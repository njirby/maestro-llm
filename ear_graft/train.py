"""Train the audio projector + <|audio_start|>/<|audio_end|> embeddings (everything else frozen). Container only:

  GPU=0 ear_graft/docker_run.sh -m ear_graft.train --run-name v0-pilot --max-steps 50
  GPU=all NAME=ear-train ear_graft/docker_run.sh -m torch.distributed.run --nproc_per_node 4 -m ear_graft.train --run-name v0

Logs to wandb (project ear-graft). Eval reports val loss with real audio vs zeroed audio (the gap is the evidence the
4B is using the audio), per-kind / per-task losses, projected-audio scale vs text embeddings, and sample generations.
Checkpoints are the trainable tensors only (~21 MB): out/<run>/step_<n>.pt and best.pt.
"""
import argparse
import json
import math
import os
import random
import re
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.distributed as dist
import wandb

from .data import Collator
from .model import AUDIO_PAD_ID, EarGraft

C = "data/ear_corpus"


def length_grouped_batches(rows, micro_bs, rng, group=64):
    """Shuffle, then sort within groups of group*micro_bs rows by length so microbatches pad little."""
    idx = list(range(len(rows)))
    rng.shuffle(idx)
    batches = []
    for s in range(0, len(idx), group * micro_bs):
        chunk = sorted(idx[s:s + group * micro_bs], key=lambda i: rows[i]["n_tok"])
        batches += [chunk[j:j + micro_bs] for j in range(0, len(chunk), micro_bs)]
    rng.shuffle(batches)
    return batches


def to_dev(b, dev):
    return {k: v.to(dev, non_blocking=True) for k, v in b.items()}


@torch.no_grad()
def evaluate(model, collate, val_rows, args, dev, step):
    model.eval()
    sums = defaultdict(float)
    cnts = defaultdict(int)
    norms = []
    text_norm = model.lm.get_input_embeddings().weight[:150000].float().norm(dim=-1).median().item()
    rows = val_rows[: args.eval_rows]
    for s in range(0, len(rows), args.micro_bs):
        chunk = rows[s:s + args.micro_bs]
        b = to_dev(collate(chunk), dev)
        n_lab = (b["labels"][:, 1:] != -100).sum().item()
        for tag, scale in (("audio", 1.0), ("zeroed", 0.0)):
            out = model(**b, audio_scale=scale)
            sums[tag] += out.loss.item() * n_lab
            cnts[tag] += n_lab
            if tag == "audio":
                # per-row losses for kind/task breakdowns
                for r, tl, lm in zip(chunk, out.tok_loss, out.mask):
                    l, n = tl[lm].sum().item(), lm.sum().item()
                    for key in [f"kind/{r['kinds'][0]}", f"task/{r['task']}", f"think/{r['think']}"]:
                        sums[key] += l
                        cnts[key] += n
        ae = model.audio_embeds(b["input_features"], b["feature_attention_mask"]).float()
        norms.append(ae.norm(dim=-1).median().item())
    logs = {f"val/loss_{k}": sums[k] / cnts[k] for k in sums if cnts[k]}
    logs = {k.replace("val/loss_kind/", "val_kind/").replace("val/loss_task/", "val_task/").replace("val/loss_think/", "val_think/"): v
            for k, v in logs.items()}
    logs["val/audio_gain"] = logs["val/loss_zeroed"] - logs["val/loss_audio"]
    logs["val/audio_embed_norm"] = sum(norms) / len(norms)
    logs["val/text_embed_norm"] = text_norm

    table = wandb.Table(columns=["step", "id", "kinds", "prompt", "student", "teacher"])
    for r in val_rows[: args.n_samples]:
        g = collate([{**r, "answer": None, "think": False}])
        g = to_dev({k: v for k, v in g.items() if k != "labels"}, dev)
        ids = model.generate(**g, max_new_tokens=args.sample_tokens, do_sample=False)
        table.add_data(step, r["id"], ",".join(r["kinds"]), r["prompt"],
                       collate.tok.decode(ids[0], skip_special_tokens=True), r["answer"][:1500])
    logs["samples"] = table
    model.train()
    return logs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--train", default=f"{C}/train.jsonl")
    ap.add_argument("--val", default=f"{C}/val.jsonl")
    ap.add_argument("--out", default="ear_graft/out")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--token-lr", type=float, default=None, help="lr for the two marker embeddings (default = --lr)")
    ap.add_argument("--warmup", type=float, default=0.03)
    ap.add_argument("--min-lr-frac", type=float, default=0.1)
    ap.add_argument("--micro-bs", type=int, default=4)
    ap.add_argument("--global-bs", type=int, default=32)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--save-every", type=int, default=200)
    ap.add_argument("--eval-rows", type=int, default=400)
    ap.add_argument("--n-samples", type=int, default=6)
    ap.add_argument("--sample-tokens", type=int, default=160)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--projector", default="linear", choices=["linear", "mlp"])
    ap.add_argument("--lora-r", type=int, default=0, help="LoRA rank on every decoder linear layer of the 4B (0 = frozen 4B)")
    ap.add_argument("--lora-lr", type=float, default=1e-4)
    ap.add_argument("--init-from", default=None, help="trainable checkpoint to start from (see --init-only)")
    ap.add_argument("--init-only", default=None, help="comma list of key prefixes to take from --init-from, e.g. audio_start,audio_end")
    ap.add_argument("--resume", default=None, help="step_<n>.pt to continue from (loads step_<n>.opt.pt too if present)")
    ap.add_argument("--rewarm", type=int, default=20, help="lr ramp after a resume without optimizer state")
    ap.add_argument("--wandb-project", default=os.environ.get("WANDB_PROJECT", "ear-graft"))
    args = ap.parse_args()

    ddp = "WORLD_SIZE" in os.environ
    if ddp:
        dist.init_process_group("nccl")
        rank, world = dist.get_rank(), dist.get_world_size()
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    else:
        rank, world = 0, 1
    dev = torch.device("cuda", torch.cuda.current_device())
    torch.manual_seed(args.seed)

    model = EarGraft(device=dev, projector=args.projector, lora_r=args.lora_r).to(dev)
    if args.init_from:
        model.load_trainable(args.init_from, only=args.init_only.split(",") if args.init_only else None)
    model.lm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.lm.config.use_cache = False
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    lora = [p for n, p in model.named_parameters() if p.requires_grad and "lora_" in n]
    assert all(n.startswith(("projector.", "audio_start", "audio_end")) or "lora_" in n for n in trainable), trainable
    assert bool(lora) == bool(args.lora_r)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    groups = [{"params": list(model.projector.parameters()), "lr": args.lr},
              {"params": [model.audio_start, model.audio_end], "lr": args.token_lr or args.lr}]
    if lora:
        groups.append({"params": lora, "lr": args.lora_lr})
    opt = torch.optim.AdamW(groups, betas=(0.9, 0.95), weight_decay=0.0)
    base_lrs = [g["lr"] for g in opt.param_groups]
    start, opt_loaded = 0, False
    if args.resume:
        model.load_trainable(args.resume)
        start = int(re.search(r"step_(\d+)", args.resume).group(1)) + 1
        opt_path = Path(args.resume).with_suffix(".opt.pt")
        if opt_path.exists():
            opt.load_state_dict(torch.load(opt_path, map_location=dev))
            opt_loaded = True
    net = torch.nn.parallel.DistributedDataParallel(model, device_ids=[dev.index], broadcast_buffers=False) if ddp else model

    train_rows = [json.loads(l) for l in open(args.train)]
    val_rows = [json.loads(l) for l in open(args.val)]
    random.Random(1).shuffle(val_rows)
    collate = Collator()
    accum = max(1, args.global_bs // (args.micro_bs * world))
    steps_per_epoch = len(train_rows) // (args.micro_bs * world * accum)
    total = args.max_steps or math.ceil(args.epochs * steps_per_epoch)
    warm = max(1, int(args.warmup * total))

    def lr_at(step):
        if args.resume and not opt_loaded and step - start < args.rewarm:  # fresh Adam moments: ramp back up
            return (step - start + 1) / args.rewarm * lr_at_sched(step)
        return lr_at_sched(step)

    def lr_at_sched(step):
        if step < warm:
            return (step + 1) / warm
        p = (step - warm) / max(1, total - warm)
        return args.min_lr_frac + (1 - args.min_lr_frac) * 0.5 * (1 + math.cos(math.pi * min(1.0, p)))

    out_dir = Path(args.out) / args.run_name
    if rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)
        cfg = {**vars(args), "world": world, "accum": accum, "total_steps": total, "steps_per_epoch": steps_per_epoch,
               "n_train_rows": len(train_rows), "n_val_rows": len(val_rows), "n_trainable": n_train,
               "start_step": start, "resumed_optimizer": opt_loaded}
        (out_dir / "config.json").write_text(json.dumps(cfg, indent=1))
        wandb.init(project=args.wandb_project, name=args.run_name, config=cfg, dir=str(out_dir))
        print(f"trainable params {n_train}; {len(train_rows)} train rows; world {world} x micro {args.micro_bs} x accum {accum}"
              f" = {args.micro_bs * world * accum}/step; {total} steps; starting at {start}"
              f"{' (optimizer restored)' if opt_loaded else ''}", flush=True)

    rng = random.Random(args.seed)
    loader_iter = None
    epoch = 0

    def next_loader(skip=0):
        nonlocal epoch
        batches = length_grouped_batches(train_rows, args.micro_bs, rng)
        batches = batches[rank::world][: steps_per_epoch * accum][skip:]  # same count on every rank
        epoch += 1
        dl = torch.utils.data.DataLoader([[train_rows[i] for i in b] for b in batches], batch_size=None,
                                         collate_fn=collate, num_workers=args.workers, prefetch_factor=4,
                                         persistent_workers=False, pin_memory=True)
        return iter(dl)

    # Resume: replay the data order (one rng shuffle per epoch) and skip the microbatches already trained on.
    done_epochs, skip = divmod(start * accum, steps_per_epoch * accum)
    for _ in range(done_epochs):
        length_grouped_batches(train_rows, args.micro_bs, rng)
        epoch += 1
    if start:
        loader_iter = next_loader(skip)

    best = float("inf")
    model.train()
    t0, seen_rows, seen_tok = time.time(), 0, 0
    for step in range(start, total):
        for g, base in zip(opt.param_groups, base_lrs):
            g["lr"] = base * lr_at(step)
        micros = []
        for micro in range(accum):
            try:
                b = next(loader_iter) if loader_iter else None
            except StopIteration:
                b = None
            if b is None:
                loader_iter = next_loader()
                b = next(loader_iter)
            micros.append(b)
        # Every labelled token in the global batch weighs the same (short-answer microbatches no longer count double).
        n_glob = torch.tensor(float(sum(int((b["labels"][:, 1:] != -100).sum()) for b in micros)), device=dev)
        if ddp:
            dist.all_reduce(n_glob)
        loss_sum = torch.zeros((), device=dev)
        for micro, b in enumerate(micros):
            b = to_dev(b, dev)
            sync = (micro == accum - 1) or not ddp
            ctx = net.no_sync() if (ddp and not sync) else torch.enable_grad()
            with ctx:
                out = net(**b)
                (out.loss_sum * world / n_glob).backward()  # DDP averages over ranks, hence * world
            loss_sum += out.loss_sum.detach()
            seen_rows += b["input_ids"].shape[0]
            seen_tok += int(b["attention_mask"].sum())
        if ddp:
            dist.all_reduce(loss_sum)
        loss_sum = (loss_sum / n_glob).item()
        gn = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)

        if rank == 0 and (step % 10 == 0 or step == total - 1):
            el = time.time() - t0
            logs = {"train/loss": loss_sum, "train/lr": opt.param_groups[0]["lr"], "train/grad_norm": gn.item(),
                    "train/epoch": epoch - 1 + (step % max(1, steps_per_epoch)) / max(1, steps_per_epoch),
                    "perf/rows_per_s": seen_rows * world / el, "perf/tok_per_s": seen_tok * world / el,
                    "perf/gpu_mem_gb": torch.cuda.max_memory_allocated() / 2**30,
                    "weights/projector_norm": torch.cat([p.flatten() for p in model.projector.parameters()]).norm().item(),
                    **({"weights/lora_norm": torch.cat([p.detach().flatten() for p in lora]).norm().item()} if lora else {}),
                    "weights/audio_start_norm": model.audio_start.norm().item(),
                    "weights/audio_end_norm": model.audio_end.norm().item()}
            wandb.log(logs, step=step)
            print(f"step {step}/{total} loss {loss_sum:.4f} gn {gn.item():.3f} lr {logs['train/lr']:.2e} "
                  f"{logs['perf/rows_per_s']:.2f} rows/s", flush=True)

        last = step == total - 1
        if (step and step % args.eval_every == 0) or last:
            if rank == 0:
                logs = evaluate(model, collate, val_rows, args, dev, step)
                wandb.log(logs, step=step)
                print(f"eval {step}: loss audio {logs['val/loss_audio']:.4f} zeroed {logs['val/loss_zeroed']:.4f} "
                      f"gain {logs['val/audio_gain']:.4f} audio-norm {logs['val/audio_embed_norm']:.3f}", flush=True)
                if logs["val/loss_audio"] < best:
                    best = logs["val/loss_audio"]
                    torch.save(model.trainable_state_dict(), out_dir / "best.pt")
            if ddp:
                dist.barrier()
        if rank == 0 and ((step and step % args.save_every == 0) or last):
            torch.save(model.trainable_state_dict(), out_dir / f"step_{step}.pt")
            torch.save(opt.state_dict(), out_dir / f"step_{step}.opt.pt")

    if rank == 0:
        wandb.summary["best_val_loss"] = best
        wandb.finish()
    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
