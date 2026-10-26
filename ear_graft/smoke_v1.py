"""Gate for v1 (MLP projector + LoRA, started from a v0 linear-projector checkpoint). One GPU, container only:

  GPU=0 ear_graft/docker_run.sh -m ear_graft.smoke_v1 --init ear_graft/out/v0-r1400/step_2749.pt

Checks: (1) the lifted v1 model at step 0 gives the same loss as the v0 model it came from; (2) after a few optimizer
steps the saved checkpoint holds every trainable tensor (MLP, both markers, all LoRA A/B); (3) every one of them
changed; (4) frozen weights are bit-identical; (5) reloading the checkpoint into a fresh model reproduces the loss.
"""
import argparse
import gc
import json

import torch

from .data import Collator
from .model import EarGraft

C = "data/ear_corpus"


def batch_loss(model, b):
    model.eval()
    with torch.no_grad():
        return model(**b).loss.item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", required=True)
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--out", default="ear_graft/out/smoke_v1.pt")
    args = ap.parse_args()
    dev = torch.device("cuda", 0)
    collate = Collator()
    rows = [json.loads(l) for l in open(f"{C}/train_v1.jsonl")]
    pick = [next(r for r in rows if r["think"] and len(r["clips"]) == 1 and r["n_tok"] < 900),
            next(r for r in rows if not r["think"] and len(r["clips"]) == 2 and r["n_tok"] < 900)]
    b = {k: v.to(dev) for k, v in collate(pick).items()}

    v0 = EarGraft(device=dev).to(dev)
    v0.load_trainable(args.init)
    l_v0 = batch_loss(v0, b)
    del v0; gc.collect(); torch.cuda.empty_cache()

    m = EarGraft(device=dev, projector="mlp", lora_r=16).to(dev)
    m.load_trainable(args.init)
    l_lift = batch_loss(m, b)
    print(f"(1) step-0 loss: v0 {l_v0:.5f}  lifted v1 {l_lift:.5f}  diff {abs(l_v0 - l_lift):.2e}")
    assert abs(l_v0 - l_lift) < 2e-3, "lifted model does not reproduce v0"

    trainable = {n: p for n, p in m.named_parameters() if p.requires_grad}
    init = {n: p.detach().clone() for n, p in trainable.items()}
    frozen_probe = {n: p.detach().clone() for n, p in m.named_parameters()
                    if not p.requires_grad and any(s in n for s in ("embed_tokens", "layers.3.linear_attn.in_proj_qkv.base",
                                                                    "layers.3.self_attn.q_proj.base", "tower.layers.0.", "lm_head"))}
    m.lm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    lora = [p for n, p in trainable.items() if "lora_" in n]
    opt = torch.optim.AdamW([{"params": list(m.projector.parameters()), "lr": 5e-4},
                             {"params": [m.audio_start, m.audio_end], "lr": 5e-4},
                             {"params": lora, "lr": 1e-4}], betas=(0.9, 0.95), weight_decay=0.0)
    m.train()
    for step in range(args.steps):
        out = m(**b)
        out.loss.backward()
        opt.step(); opt.zero_grad(set_to_none=True)
        print(f"    step {step} loss {out.loss.item():.4f}")
    l_trained = batch_loss(m, b)

    sd = m.trainable_state_dict()
    torch.save(sd, args.out)
    n_lora = sum("lora_" in k for k in sd)
    print(f"(2) saved {len(sd)} tensors: projector {sum(k.startswith('projector.') for k in sd)}, "
          f"markers {sum(k.startswith('audio_') for k in sd)}, lora {n_lora}; "
          f"{sum(v.numel() for v in sd.values()):,} params, {sum(v.numel() * v.element_size() for v in sd.values()) / 2**20:.0f} MB")
    assert set(sd) == set(trainable), set(sd) ^ set(trainable)
    assert n_lora == 400 and len(sd) == 406, (n_lora, len(sd))

    unchanged = [n for n, p in trainable.items() if torch.equal(p.detach(), init[n])]
    deltas = {g: max((trainable[n].detach() - init[n]).abs().max().item() for n in trainable if g(n))
              for g in (lambda n: n.startswith("projector."), lambda n: n.startswith("audio_"),
                        lambda n: "lora_A" in n, lambda n: "lora_B" in n)}
    print(f"(3) unchanged trainable tensors: {len(unchanged)} {unchanged[:5]}; max |delta| projector/markers/loraA/loraB:",
          " / ".join(f"{v:.2e}" for v in deltas.values()))
    assert not unchanged

    same = {n: torch.equal(p.detach(), frozen_probe[n]) for n, p in m.named_parameters() if n in frozen_probe}
    print(f"(4) frozen probes bit-identical: {sum(same.values())}/{len(same)}")
    assert same and all(same.values())

    del m, opt, trainable, init, lora; gc.collect(); torch.cuda.empty_cache()
    m2 = EarGraft(device=dev, projector="mlp", lora_r=16).to(dev)
    m2.load_trainable(args.out)
    l_reload = batch_loss(m2, b)
    print(f"(5) loss after training {l_trained:.5f}  reloaded {l_reload:.5f}  (step-0 was {l_lift:.5f})")
    assert abs(l_trained - l_reload) < 1e-4
    print("SMOKE V1 OK")


if __name__ == "__main__":
    main()
