"""Plumbing checks for the graft (run in the container: GPU=0 ear_graft/docker_run.sh -m ear_graft.smoke_test)."""
import numpy as np
import torch

from .data import SR, Collator, load_audio
from .model import AUDIO_PAD_ID, EarGraft

CLIPS = [
    "data/raw/phase1_sources/samples/cymatics_orchid_stems_pt2/Cymatics - New Flow - 102 BPM F Min Bass.wav",
    "data/raw/phase1_sources/samples/cymatics_orchid_stems_pt2/Cymatics - New Flow - 102 BPM F Min Synth.wav",
]


def rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


def main():
    torch.manual_seed(0)
    dev = "cuda"
    model = EarGraft().to(dev)
    col = Collator()

    # 1. only the projector + two marker embeddings train
    trainable = {n: p.numel() for n, p in model.named_parameters() if p.requires_grad}
    expect = 2048 * 2560 + 2560 + 2 * 2560
    print("trainable:", trainable, "total", sum(trainable.values()))
    assert set(trainable) == {"projector.weight", "projector.bias", "audio_start", "audio_end"}
    assert sum(trainable.values()) == expect

    wavs = [load_audio(p, 12.0) for p in CLIPS] + [np.zeros(SR, dtype=np.float32)]  # 12 s, 12 s, 1 s silence
    batch = [{"audio": w, "prompt": "Describe this sound.", "target": t}
             for w, t in zip(wavs, ["A deep bass line.", "A bright synth melody with a soft attack.", "Silence."])]
    b = {k: v.to(dev) for k, v in col(batch).items()}

    # 2. tower: batched == per-clip (window/clip isolation), and sdpa == flash_attention_2
    with torch.no_grad():
        lens = b["feature_attention_mask"].sum(-1)
        packed = b["input_features"].permute(0, 2, 1)[b["feature_attention_mask"].bool()].permute(1, 0).bfloat16()
        out = model.tower(packed, feature_lens=lens).last_hidden_state
        n0 = int(lens[0])
        single = model.tower(packed[:, :n0], feature_lens=lens[:1]).last_hidden_state
        print(f"tower batched vs single clip rel diff: {rel(out[: single.shape[0]], single):.4f}")
        assert rel(out[: single.shape[0]], single) < 0.02
        model.tower.config._attn_implementation = "flash_attention_2"
        fa = model.tower(packed, feature_lens=lens).last_hidden_state
        model.tower.config._attn_implementation = "sdpa"
        print(f"tower sdpa vs flash_attention_2 rel diff: {rel(out, fa):.4f}")
        assert rel(out, fa) < 0.02
        print(f"tower out tokens {out.shape[0]}, audio_pad tokens {int((b['input_ids'] == AUDIO_PAD_ID).sum())}")

    # 3. positions: hook the rotary embedding, require 0..L-1 sequential for every row
    seen = {}
    def grab(m, a, o):  # must return None, or the hook replaces the module output
        seen.setdefault("pos", a[1])
    h = model.lm.model.language_model.rotary_emb.register_forward_hook(grab)
    model.lm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    out = model(**b)
    h.remove()
    pos = seen["pos"]  # (3, B, L) M-RoPE rows
    L = b["input_ids"].shape[1]
    assert torch.equal(pos, torch.arange(L, device=dev).view(1, 1, -1).expand_as(pos)), pos[:, :, :8]
    print("positions: sequential for all 3 M-RoPE rows")

    # 4. loss finite; grads reach exactly the trainable params
    print(f"loss (untrained) {out.loss.item():.4f}")
    assert torch.isfinite(out.loss)
    out.loss.backward()
    for n in ("projector.weight", "projector.bias", "audio_start", "audio_end"):
        g = dict(model.named_parameters())[n].grad
        assert g is not None and g.abs().sum() > 0, n
    assert all(p.grad is None for n, p in model.named_parameters() if not p.requires_grad)
    print("grads: nonzero on projector/audio_start/audio_end, none on frozen params")
    model.zero_grad(set_to_none=True)

    # 5. negative control knob changes the loss; mismatched pad count fails loud
    model.eval()
    with torch.no_grad():
        l1, l0 = model(**b).loss.item(), model(**b, audio_scale=0.0).loss.item()
    print(f"loss audio {l1:.4f} vs zeroed audio {l0:.4f}")
    bad = dict(b); bad["input_ids"] = b["input_ids"].clone()
    bad["input_ids"][0, (bad["input_ids"][0] == AUDIO_PAD_ID).nonzero()[0]] = 0
    try:
        model(**bad); raise AssertionError("pad/feature mismatch not caught")
    except ValueError as e:
        print("mismatch caught:", e)

    # 6. generation through inputs_embeds (untrained projector: expect junk, just no crash)
    g = col([{"audio": wavs[1], "prompt": "Describe this sound."}])
    g = {k: v.to(dev) for k, v in g.items() if k != "labels"}
    ids = model.generate(**g, max_new_tokens=24, do_sample=False)
    print("generate (untrained):", repr(col.tok.decode(ids[0], skip_special_tokens=False)))
    print(f"peak GPU mem {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")
    print("SMOKE OK")


if __name__ == "__main__":
    main()
