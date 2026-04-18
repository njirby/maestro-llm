#!/usr/bin/env python3
"""Validation experiment: generate Stage-1 OBSERVATIONS two ways and compare.

Variant A (current): Omni listens to TARGET + DEFAULT, differential prompt.
Variant B (proposed): Omni listens to TARGET only, with a perceptual-bucket
summary of the target preset injected into the prompt as context. Text anchors
the listening so the model can't drift to pattern-matched templates.

Writes a JSONL with both observations per sample, plus an HTML comparison
(audio-embedded) so you can listen and decide which variant is more accurate
against the target audio.

Usage:
    python scripts/validate_grounded_observations.py \\
        --manifest outputs/smoke_test_v10/manifest.jsonl \\
        --n-samples 4 \\
        --omni-server http://localhost:8000 \\
        --out-jsonl outputs/smoke_v3/grounded_obs_validation.jsonl \\
        --out-html outputs/smoke_v3/grounded_obs_validation.html
"""
from __future__ import annotations

import argparse
import base64
import html
import json
import sys
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.preset_perceptual_summary import summarize_preset_perceptual


def b64(path: str | Path) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def variant_a_current(
    gt_wav: str, default_wav: str, server: str, model: str, timeout: float = 180.0
) -> str:
    """Current Stage-1 prompt: target+default, differential description."""
    content = [
        {"type": "audio_url", "audio_url": {"url": f"data:audio/wav;base64,{b64(gt_wav)}"}},
        {"type": "audio_url", "audio_url": {"url": f"data:audio/wav;base64,{b64(default_wav)}"}},
        {"type": "text", "text": (
            "You are a music production AI. Listen to two synthesizer clips.\n"
            "The first clip is the TARGET sound we need to recreate.\n"
            "The second clip is the current DEFAULT preset.\n\n"
            "Describe the perceptual differences: frequency balance (bright/warm/dark), "
            "harmonic character (clean/buzzy/rich), envelope shape (sharp/slow attack, "
            "short/long decay, sustain level), and any motion or modulation.\n"
            "3-5 short sentences. Focus on what the TARGET has that the default lacks. "
            "Use natural production language, no snake_case parameter names, no kHz numbers."
        )},
    ]
    resp = httpx.post(
        f"{server.rstrip('/')}/v1/chat/completions",
        json={"model": model, "messages": [{"role": "user", "content": content}],
              "max_tokens": 260, "temperature": 0.4},
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()


def variant_b_grounded(
    gt_wav: str, preset_summary: str, server: str, model: str, timeout: float = 180.0
) -> str:
    """Proposed Stage-1 prompt: target-only audio + perceptual preset summary.

    Summary is purely perceptual (no numbers, no param names) — it anchors the
    description without giving the model parameter-level cheating material.
    """
    content = [
        {"type": "audio_url", "audio_url": {"url": f"data:audio/wav;base64,{b64(gt_wav)}"}},
        {"type": "text", "text": (
            "You are a music production AI listening to a synth preset you need to "
            "describe in plain language.\n\n"
            "Context about the preset you're hearing (perceptual buckets, no numbers, "
            "no parameter names):\n"
            f"{preset_summary}\n\n"
            "Now listen to the clip and write 3-5 short sentences describing its "
            "perceptual character. Stay grounded in what's audible — use the context "
            "above to anchor your description, but describe how it actually sounds: "
            "tonal color, attack feel, sustain behaviour, sense of motion, spatial/effect "
            "character. Do NOT cite parameter names, numbers, Hz values, or preset fields. "
            "Do NOT start sentences with 'The target has...' — describe the sound directly."
        )},
    ]
    resp = httpx.post(
        f"{server.rstrip('/')}/v1/chat/completions",
        json={"model": model, "messages": [{"role": "user", "content": content}],
              "max_tokens": 260, "temperature": 0.4},
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()


def build_html(records: list[dict], out_path: Path) -> None:
    sections = []
    for i, r in enumerate(records):
        sid = r["sample_id"]
        archetype = r.get("archetype", "")
        t_b64 = b64(r["target_audio"])
        d_b64 = b64(r["default_audio"]) if r.get("default_audio") else ""
        summary = html.escape(r["preset_summary"])
        obs_a = html.escape(r["obs_current"] or "(empty)")
        obs_b = html.escape(r["obs_grounded"] or "(empty)")
        sections.append(f"""
<section class="sample">
  <h2>#{i+1} — {html.escape(sid)} <span class="archetype">{html.escape(archetype)}</span></h2>
  <div class="audio-row">
    <div><h4>Target audio</h4><audio controls src="data:audio/wav;base64,{t_b64}"></audio></div>
    <div><h4>Default baseline</h4><audio controls src="data:audio/wav;base64,{d_b64}"></audio></div>
  </div>
  <h4>Perceptual preset summary (injected into variant B)</h4>
  <pre class="summary">{summary}</pre>
  <div class="grid">
    <div class="col">
      <div class="side-label a">A: Current (target+default, differential prompt)</div>
      <blockquote>{obs_a}</blockquote>
    </div>
    <div class="col">
      <div class="side-label b">B: Proposed (target-only + preset summary)</div>
      <blockquote>{obs_b}</blockquote>
    </div>
  </div>
  <div class="verdict">
    <b>Which is more accurate against the target audio?</b>
    <label><input type="radio" name="v{i}" value="a"> A (current)</label>
    <label><input type="radio" name="v{i}" value="b"> B (grounded)</label>
    <label><input type="radio" name="v{i}" value="tied_good"> Both good</label>
    <label><input type="radio" name="v{i}" value="tied_bad"> Both bad</label>
  </div>
</section>
""")

    doc = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Grounded Observations Validation</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, sans-serif; max-width: 1500px; margin: 2em auto; padding: 0 1em; color: #222; }}
h1 {{ border-bottom: 2px solid #444; padding-bottom: 0.3em; }}
section.sample {{ border: 1px solid #ccc; border-radius: 8px; padding: 1em 1.5em; margin: 1.5em 0; background: #fafafa; }}
section.sample h2 {{ margin-top: 0; }}
.archetype {{ font-size: 0.55em; padding: 0.2em 0.6em; border-radius: 3px; background: #555; color: white; margin-left: 0.3em; text-transform: uppercase; }}
.audio-row {{ display: grid; grid-template-columns: 1fr 1fr; gap: 1em; margin: 0.8em 0; }}
.audio-row audio {{ width: 100%; }}
.audio-row h4 {{ margin: 0.3em 0; font-size: 0.85em; color: #555; }}
pre.summary {{ background: #fff; padding: 0.7em 1em; border: 1px solid #ddd; border-left: 3px solid #4a90e2; font-size: 0.88em; line-height: 1.5; white-space: pre-wrap; }}
.grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 1.5em; margin-top: 1em; }}
.side-label {{ font-weight: bold; padding: 0.4em 0.7em; border-radius: 4px; margin-bottom: 0.4em; }}
.side-label.a {{ background: #ffeaea; color: #822; }}
.side-label.b {{ background: #e7f5ff; color: #134; }}
blockquote {{ margin: 0; padding: 0.7em 1em; border-left: 3px solid #888; background: white; font-size: 0.92em; line-height: 1.5; }}
.verdict {{ margin-top: 1em; padding: 0.6em 1em; background: #eef5ff; border-radius: 4px; font-size: 0.9em; }}
.verdict label {{ margin-right: 1em; }}
.summary-box {{ background: #eef5ff; padding: 1em; border-radius: 6px; margin: 1em 0; }}
</style>
</head>
<body>
<h1>Stage-1 OBSERVATIONS Validation: current vs. preset-grounded</h1>
<div class="summary-box">
<b>How to use:</b> For each sample, play the target audio, read both A (current) and B (grounded)
descriptions, decide which more accurately matches what you actually hear. The preset summary
(blue box) shows what was injected into variant B — it's what would ship in training if we commit.
<p>Samples: <b>{len(records)}</b></p>
</div>
{''.join(sections)}
</body>
</html>
"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write(doc)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--n-samples", type=int, default=4)
    ap.add_argument("--omni-server", default="http://localhost:8000")
    ap.add_argument("--omni-model", default="Qwen/Qwen3-Omni-30B-A3B-Instruct")
    ap.add_argument("--out-jsonl", required=True, type=Path)
    ap.add_argument("--out-html", required=True, type=Path)
    args = ap.parse_args()

    entries: list[dict] = []
    with open(args.manifest) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    entries = entries[: args.n_samples]

    records: list[dict] = []
    for i, e in enumerate(entries):
        sid = e["sample_id"]
        print(f"[{i+1}/{len(entries)}] {sid}", flush=True)
        target_wav = e["gt_wav"]
        default_wav = e["default_wav"]
        target_preset_path = e["target_preset_path"]
        with open(target_preset_path) as f:
            preset = json.load(f)
        summary = summarize_preset_perceptual(preset)

        try:
            obs_a = variant_a_current(target_wav, default_wav, args.omni_server, args.omni_model)
        except Exception as exc:
            print(f"  variant A failed: {exc}")
            obs_a = ""
        try:
            obs_b = variant_b_grounded(target_wav, summary, args.omni_server, args.omni_model)
        except Exception as exc:
            print(f"  variant B failed: {exc}")
            obs_b = ""

        records.append({
            "sample_id": sid,
            "archetype": e.get("archetype", ""),
            "target_audio": target_wav,
            "default_audio": default_wav,
            "preset_summary": summary,
            "obs_current": obs_a,
            "obs_grounded": obs_b,
        })

    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_jsonl, "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote {args.out_jsonl}", flush=True)

    build_html(records, args.out_html)
    print(f"Wrote {args.out_html} ({args.out_html.stat().st_size // 1024} KB)", flush=True)


if __name__ == "__main__":
    main()
