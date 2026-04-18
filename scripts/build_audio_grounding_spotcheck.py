#!/usr/bin/env python3
"""Build a self-contained HTML report for manually spot-checking the
audio-grounded OBSERVATIONS judge.

For each graded record, renders:
  - Inline audio player for target WAV (base64-embedded, portable)
  - Inline audio player for default-baseline WAV (reference)
  - The OBSERVATIONS text the agent produced
  - The judge's score + reasoning
  - A preset summary (active oscillators, envelope shape, active effects,
    wavetable names) so you can see what the sound actually IS
  - A human-verdict radio button you can mark

Open the HTML in a browser, play audio, compare to observations, and tick
Match/Partial/Mismatch. Lets you validate whether the LLM judge is accurate
or too harsh on its own.

Usage:
    python scripts/build_audio_grounding_spotcheck.py \
        --grades outputs/smoke_v3/grades_main_v2_llmjudge_ext2.jsonl \
        --out outputs/smoke_v3/audio_grounding_spotcheck.html
"""
from __future__ import annotations

import argparse
import base64
import html
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.grade_agent_sft import _extract_v3_plan_and_narrations


def b64_audio(path: str | Path) -> str:
    p = Path(path)
    if not p.exists():
        return ""
    with open(p, "rb") as f:
        return base64.b64encode(f.read()).decode()


def summarize_preset(preset_path: str | Path) -> dict:
    """Extract audible-relevant preset params for human verification."""
    try:
        with open(preset_path) as f:
            p = json.load(f)
    except Exception:
        return {"error": f"could not load {preset_path}"}
    s = p.get("settings", {}) or {}
    summary: dict = {"preset_path": str(preset_path)}
    summary["active_oscillators"] = [i for i in (1, 2, 3) if s.get(f"osc_{i}_on", 0) > 0.5]
    for osc in summary["active_oscillators"]:
        summary[f"osc_{osc}_wavetable"] = s.get(f"osc_{osc}_wavetable_frame_name", "?")
        summary[f"osc_{osc}_level"] = round(float(s.get(f"osc_{osc}_level", 0)), 3)
        summary[f"osc_{osc}_unison_voices"] = int(s.get(f"osc_{osc}_unison_voices", 1) or 1)
        summary[f"osc_{osc}_unison_detune"] = round(float(s.get(f"osc_{osc}_unison_detune", 0)), 3)
    summary["env_1"] = {
        "attack_s": round(float(s.get("env_1_attack", 0)), 3),
        "decay_s": round(float(s.get("env_1_decay", 0)), 3),
        "sustain": round(float(s.get("env_1_sustain", 0)), 3),
        "release_s": round(float(s.get("env_1_release", 0)), 3),
    }
    summary["filter_1"] = {
        "on": s.get("filter_1_on", 0) > 0.5,
        "cutoff_midi": round(float(s.get("filter_1_cutoff", 0)), 1),
        "resonance": round(float(s.get("filter_1_resonance", 0)), 3),
        "model": int(s.get("filter_1_model", 0) or 0),
    }
    effects_on = {
        "chorus": s.get("chorus_on", 0) > 0.5,
        "reverb": s.get("reverb_on", 0) > 0.5,
        "delay": s.get("delay_on", 0) > 0.5,
        "distortion": s.get("distortion_on", 0) > 0.5,
        "compressor": s.get("compressor_on", 0) > 0.5,
        "eq": s.get("eq_on", 0) > 0.5,
        "phaser": s.get("phaser_on", 0) > 0.5,
        "flanger": s.get("flanger_on", 0) > 0.5,
    }
    summary["effects_active"] = [k for k, v in effects_on.items() if v]
    # LFO active?
    summary["lfos_with_depth"] = []
    for i in range(1, 9):
        # heuristic: LFO is "active" if any modulation route sources it with nonzero amount
        for mi in range(1, 65):
            src = s.get(f"modulation_{mi}_source", "") or ""
            amt = s.get(f"modulation_{mi}_amount", 0) or 0
            if f"lfo_{i}" in str(src) and abs(float(amt)) > 0.01:
                summary["lfos_with_depth"].append(i)
                break
    summary["lfos_with_depth"] = sorted(set(summary["lfos_with_depth"]))
    return summary


def _score_color(score: float | None) -> str:
    if score is None:
        return "#888"
    if score < 0.5:
        return "#d94444"
    if score < 1.0:
        return "#44a"
    return "#449944"


def _score_display(score: float | None) -> str:
    return f"{score:.2f}" if score is not None else "—"


def _build_side(record: dict | None, label: str) -> str:
    """Render one side of the comparison: label + observations + judge + narrations."""
    if record is None:
        return f"<div class='col'><h3>{html.escape(label)}</h3><p><i>(no data)</i></p></div>"
    q = record.get("quality_scores", {})
    extracted = _extract_v3_plan_and_narrations(record)
    observations = extracted.get("observations", "") or ""
    narrations = extracted.get("narrations", []) or []
    audio_score = q.get("llm_observations_audio_grounded")
    audio_reason = q.get("llm_observations_reasoning", "")
    per_batch = q.get("llm_per_batch", []) or []

    nar_rows: list[str] = []
    for i, (sub, nar) in enumerate(narrations):
        pb = per_batch[i] if i < len(per_batch) else {}
        templ = pb.get("templateness")
        plan_ref = pb.get("plan_reference")
        param_sp = pb.get("parameter_specific")
        no_h = pb.get("no_hallucination")
        badge = (
            f"<span class='mini' style='background:{_score_color(templ)}' title='templateness'>T:{_score_display(templ)}</span>"
            f"<span class='mini' style='background:{_score_color(plan_ref)}' title='plan_reference'>P:{_score_display(plan_ref)}</span>"
            f"<span class='mini' style='background:{_score_color(param_sp)}' title='parameter_specific'>S:{_score_display(param_sp)}</span>"
            f"<span class='mini' style='background:{_score_color(no_h)}' title='no_hallucination'>H:{_score_display(no_h)}</span>"
        )
        nar_rows.append(
            f"<tr><td class='sub'>{html.escape(sub)}</td><td class='nar'>{html.escape(nar[:400])}</td><td class='badges'>{badge}</td></tr>"
        )
    nar_table = "<table class='nar-table'><tr><th>subsystem</th><th>narration</th><th>judge</th></tr>" + "".join(nar_rows) + "</table>"

    return f"""
<div class="col">
  <div class="side-label">{html.escape(label)}
    <span class="score" style="background:{_score_color(audio_score)}">audio_grounded: {_score_display(audio_score)}</span>
  </div>
  <h4>OBSERVATIONS</h4>
  <blockquote>{html.escape(observations) or '<i>(empty)</i>'}</blockquote>
  <h4>Judge reasoning (audio_grounded)</h4>
  <blockquote class="judge">{html.escape(audio_reason) or '<i>(empty)</i>'}</blockquote>
  <h4>Per-batch narrations <span class="hint">(T=templateness, P=plan_ref, S=param_spec, H=no_halluc)</span></h4>
  {nar_table}
</div>
"""


def render_sample(
    record_new: dict,
    record_compare: dict | None,
    sample_idx: int,
    label_new: str = "New build",
    label_compare: str = "Previous build",
) -> str:
    sid = record_new.get("meta", {}).get("sample_id", record_new.get("id", "?"))
    extracted = _extract_v3_plan_and_narrations(record_new)
    target_audio = extracted.get("target_audio", "")
    audios = record_new.get("audios", []) or []
    default_audio = audios[1] if len(audios) > 1 else ""
    target_b64 = b64_audio(target_audio)
    default_b64 = b64_audio(default_audio) if default_audio else ""

    meta = record_new.get("meta", {})
    archetype = meta.get("archetype", "")
    sample_id_raw = meta.get("sample_id", "")
    manifest_dir = Path(target_audio).parent.parent if target_audio else Path(".")
    preset_path = manifest_dir / "paths" / f"{sample_id_raw}_target.vital"
    preset_summary = summarize_preset(preset_path) if preset_path.exists() else {}
    preset_html = "<pre class='preset'>" + html.escape(json.dumps(preset_summary, indent=2)) + "</pre>"

    # If no compare record, just show the single side full-width
    if record_compare is None:
        side_html = _build_side(record_new, label_new)
        cols_class = "grid-single"
    else:
        side_html = _build_side(record_new, label_new) + _build_side(record_compare, label_compare)
        cols_class = "grid-compare"

    return f"""
<section class="sample">
  <h2>#{sample_idx + 1} &mdash; {html.escape(sid)}
    <span class="archetype">{html.escape(archetype)}</span>
  </h2>
  <div class="top-row">
    <div class="audio-col">
      <h4>Target (what the agent heard)</h4>
      <audio controls src="data:audio/wav;base64,{target_b64}"></audio>
      <h4>Default baseline</h4>
      <audio controls src="data:audio/wav;base64,{default_b64}"></audio>
      <h4>Preset summary (ground truth)</h4>
      {preset_html}
    </div>
  </div>
  <div class="{cols_class}">
    {side_html}
  </div>
  <div class="verdict">
    <b>Your verdict (audio-to-observations match):</b>
    <label><input type="radio" name="v{sample_idx}" value="new_better"> New better</label>
    <label><input type="radio" name="v{sample_idx}" value="prev_better"> Prev better</label>
    <label><input type="radio" name="v{sample_idx}" value="both_bad"> Both bad</label>
    <label><input type="radio" name="v{sample_idx}" value="both_good"> Both good</label>
  </div>
</section>
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--grades", required=True, type=Path,
                    help="Graded JSONL (v3 main records with quality_scores).")
    ap.add_argument("--compare-grades", type=Path, default=None,
                    help="Optional second graded JSONL to show side-by-side for comparison.")
    ap.add_argument("--label-new", default="New build",
                    help="Column label for --grades.")
    ap.add_argument("--label-compare", default="Previous build",
                    help="Column label for --compare-grades.")
    ap.add_argument("--out", required=True, type=Path,
                    help="Output HTML file.")
    args = ap.parse_args()

    def _load(path: Path) -> list[dict]:
        out = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    out.append(json.loads(line))
        return out

    records_new = _load(args.grades)
    compare_map: dict[str, dict] = {}
    if args.compare_grades:
        for r in _load(args.compare_grades):
            sid = r.get("meta", {}).get("sample_id", r.get("id", ""))
            if sid:
                compare_map[sid] = r

    sections = []
    for i, r in enumerate(records_new):
        sid = r.get("meta", {}).get("sample_id", r.get("id", ""))
        compare_r = compare_map.get(sid)
        sections.append(render_sample(
            r, compare_r, i,
            label_new=args.label_new,
            label_compare=args.label_compare,
        ))
    sections_html = "\n".join(sections)

    comparison_note = (
        f"<b>Comparison mode:</b> <code>{html.escape(args.label_new)}</code> vs. <code>{html.escape(args.label_compare)}</code>"
        if args.compare_grades else
        "<b>Single-build mode.</b> Pass --compare-grades to show side-by-side with another build."
    )

    html_doc = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Audio-Grounding Spot-Check</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, sans-serif; max-width: 1600px; margin: 2em auto; padding: 0 1em; color: #222; }}
h1 {{ border-bottom: 2px solid #444; padding-bottom: 0.3em; }}
section.sample {{ border: 1px solid #ccc; border-radius: 8px; padding: 1em 1.5em; margin: 1.5em 0; background: #fafafa; }}
section.sample h2 {{ margin-top: 0; }}
.score {{ font-size: 0.7em; padding: 0.25em 0.55em; border-radius: 4px; color: white; margin-left: 0.5em; font-family: monospace; }}
.mini {{ display: inline-block; font-size: 0.7em; padding: 0.05em 0.35em; margin: 0 0.1em; border-radius: 3px; color: white; font-family: monospace; }}
.archetype {{ font-size: 0.6em; padding: 0.2em 0.6em; border-radius: 3px; background: #555; color: white; margin-left: 0.3em; text-transform: uppercase; }}
.grid-compare {{ display: grid; grid-template-columns: 1fr 1fr; gap: 1.5em; margin-top: 1em; }}
.grid-single {{ margin-top: 1em; }}
.top-row {{ display: grid; grid-template-columns: 1fr; margin-top: 0.8em; }}
.audio-col h4 {{ margin: 0.8em 0 0.3em 0; font-size: 0.85em; color: #555; }}
.col h3, .col h4 {{ margin: 1em 0 0.3em; font-size: 0.9em; color: #555; }}
.side-label {{ font-weight: bold; padding: 0.4em 0.6em; background: #eef; border-radius: 4px; margin-bottom: 0.6em; }}
blockquote {{ margin: 0.3em 0; padding: 0.55em 0.9em; border-left: 3px solid #888; background: white; font-size: 0.88em; line-height: 1.4; }}
blockquote.judge {{ border-left-color: #d94444; background: #fef3f3; }}
pre.preset {{ background: #fff; padding: 0.7em; border: 1px solid #ddd; font-size: 0.72em; line-height: 1.3; white-space: pre-wrap; word-break: break-word; max-height: 280px; overflow-y: auto; }}
audio {{ width: 100%; }}
table.nar-table {{ width: 100%; border-collapse: collapse; font-size: 0.82em; }}
table.nar-table th {{ background: #eee; text-align: left; padding: 0.3em; border-bottom: 1px solid #ccc; }}
table.nar-table td {{ padding: 0.35em; border-bottom: 1px solid #eee; vertical-align: top; }}
table.nar-table td.sub {{ width: 90px; font-weight: bold; color: #555; }}
table.nar-table td.nar {{ font-size: 0.95em; line-height: 1.35; }}
table.nar-table td.badges {{ width: 175px; white-space: nowrap; }}
.hint {{ font-weight: normal; font-size: 0.8em; color: #888; }}
.verdict {{ margin-top: 1em; padding: 0.6em 0.9em; background: #eef5ff; border-radius: 4px; font-size: 0.88em; }}
.verdict label {{ margin-right: 1em; }}
.summary {{ background: #eef5ff; padding: 1em; border-radius: 6px; margin: 1em 0; }}
</style>
</head>
<body>
<h1>Audio-Grounding Spot-Check</h1>
<div class="summary">
{comparison_note}
<p><b>How to use:</b> For each sample, play the target audio, read the OBSERVATIONS on each side, compare them to what you hear, and pick whether new or prev is better. The per-batch narrations table includes the judge's four axes (T=templateness, P=plan_ref, S=param_spec, H=no_halluc).</p>
<p>Total samples: <b>{len(records_new)}</b>, compare records matched: <b>{len(compare_map) if args.compare_grades else '—'}</b></p>
</div>
{sections_html}
</body>
</html>
"""
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        f.write(html_doc)
    print(f"Wrote {args.out} ({args.out.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
