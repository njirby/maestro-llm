#!/usr/bin/env python3
"""Search-agent SFT v3 — opencode contract, terse verdicts, evidence labels.

Redesign of v2 following the 2026-08-15 postmortem:
  - probes render with FULL preset state (fixed DawDreamer load_state snippet)
    using the sample's TRANSCRIBED melody; a builder-side discriminability
    gate (spectral-centroid spread >= 500 Hz) aborts the sample if probe
    audio carries no wavetable information.
  - shortlist labels are EVIDENCE-BASED: candidate-probe vs GT-ingredient-
    probe CLAP cosine >= threshold, plus an always-on top-2 best-available
    floor. No name-identity backdoor — the teacher only "knows" what the
    audio supports.
  - terse per-candidate verdicts templated from measured audio features
    (brightness / attack / noisiness), no per-candidate LLM calls; one
    optional batched text-only LLM call writes a sentence per shortlisted
    candidate.
  - records follow the opencode contract: leading system turn, lowercase
    bash/read tools, plain-string tool outputs.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import opencode_contract as oc
from scripts.agent_sft_common import (
    ClapEmbedder,
    DawFarmRolloutCtx,
    _wrap_as_bash,
    assert_valid_oc_record,
    build_list_wavetables_slice_snippet,
    build_render_probes_snippet,
    make_agent_id,
    oc_bash_call_msg,
    oc_bash_response_msg,
    oc_read_audio_response_msg,
    oc_read_call_msg,
)

OC_TOOLS_JSON = json.dumps(oc.TOOLS, ensure_ascii=False)

MIN_CENTROID_SPREAD_HZ = 500.0


# ---------------------------------------------------------------------------
# Audio feature descriptors (terse verdict vocabulary)
# ---------------------------------------------------------------------------

def audio_features(path: str | Path) -> dict[str, float]:
    """Raw per-probe features. Descriptors are assigned SHARD-RELATIVELY
    (see shard_descriptors) because absolute thresholds collapse a shard's
    real variation into one or two buckets (2026-08-15 defect)."""
    import soundfile as sf
    y, sr = sf.read(str(path))
    if y.ndim > 1:
        y = y.mean(axis=1)
    y = y[: sr * 6]
    if len(y) < sr // 4 or float(np.abs(y).max()) < 1e-5:
        return {"centroid": 0.0, "flatness": 0.0, "bandwidth": 0.0,
                "sustain": 0.0, "silent": 1.0}

    S = np.abs(np.fft.rfft(y))
    freqs = np.fft.rfftfreq(len(y), 1 / sr)
    tot = S.sum() + 1e-12
    centroid = float((S * freqs).sum() / tot)
    bandwidth = float(np.sqrt(((freqs - centroid) ** 2 * S).sum() / tot))
    logS = np.log(S + 1e-12)
    flatness = float(np.exp(logS.mean()) / (S.mean() + 1e-12))

    # sustain: energy in the note's tail vs its head (pluck <-> pad)
    env = np.abs(y)
    k = max(1, sr // 100)
    env = np.convolve(env, np.ones(k) / k, mode="same")
    half = len(env) // 2
    sustain = float(env[half:].mean() / (env[:half].mean() + 1e-9))
    return {"centroid": centroid, "flatness": flatness, "bandwidth": bandwidth,
            "sustain": sustain, "silent": 0.0}


def _tercile_words(values: dict[str, float], low: str, mid: str, high: str
                   ) -> dict[str, str]:
    """Assign words by within-shard rank terciles (ties broken by order)."""
    order = sorted(values, key=lambda k: values[k])
    n = len(order)
    out: dict[str, str] = {}
    for rank, key in enumerate(order):
        if n < 3:
            out[key] = mid
        elif rank < n / 3:
            out[key] = low
        elif rank < 2 * n / 3:
            out[key] = mid
        else:
            out[key] = high
    return out


def shard_descriptors(paths_by_name: dict[str, str]
                      ) -> tuple[dict[str, list[str]], dict[str, dict[str, float]]]:
    """Descriptor triples for a whole shard, computed relative to the shard's
    own feature distribution so characterizations always spread out."""
    feats = {n: audio_features(p) for n, p in paths_by_name.items()}
    # ~8 of 282 library wavetables render silent deterministically (empty or
    # degenerate tables). They are labeled "'name': silent — no" and never
    # shortlisted: correct behavior, not a render failure.
    live = {n: f for n, f in feats.items() if not f.get("silent")}
    if not live:
        return {n: ["silent"] for n in feats}, feats
    bright = _tercile_words({n: f["centroid"] for n, f in live.items()},
                            "dark", "warm", "bright")
    texture = _tercile_words({n: f["flatness"] for n, f in live.items()},
                             "pure", "harmonic", "noisy")
    body = _tercile_words({n: f["sustain"] for n, f in live.items()},
                          "plucky", "even-bodied", "sustained")
    width = _tercile_words({n: f["bandwidth"] for n, f in live.items()},
                           "narrow", "full", "wide")
    words: dict[str, list[str]] = {}
    for n in feats:
        if feats[n].get("silent"):
            words[n] = ["silent"]
            continue
        # 4 shard-relative axes => 81 possible triples-of-four; the width axis
        # only appears when it adds information beyond brightness.
        trio = [bright[n], texture[n], body[n]]
        if width[n] != {"dark": "narrow", "warm": "full", "bright": "wide"}[bright[n]]:
            trio.append(width[n])
        words[n] = trio
    return words, feats


def descriptor_spread_ok(words: dict[str, list[str]], max_share: float = 0.40
                         ) -> tuple[bool, float, tuple]:
    """Gate: no single descriptor triple may cover > max_share of the shard."""
    from collections import Counter
    counts = Counter(tuple(w) for w in words.values())
    top, n = counts.most_common(1)[0]
    share = n / max(1, len(words))
    return share <= max_share, share, top


def audio_descriptors(path: str | Path) -> tuple[dict[str, float], list[str]]:
    """Back-compat single-file entry (absolute buckets). Prefer
    shard_descriptors — this remains for gates that only need features."""
    f = audio_features(path)
    if f.get("silent"):
        return f, ["silent"]
    words = ["bright" if f["centroid"] > 3200 else
             ("warm" if f["centroid"] > 1600 else "dark"),
             "noisy" if f["flatness"] > 0.30 else
             ("pure" if f["flatness"] < 0.02 else "harmonic"),
             "sustained" if f["sustain"] > 0.9 else "plucky"]
    return f, words


def terse_verdict(name: str, words: list[str], selected: bool, hedged: bool) -> str:
    desc = ", ".join(words)
    if not selected:
        return f"'{name}': {desc} — no"
    if hedged:
        return f"'{name}': {desc} — closest available, shortlisting"
    return f"'{name}': {desc} — shortlist"


# ---------------------------------------------------------------------------
# Optional single batched LLM call: one sentence per shortlisted candidate
# ---------------------------------------------------------------------------

def shortlist_sentences(
    shortlisted: list[str],
    descriptors: dict[str, list[str]],
    archetype: str,
    server: str | None,
    model: str | None,
) -> dict[str, str]:
    if not shortlisted:
        return {}
    fallback = {
        n: (f"'{n}' ({', '.join(descriptors.get(n, ['candidate']))}) is the closest "
            f"match to the target {archetype} sound in this slice.")
        for n in shortlisted
    }
    if not server:
        return fallback
    try:
        import httpx
        lines = "\n".join(f"- {n}: {', '.join(descriptors.get(n, []))}" for n in shortlisted)
        prompt = (
            f"You are writing a search agent's shortlist notes for a {archetype} synth "
            f"sound. For each shortlisted wavetable below, write EXACTLY ONE sentence "
            f"(under 25 words) on what its measured character contributes toward the "
            f"target. Reply as JSON: {{\"<name>\": \"<sentence>\"}}.\n\n{lines}"
        )
        r = httpx.post(f"{server.rstrip('/')}/v1/chat/completions", json={
            "model": model or "default",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 90 * len(shortlisted) + 120,
            "temperature": 0.4,
        }, timeout=240)
        text = r.json()["choices"][0]["message"]["content"]
        m = re.search(r"\{.*\}", text, re.S)
        parsed = json.loads(m.group(0)) if m else {}
        return {n: str(parsed.get(n) or fallback[n]) for n in shortlisted}
    except Exception:
        return fallback


# ---------------------------------------------------------------------------
# Record builder
# ---------------------------------------------------------------------------

@dataclass
class SearchResultV3:
    record: dict | None
    shortlist: list[str]
    shard_start: int
    shard_end: int
    final_message: str | None = None
    meta: dict = field(default_factory=dict)


def _slugify(s: str) -> str:
    return (re.sub(r"[^a-zA-Z0-9]+", "_", s).strip("_") or "unnamed")[:80]


def build_search_record_v3(
    *,
    sample_id: str,
    agent_idx: int,
    archetype: str,
    target_audio_path: Path,
    gt_wavetable_names: list[str],
    shard_start: int,
    shard_end: int,
    name_to_idx: dict[str, int],
    idx_to_name: dict[int, str],
    embedder: ClapEmbedder,
    dawfarm: DawFarmRolloutCtx,
    midi_path: str,
    probe_audio_dir: Path,
    stage2_server: str | None = None,
    stage2_model: str | None = None,
    candidates_per_batch: int = 8,
    clap_threshold: float = 0.97,
    floor_k: int = 2,
    shortlist_dir: Path | None = None,
    pool_only: bool = False,
) -> SearchResultV3:
    """Build one v3 search record with real env execution (dawfarm required)."""
    empty = SearchResultV3(record=None, shortlist=[], shard_start=shard_start,
                           shard_end=shard_end)
    shard = [idx_to_name[i] for i in range(shard_start, shard_end) if i in idx_to_name]
    if not shard:
        return empty

    # --- names listing (env is the source of truth; verify against host lib)
    list_cmd = _wrap_as_bash(build_list_wavetables_slice_snippet(shard_start, shard_end))
    lres = dawfarm.real_exec(list_cmd, f"wavetable slice {shard_start}-{shard_end}")
    container_rows = json.loads(lres.stdout.strip().splitlines()[-1])["wavetables"]
    container_names = [w["name"] for w in container_rows]
    if container_names != shard:
        raise RuntimeError(
            f"container wavetable slice {shard_start}-{shard_end} disagrees with host library")

    # --- render candidate probes (fixed full-state snippet, transcribed melody)
    probe_out_dir = str(probe_audio_dir / f"{sample_id}_agent{agent_idx}")
    display_probe_dir = dawfarm.cw("", f"search_probes_a{agent_idx}")
    all_idxs = [name_to_idx[n] for n in shard if n in name_to_idx]
    render_snippet = build_render_probes_snippet(
        idxs=all_idxs, out_dir=display_probe_dir, midi_path=midi_path)
    render_cmd = "DISPLAY=:99 " + _wrap_as_bash(render_snippet)
    rres = dawfarm.real_exec(render_cmd, f"probe render agent{agent_idx}", timeout=900.0)
    rendered = json.loads(rres.stdout.strip().splitlines()[-1])["rendered"]
    if {e["name"] for e in rendered} != set(shard):
        raise RuntimeError(f"probe render mismatch agent{agent_idx}")
    dawfarm.fetch_dir(display_probe_dir, probe_out_dir)

    name_to_host = {}
    name_to_display = {}
    for e in rendered:
        fname = Path(e["out"]).name
        name_to_host[e["name"]] = f"{probe_out_dir}/{fname}"
        name_to_display[e["name"]] = e["out"]
    missing = [p for p in name_to_host.values() if not Path(p).exists()]
    if missing:
        raise RuntimeError(f"probe fetch incomplete: {len(missing)} missing")

    # --- shard-relative descriptors + DISCRIMINABILITY GATE
    words_by_name, raw_feats = shard_descriptors(
        {n: name_to_host[n] for n in shard})
    ok, top_share, top_trio = descriptor_spread_ok(words_by_name)
    if not ok:
        print(f"[search v3] WARNING {sample_id} a{agent_idx}: descriptor triple "
              f"{top_trio} covers {100 * top_share:.0f}% of the shard "
              f"(>40%) — verdicts under-discriminate", file=sys.stderr, flush=True)
    feats: dict[str, tuple[dict, list[str]]] = {
        n: (raw_feats[n], words_by_name[n]) for n in shard}
    centroids = [f[0]["centroid"] for f in feats.values()]
    spread = max(centroids) - min(centroids)
    if spread < MIN_CENTROID_SPREAD_HZ:
        raise RuntimeError(
            f"DISCRIMINABILITY GATE FAILED for {sample_id} agent{agent_idx}: "
            f"probe centroid spread {spread:.0f} Hz < {MIN_CENTROID_SPREAD_HZ:.0f} Hz "
            f"— probe audio carries no wavetable information; refusing to build "
            f"a record from it.")

    # --- GT-ingredient probes (builder-side label evidence, NOT in the record)
    gt_probe_dir_display = dawfarm.cw("", f"gt_probes_a{agent_idx}")
    gt_probe_dir_host = str(probe_audio_dir / f"{sample_id}_agent{agent_idx}_gt")
    gt_snippet = build_render_probes_snippet(
        names=gt_wavetable_names, out_dir=gt_probe_dir_display, midi_path=midi_path)
    gres = dawfarm.real_exec("DISPLAY=:99 " + _wrap_as_bash(gt_snippet),
                             f"gt probes agent{agent_idx}", timeout=600.0)
    gt_rendered = json.loads(gres.stdout.strip().splitlines()[-1])["rendered"]
    dawfarm.fetch_dir(gt_probe_dir_display, gt_probe_dir_host)
    gt_paths = [f"{gt_probe_dir_host}/{Path(e['out']).name}" for e in gt_rendered]
    gt_paths = [p for p in gt_paths if Path(p).exists()]
    if not gt_paths:
        raise RuntimeError(f"no GT ingredient probes rendered for {sample_id}")

    # --- evidence-based labels
    gt_embs = [embedder.embed_audio_path(Path(p)) for p in gt_paths]
    def _cos(a, b):
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
    best_sim: dict[str, float] = {}
    for n in shard:
        emb = embedder.embed_audio_path(Path(name_to_host[n]))
        best_sim[n] = max(_cos(emb, g) for g in gt_embs)
    above = [n for n in shard if best_sim[n] >= clap_threshold]
    ranked = sorted(shard, key=lambda n: best_sim[n], reverse=True)
    floor = [n for n in ranked[:floor_k] if n not in above]
    hedged = not above  # nothing cleared the bar; floor picks carry hedged wording
    shortlist = above + floor
    # preserve shard order for stable verdict text
    shortlist = [n for n in shard if n in set(shortlist)]

    verdict_meta = {
        "label_mode": "evidence_v3",
        "clap_threshold": clap_threshold,
        "floor_engaged": bool(floor),
        "hedged": hedged,
        "best_sim": {n: round(best_sim[n], 4) for n in shortlist},
        "gate_spread_hz": round(spread, 1),
    }

    final_shortlist = list(shortlist)
    shortlist_str = ", ".join(f'"{n}"' for n in final_shortlist)

    batches = [shard[i:i + candidates_per_batch]
               for i in range(0, len(shard), candidates_per_batch)]

    def _build_final_content() -> str:
        """The subagent's final message. Built identically whether or not the
        standalone search record is kept for training — at inference the real
        subagent always emits this full form, so the main agent's task_result
        must never see a truncated variant (contract drift, 2026-08-15)."""
        last_batch_verdicts = "\n".join(
            terse_verdict(n, feats[n][1], n in set(final_shortlist), hedged)
            for n in batches[-1]) if batches else ""
        sentences = shortlist_sentences(
            final_shortlist, {n: feats[n][1] for n in shard}, archetype,
            stage2_server, stage2_model)
        parts = []
        if last_batch_verdicts:
            parts.append(last_batch_verdicts)
        if hedged and final_shortlist:
            parts.append(
                "No candidate clears the similarity bar in this slice; flagging "
                "the closest matches as best-available for the judge.")
        parts.extend(sentences[n] for n in final_shortlist)
        parts.append(
            f"Shortlist: [{shortlist_str}]. {len(final_shortlist)} candidate(s) "
            f"flagged for the judge agent.")
        return "\n\n".join(parts)

    if pool_only:
        return SearchResultV3(record=None, shortlist=final_shortlist,
                              shard_start=shard_start, shard_end=shard_end,
                              final_message=_build_final_content(),
                              meta=verdict_meta)

    # --- conversation ---
    messages: list[dict] = []
    audio_assets: list[str] = [str(target_audio_path)]
    messages.append({
        "role": "system",
        "content": oc.system_message(oc.AGENT_PROMPTS["wavetable_search"],
                                     cwd=dawfarm.cw("", "")),
    })
    messages.append({
        "role": "user",
        "content": "<audio>\n" + oc.search_dispatch_prompt(
            target_audio_path, midi_path, shard_start, shard_end),
    })

    messages.append({"role": "assistant",
                     "content": f"Listing candidates at indices {shard_start}-{shard_end - 1}."})
    messages.append(oc_bash_call_msg(list_cmd))
    messages.append(oc_bash_response_msg(lres.stdout))

    messages.append({"role": "assistant",
                     "content": f"Rendering all {len(shard)} candidates with the transcribed melody."})
    messages.append(oc_bash_call_msg(render_cmd))
    messages.append(oc_bash_response_msg(rres.stdout))

    pending_verdicts: str | None = None
    for bi, batch in enumerate(batches):
        intro = f"Listening to batch {bi + 1} of {len(batches)}."
        if pending_verdicts:
            intro = f"{pending_verdicts}\n\n{intro}"
        messages.append({"role": "assistant", "content": intro})
        for n in batch:
            audio_assets.append(name_to_host[n])
            messages.append(oc_read_call_msg(name_to_display[n]))
            messages.append(oc_read_audio_response_msg(
                name_to_host[n], display_path=name_to_display[n]))
        lines = [terse_verdict(n, feats[n][1], n in set(final_shortlist), hedged)
                 for n in batch]
        pending_verdicts = "\n".join(lines)

    final_content = _build_final_content()
    messages.append({"role": "assistant", "content": final_content})

    shortlist_path = None
    if shortlist_dir is not None:
        shortlist_dir.mkdir(parents=True, exist_ok=True)
        agent_id = make_agent_id(sample_id, "wavetable_search", agent_idx)
        shortlist_path = str(shortlist_dir / f"{agent_id}.md")
        with open(shortlist_path, "w") as f:
            f.write(final_content + "\n")

    gt_set = set(gt_wavetable_names)
    record = {
        "id": f"{sample_id}_r1_agent{agent_idx}_search",
        "task_type": "search_v3",
        "tools": OC_TOOLS_JSON,
        "messages": messages,
        "audios": audio_assets,
        "meta": {
            "pipeline_version": "v3_search",
            "sample_id": sample_id,
            "archetype": archetype,
            "shard_start": shard_start,
            "shard_end": shard_end,
            "candidates_per_batch": candidates_per_batch,
            "gt_in_shard": [n for n in shard if n in gt_set],
            "final_shortlist": final_shortlist,
            "gt_on_shortlist": [n for n in final_shortlist if n in gt_set],
            "shortlist_output_file": shortlist_path,
            **verdict_meta,
        },
    }
    assert_valid_oc_record(record)
    return SearchResultV3(record=record, shortlist=final_shortlist,
                          shard_start=shard_start, shard_end=shard_end,
                          final_message=final_content, meta=verdict_meta)


# ---------------------------------------------------------------------------
# Slim standalone CLI (single-sample smoke; the unified v4 builder is the
# production entry point)
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--wavetable-lib", type=Path, default=Path("data/wavetable_lib.json"))
    ap.add_argument("--out-jsonl", type=Path, required=True)
    ap.add_argument("--session", required=True, help="daw-farm container name")
    ap.add_argument("--stage2-server", default=None)
    ap.add_argument("--stage2-model", default=None)
    ap.add_argument("--shard-size", type=int, default=48)
    ap.add_argument("--max-samples", type=int, default=1)
    args = ap.parse_args()

    from scripts.agent_sft_common import (
        extract_gt_wavetable_names, load_manifest_entries, load_wavetable_lib)
    from maestro.reaper.dawfarm import DockerSession

    lib = load_wavetable_lib(args.wavetable_lib)
    names_sorted = sorted(w.get("name", "") for w in lib if w.get("name"))
    name_to_idx = {n: i for i, n in enumerate(names_sorted)}
    idx_to_name = {i: n for n, i in name_to_idx.items()}
    embedder = ClapEmbedder.create("cuda")

    entries = load_manifest_entries(args.manifest)[: args.max_samples]
    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_jsonl, "w") as out:
        for entry in entries:
            sid = entry["sample_id"]
            session = DockerSession(args.session)
            dawfarm = DawFarmRolloutCtx(session=session, sample_id=sid)
            res = build_search_record_v3(
                sample_id=sid, agent_idx=1,
                archetype=entry.get("archetype", "lead"),
                target_audio_path=Path(entry["gt_wav"]),
                gt_wavetable_names=extract_gt_wavetable_names(Path(entry["target_preset"])),
                shard_start=0, shard_end=args.shard_size,
                name_to_idx=name_to_idx, idx_to_name=idx_to_name,
                embedder=embedder, dawfarm=dawfarm,
                midi_path=entry.get("notes_json") or entry.get("source_midi"),
                probe_audio_dir=args.out_jsonl.parent / "probes",
                stage2_server=args.stage2_server, stage2_model=args.stage2_model,
            )
            if res.record:
                out.write(json.dumps(res.record, ensure_ascii=False) + "\n")
                print(f"{sid}: shortlist={res.shortlist} meta={res.meta}")


if __name__ == "__main__":
    main()
