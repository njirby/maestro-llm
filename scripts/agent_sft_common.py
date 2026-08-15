from __future__ import annotations

import copy
import hashlib
import json
import logging
import random
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from maestro.render.vital import SAMPLE_RATE
from maestro.render.dawdreamer import render_preset_audio
from scripts.build_wavetable_retrieval_baseline import (
    _build_probe_preset,
    _embed_clap,
    _extract_gt_wavetable_names,
    _load_clap,
    _load_init_preset,
    _load_wavetable_lib,
)


# --------------------------------------------------------------------------
# Sub-agent dispatch convention (matches claw-code's runtime harness)
#
# claw-code generates `agent-<nanosecond_timestamp>` IDs and writes both an
# output file and a manifest file per dispatch. The parent's tool_response
# carries `agentId, status, outputFile, manifestFile, createdAt, ...` —
# the parent reads `outputFile` via bash cat to consume the result.
#
# For SFT data we use deterministic timestamp-shaped IDs derived from
# (sample_id, kind, salts) so re-runs produce identical records but the
# format matches what claw-code emits at runtime.
# --------------------------------------------------------------------------


def make_agent_id(sample_id: str, kind: str, *salts: str | int) -> str:
    """Deterministic timestamp-shaped agent ID. Matches claw-code's
    `agent-<int>` format. The integer is sha1(sample+kind+salts) → first
    16 hex chars → ~10^19, indistinguishable from real ns-precision time."""
    payload = f"{sample_id}:{kind}:" + ":".join(str(s) for s in salts)
    h = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]
    return f"agent-{int(h, 16)}"


def make_agent_manifest(
    *,
    agent_id: str,
    subagent_type: str,
    output_file: str | Path,
    manifest_file: str | Path,
    prompt: str = "",
    created_at: str = "",
    status: str = "completed",
    extra: dict | None = None,
) -> dict:
    """Build the manifest dict written alongside an agent's output file.

    Mirrors claw-code's AgentOutput shape (agentId, subagentType, status,
    outputFile, manifestFile, createdAt, startedAt). At SFT-build time
    the agent has already finished, so `status` is "completed" and
    createdAt == startedAt.
    """
    if not created_at:
        # Deterministic build-time pseudo-timestamp derived from agent_id.
        # Real ISO timestamps would break repeatability; we keep a
        # claw-code-shaped string anchored to the agent's identity.
        created_at = f"build-time:{agent_id}"
    payload: dict = {
        "agentId": agent_id,
        "subagentType": subagent_type,
        "status": status,
        "outputFile": str(output_file),
        "manifestFile": str(manifest_file),
        "createdAt": created_at,
        "startedAt": created_at,
    }
    if prompt:
        payload["prompt"] = prompt[:1000]
    if extra:
        payload.update(extra)
    return payload


def write_agent_manifest(
    *,
    agent_id: str,
    subagent_type: str,
    output_file: str | Path,
    manifest_file: str | Path,
    prompt: str = "",
    extra: dict | None = None,
) -> dict:
    """Write the manifest JSON to disk alongside the output file. Returns
    the manifest dict (for inclusion in the dispatch tool_response)."""
    manifest = make_agent_manifest(
        agent_id=agent_id,
        subagent_type=subagent_type,
        output_file=output_file,
        manifest_file=manifest_file,
        prompt=prompt,
        extra=extra,
    )
    Path(manifest_file).parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_file, "w") as f:
        json.dump(manifest, f)
        f.write("\n")
    return manifest


def slugify(s: str, max_len: int = 80) -> str:
    out = re.sub(r"[^a-zA-Z0-9]+", "_", s).strip("_")
    if not out:
        out = "unnamed"
    return out[:max_len]


def load_manifest_entries(manifest_path: Path, max_samples: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(manifest_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    if max_samples is not None:
        rows = rows[:max_samples]
    return rows


def load_index_rows(index_meta: Path) -> list[dict[str, Any]]:
    with open(index_meta) as f:
        meta = json.load(f)
    rows = meta.get("rows", [])
    if not isinstance(rows, list) or not rows:
        raise RuntimeError(f"No rows found in index metadata: {index_meta}")
    return rows


def select_probe_rows_by_name(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    selected: dict[str, dict[str, Any]] = {}
    for row in rows:
        name = str(row.get("wavetable_name", "")).strip()
        if not name:
            continue
        frame_idx = int(row.get("frame_idx", 0))
        prev = selected.get(name)
        if prev is None or frame_idx < int(prev.get("frame_idx", 0)):
            selected[name] = row
    return selected


@dataclass
class ClapEmbedder:
    device: str
    model: Any
    processor: Any
    _cache: dict[str, np.ndarray]

    @classmethod
    def create(cls, device: str, cache_path: Path | None = None) -> "ClapEmbedder":
        model, processor = _load_clap(device)
        inst = cls(device=device, model=model, processor=processor, _cache={})
        if cache_path:
            n = inst.load_cache(cache_path)
            # Two-phase runs pre-embed everything; a miss means a coverage
            # leak worth seeing in the logs.
            inst.warn_on_miss = n > 0
        return inst

    warn_on_miss: bool = False

    def embed_audio_path(self, path: Path) -> np.ndarray:
        key = str(path.resolve())
        if key in self._cache:
            return self._cache[key]
        if self.warn_on_miss:
            import sys as _sys
            print(f"[clap] CACHE MISS (embedding on {self.device}): {key}",
                  file=_sys.stderr, flush=True)
        import time as _t
        _t0 = _t.monotonic()
        audio, sr = sf.read(path, always_2d=True)
        audio = audio.T.astype(np.float32)
        emb = _embed_clap(audio, sr, self.model, self.processor, self.device)
        self._cache[key] = emb
        record_timing("clap_embed", _t.monotonic() - _t0)
        return emb

    def cosine_paths(self, a: Path, b: Path) -> float:
        ea = self.embed_audio_path(a)
        eb = self.embed_audio_path(b)
        an = np.linalg.norm(ea) + 1e-12
        bn = np.linalg.norm(eb) + 1e-12
        return float(np.dot(ea, eb) / (an * bn))

    def save_cache(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(str(path), keys=list(self._cache.keys()),
                            **{f"emb_{i}": v for i, v in enumerate(self._cache.values())})

    def load_cache(self, path: Path) -> int:
        if not path.exists():
            return 0
        data = np.load(str(path), allow_pickle=False)
        keys = list(data["keys"])
        loaded = 0
        for i, k in enumerate(keys):
            emb_key = f"emb_{i}"
            if emb_key in data:
                self._cache[str(k)] = data[emb_key]
                loaded += 1
        return loaded


def ensure_candidate_probes_for_names(
    names: list[str],
    wavetable_lib: list[dict[str, Any]],
    selected_rows: dict[str, dict[str, Any]],
    out_dir: Path,
    cache: dict[str, Path],
    probe_archetype: str = "lead",
    probe_tail_s: float = 1.0,
    trim_min_duration_s: float = 0.5,
    notes: list | None = None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    missing = [n for n in names if n not in cache and n in selected_rows]
    if not missing:
        return

    init_preset = _load_init_preset()
    if notes is None:
        from maestro.render.dawdreamer import make_probe_notes
        notes = make_probe_notes(probe_archetype)

    for name in missing:
        row = selected_rows[name]
        source_idx = int(row["source_wavetable_idx"])
        frame_idx = int(row.get("frame_idx", 0))
        wt = wavetable_lib[source_idx]
        fname = f"{slugify(name)}__src{source_idx:04d}_f{frame_idx:03d}.wav"
        path = out_dir / fname
        if not path.exists():
            preset = _build_probe_preset(init_preset, wt, frame_idx)
            render_preset_audio(preset, notes, out_path=path, tail_s=probe_tail_s)
        cache[name] = path


def build_oracle_mix_candidates(
    gt_names: list[str],
    universe_names: list[str],
    mix_size: int,
    rng: random.Random,
    hard_negative_names: list[str] | None = None,
) -> list[str]:
    mix_size = max(1, int(mix_size))
    universe_set = set(universe_names)
    gt_present = [n for n in list(dict.fromkeys(gt_names)) if n in universe_set]
    if not gt_present:
        return []

    chosen_gts = gt_present[:mix_size]
    exclude = set(gt_present)

    negatives: list[str] = []
    if hard_negative_names:
        for n in hard_negative_names:
            if n in exclude or n not in universe_set:
                continue
            if n not in negatives:
                negatives.append(n)
            if len(negatives) >= max(0, mix_size - len(chosen_gts)):
                break

    if len(negatives) < max(0, mix_size - len(chosen_gts)):
        pool = [n for n in universe_names if n not in exclude and n not in negatives]
        rng.shuffle(pool)
        need = max(0, mix_size - len(chosen_gts) - len(negatives))
        negatives.extend(pool[:need])

    candidates = chosen_gts + negatives[: max(0, mix_size - len(chosen_gts))]
    rng.shuffle(candidates)
    return candidates


def choose_candidate_pool(
    *,
    sample_id: str,
    query_audio_path: Path,
    gt_names: list[str],
    universe_names: list[str],
    candidate_source: str,
    candidate_limit: int,
    oracle_hard_pool: int,
    seed: int,
    clap_embedder: ClapEmbedder,
    selected_rows_meta: list[dict[str, Any]],
    shortlist_data: dict[str, Any] | None,
) -> list[str]:
    if candidate_source == "all":
        return universe_names[:candidate_limit]

    if shortlist_data is None:
        raise RuntimeError("shortlist_data required for clap_topn/oracle_mix8")

    q = clap_embedder.embed_audio_path(query_audio_path)
    qn = np.linalg.norm(q) + 1e-12
    rows_emb = shortlist_data["embeddings"]
    rows_norm = shortlist_data["norms"]
    row_scores = (rows_emb @ q) / (rows_norm * qn)

    # collapse rows->name max score
    name_scores: dict[str, float] = {}
    for score, row in zip(row_scores.tolist(), selected_rows_meta):
        name = str(row.get("wavetable_name", "")).strip()
        if not name:
            continue
        prev = name_scores.get(name)
        if prev is None or score > prev:
            name_scores[name] = float(score)

    ranked = [n for n, _ in sorted(name_scores.items(), key=lambda kv: kv[1], reverse=True)]

    if candidate_source == "clap_topn":
        return ranked[:candidate_limit]

    sid_seed = int(hashlib.sha1(sample_id.encode("utf-8")).hexdigest()[:8], 16)
    rng = random.Random(int(seed) + sid_seed)
    hard_pool = ranked[: max(int(oracle_hard_pool), int(candidate_limit) * 2)]
    return build_oracle_mix_candidates(gt_names, universe_names, candidate_limit, rng, hard_pool)


def build_disjoint_shards(candidates: list[str], num_agents: int) -> list[list[str]]:
    num_agents = max(1, int(num_agents))
    shards: list[list[str]] = [[] for _ in range(num_agents)]
    for i, c in enumerate(candidates):
        shards[i % num_agents].append(c)
    return shards


def build_candidate_audio_assets(
    names: list[str],
    candidate_audio_map: dict[str, Path],
    id_prefix: str = "C",
) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for i, name in enumerate(names, start=1):
        if name not in candidate_audio_map:
            continue
        out.append(
            {
                "candidate_id": f"{id_prefix}{i}",
                "wavetable_name": name,
                "audio_path": str(candidate_audio_map[name]),
            }
        )
    return out


def extract_gt_wavetable_names(target_preset_path: Path) -> list[str]:
    return _extract_gt_wavetable_names(target_preset_path)


def build_clap_shortlist_data(index_npy: Path, index_rows_meta: list[dict[str, Any]]) -> dict[str, Any]:
    idx = np.load(index_npy)
    emb = idx["embeddings"].astype(np.float32)
    if len(emb) != len(index_rows_meta):
        raise RuntimeError(
            f"Index metadata rows ({len(index_rows_meta)}) != embedding rows ({len(emb)})"
        )
    return {"embeddings": emb, "norms": np.linalg.norm(emb, axis=1) + 1e-12}


def load_wavetable_lib(path: Path) -> list[dict[str, Any]]:
    return _load_wavetable_lib(path)


def build_name_embedding_map(
    index_embeddings: np.ndarray,
    index_rows_meta: list[dict[str, Any]],
) -> dict[str, np.ndarray]:
    """Build a {wavetable_name → L2-normalized embedding} map from the index.

    Mean-pools across frames for multi-frame wavetables. Returns normalized
    vectors ready for cosine similarity via dot product.

    Reused by: build_gt_similarity_pool, search agent CLAP selection,
    main agent pool coverage check.
    """
    from collections import defaultdict as _defaultdict
    name_to_rows: dict[str, list[int]] = _defaultdict(list)
    for i, row in enumerate(index_rows_meta):
        name_to_rows[row["wavetable_name"]].append(i)

    name_to_emb: dict[str, np.ndarray] = {}
    for name, idxs in name_to_rows.items():
        emb = index_embeddings[idxs].mean(axis=0)
        name_to_emb[name] = emb / (np.linalg.norm(emb) + 1e-12)
    return name_to_emb


def is_clap_selected(
    candidate_name: str,
    gt_wavetable_names: list[str],
    name_to_emb: dict[str, np.ndarray],
    threshold: float = 0.97,
) -> bool:
    """CLAP-based selection: is this candidate similar enough to any GT wavetable?

    Returns True if candidate is an exact GT match OR has cosine similarity >= threshold
    to at least one GT wavetable in the embedding space.
    """
    if candidate_name in gt_wavetable_names:
        return True
    if candidate_name not in name_to_emb:
        return False
    cand_emb = name_to_emb[candidate_name]
    for gt in gt_wavetable_names:
        if gt in name_to_emb:
            sim = float(cand_emb @ name_to_emb[gt])
            if sim >= threshold:
                return True
    return False


def build_gt_similarity_pool(
    gt_wavetable_names: list[str],
    index_embeddings: np.ndarray,
    index_norms: np.ndarray,
    index_rows_meta: list[dict[str, Any]],
    top_k: int = 48,
    rng: random.Random | None = None,
) -> list[str]:
    """Build a candidate pool from the top-K wavetables most CLAP-similar to any GT wavetable.

    Uses the wavetable index (bare probes through default preset) for apples-to-apples
    comparison. GT wavetables are always included. Returns wavetable names, shuffled.

    Parameters
    ----------
    gt_wavetable_names : list[str]
        1-3 GT wavetable names (from the target preset's active oscillators).
    index_embeddings : np.ndarray
        (N_rows, dim) CLAP embeddings from wt_index.npz.
    index_norms : np.ndarray
        (N_rows,) L2 norms of index_embeddings (with epsilon).
    index_rows_meta : list[dict]
        Per-row metadata from wt_index_meta.json (wavetable_name, frame_idx, etc.).
    top_k : int
        Maximum pool size (default 48). GT wavetables count toward this.
    rng : random.Random | None
        For shuffling. If None, pool is returned in similarity order.

    Returns
    -------
    list[str]
        Wavetable names (unique, shuffled if rng provided). GT names always included.
    """
    # Use shared embedding helper
    name_to_emb = build_name_embedding_map(index_embeddings, index_rows_meta)
    all_names = sorted(name_to_emb.keys())

    gt_name_set = set(gt_wavetable_names)
    gt_emb_list = [name_to_emb[gn] for gn in gt_wavetable_names if gn in name_to_emb]

    if not gt_emb_list:
        # GT not in index — return GT names + random fill
        pool = list(gt_wavetable_names)
        others = [n for n in all_names if n not in gt_name_set]
        if rng:
            rng.shuffle(others)
        pool.extend(others[: max(0, top_k - len(pool))])
        if rng:
            rng.shuffle(pool)
        return pool

    gt_stack = np.stack(gt_emb_list)  # (n_gt, dim)

    # Cosine similarity: each GT against all names (embeddings already normalized)
    name_emb_matrix = np.stack([name_to_emb[n] for n in all_names])

    sim_matrix = gt_stack @ name_emb_matrix.T  # (n_gt, n_names) — both sides normalized
    max_sim = sim_matrix.max(axis=0)  # (n_names,)

    # Rank by max similarity, excluding GT
    scored = [
        (all_names[i], float(max_sim[i]))
        for i in range(len(all_names))
        if all_names[i] not in gt_name_set
    ]
    scored.sort(key=lambda x: -x[1])

    # Build pool: GT first, then top-K hard negatives
    pool = list(gt_wavetable_names)
    for name, _ in scored[: max(0, top_k - len(pool))]:
        pool.append(name)

    if rng:
        rng.shuffle(pool)
    return pool


ALLOWED_MESSAGE_ROLES = {"user", "assistant", "tool_call", "tool_response"}


def validate_ms_swift_multiturn_record(record: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    rid = str(record.get("id", ""))
    messages = record.get("messages")
    audios = record.get("audios")

    if not rid:
        errors.append("id_missing")
    if not isinstance(messages, list) or not messages:
        errors.append("messages_missing_or_empty")
        return errors
    if not isinstance(audios, list):
        errors.append("audios_missing_or_not_list")
        audios = []

    if messages[0].get("role") != "user":
        errors.append("first_message_not_user")
    if messages[-1].get("role") != "assistant":
        errors.append("last_message_not_assistant")

    seen_tool_call = False
    audio_tag_count = 0
    prev_role: str | None = None
    for i, msg in enumerate(messages):
        if not isinstance(msg, dict):
            errors.append(f"message_{i}_not_object")
            continue

        role = msg.get("role")
        if role not in ALLOWED_MESSAGE_ROLES:
            errors.append(f"message_{i}_invalid_role:{role}")
        if role == "tool_call":
            seen_tool_call = True
        if role == "tool_response" and not seen_tool_call:
            errors.append(f"message_{i}_tool_response_before_tool_call")

        # Avoid structural chatter patterns that degraded quality in spot checks.
        # EXCEPTION: consecutive tool_call messages represent parallel tool dispatch
        # (multiple tool_use blocks from a single assistant turn in Anthropic /
        # claw-code protocol). Consecutive tool_response messages are the matching
        # parallel results. Both are allowed.
        if prev_role == role and role not in {"tool_call", "tool_response"}:
            errors.append(f"message_{i}_duplicate_adjacent_role:{role}")
        prev_role = str(role) if role is not None else None

        content = msg.get("content")
        if not isinstance(content, str):
            errors.append(f"message_{i}_content_not_string")
            continue
        tags = content.count("<audio>")
        # Multiple <audio> tags are only disallowed in user/assistant turns; a tool_response
        # may legitimately return a batch of audio previews in a single message.
        if tags > 1 and role in {"user", "assistant"}:
            errors.append(f"message_{i}_multiple_audio_tags")
        audio_tag_count += tags

    if audio_tag_count != len(audios):
        errors.append(f"audio_tag_mismatch:tags={audio_tag_count}:audios={len(audios)}")

    for i, path in enumerate(audios):
        if not isinstance(path, str) or not path.strip():
            errors.append(f"audio_{i}_invalid_path")

    return errors


def assert_valid_ms_swift_multiturn_record(record: dict[str, Any]) -> None:
    errors = validate_ms_swift_multiturn_record(record)
    if errors:
        rid = str(record.get("id", "unknown"))
        raise ValueError(f"{rid}: " + "; ".join(errors))


# ---------------------------------------------------------------------------
# Shared snippet utilities
# ---------------------------------------------------------------------------

def _wrap_as_bash(python_code: str) -> str:
    """Wrap bare Python code in a shell-executable heredoc."""
    stripped = python_code.strip()
    if stripped.startswith("python"):
        return stripped
    return f"python - <<'PY'\n{stripped}\nPY"


# ---------------------------------------------------------------------------
# reapy REAPER-interaction helpers (embedded in generated snippets)
# ---------------------------------------------------------------------------

_REAPY_HELPER = """\
import json
import reapy
from reapy import reascript_api as RPR
"""

_BUILD_CHUNK_HELPER = """\
import struct as _struct

def build_vital_chunk(preset_json):
    json_bytes = json.dumps(preset_json, separators=(',', ':')).encode('utf-8')
    json_size = len(json_bytes)
    suffix = b'\\x00' * 17 + b'JUCEPrivateData' + b'\\x00' * 8
    total = 184 + json_size + len(suffix)
    prefix = bytearray(184)
    _struct.pack_into('<I', prefix, 0, total - 16)
    _struct.pack_into('<I', prefix, 4, 1)
    prefix[8:12] = b'VstW'
    _struct.pack_into('>I', prefix, 12, 8)
    _struct.pack_into('>I', prefix, 16, 1)
    prefix[24:28] = b'CcnK'
    _struct.pack_into('>I', prefix, 28, total - 40)
    prefix[32:36] = b'FBCh'
    _struct.pack_into('>I', prefix, 36, 2)
    prefix[40:44] = b'Vita'
    _struct.pack_into('>I', prefix, 44, 0x00010600)
    _struct.pack_into('>I', prefix, 180, json_size + 32)
    return bytes(prefix) + json_bytes + suffix
"""

_READ_CHUNK_HELPER = """\
import base64 as _b64

def _get_fx_chunk(track, fx_idx, param_name):
    result = RPR.TrackFX_GetNamedConfigParm(track, fx_idx, param_name, '', 4*1024*1024)
    ok = result[0]
    raw = result[-2]
    return ok, raw

def read_vital_preset(track_idx=0, fx_idx=0):
    with reapy.inside_reaper():
        track = RPR.GetTrack(0, track_idx)
        ok, raw = _get_fx_chunk(track, fx_idx, 'vst_chunk')
        if not ok or not raw:
            ok, raw = _get_fx_chunk(track, fx_idx, 'vst3_chunk')
        chunk = _b64.b64decode(raw)
    start = chunk.index(b'{"')
    end = chunk.rindex(b'}') + 1
    return json.loads(chunk[start:end])
"""


# ---------------------------------------------------------------------------
# claw-code tool response helpers
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# opencode-contract emission layer (new builders route through these; the
# legacy claw-code helpers below remain for old scripts)
# ---------------------------------------------------------------------------

from scripts import opencode_contract as oc  # noqa: E402


def oc_tool_call_msg(name: str, arguments: dict) -> dict:
    return {"role": "tool_call", "content": oc.tool_call(name, arguments)}


def oc_bash_call_msg(command: str, timeout_ms: int | None = None,
                     workdir: str | None = None) -> dict:
    return {"role": "tool_call", "content": oc.bash_call(command, timeout_ms, workdir)}


def oc_bash_response_msg(stdout: str, exit_code: int = 0, stderr: str = "") -> dict:
    return {"role": "tool_response",
            "content": oc.bash_output(stdout, exit_code=exit_code, stderr=stderr)}


def oc_read_call_msg(file_path: str) -> dict:
    return {"role": "tool_call", "content": oc.read_call(file_path)}


def oc_read_audio_response_msg(audio_path: str | Path,
                               display_path: str | None = None) -> dict:
    """Audio read result: <audio> placeholder + attachment notice. The
    placeholder aligns positionally with the record's `audios` list; at
    serving time the harness sends the clip as an input_audio part."""
    p = Path(audio_path)
    duration, sr = 0.0, 44100
    try:
        info = sf.info(str(p))
        duration, sr = float(info.duration), int(info.samplerate)
    except Exception:
        pass
    fname = Path(display_path).name if display_path else p.name
    return {"role": "tool_response",
            "content": "<audio>\n" + oc.read_output_audio(fname, duration, sr)}


def oc_task_call_msg(description: str, prompt: str, subagent_type: str) -> dict:
    return {"role": "tool_call",
            "content": oc.task_call(description, prompt, subagent_type)}


def oc_task_result_msg(session_id: str, final_text: str,
                       state: str = "completed") -> dict:
    return {"role": "tool_response", "content": oc.task_result(session_id, final_text, state)}


def oc_emit_listen_sequence(
    messages: list[dict],
    audio_assets: list[str],
    audio_path: str | Path,
    probe_stdout: str | None = None,
    display_path: str | None = None,
) -> None:
    """New-contract render-listen: plain-string bash result -> read -> audio.
    Call AFTER the assistant text and bash render tool_call; ``audio_path``
    must already be appended to ``audio_assets``."""
    path_str = str(display_path or audio_path)
    if probe_stdout is None:
        try:
            info = sf.info(str(audio_path))
            probe_stdout = (f"Rendered {path_str} "
                            f"({info.duration:.2f}s, {int(info.samplerate)} Hz)")
        except Exception:
            probe_stdout = f"Rendered {path_str}"
    messages.append(oc_bash_response_msg(probe_stdout))
    messages.append(oc_read_call_msg(path_str))
    messages.append(oc_read_audio_response_msg(audio_path, display_path=path_str))


# --- compat shims: legacy call-shape, new-contract emission -----------------
# Builders that historically used the claw-code helpers can switch to the
# opencode contract by importing these under the old names. They translate
# tool names and argument keys and emit plain-string outputs.

_OC_NAME_MAP = {"Bash": "bash", "Read": "read", "Skill": "skill", "Agent": "task"}


def oc_compat_tool_call(name: str, arguments: dict) -> dict:
    oc_name = _OC_NAME_MAP.get(name, name)
    args = dict(arguments)
    if oc_name == "read" and "file_path" in args:
        args = {"filePath": args["file_path"]}
    elif oc_name == "skill":
        args = {"name": args.get("skill") or args.get("name", "")}
    elif oc_name == "task":
        args = {
            "description": args.get("description", ""),
            "prompt": args.get("prompt", ""),
            "subagent_type": args.get("subagent_type", ""),
        }
    elif oc_name == "bash":
        out = {"command": args.get("command", "")}
        if args.get("timeout") is not None:
            t = args["timeout"]
            out["timeout"] = int(t * 1000) if t < 10000 else int(t)
        if args.get("cwd") or args.get("workdir"):
            out["workdir"] = args.get("cwd") or args.get("workdir")
        args = out
    return {"role": "tool_call", "content": oc.tool_call(oc_name, args)}


def oc_compat_bash_response(stdout: str, stderr: str = "",
                            interrupted: bool = False) -> dict:
    exit_code = 1 if (stderr and stderr.strip()) or interrupted else 0
    return {"role": "tool_response",
            "content": oc.bash_output(stdout, exit_code=exit_code, stderr=stderr)}


def oc_compat_read_response_audio(audio_path: str | Path | None = None,
                                  display_path: str | None = None) -> dict:
    """Audio read result: <audio> placeholder AND the contract's attachment
    line. Callers that pass no path still get a valid (path-less) notice."""
    if audio_path is None and display_path is None:
        return {"role": "tool_response",
                "content": "<audio>\n" + oc.read_output_audio("audio", 0.0, 44100)}
    return oc_read_audio_response_msg(audio_path or display_path,
                                      display_path=display_path)


def oc_compat_emit_listen_sequence(
    messages: list[dict],
    audio_assets: list[str],
    audio_path: str | Path,
    probe_stdout: str | None = None,
    display_path: str | None = None,
) -> None:
    path_str = str(display_path or audio_path)
    if probe_stdout is None:
        try:
            info = sf.info(str(audio_path))
            probe_stdout = (f"Rendered {path_str} "
                            f"({info.duration:.2f}s, {int(info.samplerate)} Hz)")
        except Exception:
            probe_stdout = f"Rendered {path_str}"
    messages.append(oc_compat_bash_response(probe_stdout))
    messages.append(oc_compat_tool_call("Read", {"file_path": path_str}))
    messages.append(oc_compat_read_response_audio(audio_path, display_path=path_str))


OC_ALLOWED_ROLES = {"system", "user", "assistant", "tool_call", "tool_response"}


def validate_oc_record(record: dict[str, Any]) -> list[str]:
    """Validator for opencode-contract records: leading system turn, lowercase
    tool names, plain-string contents, audio-tag/audios alignment."""
    errors: list[str] = []
    rid = str(record.get("id", ""))
    messages = record.get("messages")
    audios = record.get("audios")
    if not rid:
        errors.append("id_missing")
    if not isinstance(messages, list) or len(messages) < 3:
        errors.append("messages_missing_or_too_short")
        return errors
    if not isinstance(audios, list):
        errors.append("audios_missing_or_not_list")
        audios = []
    if messages[0].get("role") != "system":
        errors.append("first_message_not_system")
    if messages[1].get("role") != "user":
        errors.append("second_message_not_user")
    if messages[-1].get("role") != "assistant":
        errors.append("last_message_not_assistant")
    seen_tool_call = False
    audio_tag_count = 0
    prev_role: str | None = None
    for i, msg in enumerate(messages):
        if not isinstance(msg, dict):
            errors.append(f"message_{i}_not_object")
            continue
        role = msg.get("role")
        if role not in OC_ALLOWED_ROLES:
            errors.append(f"message_{i}_invalid_role:{role}")
        if role == "system" and i != 0:
            errors.append(f"message_{i}_system_not_first")
        if role == "tool_call":
            seen_tool_call = True
            try:
                payload = json.loads(msg.get("content", ""))
                if payload.get("name") not in oc.TOOL_NAMES:
                    errors.append(f"message_{i}_unknown_tool:{payload.get('name')}")
            except Exception:
                errors.append(f"message_{i}_tool_call_not_json")
        if role == "tool_response" and not seen_tool_call:
            errors.append(f"message_{i}_tool_response_before_tool_call")
        if prev_role == role and role not in {"tool_call", "tool_response"}:
            errors.append(f"message_{i}_duplicate_adjacent_role:{role}")
        prev_role = str(role) if role is not None else None
        content = msg.get("content")
        if not isinstance(content, str):
            errors.append(f"message_{i}_content_not_string")
            continue
        tags = content.count("<audio>")
        if tags > 1 and role in {"user", "assistant"}:
            errors.append(f"message_{i}_multiple_audio_tags")
        if tags and role in {"assistant", "system"}:
            errors.append(f"message_{i}_audio_tag_in_{role}_turn")
        audio_tag_count += tags
    if audio_tag_count != len(audios):
        errors.append(f"audio_tag_mismatch:tags={audio_tag_count}:audios={len(audios)}")
    for i, path in enumerate(audios):
        if not isinstance(path, str) or not path.strip():
            errors.append(f"audio_{i}_invalid_path")
    return errors


def assert_valid_oc_record(record: dict[str, Any]) -> None:
    errors = validate_oc_record(record)
    if errors:
        raise ValueError(f"{record.get('id', 'unknown')}: " + "; ".join(errors))


def build_probe_render_snippet(*args, **kwargs) -> str:
    """Alias for build_render_probes_snippet (opencode-era name)."""
    return build_render_probes_snippet(*args, **kwargs)


def _tool_call(name: str, arguments: dict) -> dict:
    """Build a tool_call message matching claw-code's format."""
    return {"role": "tool_call", "content": json.dumps({"name": name, "arguments": arguments}, ensure_ascii=False)}


def _bash_tool_response(stdout: str, stderr: str = "", interrupted: bool = False) -> dict:
    """Build a tool_response matching claw-code's BashCommandOutput schema."""
    payload: dict[str, Any] = {"stdout": stdout, "stderr": stderr, "interrupted": interrupted}
    return {"role": "tool_response", "content": json.dumps(payload, ensure_ascii=False)}


def _read_tool_response_audio() -> dict:
    """Build a tool_response for a read tool call that returns audio content."""
    return {"role": "tool_response", "content": "<audio>"}


def _emit_listen_sequence(
    messages: list[dict],
    audio_assets: list[str],
    audio_path: str | Path,
    probe_stdout: str | None = None,
    display_path: str | None = None,
) -> None:
    """Append the BashCommandOutput → Read → audio sequence.

    Call AFTER appending the assistant text and bash render tool_call.
    The Read tool_call follows immediately (same assistant turn, no
    intermediate assistant message). ``audio_path`` must already be in
    ``audio_assets`` before calling this.

    ``display_path`` — path shown in the conversation (Read file_path and
    the default probe stdout). Used by daw-farm builds where the render
    lives at a container path while ``audio_path`` is the fetched host copy;
    defaults to ``audio_path``.
    """
    path_str = str(display_path or audio_path)
    if probe_stdout is None:
        probe_out: dict[str, Any] = {"listen_probe": {"path": path_str, "exists": True}}
        try:
            info = sf.info(str(audio_path))
            probe_out["listen_probe"]["duration_s"] = round(info.duration, 4)
        except Exception:
            pass
        probe_stdout = json.dumps(probe_out, ensure_ascii=False) + "\n"
    messages.append(_bash_tool_response(probe_stdout))
    messages.append(_tool_call("Read", {"file_path": path_str}))
    messages.append(_read_tool_response_audio())


# ---------------------------------------------------------------------------
# daw-farm rollout context — real-environment execution for builders
# ---------------------------------------------------------------------------

import collections as _collections
import threading as _threading
import time as _time_mod
from dataclasses import dataclass as _dataclass, field as _field

# Aggregate wall-time per pipeline category, for bottleneck analysis.
PIPELINE_TIMINGS: "_collections.Counter[str]" = _collections.Counter()
PIPELINE_COUNTS: "_collections.Counter[str]" = _collections.Counter()
_TIMINGS_LOCK = _threading.Lock()


def record_timing(category: str, dt: float) -> None:
    with _TIMINGS_LOCK:
        PIPELINE_TIMINGS[category] += dt
        PIPELINE_COUNTS[category] += 1


def format_pipeline_timings() -> str:
    with _TIMINGS_LOCK:
        rows = sorted(PIPELINE_TIMINGS.items(), key=lambda kv: -kv[1])
        lines = [f"  {cat:<24} {total:8.1f}s  n={PIPELINE_COUNTS[cat]:<5} "
                 f"avg={total / max(1, PIPELINE_COUNTS[cat]):6.2f}s"
                 for cat, total in rows]
    return "\n".join(lines) if lines else "  (no timings recorded)"


def _env_category(what: str) -> str:
    w = what.lower()
    for key in ("render", "search", "apply", "creation", "insert", "slice",
                "grep", "cat", "count", "discovery"):
        if key in w:
            return f"env_{key}"
    return "env_other"


@_dataclass
class DawFarmRolloutCtx:
    """Shared handle for executing a sample's snippets in its daw-farm session.

    One instance per sample rollout; passed into sub-builders (search, judge,
    transcription) so all env execution serializes on the sample's single
    REAPER container via ``lock`` — reapy allows one client at a time, and at
    inference subagents share the container the same way.
    """

    session: Any  # maestro.reaper.dawfarm.DawFarmSession
    sample_id: str
    exec_timeout: float = 300.0
    lock: _threading.Lock = _field(default_factory=_threading.Lock)

    def cw(self, host_path: str | Path, subdir: str = "") -> str:
        """Container path a snippet should write to (mirrors v3's _cw)."""
        from maestro.reaper.dawfarm import rollout_dir
        base = rollout_dir(self.sample_id)
        if subdir:
            base = f"{base}/{subdir}"
        name = Path(host_path).name if str(host_path) else ""
        return f"{base}/{name}" if name else base

    def real_exec(self, cmd: str, what: str, timeout: float | None = None):
        _t0 = _time_mod.monotonic()
        with self.lock:
            _t1 = _time_mod.monotonic()
            res = self.session.exec_bash(cmd, timeout=timeout or self.exec_timeout)
            _t2 = _time_mod.monotonic()
        record_timing("env_lock_wait", _t1 - _t0)
        record_timing(_env_category(what), _t2 - _t1)
        if res.returncode != 0:
            raise RuntimeError(
                f"daw-farm {what} failed on {self.session.name} (rc={res.returncode}): "
                f"{(res.stderr or res.stdout)[-2000:]}"
            )
        return res

    def fetch_wav(self, container_path: str, host_path: str | Path) -> None:
        _t0 = _time_mod.monotonic()
        with self.lock:
            # The render snippets wait for a stable output file before their
            # exec returns, so try the direct copy first and only fall back
            # to polling if it fails (each poll is a docker exec round trip).
            try:
                self.session.get(container_path, host_path)
            except Exception:
                if not self.session.wait_for_file(container_path, timeout=60):
                    raise RuntimeError(f"daw-farm render never appeared: {container_path}")
                self.session.get(container_path, host_path)
        record_timing("env_fetch", _time_mod.monotonic() - _t0)

    def fetch_dir(self, container_dir: str, host_dir: str | Path) -> None:
        _t0 = _time_mod.monotonic()
        with self.lock:
            self.session.get_dir(container_dir, host_dir)
        record_timing("env_fetch_dir", _time_mod.monotonic() - _t0)


def warm_candidate_probe_cache_dawfarm(
    session: Any,
    names: list[str],
    out_dir: Path,
    cache: dict,
    probe_archetype: str = "lead",
) -> int:
    """Render the shared candidate-describe probes inside a daw-farm session.

    One DawDreamer engine spawn in the container renders every missing
    wavetable probe with fixed archetype notes (env-exact audio for the
    describe calls; the on-disk cache persists across runs). Call BEFORE the
    sample worker fan-out — never mid-rollout (it resets the project).
    Populates *cache* (name → host wav path) and returns #rendered.
    """
    import re as _re
    from maestro.render.dawdreamer import make_probe_notes
    from maestro.reaper import dawfarm as _df

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def _slug(s: str) -> str:
        return (_re.sub(r"[^a-zA-Z0-9]+", "_", s).strip("_") or "unnamed")[:80]

    expected = {n: out_dir / f"wt_{i:04d}_{_slug(n)}.wav" for i, n in enumerate(names)}
    missing = [n for n, p in expected.items() if not p.exists()]
    for n, p in expected.items():
        if p.exists():
            cache.setdefault(n, p)
    if not missing:
        return 0

    _df.reset_project(session)
    _df.create_vital_track(session)
    notes = [
        {"pitch": int(p), "velocity": int(v), "start_s": float(s), "dur_s": float(d)}
        for (p, v, s, d) in make_probe_notes(probe_archetype)
    ]
    notes_file = "/work/rollouts/_probe_cache/notes.json"
    res = session.exec_bash(
        f"mkdir -p /work/rollouts/_probe_cache && cat > {notes_file} <<'EOF'\n"
        + json.dumps({"notes": notes}) + "\nEOF")
    if not res.ok:
        raise RuntimeError(f"probe cache notes push failed: {res.stderr}")
    container_dir = "/work/rollouts/_probe_cache/probes"
    snippet = _wrap_as_bash(build_render_probes_snippet(
        names=names, out_dir=container_dir, midi_path=notes_file))
    res = session.exec_bash(snippet, timeout=1800)
    if not res.ok:
        raise RuntimeError(f"probe cache render failed: {res.stderr[-1500:]}")
    session.get_dir(container_dir, out_dir)
    n_done = 0
    for n, p in expected.items():
        if p.exists():
            cache[n] = p
            n_done += 1
        elif n in missing:
            logger.warning("probe cache: %s not rendered", n)
    _df.reset_project(session)
    return n_done


# ---------------------------------------------------------------------------
# Inline snippet builders — replace prebuilt skill scripts
# ---------------------------------------------------------------------------

def build_list_wavetables_total_snippet() -> str:
    """Inline Python that scans Vital's data dirs and prints wavetable count."""
    return (
        "import json, os, glob\n"
        "_VITAL_DIRS = os.environ.get('VITAL_DATA_DIRS', os.path.expanduser('~/.local/share/vital')).split(':')\n"
        "_seen, _count = set(), 0\n"
        "for _vd in _VITAL_DIRS:\n"
        "    if not os.path.isdir(_vd): continue\n"
        "    for _vt in sorted(glob.glob(os.path.join(_vd, '**', '*.vitaltable'), recursive=True)):\n"
        "        try:\n"
        "            _w = json.load(open(_vt)); _n = _w.get('name','')\n"
        "            if not isinstance(_n, str): _n = ''\n"
        "            if _n and _n not in _seen and 'groups' in _w: _seen.add(_n); _count += 1\n"
        "        except: pass\n"
        "    for _vp in sorted(glob.glob(os.path.join(_vd, '**', '*.vital'), recursive=True)):\n"
        "        try:\n"
        "            for _w in json.load(open(_vp)).get('settings',{}).get('wavetables',[]):\n"
        "                _n = _w.get('name','') if isinstance(_w,dict) else ''\n"
        "                if _n and _n not in _seen and 'groups' in _w: _seen.add(_n); _count += 1\n"
        "        except: pass\n"
        "print(json.dumps({'total': _count}))\n"
    )


def build_list_wavetables_slice_snippet(start: int, end: int) -> str:
    """Inline Python that scans Vital's data dirs and prints wavetable names in a range."""
    return (
        "import json, os, glob\n"
        "_VITAL_DIRS = os.environ.get('VITAL_DATA_DIRS', os.path.expanduser('~/.local/share/vital')).split(':')\n"
        "_seen, _names = set(), []\n"
        "for _vd in _VITAL_DIRS:\n"
        "    if not os.path.isdir(_vd): continue\n"
        "    for _vt in sorted(glob.glob(os.path.join(_vd, '**', '*.vitaltable'), recursive=True)):\n"
        "        try:\n"
        "            _w = json.load(open(_vt)); _n = _w.get('name','')\n"
        "            if not isinstance(_n, str): _n = ''\n"
        "            if _n and _n not in _seen and 'groups' in _w: _seen.add(_n); _names.append(_n)\n"
        "        except: pass\n"
        "    for _vp in sorted(glob.glob(os.path.join(_vd, '**', '*.vital'), recursive=True)):\n"
        "        try:\n"
        "            for _w in json.load(open(_vp)).get('settings',{}).get('wavetables',[]):\n"
        "                _n = _w.get('name','') if isinstance(_w,dict) else ''\n"
        "                if _n and _n not in _seen and 'groups' in _w: _seen.add(_n); _names.append(_n)\n"
        "        except: pass\n"
        "_names.sort()\n"
        f"_start, _end = {start}, min({end}, len(_names))\n"
        "_rows = [{'idx': i, 'name': _names[i]} for i in range(_start, _end)]\n"
        "print(json.dumps({'wavetables': _rows, 'start': _start, 'end': _end, "
        "'count': len(_rows), 'total': len(_names)}))\n"
    )


_WT_DISCOVER_SNIPPET = (
    "import os, glob as _glob\n"
    "_VITAL_DIRS = os.environ.get('VITAL_DATA_DIRS', os.path.expanduser('~/.local/share/vital')).split(':')\n"
    "_seen_wt, lib = set(), []\n"
    "for _vd in _VITAL_DIRS:\n"
    "    if not os.path.isdir(_vd): continue\n"
    "    for _vt in sorted(_glob.glob(os.path.join(_vd, '**', '*.vitaltable'), recursive=True)):\n"
    "        try:\n"
    "            _w = json.load(open(_vt)); _n = _w.get('name','')\n"
    "            if not isinstance(_n, str): _n = ''\n"
    "            if _n and _n not in _seen_wt and 'groups' in _w: _seen_wt.add(_n); lib.append(_w)\n"
    "        except: pass\n"
    "    for _vp in sorted(_glob.glob(os.path.join(_vd, '**', '*.vital'), recursive=True)):\n"
    "        try:\n"
    "            for _w in json.load(open(_vp)).get('settings',{}).get('wavetables',[]):\n"
    "                _n = _w.get('name','') if isinstance(_w,dict) else ''\n"
    "                if _n and _n not in _seen_wt and 'groups' in _w: _seen_wt.add(_n); lib.append(_w)\n"
    "        except: pass\n"
    "lib.sort(key=lambda w: w.get('name',''))\n"
)

_DAWDREAMER_RENDER_HELPER = r"""
import ctypes, ctypes.util, json, os, re, struct, tempfile

def _install_x11_error_handler():
    # MUST run before importing dawdreamer: Vital's state load touches the
    # GUI layer and an unhandled X error aborts the process under Xvfb.
    _path = ctypes.util.find_library("X11")
    if not _path:
        return
    _libx11 = ctypes.CDLL(_path)
    _libx11.XInitThreads()
    _H = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p)
    _ign = _H(lambda d, e: 0)
    _libx11.XSetErrorHandler(_ign)
    _IOH = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p)
    _io_ign = _IOH(lambda d: 0)
    _libx11.XSetIOErrorHandler(_io_ign)
    _install_x11_error_handler._keep = (_ign, _io_ign)

_install_x11_error_handler()
import numpy as np
import soundfile as sf
import dawdreamer as daw

VITAL_VST3 = os.environ.get("VITAL_VST3", os.path.expanduser("~/.vst3/Vital.vst3"))
SAMPLE_RATE = 44100
BLOCK_SIZE = 512

_JUCE_B64 = ".ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+"

def _juce_b64_encode(data):
    size = len(data)
    out = []
    for i in range(((size * 8) + 5) // 6):
        bit = i * 6
        byte, off = bit >> 3, bit & 7
        if byte + 1 < size:
            word = data[byte] | (data[byte + 1] << 8)
        else:
            word = data[byte] if byte < size else 0
        out.append(_JUCE_B64[(word >> off) & 0x3F])
    return str(size) + "." + "".join(out)

def _build_vst3_state(preset):
    # Vital VST3 state blob (VC2! magic + XML + JUCE-b64 IComponent). Raw
    # preset JSON is NOT a valid load_state input — passing it silently
    # renders the default wavetable (2026-08-15 postmortem).
    json_bytes = json.dumps(preset, separators=(",", ":")).encode() + b"\x00"
    chunk = json_bytes + b"\x00" * 16 + b"JUCEPrivateData"
    body = 4 + 4 + 4 + 4 + 4 + 128 + 4 + len(chunk)
    icomp = (b"VstW" + struct.pack(">III", 8, 1, 0)
             + b"CcnK" + struct.pack(">i", body)
             + b"FBCh" + struct.pack(">i", 2)
             + b"Vita" + struct.pack(">i", 0x00010600)
             + struct.pack(">i", 0) + b"\x00" * 128
             + struct.pack(">i", len(chunk)) + chunk)
    xml = ('<?xml version="1.0" encoding="UTF-8"?>\n'
           "<VST3PluginState><IComponent>" + _juce_b64_encode(icomp)
           + "</IComponent></VST3PluginState>").encode() + b"\x00"
    return struct.pack("<II", 0x21324356, len(xml)) + xml

_engine = daw.RenderEngine(SAMPLE_RATE, BLOCK_SIZE)
_synth = _engine.make_plugin_processor("vital", VITAL_VST3)
_engine.load_graph([(_synth, [])])
_STATE_PATH = os.path.join(tempfile.gettempdir(), "dd_state_%d" % os.getpid())

def load_midi_notes(path):
    with open(path) as f:
        data = json.load(f)
    notes = data["notes"]
    return [(n["pitch"], n["velocity"], n["start_s"], n["dur_s"]) for n in notes]

def render_vital_preset(preset_dict, out_path, midi_notes):
    with open(_STATE_PATH, "wb") as f:
        f.write(_build_vst3_state(preset_dict))
    _synth.load_state(_STATE_PATH)
    _synth.clear_midi()
    for pitch, vel, start, dur in midi_notes:
        _synth.add_midi_note(int(pitch), int(vel), float(start), float(dur))
    duration = max((start + dur for _, _, start, dur in midi_notes), default=10.0) + 1.0
    _engine.render(duration)
    audio = _synth.get_audio()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    sf.write(out_path, audio.T, SAMPLE_RATE)
    return float(np.abs(audio).max())
"""


def build_render_probes_snippet(
    idxs: list[int] | None = None,
    names: list[str] | None = None,
    out_dir: str = "/tmp/probes",
    midi_path: str | None = None,
    track_idx: int = 0,
    fx_idx: int = 0,
) -> str:
    """Snippet that swaps a wavetable into osc 0 and renders via DawDreamer.

    Loads the target's transcribed MIDI from ``midi_path`` (falls back to a
    fixed 4-triad pattern when unavailable). For each probe: Python swaps the
    wavetable in the init preset and renders directly through DawDreamer — no
    REAPER, no Lua, no chunk binary.
    """
    if names:
        names_literal = repr(names)
        # names-mode (judge): filenames use the local enumerate index.
        assignments_code = f"probe_items = list(enumerate({names_literal}))\n"
    else:
        idxs_literal = repr(idxs or [])
        # idxs-mode (search): filenames must carry the GLOBAL library index —
        # the search builder's tool responses and audios key off it.
        assignments_code = (
            f"all_idxs = {idxs_literal}\n"
            "probe_items = [(i, all_names[i]) for i in all_idxs if i < len(all_names)]\n"
        )

    midi_path_literal = repr(midi_path)
    out_dir_json = json.dumps(out_dir)

    return (
        _REAPY_HELPER
        + _READ_CHUNK_HELPER
        + _DAWDREAMER_RENDER_HELPER
        + f"midi_notes = load_midi_notes({midi_path_literal})\n"
        + f"OUT_DIR = {out_dir_json}\n"
        "os.makedirs(OUT_DIR, exist_ok=True)\n"
        + _WT_DISCOVER_SNIPPET
        + "all_names = [wt.get('name','') for wt in lib]\n"
        + assignments_code
        + "name_to_wt = {wt['name']: wt for wt in lib if 'name' in wt}\n"
        "base_preset = read_vital_preset()\n"
        "rendered = []\n"
        "for idx, wt_name in probe_items:\n"
        "    if wt_name not in name_to_wt: continue\n"
        "    preset = json.loads(json.dumps(base_preset))\n"
        "    preset['settings']['wavetables'][0] = name_to_wt[wt_name]\n"
        "    slug = re.sub(r'[^a-zA-Z0-9]+', '_', wt_name).strip('_')[:80] or 'unnamed'\n"
        "    out_path = f'{OUT_DIR}/wt_{idx:04d}_{slug}.wav'\n"
        "    render_vital_preset(preset, out_path, midi_notes)\n"
        "    rendered.append({'idx': idx, 'name': wt_name, 'out': out_path})\n"
        "print(json.dumps({'status': 'ok', 'rendered': rendered}))"
    )


def build_render_tuple_snippet(
    osc_names: dict[int, str],
    out_path: str,
    midi_path: str | None = None,
    track_idx: int = 0,
    fx_idx: int = 0,
) -> str:
    """Snippet that swaps wavetables into multiple oscillators and renders via DawDreamer.

    Loads the target's transcribed MIDI from ``midi_path`` (falls back to a
    fixed 4-triad pattern). No REAPER, no Lua, no chunk binary.
    """
    assignments = [(osc_idx, name) for osc_idx, name in sorted(osc_names.items())]
    assignments_literal = repr(assignments)
    apply_names = [name for _, name in assignments]
    apply_names_literal = repr(apply_names)
    out_dir = str(Path(out_path).parent)
    midi_path_literal = repr(midi_path)

    return (
        _REAPY_HELPER
        + _READ_CHUNK_HELPER
        + _DAWDREAMER_RENDER_HELPER
        + f"midi_notes = load_midi_notes({midi_path_literal})\n"
        + f"os.makedirs({json.dumps(out_dir)}, exist_ok=True)\n"
        + _WT_DISCOVER_SNIPPET
        + "name_to_wt = {wt['name']: wt for wt in lib if 'name' in wt}\n"
        + "preset = read_vital_preset()\n"
        f"for osc_idx, wt_name in {assignments_literal}:\n"
        "    if wt_name in name_to_wt:\n"
        "        preset['settings']['wavetables'][osc_idx] = name_to_wt[wt_name]\n"
        f"render_vital_preset(preset, {json.dumps(out_path)}, midi_notes)\n"
        f"print(json.dumps({{'status': 'ok', 'out': {json.dumps(out_path)}, "
        f"'wavetables': {apply_names_literal}}}))"
    )


def build_render_verify_snippet(
    out_path: str,
    midi_path: str | None = None,
    notes_override: list[tuple] | None = None,
) -> str:
    """Render init preset + transcribed MIDI via DawDreamer for transcription verification.

    If *notes_override* is provided, embeds those tuples directly instead of
    loading from *midi_path* (used for wrong-transcription verification).
    """
    out_path_json = json.dumps(out_path)
    out_dir = str(Path(out_path).parent)
    if notes_override is not None:
        midi_line = f"midi_notes = {repr(notes_override)}\n"
    else:
        midi_line = f"midi_notes = load_midi_notes({repr(midi_path)})\n"

    return (
        _REAPY_HELPER
        + _READ_CHUNK_HELPER
        + _DAWDREAMER_RENDER_HELPER
        + midi_line
        + f"os.makedirs({json.dumps(out_dir)}, exist_ok=True)\n"
        "preset = read_vital_preset()\n"
        f"render_vital_preset(preset, {out_path_json}, midi_notes)\n"
        f"print(json.dumps({{'listen_probe': {{'path': {out_path_json}, 'exists': True}}}}))"
    )


def build_render_cumulative_snippet(
    out_path: str,
    midi_path: str | None,
    wt_assignments: list[tuple[int, str]],
    modulations: list[dict],
    cumulative_params: dict[str, float],
) -> str:
    """Reconstruct the cumulative preset and render via DawDreamer.

    *cumulative_params*: ``{json_key: native_value}`` — all params applied so
    far (denormalized).  Embedded as a dict literal in the generated code.
    """
    out_path_json = json.dumps(out_path)
    out_dir = str(Path(out_path).parent)
    midi_line = f"midi_notes = load_midi_notes({repr(midi_path)})\n"
    assignments_literal = repr(list(wt_assignments))
    mods_literal = repr(modulations)
    params_literal = repr(cumulative_params)

    return (
        _REAPY_HELPER
        + _READ_CHUNK_HELPER
        + _DAWDREAMER_RENDER_HELPER
        + midi_line
        + f"os.makedirs({json.dumps(out_dir)}, exist_ok=True)\n"
        + _WT_DISCOVER_SNIPPET
        + "name_to_wt = {wt['name']: wt for wt in lib if 'name' in wt}\n"
        + "preset = read_vital_preset()\n"
        f"for osc_idx, wt_name in {assignments_literal}:\n"
        "    if wt_name in name_to_wt:\n"
        "        preset['settings']['wavetables'][osc_idx] = name_to_wt[wt_name]\n"
        f"preset['settings']['modulations'] = {mods_literal}\n"
        f"for k, v in {params_literal}.items():\n"
        "    preset['settings'][k] = v\n"
        f"render_vital_preset(preset, {out_path_json}, midi_notes)\n"
        f"print(json.dumps({{'listen_probe': {{'path': {out_path_json}, 'exists': True}}}}))"
    )


# ---------------------------------------------------------------------------
# REAPER render snippet (inline reapy — replaces DawDreamer for iterative listening)
# ---------------------------------------------------------------------------


def build_reaper_render_snippet(
    out_path: str,
    track_idx: int = 0,
) -> str:
    """Build a compact reapy snippet that renders the current REAPER project to WAV.

    Used for iterative listening after param changes — the synth state is
    already in REAPER (wavetables via chunk, params via SetParam), so no
    preset reconstruction is needed.  The model learns to write this pattern
    at inference time.
    """
    return (
        _REAPY_HELPER
        + "import os, time\n"
        f"out_path = os.path.abspath({out_path!r})\n"
        "os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)\n"
        "# remove a stale render first: REAPER blocks on an overwrite prompt\n"
        "# for existing files, which would hang a headless session\n"
        "if os.path.isfile(out_path):\n"
        "    os.remove(out_path)\n"
        "with reapy.inside_reaper():\n"
        f"    proj = RPR.EnumProjects(-1, '', 512)[0]\n"
        f"    track = RPR.GetTrack(0, {track_idx})\n"
        "    end = 0.0\n"
        "    for i in range(RPR.GetTrackNumMediaItems(track)):\n"
        "        item = RPR.GetTrackMediaItem(track, i)\n"
        "        end = max(end, RPR.GetMediaItemInfo_Value(item, 'D_POSITION')"
        " + RPR.GetMediaItemInfo_Value(item, 'D_LENGTH'))\n"
        "    RPR.GetSet_LoopTimeRange(True, False, 0.0, end + 1.0, False)\n"
        "    RPR.GetSetProjectInfo_String(proj, 'RENDER_FILE', os.path.dirname(out_path), True)\n"
        "    RPR.GetSetProjectInfo_String(proj, 'RENDER_PATTERN', "
        "os.path.splitext(os.path.basename(out_path))[0], True)\n"
        "    RPR.Main_OnCommand(42230, 0)\n"
        "# Render completes asynchronously; wait for a stable file. REAPER\n"
        "# writes to <name>-001.wav when it detects a conflict.\n"
        "prev_size = -1\n"
        "for _ in range(120):\n"
        "    alt = out_path[:-len('.wav')] + '-001.wav'\n"
        "    if not os.path.isfile(out_path) and os.path.isfile(alt):\n"
        "        os.rename(alt, out_path)\n"
        "    if os.path.isfile(out_path):\n"
        "        size = os.path.getsize(out_path)\n"
        "        if size > 44 and size == prev_size:\n"
        "            break\n"
        "        prev_size = size\n"
        "    time.sleep(0.25)\n"
        "print(json.dumps({'listen_probe': {'path': out_path, 'exists': os.path.isfile(out_path)}}))\n"
    )


# ---------------------------------------------------------------------------
# Param search snippets (inline reapy code the model learns to write)
# ---------------------------------------------------------------------------


def build_param_search_snippet(
    query: str,
    track_idx: int = 0,
    fx_idx: int = 0,
) -> str:
    """Build an inline reapy snippet that searches REAPER params by keyword.

    The model learns to write this pattern at inference time — no premade
    scripts, just inline Python that queries REAPER's TrackFX API.
    """
    return (
        _REAPY_HELPER
        + f"query = {query!r}\n"
        "keywords = query.lower().split()\n"
        "results = []\n"
        f"with reapy.inside_reaper():\n"
        f"    track = RPR.GetTrack(0, {track_idx})\n"
        f"    n = RPR.TrackFX_GetNumParams(track, {fx_idx})\n"
        f"    for i in range(n):\n"
        f"        _pn = RPR.TrackFX_GetParamName(track, {fx_idx}, i, '', 2048)\n"
        f"        name = _pn[-2] if len(_pn) > 2 else ''\n"
        f"        if all(kw in name.strip().lower() for kw in keywords):\n"
        f"            _pv = RPR.TrackFX_GetParam(track, {fx_idx}, i, 0.0, 0.0)\n"
        f"            val, mn, mx = float(_pv[0]), float(_pv[-2]), float(_pv[-1])\n"
        f"            _dv = RPR.TrackFX_GetFormattedParamValue(track, {fx_idx}, i, '', 2048)\n"
        f"            disp = _dv[-2] if len(_dv) > 2 else ''\n"
        f"            _ss = RPR.TrackFX_GetParameterStepSizes(track, {fx_idx}, i, 0, 0, 0, 0)\n"
        f"            step = float(_ss[4]) if len(_ss) > 4 else 0\n"
        f"            entry = {{'idx': i, 'name': name.strip(), 'display': disp.strip()}}\n"
        f"            if step > 0:\n"
        f"                n_options = round(1/step) + 1\n"
        f"                options = []\n"
        f"                for oi in range(n_options):\n"
        f"                    ov = oi / max(1, n_options - 1)\n"
        f"                    # format-only: never TrackFX_SetParam here — a set (even\n"
        f"                    # set-then-restore) stamps REAPER's VST3 param cache and the\n"
        f"                    # next render re-asserts it over chunk-applied state\n"
        f"                    _od = RPR.TrackFX_FormatParamValueNormalized(track, {fx_idx}, i, ov, '', 2048)\n"
        f"                    options.append((_od[-2] if len(_od) > 2 else '').strip())\n"
        f"                entry['type'] = 'discrete'\n"
        f"                entry['options'] = options\n"
        f"                entry['current_index'] = round(val * max(1, n_options - 1))\n"
        f"            else:\n"
        f"                entry['type'] = 'continuous'\n"
        f"                entry['value'] = round(val, 6)\n"
        f"                entry['min'] = round(mn, 6)\n"
        f"                entry['max'] = round(mx, 6)\n"
        f"            results.append(entry)\n"
        "print(json.dumps({'query': query, 'count': len(results), 'params': results}, indent=2))\n"
    )


def simulate_param_search(
    query: str,
    dump: list[dict],
    value_overrides: dict[int, float] | None = None,
    max_results: int = 30,
) -> list[dict]:
    """Simulate a keyword search against the static REAPER param dump.

    Build-time only — produces the same JSON the inline reapy snippet would
    return from a live REAPER session. The dump now includes type info
    (discrete with options list, or continuous with value/min/max).
    *value_overrides* patches in current [0,1] values for continuous params.
    """
    keywords = query.lower().split()
    results: list[dict] = []
    for p in dump:
        if all(kw in p["name"].lower() for kw in keywords):
            entry = dict(p)
            if value_overrides and p["idx"] in value_overrides:
                ov = value_overrides[p["idx"]]
                if entry.get("type") == "discrete" and "options" in entry:
                    n_opts = len(entry["options"])
                    entry["current_index"] = round(ov * max(1, n_opts - 1))
                    entry["display"] = entry["options"][min(entry["current_index"], n_opts - 1)]
                elif entry.get("type") == "continuous":
                    entry["value"] = round(ov, 6)
            results.append(entry)
    return results[:max_results]


# ---------------------------------------------------------------------------
# Code-mistake injection — mutation catalog + traceback execution
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)


@dataclass
class CodeMutation:
    name: str
    category: str  # "dawdreamer" | "reapy" | "syntax"
    target_snippets: list[str]
    diagnosis: str
    _old: str = ""
    _new: str = ""

    def mutate(self, code: str) -> str | None:
        if self._old not in code:
            return None
        return code.replace(self._old, self._new, 1)


def _make_syntax_paren_mutation() -> CodeMutation:
    """SyntaxError: extra paren in json.dumps — uses rfind for last occurrence."""
    m = CodeMutation(
        name="syntax_error_paren",
        category="syntax",
        target_snippets=["probes", "tuple", "verify", "cumulative", "reaper_render", "param_search"],
        diagnosis=(
            "The script failed to parse — `SyntaxError: '(' was never closed`. "
            "There's an extra parenthesis in the `json.dumps` call. Removing it."
        ),
    )
    _orig_mutate = m.mutate

    def _mutate_last_json_dumps(code: str) -> str | None:
        target = "json.dumps({"
        idx = code.rfind(target)
        if idx < 0:
            return None
        return code[:idx] + "json.dumps(({" + code[idx + len(target):]

    m.mutate = _mutate_last_json_dumps  # type: ignore[assignment]
    return m


def _make_indentation_mutation() -> CodeMutation:
    m = CodeMutation(
        name="indentation_error",
        category="syntax",
        target_snippets=["probes", "tuple", "verify", "cumulative"],
        diagnosis=(
            "The script failed to parse — `IndentationError: unexpected indent`. "
            "The `finally` block has wrong indentation. Fixing it."
        ),
        _old="\n    finally:\n",
        _new="\n      finally:\n",
    )
    return m


CODE_MUTATIONS: list[CodeMutation] = [
    # --- DawDreamer ---
    CodeMutation(
        name="key_error_settings",
        category="dawdreamer",
        target_snippets=["probes", "tuple", "verify", "cumulative"],
        diagnosis=(
            "The render failed — `KeyError: 'setting'`. The preset dictionary key "
            "should be `'settings'` with a trailing 's'. Fixing and re-rendering."
        ),
        _old="preset['settings']",
        _new="preset['setting']",
    ),
    CodeMutation(
        name="glob_ext_typo",
        category="dawdreamer",
        target_snippets=["probes", "tuple", "cumulative"],
        diagnosis=(
            "The render failed — no wavetables were found. The glob pattern used "
            "`*.vitaltable2` but the correct extension is `*.vitaltable`. Fixing the typo."
        ),
        _old="*.vitaltable",
        _new="*.vitaltable2",
    ),
    CodeMutation(
        name="name_error_midi",
        category="dawdreamer",
        target_snippets=["probes", "tuple", "verify"],
        diagnosis=(
            "The render failed — `NameError: name 'midi_note' is not defined`. "
            "The variable is `midi_notes` (plural). Fixing the typo."
        ),
        _old=", midi_notes)\n",
        _new=", midi_note)\n",
    ),
    CodeMutation(
        name="index_error_wt",
        category="dawdreamer",
        target_snippets=["probes"],
        diagnosis=(
            "The render failed — `IndexError: list index out of range`. The wavetable "
            "slot index exceeds the number of oscillators. Correcting to index 0."
        ),
        _old="wavetables'][0]",
        _new="wavetables'][3]",
    ),
    CodeMutation(
        name="attr_error_dawdreamer",
        category="dawdreamer",
        target_snippets=["probes", "tuple", "verify", "cumulative"],
        diagnosis=(
            "The render failed — `AttributeError: 'reset_midi'`. The DawDreamer synth "
            "method is `clear_midi()`, not `reset_midi()`. Fixing."
        ),
        _old="_synth.clear_midi()",
        _new="_synth.reset_midi()",
    ),
    # --- reapy ---
    CodeMutation(
        name="attr_error_reapy_func",
        category="reapy",
        target_snippets=["param_search"],
        diagnosis=(
            "The command failed — `AttributeError: 'TrackFX_GetParamCount'`. The REAPER "
            "API function is `TrackFX_GetNumParams`, not `TrackFX_GetParamCount`. Fixing."
        ),
        _old="RPR.TrackFX_GetNumParams",
        _new="RPR.TrackFX_GetParamCount",
    ),
    CodeMutation(
        name="type_error_reapy_args",
        category="reapy",
        target_snippets=["param_search"],
        diagnosis=(
            "The command failed — `TypeError: missing 1 required positional argument`. "
            "`TrackFX_GetParam` requires 5 arguments (track, fx_idx, param_idx, minval, "
            "maxval). Adding the missing argument."
        ),
        _old=", 0.0, 0.0)",
        _new=", 0.0)",
    ),
    CodeMutation(
        name="attr_error_reapy_render",
        category="reapy",
        target_snippets=["reaper_render"],
        diagnosis=(
            "The render command failed — `AttributeError: 'Main_OnCommand_Ex'`. The REAPER "
            "API function is `Main_OnCommand`, not `Main_OnCommand_Ex`. Fixing."
        ),
        _old="RPR.Main_OnCommand(",
        _new="RPR.Main_OnCommand_Ex(",
    ),
    # --- Syntax ---
    _make_syntax_paren_mutation(),
    _make_indentation_mutation(),
]


_reaper_available_cache: dict[str, bool] = {}


def reaper_is_running(cwd: str) -> bool:
    """Check whether REAPER is reachable via reapy. Cached per cwd for the build run."""
    if cwd in _reaper_available_cache:
        return _reaper_available_cache[cwd]
    try:
        proc = subprocess.run(
            ["python3", "-c", "import reapy; reapy.is_inside_reaper() or reapy.connect()"],
            capture_output=True, timeout=5.0, cwd=cwd,
        )
        available = proc.returncode == 0
    except Exception:
        available = False
    _reaper_available_cache[cwd] = available
    logger.info("REAPER availability check: %s (cwd=%s)", available, cwd)
    return available


def select_and_apply_mutation(
    snippet_name: str,
    code: str,
    rng: random.Random,
    reaper_available: bool = False,
) -> tuple[CodeMutation, str] | None:
    """Pick a random applicable mutation and apply it.

    Returns (mutation, broken_code) or None if no mutation fits.
    """
    candidates = [
        m for m in CODE_MUTATIONS
        if snippet_name in m.target_snippets
        and (m.category != "reapy" or reaper_available)
    ]
    rng.shuffle(candidates)
    for m in candidates:
        broken = m.mutate(code)
        if broken is not None and broken != code:
            return m, broken
    return None


def execute_for_traceback(
    broken_code: str,
    cwd: str,
    timeout: float = 15.0,
    session: Any | None = None,
) -> str | None:
    """Run broken Python in subprocess, return stderr if it failed.

    With *session* (a maestro.reaper.dawfarm.DawFarmSession), the code runs
    inside the daw-farm container instead — the traceback then comes from
    the same environment the rest of the rollout executed in.
    """
    if session is not None:
        try:
            res = session.exec_python(broken_code, timeout=timeout)
        except Exception as exc:
            logger.warning("execute_for_traceback (daw-farm) failed: %s", exc)
            return None
        if res.returncode != 0 and res.stderr.strip():
            return res.stderr.strip()
        return None
    try:
        proc = subprocess.run(
            ["bash", "-c", f"python3 - <<'PY'\n{broken_code}\nPY"],
            capture_output=True, text=True, timeout=timeout, cwd=cwd,
        )
    except subprocess.TimeoutExpired:
        logger.warning("execute_for_traceback timed out after %.0fs", timeout)
        return None
    except Exception as exc:
        logger.warning("execute_for_traceback failed: %s", exc)
        return None
    if proc.returncode != 0 and proc.stderr.strip():
        return proc.stderr.strip()
    return None


def emit_code_mistake_sequence(
    messages: list[dict],
    broken_cmd: str,
    correct_cmd: str,
    traceback_stderr: str,
    mutation: CodeMutation,
) -> None:
    """Inject the error→fix sequence into the conversation.

    Caller has already appended whatever assistant message leads into the
    broken tool call. This emits:
      1. tool_call  (broken bash)
      2. tool_response (traceback)
      3. assistant  (diagnosis + "fixing")
    The caller's existing code then appends the correct tool_call + success
    tool_response, completing the retry pattern.
    """
    messages.append(_tool_call("Bash", {"command": _wrap_as_bash(broken_cmd)}))
    messages.append(_bash_tool_response(stdout="", stderr=traceback_stderr))
    messages.append({"role": "assistant", "content": mutation.diagnosis})


# ---------------------------------------------------------------------------
# Transcription-mistake injection — note mutation catalog
# ---------------------------------------------------------------------------


@dataclass
class TranscriptionMutation:
    name: str
    narration_templates: list[str]

    def apply(self, notes: list[dict], rng: random.Random) -> tuple[list[dict], dict]:
        raise NotImplementedError


class _PitchShiftSmall(TranscriptionMutation):
    def __init__(self) -> None:
        super().__init__(
            name="pitch_shift_small",
            narration_templates=[
                "The melody is slightly off — one note doesn’t match the target pitch. Re-transcribing from scratch.",
                "One of the notes sounds like it’s the wrong pitch. Re-transcribing.",
                "A note in the middle doesn’t sound right compared to the target. Starting over.",
            ],
        )

    def apply(self, notes: list[dict], rng: random.Random) -> tuple[list[dict], dict]:
        wrong = copy.deepcopy(notes)
        idx = rng.randrange(len(wrong))
        delta = rng.choice([-2, -1, 1, 2])
        orig = int(wrong[idx]["pitch"])
        wrong[idx]["pitch"] = max(0, min(127, orig + delta))
        return wrong, {"type": self.name, "note_idx": idx, "delta": delta}


class _OctaveError(TranscriptionMutation):
    def __init__(self) -> None:
        super().__init__(
            name="octave_error",
            narration_templates=[
                "One note jumps to the wrong octave — the contour breaks there. Re-transcribing.",
                "A note sounds like it’s in the wrong octave register. Starting over.",
                "The melody jumps an octave at one point — that’s not in the original. Re-transcribing.",
            ],
        )

    def apply(self, notes: list[dict], rng: random.Random) -> tuple[list[dict], dict]:
        wrong = copy.deepcopy(notes)
        idx = rng.randrange(len(wrong))
        delta = rng.choice([-12, 12])
        orig = int(wrong[idx]["pitch"])
        new_pitch = orig + delta
        if new_pitch < 0 or new_pitch > 127:
            new_pitch = orig - delta
        wrong[idx]["pitch"] = max(0, min(127, new_pitch))
        return wrong, {"type": self.name, "note_idx": idx, "delta": delta}


class _NoteDeletion(TranscriptionMutation):
    def __init__(self) -> None:
        super().__init__(
            name="note_deletion",
            narration_templates=[
                "The melody sounds incomplete — a note seems to be missing. Re-transcribing.",
                "Compared to the target, the transcription is missing a note. Starting over.",
                "The phrase feels like it’s missing a beat — a note was dropped. Re-transcribing.",
            ],
        )

    def apply(self, notes: list[dict], rng: random.Random) -> tuple[list[dict], dict]:
        wrong = copy.deepcopy(notes)
        idx = rng.randrange(len(wrong))
        removed = wrong.pop(idx)
        return wrong, {"type": self.name, "note_idx": idx, "removed_pitch": removed["pitch"]}


class _NoteInsertion(TranscriptionMutation):
    def __init__(self) -> None:
        super().__init__(
            name="note_insertion",
            narration_templates=[
                "There’s an extra note that wasn’t in the original melody. Re-transcribing.",
                "I’m hearing a note that doesn’t belong in this phrase. Starting over.",
                "The transcription has an extra event that the target doesn’t. Re-transcribing.",
            ],
        )

    def apply(self, notes: list[dict], rng: random.Random) -> tuple[list[dict], dict]:
        wrong = copy.deepcopy(notes)
        pitches = [n["pitch"] for n in wrong]
        min_p, max_p = min(pitches), max(pitches)
        phantom_pitch = rng.randint(max(0, min_p - 3), min(127, max_p + 3))
        max_time = max(n["start_s"] + n["dur_s"] for n in wrong)
        phantom_start = round(rng.uniform(0, max(0.1, max_time - 0.3)), 4)
        phantom = {
            "pitch": phantom_pitch,
            "start_s": phantom_start,
            "dur_s": round(rng.uniform(0.1, 0.5), 4),
            "velocity": rng.randint(60, 110),
        }
        wrong.append(phantom)
        wrong.sort(key=lambda n: (n["start_s"], n["pitch"]))
        return wrong, {"type": self.name, "phantom": phantom}


class _TimingShift(TranscriptionMutation):
    def __init__(self) -> None:
        super().__init__(
            name="timing_shift",
            narration_templates=[
                "The rhythm is off — a note’s timing doesn’t match the target. Re-transcribing.",
                "One note sounds out of time compared to the original. Starting over.",
                "The timing doesn’t line up for one of the notes. Re-transcribing.",
            ],
        )

    def apply(self, notes: list[dict], rng: random.Random) -> tuple[list[dict], dict]:
        wrong = copy.deepcopy(notes)
        idx = rng.randrange(len(wrong))
        shift = rng.choice([-1, 1]) * round(rng.uniform(0.15, 0.3), 4)
        orig_start = wrong[idx]["start_s"]
        wrong[idx]["start_s"] = round(max(0, orig_start + shift), 4)
        wrong.sort(key=lambda n: (n["start_s"], n["pitch"]))
        return wrong, {"type": self.name, "note_idx": idx, "shift_s": shift}


class _MultiNoteCorruption(TranscriptionMutation):
    def __init__(self) -> None:
        super().__init__(
            name="multi_note_corruption",
            narration_templates=[
                "Multiple notes sound wrong compared to the target. Re-transcribing from scratch.",
                "A couple of notes don’t match the original melody. Starting over.",
                "The melody deviates in several places. Re-transcribing.",
            ],
        )

    def apply(self, notes: list[dict], rng: random.Random) -> tuple[list[dict], dict]:
        wrong = copy.deepcopy(notes)
        n_corrupt = min(2, len(wrong))
        indices = rng.sample(range(len(wrong)), n_corrupt)
        deltas = []
        for idx in indices:
            delta = rng.choice([-2, -1, 1, 2])
            orig = int(wrong[idx]["pitch"])
            wrong[idx]["pitch"] = max(0, min(127, orig + delta))
            deltas.append({"note_idx": idx, "delta": delta})
        return wrong, {"type": self.name, "corrupted": deltas}


TRANSCRIPTION_MUTATIONS: list[TranscriptionMutation] = [
    _PitchShiftSmall(),
    _OctaveError(),
    _NoteDeletion(),
    _NoteInsertion(),
    _TimingShift(),
    _MultiNoteCorruption(),
]


_COMPOUND_NARRATIONS: list[str] = [
    "The melody doesn't match — several notes sound off compared to the target. Re-transcribing from scratch.",
    "This doesn't sound right. Multiple issues — some pitches are wrong and the phrasing is off. Starting over.",
    "The transcription deviates from the target in several places. Re-transcribing.",
    "Comparing against the target, the melody has multiple errors. Starting over from scratch.",
    "The transcription has noticeable differences from the target — re-transcribing.",
]


def apply_transcription_mutations(
    notes: list[dict],
    rng: random.Random,
) -> tuple[list[dict], list[dict], str] | None:
    """Apply multiple mutations to corrupt a transcription.

    The number of mutations scales with melody length (2-6 for typical
    20-40 note melodies). Returns ``(wrong_notes, info_list, narration)``
    or ``None`` if notes are too short to corrupt.
    """
    if len(notes) < 3:
        return None

    n_muts = rng.randint(2, max(2, min(6, len(notes) // 5)))

    wrong = copy.deepcopy(notes)
    infos: list[dict] = []

    for _ in range(n_muts):
        candidates = list(TRANSCRIPTION_MUTATIONS)
        if len(wrong) < 3:
            candidates = [m for m in candidates if m.name not in ("note_deletion", "multi_note_corruption")]
        if not candidates:
            break
        mut = rng.choice(candidates)
        wrong, info = mut.apply(wrong, rng)
        infos.append(info)

    if not infos:
        return None

    narration = rng.choice(_COMPOUND_NARRATIONS)
    return wrong, infos, narration


def select_transcription_mutation(
    notes: list[dict],
    rng: random.Random,
    exclude_names: list[str] | None = None,
) -> tuple[TranscriptionMutation, list[dict], dict] | None:
    """Pick a random applicable mutation and apply it.

    .. deprecated:: Use ``apply_transcription_mutations`` instead.
    """
    exclude = set(exclude_names or [])
    candidates = [m for m in TRANSCRIPTION_MUTATIONS if m.name not in exclude]
    if len(notes) < 3:
        candidates = [m for m in candidates if m.name not in ("note_deletion", "multi_note_corruption")]
    if not candidates:
        return None
    mut = rng.choice(candidates)
    wrong, info = mut.apply(notes, rng)
    return mut, wrong, info
