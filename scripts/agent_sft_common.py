from __future__ import annotations

import hashlib
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from maestro.render.vital import SAMPLE_RATE, _load_vital, _render_note_list, make_probe_notes, trim_silence
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
    def create(cls, device: str) -> "ClapEmbedder":
        model, processor = _load_clap(device)
        return cls(device=device, model=model, processor=processor, _cache={})

    def embed_audio_path(self, path: Path) -> np.ndarray:
        key = str(path.resolve())
        if key in self._cache:
            return self._cache[key]
        audio, sr = sf.read(path, always_2d=True)
        audio = audio.T.astype(np.float32)
        emb = _embed_clap(audio, sr, self.model, self.processor, self.device)
        self._cache[key] = emb
        return emb

    def cosine_paths(self, a: Path, b: Path) -> float:
        ea = self.embed_audio_path(a)
        eb = self.embed_audio_path(b)
        an = np.linalg.norm(ea) + 1e-12
        bn = np.linalg.norm(eb) + 1e-12
        return float(np.dot(ea, eb) / (an * bn))


def ensure_candidate_probes_for_names(
    names: list[str],
    wavetable_lib: list[dict[str, Any]],
    selected_rows: dict[str, dict[str, Any]],
    out_dir: Path,
    cache: dict[str, Path],
    probe_archetype: str = "lead",
    probe_tail_s: float = 1.0,
    trim_min_duration_s: float = 0.5,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    missing = [n for n in names if n not in cache and n in selected_rows]
    if not missing:
        return

    init_preset = _load_init_preset()
    notes = make_probe_notes(probe_archetype)
    synth = _load_vital()

    for name in missing:
        row = selected_rows[name]
        source_idx = int(row["source_wavetable_idx"])
        frame_idx = int(row.get("frame_idx", 0))
        wt = wavetable_lib[source_idx]
        fname = f"{slugify(name)}__src{source_idx:04d}_f{frame_idx:03d}.wav"
        path = out_dir / fname
        if not path.exists():
            preset = _build_probe_preset(init_preset, wt, frame_idx)
            synth.load_json(json.dumps(preset))
            audio = _render_note_list(synth, notes, SAMPLE_RATE, tail_s=probe_tail_s)
            audio = trim_silence(audio, SAMPLE_RATE, min_duration_s=trim_min_duration_s)
            sf.write(path, audio.T, SAMPLE_RATE)
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
    threshold: float = 0.92,
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
