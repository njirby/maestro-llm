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
        if prev_role in {"user", "assistant"} and role == prev_role:
            errors.append(f"message_{i}_duplicate_adjacent_role:{role}")
        prev_role = str(role) if role is not None else None

        content = msg.get("content")
        if not isinstance(content, str):
            errors.append(f"message_{i}_content_not_string")
            continue
        tags = content.count("<audio>")
        if tags > 1:
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
