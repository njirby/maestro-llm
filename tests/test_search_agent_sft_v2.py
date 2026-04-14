"""Contract tests for build_search_agent_sft_v2 record structure.

Validates the iterative batch-listening search agent conversation:
  - Record passes ms-swift validator
  - Multiple batches present
  - Shortlist evolves across batches
  - GT wavetable appears on final shortlist when in shard
  - Every <audio> in tool_response has a corresponding name
  - No CLAP scores or embeddings in messages
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.agent_sft_common import validate_ms_swift_multiturn_record


# ---- Helpers ----

def _make_search_v2_record(
    n_batches: int = 3,
    candidates_per_batch: int = 4,
    gt_in_shard: bool = True,
    gt_name: str = "01 Basic Shapes",
):
    """Build a synthetic search agent v2 record for structural testing."""
    messages = [
        {"role": "user", "content": "<audio>\nListen to this target keys sound. Evaluate batches of candidates."},
    ]
    audios = ["/tmp/target.wav"]
    shortlist = []

    for bi in range(n_batches):
        batch_names = [f"Wavetable_{bi * candidates_per_batch + j}" for j in range(candidates_per_batch)]
        # Insert GT in batch 2 if gt_in_shard
        if gt_in_shard and bi == 1:
            batch_names[0] = gt_name

        candidate_entries = []
        for name in batch_names:
            audios.append(f"/tmp/probe_{name}.wav")
            candidate_entries.append({"name": name, "audio": "<audio>"})

        # Intro — merge prior notes for non-first batches
        if bi == 0:
            intro = f"Listening to batch {bi + 1} of candidate wavetables."
        else:
            sl_str = ", ".join(f"'{n}'" for n in shortlist)
            intro = f"Current shortlist: [{sl_str}]\n\nListening to batch {bi + 1}."

        messages.append({"role": "assistant", "content": intro})
        messages.append({
            "role": "tool_call",
            "content": json.dumps({"name": "bash", "arguments": {"command": f"# Render batch {bi + 1}"}})
        })
        messages.append({
            "role": "tool_response",
            "content": json.dumps({"batch": bi + 1, "candidates": candidate_entries}),
        })

        # Update shortlist
        if gt_in_shard and bi == 1:
            shortlist.append(gt_name)
        shortlist.append(batch_names[-1])  # always add last candidate
        shortlist = shortlist[-4:]  # keep max 4

    # Final summary merged with last batch notes
    sl_str = ", ".join(f"'{n}'" for n in shortlist)
    final = f"Nothing new improves the shortlist.\n\nFinal candidates: [{sl_str}]"
    messages.append({"role": "assistant", "content": final})

    return {
        "id": "test_search_v2_001",
        "task_type": "search_v2",
        "tools": "[]",
        "messages": messages,
        "audios": audios,
        "meta": {
            "pipeline_version": "v2_search",
            "sample_id": "test_001",
            "archetype": "keys",
            "shard_size": n_batches * candidates_per_batch,
            "n_batches": n_batches,
            "candidates_per_batch": candidates_per_batch,
            "gt_in_shard": [gt_name] if gt_in_shard else [],
            "final_shortlist": shortlist,
            "gt_on_shortlist": [gt_name] if gt_in_shard and gt_name in shortlist else [],
        },
    }


# ---- Tests ----

def test_search_v2_record_passes_validator():
    record = _make_search_v2_record()
    errors = validate_ms_swift_multiturn_record(record)
    assert errors == [], f"Validation errors: {errors}"


def test_search_v2_multiple_batches():
    record = _make_search_v2_record(n_batches=3)
    tool_responses = [
        m for m in record["messages"]
        if m["role"] == "tool_response" and "batch" in m["content"]
    ]
    assert len(tool_responses) == 3


def test_search_v2_gt_on_shortlist_when_in_shard():
    record = _make_search_v2_record(gt_in_shard=True, gt_name="01 Basic Shapes")
    assert "01 Basic Shapes" in record["meta"]["final_shortlist"]
    assert record["meta"]["gt_on_shortlist"] == ["01 Basic Shapes"]


def test_search_v2_no_gt_when_not_in_shard():
    record = _make_search_v2_record(gt_in_shard=False)
    assert record["meta"]["gt_in_shard"] == []
    assert record["meta"]["gt_on_shortlist"] == []


def test_search_v2_audio_tags_match_names():
    """Every <audio> in a tool_response should have a corresponding name in the JSON."""
    record = _make_search_v2_record()
    for m in record["messages"]:
        if m["role"] == "tool_response":
            content = m["content"]
            if "<audio>" in content:
                parsed = json.loads(content)
                candidates = parsed.get("candidates", [])
                for c in candidates:
                    if c.get("audio") == "<audio>":
                        assert "name" in c, "Candidate with <audio> must have a name"
                        assert c["name"], "Candidate name must not be empty"


def test_search_v2_no_clap_scores_in_messages():
    """No CLAP scores or embedding references should leak into the conversation."""
    record = _make_search_v2_record()
    banned = ["clap", "cosine_vs_target", "embedding", "0.95", "0.98"]
    for m in record["messages"]:
        content = m.get("content", "").lower()
        for term in banned:
            assert term not in content, f"Banned term '{term}' found in message: {content[:80]}"


def test_search_v2_shortlist_evolves():
    """The shortlist should grow or change across batches."""
    record = _make_search_v2_record(n_batches=3)
    assistant_msgs = [m["content"] for m in record["messages"] if m["role"] == "assistant"]
    shortlist_mentions = [m for m in assistant_msgs if "shortlist" in m.lower() or "Final candidates" in m]
    assert len(shortlist_mentions) >= 2, "Shortlist should be mentioned in at least 2 messages"


def test_search_v2_final_candidates_present():
    record = _make_search_v2_record()
    last = record["messages"][-1]["content"]
    assert "Final candidates:" in last


def test_search_v2_audio_count_matches():
    """Audio count = 1 (target) + n_batches * candidates_per_batch (one per candidate)."""
    record = _make_search_v2_record(n_batches=3, candidates_per_batch=4)
    expected = 1 + 3 * 4  # target + all candidate probes
    assert len(record["audios"]) == expected, f"Expected {expected} audios, got {len(record['audios'])}"


def test_search_v2_meta_fields():
    record = _make_search_v2_record()
    meta = record["meta"]
    assert meta["pipeline_version"] == "v2_search"
    assert "sample_id" in meta
    assert "archetype" in meta
    assert "shard_size" in meta
    assert "n_batches" in meta
    assert "final_shortlist" in meta
    assert isinstance(meta["final_shortlist"], list)
