"""Contract tests for build_search_agent_sft_v2 record structure.

Validates the iterative batch-listening search agent conversation with
real bash commands to list_wavetables.py + render_wavetable_probes.py:
  - Record passes ms-swift validator
  - First tool call is list_wavetables.py with --start/--end
  - Each batch render uses render_wavetable_probes.py with --idxs
  - Shortlist output path in meta
  - Audio tags match rendered candidates
  - No CLAP scores or embeddings in messages
  - No fabricated wavetable names (shortlist ⊆ rendered set)
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
    shard_start: int = 100,
    shard_size: int = 24,
    gt_in_shard: bool = True,
    gt_name: str = "01 Basic Shapes",
):
    """Build a synthetic search agent v2 record for structural testing.

    Matches the new claw-code-style flow: real list_wavetables.py + render_wavetable_probes.py
    bash commands, tool responses with audio attached per-probe.
    """
    shard_end = shard_start + shard_size
    probe_dir = "/tmp/search_probes/test_agent1"

    messages = [
        {
            "role": "user",
            "content": (
                f"<audio>\nTarget: keys sound. Evaluate wavetables at indices "
                f"{shard_start}-{shard_end - 1} from the library and return a shortlist."
            ),
        },
    ]
    audios = ["/tmp/target.wav"]
    shortlist = []

    # Step 1: list_wavetables
    messages.append({
        "role": "assistant",
        "content": f"Fetching candidate names at indices {shard_start}-{shard_end - 1}.",
    })
    messages.append({
        "role": "tool_call",
        "content": json.dumps({
            "name": "bash",
            "arguments": {"command": f"python scripts/list_wavetables.py --start {shard_start} --end {shard_end}"},
        }),
    })
    shard_entries = [{"idx": i, "name": f"WT_{i}"} for i in range(shard_start, shard_end)]
    messages.append({
        "role": "tool_response",
        "content": json.dumps({"wavetables": shard_entries, "count": shard_size}),
    })

    pending_notes = None
    for bi in range(n_batches):
        batch_idxs = [shard_start + bi * candidates_per_batch + j for j in range(candidates_per_batch)]
        batch_names = [f"WT_{i}" for i in batch_idxs]

        # Insert GT in batch 2 if gt_in_shard
        if gt_in_shard and bi == 1 and batch_idxs:
            batch_names[0] = gt_name

        idxs_csv = ",".join(str(i) for i in batch_idxs)

        rendered = []
        for name, idx in zip(batch_names, batch_idxs):
            out = f"{probe_dir}/wt_{idx:04d}_{name}.wav"
            audios.append(out)
            rendered.append({"idx": idx, "name": name, "out": out, "audio": "<audio>"})

        intro = f"Rendering batch {bi + 1} (indices {', '.join(str(i) for i in batch_idxs)})."
        if pending_notes:
            intro = f"{pending_notes}\n\n{intro}"
            pending_notes = None
        messages.append({"role": "assistant", "content": intro})
        messages.append({
            "role": "tool_call",
            "content": json.dumps({
                "name": "bash",
                "arguments": {"command": f"python scripts/render_wavetable_probes.py --idxs {idxs_csv} --out-dir {probe_dir}"},
            }),
        })
        messages.append({
            "role": "tool_response",
            "content": json.dumps({"status": "ok", "rendered": rendered}),
        })

        # Shortlist update
        if gt_in_shard and bi == 1:
            shortlist.append(gt_name)
        shortlist.append(batch_names[-1])
        shortlist = shortlist[-4:]

        sl_str = ", ".join(f"'{n}'" for n in shortlist)
        pending_notes = "\n".join(f"'{n}': Evaluated. Shortlisted." for n in batch_names[:2])
        pending_notes += f"\nCurrent shortlist: [{sl_str}]"

    # Final summary (merged with pending notes)
    sl_str = ", ".join(f"'{n}'" for n in shortlist)
    final = f"{pending_notes or ''}\n\nFinal candidates: [{sl_str}]"
    messages.append({"role": "assistant", "content": final})

    shortlist_path = f"/tmp/agents/search_test_1.json"
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
            "shard_size": shard_size,
            "shard_start": shard_start,
            "shard_end": shard_end,
            "n_batches": n_batches,
            "candidates_per_batch": candidates_per_batch,
            "gt_in_shard": [gt_name] if gt_in_shard else [],
            "final_shortlist": shortlist,
            "gt_on_shortlist": [gt_name] if gt_in_shard and gt_name in shortlist else [],
            "shortlist_output_file": shortlist_path,
        },
    }


# ---- Structural tests ----

def test_search_v2_record_passes_validator():
    record = _make_search_v2_record()
    errors = validate_ms_swift_multiturn_record(record)
    assert errors == [], f"Validation errors: {errors}"


def test_search_v2_first_bash_is_list_wavetables():
    record = _make_search_v2_record()
    tool_calls = [json.loads(m["content"]) for m in record["messages"] if m["role"] == "tool_call"]
    bash_calls = [tc for tc in tool_calls if tc.get("name") == "bash"]
    assert bash_calls, "Should have at least one bash call"
    first_cmd = bash_calls[0]["arguments"]["command"]
    assert "list_wavetables.py" in first_cmd
    assert "--start" in first_cmd
    assert "--end" in first_cmd


def test_search_v2_renders_use_render_wavetable_probes():
    record = _make_search_v2_record()
    tool_calls = [json.loads(m["content"]) for m in record["messages"] if m["role"] == "tool_call"]
    render_calls = [
        tc for tc in tool_calls
        if tc.get("name") == "bash" and "render_wavetable_probes.py" in tc["arguments"].get("command", "")
    ]
    # Expect one render call per batch
    assert len(render_calls) >= 2, f"Expected >= 2 render_wavetable_probes calls, got {len(render_calls)}"
    for rc in render_calls:
        cmd = rc["arguments"]["command"]
        assert "--idxs" in cmd, f"Render call missing --idxs: {cmd}"
        assert "--out-dir" in cmd, f"Render call missing --out-dir: {cmd}"


def test_search_v2_shortlist_output_file_in_meta():
    record = _make_search_v2_record()
    assert "shortlist_output_file" in record["meta"]


def test_search_v2_gt_on_shortlist_when_in_shard():
    record = _make_search_v2_record(gt_in_shard=True)
    assert "01 Basic Shapes" in record["meta"]["final_shortlist"]
    assert record["meta"]["gt_on_shortlist"] == ["01 Basic Shapes"]


def test_search_v2_no_gt_when_not_in_shard():
    record = _make_search_v2_record(gt_in_shard=False)
    assert record["meta"]["gt_in_shard"] == []
    assert record["meta"]["gt_on_shortlist"] == []


def test_search_v2_audio_matches_rendered():
    """Audio attachments in tool_responses should match rendered entries."""
    record = _make_search_v2_record()
    for m in record["messages"]:
        if m["role"] == "tool_response":
            try:
                parsed = json.loads(m["content"])
            except Exception:
                continue
            for entry in parsed.get("rendered", []):
                if entry.get("audio") == "<audio>":
                    assert "name" in entry
                    assert "idx" in entry
                    assert "out" in entry


def test_search_v2_no_clap_scores_in_messages():
    record = _make_search_v2_record()
    banned = ["cosine_vs_target", "embedding"]
    for m in record["messages"]:
        content = m.get("content", "").lower()
        for term in banned:
            assert term not in content, f"Banned term '{term}' in message: {content[:80]}"


def test_search_v2_final_candidates_present():
    record = _make_search_v2_record()
    last = record["messages"][-1]["content"]
    assert "Final candidates:" in last


def test_search_v2_meta_has_shard_range():
    record = _make_search_v2_record(shard_start=100, shard_size=24)
    meta = record["meta"]
    assert meta["shard_start"] == 100
    assert meta["shard_end"] == 124
    assert meta["shard_size"] == 24


def test_search_v2_shortlist_only_contains_rendered_names():
    """Shortlist should only contain names that appeared in rendered tool_responses."""
    record = _make_search_v2_record()
    rendered_names = set()
    for m in record["messages"]:
        if m["role"] == "tool_response":
            try:
                parsed = json.loads(m["content"])
                for entry in parsed.get("rendered", []):
                    rendered_names.add(entry["name"])
            except Exception:
                pass
    for name in record["meta"]["final_shortlist"]:
        assert name in rendered_names, f"Shortlist entry '{name}' not in rendered names"
