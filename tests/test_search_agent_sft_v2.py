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
            "arguments": {"command": f"python skills/vital/scripts/list_wavetables.py --start {shard_start} --end {shard_end}"},
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
                "arguments": {"command": f"python skills/vital/scripts/render_probes.py --idxs {idxs_csv} --out-dir {probe_dir}"},
            }),
        })
        messages.append({
            "role": "tool_response",
            "content": json.dumps({"status": "ok", "rendered": rendered}),
        })

        # Shortlist update (builder-managed state, NOT echoed in narration —
        # claw-code-style: running state is implicit, final state is a file write)
        if gt_in_shard and bi == 1:
            shortlist.append(gt_name)
        shortlist.append(batch_names[-1])
        shortlist = shortlist[-4:]

        pending_notes = "\n".join(f"'{n}': Evaluated. Selected." for n in batch_names[:2])

    # Final narration + "writing file" intro merged into ONE assistant turn
    # (two back-to-back assistants would fail the validator). Claw-code-style
    # handoff: file write emitted as a visible tool_call.
    shortlist_path = f"/tmp/agents/search_test_1.json"
    payload = {
        "status": "completed",
        "agentId": "search_test_001_1",
        "shardStart": shard_start,
        "shardEnd": shard_end,
        "shortlist": shortlist,
        "nBatches": n_batches,
    }
    final = (
        f"{pending_notes or ''}\n\n"
        f"Shortlisted candidates each offer distinct qualities for the target sound.\n\n"
        f"Writing the final shortlist to the output file for the dispatcher to consume."
    )
    messages.append({"role": "assistant", "content": final})
    messages.append({
        "role": "tool_call",
        "content": json.dumps({
            "name": "bash",
            "arguments": {"command": f"python - <<'PY'\nimport json; json.dump({json.dumps(payload)}, open('{shortlist_path}','w'))\nPY"},
        }),
    })
    messages.append({
        "role": "tool_response",
        "content": json.dumps({"status": "ok", "file": shortlist_path}),
    })
    messages.append({
        "role": "assistant",
        "content": f"Shortlist written. {len(shortlist)} candidate(s) flagged for the judge agent.",
    })

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


def test_search_v2_renders_use_render_probes():
    record = _make_search_v2_record()
    tool_calls = [json.loads(m["content"]) for m in record["messages"] if m["role"] == "tool_call"]
    render_calls = [
        tc for tc in tool_calls
        if tc.get("name") == "bash" and "skills/vital/scripts/render_probes.py" in tc["arguments"].get("command", "")
    ]
    # Expect one render call per batch
    assert len(render_calls) >= 2, f"Expected >= 2 render_probes calls, got {len(render_calls)}"
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


def test_search_v2_ends_with_shortlist_file_write():
    """Claw-code-style handoff: the final turns should be a bash tool_call that
    writes the shortlist file, a tool_response confirming the write, and a
    closing assistant turn. No in-token 'Final candidates: [...]' echo."""
    record = _make_search_v2_record()
    msgs = record["messages"]

    # Last message must be assistant (validator invariant)
    assert msgs[-1]["role"] == "assistant"

    # Find the final bash write tool_call + response pair
    tool_calls = [
        (i, json.loads(m["content"]))
        for i, m in enumerate(msgs)
        if m["role"] == "tool_call"
    ]
    bash_calls = [(i, tc) for i, tc in tool_calls if tc.get("name") == "bash"]
    assert bash_calls, "Expected at least one bash tool_call"
    # The LAST bash tool_call should be the shortlist file write
    last_bash_idx, last_bash = bash_calls[-1]
    cmd = last_bash["arguments"]["command"]
    assert "shortlist" in cmd.lower() or ".json" in cmd, (
        f"Last bash call should write the shortlist; got: {cmd[:120]}"
    )

    # And its tool_response should confirm file ok
    assert msgs[last_bash_idx + 1]["role"] == "tool_response"
    resp = json.loads(msgs[last_bash_idx + 1]["content"])
    assert resp.get("status") == "ok"
    assert "file" in resp

    # The "Final candidates: [..]" in-token echo is explicitly removed.
    combined = " ".join(m["content"] for m in msgs if m["role"] == "assistant")
    assert "Final candidates:" not in combined, (
        "The in-token 'Final candidates: [...]' echo should not appear — "
        "shortlist is communicated via the file write only."
    )


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
