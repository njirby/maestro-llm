#!/usr/bin/env python3
"""Quality grader for agent SFT JSONL conversations.

Scores each record on multiple axes and writes a scored JSONL.
Main records require ``meta.step_labels`` (embedded by build_main_agent_sft_v2.py).

Scores
------
main records
    plan_param_alignment  – fraction of steps where PLAN text mentions ≥1 actual
                            changed parameter (0-1). Primary signal. Heuristic.
    section_structure     – fraction of commentary turns with all 3 headers (0-1).
    snake_case_clean      – 1 if no snake_case in assistant turns, graded down per hit.
    commentary_diversity  – 1 − mean pairwise Jaccard overlap across commentary turns.
    format_consistent     – 1 if no **BOLD:** section headers appear.
    llm_plan_alignment    – (optional) LLM judge score: does PLAN semantically match
                            params_delta? Activated via --llm-judge-server.
    overall               – weighted sum of the above (llm replaces heuristic if present).

search records
    has_cosine_scores     – audition tool_response carries cosine_vs_target fields.
    proposal_diversity    – are proposal reason strings distinct?
    no_gt_leak            – no "source wavetable" or "(gt)" in assistant turns.
    overall               – weighted sum.

judge records
    has_score_in_reason   – final assistant JSON reason references numeric scores.
    has_cosine_scores     – audition tool_response carries cosine_vs_target fields.
    gt_recall             – if gt_candidate_ids known, ≥1 in selected (1/0/None).
    overall               – weighted sum.

LLM-as-judge (--llm-judge-server)
    For main records, each PLAN section is evaluated by a text LLM against the actual
    params_delta list. The LLM can catch paraphrasing that bigram matching misses
    (e.g. "engage the bypass on modulation slot 10" correctly matching modulation_10_bypass).
    Activate with:
        --llm-judge-server http://localhost:8000
        --llm-judge-model  Qwen/Qwen3-Omni-30B-A3B-Instruct   (or any OpenAI-compatible model name)
    The judge returns a JSON {\"score\": 0.0..1.0, \"reason\": \"...\"} per step.
    Falls back to heuristic score silently if the server is unavailable.
"""
from __future__ import annotations

import argparse
import base64
import json
import random as _random
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


def _b64(path: str | Path) -> str:
    """Base64-encode a WAV file for inline embedding in an OpenAI-style data URL."""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import httpx as _httpx
    _HTTPX_AVAILABLE = True
except ImportError:
    _HTTPX_AVAILABLE = False

from scripts.build_main_agent_sft_v2 import _json_key_to_display

# ---------------------------------------------------------------------------
# Regex helpers
# ---------------------------------------------------------------------------

_SNAKE_CASE_RE = re.compile(r"\b[a-z][a-z0-9]*(?:_[a-z0-9]+){2,}\b")
_BOLD_HEADER_RE = re.compile(r"\*\*(HEARD|HYPOTHESIS|PLAN):\*\*")

# Markers that indicate the assistant prose is leaking oracle/GT info that
# the model wouldn't have at inference time. Catches phrases like
# "(GT — target processes this with: ...)" or "ground truth:" or
# "this is the correct answer".
_GT_LEAK_MARKERS = (
    r"\bgt\b\s*[—\-:]",
    r"\(gt\b",
    r"\bground\s+truth\b",
    r"\boracle\b",
    r"\bcorrect\s+answer\b",
    r"\bknown\s+(right|correct)\b",
    r"\bthis\s+is\s+the\s+(target|gt)\b",
)
_GT_LEAK_RE = re.compile("|".join(_GT_LEAK_MARKERS), re.IGNORECASE)


def _gt_leak_score(assistant_turns: list[str]) -> float:
    """Return 1.0 when no oracle/GT-leak markers appear in any assistant turn,
    else 0.0. Markers include 'GT —', '(GT', 'ground truth', 'oracle',
    'correct answer', etc. — phrases that identify a candidate or value as
    the known-right answer at inference, which the model shouldn't be able
    to do."""
    return 0.0 if any(_GT_LEAK_RE.search(t or "") for t in assistant_turns) else 1.0
_NUMERIC_SCORE_RE = re.compile(r"\d\.\d{3}")
_SECTION_HEADERS = ("HEARD:", "HYPOTHESIS:", "PLAN:")

# Module-level entity words that, combined with a number, are too generic to
# count as a specific parameter reference.  "oscillator 2" alone doesn't tell
# us which oscillator parameter was changed; "2 on" or "2 distortion type" does.
_ENTITY_WORDS = frozenset({"oscillator", "filter", "lfo", "envelope", "modulation", "unison"})


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _extract_section(text: str, header: str) -> str:
    """Return content of a section starting with *header*, until the next header."""
    idx = text.find(header)
    if idx < 0:
        # Also try bold variant
        bold = header.replace(":", ":**")
        idx = text.find(bold)
        if idx < 0:
            return ""
        header = bold
    start = idx + len(header)
    end = len(text)
    for other in _SECTION_HEADERS:
        for variant in (other, other.replace(":", ":**")):
            ni = text.find(variant, start + 1)
            if ni > 0:
                end = min(end, ni)
    return text[start:end].strip()


def _has_all_sections(text: str) -> bool:
    return all(
        (h in text) or (h.replace(":", ":**") in text)
        for h in _SECTION_HEADERS
    )


def _bigram_is_specific(bigram: str) -> bool:
    """Return True if a bigram carries parameter-specific information.

    Rejects module+number bigrams like "oscillator 2" or "lfo 1" that only name
    the synthesis module without specifying which parameter changed.  These appear
    in low-quality template commentary and should not count as alignment evidence.
    """
    words = bigram.split()
    if len(words) != 2:
        return True
    w0, w1 = words
    # Reject: entity-word followed by a bare number ("oscillator 2", "modulation 10")
    if w0 in _ENTITY_WORDS and w1.isdigit():
        return False
    # Reject: bare number followed by an entity-word ("2 oscillator") — unusual but safe
    if w0.isdigit() and w1 in _ENTITY_WORDS:
        return False
    return True


def _param_mentioned_in_text(text: str, param_name: str) -> bool:
    """True if a specific sub-parameter reference from *display_name* appears in *text*.

    Requires a bigram that contains at least one word beyond the module name and
    number (e.g. "2 on", "10 bypass", "distortion type") — plain module references
    like "oscillator 2" or "modulation 10" are explicitly rejected.
    """
    # Strip possessives (e.g. "LFO 8's phase" → "lfo 8 phase") before bigram search.
    text_lower = re.sub(r"'s\b", "", text.lower())
    display = _json_key_to_display(param_name).lower()
    words = display.split()
    for i in range(len(words) - 1):
        bigram = words[i] + " " + words[i + 1]
        if bigram in text_lower and _bigram_is_specific(bigram):
            return True
    return False


def _word_set(text: str) -> set[str]:
    return set(re.findall(r"\b\w+\b", text.lower()))


def _jaccard(a: set, b: set) -> float:
    union = len(a | b)
    return len(a & b) / union if union else 0.0


def _parse_json_object(text: str) -> dict[str, Any] | None:
    """Best-effort parse of a JSON object from *text*."""
    raw = text.strip()
    if not raw:
        return None
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    for ln in reversed(lines):
        try:
            parsed = json.loads(ln)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            continue
    return None


def _normalise_path_string(path_str: str | None) -> str | None:
    if not isinstance(path_str, str) or not path_str:
        return None
    return str(Path(path_str).expanduser())


def _runtime_path_from_payload(payload: dict[str, Any] | None) -> str | None:
    """Extract a comparable path from runtime JSON payloads."""
    if not isinstance(payload, dict):
        return None
    if isinstance(payload.get("path"), str):
        return _normalise_path_string(payload.get("path"))
    listen_probe = payload.get("listen_probe")
    if isinstance(listen_probe, dict) and isinstance(listen_probe.get("path"), str):
        return _normalise_path_string(listen_probe.get("path"))
    return None


def _validate_bash_execution_against_expected(
    command: str,
    expected_payload: dict[str, Any] | None,
    runtime_payload: dict[str, Any] | None,
    exit_code: int,
) -> list[str]:
    """Schema + key-field validation for a bash tool execution."""
    errors: list[str] = []
    if exit_code != 0:
        errors.append(f"nonzero_exit:{exit_code}")

    if isinstance(expected_payload, dict):
        expected_status = expected_payload.get("status")
        if expected_status == "ok" and exit_code != 0:
            errors.append("expected_status_ok_but_command_failed")
        if expected_status is not None and isinstance(runtime_payload, dict) and "status" in runtime_payload:
            if runtime_payload.get("status") != expected_status:
                errors.append(
                    f"status_mismatch:expected={expected_status!r}:got={runtime_payload.get('status')!r}"
                )

        if "path" in expected_payload:
            expected_path = _normalise_path_string(expected_payload.get("path"))
            got_path = _runtime_path_from_payload(runtime_payload)
            if expected_path and got_path is None:
                errors.append("missing_runtime_path")
            elif expected_path and got_path and expected_path != got_path:
                errors.append(f"path_mismatch:expected={expected_path}:got={got_path}")

        if "applied_wavetable" in expected_payload:
            if not isinstance(runtime_payload, dict):
                errors.append("missing_runtime_json_for_applied_wavetable")
            elif runtime_payload.get("applied_wavetable") != expected_payload.get("applied_wavetable"):
                errors.append(
                    "applied_wavetable_mismatch:"
                    f"expected={expected_payload.get('applied_wavetable')!r}:"
                    f"got={runtime_payload.get('applied_wavetable')!r}"
                )
        if "applied_tuple_id" in expected_payload:
            if not isinstance(runtime_payload, dict):
                errors.append("missing_runtime_json_for_applied_tuple_id")
            elif runtime_payload.get("applied_tuple_id") != expected_payload.get("applied_tuple_id"):
                errors.append(
                    "applied_tuple_id_mismatch:"
                    f"expected={expected_payload.get('applied_tuple_id')!r}:"
                    f"got={runtime_payload.get('applied_tuple_id')!r}"
                )

    # "Intended effect" heuristics for common command families.
    if "listen_probe" in command:
        if not isinstance(runtime_payload, dict) or not isinstance(runtime_payload.get("listen_probe"), dict):
            errors.append("listen_probe_payload_missing")
        else:
            lp = runtime_payload["listen_probe"]
            if lp.get("exists") is False:
                errors.append("listen_probe_reported_missing_file")
            if isinstance(expected_payload, dict) and "path" in expected_payload:
                expected_path = _normalise_path_string(expected_payload.get("path"))
                got_path = _normalise_path_string(lp.get("path"))
                if expected_path and got_path and expected_path != got_path:
                    errors.append(
                        f"listen_probe_path_mismatch:expected={expected_path}:got={got_path}"
                    )

    if "set_params(" in command or "TrackFX_SetParam" in command:
        if not isinstance(runtime_payload, dict):
            errors.append("set_params_missing_runtime_json")
        else:
            if isinstance(runtime_payload.get("not_found"), list) and runtime_payload["not_found"]:
                errors.append(f"set_params_not_found:{len(runtime_payload['not_found'])}")
            if "applied" in runtime_payload and isinstance(runtime_payload.get("applied"), (list, int)):
                applied = runtime_payload["applied"]
                if (isinstance(applied, list) and len(applied) == 0) or applied == 0:
                    errors.append("set_params_applied_empty")
            elif "status" not in runtime_payload:
                errors.append("set_params_missing_status_or_applied")

    return errors


def _classify_bash_command(command: str) -> str:
    if "listen_probe" in command:
        return "listen_probe"
    if "set_params(" in command or "TrackFX_SetParam" in command:
        return "set_params"
    if "applied_wavetable" in command or "vc.set_preset(" in command or "TrackFX_SetNamedConfigParm" in command:
        return "apply_wavetable"
    if "applied_tuple_id" in command:
        return "apply_tuple"
    return "bash_generic"


_TAUTOLOGY_READ_RE = re.compile(r"^\s*(cat|head|tail)\b")
_TAUTOLOGY_ECHO_RE = re.compile(r"^\s*(echo|printf)\b")


def _is_tautological_bash_command(command: str) -> bool:
    """Return True when execing *command* can't surface a real failure.

    These are builder round-trips — reads of files the builder itself wrote,
    or bare echo/printf of literals. Their tool_response is predetermined by
    conversation state, not by any system behaviour worth verifying.
    """
    first_line = command.strip().splitlines()[0] if command.strip() else ""
    if _TAUTOLOGY_READ_RE.match(first_line):
        return True
    if _TAUTOLOGY_ECHO_RE.match(first_line) and not any(
        tok in first_line for tok in ("|", "`", "$(", "$")
    ):
        return True
    return False


def _try_reapy_handshake(timeout_sec: float = 10.0) -> bool:
    """Return True if reapy can talk to a running REAPER session.

    Uses a lightweight TCP+reapy probe in a subprocess so a hung server
    can't block the caller (the subprocess gets killed on timeout).
    """
    import subprocess as _sp

    try:
        r = _sp.run(
            [
                "python", "-c",
                "import reapy\n"
                "with reapy.inside_reaper():\n"
                "    print(len(reapy.Project().tracks))\n",
            ],
            capture_output=True, text=True, timeout=timeout_sec,
        )
        return r.returncode == 0
    except Exception:
        return False


def _start_reaper_in_background(reaper_bin: str, log_path: str = "/tmp/reaper_start.log") -> None:
    """Spawn REAPER detached. Inherits DISPLAY when set; auto-detects the
    active X11 session otherwise. Output streams to ``log_path``."""
    import os as _os
    import subprocess as _sp

    env = _os.environ.copy()
    if "DISPLAY" not in env:
        for candidate in (":1", ":0"):
            xauth = f"/run/user/{_os.getuid()}/gdm/Xauthority"
            if _os.path.exists(xauth):
                env["XAUTHORITY"] = xauth
            try:
                _sp.run(
                    ["xdpyinfo", "-display", candidate],
                    capture_output=True, timeout=3, env=env,
                )
                env["DISPLAY"] = candidate
                break
            except Exception:
                continue
        else:
            env.setdefault("DISPLAY", ":0")
    with open(log_path, "ab") as logf:
        _sp.Popen(
            [reaper_bin, "-nosplash"],
            stdout=logf, stderr=logf, stdin=_sp.DEVNULL,
            preexec_fn=_os.setsid,  # detach so REAPER survives the grader's exit
            env=env,
        )


def _check_live_exec_environment(auto_start: bool = True) -> None:
    """Ensure a live REAPER + reapy session is reachable before executing checks.

    If ``auto_start`` is True (default) and the handshake fails, try to spawn
    REAPER (via $REAPER_BIN or the canonical install path) and wait up to ~25s
    for its reapy server to come up. Raises RuntimeError if no REAPER is
    reachable after the wait.
    """
    if _try_reapy_handshake():
        return

    if not auto_start:
        raise RuntimeError(
            "Live execution checks require a running REAPER session with reapy server available."
        )

    import os as _os
    import shutil as _sh
    import time as _time

    reaper_bin = (
        _os.environ.get("REAPER_BIN")
        or _sh.which("reaper")
        or "/home/nate/opt/REAPER/REAPER/reaper"
    )
    if not _os.path.exists(reaper_bin):
        raise RuntimeError(
            "Live execution checks require REAPER. Tried to auto-start but "
            f"could not find the binary (looked at $REAPER_BIN, PATH, and "
            f"{reaper_bin!r}). Start REAPER manually or set REAPER_BIN."
        )

    print(f"REAPER not reachable — auto-starting from {reaper_bin}", flush=True)
    try:
        _start_reaper_in_background(reaper_bin)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to spawn REAPER ({reaper_bin}): {exc}. Start it manually."
        ) from exc

    # Wait for the reapy server to come up. REAPER startup is ~5-10s plus the
    # defer-loop server takes another few seconds to bind.
    timeout_s = float(_os.environ.get("REAPER_START_TIMEOUT_S", "60"))
    deadline = _time.monotonic() + timeout_s
    delay = 2.0
    last_print = _time.monotonic()
    while _time.monotonic() < deadline:
        _time.sleep(delay)
        if _try_reapy_handshake():
            print("REAPER + reapy ready.", flush=True)
            return
        # Heartbeat every ~10s so the user sees we're still trying.
        if _time.monotonic() - last_print > 10.0:
            print(f"  ...still waiting for reapy server (timeout in {int(deadline - _time.monotonic())}s)", flush=True)
            last_print = _time.monotonic()
        delay = min(delay * 1.3, 4.0)

    raise RuntimeError(
        f"Auto-started REAPER but reapy never connected within {int(timeout_s)}s. "
        "Check that the reapy server script is configured to autorun "
        "(python -c 'import reapy; reapy.configure_reaper()' once). "
        "Override the timeout with REAPER_START_TIMEOUT_S=120."
    )


def _reset_reaper_project(record: dict[str, Any] | None = None) -> None:
    """Reset tracks/items in REAPER so each record starts from a clean slate.

    Runs in a subprocess so the reapy connection is fully closed when done,
    avoiding stale HOLD connections that block the single-threaded reapy server.
    """
    required_tracks = 0
    if record is not None:
        task_type = record.get("task_type", "")
        if task_type == "melody_transcription":
            meta = record.get("meta") or {}
            tidx = meta.get("track_idx")
            if isinstance(tidx, int) and tidx >= 0:
                required_tracks = tidx + 1

    script = (
        "import reapy\n"
        "with reapy.inside_reaper():\n"
        "    rpr = reapy.reascript_api\n"
        "    project = reapy.Project()\n"
        "    for track in reversed(list(project.tracks)):\n"
        "        try:\n"
        "            track.delete()\n"
        "        except Exception:\n"
        "            pass\n"
        f"    for i in range({required_tracks}):\n"
        "        rpr.InsertTrackAtIndex(i, False)\n"
    )
    try:
        subprocess.run(
            ["python", "-c", script],
            capture_output=True, text=True, timeout=15,
            start_new_session=True,
        )
    except Exception:
        pass


def run_live_execution_checks_for_record(
    record: dict[str, Any],
    timeout_sec: float = 30.0,
    max_calls: int | None = None,
) -> dict[str, Any]:
    """Execute bash tool calls and compare runtime output against paired tool_response."""
    messages = record.get("messages", [])
    checks: list[dict[str, Any]] = []
    assessed = 0
    passed = 0
    skipped_tautologies = 0
    call_budget = max_calls if (max_calls is not None and max_calls > 0) else None

    for i, msg in enumerate(messages):
        if call_budget is not None and assessed >= call_budget:
            break
        if msg.get("role") != "tool_call":
            continue
        tc = _parse_json_object(msg.get("content", ""))
        if not isinstance(tc, dict) or tc.get("name") not in ("bash", "Bash"):
            continue

        command = tc.get("arguments", {}).get("command", "")
        if not isinstance(command, str) or not command.strip():
            continue

        if _is_tautological_bash_command(command):
            skipped_tautologies += 1
            continue

        expected_payload: dict[str, Any] | None = None
        if i + 1 < len(messages) and messages[i + 1].get("role") == "tool_response":
            expected_payload = _parse_json_object(messages[i + 1].get("content", ""))

        assessed += 1
        category = _classify_bash_command(command)
        try:
            proc = subprocess.Popen(
                ["bash", "-lc", command],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                start_new_session=True,
            )
            try:
                stdout, stderr = proc.communicate(timeout=timeout_sec)
            except subprocess.TimeoutExpired:
                import os as _os
                import signal as _sig
                try:
                    _os.killpg(proc.pid, _sig.SIGKILL)
                except OSError:
                    proc.kill()
                proc.wait()
                checks.append(
                    {
                        "tool_call_index": i,
                        "category": category,
                        "ok": False,
                        "exit_code": None,
                        "errors": [f"timeout:{timeout_sec}s"],
                        "stderr_snippet": "",
                    }
                )
                continue
            runtime_payload = _parse_json_object(stdout)
            errors = _validate_bash_execution_against_expected(
                command=command,
                expected_payload=expected_payload,
                runtime_payload=runtime_payload,
                exit_code=int(proc.returncode),
            )
            ok = len(errors) == 0
            if ok:
                passed += 1
            checks.append(
                {
                    "tool_call_index": i,
                    "category": category,
                    "ok": ok,
                    "exit_code": int(proc.returncode),
                    "errors": errors,
                    "stderr_snippet": (stderr or "").strip()[:300],
                }
            )
        except Exception as exc:
            checks.append(
                {
                    "tool_call_index": i,
                    "category": category,
                    "ok": False,
                    "exit_code": None,
                    "errors": [f"exec_error:{exc!r}"],
                    "stderr_snippet": "",
                }
            )

    fidelity = (passed / assessed) if assessed > 0 else None
    return {
        "execution_fidelity": round(float(fidelity), 4) if fidelity is not None else None,
        "execution_checks": checks,
        "execution_calls_assessed": assessed,
        "execution_calls_passed": passed,
        "execution_calls_skipped_tautology": skipped_tautologies,
    }


# ---------------------------------------------------------------------------
# LLM-as-judge
# ---------------------------------------------------------------------------

_LLM_JUDGE_SYSTEM = (
    "You are a precise quality assessor for AI synthesizer training data. "
    "You evaluate whether a PLAN section correctly documents the specific parameter "
    "changes for a given step — not whether the changes are musically good, only whether "
    "the PLAN accurately names what is being changed."
)

_LLM_JUDGE_PROMPT_TEMPLATE = """\
PARAMETERS CHANGED THIS STEP (ground truth):
{param_list}

PLAN TEXT TO EVALUATE:
{plan_text}

Evaluate in three steps, then give a score:

STEP 1 — List every parameter from the ground truth above.
STEP 2 — For each, mark whether the PLAN names it (✓) or not (✗). Accept clear paraphrases:
  • "oscillator distortion" for osc_1_distortion_type ✓
  • "the filter" for filter_1_cutoff ✗  (too generic — no subsystem identifier)
  • "9 oscillator 1 parameters" or "N oscillator 1 parameters" covers ALL osc_1_* params in the list ✓
  • "N [subsystem] parameters" (e.g. "6 compressor parameters", "8 reverb parameters") counts as \
naming EVERY parameter from that subsystem in the ground truth list ✓
STEP 3 — Check if the PLAN names any parameters NOT in the ground truth list.

SCORING RUBRIC (use ONLY these three values):
  1.0 — PLAN names ≥1 step parameter AND does not name parameters outside the step list
  0.5 — PLAN names ≥1 step parameter BUT also names parameters outside the step list
  0.0 — PLAN names NO step parameters (references only unrelated or invented parameters)

Respond with JSON only, no other text:
{{"reasoning": "<your STEP 1-3 analysis in 2-3 sentences>", "score": <1.0|0.5|0.0>}}
"""


def _llm_judge_plan_step(
    plan_text: str,
    params_delta: list[dict],
    server_url: str,
    model: str,
    timeout: float = 10.0,
) -> tuple[float, str] | None:
    """Call a text LLM to judge whether *plan_text* references the changed parameters.

    Returns (score, reasoning) where score is in [0, 1] and reasoning is the CoT text,
    or None if the server is unavailable / response invalid.

    Scores are discrete: 1.0 (correct params only), 0.5 (correct + extras), 0.0 (wrong params).
    """
    if not _HTTPX_AVAILABLE:
        return None
    if not plan_text.strip() or not params_delta:
        return None

    param_list = "\n".join(
        f"  - {_json_key_to_display(p['name'])} ({p['name']})" for p in params_delta
    )
    prompt = _LLM_JUDGE_PROMPT_TEMPLATE.format(
        plan_text=plan_text.strip()[:600],
        param_list=param_list,
    )

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": _LLM_JUDGE_SYSTEM},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": 300,
        "temperature": 0.0,
    }

    try:
        url = f"{server_url.rstrip('/')}/v1/chat/completions"
        resp = _httpx.post(url, json=payload, timeout=timeout)
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"].strip()
        # Strip markdown code fences if present
        content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content, flags=re.MULTILINE).strip()
        parsed = json.loads(content)
        score = float(parsed["score"])
        reasoning = str(parsed.get("reasoning", ""))
        return max(0.0, min(1.0, score)), reasoning
    except Exception:
        return None


def llm_judge_main_record(
    record: dict[str, Any],
    server_url: str,
    model: str,
    timeout: float = 10.0,
) -> dict[str, Any]:
    """Run LLM judge on all commentary steps for a main record.

    Returns a dict with:
        llm_plan_alignment  – mean score across assessable steps (or None)
        llm_per_step        – list of {step, score, reason} dicts
    """
    messages = record.get("messages", [])
    meta = record.get("meta", {})
    step_labels: list[dict] = meta.get("step_labels", [])

    commentary_turns = [
        m["content"] for m in messages
        if m["role"] == "assistant"
        and ("HEARD" in m.get("content", "") or "PLAN" in m.get("content", ""))
    ]

    if not step_labels or not commentary_turns:
        return {"llm_plan_alignment": None, "llm_per_step": []}

    step_scores: list[float] = []
    per_step: list[dict] = []
    for label, turn in zip(step_labels, commentary_turns):
        delta: list[dict] = label.get("params_delta") or []
        if not delta:
            per_step.append({"step": label.get("step"), "assessable": False, "score": None})
            continue
        plan_text = _extract_section(turn, "PLAN:")
        result = _llm_judge_plan_step(plan_text, delta, server_url, model, timeout)
        if result is not None:
            score, reasoning = result
            step_scores.append(score)
        else:
            score, reasoning = None, ""
        per_step.append({
            "step": label.get("step"),
            "assessable": True,
            "score": round(score, 4) if score is not None else None,
            "reasoning": reasoning,
        })

    alignment = sum(step_scores) / len(step_scores) if step_scores else None
    return {
        "llm_plan_alignment": round(alignment, 4) if alignment is not None else None,
        "llm_per_step": per_step,
    }


# ---------------------------------------------------------------------------
# Per-task-type scorers
# ---------------------------------------------------------------------------

def _score_clap_net_improvement(step_labels: list[dict]) -> float | None:
    """Net CLAP improvement from first step to last, normalized to [0, 1].

    Measures whether the path made meaningful net progress toward the GT audio,
    regardless of noisy intermediate steps. A delta of +0.30 cosine units or more
    scores 1.0; a delta of 0 or below scores 0.0; values in between scale linearly.

    This replaced clap_monotonic, which penalized intentional path_gen noise injection
    that causes individual steps to fluctuate even when the path is converging.
    """
    scores = [sl.get("clap_score") for sl in step_labels if sl.get("clap_score") is not None]
    if len(scores) < 2:
        return None
    delta = scores[-1] - scores[0]
    return round(min(1.0, max(0.0, delta / 0.30)), 4)


def _score_plan_rationale_unique(commentary_turns: list[str]) -> float | None:
    """Fraction of consecutive PLAN Sentence-2 pairs that are sufficiently different.

    PLAN Sentence 1 is pre-seeded with the parameter inventory and is always correct.
    Sentence 2 is the free-form rationale. This metric flags paths where the model
    degrades into copy-pasting the same boilerplate rationale across steps.

    For each consecutive pair (step N, step N+1), computes Jaccard word-overlap on
    their Sentence 2 texts. A pair scores 1.0 if overlap < 0.60, else 0.0.
    Returns None when fewer than 2 turns have an extractable Sentence 2.
    """
    def _plan_sent2(turn: str) -> str | None:
        plan = _extract_section(turn, "PLAN:")
        if not plan:
            return None
        sents = [s.strip() for s in plan.split(".") if len(s.strip()) > 15]
        return sents[1] if len(sents) > 1 else (sents[0] if sents else None)

    sent2s = [s for t in commentary_turns if (s := _plan_sent2(t)) is not None]
    if len(sent2s) < 2:
        return None

    scores: list[float] = []
    for a, b in zip(sent2s, sent2s[1:]):
        a_words = set(a.lower().split())
        b_words = set(b.lower().split())
        union = a_words | b_words
        jaccard = len(a_words & b_words) / len(union) if union else 1.0
        scores.append(1.0 if jaccard < 0.60 else 0.0)
    return round(sum(scores) / len(scores), 4)


def _score_hypothesis_grounding(
    commentary_turns: list[str],
    step_labels: list[dict],
) -> float | None:
    """Score HYPOTHESIS grounding against the labeled remaining subsystem hints.

    For each step with a non-empty ``remaining_top_2`` label, this checks whether the
    HYPOTHESIS text names at least one of those subsystem tokens.
    Returns None when no step is assessable.
    """
    assessable = 0
    hits = 0

    for label, turn in zip(step_labels, commentary_turns):
        remaining = label.get("remaining_top_2")
        if not remaining:
            continue

        if isinstance(remaining, str):
            cleaned = remaining.lower().replace("/", " and ").replace(",", " and ")
            candidates = [part.strip() for part in re.split(r"\band\b", cleaned)]
        elif isinstance(remaining, list):
            candidates = [str(x).strip().lower() for x in remaining if str(x).strip()]
        else:
            continue

        tokens = []
        for c in candidates:
            t = re.sub(r"[^a-z0-9 ]+", " ", c).strip()
            if len(t) >= 3:
                tokens.append(t)
        if not tokens:
            continue

        assessable += 1
        hypo_text = _extract_section(turn, "HYPOTHESIS:").lower()
        if not hypo_text:
            continue

        if any(re.search(rf"\b{re.escape(tok)}\b", hypo_text) for tok in tokens):
            hits += 1

    if assessable == 0:
        return None
    return round(hits / assessable, 4)


def score_main_record(
    record: dict[str, Any],
    llm_judge_server: str | None = None,
    llm_judge_model: str = "Qwen/Qwen3-Omni-30B-A3B-Instruct",
) -> dict[str, Any]:
    messages = record.get("messages", [])
    meta = record.get("meta", {})
    labels = record.get("labels", {})
    step_labels: list[dict] = meta.get("step_labels") or labels.get("step_labels", [])

    commentary_turns = [
        m["content"] for m in messages
        if m["role"] == "assistant"
        and ("HEARD" in m.get("content", "") or "PLAN" in m.get("content", ""))
    ]
    assistant_turns = [m["content"] for m in messages if m["role"] == "assistant"]

    # Separate listening turns (HEARD/HYPOTHESIS/PLAN) from planning-only turns (PLAN only).
    # Planning turns are exempt from section_structure and hypothesis_grounding checks —
    # they are intentionally PLAN-only by design (audio gate suppressed the listen turn).
    listening_turns = [t for t in commentary_turns if "HEARD:" in t]
    planning_only_turns = [t for t in commentary_turns if "HEARD:" not in t and "PLAN:" in t]

    # 1. Section structure — only applies to listening turns (full HEARD/HYPOTHESIS/PLAN expected).
    if listening_turns:
        section_structure = sum(_has_all_sections(t) for t in listening_turns) / len(listening_turns)
    elif commentary_turns:
        # All turns are planning-only — no structure to check; treat as N/A → 1.0.
        section_structure = 1.0
    else:
        section_structure = 0.0

    # 2. Snake-case cleanliness — checked on section bodies only.
    # Commentary turns may have a preamble line ("Selected candidates: wt_name_v1")
    # before the HEARD/HYPOTHESIS/PLAN sections that legitimately contains wavetable
    # filenames with underscores.  Extracting the section text avoids those false positives.
    # For non-commentary turns, check the whole content.
    def _section_body(t: str) -> str:
        """Return the concatenated HEARD+HYPOTHESIS+PLAN body, or the full text."""
        parts = [_extract_section(t, h) for h in _SECTION_HEADERS]
        body = " ".join(p for p in parts if p)
        return body if body else t
    snake_check_texts = [_section_body(t) for t in (commentary_turns if commentary_turns else assistant_turns)]
    snake_hits = sum(bool(_SNAKE_CASE_RE.search(t)) for t in snake_check_texts)
    snake_case_clean = 1.0 - min(1.0, snake_hits / max(1, len(snake_check_texts)))

    # 3. Format consistency (no **BOLD:** headers)
    bold_hits = sum(bool(_BOLD_HEADER_RE.search(t)) for t in commentary_turns)
    format_consistent = 1.0 - min(1.0, bold_hits / max(1, len(commentary_turns)))

    # 4. Commentary diversity (low pairwise Jaccard = diverse = good)
    if len(commentary_turns) >= 2:
        word_sets = [_word_set(t) for t in commentary_turns]
        pairs = [
            _jaccard(word_sets[i], word_sets[j])
            for i in range(len(word_sets))
            for j in range(i + 1, len(word_sets))
        ]
        commentary_diversity = 1.0 - (sum(pairs) / len(pairs))
    else:
        commentary_diversity = 1.0

    # 5. Plan-param alignment
    plan_param_alignment: float | None = None
    per_step_alignment: list[dict] = []
    if step_labels and commentary_turns:
        aligned = 0
        assessable = 0
        for label, turn in zip(step_labels, commentary_turns):
            delta: list[dict] = label.get("params_delta") or []
            if not delta:
                per_step_alignment.append({"step": label.get("step"), "assessable": False})
                continue
            assessable += 1
            plan_text = _extract_section(turn, "PLAN:")
            matched_names = [
                d["name"] for d in delta if _param_mentioned_in_text(plan_text, d["name"])
            ]
            hit = len(matched_names) > 0
            if hit:
                aligned += 1
            per_step_alignment.append({
                "step": label.get("step"),
                "assessable": True,
                "hit": hit,
                "matched_params": matched_names,
                "plan_snippet": plan_text[:120],
            })
        plan_param_alignment = aligned / assessable if assessable > 0 else None

    # 6. Plan rationale uniqueness — penalises paths where PLAN Sentence 2 degrades
    # into repeated boilerplate. plan_param_alignment is always 1.0 because Sentence 1
    # is pre-seeded with the correct param inventory; this metric captures the quality
    # of the free-form Sentence 2 rationale instead.
    plan_rationale_unique = _score_plan_rationale_unique(commentary_turns)

    # 7. Hypothesis grounding — fraction of labeled steps where HYPOTHESIS names
    # at least one subsystem in remaining_top_2.
    hypothesis_grounding = _score_hypothesis_grounding(listening_turns, step_labels)

    # 8. CLAP net improvement — did the path make meaningful net progress toward GT audio?
    # Measures final-vs-initial CLAP delta normalized to [0,1]. Replaces clap_monotonic
    # which penalized intentional path_gen noise on intermediate steps.
    clap_net_improvement = _score_clap_net_improvement(step_labels)

    # 9. Optional LLM judge — when enabled and available, use this for plan alignment
    # in overall scoring (it can catch paraphrases missed by the heuristic matcher).
    llm_results: dict[str, Any] = {}
    if llm_judge_server:
        llm_results = llm_judge_main_record(record, llm_judge_server, llm_judge_model)

    llm_alignment: float | None = llm_results.get("llm_plan_alignment")
    plan_alignment_for_overall = llm_alignment if llm_alignment is not None else plan_param_alignment

    # Overall weighted score.
    # commentary_diversity is also excluded — it has a structural ceiling (~0.67) in this
    # domain regardless of content quality.
    weights = {
        "plan_alignment_for_overall": 0.45,
        "clap_net_improvement":       0.20,
        "plan_rationale_unique":      0.15,
        "hypothesis_grounding":       0.10,
        "section_structure":          0.05,
        "snake_case_clean":           0.025,
        "format_consistent":          0.025,
    }
    raw = {
        "plan_alignment_for_overall": plan_alignment_for_overall,
        "clap_net_improvement": clap_net_improvement,
        "plan_rationale_unique": plan_rationale_unique,
        "hypothesis_grounding": hypothesis_grounding,
        "section_structure": section_structure,
        "snake_case_clean": snake_case_clean,
        "commentary_diversity": commentary_diversity,
        "format_consistent": format_consistent,
        # Keep alignment scores as diagnostics
        "plan_param_alignment": plan_param_alignment,
    }
    weighted_sum = 0.0
    weight_sum = 0.0
    for k, w in weights.items():
        v = raw[k]
        if v is not None:
            weighted_sum += v * w
            weight_sum += w
    overall = weighted_sum / weight_sum if weight_sum > 0 else 0.0

    result: dict[str, Any] = {
        **raw,
        "heuristic_plan_param_alignment": plan_param_alignment,
        "overall": round(overall, 4),
        "per_step_alignment": per_step_alignment,
    }
    if llm_judge_server:
        result["llm_plan_alignment"] = llm_alignment
        result["llm_per_step"] = llm_results.get("llm_per_step", [])
    return result


def score_search_record(record: dict[str, Any]) -> dict[str, Any]:
    messages = record.get("messages", [])
    assistant_turns = [m["content"] for m in messages if m["role"] == "assistant"]

    # has_cosine_scores: audition tool_response carries cosine_vs_target
    has_cosine = 0.0
    for m in messages:
        if m["role"] == "tool_response":
            try:
                parsed = json.loads(m["content"])
                results = parsed.get("audition_results", [])
                if results and all("cosine_vs_target" in r for r in results):
                    has_cosine = 1.0
                    break
            except Exception:
                pass

    # proposal_diversity: are proposal reasons distinct?
    proposal_diversity = 1.0
    for m in messages:
        if m["role"] == "assistant":
            try:
                parsed = json.loads(m["content"])
                proposals = parsed.get("proposals", [])
                if proposals and len(proposals) >= 2:
                    reasons = [p.get("reason", "") for p in proposals]
                    unique = len(set(reasons))
                    proposal_diversity = unique / len(reasons)
                    break
            except Exception:
                pass

    # no_gt_leak: no oracle identity text
    gt_leak_patterns = re.compile(r"\(gt\)|source wavetable", re.IGNORECASE)
    gt_leaks = sum(bool(gt_leak_patterns.search(t)) for t in assistant_turns)
    no_gt_leak = 1.0 if gt_leaks == 0 else 0.0

    weights = {"has_cosine_scores": 0.40, "proposal_diversity": 0.35, "no_gt_leak": 0.25}
    raw = {"has_cosine_scores": has_cosine, "proposal_diversity": proposal_diversity, "no_gt_leak": no_gt_leak}
    overall = sum(raw[k] * w for k, w in weights.items())

    return {**raw, "overall": round(overall, 4)}


def score_judge_record(record: dict[str, Any]) -> dict[str, Any]:
    messages = record.get("messages", [])
    labels = record.get("labels", {})

    # has_score_in_reason: final assistant JSON reason has numeric scores
    has_score = 0.0
    for m in reversed(messages):
        if m["role"] == "assistant":
            try:
                parsed = json.loads(m["content"])
                reason = parsed.get("reason", "")
                if _NUMERIC_SCORE_RE.search(reason):
                    has_score = 1.0
            except Exception:
                pass
            break

    # has_cosine_scores: audition tool_response carries cosine_vs_target
    has_cosine = 0.0
    for m in messages:
        if m["role"] == "tool_response":
            try:
                parsed = json.loads(m["content"])
                results = parsed.get("audition_results", [])
                if results and all("cosine_vs_target" in r for r in results):
                    has_cosine = 1.0
                    break
            except Exception:
                pass

    # gt_recall: if gt known, is ≥1 in selected?
    gt_ids = set(labels.get("gt_candidate_ids", []))
    selected = set(labels.get("selected", []))
    gt_recall: float | None = None
    if gt_ids:
        gt_recall = 1.0 if gt_ids & selected else 0.0

    weights = {"has_score_in_reason": 0.45, "has_cosine_scores": 0.30, "gt_recall": 0.25}
    raw = {"has_score_in_reason": has_score, "has_cosine_scores": has_cosine, "gt_recall": gt_recall}
    weight_sum = sum(w for k, w in weights.items() if raw[k] is not None)
    weighted_sum = sum(raw[k] * w for k, w in weights.items() if raw[k] is not None)
    overall = weighted_sum / weight_sum if weight_sum > 0 else 0.0

    return {**raw, "overall": round(overall, 4)}


# ---------------------------------------------------------------------------
# v3 LLM judge — targets narration-template filler, verdict pattern-matching,
# and plan↔narration grounding (failure modes the structural grader misses).
# ---------------------------------------------------------------------------

def _extract_v3_plan_and_narrations(record: dict) -> dict:
    """Pull plan text, per-batch narrations, and verdict from a v3 record.

    Returns {"plan": str, "observations": str, "narrations": [(subsystem, narration_text), ...],
             "verdict": str, "target_audio": str | None}. Batch narrations are
    the assistant turn that *follows* each ``batch_audio`` tool_response; the
    subsystem label is taken from meta.batch_labels in order (correction
    batches are skipped). ``target_audio`` is the first audio path (GT).
    """
    messages = record.get("messages", [])
    meta = record.get("meta", {})
    batch_labels = meta.get("batch_labels", []) or []
    audios = record.get("audios", []) or []
    target_audio = audios[0] if audios else None

    plan_text = ""
    for m in messages:
        if m.get("role") != "assistant":
            continue
        c = m.get("content", "")
        if isinstance(c, str) and "OBSERVATIONS" in c and "PLAN" in c:
            plan_text = c
            break

    observations = ""
    if plan_text:
        obs_match = re.search(
            r"OBSERVATIONS:\s*(.+?)(?:\n\s*PLAN:|\Z)", plan_text, flags=re.DOTALL
        )
        if obs_match:
            observations = obs_match.group(1).strip()

    narrations: list[tuple[str, str]] = []
    non_correction_labels = [bl for bl in batch_labels if not bl.get("is_correction")]
    bi = 0
    # Strip the trailing "Applying <next>_ changes." / "Correcting ..." / "Listening ..."
    # transition lines that reference the NEXT batch and would confuse per-batch
    # judges into seeing a hallucination where there is none.
    _trailer_re = re.compile(
        r"\n+\s*(Applying|Correcting|Listening)\b[^\n]*$", flags=re.IGNORECASE
    )
    for i, m in enumerate(messages):
        if m.get("role") != "tool_response":
            continue
        c = str(m.get("content", ""))[:300]
        if '"batch_audio"' not in c:
            continue
        if i + 1 >= len(messages):
            continue
        nxt = messages[i + 1]
        if nxt.get("role") != "assistant":
            continue
        text = str(nxt.get("content", "") or "")
        # Also strip any FINAL ASSESSMENT fold-in on the last batch turn.
        if "FINAL ASSESSMENT" in text:
            text = text.split("FINAL ASSESSMENT", 1)[0].rstrip()
        # Strip trailing transition line.
        text = _trailer_re.sub("", text).rstrip()
        if bi < len(non_correction_labels):
            narrations.append((non_correction_labels[bi].get("subsystem", ""), text))
        bi += 1

    verdict = ""
    for m in reversed(messages):
        c = m.get("content", "") or ""
        if m.get("role") == "assistant" and "FINAL ASSESSMENT" in str(c):
            verdict = str(c)
            break

    return {
        "plan": plan_text,
        "observations": observations,
        "narrations": narrations,
        "verdict": verdict,
        "target_audio": target_audio,
    }


_V3_NARRATION_JUDGE_SYSTEM = (
    "You are a strict quality assessor for synth-design training data. "
    "You evaluate whether a one-line narration of a parameter-edit batch is "
    "grounded in the plan and specific to the params edited, or whether it is "
    "generic template filler."
)

_V3_NARRATION_JUDGE_PROMPT = """\
PLAN:
{plan_text}

THIS BATCH:
  subsystem: {subsystem}
  params edited: {param_list}

OTHER NARRATIONS from the SAME subsystem ({subsystem}) in OTHER samples in this run
(for template-detection reference — compare against the narration under review):
{other_narrations_block}

NARRATION UNDER REVIEW (one sentence by the agent after applying the batch):
  "{narration}"

Score on three INDEPENDENT axes. Each axis measures ONE thing — do not bleed concerns across axes.

1. plan_reference (0.0 / 0.5 / 1.0)  — plan consistency
   1.0 = narration picks up a concrete intent from the plan's {subsystem} bullet
   0.5 = loose thematic alignment with the plan
   0.0 = narration is independent of the plan OR contradicts the plan (e.g. plan says
         "remove all modulation" and narration describes modulation being added)

2. parameter_specific (0.0 / 0.5 / 1.0)  — does the narration name audible consequences of THESE params
   1.0 = names an audible consequence clearly tied to the listed param families
   0.5 = correctly within {subsystem} domain but no specifics
   0.0 = generic synth-speak ("richer", "more complex", "subtle movement")

3. templateness (0.0 / 1.0 — BINARY)  — compare the narration UNDER REVIEW to the OTHER narrations above
   1.0 = different phrasing/structure from the other samples — would NOT fit if dropped into another
         sample's slot for the same subsystem (sample-specific language, unique anchors)
   0.0 = same template shape as the other samples — you could swap this narration into any other
         {subsystem} slot and it would fit (interchangeable phrasing)
   Pick 0.0 or 1.0. Do NOT pick 0.5 for templateness.

IMPORTANT: templateness is purely about phrasing uniqueness vs. the reference narrations above.
Do NOT mark templateness=0 because the narration contradicts the plan — that belongs to plan_reference.
Do NOT mark templateness=0 because it's generic — that belongs to parameter_specific.

Respond ONLY with JSON, no other text:
{{"plan_reference": <0.0|0.5|1.0>, "parameter_specific": <0.0|0.5|1.0>, "templateness": <0.0|1.0>, "reasoning": "<one sentence explicitly comparing this narration's phrasing to the reference ones>"}}
"""


_V3_VERDICT_JUDGE_SYSTEM = (
    "You evaluate FINAL ASSESSMENT text for synth-recreation training data. "
    "You judge whether the assessment cites a specific, plausible residual or "
    "pattern-matches to a subsystem name without justification."
)

_V3_VERDICT_JUDGE_PROMPT = """\
PLAN (what was planned):
{plan_text}

SUBSYSTEMS TOUCHED: {subsystems}

OTHER FINAL ASSESSMENTS FROM DIFFERENT SAMPLES IN THIS RUN (for template-detection reference):
{other_verdicts_block}

VERDICT UNDER REVIEW:
"{verdict_text}"

Score on two axes. Use ONLY 0.0, 0.5, or 1.0.

1. residual_grounded
   1.0 = cites a concrete residual discrepancy with specifics ("attack still softer than target",
         "high-frequency shimmer missing")
   0.5 = names a subsystem with at least some audible rationale
   0.0 = boilerplate naming a subsystem ("remaining difference lies in envelope 6") without
         perceptual grounding

2. novelty — compare the verdict UNDER REVIEW against the OTHER verdicts above. Use a BINARY scale (NO 0.5):
   1.0 = the verdict has specific content that would NOT fit if you swapped it into another sample's slot
   0.0 = same template shape as the other samples — same opening phrase pattern, same "envelope N" pattern,
         same generic "affecting X" filler — you could swap this verdict into any other sample's slot

Pick 0.0 or 1.0 — do not pick 0.5 for novelty.

Respond ONLY with JSON:
{{"residual_grounded": <0.0|0.5|1.0>, "novelty": <0.0|1.0>, "reasoning": "<one sentence comparing structure to other samples>"}}
"""


_CANDIDATE_AUDIO_JUDGE_SYSTEM = (
    "You evaluate whether a written TIMBRE description of a wavetable audio "
    "clip matches what is audible. Focus only on TIMBRAL qualities (harmonic "
    "content, brightness, texture, roughness, formants, attack shape) — NOT "
    "on the note pattern, melody, or number of notes in the clip."
)

_CANDIDATE_AUDIO_JUDGE_PROMPT = """\
You will hear a wavetable audio clip (a probe render: a fixed test pattern of 4 triads over ~10s through a default synth preset) and a one-sentence description written about the wavetable's TIMBRE.

Description:
"{desc}"

Judge only whether the TIMBRAL claims in the description are audible. Ignore anything the description says — or fails to say — about melody, notes, or note count. The note pattern is identical across all probes; what varies is the wavetable's timbre.

Examples of timbral claims to verify:
  • harmonic content: pure/rich/bright/dull, odd-only/even-only/inharmonic, buzzy/smooth
  • attack/decay shape: sharp pluck vs slow swell, static vs evolving
  • formant/resonance: nasal, hollow, vowel-like, peaky
  • overall character: warm/cold, clean/dirty, metallic, vocal, etc.

Scoring:
  1.0 = the timbral claims are clearly audible in the clip.
  0.5 = the description is generic/vague or only weakly supported, but not contradicted.
  0.0 = at least one concrete TIMBRAL claim is contradicted by what you hear (e.g. "bright and aggressive" but the clip is mellow and smooth).

Respond ONLY with JSON:
{{"audio_grounded": <0.0|0.5|1.0>, "reasoning": "<one sentence citing one audible timbral attribute that did or did not match>"}}
"""


def _resolve_audio(path: str | Path) -> Path | None:
    p = Path(str(path))
    if p.is_absolute():
        return p if p.exists() else None
    abs_p = ROOT / p
    return abs_p if abs_p.exists() else None


def _llm_judge_candidate_audio(
    description: str,
    audio_path: str,
    server_url: str,
    model: str,
    timeout: float = 30.0,
) -> dict | None:
    """Single-audio grounding check: does the description match this clip?

    Catches multi-audio position-confusion hallucinations that originate in
    the build-time batch Omni call (where N audios are sent in one request
    and the model can scramble which description belongs to which clip).
    """
    if not _HTTPX_AVAILABLE or not description.strip() or not audio_path:
        return None
    resolved = _resolve_audio(audio_path)
    if resolved is None:
        return None
    try:
        audio_b64 = _b64(resolved)
    except Exception:
        return None
    prompt = _CANDIDATE_AUDIO_JUDGE_PROMPT.format(desc=description.strip()[:400])
    content = [
        {"type": "audio_url", "audio_url": {"url": f"data:audio/wav;base64,{audio_b64}"}},
        {"type": "text", "text": prompt},
    ]
    try:
        resp = _httpx.post(
            f"{server_url.rstrip('/')}/v1/chat/completions",
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": _CANDIDATE_AUDIO_JUDGE_SYSTEM},
                    {"role": "user", "content": content},
                ],
                "max_tokens": 200,
                "temperature": 0.0,
            },
            timeout=timeout,
        )
        resp.raise_for_status()
        text = resp.json()["choices"][0]["message"]["content"].strip()
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.MULTILINE).strip()
        parsed = json.loads(text)
        return {
            "audio_grounded": float(parsed.get("audio_grounded", 0.0)),
            "reasoning": str(parsed.get("reasoning", "")),
        }
    except Exception:
        return None


def _extract_candidate_description(notes: str, name: str) -> str | None:
    """Parse a per-candidate line of form '<name>': <description>[. Selected|Not selected].

    Search agent output includes a trailing 'Selected' or 'Not selected' label;
    judge agent output omits that (uses a single 'RECOMMENDATION:' line at the
    end instead). Either format is accepted. Works line-by-line so a name-match
    in one line can't capture description text from a later line.
    """
    if not notes:
        return None
    esc = re.escape(name)
    line_pat = re.compile(rf"""^\s*['"]{esc}['"]\s*:\s*(.+?)\s*$""")
    for line in notes.splitlines():
        m = line_pat.match(line)
        if not m:
            continue
        desc = m.group(1).strip()
        # Strip trailing Selected/Not selected label if present (search format).
        desc = re.sub(
            r"\.\s*(Selected|Not selected)\s*\.?\s*$",
            "",
            desc,
            flags=re.IGNORECASE,
        ).strip()
        return desc or None
    return None


def _extract_search_candidate_triples(
    record: dict[str, Any],
) -> list[tuple[str, str, str]]:
    """Yield (name, description, audio_path) for each candidate in a search record.

    Walks tool_responses with a `rendered` list; each entry flagged
    `"audio": "<audio>"` consumes one slot from record["audios"][1:]
    (index 0 is the target). The description is pulled from the assistant
    turn immediately following each rendered-batch response.
    """
    messages = record.get("messages", []) or []
    audios = record.get("audios", []) or []
    if len(audios) <= 1:
        return []
    cursor = 1  # skip target audio
    triples: list[tuple[str, str, str]] = []
    for i, m in enumerate(messages):
        if m.get("role") != "tool_response":
            continue
        try:
            resp = json.loads(m.get("content", "") or "")
        except Exception:
            continue
        rendered = resp.get("rendered")
        if not isinstance(rendered, list):
            continue
        batch_pairs: list[tuple[str, str]] = []
        for entry in rendered:
            if not isinstance(entry, dict):
                continue
            name = entry.get("name")
            if not name:
                continue
            if entry.get("audio") != "<audio>":
                continue
            if cursor >= len(audios):
                break
            batch_pairs.append((str(name), str(audios[cursor])))
            cursor += 1
        nxt = messages[i + 1] if i + 1 < len(messages) else None
        if not nxt or nxt.get("role") != "assistant":
            continue
        notes = nxt.get("content", "") or ""
        for name, path in batch_pairs:
            desc = _extract_candidate_description(notes, name)
            if desc:
                triples.append((name, desc, path))
    return triples


def _extract_judge_candidate_triples(
    record: dict[str, Any],
) -> list[tuple[str, str, str]]:
    """Yield (name, description, audio_path) for each pool candidate in a judge record."""
    messages = record.get("messages", []) or []
    audios = record.get("audios", []) or []
    if len(audios) <= 1:
        return []
    # audios[0] is target; audios[1:] are pool candidate audios in append order.
    # The judge dumps per-candidate lines in the assistant deliberation turn(s).
    pool: list[str] = list((record.get("meta") or {}).get("pool") or [])
    if not pool:
        return []
    # Join all assistant content so the regex can find each candidate line.
    assistant_text = "\n".join(
        m.get("content", "") or ""
        for m in messages
        if m.get("role") == "assistant"
    )
    triples: list[tuple[str, str, str]] = []
    for i, name in enumerate(pool):
        audio_idx = i + 1  # offset past target
        if audio_idx >= len(audios):
            break
        desc = _extract_candidate_description(assistant_text, name)
        if desc:
            triples.append((name, desc, str(audios[audio_idx])))
    return triples


_V3_OBSERVATIONS_JUDGE_SYSTEM = (
    "You are evaluating whether the agent's written OBSERVATIONS about a target "
    "audio clip match the actual audible content of that clip. You have access "
    "to the audio and the text description — score how consistent they are."
)

_V3_OBSERVATIONS_JUDGE_PROMPT = """\
You will hear the TARGET audio, then read the agent's OBSERVATIONS text.

OBSERVATIONS text (what the agent claims to hear):
"{observations}"

Listen to the audio and evaluate:

1. audio_grounded
   1.0 = every attribute named in OBSERVATIONS is audibly present in the clip
   0.5 = most attributes are plausible but at least one claim is generic or weakly supported
   0.0 = one or more claims are clearly absent or contradicted by the audio

Respond ONLY with JSON:
{{"audio_grounded": <0.0|0.5|1.0>, "reasoning": "<one sentence citing one audible attribute that did or did not match>"}}
"""


_V3_HALLUCINATION_JUDGE_SYSTEM = (
    "You are checking a one-sentence synth-edit narration for parameter-hallucination. "
    "The narration should only reference effects/params that align with the batch's "
    "param list. If the narration names a param family that is NOT in the list, that "
    "is a hallucination."
)

_V3_HALLUCINATION_JUDGE_PROMPT = """\
This batch edited these parameters (grouped by the synth subsystem):
  subsystem: {subsystem}
  params: {param_list}

NARRATION written by the agent after applying the batch:
  "{narration}"

Does the narration reference any param-family effect that is NOT in the param list?

Examples of hallucinations:
  • Batch edited 'chorus', narration mentions "phaser modulation" → hallucination (phaser ≠ chorus)
  • Batch edited 'lfo_1_*', narration mentions "chorus and phaser" → hallucination
  • Batch edited 'filter_1_cutoff', narration mentions "reverb tail" → hallucination
  • Batch edited 'fx' (chorus only), narration mentions "spatial reverb" → hallucination

NOT hallucinations (these are fine):
  • Describing abstract audio qualities ("brighter", "darker", "richer")
  • Mentioning the subsystem generically ("filter", "envelope")
  • Mentioning an effect that's a subtype of what was edited (e.g. "resonance sweep" for filter params)

Respond ONLY with JSON:
{{"no_hallucination": <0.0|1.0>, "reasoning": "<one sentence — if hallucinated, name the offending word>"}}
"""


def _llm_judge_v3_narration(
    plan_text: str,
    subsystem: str,
    param_names: list[str],
    narration: str,
    server_url: str,
    model: str,
    other_narrations: list[str] | None = None,
    timeout: float = 15.0,
) -> dict | None:
    if not _HTTPX_AVAILABLE or not plan_text.strip() or not narration.strip():
        return None
    others = other_narrations or []
    if others:
        others_block = "\n".join(
            f"  [sample {i+1}]: \"{v.strip()[:280]}\"" for i, v in enumerate(others)
        )
    else:
        others_block = "  (no other same-subsystem narrations available — evaluate templateness standalone)"
    prompt = _V3_NARRATION_JUDGE_PROMPT.format(
        plan_text=plan_text.strip()[:900],
        subsystem=subsystem,
        param_list=", ".join(param_names[:8]) + ("…" if len(param_names) > 8 else ""),
        other_narrations_block=others_block,
        narration=narration.strip()[:400],
    )
    try:
        resp = _httpx.post(
            f"{server_url.rstrip('/')}/v1/chat/completions",
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": _V3_NARRATION_JUDGE_SYSTEM},
                    {"role": "user", "content": prompt},
                ],
                "max_tokens": 250,
                "temperature": 0.0,
            },
            timeout=timeout,
        )
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"].strip()
        content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content, flags=re.MULTILINE).strip()
        parsed = json.loads(content)
        return {
            "plan_reference": float(parsed["plan_reference"]),
            "parameter_specific": float(parsed["parameter_specific"]),
            "templateness": float(parsed["templateness"]),
            "reasoning": str(parsed.get("reasoning", "")),
        }
    except Exception:
        return None


def _llm_judge_v3_verdict(
    plan_text: str,
    subsystems: list[str],
    verdict: str,
    server_url: str,
    model: str,
    other_verdicts: list[str] | None = None,
    timeout: float = 15.0,
) -> dict | None:
    if not _HTTPX_AVAILABLE or not verdict.strip():
        return None
    others = other_verdicts or []
    if others:
        others_block = "\n".join(
            f"  [sample {i+1}]: \"{v.strip()[:400]}\"" for i, v in enumerate(others)
        )
    else:
        others_block = "  (no other verdicts available — evaluate novelty standalone)"
    prompt = _V3_VERDICT_JUDGE_PROMPT.format(
        plan_text=plan_text.strip()[:900] or "(plan unavailable)",
        subsystems=", ".join(subsystems),
        other_verdicts_block=others_block,
        verdict_text=verdict.strip()[:600],
    )
    try:
        resp = _httpx.post(
            f"{server_url.rstrip('/')}/v1/chat/completions",
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": _V3_VERDICT_JUDGE_SYSTEM},
                    {"role": "user", "content": prompt},
                ],
                "max_tokens": 250,
                "temperature": 0.0,
            },
            timeout=timeout,
        )
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"].strip()
        content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content, flags=re.MULTILINE).strip()
        parsed = json.loads(content)
        return {
            "residual_grounded": float(parsed["residual_grounded"]),
            "novelty": float(parsed["novelty"]),
            "reasoning": str(parsed.get("reasoning", "")),
        }
    except Exception:
        return None


def _llm_judge_v3_observations_audio(
    observations: str,
    target_audio_path: str | None,
    server_url: str,
    model: str,
    timeout: float = 45.0,
) -> dict | None:
    """Send target WAV + OBSERVATIONS text to Qwen-Omni and check if the
    description matches what's audible in the clip."""
    if not _HTTPX_AVAILABLE or not observations.strip() or not target_audio_path:
        return None
    try:
        audio_b64 = _b64(target_audio_path)
    except Exception:
        return None
    prompt = _V3_OBSERVATIONS_JUDGE_PROMPT.format(
        observations=observations.strip()[:600],
    )
    content = [
        {"type": "audio_url", "audio_url": {"url": f"data:audio/wav;base64,{audio_b64}"}},
        {"type": "text", "text": prompt},
    ]
    try:
        resp = _httpx.post(
            f"{server_url.rstrip('/')}/v1/chat/completions",
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": _V3_OBSERVATIONS_JUDGE_SYSTEM},
                    {"role": "user", "content": content},
                ],
                "max_tokens": 250,
                "temperature": 0.0,
            },
            timeout=timeout,
        )
        resp.raise_for_status()
        text = resp.json()["choices"][0]["message"]["content"].strip()
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.MULTILINE).strip()
        parsed = json.loads(text)
        return {
            "audio_grounded": float(parsed["audio_grounded"]),
            "reasoning": str(parsed.get("reasoning", "")),
        }
    except Exception:
        return None


def _llm_judge_v3_param_hallucination(
    subsystem: str,
    param_names: list[str],
    narration: str,
    server_url: str,
    model: str,
    timeout: float = 15.0,
) -> dict | None:
    """Binary check: does the narration reference any param family not in the batch?"""
    if not _HTTPX_AVAILABLE or not narration.strip():
        return None
    prompt = _V3_HALLUCINATION_JUDGE_PROMPT.format(
        subsystem=subsystem,
        param_list=", ".join(param_names[:12]) + ("…" if len(param_names) > 12 else ""),
        narration=narration.strip()[:400],
    )
    try:
        resp = _httpx.post(
            f"{server_url.rstrip('/')}/v1/chat/completions",
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": _V3_HALLUCINATION_JUDGE_SYSTEM},
                    {"role": "user", "content": prompt},
                ],
                "max_tokens": 200,
                "temperature": 0.0,
            },
            timeout=timeout,
        )
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"].strip()
        content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content, flags=re.MULTILINE).strip()
        parsed = json.loads(content)
        return {
            "no_hallucination": float(parsed["no_hallucination"]),
            "reasoning": str(parsed.get("reasoning", "")),
        }
    except Exception:
        return None


def llm_judge_v3_record(
    record: dict[str, Any],
    server_url: str,
    model: str,
    other_verdicts: list[str] | None = None,
    other_narrations_by_subsystem: dict[str, list[str]] | None = None,
    timeout: float = 15.0,
) -> dict[str, Any]:
    """Run v3-aware LLM judge: narration grounding per batch, verdict
    substantiveness (with cross-sample context), parameter-hallucination
    binary per batch, and audio-grounded observations check per sample.

    When ``other_narrations_by_subsystem`` is supplied, the narration judge
    sees 2-3 same-subsystem narrations from OTHER samples as reference for
    template detection — lets it produce a cleaner binary templateness signal.

    Returns a dict with aggregated scores and per-batch details. Gracefully
    returns ``None`` values when httpx is unavailable or the server fails.
    """
    extracted = _extract_v3_plan_and_narrations(record)
    plan_text = extracted["plan"]
    observations = extracted["observations"]
    narrations = extracted["narrations"]
    verdict = extracted["verdict"]
    target_audio = extracted["target_audio"]
    meta = record.get("meta", {})
    batch_labels = [bl for bl in meta.get("batch_labels", []) if not bl.get("is_correction")]

    other_by_sub = other_narrations_by_subsystem or {}

    plan_refs: list[float] = []
    param_specs: list[float] = []
    templateness: list[float] = []
    no_halluc: list[float] = []
    # Parallelize: every batch's narration + hallucination judge calls are
    # independent of every other batch's. Plus verdict + observations are
    # independent of all batches. Fire everything in one ThreadPoolExecutor
    # so vLLM's batching can absorb the whole record's LLM-judge workload.
    from concurrent.futures import ThreadPoolExecutor as _TPE
    n_batches = len(narrations)
    max_workers = max(2, min(32, 2 * n_batches + 2))  # 2 calls per batch + verdict + obs

    subsystems = [bl.get("subsystem", "") for bl in batch_labels]
    per_batch_entries: list[dict | None] = [None] * n_batches

    def _run_narr(i: int, subsystem: str, narration: str, param_names: list, other_nars: list):
        return i, _llm_judge_v3_narration(
            plan_text=plan_text, subsystem=subsystem, param_names=param_names,
            narration=narration, server_url=server_url, model=model,
            other_narrations=other_nars, timeout=timeout,
        )

    def _run_hall(i: int, subsystem: str, narration: str, param_names: list):
        return i, _llm_judge_v3_param_hallucination(
            subsystem=subsystem, param_names=param_names, narration=narration,
            server_url=server_url, model=model, timeout=timeout,
        )

    def _run_verdict():
        return _llm_judge_v3_verdict(
            plan_text=plan_text, subsystems=subsystems, verdict=verdict,
            server_url=server_url, model=model, other_verdicts=other_verdicts,
            timeout=timeout,
        )

    def _run_obs():
        return _llm_judge_v3_observations_audio(
            observations=observations, target_audio_path=target_audio,
            server_url=server_url, model=model,
        )

    nres_by_idx: dict[int, dict | None] = {}
    hres_by_idx: dict[int, dict | None] = {}
    verdict_res: dict | None = None
    obs_res: dict | None = None
    with _TPE(max_workers=max_workers) as ex:
        narr_futs = []
        hall_futs = []
        for i, ((subsystem, narration), label) in enumerate(zip(narrations, batch_labels)):
            param_names = list(label.get("param_names", []))
            other_nars = list(other_by_sub.get(subsystem, []))[:3]
            narr_futs.append(ex.submit(_run_narr, i, subsystem, narration, param_names, other_nars))
            hall_futs.append(ex.submit(_run_hall, i, subsystem, narration, param_names))
        verdict_fut = ex.submit(_run_verdict)
        obs_fut = ex.submit(_run_obs)
        for fut in narr_futs:
            i, nres = fut.result()
            nres_by_idx[i] = nres
        for fut in hall_futs:
            i, hres = fut.result()
            hres_by_idx[i] = hres
        verdict_res = verdict_fut.result()
        obs_res = obs_fut.result()

    per_batch: list[dict] = []
    for i, ((subsystem, _narration), _label) in enumerate(zip(narrations, batch_labels)):
        nres = nres_by_idx.get(i)
        hres = hres_by_idx.get(i)
        entry: dict[str, Any] = {"subsystem": subsystem}
        if nres is not None:
            plan_refs.append(nres["plan_reference"])
            param_specs.append(nres["parameter_specific"])
            templateness.append(nres["templateness"])
            entry.update(nres)
        else:
            entry.update({"plan_reference": None, "parameter_specific": None,
                          "templateness": None, "reasoning": "(narration judge unavailable)"})
        if hres is not None:
            no_halluc.append(hres["no_hallucination"])
            entry["no_hallucination"] = hres["no_hallucination"]
            entry["hallucination_reasoning"] = hres["reasoning"]
        else:
            entry["no_hallucination"] = None
            entry["hallucination_reasoning"] = "(hallucination judge unavailable)"
        per_batch.append(entry)

    def _mean(xs: list[float]) -> float | None:
        return round(sum(xs) / len(xs), 4) if xs else None

    return {
        "llm_narration_plan_ref": _mean(plan_refs),
        "llm_narration_param_specific": _mean(param_specs),
        "llm_narration_templateness": _mean(templateness),
        "llm_narration_no_hallucination": _mean(no_halluc),
        "llm_observations_audio_grounded": (
            round(obs_res["audio_grounded"], 4) if obs_res else None
        ),
        "llm_verdict_residual_grounded": (
            round(verdict_res["residual_grounded"], 4) if verdict_res else None
        ),
        "llm_verdict_novelty": (
            round(verdict_res["novelty"], 4) if verdict_res else None
        ),
        "llm_per_batch": per_batch,
        "llm_verdict_reasoning": (verdict_res or {}).get("reasoning", ""),
        "llm_observations_reasoning": (obs_res or {}).get("reasoning", ""),
    }


# ---------------------------------------------------------------------------
# Dispatcher and summary
# ---------------------------------------------------------------------------

def score_main_v3_record(
    record: dict[str, Any],
    llm_judge_server: str | None = None,
    llm_judge_model: str = "Qwen/Qwen3-Omni-30B-A3B-Instruct",
    other_verdicts: list[str] | None = None,
    other_narrations_by_subsystem: dict[str, list[str]] | None = None,
) -> dict[str, Any]:
    """Score a v3 pipeline main record.

    v3 records use ``meta.batch_labels`` instead of ``step_labels`` and have
    a DIAGNOSIS → subsystem-batch → CORRECTION → VERDICT topology.

    Scoring dimensions:
      clap_net_improvement    (20%) — same semantics as v2, uses batch_labels
      diagnosis_subsystem_coverage (25%) — precision+recall of plan vs truth
      batch_param_alignment   (30%) — each batch's set_params matches subsystem label
      snake_case_clean        (2.5%) — no snake_case in assistant text
      format_consistent       (2.5%) — no **BOLD:** headers
      verdict_grounded        (10%) — FINAL ASSESSMENT names a residual subsystem
      mistake_recovery        (10%, conditional) — correction block for mistake samples
    """
    meta = record.get("meta", {})
    messages = record.get("messages", [])
    batch_labels: list[dict] = meta.get("batch_labels", [])

    # -- CLAP net improvement (from batch_labels[].clap_score_after_batch) --
    clap_scores = [bl["clap_score_after_batch"] for bl in batch_labels
                   if bl.get("clap_score_after_batch") is not None]
    if len(clap_scores) >= 2:
        delta = clap_scores[-1] - clap_scores[0]
        clap_net_improvement: float | None = round(min(1.0, max(0.0, delta / 0.30)), 4)
    else:
        clap_net_improvement = None

    # -- Diagnosis subsystem coverage (F1-style) --
    truth = set(meta.get("diagnosis_subsystems_truth") or [])
    mentioned = set(meta.get("diagnosis_subsystems_mentioned") or [])
    if truth:
        precision = len(mentioned & truth) / max(1, len(mentioned)) if mentioned else 0.0
        recall = len(mentioned & truth) / len(truth)
        f1 = 2 * precision * recall / max(1e-9, precision + recall)
        diagnosis_subsystem_coverage: float | None = round(f1, 4)
    else:
        diagnosis_subsystem_coverage = 1.0  # nothing to predict

    # -- Batch param alignment --
    # For each regular batch, check if all params in the set_params tool call belong
    # to the named subsystem. Uses _param_family from path_gen + v3's presentation mapping.
    from scripts.build_main_agent_sft_v3 import presentation_subsystem
    from maestro.synth.path_gen import _param_family

    aligned_batches = 0
    assessable_batches = 0
    for bl in batch_labels:
        if bl.get("is_correction"):
            continue
        assessable_batches += 1
        param_names: list[str] = bl.get("param_names", [])
        expected_sub = bl.get("subsystem", "")
        if not param_names:
            continue
        all_match = all(
            presentation_subsystem(_param_family(n)) == expected_sub
            for n in param_names
        )
        if all_match:
            aligned_batches += 1
    batch_param_alignment: float | None = (
        round(aligned_batches / assessable_batches, 4) if assessable_batches > 0 else None
    )

    # -- Snake case + format checks --
    assistant_turns = [m["content"] for m in messages if m.get("role") == "assistant"]
    snake_hits = sum(bool(_SNAKE_CASE_RE.search(t)) for t in assistant_turns)
    snake_case_clean = 1.0 - min(1.0, snake_hits / max(1, len(assistant_turns)))
    bold_hits = sum(bool(_BOLD_HEADER_RE.search(t)) for t in assistant_turns)
    format_consistent = 1.0 - min(1.0, bold_hits / max(1, len(assistant_turns)))

    # -- Verdict grounded: FINAL ASSESSMENT names a residual subsystem --
    verdict_turn = ""
    for m in reversed(messages):
        if m.get("role") == "assistant" and "FINAL ASSESSMENT" in m.get("content", ""):
            verdict_turn = m["content"]
            break
    residual_subs = list(((meta.get("batch_labels") or [{}])[-1] or {}).get("param_names", []))[:3]
    # Try simpler: check that verdict mentions any subsystem name from SUBSYSTEM_ORDER
    from scripts.build_main_agent_sft_v3 import _SUBSYSTEM_ALIASES
    verdict_lower = verdict_turn.lower()
    verdict_mentions_sub = any(
        any(re.search(rf"\b{re.escape(alias)}\b", verdict_lower)
            for alias in aliases)
        for label, aliases in _SUBSYSTEM_ALIASES.items()
    )
    verdict_grounded: float = 1.0 if verdict_mentions_sub else 0.0

    # -- Mistake recovery (conditional) --
    mistake_recovery: float | None = None
    if meta.get("injected_mistake") is not None:
        mistake_recovery = 1.0 if meta.get("mistake_caught") else 0.0

    # -- LLM judge (optional): narration grounding + verdict substantiveness +
    #    audio-grounded observations + param-hallucination binary check --
    llm_scores: dict[str, Any] = {}
    if llm_judge_server:
        llm_scores = llm_judge_v3_record(
            record, llm_judge_server, llm_judge_model,
            other_verdicts=other_verdicts,
            other_narrations_by_subsystem=other_narrations_by_subsystem,
        )

    no_gt_leak = _gt_leak_score(assistant_turns)

    # -- Overall weights (structural 50%, LLM-judge 50% when enabled) --
    weights: dict[str, float] = {
        "batch_param_alignment": 0.15,
        "diagnosis_subsystem_coverage": 0.10,
        "clap_net_improvement": 0.15,
        "verdict_grounded": 0.05,
        "no_gt_leak": 0.05,
        "snake_case_clean": 0.025,
        "format_consistent": 0.025,
        # LLM judge dimensions — enabled when llm_judge_server is set.
        "llm_narration_plan_ref": 0.05,
        "llm_narration_param_specific": 0.05,
        "llm_narration_templateness": 0.05,
        "llm_narration_no_hallucination": 0.10,
        "llm_observations_audio_grounded": 0.10,
        "llm_verdict_residual_grounded": 0.05,
        "llm_verdict_novelty": 0.05,
    }
    if mistake_recovery is not None:
        weights["mistake_recovery"] = 0.05

    raw: dict[str, Any] = {
        "clap_net_improvement": clap_net_improvement,
        "diagnosis_subsystem_coverage": diagnosis_subsystem_coverage,
        "batch_param_alignment": batch_param_alignment,
        "snake_case_clean": snake_case_clean,
        "format_consistent": format_consistent,
        "verdict_grounded": verdict_grounded,
        "no_gt_leak": no_gt_leak,
        "mistake_recovery": mistake_recovery,
        "llm_narration_plan_ref": llm_scores.get("llm_narration_plan_ref"),
        "llm_narration_param_specific": llm_scores.get("llm_narration_param_specific"),
        "llm_narration_templateness": llm_scores.get("llm_narration_templateness"),
        "llm_narration_no_hallucination": llm_scores.get("llm_narration_no_hallucination"),
        "llm_observations_audio_grounded": llm_scores.get("llm_observations_audio_grounded"),
        "llm_verdict_residual_grounded": llm_scores.get("llm_verdict_residual_grounded"),
        "llm_verdict_novelty": llm_scores.get("llm_verdict_novelty"),
    }
    weighted_sum = 0.0
    weight_sum = 0.0
    for k, w in weights.items():
        v = raw.get(k)
        if v is not None:
            weighted_sum += v * w
            weight_sum += w
    overall = round(weighted_sum / weight_sum, 4) if weight_sum > 0 else 0.0

    out = {**raw, "overall": overall}
    if llm_scores:
        out["llm_per_batch"] = llm_scores.get("llm_per_batch", [])
        out["llm_verdict_reasoning"] = llm_scores.get("llm_verdict_reasoning", "")
        out["llm_observations_reasoning"] = llm_scores.get("llm_observations_reasoning", "")
    return out


# ---------------------------------------------------------------------------
# Search v2 scoring — narrow, structural + correctness
# ---------------------------------------------------------------------------

def _score_candidate_audio_grounding(
    triples: list[tuple[str, str, str]],
    llm_judge_server: str | None,
    llm_judge_model: str,
    max_candidates: int | None = None,
    sample_rate: float = 1.0,
    sample_seed: int | None = None,
) -> tuple[float | None, list[dict]]:
    """Run the per-candidate audio-grounding check. Returns (mean_score, details).

    Returns (None, []) if llm_judge_server is not set or no triples can be
    checked. Catches multi-audio position-confusion hallucinations from the
    build-time batch Omni call.

    - `sample_rate` in (0, 1]: fraction of triples to evaluate (rounded up,
      at least 1). Random subset by default; set `sample_seed` for reproducible
      sampling. 1.0 (default) checks every candidate.
    - `max_candidates`: optional hard cap applied after sampling.
    """
    if not llm_judge_server or not triples:
        return None, []
    import math
    import random
    rng = random.Random(sample_seed) if sample_seed is not None else random
    target_n = len(triples)
    if 0.0 < sample_rate < 1.0:
        target_n = max(1, math.ceil(len(triples) * sample_rate))
    if max_candidates is not None and max_candidates > 0:
        target_n = min(target_n, max_candidates)
    if target_n < len(triples):
        triples = rng.sample(triples, target_n)
    details: list[dict] = []
    scores: list[float] = []
    # Parallelize the per-candidate Omni calls — vLLM's continuous batching
    # serves N concurrent requests far faster than N sequential ones. With
    # `--workers W` records in flight and `_per_record_concurrency` checks
    # per record, the in-flight Omni-request count is roughly W * concurrency.
    _per_record_concurrency = min(16, len(triples))
    if _per_record_concurrency <= 1:
        for name, desc, path in triples:
            result = _llm_judge_candidate_audio(
                description=desc, audio_path=path,
                server_url=llm_judge_server, model=llm_judge_model,
            )
            if result is None:
                continue
            scores.append(result["audio_grounded"])
            details.append({"name": name, "audio_grounded": result["audio_grounded"], "reasoning": result["reasoning"]})
        if not scores:
            return None, details
        return round(sum(scores) / len(scores), 4), details

    from concurrent.futures import ThreadPoolExecutor as _TPE
    futures_in_order: list = []
    with _TPE(max_workers=_per_record_concurrency) as ex:
        for name, desc, path in triples:
            futures_in_order.append((name, ex.submit(
                _llm_judge_candidate_audio,
                desc, path, llm_judge_server, llm_judge_model,
            )))
        for name, fut in futures_in_order:
            result = fut.result()
            if result is None:
                continue
            scores.append(result["audio_grounded"])
            details.append({"name": name, "audio_grounded": result["audio_grounded"], "reasoning": result["reasoning"]})
    if not scores:
        return None, details
    return round(sum(scores) / len(scores), 4), details


def score_search_v2_record(
    record: dict[str, Any],
    llm_judge_server: str | None = None,
    llm_judge_model: str = "Qwen/Qwen3-Omni-30B-A3B-Instruct",
    max_audio_grounding_candidates: int | None = None,
    audio_grounding_sample_rate: float = 1.0,
) -> dict[str, Any]:
    """Score a search_v2 record.

    The search agent evaluates one slice of the wavetable library across 6
    batches, accumulates a shortlist via CLAP-grounded labels, then writes
    the shortlist to an output file via an explicit bash tool_call.

    Dimensions:
      gt_recovery                    (35% conditional) — fraction of GTs in the shard that made it onto the shortlist. Only assessable when gt_in_shard is non-empty.
      shortlist_file_written         (20%)             — last bash tool_call writes the shortlist JSON, tool_response confirms.
      llm_candidates_audio_grounded  (15% conditional) — mean single-audio Omni grounding check across all (candidate, description) pairs. Catches batch-call audio-position confusion. Requires --llm-judge-server.
      has_render_probes              (10%)             — ≥1 bash tool_call renders wavetable probes (inline chunk manipulation or legacy render_probes.py).
      shortlist_nonempty             (10%)             — final shortlist has ≥1 name.
      closing_assistant              (5%)              — last message is assistant.
      snake_case_clean               (2.5%)            — no snake_case param leak.
      format_consistent              (2.5%)            — no **BOLD:** headers.
    """
    messages = record.get("messages", [])
    meta = record.get("meta", {})
    assistant_turns = [m.get("content", "") or "" for m in messages if m.get("role") == "assistant"]

    # GT recovery
    gt_in_shard: list[str] = list(meta.get("gt_in_shard") or [])
    gt_on_shortlist: list[str] = list(meta.get("gt_on_shortlist") or [])
    if gt_in_shard:
        gt_recovery: float | None = round(len(gt_on_shortlist) / len(gt_in_shard), 4)
    else:
        gt_recovery = None  # not assessable

    # Shortlist file write: the LAST bash tool_call should write a *_search_*.json
    # and its matching tool_response should return {"status": "ok", "file": ...}.
    shortlist_file_written: float = 0.0
    last_bash_idx = -1
    for i, m in enumerate(messages):
        if m.get("role") == "tool_call":
            try:
                tc = json.loads(m.get("content", ""))
                if tc.get("name") in ("bash", "Bash"):
                    last_bash_idx = i
            except Exception:
                pass
    if last_bash_idx >= 0:
        try:
            tc = json.loads(messages[last_bash_idx]["content"])
            cmd = tc.get("arguments", {}).get("command", "")
            if "_search_" in cmd and ".json" in cmd:
                # Check next tool_response is ok
                nxt = messages[last_bash_idx + 1] if last_bash_idx + 1 < len(messages) else None
                if nxt and nxt.get("role") == "tool_response":
                    try:
                        resp = json.loads(nxt.get("content", ""))
                        if resp.get("status") == "ok" and resp.get("file"):
                            shortlist_file_written = 1.0
                    except Exception:
                        pass
        except Exception:
            pass

    # Closing assistant
    closing_assistant: float = 1.0 if messages and messages[-1].get("role") == "assistant" else 0.0

    # Has render_probes invocation (inline chunk manipulation or legacy script)
    has_render_probes: float = 0.0
    for m in messages:
        if m.get("role") != "tool_call":
            continue
        try:
            tc = json.loads(m.get("content", ""))
            cmd = tc.get("arguments", {}).get("command", "") if tc.get("name") in ("bash", "Bash") else ""
            if "skills/vital/scripts/render_probes.py" in cmd or "TrackFX_SetNamedConfigParm" in cmd:
                has_render_probes = 1.0
                break
        except Exception:
            pass

    # Shortlist nonempty
    final_shortlist: list[str] = list(meta.get("final_shortlist") or [])
    shortlist_nonempty: float = 1.0 if final_shortlist else 0.0

    # Snake case / format
    snake_hits = sum(bool(_SNAKE_CASE_RE.search(t)) for t in assistant_turns)
    snake_case_clean = 1.0 - min(1.0, snake_hits / max(1, len(assistant_turns)))
    bold_hits = sum(bool(_BOLD_HEADER_RE.search(t)) for t in assistant_turns)
    format_consistent = 1.0 - min(1.0, bold_hits / max(1, len(assistant_turns)))

    # Per-candidate audio grounding (optional, requires LLM judge server).
    triples = _extract_search_candidate_triples(record)
    audio_grounded_mean, audio_grounded_details = _score_candidate_audio_grounding(
        triples, llm_judge_server, llm_judge_model, max_audio_grounding_candidates,
        sample_rate=audio_grounding_sample_rate,
    )

    no_gt_leak = _gt_leak_score(assistant_turns)

    weights: dict[str, float] = {
        "gt_recovery": 0.30,
        "shortlist_file_written": 0.20,
        "llm_candidates_audio_grounded": 0.15,
        "has_render_probes": 0.10,
        "shortlist_nonempty": 0.10,
        "no_gt_leak": 0.10,
        "closing_assistant": 0.025,
        "snake_case_clean": 0.0125,
        "format_consistent": 0.0125,
    }
    raw: dict[str, Any] = {
        "gt_recovery": gt_recovery,
        "shortlist_file_written": shortlist_file_written,
        "llm_candidates_audio_grounded": audio_grounded_mean,
        "has_render_probes": has_render_probes,
        "shortlist_nonempty": shortlist_nonempty,
        "no_gt_leak": no_gt_leak,
        "closing_assistant": closing_assistant,
        "snake_case_clean": snake_case_clean,
        "format_consistent": format_consistent,
        "llm_candidates_audio_grounded_details": audio_grounded_details,
    }
    weighted_sum = 0.0
    weight_sum = 0.0
    for k, w in weights.items():
        v = raw.get(k)
        if v is not None:
            weighted_sum += v * w
            weight_sum += w
    overall = round(weighted_sum / weight_sum, 4) if weight_sum > 0 else 0.0
    return {**raw, "overall": overall}


# ---------------------------------------------------------------------------
# Judge v3 scoring — narrow, structural + correctness
# ---------------------------------------------------------------------------

def score_judge_v3_record(
    record: dict[str, Any],
    llm_judge_server: str | None = None,
    llm_judge_model: str = "Qwen/Qwen3-Omni-30B-A3B-Instruct",
    max_audio_grounding_candidates: int | None = None,
    audio_grounding_sample_rate: float = 1.0,
) -> dict[str, Any]:
    """Score a v3 judge record.

    Verdict-aware: judge emits verdict ∈ {"good", "no_match"}. Tuple-related
    dimensions only apply when verdict == "good"; missing-character + no_match
    narration dimensions apply only when verdict == "no_match". Records without
    a verdict field default to "good" for back-compat with v18/v19.

    Dimensions:
      verdict_correct                (25%)            — judge's emitted verdict matches the oracle (gt ⊆ pool).
      output_file_written            (15%)            — final bash writes judge JSON; tool_response ok.
      output_schema_valid            (5%)             — embedded write_cmd JSON has all required keys.
      judge_correct                  (15% cond:good)  — meta.judge_correct: tuple matches GTs in pool.
      tuple_size_correct             (5%  cond:good)  — len(selected_tuple) == n_osc_slots.
      tuple_names_in_pool            (5%  cond:good)  — every selected name is from the pool.
      missing_character_nonempty     (5%  cond:no_match) — meta.missing_character has ≥3 words.
      no_match_narration_present     (5%  cond:no_match) — assistant text contains no_match/re-search hints.
      llm_candidates_audio_grounded  (10% conditional) — single-audio Omni grounding (requires --llm-judge-server).
      pool_candidates_discussed      (5%)             — fraction of pool names mentioned in deliberation.
      has_render_probes              (2.5%)           — ≥1 bash tool_call renders wavetable probes (inline chunk manipulation or legacy render_probes.py).
      closing_assistant              (2.5%)           — last message is assistant.
      format_consistent              (2.5%)           — no **BOLD:** headers.
      snake_case_clean               (2.5%)           — no snake_case in assistant text.
    """
    messages = record.get("messages", [])
    meta = record.get("meta", {})
    assistant_turns = [m.get("content", "") or "" for m in messages if m.get("role") == "assistant"]

    # Verdict resolution: explicit field wins; default to "good" for back-compat.
    verdict = meta.get("verdict") or "good"
    is_good = verdict == "good"
    is_no_match = verdict == "no_match"

    selected_tuple: list[str] = list(meta.get("selected_tuple") or [])
    pool: list[str] = list(meta.get("pool") or [])
    n_osc_slots: int = int(meta.get("n_osc_slots") or 0)
    gt_names: list[str] = list(meta.get("gt_wavetable_names") or [])
    missing_character = str(meta.get("missing_character") or "")

    # Verdict correctness: oracle = "good" iff every GT name appears in pool.
    expected_verdict = "good" if (gt_names and all(g in pool for g in gt_names)) else "no_match"
    if not gt_names:
        # No GT info available — can't grade verdict correctness.
        verdict_correct: float | None = None
    else:
        verdict_correct = 1.0 if verdict == expected_verdict else 0.0

    # Tuple-related dimensions (good only)
    if is_good:
        judge_correct: float | None = 1.0 if meta.get("judge_correct") else 0.0
        tuple_size_correct: float | None = 1.0 if len(selected_tuple) == n_osc_slots else 0.0
        tuple_names_in_pool: float | None = (
            1.0 if selected_tuple and all(n in pool for n in selected_tuple) else 0.0
        )
    else:
        judge_correct = None
        tuple_size_correct = None
        tuple_names_in_pool = None

    # No_match-specific dimensions
    if is_no_match:
        missing_character_nonempty: float | None = (
            1.0 if len(missing_character.split()) >= 3 else 0.0
        )
        _no_match_kw = ("no_match", "no-match", "re-search", "research", "doesn't contain", "does not contain")
        _hits = any(any(kw in (t or "").lower() for kw in _no_match_kw) for t in assistant_turns)
        no_match_narration_present: float | None = 1.0 if _hits else 0.0
    else:
        missing_character_nonempty = None
        no_match_narration_present = None

    # Output file written + schema valid
    output_file_written: float = 0.0
    output_schema_valid: float = 0.0
    last_bash_idx = -1
    for i, m in enumerate(messages):
        if m.get("role") == "tool_call":
            try:
                tc = json.loads(m.get("content", ""))
                if tc.get("name") in ("bash", "Bash"):
                    last_bash_idx = i
            except Exception:
                pass
    if last_bash_idx >= 0:
        try:
            tc = json.loads(messages[last_bash_idx]["content"])
            cmd = tc.get("arguments", {}).get("command", "")
            # Loosened heuristic: must reference n_osc_slots + reasoning. The "tuple"
            # check is omitted because tuple may be null on no_match records, but
            # the literal "tuple" key is still present so we keep that as a soft check.
            if "n_osc_slots" in cmd and "reasoning" in cmd and "tuple" in cmd:
                nxt = messages[last_bash_idx + 1] if last_bash_idx + 1 < len(messages) else None
                if nxt and nxt.get("role") == "tool_response":
                    try:
                        resp = json.loads(nxt.get("content", ""))
                        if resp.get("status") == "ok" and resp.get("file"):
                            output_file_written = 1.0
                    except Exception:
                        pass
            # Schema-valid: parse the embedded JSON literal in the heredoc and
            # verify the new keys are present. Lenient — for back-compat with
            # records lacking verdict/missing_character we only require the
            # legacy {tuple, n_osc_slots, reasoning} keys.
            required_legacy = {"tuple", "n_osc_slots", "reasoning"}
            required_new = required_legacy | {"verdict", "missing_character"}
            try:
                # Find the largest brace-balanced JSON-looking substring
                _braces = re.findall(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", cmd, re.DOTALL)
                _payload_keys: set[str] = set()
                for blob in _braces:
                    try:
                        _parsed = json.loads(blob)
                        if isinstance(_parsed, dict) and required_legacy.issubset(_parsed.keys()):
                            _payload_keys = set(_parsed.keys())
                            break
                    except Exception:
                        continue
                target_keys = required_new if (verdict and verdict in ("good", "no_match")) else required_legacy
                if target_keys.issubset(_payload_keys):
                    output_schema_valid = 1.0
            except Exception:
                pass
        except Exception:
            pass

    # Pool candidates discussed in deliberation
    if pool and assistant_turns:
        deliberation_text = " ".join(assistant_turns).lower()
        mentioned = sum(1 for n in pool if n.lower() in deliberation_text)
        pool_candidates_discussed: float | None = round(mentioned / len(pool), 4)
    else:
        pool_candidates_discussed = None

    # Has render_probes call (inline chunk manipulation or legacy script)
    has_render_probes: float = 0.0
    for m in messages:
        if m.get("role") != "tool_call":
            continue
        try:
            tc = json.loads(m.get("content", ""))
            cmd = tc.get("arguments", {}).get("command", "") if tc.get("name") in ("bash", "Bash") else ""
            if "skills/vital/scripts/render_probes.py" in cmd or "TrackFX_SetNamedConfigParm" in cmd:
                has_render_probes = 1.0
                break
        except Exception:
            pass

    closing_assistant: float = 1.0 if messages and messages[-1].get("role") == "assistant" else 0.0

    snake_hits = sum(bool(_SNAKE_CASE_RE.search(t)) for t in assistant_turns)
    snake_case_clean = 1.0 - min(1.0, snake_hits / max(1, len(assistant_turns)))
    bold_hits = sum(bool(_BOLD_HEADER_RE.search(t)) for t in assistant_turns)
    format_consistent = 1.0 - min(1.0, bold_hits / max(1, len(assistant_turns)))

    triples = _extract_judge_candidate_triples(record)
    audio_grounded_mean, audio_grounded_details = _score_candidate_audio_grounding(
        triples, llm_judge_server, llm_judge_model, max_audio_grounding_candidates,
        sample_rate=audio_grounding_sample_rate,
    )

    weights: dict[str, float] = {
        "verdict_correct": 0.25,
        "output_file_written": 0.15,
        "output_schema_valid": 0.05,
        "judge_correct": 0.15,
        "tuple_size_correct": 0.05,
        "tuple_names_in_pool": 0.05,
        "missing_character_nonempty": 0.05,
        "no_match_narration_present": 0.05,
        "llm_candidates_audio_grounded": 0.10,
        "pool_candidates_discussed": 0.05,
        "has_render_probes": 0.025,
        "closing_assistant": 0.025,
        "format_consistent": 0.025,
        "snake_case_clean": 0.025,
    }
    raw: dict[str, Any] = {
        "verdict": verdict,
        "verdict_correct": verdict_correct,
        "output_file_written": output_file_written,
        "output_schema_valid": output_schema_valid,
        "judge_correct": judge_correct,
        "tuple_size_correct": tuple_size_correct,
        "tuple_names_in_pool": tuple_names_in_pool,
        "missing_character_nonempty": missing_character_nonempty,
        "no_match_narration_present": no_match_narration_present,
        "llm_candidates_audio_grounded": audio_grounded_mean,
        "pool_candidates_discussed": pool_candidates_discussed,
        "has_render_probes": has_render_probes,
        "closing_assistant": closing_assistant,
        "format_consistent": format_consistent,
        "snake_case_clean": snake_case_clean,
        "llm_candidates_audio_grounded_details": audio_grounded_details,
    }
    weighted_sum = 0.0
    weight_sum = 0.0
    for k, w in weights.items():
        v = raw.get(k)
        if v is not None:
            weighted_sum += v * w
            weight_sum += w
    overall = round(weighted_sum / weight_sum, 4) if weight_sum > 0 else 0.0
    return {**raw, "overall": overall}


# ---------------------------------------------------------------------------
# Melody transcription v3 scoring — structural + oracle correctness
# ---------------------------------------------------------------------------

def score_transcription_record(record: dict[str, Any]) -> dict[str, Any]:
    """Score a melody_transcription record.

    The transcription subagent listens to the target audio, writes Python
    (reapy) code that inserts MIDI notes on a REAPER track, and saves the
    note list as a JSON handoff file.

    Dimensions:
      has_reapy_midi_insert  (20%) — ≥1 bash tool_call contains MIDI_InsertNote
                                    (or RPR_MIDI_InsertNote).
      output_file_written    (25%) — final bash writes transcription.json with
                                    notes/n_notes/duration_s + ok tool_response.
      note_count_match       (20%) — n notes in payload matches meta.n_notes
                                    (oracle count).
      pitch_coverage         (10%) — fraction of oracle pitches that appear in
                                    deliberation or insert command.
      has_render_listen      (10%) — first user message has <audio> (subagent
                                    actually received the target for listening).
      closing_assistant      (5%)  — last message is an assistant turn.
      snake_case_clean       (5%)  — no snake_case in assistant prose.
      format_consistent      (5%)  — no **BOLD:** headers.
    """
    messages = record.get("messages", [])
    meta = record.get("meta", {})
    assistant_turns = [m.get("content", "") or "" for m in messages if m.get("role") == "assistant"]

    oracle_notes: list[dict] = list(meta.get("notes") or [])
    oracle_n: int = int(meta.get("n_notes") or 0)
    # Pitch may be a MIDI int (legacy format) or a note-name string ("C2", "F#3", etc.).
    oracle_pitches: set[int] = set()
    for n in oracle_notes:
        p = _pitch_to_int(n.get("pitch"))
        if p is not None:
            oracle_pitches.add(p)

    # has_reapy_midi_insert: walk bash tool_calls for MIDI_InsertNote
    has_reapy_midi_insert: float = 0.0
    insert_cmd_text = ""
    for m in messages:
        if m.get("role") != "tool_call":
            continue
        try:
            tc = json.loads(m.get("content", ""))
            if tc.get("name") not in ("bash", "Bash"):
                continue
            cmd = tc.get("arguments", {}).get("command", "") or ""
            if "MIDI_InsertNote" in cmd:
                has_reapy_midi_insert = 1.0
                insert_cmd_text = cmd
                break
        except Exception:
            pass

    # output_file_written: final bash tool_call writes transcription JSON with
    # the expected shape + matching ok tool_response.
    output_file_written: float = 0.0
    note_count_match: float = 0.0
    last_bash_idx = -1
    for i, m in enumerate(messages):
        if m.get("role") == "tool_call":
            try:
                tc = json.loads(m.get("content", ""))
                if tc.get("name") in ("bash", "Bash"):
                    last_bash_idx = i
            except Exception:
                pass
    if last_bash_idx >= 0:
        try:
            tc = json.loads(messages[last_bash_idx]["content"])
            cmd = tc.get("arguments", {}).get("command", "") or ""
            # require n_notes + notes + duration_s schema in the heredoc
            has_schema = all(tok in cmd for tok in ('"n_notes"', '"notes"', '"duration_s"'))
            if has_schema and ".json" in cmd:
                nxt = messages[last_bash_idx + 1] if last_bash_idx + 1 < len(messages) else None
                if nxt and nxt.get("role") == "tool_response":
                    try:
                        resp = json.loads(nxt.get("content", ""))
                        if resp.get("status") == "ok" and resp.get("file"):
                            output_file_written = 1.0
                    except Exception:
                        pass
                # note count match: parse n_notes from the heredoc payload string
                m_count = re.search(r'"n_notes"\s*:\s*(\d+)', cmd)
                if m_count:
                    emitted_n = int(m_count.group(1))
                    if oracle_n > 0 and emitted_n == oracle_n:
                        note_count_match = 1.0
        except Exception:
            pass

    # pitch_coverage: how many oracle pitches appear in the assistant/insert_cmd text
    combined_text = " ".join(assistant_turns) + " " + insert_cmd_text
    if oracle_pitches:
        seen = 0
        for p in oracle_pitches:
            # Match as integer token (word-bounded) OR note name
            if re.search(rf"\b{p}\b", combined_text):
                seen += 1
                continue
            name = _pitch_name_for_grader(p)
            if name and name in combined_text:
                seen += 1
        pitch_coverage: float | None = round(seen / len(oracle_pitches), 4)
    else:
        pitch_coverage = None

    # has_render_listen: first user message carries an <audio> tag
    has_render_listen: float = 0.0
    if messages and messages[0].get("role") == "user":
        if "<audio>" in str(messages[0].get("content", "")):
            has_render_listen = 1.0

    closing_assistant: float = 1.0 if messages and messages[-1].get("role") == "assistant" else 0.0

    snake_hits = sum(bool(_SNAKE_CASE_RE.search(t)) for t in assistant_turns)
    snake_case_clean = 1.0 - min(1.0, snake_hits / max(1, len(assistant_turns)))
    bold_hits = sum(bool(_BOLD_HEADER_RE.search(t)) for t in assistant_turns)
    format_consistent = 1.0 - min(1.0, bold_hits / max(1, len(assistant_turns)))

    weights: dict[str, float] = {
        "has_reapy_midi_insert": 0.20,
        "output_file_written": 0.25,
        "note_count_match": 0.20,
        "pitch_coverage": 0.10,
        "has_render_listen": 0.10,
        "closing_assistant": 0.05,
        "snake_case_clean": 0.05,
        "format_consistent": 0.05,
    }
    raw: dict[str, Any] = {
        "has_reapy_midi_insert": has_reapy_midi_insert,
        "output_file_written": output_file_written,
        "note_count_match": note_count_match,
        "pitch_coverage": pitch_coverage,
        "has_render_listen": has_render_listen,
        "closing_assistant": closing_assistant,
        "snake_case_clean": snake_case_clean,
        "format_consistent": format_consistent,
    }
    weighted_sum = 0.0
    weight_sum = 0.0
    for k, w in weights.items():
        v = raw.get(k)
        if v is not None:
            weighted_sum += v * w
            weight_sum += w
    overall = round(weighted_sum / weight_sum, 4) if weight_sum > 0 else 0.0
    return {**raw, "overall": overall}


_PITCH_NAMES_FOR_GRADER = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
_PITCH_CLASS_TO_INT = {
    **{n: i for i, n in enumerate(_PITCH_NAMES_FOR_GRADER)},
    "Db": 1, "Eb": 3, "Gb": 6, "Ab": 8, "Bb": 10,
}


def _pitch_name_for_grader(pitch: int) -> str:
    octave = (pitch // 12) - 1
    return f"{_PITCH_NAMES_FOR_GRADER[pitch % 12]}{octave}"


def _pitch_to_int(value) -> int | None:
    """MIDI int or float → int. Strings are accepted for back-compat only
    (an earlier builder emitted note-name strings like 'C2'; the current
    builder uses MIDI ints directly)."""
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if not isinstance(value, str):
        return None
    # Split "<pitch-class><octave>" — the octave may be negative.
    i = len(value) - 1
    while i > 0 and value[i] not in "-0123456789":
        i -= 1
    pc = value[:i]
    try:
        octave = int(value[i:])
    except ValueError:
        return None
    cls = _PITCH_CLASS_TO_INT.get(pc)
    if cls is None:
        return None
    return cls + (octave + 1) * 12


def score_record(
    record: dict[str, Any],
    llm_judge_server: str | None = None,
    llm_judge_model: str = "Qwen/Qwen3-Omni-30B-A3B-Instruct",
    live_exec_check: bool = False,
    live_exec_timeout_sec: float = 30.0,
    live_exec_max_calls: int | None = None,
    other_verdicts: list[str] | None = None,
    other_narrations_by_subsystem: dict[str, list[str]] | None = None,
    audio_grounding_sample_rate: float = 1.0,
) -> dict[str, Any]:
    task = record.get("task_type", "")
    meta = record.get("meta", {})
    # Branch: v3 pipeline records use a different scoring path.
    if task == "main" and meta.get("pipeline_version") == "v3":
        scores = score_main_v3_record(
            record, llm_judge_server=llm_judge_server, llm_judge_model=llm_judge_model,
            other_verdicts=other_verdicts,
            other_narrations_by_subsystem=other_narrations_by_subsystem,
        )
    elif task == "main":
        scores = score_main_record(record, llm_judge_server=llm_judge_server, llm_judge_model=llm_judge_model)
    elif task == "search_v2":
        scores = score_search_v2_record(
            record, llm_judge_server=llm_judge_server, llm_judge_model=llm_judge_model,
            audio_grounding_sample_rate=audio_grounding_sample_rate,
        )
    elif task == "judge" and meta.get("pipeline_version") == "v3_judge":
        scores = score_judge_v3_record(
            record, llm_judge_server=llm_judge_server, llm_judge_model=llm_judge_model,
            audio_grounding_sample_rate=audio_grounding_sample_rate,
        )
    elif task == "melody_transcription":
        scores = score_transcription_record(record)
    elif task == "search":
        scores = score_search_record(record)  # legacy v1
    elif task == "judge":
        scores = score_judge_record(record)   # legacy v1
    else:
        scores = {"overall": None, "error": f"unknown task_type: {task!r}"}

    # Optional live runtime validation: execute bash tool calls and check that
    # runtime outputs/effects align with emitted tool_response payloads.
    # Hard-fail filter: any non-passing call forces overall=0 so the record is
    # filtered out at `--min-score` thresholds. The record stays in the scored
    # JSONL with the detailed errors intact for postmortem.
    if live_exec_check:
        exec_results = run_live_execution_checks_for_record(
            record,
            timeout_sec=live_exec_timeout_sec,
            max_calls=live_exec_max_calls,
        )
        scores.update(exec_results)
        exec_fidelity = exec_results.get("execution_fidelity")
        if exec_fidelity is not None and float(exec_fidelity) < 1.0 and scores.get("overall") is not None:
            scores["overall"] = 0.0

    return scores


def grade_file(
    input_path: Path,
    output_path: Path,
    min_score: float | None = None,
    verbose: bool = False,
    llm_judge_server: str | None = "http://localhost:8000",
    llm_judge_model: str = "Qwen/Qwen3-Omni-30B-A3B-Instruct",
    workers: int = 1,
    live_exec_check: bool = False,
    live_exec_timeout_sec: float = 30.0,
    live_exec_max_calls: int | None = None,
    audio_grounding_sample_rate: float = 1.0,
) -> dict[str, Any]:
    """Grade all records in *input_path*, write scored output to *output_path*.

    LLM-as-judge is **on by default** using the local inference server at
    ``http://localhost:8000``.  Pass ``llm_judge_server=None`` to disable it and
    fall back to heuristic bigram matching only.

    ``workers`` controls how many records are scored concurrently. Each record's
    scoring is independent HTTP calls, so threading scales well. Defaults to 1
    (serial). Output order always matches input order regardless of workers.

    Returns a summary dict with per-task-type stats.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    if live_exec_check:
        _check_live_exec_environment()
        if workers > 1:
            # Live REAPER execution is stateful; serialise checks for determinism.
            workers = 1

    rows: list[dict] = []
    with open(input_path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    # Pre-collect verdicts AND narrations-by-subsystem from all v3 records so
    # cross-sample template-detection has reference material. Each record is
    # passed 3 verdicts + 3 same-subsystem narrations from OTHER samples
    # (deterministic pick via record index + fixed seed).
    verdicts_by_idx: dict[int, str] = {}
    narrations_by_idx: dict[int, list[tuple[str, str]]] = {}
    for i, record in enumerate(rows):
        meta = record.get("meta", {})
        if record.get("task_type") == "main" and meta.get("pipeline_version") == "v3":
            extracted = _extract_v3_plan_and_narrations(record)
            if extracted["verdict"]:
                verdicts_by_idx[i] = extracted["verdict"]
            narrations_by_idx[i] = extracted["narrations"]

    def _other_verdicts_for(idx: int, k: int = 3) -> list[str]:
        pool = [v for j, v in verdicts_by_idx.items() if j != idx]
        if not pool:
            return []
        rng = _random.Random(1337 + idx)
        rng.shuffle(pool)
        return pool[:k]

    def _other_narrations_by_subsystem_for(idx: int, k: int = 3) -> dict[str, list[str]]:
        """For the record at idx, return {subsystem: [narrations from OTHER samples]}."""
        pool: dict[str, list[str]] = {}
        for j, nars in narrations_by_idx.items():
            if j == idx:
                continue
            for subsystem, narration in nars:
                pool.setdefault(subsystem, []).append(narration)
        rng = _random.Random(2024 + idx)
        result: dict[str, list[str]] = {}
        for subsystem, lst in pool.items():
            rng.shuffle(lst)
            result[subsystem] = lst[:k]
        return result

    # Score all records, in parallel if workers > 1. Results keyed by input index
    # so output order is always deterministic regardless of completion order.
    scores_by_idx: dict[int, dict] = {}
    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futs = {
                pool.submit(score_record, record,
                            llm_judge_server=llm_judge_server,
                            llm_judge_model=llm_judge_model,
                            live_exec_check=live_exec_check,
                            live_exec_timeout_sec=live_exec_timeout_sec,
                            live_exec_max_calls=live_exec_max_calls,
                            other_verdicts=_other_verdicts_for(i),
                            other_narrations_by_subsystem=_other_narrations_by_subsystem_for(i),
                            audio_grounding_sample_rate=audio_grounding_sample_rate): i
                for i, record in enumerate(rows)
            }
            for fut in as_completed(futs):
                i = futs[fut]
                try:
                    scores_by_idx[i] = fut.result()
                except Exception as exc:
                    print(f"WARNING: record {i} ({rows[i].get('id','?')}) failed scoring: {exc}")
                    scores_by_idx[i] = {}
    else:
        for i, record in enumerate(rows):
            if live_exec_check:
                _reset_reaper_project(record)
            scores_by_idx[i] = score_record(
                record,
                llm_judge_server=llm_judge_server,
                llm_judge_model=llm_judge_model,
                live_exec_check=live_exec_check,
                live_exec_timeout_sec=live_exec_timeout_sec,
                live_exec_max_calls=live_exec_max_calls,
                other_verdicts=_other_verdicts_for(i),
                other_narrations_by_subsystem=_other_narrations_by_subsystem_for(i),
                audio_grounding_sample_rate=audio_grounding_sample_rate,
            )

    # Cross-sample narration diversity: Jaccard on trigram-word-sets between
    # same-subsystem narrations across v3 records. Low diversity = same boilerplate
    # phrases recurring across samples, which structural metrics can't catch.
    v3_indices = [i for i, r in enumerate(rows)
                  if r.get("task_type") == "main" and r.get("meta", {}).get("pipeline_version") == "v3"]
    diversity_by_idx: dict[int, float | None] = {}
    if len(v3_indices) >= 2:
        narrations_by_idx: dict[int, list[tuple[str, str]]] = {}
        for i in v3_indices:
            narrations_by_idx[i] = _extract_v3_plan_and_narrations(rows[i])["narrations"]

        def _trigrams(text: str) -> set[tuple[str, ...]]:
            words = re.findall(r"[a-z]+", text.lower())
            return {tuple(words[j:j+3]) for j in range(len(words) - 2)}

        for i in v3_indices:
            per_batch_novelty: list[float] = []
            for subsystem, narration in narrations_by_idx[i]:
                this_tg = _trigrams(narration)
                if not this_tg:
                    continue
                overlaps: list[float] = []
                for j in v3_indices:
                    if j == i:
                        continue
                    for other_sub, other_nar in narrations_by_idx[j]:
                        if other_sub != subsystem:
                            continue
                        other_tg = _trigrams(other_nar)
                        if not other_tg:
                            continue
                        jacc = len(this_tg & other_tg) / len(this_tg | other_tg)
                        overlaps.append(jacc)
                if overlaps:
                    per_batch_novelty.append(1.0 - (sum(overlaps) / len(overlaps)))
            diversity_by_idx[i] = (
                round(sum(per_batch_novelty) / len(per_batch_novelty), 4)
                if per_batch_novelty else None
            )

    scored: list[dict] = []
    kept = 0
    all_scores: list[tuple[str, float]] = []  # (task, overall) for summary stats
    for i, record in enumerate(rows):
        scores = scores_by_idx[i]
        if i in diversity_by_idx:
            scores["cross_sample_narration_novelty"] = diversity_by_idx[i]
        record_out = dict(record)
        record_out["quality_scores"] = scores
        overall = scores.get("overall")
        passes = (min_score is None) or (overall is not None and overall >= min_score)
        if passes:
            scored.append(record_out)
            kept += 1
        if overall is not None:
            all_scores.append((record.get("task_type", "unknown"), overall))
        if verbose:
            task = record.get("task_type", "?")
            ov_str = f"{overall:.4f}" if overall is not None else "N/A"
            status = "KEPT" if passes else "FILTERED"
            print(f"  [{task}] {record.get('id','?')} overall={ov_str} {status}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for r in scored:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Summary stats — built from scores already computed above (no second pass)
    by_task: dict[str, list[float]] = {}
    for task, ov in all_scores:
        by_task.setdefault(task, []).append(ov)

    summary: dict[str, Any] = {
        "input": str(input_path),
        "output": str(output_path),
        "total": len(rows),
        "kept": kept,
        "filtered": len(rows) - kept,
        "min_score_threshold": min_score,
        "by_task": {
            task: {
                "n": len(scores),
                "mean": round(sum(scores) / len(scores), 4),
                "min": round(min(scores), 4),
                "max": round(max(scores), 4),
            }
            for task, scores in by_task.items()
        },
    }
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Grade agent SFT JSONL conversations.")
    ap.add_argument("--input", required=True, type=Path, help="Input JSONL file to grade.")
    ap.add_argument("--output", required=True, type=Path, help="Output JSONL with quality_scores field.")
    ap.add_argument("--min-score", type=float, default=None, help="Filter records below this overall score.")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument(
        "--llm-judge-server",
        default="http://localhost:8000",
        help="OpenAI-compatible server URL for LLM-as-judge scoring. "
             "Defaults to http://localhost:8000. Uses semantic plan-param alignment "
             "instead of heuristic bigram matching for main records.",
    )
    ap.add_argument(
        "--llm-judge-model",
        default="Qwen/Qwen3-Omni-30B-A3B-Instruct",
        help="Model name to use for LLM-as-judge calls (default: Qwen/Qwen3-Omni-30B-A3B-Instruct).",
    )
    ap.add_argument(
        "--no-llm-judge",
        action="store_true",
        help="Disable LLM-as-judge scoring and use heuristic bigram matching only.",
    )
    ap.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of records to score concurrently (default: 4). Each record's scoring "
             "is independent HTTP calls so threading scales well.",
    )
    ap.add_argument(
        "--live-exec-check",
        action="store_true",
        help="Execute bash tool calls live and score execution_fidelity from runtime output/effect checks.",
    )
    ap.add_argument(
        "--live-exec-timeout-sec",
        type=float,
        default=30.0,
        help="Timeout in seconds for each live-executed bash tool call (default: 30).",
    )
    ap.add_argument(
        "--live-exec-max-calls",
        type=int,
        default=0,
        help="Optional cap on executed bash tool calls per record for faster checks (0 = all).",
    )
    ap.add_argument(
        "--audio-grounding-sample-rate",
        type=float,
        default=1.0,
        help=(
            "Fraction of per-candidate audio-grounding checks to run per search/judge "
            "record (random sample). 1.0 = all (default). E.g. 0.125 ≈ 6/48 for search, "
            "or 0.33 ≈ 4/12 for judge. Only used when --llm-judge-server is set."
        ),
    )
    args = ap.parse_args()

    summary = grade_file(
        args.input,
        args.output,
        min_score=args.min_score,
        verbose=args.verbose,
        llm_judge_server=None if args.no_llm_judge else args.llm_judge_server,
        llm_judge_model=args.llm_judge_model,
        workers=args.workers,
        live_exec_check=args.live_exec_check,
        live_exec_timeout_sec=float(args.live_exec_timeout_sec),
        live_exec_max_calls=(None if int(args.live_exec_max_calls) <= 0 else int(args.live_exec_max_calls)),
        audio_grounding_sample_rate=float(args.audio_grounding_sample_rate),
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
