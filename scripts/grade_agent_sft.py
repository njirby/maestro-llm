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
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

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

    if "set_params(" in command:
        if not isinstance(runtime_payload, dict):
            errors.append("set_params_missing_runtime_json")
        else:
            if isinstance(runtime_payload.get("not_found"), list) and runtime_payload["not_found"]:
                errors.append(f"set_params_not_found:{len(runtime_payload['not_found'])}")
            if "applied" in runtime_payload and isinstance(runtime_payload.get("applied"), list):
                if len(runtime_payload["applied"]) == 0:
                    errors.append("set_params_applied_empty")
            elif "status" not in runtime_payload:
                errors.append("set_params_missing_status_or_applied")

    return errors


def _classify_bash_command(command: str) -> str:
    if "listen_probe" in command:
        return "listen_probe"
    if "set_params(" in command:
        return "set_params"
    if "applied_wavetable" in command or "vc.set_preset(" in command:
        return "apply_wavetable"
    if "applied_tuple_id" in command:
        return "apply_tuple"
    return "bash_generic"


def _check_live_exec_environment() -> None:
    """Ensure a live REAPER + reapy session is reachable before executing checks."""
    try:
        import reapy  # noqa: PLC0415

        with reapy.inside_reaper():
            project = reapy.Project()
            _ = len(project.tracks)
    except Exception as exc:
        raise RuntimeError(
            "Live execution checks require a running REAPER session with reapy server available."
        ) from exc


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
    call_budget = max_calls if (max_calls is not None and max_calls > 0) else None

    for i, msg in enumerate(messages):
        if call_budget is not None and assessed >= call_budget:
            break
        if msg.get("role") != "tool_call":
            continue
        tc = _parse_json_object(msg.get("content", ""))
        if not isinstance(tc, dict) or tc.get("name") != "bash":
            continue

        command = tc.get("arguments", {}).get("command", "")
        if not isinstance(command, str) or not command.strip():
            continue

        expected_payload: dict[str, Any] | None = None
        if i + 1 < len(messages) and messages[i + 1].get("role") == "tool_response":
            expected_payload = _parse_json_object(messages[i + 1].get("content", ""))

        assessed += 1
        category = _classify_bash_command(command)
        try:
            proc = subprocess.run(
                ["bash", "-lc", command],
                capture_output=True,
                text=True,
                timeout=timeout_sec,
            )
            runtime_payload = _parse_json_object(proc.stdout)
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
                    "stderr_snippet": (proc.stderr or "").strip()[:300],
                }
            )
        except subprocess.TimeoutExpired:
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

    fidelity = (passed / assessed) if assessed > 0 else None
    return {
        "execution_fidelity": round(float(fidelity), 4) if fidelity is not None else None,
        "execution_checks": checks,
        "execution_calls_assessed": assessed,
        "execution_calls_passed": passed,
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
# Dispatcher and summary
# ---------------------------------------------------------------------------

def score_main_v3_record(record: dict[str, Any]) -> dict[str, Any]:
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
    residual_subs = list((meta.get("batch_labels", [{}])[-1] or {}).get("param_names", []))[:3]
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

    # -- Overall --
    weights: dict[str, float] = {
        "batch_param_alignment": 0.30,
        "diagnosis_subsystem_coverage": 0.25,
        "clap_net_improvement": 0.20,
        "verdict_grounded": 0.10,
        "snake_case_clean": 0.025,
        "format_consistent": 0.025,
    }
    if mistake_recovery is not None:
        weights["mistake_recovery"] = 0.10
    else:
        # Redistribute 10% among other weights proportionally.
        base = sum(weights.values())
        weights = {k: v / base for k, v in weights.items()}

    raw: dict[str, Any] = {
        "clap_net_improvement": clap_net_improvement,
        "diagnosis_subsystem_coverage": diagnosis_subsystem_coverage,
        "batch_param_alignment": batch_param_alignment,
        "snake_case_clean": snake_case_clean,
        "format_consistent": format_consistent,
        "verdict_grounded": verdict_grounded,
        "mistake_recovery": mistake_recovery,
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


def score_record(
    record: dict[str, Any],
    llm_judge_server: str | None = None,
    llm_judge_model: str = "Qwen/Qwen3-Omni-30B-A3B-Instruct",
    live_exec_check: bool = False,
    live_exec_timeout_sec: float = 30.0,
    live_exec_max_calls: int | None = None,
) -> dict[str, Any]:
    task = record.get("task_type", "")
    meta = record.get("meta", {})
    # Branch: v3 pipeline records use a different scoring path.
    if task == "main" and meta.get("pipeline_version") == "v3":
        scores = score_main_v3_record(record)
    elif task == "main":
        scores = score_main_record(record, llm_judge_server=llm_judge_server, llm_judge_model=llm_judge_model)
    elif task == "search":
        scores = score_search_record(record)
    elif task == "judge":
        scores = score_judge_record(record)
    else:
        scores = {"overall": None, "error": f"unknown task_type: {task!r}"}

    # Optional live runtime validation: execute bash tool calls and check that
    # runtime outputs/effects align with emitted tool_response payloads.
    if live_exec_check:
        exec_results = run_live_execution_checks_for_record(
            record,
            timeout_sec=live_exec_timeout_sec,
            max_calls=live_exec_max_calls,
        )
        scores.update(exec_results)
        exec_fidelity = exec_results.get("execution_fidelity")
        base_overall = scores.get("overall")
        if exec_fidelity is not None and base_overall is not None:
            # Keep existing quality signal dominant while making runtime fidelity
            # a meaningful factor when live checks are enabled.
            scores["overall"] = round((0.85 * float(base_overall)) + (0.15 * float(exec_fidelity)), 4)

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
                            live_exec_max_calls=live_exec_max_calls): i
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
            scores_by_idx[i] = score_record(
                record,
                llm_judge_server=llm_judge_server,
                llm_judge_model=llm_judge_model,
                live_exec_check=live_exec_check,
                live_exec_timeout_sec=live_exec_timeout_sec,
                live_exec_max_calls=live_exec_max_calls,
            )

    scored: list[dict] = []
    kept = 0
    all_scores: list[tuple[str, float]] = []  # (task, overall) for summary stats
    for i, record in enumerate(rows):
        scores = scores_by_idx[i]
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
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
