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
    """Score how well each step's HYPOTHESIS references the correct remaining subsystems.

    For each step that has a remaining_top_2 label, checks whether the HYPOTHESIS text
    mentions at least one of those subsystem names. Returns the fraction of assessable
    steps that pass, or None if no steps have remaining_top_2 data.
    """
    scores: list[float] = []
    for turn, label in zip(commentary_turns, step_labels):
        top2: str | None = label.get("remaining_top_2")
        if not top2 or top2 == "the remaining parameters":
            continue  # no ground-truth remaining info for this step
        hypo_text = _extract_section(turn, "HYPOTHESIS:").lower()
        if not hypo_text:
            scores.append(0.0)
            continue
        # top2 is like "compressor and chorus" — split on " and " and ","
        sub_names = [s.strip() for s in top2.replace(" and ", ",").split(",") if s.strip()]
        mentioned = any(name in hypo_text for name in sub_names)
        scores.append(1.0 if mentioned else 0.0)
    if not scores:
        return None
    return round(sum(scores) / len(scores), 4)


def score_main_record(
    record: dict[str, Any],
    llm_judge_server: str | None = None,
    llm_judge_model: str = "Qwen/Qwen3-Omni-30B-A3B-Instruct",
) -> dict[str, Any]:
    messages = record.get("messages", [])
    meta = record.get("meta", {})
    step_labels: list[dict] = meta.get("step_labels", [])

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

    # 7. Hypothesis grounding — fraction of steps where HYPOTHESIS mentions a top remaining
    # subsystem from the GT-preset gap. Requires remaining_top_2 in step_labels (built by
    # build_main_agent_sft_v2 when target_preset is available).
    # Only listening turns have HYPOTHESIS sections; planning-only turns are exempt.
    listening_step_labels = [
        lbl for lbl in step_labels if not lbl.get("is_planning_step", False)
    ] if step_labels else step_labels
    hypothesis_grounding = _score_hypothesis_grounding(listening_turns, listening_step_labels)

    # 8. CLAP net improvement — did the path make meaningful net progress toward GT audio?
    # Measures final-vs-initial CLAP delta normalized to [0,1]. Replaces clap_monotonic
    # which penalized intentional path_gen noise on intermediate steps.
    clap_net_improvement = _score_clap_net_improvement(step_labels)

    # 9. Optional LLM judge — kept for diagnostics/transparency but excluded from overall
    # score because plan_param_alignment is structurally guaranteed to be 1.0 (Sentence 1
    # is pre-seeded with ground-truth params), so the judge score carries no signal.
    llm_results: dict[str, Any] = {}
    if llm_judge_server:
        llm_results = llm_judge_main_record(record, llm_judge_server, llm_judge_model)

    llm_alignment: float | None = llm_results.get("llm_plan_alignment")

    # Overall weighted score.
    # plan_param_alignment / llm_plan_alignment are excluded from the overall score because
    # Sentence 1 of PLAN is pre-seeded with the ground-truth param inventory, making
    # alignment structurally guaranteed — it carries no training-quality signal.
    # commentary_diversity is also excluded — it has a structural ceiling (~0.67) in this
    # domain regardless of content quality.
    # section_structure is kept at a small weight as a correctness sanity check; it has
    # been 1.0 in all observed data (format is reliable) so it carries minimal influence.
    weights = {
        "clap_net_improvement":  0.30,
        "plan_rationale_unique": 0.25,
        "hypothesis_grounding":  0.25,
        "section_structure":     0.10,
        "snake_case_clean":      0.05,
        "format_consistent":     0.05,
    }
    raw = {
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

def score_record(
    record: dict[str, Any],
    llm_judge_server: str | None = None,
    llm_judge_model: str = "Qwen/Qwen3-Omni-30B-A3B-Instruct",
) -> dict[str, Any]:
    task = record.get("task_type", "")
    if task == "main":
        return score_main_record(record, llm_judge_server=llm_judge_server, llm_judge_model=llm_judge_model)
    if task == "search":
        return score_search_record(record)
    if task == "judge":
        return score_judge_record(record)
    return {"overall": None, "error": f"unknown task_type: {task!r}"}


def grade_file(
    input_path: Path,
    output_path: Path,
    min_score: float | None = None,
    verbose: bool = False,
    llm_judge_server: str | None = "http://localhost:8000",
    llm_judge_model: str = "Qwen/Qwen3-Omni-30B-A3B-Instruct",
    workers: int = 1,
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
                            llm_judge_model=llm_judge_model): i
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
                record, llm_judge_server=llm_judge_server, llm_judge_model=llm_judge_model
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
    args = ap.parse_args()

    summary = grade_file(
        args.input,
        args.output,
        min_score=args.min_score,
        verbose=args.verbose,
        llm_judge_server=None if args.no_llm_judge else args.llm_judge_server,
        llm_judge_model=args.llm_judge_model,
        workers=args.workers,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
