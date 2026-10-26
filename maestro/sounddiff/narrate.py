"""Narrate facts (facts.py) in a producer's voice with a text-only LLM behind an OpenAI-compatible endpoint (vLLM), one
prompt per mode, hedging tied to magnitude; every narration goes through the score.py filter: on rejection the prompt
is re-sent once with the problem named (the prompt changes, the output is never edited), then a template fallback.

API (the rollout builder calls this):

    from maestro.sounddiff.narrate import Narrator, narrate
    nar = Narrator(url="http://127.0.0.1:8001/v1", model="qwen3.8-27b")       # default: thinking on, effort xhigh
    rows = narrate("self_edit_delta", [facts, ...], nar)   # batched (thread pool), order preserved
    rows[i]["text"], rows[i]["fallback"], rows[i]["attempts"]   # attempts: [{"text", "check", "reasoning_chars"}]

Narrator(think=False, sys_instruction=True) is thinking off with the template's xhigh correctness instruction moved
into the system prompt (the Qwen3.8 template only adds it when thinking is on).

CLI over a probe dir (items.jsonl + measured renders):
  .venv/bin/python -m maestro.sounddiff.narrate outputs/sounddiff/probe1 --tag 27b-xhigh
  .venv/bin/python -m maestro.sounddiff.narrate outputs/sounddiff/probe1 --tag 27b-nothink-sys --no-think --sys-instruction
"""
from __future__ import annotations

import argparse
import json
import random
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import requests

from . import score
from .facts import TONE, WORDS, Measured, facts_for, near_silent

SYSTEM = (
    "You are an experienced electronic music producer sound-designing in Vital. You talk like a producer in the studio: "
    "short, concrete, about what you hear, not about knob values or numbers.")
# what the Qwen3.8 chat template injects for reasoning_effort=xhigh (only when thinking is on)
XHIGH = ("Please think carefully through the task, validate key assumptions, consider plausible alternatives, and "
         "prioritize correctness, consistency, and clarity in the final answer.")

COMMON = ("Only describe what the facts support: don't mention any quality of the sound the facts don't list, and "
          "don't quote numbers, units or parameter names. Don't recite the list - pick what matters most. Match your "
          "confidence to the size of each change: 'much' / 'clearly' you can say plainly; 'slightly' gets a hedge "
          "(\"maybe a touch...\", \"a little...\"). Tone words the facts mark as uncertain must be hedged. Reply with "
          "just the sentence(s).")

PROMPTS = {
    "self_edit_delta": """You just made an edit to your patch and played the part again to compare. Here is the edit and \
what the before/after renders measure:

{facts}

In one or two sentences, say what you hear changed, the way you'd say it to a collaborator sitting next to you: the \
one or two changes that matter most, the biggest first. Go by what was measured, not by what the edit was supposed to \
do. If the overall difference is none, say you can't really hear a difference. """ + COMMON,

    "target_vs_current": """You're recreating a target sound. You just made an edit and A/B'd your patch against the \
target. Here is the edit and what the renders measure:

{facts}

In one or two sentences, think out loud while working: did the last edit get you closer, and what's the biggest \
remaining difference from the target? If the facts give your next move, say that's what you'll go after next; if they \
don't, don't propose a next step. If it basically matches, say so. """ + COMMON,

    "first_impression": """You're auditioning a patch for the first time. Here is what stands out in its render, \
compared with lots of other synth patches:

{facts}

In one sentence, give your quick first impression, the way you'd describe it to a collaborator ("sounds like...", \
"kind of a ... with ..."): the two or three things that stand out, not a list of everything. """ + COMMON,
}

RETRY = ("\n\nImportant: a previous draft of this description was rejected because it {problems}. Describe only what the "
         "facts above list.")


@dataclass
class Narrator:
    url: str = "http://127.0.0.1:8001/v1"
    model: str = "qwen3.8-27b"
    think: bool = True
    effort: str | None = None          # Qwen3.8 template: None -> xhigh (template default), "medium", "low"
    sys_instruction: bool = False      # put the xhigh correctness instruction into the system prompt
    temperature: float | None = None   # default 1.0 thinking / 0.7 not
    workers: int = 24
    max_tokens: int = 8000

    def system(self) -> str:
        return SYSTEM + ("\n\n" + XHIGH if self.sys_instruction else "")

    def call(self, prompt: str, system: str | None = None, json_mode=False) -> tuple[str, str]:
        temp = self.temperature if self.temperature is not None else (1.0 if self.think else 0.7)
        kw = {"enable_thinking": self.think, **({"reasoning_effort": self.effort} if self.effort else {})}
        body = {"model": self.model, "temperature": temp, "top_p": 0.95 if self.think else 0.8, "top_k": 20,
                "max_tokens": self.max_tokens, "chat_template_kwargs": kw,
                "messages": [{"role": "system", "content": system or self.system()}, {"role": "user", "content": prompt}]}
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        r = requests.post(f"{self.url}/chat/completions", json=body, timeout=900)
        r.raise_for_status()
        msg = r.json()["choices"][0]["message"]
        # content is None when thinking used the whole token budget
        return (msg["content"] or "").strip(), (msg.get("reasoning_content") or msg.get("reasoning") or "")


def _problems_text(problems: list[str]) -> str:
    return "; ".join(p.replace("claims not supported by the facts:", "mentioned things the facts don't support:")
                     for p in problems)


def narrate_one(mode: str, facts: dict, nar: Narrator, check=True) -> dict:
    prompt = PROMPTS[mode].format(facts=facts["text"])
    attempts = []
    for k in range(2 if check else 1):
        p = prompt if k == 0 else prompt + RETRY.format(problems=_problems_text(attempts[-1]["check"]["problems"]))
        text, reasoning = nar.call(p)
        c = score.check(facts, text)
        attempts.append({"text": text, "check": c, "reasoning_chars": len(reasoning)})
        if c["ok"] or not check:
            return {"id": facts.get("id"), "mode": mode, "text": text, "fallback": False, "attempts": attempts,
                    "audibility": facts.get("audibility")}
    return {"id": facts.get("id"), "mode": mode, "text": template(facts), "fallback": True, "attempts": attempts,
            "audibility": facts.get("audibility")}


def narrate(mode: str, facts_list: list[dict], nar: Narrator | None = None, check=True) -> list[dict]:
    """Narrate many facts of one mode concurrently; output order matches input."""
    nar = nar or Narrator()
    with ThreadPoolExecutor(nar.workers) as ex:
        return list(ex.map(lambda f: narrate_one(mode, f, nar, check), facts_list))


# --- template fallback ----------------------------------------------------------------------------------------------

def _phrase(c: dict) -> str:
    if c["dim"] == TONE:
        return {"much": "the whole character changed", "clearly": "the tone's clearly different",
                "slightly": "the tone's maybe a little different"}[c["size"]]
    hedge = {"much": "a lot ", "clearly": "", "slightly": "maybe a touch "}[c["size"]]
    return f"{hedge}{c['direction'].split(' / ')[0].split(':')[0]}"


def template(facts: dict) -> str:
    rng = random.Random(str(facts.get("id")))
    mode = facts["mode"]
    if mode == "self_edit_delta":
        if facts["audibility"] == "none":
            return rng.choice(["Honestly, I can't really hear a difference.", "Can't really hear that one, sounds the same."])
        if facts.get("gate") and "nearly silent / almost gone" in facts["gate"]:
            return "That pretty much killed it - it's almost gone now."
        sh = facts["shown"][:2]
        if not sh:
            return "There's a small difference, maybe in the overall tone, but it's subtle."
        return (" and ".join(_phrase(c) for c in sh).capitalize() + ".").strip()
    if mode == "target_vs_current":
        traj = {"closer": "That got me closer.", "further": "Hm, that took me further away.",
                "same": "Not much closer with that one."}[facts["trajectory"]]
        if facts["audibility"] == "none":
            return traj + " It basically matches the target now."
        gaps = facts["shown"][:1]
        g = " and ".join("a different overall tone" if c["dim"] == TONE else
                         f"{c['size'] if c['size'] != 'slightly' else 'a bit'} {c['direction'].split(' / ')[0]}"
                         for c in gaps)
        s = f"{traj} The target's still {g}." if g else traj
        if facts.get("focus"):
            s += f" I'll go after the {facts['focus'].split(' / ')[0]} next."
        return s
    parts = [v.split(",")[0].split(" / ")[0].split(":")[0].split(" (")[0] for v in list(facts["shown"].values())[:2]]
    s = "Kind of a " + " and ".join(parts) + " sound" if parts else "Pretty neutral sound"
    if facts["adjectives"]:
        s += f", maybe a little {facts['adjectives'][0]}"
    return s + "."


# --- CLI ------------------------------------------------------------------------------------------------------------

def load_probe(d: Path, peak_min_db=-55.0):
    """Items + their facts for a probe dir. Skipped: items whose reference render is (nearly) silent below peak_min_db,
    and target / first-impression items whose target is under the near-silence gate (nothing to recreate)."""
    m = Measured(d)
    items = [json.loads(l) for l in open(d / "items.jsonl")]
    keep = []
    for it in items:
        ref = it["wavs"].get("target") or it["wavs"]["before"]
        if m.desc(ref)["peak_db"] < peak_min_db or m.desc(it["wavs"].get("after", ref))["peak_db"] < peak_min_db:
            continue
        if it["mode"] != "self_edit_delta" and near_silent(m.desc(it["wavs"]["target"])):
            continue
        keep.append((it, facts_for(it["mode"], it, m)))
    return m, keep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--url", default="http://127.0.0.1:8001/v1")
    ap.add_argument("--model", default="qwen3.8-27b")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--no-think", action="store_true")
    ap.add_argument("--effort", choices=["low", "medium", "xhigh"])
    ap.add_argument("--sys-instruction", action="store_true")
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--no-check", action="store_true")
    a = ap.parse_args()
    d = Path(a.dir)
    nar = Narrator(a.url, a.model, think=not a.no_think, effort=a.effort, sys_instruction=a.sys_instruction,
                   workers=a.workers)
    _, items = load_probe(d)
    with open(d / f"narr_{a.tag}.jsonl", "w") as f:
        for mode in PROMPTS:
            fl = [fa for it, fa in items if it["mode"] == mode]
            for r in narrate(mode, fl, nar, check=not a.no_check):
                f.write(json.dumps({**r, "tag": a.tag}, ensure_ascii=False) + "\n")
            print(f"{mode}: {len(fl)} narrated", flush=True)
    print(f"-> {d}/narr_{a.tag}.jsonl")


if __name__ == "__main__":
    main()
