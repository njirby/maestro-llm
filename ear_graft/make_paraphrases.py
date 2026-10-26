"""Write ear_graft/prompt_paraphrases.json: ~20 new phrasings per prompt family, written by the served teacher (text only).

  .venv/bin/python -m ear_graft.make_paraphrases

These only widen the prompt pool used to BUILD new tasks; every prompt is sent to the teacher as-is and answered there.
"""
import json
import re
from pathlib import Path

import httpx

from .prompts import PAIR, SINGLE

URL, MODEL = "http://127.0.0.1:8100/v1/chat/completions", "omni30b-thinking"
OUT = Path(__file__).with_name("prompt_paraphrases.json")

ASK = """I'm building a dataset of requests that people send to an AI model along with {what}. Here are example requests for one kind of task:

{examples}

Write {n} new requests for this same kind of task. Vary the wording, length and register: some short and casual, some precise and technical, some questions, some instructions. Only a few should mention who is asking. Keep the intent the same. {extra}

Finish with a JSON list of strings."""


KIND_DESC = {
    "music": "a music recording (a song, a loop, or an instrument stem)",
    "synth": "a synthesizer or electronic sound (a synth patch, loop or rendered MIDI part)",
    "note": "a single isolated musical note played on one instrument",
    "speech": "a recording of someone speaking",
    "sound": "an everyday or environmental sound recording (animals, machines, nature, household, crowds...)",
}
ANY = "Every request must make sense whatever the clip contains (music, speech, synth sounds, single notes or everyday sounds), so don't assume a genre, instrument, lyrics or speech."


def families():
    # single-clip families: one set per kind of audio the family is used for, so every prompt fits its clip
    for name, (templates, weights) in SINGLE.items():
        for kind, w in weights.items():
            if w > 0:
                yield f"{name}|{kind}", templates, f"a single audio clip that is {KIND_DESC[kind]}", "Each request must make sense for that kind of clip."
    for name, templates in PAIR.items():
        yield "pair_" + name, templates, 'two audio clips (labelled "Audio 1" and "Audio 2")', ANY
    yield ("yes_no", ["Is there {label} in this recording? Answer yes or no, then explain."], "a single audio clip",
           "Every request must contain the literal placeholder {label} (a sound class such as 'dog barking') and must ask for a yes/no answer followed by an explanation. " + ANY)


def main():
    out = {}
    with httpx.Client(timeout=1800) as client:
        for name, templates, what, extra in families():
            n = 30 if name.startswith("pair_") else 12
            msg = ASK.format(what=what, examples="\n".join(f"- {t}" for t in templates), n=n, extra=extra)
            r = client.post(URL, json={"model": MODEL, "messages": [{"role": "user", "content": msg}]}).json()
            text = r["choices"][0]["message"]["content"]
            m = re.findall(r"\[[\s\S]*\]", text)
            items = json.loads(m[-1]) if m else []
            items = [s.strip() for s in items if isinstance(s, str) and s.strip() and s.strip() not in templates]
            if name == "yes_no":
                items = [s for s in items if "{label}" in s]
            out[name] = sorted(set(items))
            print(f"{name}: {len(out[name])}  e.g. {out[name][:2]}", flush=True)
    OUT.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print("wrote", OUT, sum(map(len, out.values())), "paraphrases")


if __name__ == "__main__":
    main()
