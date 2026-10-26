"""First impression of a target sound: ablation-verified session notes + one Qwen3-Omni-30B-A3B-Instruct call.

    from maestro.sounddiff.renderer import ContainerRenderer
    from maestro.sounddiff.first_impression import notes_first_impression, narrate_first_impression

    with ContainerRenderer() as r:
        n = notes_first_impression("target.vital", {"bpm": 120, "notes": [...]}, renderer=r)
    n["text"]      # the session notes block (every core dimension, listener words)
    n["facts"]     # flags the checker tests narrations against (dry, steady, bright, plucky, continuous, ...)
    n["audible"]   # {module: [strength, phrase]} that survived ablation
    n["audio"]     # host path of the full render the notes describe (ContainerRenderer only)

    out = narrate_first_impression(n["audio"], n, url="http://localhost:8000/v1")
    out["text"], out["attempts"], out["problems"]    # problems of the returned text ([] = clean)

notes_first_impression(target_preset_path_or_dict, notes, *, renderer) -> {text, facts, audible, audio}
    notes: {"bpm", "notes": [{pitch, velocity, start_s, dur_s}]}, a bare note list, or a path to such a json.
    renderer: maestro.sounddiff.renderer.ContainerRenderer (or any callable with its render signature).
narrate_first_impression(audio_wav_path, notes_text, *, url, model=None, facts=None, retries=1) -> {text, attempts,
    problems}
    notes_text: the notes text, or the dict from notes_first_impression (then its facts are used for the check).
    One call (audio peak-normalised to mono + notes + zero-shot instruction; no sampling params), checked; on a flagged
    sample one fresh regenerate with the same prompt (never post-edited). Returns the last sample and its problems.

Recipe and numbers: README.md here. The probe scripts next to this file (fi_notes.py CLI, omni_fi.py, iterate.py,
oscshape.py) are the experiment record.
"""
from __future__ import annotations

from ..check import check_state
from ..omni import audio_parts, chat
from ..verified import verify_state

PROMPT = ("This clip is a single synth patch playing a part. {notes}\n\nThe notes are accurate. Using them, give your "
          "first impression the way a producer would to a collaborator, in 2-3 sentences, in your own words: the part, "
          "how the notes behave, the space and movement, and the core tone. Keep subtle things subtle, don't add "
          "anything the notes rule out, and don't read the notes back as a list. No preset or wavetable names and no "
          "synth-engineering jargon (oscillators, layers, polyphony, LFO names). Reply with just the impression, in "
          "English.")
HEADER = "Session notes (verified against the session):"


def notes_text(st) -> str:
    if st.silent:
        return HEADER + "\n- The render is silent: nothing can be heard."
    d = st.dims()
    return "\n".join([HEADER, f"- Part: {d['part']}.", f"- Space: {d['space']}.", f"- Movement: {d['movement']}.",
                      f"- Distortion: {d['distortion']}.", f"- Stereo: {d['stereo']}.", f"- Tone: {d['tone']}."])


def notes_first_impression(target, notes, *, renderer) -> dict:
    st = verify_state(target, notes, renderer)
    out = {"text": notes_text(st), "facts": st.flags(), "audible": {k: list(v) for k, v in st.aud.items()},
           "silent": st.silent, "floor": st.floor}
    if hasattr(renderer, "path_of"):
        out["audio"] = str(renderer.path_of({"settings": st.settings}, st.notes))
    return out


def narrate_first_impression(audio_wav_path, notes_text, *, url, model=None, facts=None, retries=1) -> dict:
    if isinstance(notes_text, dict):
        facts = notes_text["facts"] if facts is None else facts
        notes_text = notes_text["text"]
    content = audio_parts([audio_wav_path]) + [{"type": "text", "text": PROMPT.format(notes=notes_text)}]
    attempts = []
    for _ in range(1 + retries):
        text, _ = chat(url, content, model=model)
        probs = check_state(text, facts or {})
        attempts.append({"text": text, "problems": probs})
        if not probs:
            break
    return {"text": attempts[-1]["text"], "attempts": attempts, "problems": attempts[-1]["problems"]}
