"""Prompt bank for teacher data: one- and two-audio tasks, weighted by what kind of audio a clip is.

Each task samples a prompt for a manifest row (or pair of rows). Rows look like
  {"id", "source", "kind": music|note|speech|sound|synth, "path", "dur", "meta": {...}}
Prompts never leak labels into the question except for verifiable yes/no tasks, whose expected answer is kept in
`check` so bad teacher answers can be filtered later.
With paraphrase=True the pool also includes teacher-written rephrasings (prompt_paraphrases.json, per family and kind
of audio). Whatever string is chosen is the exact prompt sent to the teacher.
"""
import json
import random
from pathlib import Path

_PARA_FILE = Path(__file__).with_name("prompt_paraphrases.json")
PARAPHRASES = json.loads(_PARA_FILE.read_text()) if _PARA_FILE.exists() else {}

SINGLE = {
    "caption_detailed": (
        ["Describe this audio in detail.",
         "Listen carefully and describe everything you hear in this recording.",
         "Give a thorough description of this sound: what it is, how it sounds, and how it changes over time.",
         "Write a detailed caption for this audio clip."],
        {"music": 3, "note": 2, "speech": 2, "sound": 3, "synth": 3}),
    "caption_brief": (
        ["Describe this audio in one sentence.",
         "Give a short caption for this sound.",
         "In a few words, what is this?"],
        {"music": 2, "note": 2, "speech": 1, "sound": 2, "synth": 2}),
    "sources": (
        ["List every distinct sound source you can hear, and roughly when each one occurs.",
         "What is making the sounds in this recording? Identify each source.",
         "Break this recording down into its individual sound events in order."],
        {"music": 1, "note": 0, "speech": 1, "sound": 3, "synth": 0}),
    "music": (
        ["Analyze this music: genre, instrumentation, tempo, key or mode, mood, and production style.",
         "What instruments are playing, and what role does each one play in the arrangement?",
         "Describe the rhythm, harmony and melody of this piece.",
         "How would you describe the mix and production of this track?"],
        {"music": 4, "note": 0, "speech": 0, "sound": 0, "synth": 1}),
    "timbre": (
        ["Describe the timbre of this sound: its brightness, harmonic content, noisiness, and texture.",
         "Describe the envelope of this sound: attack, decay, sustain and release.",
         "What is the pitch or pitch range of this sound, and is it stable or moving?",
         "What frequency ranges dominate this sound?"],
        {"music": 1, "note": 4, "speech": 0, "sound": 1, "synth": 4}),
    "synth_recreate": (
        ["If you had to recreate this sound on a synthesizer, what oscillators, filter, envelopes, modulation and effects would you use? Be specific.",
         "Explain how this sound was probably designed or produced, as a sound designer would.",
         "What synthesis technique does this sound resemble, and what parameter settings would get close to it?"],
        {"music": 1, "note": 2, "speech": 0, "sound": 0, "synth": 5}),
    "temporal": (
        ["Describe how this audio changes over time, from start to finish.",
         "What happens at the beginning, middle and end of this clip?"],
        {"music": 1, "note": 1, "speech": 1, "sound": 2, "synth": 2}),
    "scene": (
        ["Where do you think this was recorded? Describe the acoustic environment, distance, reverberation and recording quality.",
         "Describe the space this sound is in and the quality of the recording."],
        {"music": 0, "note": 0, "speech": 1, "sound": 3, "synth": 0}),
    "speech_transcribe": (
        ["Transcribe the speech in this recording exactly.",
         "What is being said?"],
        {"music": 0, "note": 0, "speech": 4, "sound": 0, "synth": 0}),
    "speaker": (
        ["Describe the speaker: voice, likely gender and age, accent, emotion, and speaking pace.",
         "How does the speaker sound? Describe their voice and delivery."],
        {"music": 0, "note": 0, "speech": 3, "sound": 0, "synth": 0}),
}

PAIR = {
    "compare": [
        "Compare these two recordings. How are they similar and how do they differ?",
        "Describe each audio clip, then explain the main differences between them.",
        "What are the most important differences between Audio 1 and Audio 2?",
    ],
    "which": [
        "Which of the two sounds is brighter? Explain why.",
        "Which of the two sounds is higher in pitch? Explain.",
        "Which clip is more rhythmic or faster? Explain.",
        "Which clip sounds more distant or reverberant? Explain.",
        "Which clip has the sharper attack? Explain.",
        "Which clip sounds more distorted or noisy? Explain.",
    ],
    "same": [
        "Do these two clips come from the same kind of source? Explain your reasoning.",
        "Could these two sounds be from the same instrument or device? Why or why not?",
    ],
    "transform": [
        "Describe what you would change to turn the first sound into the second.",
        "If Audio 2 were a processed version of Audio 1, what processing would explain the difference?",
    ],
}


def _weighted(rng, items):
    keys, weights = zip(*items)
    return rng.choices(keys, weights=weights, k=1)[0]


def single_task(row, rng: random.Random, label_pool=None, paraphrase=False):
    kind = row["kind"]
    labels = row.get("meta", {}).get("labels")
    # verifiable yes/no about AudioSet/ESC labels: half true labels, half absent ones
    if labels and label_pool and rng.random() < 0.12:
        if rng.random() < 0.5:
            lab, ans = rng.choice(labels), "yes"
        else:
            lab = rng.choice([l for l in rng.sample(label_pool, 20) if l not in labels] or label_pool)
            ans = "no"
        pool = ["Is there {label} in this recording? Answer yes or no, then explain."]
        if paraphrase:
            pool += PARAPHRASES.get("yes_no", [])
        return {"task": "yes_no", "prompt": rng.choice(pool).replace("{label}", lab.lower()),
                "check": {"label": lab, "expected": ans}}
    task = _weighted(rng, [(t, w[kind]) for t, (_, w) in SINGLE.items() if w.get(kind, 0) > 0])
    pool = SINGLE[task][0] + (PARAPHRASES.get(f"{task}|{kind}", []) if paraphrase else [])
    return {"task": task, "prompt": rng.choice(pool)}


def pair_task(a, b, rng: random.Random, paraphrase=False):
    task = rng.choices(list(PAIR), weights=[4, 4, 2, 1 if a["kind"] == b["kind"] else 0], k=1)[0]
    pool = PAIR[task] + (PARAPHRASES.get("pair_" + task, []) if paraphrase else [])
    return {"task": "pair_" + task, "prompt": rng.choice(pool)}
