"""Rollout inputs: WS-C's manifest rows (outputs/v5_manifest/<name>/manifest.jsonl), or small test inputs built here.

Manifest row schema (consumed exactly):
  {id, archetype, gt_preset_path, init_preset_path, notes_path, target_wav_path, bpm,
   clip: {midi_path, track_idx, start_s, program}, target_rms_dbfs}
notes_path holds notes.json in the vital skill format (measures and note values, maestro.notation:
{"bpm", "time_signature", "notes": [{pitch, velocity, bar, beat, duration}]}; the legacy start_s/dur_s
form and a bare note list are still accepted). notes() returns canonical seconds for the host-side
plumbing (gate, narration, renders). Test inputs leave target_wav_path empty: the rollout renders the
target in its session with the skill's render.py (GT preset + GT notes), exactly as WS-C does.
"""
from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path

from maestro import notation

REPO = Path(__file__).resolve().parents[2]
INIT_PRESET = REPO / "opencode" / "skills" / "vital" / "data" / "init_preset.json"
CLIP_CATALOG = REPO / "outputs" / "midi_clips" / "lakh_catalog_small.jsonl"
WAVETABLE_LIB = REPO / "data" / "wavetable_lib.json"


@dataclass
class Sample:
    id: str
    archetype: str
    gt_preset_path: str
    init_preset_path: str
    notes_path: str
    target_wav_path: str | None
    bpm: float
    clip: dict = field(default_factory=dict)
    target_rms_dbfs: float | None = None
    canonical: bool = False             # scripts/canonicalize_manifest.py: every change from init is audible
    canonical_reset: list = field(default_factory=list)
    impact: dict = field(default_factory=dict)  # key -> audible impact (one-key revert distance), canonical rows

    def gt_preset(self) -> dict:
        return json.loads(Path(self.gt_preset_path).read_text())

    def init_preset(self) -> dict:
        return json.loads(Path(self.init_preset_path).read_text())

    def notes(self) -> list[dict]:
        """The GT notes as canonical seconds ({pitch, velocity, start_s, dur_s}); the file itself may
        be in the skill's beat form."""
        return notation.to_seconds(Path(self.notes_path).read_text())[0]

    def to_dict(self) -> dict:
        return asdict(self)


def load_manifest(path: str | Path) -> list[Sample]:
    keys = set(Sample.__dataclass_fields__)
    return [Sample(**{k: v for k, v in json.loads(line).items() if k in keys})
            for line in Path(path).read_text().splitlines() if line.strip()]


def make_test_samples(n: int, out_dir: str | Path, seed: int = 0) -> list[Sample]:
    """n (GT preset, Lakh clip) pairs: preset_gen.generate_preset + sounddiff.edits.pick_clip. Writes
    <out_dir>/<id>/{gt_preset.json, notes.json} and <out_dir>/manifest.jsonl."""
    from maestro.sounddiff.edits import load_clips, pick_clip
    from maestro.synth.preset_gen import ARCHETYPES, generate_preset
    from maestro.synth.wavetable_lib import load_wavetable_lib

    out_dir = Path(out_dir)
    lib = load_wavetable_lib(WAVETABLE_LIB)
    clips = load_clips(str(CLIP_CATALOG))
    rng = random.Random(seed)
    samples = []
    for i in range(n):
        sid = f"t{seed:03d}_{i:03d}"
        arch = rng.choice(ARCHETYPES)
        gt = generate_preset(arch, random.Random(rng.randrange(1 << 30)), wavetable_lib=lib)
        clip, notes = pick_clip(clips, rng)
        notes = [x for x in notes if x["dur_s"] >= 0.02]  # clip_notes cuts notes at the window end (render.py rejects 0 s)
        notes = notation.quantize(float(clip["bpm"]), notes)
        d = out_dir / sid
        d.mkdir(parents=True, exist_ok=True)
        (d / "gt_preset.json").write_text(json.dumps(gt, separators=(",", ":")))
        (d / "notes.json").write_text(notation.beat_form_json(float(clip["bpm"]), notes))
        samples.append(Sample(sid, arch, str(d / "gt_preset.json"), str(INIT_PRESET), str(d / "notes.json"), None,
                              float(clip["bpm"]), {k: clip[k] for k in ("midi_path", "track_idx", "start_s",
                                                                        "program")}))
    (out_dir / "manifest.jsonl").write_text("".join(json.dumps(s.to_dict()) + "\n" for s in samples))
    return samples
