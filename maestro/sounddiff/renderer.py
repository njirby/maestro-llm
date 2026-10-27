"""Batch renderer for the verified-notes builders: presets + notes -> float32 stereo arrays, rendered by the vital
skill's own render.py (opencode/skills/vital/scripts/render.py) inside the daw-farm image, i.e. exactly what the agent
hears when it renders.

    from maestro.sounddiff.renderer import ContainerRenderer
    with ContainerRenderer() as r:                       # one long-lived daw-farm container (DawDreamerSession)
        wavs = r.render({"full": preset, "no_reverb": p2}, {"bpm": 120, "notes": [...]})
        wavs["full"]                                     # np.ndarray (n, 2) float32, 44.1 kHz
        r.path_of(preset, notes)                         # host path of the cached wav (for an audio model)

A renderer is any callable `render(presets: {name: preset dict (full .vital json or bare settings)}, notes) ->
{name: array}`; `ContainerRenderer.render` is the reference one. Renders are cached by content (preset + notes + bpm)
in `cache_dir`, so a rollout that keeps rendering the same states (the 'after' of one edit is the 'before' of the next)
pays once. Two renders of the same preset that must differ (the render-noise floor) use `variant=` keys: names ending
in "#<k>" are rendered as separate cache entries.
"""
from __future__ import annotations

import hashlib
import json
import os
import shlex
import threading
from pathlib import Path

import numpy as np
import soundfile as sf

REPO = Path(__file__).resolve().parents[2]
DEFAULT_CACHE = REPO / "outputs" / "sounddiff" / "render_cache"


def clean_notes(notes) -> dict:
    """{"bpm", "notes"} in seconds with zero-length notes dropped (Lakh has them; render.py rejects
    them). Accepts the skill's measures-and-note-values form, the legacy start_s/dur_s form, or
    plain seconds notes (maestro.notation.to_seconds)."""
    from maestro import notation
    if isinstance(notes, (str, Path)):
        notes = json.loads(Path(notes).read_text())
    ns, bpm = notation.to_seconds(notes)
    ns = [n for n in ns if n["dur_s"] > 0]
    return {"bpm": bpm, "notes": ns}


def _as_preset(p) -> dict:
    if isinstance(p, (str, Path)):
        p = json.loads(Path(p).read_text())
    return p if "settings" in p else {"settings": p}


def _key(preset: dict, notes: dict, variant: str) -> str:
    """Cache key of a render: the preset's settings only (top-level metadata such as names / comments does not sound),
    the notes and the take variant. A bare settings dict and the full preset holding it get the same key."""
    h = hashlib.sha1(json.dumps(_as_preset(preset)["settings"], sort_keys=True, separators=(",", ":")).encode())
    h.update(json.dumps(notes, sort_keys=True).encode())
    h.update(variant.encode())
    return h.hexdigest()[:20]


class ContainerRenderer:
    """Renders through render.py in one daw-farm container (maestro.dawfarm_dd.DawDreamerSession), `jobs` in parallel.
    The cache dir is the container's rollout root, so outputs land on the host without copies."""

    def __init__(self, cache_dir: str | Path = DEFAULT_CACHE, jobs: int = 12, name: str | None = None,
                 session=None):
        self.cache = Path(cache_dir).resolve()
        self.cache.mkdir(parents=True, exist_ok=True)
        self.jobs = jobs
        self._own = session is None
        self._sess = session
        self._name = name or f"sounddiff-render-{os.getpid()}"
        self._lock = threading.Lock()

    # a caller-provided DawDreamerSession renders inside its own rollout workdir (its host_root is the cache)
    def _session(self):
        if self._sess is None:
            from maestro.dawfarm_dd import DawDreamerSession
            self._sess = DawDreamerSession(self._name, host_root=self.cache)
            self._sess.start()
        return self._sess

    def close(self):
        if self._own and self._sess is not None:
            self._sess.close()
            self._sess = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def _host_dir(self) -> tuple[Path, str]:
        s = self._session()
        if self._own:
            return self.cache, "/work/rollouts"
        return s.host_path("sounddiff"), f"{s.workdir}/sounddiff"

    def path_of(self, preset, notes, variant: str = "") -> Path:
        hd, _ = self._host_dir()
        return hd / "wav" / f"{_key(_as_preset(preset), clean_notes(notes), variant)}.wav"

    def render(self, presets: dict, notes) -> dict:
        """{name: preset} -> {name: (n, 2) float32}. Names ending in '#k' are extra takes of the same preset."""
        notes = clean_notes(notes)
        hd, cd = self._host_dir()
        for sub in ("wav", "in"):
            (hd / sub).mkdir(parents=True, exist_ok=True)
        keys, todo = {}, []
        for name, p in presets.items():
            p = _as_preset(p)
            k = _key(p, notes, name.partition("#")[2])
            keys[name] = k
            if not (hd / "wav" / f"{k}.wav").exists() and k not in [t for t, _ in todo]:
                (hd / "in" / f"{k}.vital").write_text(json.dumps(p, separators=(",", ":")))
                todo.append((k, p))
        if todo:
            nk = hashlib.sha1(json.dumps(notes, sort_keys=True).encode()).hexdigest()[:12]
            (hd / "in" / f"{nk}.notes.json").write_text(json.dumps(notes))
            lst = " ".join(k for k, _ in todo)
            cmd = (f"cd {shlex.quote(cd)} && printf '%s\\n' {lst} | xargs -P {self.jobs} -I{{}} sh -c "
                   f"'python3 \"$VITAL_SKILL/scripts/render.py\" in/{{}}.vital in/{nk}.notes.json wav/{{}}.tmp.wav "
                   f">/dev/null && mv wav/{{}}.tmp.wav wav/{{}}.wav || echo FAIL {{}}'")
            with self._lock:
                res = self._session().bash(cmd, timeout_ms=max(120_000, 20_000 * len(todo) // self.jobs + 60_000),
                                           workdir=cd)
            if "FAIL" in res.output or res.exit_code not in (0, None) or res.timed_out:
                raise RuntimeError(f"render failed: {res.output[-2000:]}")
            for k, _ in todo:
                (hd / "in" / f"{k}.vital").unlink(missing_ok=True)
        return {name: sf.read(hd / "wav" / f"{k}.wav", dtype="float32", always_2d=True)[0]
                for name, k in keys.items()}

    __call__ = render
