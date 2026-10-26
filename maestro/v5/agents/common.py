"""Helpers shared by the agent builders: run a bash call / listen to a file and append it to a Record."""
from __future__ import annotations

import json
import re

from maestro.dawfarm_dd import BashResult

from ..ctx import Ctx
from ..record import Record

RENDER = "python3 $VITAL_SKILL/scripts/render.py"
INIT = "$VITAL_SKILL/data/init_preset.json"


class StepFailed(RuntimeError):
    """A command the plan expects to succeed failed (the rollout is dropped, never recorded half-right)."""


def run(rec: Record, ctx: Ctx, cmd: str, *, expect_ok: bool = True, timeout_ms: int | None = None) -> BashResult:
    call, out, res = ctx.bash(cmd, timeout_ms=timeout_ms)
    # exit code of a multi-line command is its last line's: also catch a CLI error on any line
    if expect_ok and (res.exit_code != 0 or re.search(r"^(error: |usage: |Traceback )", res.output, re.M)):
        raise StepFailed(f"{rec.agent}: `{cmd[:200]}` exit {res.exit_code}: {res.output[-800:]}")
    rec.step([call], [out])
    return res


def listen(rec: Record, ctx: Ctx, rel: str) -> str:
    """One read of an audio file (its own step); returns the host copy."""
    host = ctx.fetch(rel)
    rec.audio_step([ctx.path(rel)], [host])
    return host


def notes_json(bpm: float, notes: list[dict]) -> str:
    """notes.json as an agent writes it: one note per line."""
    rows = ",\n".join("  " + json.dumps({k: n[k] for k in ("pitch", "velocity", "start_s", "dur_s")}) for n in notes)
    return f'{{"bpm": {json.dumps(float(bpm))}, "notes": [\n{rows}\n]}}\n'
