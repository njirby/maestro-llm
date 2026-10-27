"""Helpers shared by the agent builders: run a bash call / listen to a file and append it to a Record."""
from __future__ import annotations

import json
import re

from maestro.dawfarm_dd import BashResult
from maestro import notation
from scripts import opencode_contract as oc

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
    """notes.json as an agent writes it: measures and note values, one note per line (maestro.notation)."""
    return notation.beat_form_json(bpm, notes)


SKILL_MD = (oc.OPENCODE_DIR / "skills" / "vital" / "SKILL.md").read_text()


def skill_files() -> list[str]:
    """The <skill_files> sample exactly as captured from OpenCode (ripgrep order), rebased onto CONFIG_DIR."""
    import re
    case = next(c for c in json.loads((oc.GOLDENS_DIR / "with_plugin" / "tool_results.json").read_text())
                if c["case"] == "skill_vital")
    return [f.replace("{CONFIG_DIR}", oc.CONFIG_DIR) for f in re.findall(r"<file>(.*?)</file>", case["output"])]


def skill_response() -> str:
    return oc.skill_output("vital", f"{oc.CONFIG_DIR}/skills/vital/SKILL.md", SKILL_MD, files=skill_files())


def open_session(rec: Record, ctx: Ctx) -> str:
    """How every agent's session starts (what real models do with these prompts): load the `vital` skill, then
    listen to the target. Returns the target's host copy."""
    rec.step([oc.skill_call("vital")], [skill_response()])
    return listen(rec, ctx, "target.wav")
