"""Per-rollout execution context: one DawDreamer session workdir shared by the main agent and its subagents (OpenCode
subagents run in the parent's cwd), real tool execution, and host copies of every clip an agent listens to."""
from __future__ import annotations

import hashlib
import itertools
import json
import random
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from maestro.dawfarm_dd import ROLLOUT_ROOT, BashResult, DawDreamerSession
from scripts import opencode_contract as oc

from .inputs import Sample

TRUNC_HEAD = "...output truncated...\n\n"


def opencode_bash_text(res: BashResult, spill_path: str) -> str:
    """DawDreamerSession.bash output as OpenCode shows it: the session already tail-truncates and adds the timeout
    footer but omits the "Full output saved to: <spill file>" line OpenCode puts under the truncation header."""
    out = res.output
    if res.truncated and out.startswith(TRUNC_HEAD):
        out = f"{TRUNC_HEAD}Full output saved to: {spill_path}\n\n{out[len(TRUNC_HEAD):]}"
    return out


@dataclass
class Ctx:
    sess: DawDreamerSession
    sample: Sample
    rng: random.Random
    run_dir: Path
    wd: str = ""
    events: list = field(default_factory=list)       # build log: what was injected where (for the index/meta)
    _spill: itertools.count = field(default_factory=itertools.count)
    _fetch: itertools.count = field(default_factory=itertools.count)
    _copies: dict = field(default_factory=dict)

    @property
    def audio_dir(self) -> Path:
        return self.run_dir / "audio" / self.sample.id

    def path(self, rel: str) -> str:
        """Absolute container path (what the agents write in tool calls) of a workdir-relative file."""
        return rel if rel.startswith("/") else f"{self.wd}/{rel}"

    def bash(self, command: str, timeout_ms: int | None = None) -> tuple[str, str, BashResult]:
        """Run an agent bash command for real; returns (tool_call content, tool_response content, raw result)."""
        res = self.sess.bash(command, timeout_ms=timeout_ms)
        spill = oc.tool_output_path(seed=f"{self.sample.id}/bash/{next(self._spill)}")
        return oc.bash_call(command, timeout_ms=timeout_ms), opencode_bash_text(res, spill), res

    def write(self, rel: str, content: str) -> tuple[str, str]:
        """The write tool: writes the file in the workdir; returns (tool_call, tool_response)."""
        self.sess.put_text(content, rel)
        return oc.write_call(self.path(rel), content), oc.WRITE_OK

    def fetch(self, rel: str) -> str:
        """Host snapshot of a workdir file the agent reads (audio); returns the host path for `audios`. Every distinct
        content gets its own copy (<n>_<name>; agents re-render to the same name), identical content is shared."""
        rel = rel[len(self.wd) + 1:] if rel.startswith(self.wd + "/") else rel
        data = self.host(rel).read_bytes()
        key = hashlib.sha1(data).hexdigest()
        if key not in self._copies:
            dst = self.audio_dir / f"{next(self._fetch):03d}_{rel.replace('/', '_')}"
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(data)
            self._copies[key] = str(dst.resolve())
        return self._copies[key]

    def host(self, rel: str) -> Path:
        """Live host path of a workdir file (through the session's bind mount)."""
        return self.sess.host_path(rel)

    def put(self, host_path: str | Path, rel: str) -> None:
        self.sess.put(host_path, rel)

    def read_json(self, rel: str) -> dict:
        return json.loads(self.host(rel).read_text())

    def note(self, **event) -> None:
        self.events.append(event)

    # -- oracle side (never in a record): renders the builder needs for measures and narration ablations ---------
    # GT/oracle material lives host-side only (<run>/oracle/<sample>/). A render stages its preset + notes in a
    # one-off dir under the rollout root, renders, moves the wav out and deletes the dir before any agent command
    # runs again (agent commands and oracle renders never overlap in a session), so nothing reachable from the
    # agents' workdir, even via ../, holds answers.

    @property
    def oracle_host_dir(self) -> Path:
        return self.run_dir / "oracle" / self.sample.id

    def oracle_render(self, preset: dict | str | Path, name: str, notes: list[dict] | None = None) -> Path:
        """Render a preset (dict, or host path of a preset file) through the GT notes (or `notes`); returns the
        host-side wav path."""
        d = f"{ROLLOUT_ROOT}/.render-{hashlib.sha1(f'{self.sample.id}/{name}'.encode()).hexdigest()[:12]}"
        hd = self.sess.host_path(d)
        if hd.exists():
            shutil.rmtree(hd)
        hd.mkdir(parents=True)
        try:
            (hd / "notes.json").write_text(json.dumps({"bpm": self.sample.bpm, "notes": notes or self.sample.notes()}))
            if isinstance(preset, dict):
                (hd / "p.json").write_text(json.dumps(preset))
            else:
                shutil.copyfile(preset, hd / "p.json")
            res = self.sess.bash("python3 $VITAL_SKILL/scripts/render.py p.json notes.json out.wav", workdir=d)
            if res.exit_code != 0:
                raise RuntimeError(f"oracle render {name} failed: {res.output}")
            dst = self.oracle_host_dir / f"{name}.wav"
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(hd / "out.wav", dst)
            return dst
        finally:
            shutil.rmtree(hd, ignore_errors=True)


def copy_tree(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


class OracleRenderer:
    """maestro.sounddiff.renderer.ContainerRenderer's interface (render(presets, notes) -> {name: array}, path_of)
    for WS-B's verified-notes builders, on this rollout's own pool session and host-side only.

    ContainerRenderer(session=...) would stage and cache inside the agents' workdir (<workdir>/sounddiff); this one
    stages each batch in a one-off dir under the rollout root, renders it there (render.py, `jobs` in parallel), moves
    the wavs to a host cache (<run>/oracle/<sample>/render_cache) and deletes the staging dir before the next agent
    command, like Ctx.oracle_render. Same cache keys as ContainerRenderer. cache_dir: <run>/oracle/<sample>/render_cache
    (Ctx: OracleRenderer(ctx.sess, ctx.oracle_host_dir / "render_cache"))."""

    def __init__(self, sess: DawDreamerSession, cache_dir: Path, jobs: int = 8):
        self.sess, self.jobs = sess, jobs
        self.cache = Path(cache_dir)

    def path_of(self, preset, notes, variant: str = "") -> Path:
        from maestro.sounddiff.renderer import _as_preset, _key, clean_notes
        return self.cache / f"{_key(_as_preset(preset), clean_notes(notes), variant)}.wav"

    def render(self, presets: dict, notes) -> dict:
        import soundfile as sf
        from maestro.sounddiff.renderer import _as_preset, _key, clean_notes
        notes = clean_notes(notes)
        self.cache.mkdir(parents=True, exist_ok=True)
        keys, todo = {}, {}
        for name, p in presets.items():
            p = _as_preset(p)
            k = _key(p, notes, name.partition("#")[2])
            keys[name] = k
            if not (self.cache / f"{k}.wav").exists():
                todo[k] = p
        if todo:
            d = f"{ROLLOUT_ROOT}/.render-{hashlib.sha1(json.dumps(sorted(todo)).encode()).hexdigest()[:12]}"
            hd = self.sess.host_path(d)
            shutil.rmtree(hd, ignore_errors=True)
            hd.mkdir(parents=True)
            try:
                (hd / "notes.json").write_text(json.dumps(notes))
                for k, p in todo.items():
                    (hd / f"{k}.json").write_text(json.dumps(p, separators=(",", ":")))
                cmd = (f"printf '%s\\n' {' '.join(todo)} | xargs -P {self.jobs} -I{{}} sh -c "
                       f"'python3 \"$VITAL_SKILL/scripts/render.py\" {{}}.json notes.json {{}}.wav >/dev/null "
                       f"|| echo FAIL {{}}'")
                res = self.sess.bash(cmd, timeout_ms=max(120_000, 20_000 * len(todo) // self.jobs + 60_000),
                                         workdir=d)
                if "FAIL" in res.output or res.exit_code not in (0, None) or res.timed_out:
                    raise RuntimeError(f"oracle batch render failed: {res.output[-2000:]}")
                for k in todo:
                    shutil.move(hd / f"{k}.wav", self.cache / f"{k}.wav")
            finally:
                shutil.rmtree(hd, ignore_errors=True)
        return {name: sf.read(self.cache / f"{k}.wav", dtype="float32", always_2d=True)[0]
                for name, k in keys.items()}

    __call__ = render
