"""
dawfarm_dd.py — lightweight agent execution environment for the DawDreamer-based vital skill.

Each DawDreamerSession is one long-lived container from the daw-farm image, started with `--entrypoint bash`
(no REAPER): Xvfb on :99 (respawned if it dies), `--network none`, uid 1000, and only two mounts:

  <host_root>  -> /work/rollouts          (rw; per-rollout workdirs /work/rollouts/<sid> = <host_root>/<sid>)
  <skill_dir>  -> <skill_mount>           (ro; defaults to the same absolute path as on the host, so the
                                           "Base directory for this skill" OpenCode reports resolves in both places)

Nothing else of the host is visible, so an agent cannot read target presets or answers. Agent commands run through
`bash(cmd, timeout_ms)`, which mirrors OpenCode's bash tool: stdout+stderr combined in arrival order, run in the
rollout workdir, killed at the timeout (default 120000 ms), output tail-truncated to 2000 lines / 50 KB.
DISPLAY, VITAL_VST3 and VITAL_SKILL are set container-wide, so the skill scripts run as documented.

Files move by plain host copies through the bind mount (no `docker cp`), so put/get are cheap.

    pool = DawDreamerPool.start(2)
    with pool.acquire() as sess:
        wd = sess.begin_rollout("sample-001")
        sess.put("init_preset.json", "preset.json")
        out, code = sess.bash(f"python3 {sess.skill_mount}/scripts/render.py preset.json notes.json a.wav")
        sess.get("a.wav", "/tmp/a.wav")
    pool.close()

Transport (docker exec, per-session lock, acquire/release pool) is reused from maestro.reaper.dawfarm.
"""

from __future__ import annotations

import atexit
import logging
import shlex
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from maestro.reaper.dawfarm import DawFarmPool, DockerSession

logger = logging.getLogger(__name__)

REPO = Path(__file__).resolve().parents[1]
IMAGE = "daw-farm/reaper:latest"
SKILL_DIR = REPO / "opencode" / "skills" / "vital"
HOST_ROOT = REPO / "outputs" / "dd_sessions"
ROLLOUT_ROOT = "/work/rollouts"
DISPLAY = ":99"
VITAL_VST3 = "/usr/lib/vst3/Vital.vst3"
NAME_PREFIX = "dd-session-"

# OpenCode bash tool semantics (v1.19): default timeout, output limits (tail kept), timeout notice.
DEFAULT_TIMEOUT_MS = 120_000
MAX_LINES, MAX_BYTES = 2000, 50 * 1024
TIMEOUT_NOTE = ("shell tool terminated command after exceeding timeout {ms} ms. If this command is expected to take "
                "longer and is not waiting for interactive input, retry with a larger timeout value in milliseconds.")

_BOOT = (f"while true; do rm -f /tmp/.X99-lock /tmp/.X11-unix/X99; "
         f"Xvfb {DISPLAY} -screen 0 1024x768x24 -nolisten tcp >/dev/null 2>&1; sleep 1; done & "
         "exec sleep infinity")


# Kill everything but init, the Xvfb respawn loop / Xvfb and `sleep infinity` (left-over agent jobs, e.g. `cmd &`).
_KILL_STRAYS = ('for p in $(ps -eo pid=); do [ "$p" = 1 ] || [ "$p" = $$ ] && continue; '
                'a=$(tr "\\0" " " < /proc/$p/cmdline 2>/dev/null); case "$a" in *Xvfb*|"sleep infinity"*|"") continue;; esac; '
                'kill -9 "$p" 2>/dev/null; done; rm -rf /tmp/tmp* 2>/dev/null; true')


@dataclass
class BashResult:
    output: str          # combined stdout+stderr, OpenCode-formatted (truncation / timeout notes, "(no output)")
    exit_code: int | None  # None when killed by the timeout
    timed_out: bool
    truncated: bool
    seconds: float

    def __iter__(self):  # out, code = sess.bash(...)
        return iter((self.output, self.exit_code))


def _truncate_tail(text: str) -> tuple[str, bool]:
    """Keep the last MAX_LINES lines / MAX_BYTES bytes, as OpenCode does."""
    lines = text.split("\n")
    if len(lines) <= MAX_LINES and len(text.encode()) <= MAX_BYTES:
        return text, False
    kept, size = [], 0
    for line in reversed(lines):
        if len(kept) >= MAX_LINES:
            break
        n = len(line.encode()) + (1 if kept else 0)
        if size + n > MAX_BYTES:
            if not kept:
                kept.append(line.encode()[-MAX_BYTES:].decode(errors="ignore"))
            break
        kept.append(line)
        size += n
    return "\n".join(reversed(kept)), True


class DawDreamerSession(DockerSession):
    """One container + a current rollout workdir. Owned by one worker at a time (via DawDreamerPool.acquire)."""

    def __init__(self, name: str, host_root: str | Path | None = None, skill_dir: str | Path = SKILL_DIR,
                 skill_mount: str | None = None, image: str = IMAGE):
        super().__init__(name)
        self.host_root = Path(host_root or HOST_ROOT / name).resolve()
        self.skill_dir = Path(skill_dir).resolve()
        self.skill_mount = skill_mount or str(self.skill_dir)
        self.image = image
        self.sid: str | None = None
        self.started = False

    # -- lifecycle -----------------------------------------------------------

    def start(self, cpus: float | None = None, memory: str | None = None, ready_timeout: float = 30.0) -> None:
        if not (self.skill_dir / "SKILL.md").is_file():
            raise RuntimeError(f"no skill at {self.skill_dir}")
        self.host_root.mkdir(parents=True, exist_ok=True)
        subprocess.run(["docker", "rm", "-f", self.name], capture_output=True, timeout=60)
        argv = ["docker", "run", "-d", "--init", "--name", self.name, "--user", "1000:1000", "--network", "none",
                "--label", "maestro/dd-session=1",
                "-v", f"{self.host_root}:{ROLLOUT_ROOT}", "-v", f"{self.skill_dir}:{self.skill_mount}:ro",
                "-e", f"DISPLAY={DISPLAY}", "-e", f"VITAL_VST3={VITAL_VST3}", "-e", f"VITAL_SKILL={self.skill_mount}",
                "-e", "PYTHONDONTWRITEBYTECODE=1", "-w", ROLLOUT_ROOT, "--entrypoint", "bash"]
        if cpus:
            argv += ["--cpus", str(cpus)]
        if memory:
            argv += ["--memory", memory]
        proc = subprocess.run(argv + [self.image, "-c", _BOOT], capture_output=True, text=True, timeout=120)
        if proc.returncode != 0:
            raise RuntimeError(f"{self.name}: docker run failed: {proc.stderr.strip()}")
        self.started = True
        deadline = time.monotonic() + ready_timeout
        while time.monotonic() < deadline:
            if self.healthy(timeout=10):
                return
            time.sleep(0.25)
        raise RuntimeError(f"{self.name}: Xvfb not ready after {ready_timeout:.0f}s")

    def healthy(self, timeout: float = 30.0) -> bool:
        res = self.exec_argv(["bash", "-c", f"test -S /tmp/.X11-unix/X99 && test -e {VITAL_VST3} "
                                            f"&& test -r {shlex.quote(self.skill_mount)}/SKILL.md"],
                             stdin="", timeout=timeout)
        return res.ok

    def close(self) -> None:
        subprocess.run(["docker", "rm", "-f", self.name], capture_output=True, timeout=60)
        self.started = False

    # -- rollouts ------------------------------------------------------------

    @property
    def workdir(self) -> str:
        if self.sid is None:
            raise RuntimeError(f"{self.name}: no rollout started (begin_rollout)")
        return f"{ROLLOUT_ROOT}/{self.sid}"

    def host_path(self, container_path: str) -> Path:
        """Host path of a file under /work/rollouts (relative paths are relative to the rollout workdir)."""
        p = container_path if container_path.startswith("/") else f"{self.workdir}/{container_path}"
        p = str(Path(p))
        if p != ROLLOUT_ROOT and not p.startswith(ROLLOUT_ROOT + "/"):
            raise ValueError(f"{container_path!r} is outside {ROLLOUT_ROOT}")
        return self.host_root / Path(p).relative_to(ROLLOUT_ROOT)

    def begin_rollout(self, sid: str, keep_previous: bool = False) -> str:
        """Fresh workdir for a rollout; kills stray agent processes and (by default) purges earlier rollouts' files.
        Hygiene is pull-based like dawfarm.prepare_sample_dirs: the previous rollout stays inspectable until now."""
        if not sid or "/" in sid or sid in (".", ".."):
            raise ValueError(f"bad rollout id {sid!r}")
        self.exec_argv(["bash", "-c", _KILL_STRAYS], stdin="", timeout=30)
        if not keep_previous:
            for child in self.host_root.iterdir():
                shutil.rmtree(child) if child.is_dir() else child.unlink()
        rd = self.host_root / sid
        if rd.exists():
            shutil.rmtree(rd)
        rd.mkdir(parents=True)
        self.sid = sid
        return self.workdir

    # -- agent tool ----------------------------------------------------------

    def bash(self, cmd: str, timeout_ms: int | None = None, workdir: str | None = None) -> BashResult:
        """Run an agent bash command like OpenCode's bash tool: combined output, workdir cwd, timeout kill."""
        timeout_ms = DEFAULT_TIMEOUT_MS if timeout_ms is None else int(timeout_ms)
        if timeout_ms <= 0:
            raise ValueError(f"Invalid timeout value: {timeout_ms}. Timeout must be a positive number.")
        wd = workdir or self.workdir
        secs = timeout_ms / 1000
        script = f"exec 2>&1\ncd {shlex.quote(wd)} || exit 1\n{cmd}\n"
        argv = self._exec_argv(["timeout", "-k", "3", f"{secs:.3f}", "bash", "-c", script])
        t0 = time.monotonic()
        with self._lock:
            try:
                proc = subprocess.run(argv, input=b"", capture_output=True, timeout=secs + 30)
                raw, code = proc.stdout, proc.returncode
            except subprocess.TimeoutExpired as exc:  # docker exec itself hung; the in-container timeout failed
                raw, code = exc.stdout or b"", 124
        elapsed = time.monotonic() - t0
        timed_out = code in (124, 137) and elapsed >= secs - 0.05
        text, truncated = _truncate_tail(raw.decode("utf-8", errors="replace"))
        if truncated:
            text = "...output truncated...\n\n" + text
        if not text:
            text = "(no output)"
        if timed_out:
            text += f"\n\n<shell_metadata>\n{TIMEOUT_NOTE.format(ms=timeout_ms)}\n</shell_metadata>"
        return BashResult(text, None if timed_out else code, timed_out, truncated, elapsed)

    # -- files (host copies through the bind mount) ---------------------------

    def put(self, host_path: str | Path, container_path: str) -> None:
        dst = self.host_path(container_path)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(host_path, dst)

    def put_text(self, text: str, container_path: str) -> None:
        dst = self.host_path(container_path)
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(text)

    def get(self, container_path: str, host_path: str | Path) -> Path:
        src = self.host_path(container_path)
        if not src.is_file():
            raise FileNotFoundError(f"{self.name}: {container_path} does not exist")
        host_path = Path(host_path)
        host_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, host_path)
        return host_path


class DawDreamerPool(DawFarmPool):
    """Thread-safe pool of DawDreamerSessions (acquire() from DawFarmPool). Owns its containers: close() removes them."""

    @classmethod
    def start(cls, n: int, prefix: str = NAME_PREFIX, host_root: str | Path = HOST_ROOT,
              skill_dir: str | Path = SKILL_DIR, skill_mount: str | None = None, image: str = IMAGE,
              cpus: float | None = None, memory: str | None = None) -> "DawDreamerPool":
        sessions = [DawDreamerSession(f"{prefix}{i}", Path(host_root) / f"{prefix}{i}", skill_dir, skill_mount, image)
                    for i in range(n)]
        try:
            with ThreadPoolExecutor(max_workers=min(n, 8)) as ex:
                list(ex.map(lambda s: s.start(cpus=cpus, memory=memory), sessions))
        except Exception:
            for s in sessions:
                s.close()
            raise
        pool = cls(sessions)
        atexit.register(pool.close)
        print(f"dawdreamer pool: {n} session(s): {', '.join(s.name for s in sessions)}", flush=True)
        return pool

    def close(self) -> None:
        for s in self.sessions:
            if s.started:
                s.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def cleanup_stale(prefix: str = NAME_PREFIX) -> list[str]:
    """Remove leftover session containers (label maestro/dd-session) whose names start with prefix."""
    proc = subprocess.run(["docker", "ps", "-a", "--filter", "label=maestro/dd-session=1", "--format", "{{.Names}}"],
                          capture_output=True, text=True, timeout=30)
    names = [n for n in proc.stdout.split() if n.startswith(prefix)]
    for n in names:
        subprocess.run(["docker", "rm", "-f", n], capture_output=True, timeout=60)
    return names
