"""
dawfarm.py — client for daw-farm REAPER session containers.

daw-farm (https://github.com/.../daw-farm, local checkout at ~/daw-farm) runs
real, unmodified desktop REAPER instances in containers. There is no network
API: control is `docker exec` / `kubectl exec` into the container, where
 - `reaper-exec` runs Lua ReaScript in the live instance (file-drop queue),
 - the reapy dist API lets Python drive the same instance, and
 - `docker cp` / `kubectl cp` move files (rendered WAVs, MIDI JSON) in/out.

This module wraps that transport in a thread-safe session pool so the SFT
builders can execute their emitted tool-call snippets against a real DAW
instead of fabricating the tool responses.

Spec strings (``--daw-farm``):
    docker                 all running containers with image daw-farm/reaper*
    docker:c1,c2           explicit container names
    k8s                    all Ready pods labelled daw-farm/daw=reaper
    k8s:pod1,pod2          explicit pod names (with or without the daw- prefix)

Sessions map host paths 1:1 onto container paths under /work/rollouts/<sid>
and /tmp/agents/<sid>; renders happen in the container and are fetched back.
"""

from __future__ import annotations

import json
import logging
import queue
import shlex
import subprocess
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_EXEC_TIMEOUT = 180.0
K8S_NAMESPACE = "daw-farm"
ROLLOUT_ROOT = "/work/rollouts"


@dataclass
class ExecResult:
    returncode: int
    stdout: str
    stderr: str

    @property
    def ok(self) -> bool:
        return self.returncode == 0


class DawFarmSession:
    """One live REAPER container. Not thread-safe; owned by one worker at a time."""

    name: str

    def __init__(self, name: str):
        self.name = name
        self._lock = threading.Lock()
        self.vital_data_synced = False

    # -- transport primitives (backend-specific) ----------------------------

    def _exec_argv(self, argv: list[str]) -> list[str]:
        raise NotImplementedError

    def _cp_to(self, host: str, container: str) -> list[str]:
        raise NotImplementedError

    def _cp_from(self, container: str, host: str) -> list[str]:
        raise NotImplementedError

    # -- generic operations --------------------------------------------------

    def exec_argv(self, argv: list[str], stdin: str | None = None,
                  timeout: float = DEFAULT_EXEC_TIMEOUT) -> ExecResult:
        with self._lock:
            try:
                proc = subprocess.run(
                    self._exec_argv(argv), input=stdin, capture_output=True,
                    text=True, timeout=timeout,
                )
            except subprocess.TimeoutExpired as exc:
                return ExecResult(
                    124,
                    (exc.stdout or b"").decode() if isinstance(exc.stdout, bytes) else (exc.stdout or ""),
                    f"timed out after {timeout:.0f}s",
                )
        return ExecResult(proc.returncode, proc.stdout, proc.stderr)

    def exec_bash(self, script: str, cwd: str = "/work",
                  timeout: float = DEFAULT_EXEC_TIMEOUT) -> ExecResult:
        """Run a bash snippet (e.g. an emitted tool-call command) in the container."""
        wrapped = f"cd {shlex.quote(cwd)} || exit 1\n{script}"
        return self.exec_argv(["bash", "-s"], stdin=wrapped, timeout=timeout)

    def exec_python(self, code: str, cwd: str = "/work",
                    timeout: float = DEFAULT_EXEC_TIMEOUT) -> ExecResult:
        return self.exec_bash(f"python3 - <<'PY'\n{code}\nPY", cwd=cwd, timeout=timeout)

    def exec_lua(self, lua: str, timeout: float = DEFAULT_EXEC_TIMEOUT) -> ExecResult:
        """Run Lua in the live REAPER via the daw-farm job queue."""
        return self.exec_argv(["reaper-exec", "-"], stdin=lua, timeout=timeout)

    def put(self, host_path: str | Path, container_path: str) -> None:
        if not container_path.startswith("/"):
            raise ValueError(f"container path must be absolute: {container_path!r}")
        parent = str(Path(container_path).parent)
        res = self.exec_argv(["mkdir", "-p", parent])
        if not res.ok:
            raise RuntimeError(f"{self.name}: mkdir -p {parent} failed: {res.stderr}")
        proc = subprocess.run(self._cp_to(str(host_path), container_path),
                              capture_output=True, text=True, timeout=120)
        if proc.returncode != 0:
            raise RuntimeError(f"{self.name}: cp to {container_path} failed: {proc.stderr}")

    def get(self, container_path: str, host_path: str | Path) -> Path:
        host_path = Path(host_path)
        host_path.parent.mkdir(parents=True, exist_ok=True)
        proc = subprocess.run(self._cp_from(container_path, str(host_path)),
                              capture_output=True, text=True, timeout=120)
        if proc.returncode != 0 or not host_path.exists():
            raise RuntimeError(f"{self.name}: cp from {container_path} failed: {proc.stderr}")
        return host_path

    def get_dir(self, container_dir: str, host_dir: str | Path) -> Path:
        """Fetch a whole directory's contents in one cp (per-file fetching of
        large probe batches dominates wall time otherwise)."""
        host_dir = Path(host_dir)
        host_dir.mkdir(parents=True, exist_ok=True)
        src = container_dir.rstrip("/") + ("/." if isinstance(self, DockerSession) else "")
        proc = subprocess.run(self._cp_from(src, str(host_dir)),
                              capture_output=True, text=True, timeout=600)
        if proc.returncode != 0:
            raise RuntimeError(f"{self.name}: dir cp from {container_dir} failed: {proc.stderr}")
        return host_dir

    def healthy(self, timeout: float = 30.0) -> bool:
        """Queue round-trip + reapy socket (mirrors daw-farm's reaper-ready)."""
        res = self.exec_argv(["reaper-ready"], timeout=timeout)
        if res.ok:
            return True
        logger.warning("session %s unhealthy: %s", self.name, (res.stderr or res.stdout).strip())
        return False

    def wait_for_file(self, container_path: str, timeout: float = 30.0,
                      poll_s: float = 0.5) -> bool:
        """Wait until a file exists in the container with a stable size."""
        deadline = time.monotonic() + timeout
        last_size = -1
        while time.monotonic() < deadline:
            res = self.exec_argv(["stat", "-c", "%s", container_path], timeout=15)
            if res.ok:
                size = int(res.stdout.strip() or 0)
                if size > 0 and size == last_size:
                    return True
                last_size = size
            time.sleep(poll_s)
        return False


class DockerSession(DawFarmSession):
    def _exec_argv(self, argv: list[str]) -> list[str]:
        return ["docker", "exec", "-i", self.name] + argv

    def _cp_to(self, host: str, container: str) -> list[str]:
        return ["docker", "cp", host, f"{self.name}:{container}"]

    def _cp_from(self, container: str, host: str) -> list[str]:
        return ["docker", "cp", f"{self.name}:{container}", host]


    def recycle(self, timeout: float = 180.0) -> None:
        """Restart the container and wait for REAPER readiness — the
        strongest between-rollout hygiene (kills plugin param caches, stray
        renders, leaked project state; 2026-08-15 policy)."""
        import time as _time
        subprocess.run(["docker", "restart", self.name], check=True,
                       capture_output=True, timeout=timeout)
        deadline = _time.time() + timeout
        ready = False
        while _time.time() < deadline:
            if self.healthy(timeout=20.0):
                ready = True
                break
            _time.sleep(3.0)
        if not ready:
            raise RuntimeError(f"{self.name}: not ready {timeout}s after recycle")
        # docker restart keeps the container filesystem — clear rollout litter
        # explicitly (process state, e.g. plugin param caches, died with the
        # restart; /tmp does not).
        self.exec_bash(
            "rm -rf /tmp/agents /tmp/search_probes /tmp/gate && "
            "find /tmp -name 'wt_*.wav' -delete 2>/dev/null; true", timeout=60.0)


class K8sSession(DawFarmSession):
    def __init__(self, name: str, namespace: str = K8S_NAMESPACE):
        super().__init__(name)
        self.namespace = namespace

    def _exec_argv(self, argv: list[str]) -> list[str]:
        return ["kubectl", "-n", self.namespace, "exec", "-i", self.name, "--"] + argv

    def _cp_to(self, host: str, container: str) -> list[str]:
        return ["kubectl", "-n", self.namespace, "cp", host,
                f"{self.name}:{container}", "--retries=3"]

    def _cp_from(self, container: str, host: str) -> list[str]:
        return ["kubectl", "-n", self.namespace, "cp",
                f"{self.name}:{container}", host, "--retries=3"]


# ---------------------------------------------------------------------------
# Discovery + pool
# ---------------------------------------------------------------------------


def _discover_docker() -> list[str]:
    proc = subprocess.run(
        ["docker", "ps", "--format", "{{.Names}}\t{{.Image}}"],
        capture_output=True, text=True, timeout=30,
    )
    names = []
    for line in proc.stdout.splitlines():
        parts = line.split("\t")
        if len(parts) == 2 and parts[1].startswith("daw-farm/reaper"):
            names.append(parts[0])
    return sorted(names)


def _discover_k8s(namespace: str = K8S_NAMESPACE) -> list[str]:
    proc = subprocess.run(
        ["kubectl", "-n", namespace, "get", "pods", "-l", "daw-farm/daw=reaper",
         "-o", "jsonpath={range .items[*]}{.metadata.name} "
         "{.status.conditions[?(@.type=='Ready')].status}{'\\n'}{end}"],
        capture_output=True, text=True, timeout=30,
    )
    names = []
    for line in proc.stdout.splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[1] == "True":
            names.append(parts[0])
    return sorted(names)


class DawFarmPool:
    """Thread-safe pool of healthy sessions; one session per concurrent sample."""

    def __init__(self, sessions: list[DawFarmSession]):
        if not sessions:
            raise RuntimeError("daw-farm pool is empty — no healthy sessions")
        self.sessions = sessions
        self._free: queue.Queue[DawFarmSession] = queue.Queue()
        for s in sessions:
            self._free.put(s)

    @classmethod
    def from_spec(cls, spec: str, namespace: str = K8S_NAMESPACE) -> "DawFarmPool":
        backend, _, rest = spec.partition(":")
        names = [n.strip() for n in rest.split(",") if n.strip()]
        if backend == "docker":
            names = names or _discover_docker()
            candidates: list[DawFarmSession] = [DockerSession(n) for n in names]
        elif backend == "k8s":
            if names:
                names = [n if n.startswith("daw-") else f"daw-{n}" for n in names]
            else:
                names = _discover_k8s(namespace)
            candidates = [K8sSession(n, namespace) for n in names]
        else:
            raise ValueError(f"unknown daw-farm spec {spec!r} (want docker[:...] or k8s[:...])")
        if not candidates:
            raise RuntimeError(f"no daw-farm sessions found for spec {spec!r}")
        healthy = [s for s in candidates if s.healthy()]
        dropped = {s.name for s in candidates} - {s.name for s in healthy}
        if dropped:
            logger.warning("dropping unhealthy daw-farm sessions: %s", sorted(dropped))
        print(f"daw-farm pool: {len(healthy)} healthy session(s): "
              f"{', '.join(s.name for s in healthy)}", flush=True)
        return cls(healthy)

    @contextmanager
    def acquire(self, timeout: float | None = None):
        session = self._free.get(timeout=timeout)
        try:
            yield session
        finally:
            self._free.put(session)


# ---------------------------------------------------------------------------
# Sample-level REAPER operations (infra plumbing, not learned conversation)
# ---------------------------------------------------------------------------


_RESET_LUA = """\
reaper.PreventUIRefresh(1)
for i = reaper.CountTracks(0) - 1, 0, -1 do
  reaper.DeleteTrack(reaper.GetTrack(0, i))
end
reaper.GetSet_LoopTimeRange(true, false, 0, 0, false)
reaper.SetEditCurPos(0, false, false)
reaper.PreventUIRefresh(-1)
reaper.UpdateArrange()
return "reset"
"""

# Standalone Vital chunk builder — same binary layout as the snippet helpers
# in scripts/agent_sft_common.py (VstW/CcnK/FBCh header + JSON + JUCE suffix).
_CHUNK_BUILDER_PY = """\
import base64, json, struct

def build_vital_chunk(preset_json):
    json_bytes = json.dumps(preset_json, separators=(',', ':')).encode('utf-8')
    json_size = len(json_bytes)
    suffix = b'\\x00' * 17 + b'JUCEPrivateData' + b'\\x00' * 8
    total = 184 + json_size + len(suffix)
    prefix = bytearray(184)
    struct.pack_into('<I', prefix, 0, total - 16)
    struct.pack_into('<I', prefix, 4, 1)
    prefix[8:12] = b'VstW'
    struct.pack_into('>I', prefix, 12, 8)
    struct.pack_into('>I', prefix, 16, 1)
    prefix[24:28] = b'CcnK'
    struct.pack_into('>I', prefix, 28, total - 40)
    prefix[32:36] = b'FBCh'
    struct.pack_into('>I', prefix, 36, 2)
    prefix[40:44] = b'Vita'
    struct.pack_into('>I', prefix, 44, 0x00010600)
    struct.pack_into('>I', prefix, 180, json_size + 32)
    return bytes(prefix) + json_bytes + suffix
"""


ASSERT_CLEAN_SNIPPET = '''
import glob, json, os
import reapy
from reapy import reascript_api as RPR
problems = []
with reapy.inside_reaper():
    n_tracks = RPR.CountTracks(0)
    if n_tracks != 0:
        problems.append(f"tracks={n_tracks}")
    bpm = RPR.Master_GetTempo()
    if abs(bpm - 120.0) > 0.01:
        problems.append(f"tempo={bpm}")
for d in ("/tmp/agents", "/tmp/search_probes", "/tmp/gate"):
    if os.path.isdir(d) and os.listdir(d):
        problems.append(f"dirty:{d}")
stray = glob.glob("/tmp/**/wt_*.wav", recursive=True)
if stray:
    problems.append(f"stray_probes={len(stray)}")
print(json.dumps({"clean": not problems, "problems": problems}))
'''


def assert_clean(session: DawFarmSession) -> None:
    """Verify the container is pristine before starting a rollout. Fails
    LOUDLY instead of cleaning: silent cleanup hides state-leak bugs."""
    res = session.exec_bash(f"python3 - <<'PY'\n{ASSERT_CLEAN_SNIPPET}\nPY",
                            timeout=60.0)
    try:
        verdict = json.loads((res.stdout or "").strip().splitlines()[-1])
    except Exception:
        raise RuntimeError(
            f"{session.name}: assert_clean unparseable: {res.stdout[:200]!r} "
            f"{res.stderr[:200]!r}")
    if not verdict.get("clean"):
        raise RuntimeError(
            f"{session.name}: container dirty at rollout start: "
            f"{verdict.get('problems')} — recycle it or fix the leak")


def reset_project(session: DawFarmSession) -> None:
    res = session.exec_lua(_RESET_LUA, timeout=60)
    if not res.ok:
        raise RuntimeError(f"{session.name}: project reset failed: {res.stderr or res.stdout}")


def create_vital_track(session: DawFarmSession, track_name: str = "target_melody") -> None:
    """Infra-level track+Vital creation, for samples whose conversation doesn't emit it."""
    code = (
        "import json\n"
        "import reapy\n"
        "from reapy import reascript_api as RPR\n"
        f"track_name = {json.dumps(track_name)}\n"
        "with reapy.inside_reaper():\n"
        "    RPR.InsertTrackAtIndex(0, True)\n"
        "    track = RPR.GetTrack(0, 0)\n"
        "    RPR.GetSetMediaTrackInfo_String(track, 'P_NAME', track_name, True)\n"
        "    fx = RPR.TrackFX_AddByName(track, 'Vital', False, 1)\n"
        "    assert fx >= 0, 'Vital not found'\n"
        "print(json.dumps({'status': 'ok', 'fx': True}))\n"
    )
    res = session.exec_python(code, timeout=120)
    if not res.ok:
        raise RuntimeError(f"{session.name}: create_vital_track failed: {res.stderr}")


def apply_vital_preset(session: DawFarmSession, preset: dict,
                       track_idx: int = 0, fx_idx: int = 0) -> None:
    """Load a full Vital preset dict into the live instance via the chunk API."""
    preset_json = json.dumps(preset, separators=(",", ":"))
    code = (
        _CHUNK_BUILDER_PY
        + "import reapy\n"
        "from reapy import reascript_api as RPR\n"
        f"preset = json.loads({preset_json!r})\n"
        "chunk = build_vital_chunk(preset)\n"
        "encoded = base64.b64encode(chunk).decode('ascii')\n"
        "with reapy.inside_reaper():\n"
        f"    track = RPR.GetTrack(0, {track_idx})\n"
        f"    if not RPR.TrackFX_SetNamedConfigParm(track, {fx_idx}, 'vst_chunk', encoded):\n"
        f"        RPR.TrackFX_SetNamedConfigParm(track, {fx_idx}, 'vst3_chunk', encoded)\n"
        "print('applied')\n"
    )
    res = session.exec_python(code, timeout=120)
    if not res.ok:
        raise RuntimeError(f"{session.name}: apply_vital_preset failed: {res.stderr}")


def insert_midi_notes(session: DawFarmSession, notes: list[dict],
                      track_idx: int = 0) -> None:
    """Insert notes ({pitch, velocity, start_s, dur_s}) as one MIDI item.

    This is the project-state effect the transcription subagent produces at
    inference time; the builder applies it directly so subsequent timeline
    renders are audible.
    """
    end_s = max((n["start_s"] + n["dur_s"] for n in notes), default=4.0)
    notes_json = json.dumps([
        {"pitch": int(n["pitch"]), "velocity": int(n["velocity"]),
         "start_s": float(n["start_s"]), "dur_s": float(n["dur_s"])}
        for n in notes
    ])
    code = (
        "import json\n"
        "import reapy\n"
        "from reapy import reascript_api as RPR\n"
        f"notes = json.loads({notes_json!r})\n"
        f"end_s = {end_s!r}\n"
        "with reapy.inside_reaper():\n"
        f"    track = RPR.GetTrack(0, {track_idx})\n"
        "    RPR.CreateNewMIDIItemInProj(track, 0.0, end_s, False)\n"
        "    # reapy's RPR wrappers return the full arg list, not the created\n"
        "    # item pointer — re-fetch the item from the track instead.\n"
        "    item = RPR.GetTrackMediaItem(track, RPR.GetTrackNumMediaItems(track) - 1)\n"
        "    take = RPR.GetActiveTake(item)\n"
        "    for n in notes:\n"
        "        sp = RPR.MIDI_GetPPQPosFromProjTime(take, n['start_s'])\n"
        "        ep = RPR.MIDI_GetPPQPosFromProjTime(take, n['start_s'] + n['dur_s'])\n"
        "        RPR.MIDI_InsertNote(take, False, False, sp, ep, 0, n['pitch'], n['velocity'], True)\n"
        "    RPR.MIDI_Sort(take)\n"
        "    n_inserted = RPR.MIDI_CountEvts(take, 0, 0, 0)[2]\n"
        "assert n_inserted == len(notes), f'inserted {n_inserted}/{len(notes)} notes'\n"
        "print(json.dumps({'status': 'ok', 'n_notes': n_inserted}))\n"
    )
    res = session.exec_python(code, timeout=120)
    if not res.ok:
        raise RuntimeError(f"{session.name}: insert_midi_notes failed: {res.stderr}")


def sync_vital_data(session: DawFarmSession, host_dir: str | Path | None) -> None:
    """Push the host's Vital data dir (wavetables) so the container library
    matches the one preset generation sampled from. No-op if already synced
    or *host_dir* is None/missing (the image bakes a copy at build time)."""
    if session.vital_data_synced or not host_dir:
        return
    host_dir = Path(host_dir).expanduser()
    if not host_dir.is_dir():
        logger.warning("vital data dir %s missing; relying on image-baked copy", host_dir)
        session.vital_data_synced = True
        return
    session.exec_argv(["rm", "-rf", "/home/daw/.local/share/vital"])
    session.exec_argv(["mkdir", "-p", "/home/daw/.local/share"])
    proc = subprocess.run(session._cp_to(str(host_dir), "/home/daw/.local/share/vital"),
                          capture_output=True, text=True, timeout=300)
    if proc.returncode != 0:
        raise RuntimeError(f"{session.name}: vital data sync failed: {proc.stderr}")
    session.vital_data_synced = True


def set_project_tempo(session: DawFarmSession, bpm: float) -> None:
    """Match REAPER's project tempo to the sample's assumed BPM.

    Vital takes tempo from the host — the preset's beats_per_minute key is
    ignored inside REAPER — so tempo-synced LFOs/delays only match the GT
    render if the project tempo agrees with the preset's assumption.
    """
    res = session.exec_lua(f"reaper.SetCurrentBPM(0, {float(bpm)!r}, true)\nreturn 'ok'", timeout=60)
    if not res.ok:
        raise RuntimeError(f"{session.name}: set_project_tempo failed: {res.stderr or res.stdout}")


_REAPER_RENDER_PY = """\
import json, os, time
import reapy
from reapy import reascript_api as RPR
out_path = {out_path!r}
os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
if os.path.isfile(out_path):
    os.remove(out_path)  # existing file triggers REAPER's overwrite prompt
with reapy.inside_reaper():
    proj = RPR.EnumProjects(-1, '', 512)[0]
    track = RPR.GetTrack(0, 0)
    end = 0.0
    for i in range(RPR.GetTrackNumMediaItems(track)):
        item = RPR.GetTrackMediaItem(track, i)
        end = max(end, RPR.GetMediaItemInfo_Value(item, 'D_POSITION')
                  + RPR.GetMediaItemInfo_Value(item, 'D_LENGTH'))
    RPR.GetSet_LoopTimeRange(True, False, 0.0, end + 1.0, False)
    RPR.GetSetProjectInfo_String(proj, 'RENDER_FILE', os.path.dirname(out_path), True)
    RPR.GetSetProjectInfo_String(proj, 'RENDER_PATTERN',
                                 os.path.splitext(os.path.basename(out_path))[0], True)
    RPR.Main_OnCommand(42230, 0)
prev = -1
for _ in range(240):
    alt = out_path[:-len('.wav')] + '-001.wav'
    if not os.path.isfile(out_path) and os.path.isfile(alt):
        os.rename(alt, out_path)
    if os.path.isfile(out_path):
        size = os.path.getsize(out_path)
        if size > 44 and size == prev:
            break
        prev = size
    time.sleep(0.25)
assert os.path.isfile(out_path), 'render did not appear'
print('rendered')
"""


def render_preset_in_reaper(
    session: DawFarmSession,
    preset: dict,
    notes: list[dict],
    host_out: str | Path,
    tag: str = "_stage_a",
) -> Path:
    """Render *preset* + *notes* through the exact rollout environment.

    Same path a rollout's listen renders take: live REAPER project, Vital
    VST3 chunk load, project tempo from the preset, MIDI item on the
    timeline, offline render via action 42230 into the master mix. Used to
    produce ground-truth/baseline audio so training targets come from the
    environment the model acts in, not from a different engine.
    """
    host_out = Path(host_out)
    reset_project(session)
    set_project_tempo(session, 60.0 * float(preset.get("settings", {}).get("beats_per_minute", 2.0)))
    create_vital_track(session)
    apply_vital_preset(session, preset)
    insert_midi_notes(session, notes)
    container_out = f"{ROLLOUT_ROOT}/{tag}/{host_out.name}"
    res = session.exec_python(_REAPER_RENDER_PY.format(out_path=container_out), timeout=300)
    if not res.ok:
        raise RuntimeError(f"{session.name}: env render failed: {res.stderr[-1500:]}")
    return session.get(container_out, host_out)


def rollout_dir(sample_id: str) -> str:
    return f"{ROLLOUT_ROOT}/{sample_id}"


def prepare_sample_dirs(session: DawFarmSession, sample_id: str) -> str:
    """Start-of-rollout workspace prep for a (reused) session.

    Rollout hygiene is pull-based — each rollout cleans at acquire time
    rather than trusting its predecessor to have cleaned at release:
    - reset_project() deletes all tracks, which destroys the FX instances
      and with them any REAPER-cached VST3 parameter state from the
      previous rollout;
    - this purge drops previous rollouts' renders and agent files so a new
      sample can never read or fetch stale artifacts (and disk usage stays
      bounded on long-lived sessions).
    Leaving the previous rollout's files in place until now (instead of
    deleting at release) keeps the last run inspectable for debugging.
    """
    rd = rollout_dir(sample_id)
    res = session.exec_bash(
        f"rm -rf {ROLLOUT_ROOT}/* /tmp/agents/* && mkdir -p {rd} /tmp/agents/{sample_id}"
    )
    if not res.ok:
        raise RuntimeError(f"{session.name}: rollout dir prep failed: {res.stderr}")
    return rd
