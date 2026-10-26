"""Replay records through the real OpenCode 1.19.3 binary and diff what it sends against the records.

Execution model: one throwaway daw-farm container per replay (`wsd-replay-<run>`, `--network none`)
runs the mock server, OpenCode and every tool OpenCode executes. Bash, read, write, glob... all
act on the container's filesystem, where the rollout workdir is /work/rollouts/<sid> exactly as in
maestro.dawfarm_dd.DawDreamerSession, and the skill is reachable at its model-visible path
/tmp/maestro-oc/config/skills/vital. So the paths OpenCode puts in prompts and tool results equal
the ones in the records, and skill scripts run in the same image the records were built in.

Mounts:
  <run>/oc_root           -> /tmp/maestro-oc            staging root of opencode/README.md (rw)
  <run>/rollouts          -> /work/rollouts             rollout workdirs (rw)
  <repo>/opencode         -> <repo>/opencode (ro)       targets of the staging symlinks
  ~/.opencode/bin/opencode-> /opt/opencode/bin/opencode (ro)
  rg 15.1.0               -> /usr/local/bin/rg (ro)     ahead of the image's /usr/bin/rg 14.1.0 on PATH
  <repo>/opencode/replay  -> /opt/replay (ro)           mock server
  <run>/io                -> /replay                    scenario, request log, OpenCode output

The mock listens on the container's loopback; nothing can reach the network.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any

from . import audio_eq, records as R
from .records import unified

REPO = Path(__file__).resolve().parents[2]
OPENCODE_BIN = Path(os.environ.get("OPENCODE_BIN", Path.home() / ".opencode/bin/opencode"))
RG_BIN = Path(os.environ.get("RG_BIN", REPO / "outputs/replay/bin/rg-15.1.0"))
RG_SHA256 = "ebeaf56f8a25e102e9419933423738b3a2a613a444fd749d695e15eba53f71f2"  # ripgrep 15.1.0 musl
IMAGE = "daw-farm/reaper:latest"
OC_ROOT = "/tmp/maestro-oc"
ROLLOUT_ROOT = "/work/rollouts"
CONTAINER_PREFIX = "wsd-replay-"
STAGE_SKIP = {"replay", "README.md", "__pycache__"}  # opencode/ entries that are not OpenCode config
GOLDEN_PARAMS = REPO / "docs/opencode_capture/goldens/request_params.json"
# Same as maestro.dawfarm_dd (container-wide there; OpenCode's env -i drops them, so pass explicitly).
SKILL_ENV = {"DISPLAY": ":99", "VITAL_VST3": "/usr/lib/vst3/Vital.vst3",
             "VITAL_SKILL": f"{OC_ROOT}/config/skills/vital", "PYTHONDONTWRITEBYTECODE": "1"}
_BOOT = ("while true; do rm -f /tmp/.X99-lock /tmp/.X11-unix/X99; "
         "Xvfb :99 -screen 0 1024x768x24 -nolisten tcp >/dev/null 2>&1; sleep 1; done & "
         "exec sleep infinity")


def _sha256(p: Path) -> str:
    import hashlib
    return hashlib.sha256(p.read_bytes()).hexdigest()


def stage_root(root: Path, with_plugin: bool = True) -> None:
    """opencode/README.md staging: config dir of symlinks into the repo, HOME/XDG under the root,
    ripgrep seeded into the cache (also mounted on PATH). Every top-level entry of opencode/ is staged
    (opencode.json, agent/, skills/, plugin/, AGENTS.md or whatever the config grows) except the
    replay harness and docs; plugin/ only with_plugin."""
    if root.exists():
        shutil.rmtree(root)
    for d in ("config", "home", "data", "cache/opencode/bin"):
        (root / d).mkdir(parents=True)
    skip = STAGE_SKIP | (set() if with_plugin else {"plugin"})
    for src in sorted((REPO / "opencode").iterdir()):
        if src.name not in skip and not src.name.startswith("."):
            os.symlink(src, root / "config" / src.name)
    shutil.copy(RG_BIN, root / "cache/opencode/bin/rg")


def opencode_env(port: int, model: str) -> dict[str, str]:
    return {
        "PATH": "/usr/local/bin:/usr/bin:/bin", "LANG": "C.UTF-8",
        "HOME": f"{OC_ROOT}/home", "XDG_CONFIG_HOME": f"{OC_ROOT}/home/.config",
        "XDG_DATA_HOME": f"{OC_ROOT}/data", "XDG_CACHE_HOME": f"{OC_ROOT}/cache",
        "XDG_STATE_HOME": f"{OC_ROOT}/home/.local/state",
        "OPENCODE_CONFIG_DIR": f"{OC_ROOT}/config",
        "OPENCODE_DISABLE_AUTOUPDATE": "1", "OPENCODE_DISABLE_MODELS_FETCH": "1",
        "OPENCODE_DISABLE_LSP_DOWNLOAD": "1", "OPENCODE_DISABLE_CLAUDE_CODE": "1",
        "OPENCODE_DISABLE_EXTERNAL_SKILLS": "1",
        "NPM_CONFIG_REGISTRY": "http://127.0.0.1:9/",
        "MAESTRO_BASE_URL": f"http://127.0.0.1:{port}/v1", "MAESTRO_MODEL": model,
        **SKILL_ENV,
    }


def _system_field(system: str, rx: str, default: str | None = None) -> str | None:
    m = re.search(rx, system)
    return m.group(1) if m else default


def _docker(*args: str, **kw: Any) -> subprocess.CompletedProcess:
    return subprocess.run(["docker", *args], capture_output=True, text=True, **kw)


def replay(main: dict[str, Any], subagents: list[dict[str, Any]], run_dir: str | Path, *,
           init_dir: str | Path | None = None, prompt: str | None = None,
           attach: list[str] | None = None, port: int = 8911, timeout_s: int = 1800,
           with_plugin: bool = True, keep_container: bool = False,
           base_dir: str | Path | None = None, expected_params: dict[str, Any] | None = None,
           extra_records: dict[str, dict[str, Any]] | None = None, max_mel_db: float = audio_eq.DEFAULT_MAX_DB,
           final_dir: str | Path | None = None) -> dict[str, Any]:
    """Run one replay and write <run_dir>/report.json. Returns the report.

    main/subagents: records (ms-swift dicts). init_dir: files to place in the workdir before the run
    (the rollout's initial state). prompt: user text sent on stdin (default: the main record's first
    user message with media placeholders removed). attach: container paths passed as `-f`.
    base_dir resolves relative `audios` paths. max_mel_db: audio-equivalence threshold for re-rendered
    clips (audio_eq); clips whose bytes were in init_dir must match exactly. final_dir: the rollout's
    final workdir, compared with the replay's (compare_workdirs)."""
    run_dir = Path(run_dir).resolve()
    if not RG_BIN.is_file() or _sha256(RG_BIN) != RG_SHA256:
        raise RuntimeError(f"ripgrep 15.1.0 missing or wrong at {RG_BIN} (set RG_BIN)")
    records = {"main": main}
    for i, s in enumerate(subagents):
        records[s.get("id") or f"sub{i}"] = s
    records.update(extra_records or {})
    sysm = main["messages"][0]["content"]
    workdir = _system_field(sysm, r"Working directory: (\S+)")
    model = _system_field(sysm, r"You are powered by the model named (\S+)\.", "maestro")
    if not workdir or not workdir.startswith(ROLLOUT_ROOT + "/"):
        raise ValueError(f"record workdir {workdir!r} is not under {ROLLOUT_ROOT}/")
    sid = workdir[len(ROLLOUT_ROOT) + 1:]
    if "/" in sid:
        raise ValueError(f"nested rollout workdir {workdir!r}")

    if run_dir.exists():
        shutil.rmtree(run_dir)
    io, rollouts, root = run_dir / "io", run_dir / "rollouts", run_dir / "oc_root"
    io.mkdir(parents=True)
    stage_root(root, with_plugin)
    wd_host = rollouts / sid
    if init_dir:
        shutil.copytree(init_dir, wd_host, symlinks=True)
    else:
        wd_host.mkdir(parents=True)
    (io / "scenario.json").write_text(json.dumps(R.build_scenario(records), ensure_ascii=False))
    (io / "records.json").write_text(json.dumps(records, ensure_ascii=False))
    if prompt is None:
        fu = R.first_user_text(main["messages"]) or ""
        prompt = fu.replace(R.AUDIO_PLACEHOLDER, "").replace(R.IMAGE_PLACEHOLDER, "")
    (io / "prompt.txt").write_text(prompt)
    meta = {"init_shas": sorted(_tree_shas(init_dir)) if init_dir else [], "max_mel_db": max_mel_db,
            "base_dir": str(base_dir) if base_dir else None}
    (io / "meta.json").write_text(json.dumps(meta, indent=1))

    name = CONTAINER_PREFIX + run_dir.name.replace("_", "-")[:40] + "-" + uuid.uuid4().hex[:6]
    argv = ["run", "-d", "--init", "--name", name, "--user", f"{os.getuid()}:{os.getgid()}",
            "--network", "none", "--label", "maestro/wsd-replay=1",
            "-v", f"{rollouts}:{ROLLOUT_ROOT}", "-v", f"{root}:{OC_ROOT}",
            "-v", f"{REPO / 'opencode'}:{REPO / 'opencode'}:ro",
            "-v", f"{OPENCODE_BIN}:/opt/opencode/bin/opencode:ro",
            "-v", f"{RG_BIN}:/usr/local/bin/rg:ro",
            "-v", f"{Path(__file__).resolve().parent}:/opt/replay:ro",
            "-v", f"{io}:/replay", "-w", ROLLOUT_ROOT, "--entrypoint", "bash", IMAGE, "-c", _BOOT]
    p = _docker(*argv, timeout=120)
    if p.returncode != 0:
        raise RuntimeError(f"docker run failed: {p.stderr.strip()}")
    t0 = time.time()
    try:
        _docker("exec", "-d", name, "python3", "-B", "/opt/replay/mock_server.py", "/replay/scenario.json",
                "/replay/requests.jsonl", str(port), timeout=60)
        probe = f"import urllib.request as u; u.urlopen('http://127.0.0.1:{port}/v1/models', timeout=1)"
        for _ in range(100):
            if _docker("exec", name, "python3", "-c", probe, timeout=30).returncode == 0:
                break
            time.sleep(0.2)
        else:
            raise RuntimeError("mock server did not come up")
        env = [f"{k}={v}" for k, v in opencode_env(port, model).items()]
        cmd = ["exec", "-i", "-w", workdir, name, "env", "-i", *env, "timeout", str(timeout_s),
               "/opt/opencode/bin/opencode", "run", "--agent", "maestro", "--format", "json"]
        for a in attach or []:
            cmd += ["-f", a]
        with open(io / "prompt.txt", "rb") as fin, open(io / "run.jsonl", "wb") as fout, \
                open(io / "run.err", "wb") as ferr:
            rc = subprocess.run(["docker", *cmd], stdin=fin, stdout=fout, stderr=ferr,
                                timeout=timeout_s + 60).returncode
    finally:
        if not keep_container:
            _docker("rm", "-f", name, timeout=60)
    elapsed = time.time() - t0
    report = build_report(records, io / "requests.jsonl", base_dir=base_dir, expected_params=expected_params,
                          prompt=prompt, init_shas=set(meta["init_shas"]), max_mel_db=max_mel_db)
    if final_dir:
        wd = compare_workdirs(Path(final_dir), wd_host, max_mel_db)
        report["workdir_check"] = {k: v for k, v in wd.items() if k != "diffs"}
        if wd["diffs"]:
            report["rows"].append({"seq": None, "route": "workdir", "kind": "workdir", "step": None,
                                   "status": "diff", "diffs": wd["diffs"]})
            report["diffs"] += 1
            for d in wd["diffs"]:
                report["by_kind"][d["kind"]] = report["by_kind"].get(d["kind"], 0) + 1
    report.update({"opencode_exit": rc, "elapsed_s": round(elapsed, 1), "container": name,
                   "workdir": workdir, "run_dir": str(run_dir)})
    (run_dir / "report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False))
    (run_dir / "report.md").write_text(report_markdown(report))
    return report


def _tree_shas(d: str | Path) -> set[str]:
    return {R.file_digest(str(p))["sha256"] for p in Path(d).rglob("*") if p.is_file() and not p.is_symlink()}


def _audio_paths(record: dict[str, Any], base_dir: str | Path | None) -> list[Path]:
    out = []
    for a in record.get("audios") or []:
        p = Path(a)
        if not p.is_absolute() and base_dir:
            p = Path(base_dir) / p
        out.append(p)
    return out


AUDIO_EXT = (".wav", ".flac", ".mp3", ".ogg")


def compare_workdirs(orig: Path, replayed: Path, max_mel_db: float = audio_eq.DEFAULT_MAX_DB) -> dict[str, Any]:
    """The rollout's final workdir vs the replay's. Same file set and byte-identical non-audio files
    (preset.json, notes.json, probe/judge JSON...) are required; audio files are compared with audio_eq
    and reported (not failed: renders are not bit-identical and same-preset noise can exceed the
    threshold, see README)."""
    def files(d: Path) -> dict[str, Path]:
        return {str(p.relative_to(d)): p for p in d.rglob("*") if p.is_file()}
    a, b = files(orig), files(replayed)
    diffs: list[dict[str, Any]] = []
    if set(a) != set(b):
        diffs.append({"kind": "workdir_files", "detail": {"missing_in_replay": sorted(set(a) - set(b)),
                                                          "extra_in_replay": sorted(set(b) - set(a))}})
    audio, changed = {}, []
    for rel in sorted(set(a) & set(b)):
        if R.file_digest(str(a[rel]))["sha256"] == R.file_digest(str(b[rel]))["sha256"]:
            continue
        if rel.lower().endswith(AUDIO_EXT):
            audio[rel] = audio_eq.compare(a[rel], b[rel], max_db=max_mel_db)
        else:
            changed.append(rel)
    if changed:
        diffs.append({"kind": "workdir_content", "detail": {
            rel: unified(a[rel].read_text(errors="replace").split("\n"), b[rel].read_text(errors="replace").split("\n"),
                         "rollout", "replay", max_lines=30) for rel in changed}})
    return {"files": len(a), "identical": len(a) - len(audio) - len(changed) - len(set(a) - set(b)),
            "audio_rerendered": {k: v.get("mel_db") for k, v in audio.items()},
            "audio_over_threshold": sorted(k for k, v in audio.items() if not v.get("equivalent")),
            "diffs": diffs}


def expected_title_request(prompt: str, model: str) -> dict[str, Any] | None:
    """goldens/title_request.json with the replay's prompt and model (OpenCode sends the user's text)."""
    g = GOLDEN_PARAMS.parent / "title_request.json"
    if not g.is_file():
        return None
    t = json.loads(g.read_text())
    assert t["messages"][-1]["role"] == "user"
    t["messages"][-1] = {"role": "user", "content": prompt}
    t["model"] = model
    return t


def _title_row(body: dict[str, Any], expected: dict[str, Any] | None) -> dict[str, Any]:
    if expected is None:
        return {"status": "skipped"}
    if body == expected:
        return {"status": "match", "diffs": []}
    detail = unified(json.dumps(expected, indent=1, ensure_ascii=False).split("\n"),
                     json.dumps(body, indent=1, ensure_ascii=False).split("\n"), "golden", "opencode")
    return {"status": "diff", "diffs": [{"kind": "title", "detail": detail}]}


def build_report(records: dict[str, dict[str, Any]], log: Path, base_dir: str | Path | None = None,
                 expected_params: dict[str, Any] | None = None, prompt: str | None = None,
                 init_shas: set[str] | None = None, max_mel_db: float = audio_eq.DEFAULT_MAX_DB) -> dict[str, Any]:
    """Diff every logged request against its record. expected_params defaults to the golden
    request_params.json with the main record's model name; the title request is diffed against the
    golden title request with `prompt` (default <log dir>/prompt.txt). A media item whose sha256 differs
    passes as re-rendered audio (audio_eq, saved bytes in <log dir>/media) unless its record clip's
    bytes are among init_shas (files of the initial workdir: exact match required)."""
    entries = [json.loads(line) for line in open(log)] if log.exists() else []
    main = records.get("main") or next(iter(records.values()))
    model = _system_field(main["messages"][0]["content"], r"You are powered by the model named (\S+)\.", "maestro")
    if expected_params is None and GOLDEN_PARAMS.is_file():
        expected_params = {**json.loads(GOLDEN_PARAMS.read_text()), "model": model}
    if prompt is None and (log.parent / "prompt.txt").is_file():
        prompt = (log.parent / "prompt.txt").read_text()
    title = expected_title_request(prompt, model) if prompt is not None else None
    init_shas = init_shas or set()
    paths = {k: _audio_paths(r, base_dir) for k, r in records.items()}
    digests = {k: [R.file_digest(str(p)) for p in ps] for k, ps in paths.items()}
    media_dir = log.parent / "media"

    def media_eq_for(key: str):
        def eq(i: int, got_sha: str) -> dict[str, Any]:
            if digests[key][i]["sha256"] in init_shas:
                return {"equivalent": False, "reason": "clip from the initial workdir: exact match required"}
            got = sorted(media_dir.glob(got_sha + ".*"))
            if not got:
                return {"equivalent": False, "reason": "replayed media bytes not saved"}
            return audio_eq.compare(paths[key][i], got[0], max_db=max_mel_db)
        return eq

    rows, seen = [], {k: set() for k in records}
    for e in entries:
        row = {"seq": e["seq"], "route": e["route"], "kind": e["kind"], "step": e["step"]}
        if e["kind"] == "title":
            row.update(_title_row(e["body"], title))
        elif e["route"] is None or e.get("error"):
            row.update(status="diff", diffs=[{"kind": "routing", "detail": e.get("error") or e["kind"]}])
        else:
            res = R.compare_request(e["body"], records[e["route"]], e["step"], digests[e["route"]],
                                    expected_params, media_eq_for(e["route"]))
            row.update(res)
            if e["kind"] == "record_fallback":
                row["diffs"].append({"kind": "routing", "detail": "first user message matched no record "
                                                                 "exactly; routed by system prompt only"})
                row["status"] = "diff"
            seen[e["route"]].add(e["step"])
        rows.append(row)
    coverage = {}
    for k, r in records.items():
        n = len(R.record_steps(r["messages"]))
        missing = sorted(set(range(n)) - seen[k])
        coverage[k] = {"steps": n, "requested": len(seen[k]), "missing": missing}
        if missing:
            rows.append({"seq": None, "route": k, "kind": "coverage", "step": missing[0], "status": "diff",
                         "diffs": [{"kind": "coverage", "detail": f"steps never requested: {missing}"}]})
    by_kind: dict[str, int] = {}
    for r in rows:
        for d in r.get("diffs", []):
            by_kind[d["kind"]] = by_kind.get(d["kind"], 0) + 1
        for w in r.get("whitelisted", []):
            by_kind["whitelisted:" + w["kind"]] = by_kind.get("whitelisted:" + w["kind"], 0) + 1
    n_diff = sum(1 for r in rows if r["status"] == "diff")
    return {"requests": len(entries), "compared": sum(1 for r in rows if r["status"] in ("match", "diff")),
            "matches": sum(1 for r in rows if r["status"] == "match"), "diffs": n_diff, "by_kind": by_kind,
            "max_mel_db": max_mel_db, "whitelist": R.WHITELIST, "coverage": coverage, "rows": rows}


def report_markdown(rep: dict[str, Any]) -> str:
    lines = [f"# Replay report", "",
             f"- requests: {rep['requests']}, compared: {rep['compared']}, match: {rep['matches']}, "
             f"diff: {rep['diffs']}",
             f"- diff/whitelist items by kind: {json.dumps(rep.get('by_kind', {}))}",
             f"- opencode exit: {rep.get('opencode_exit')}, elapsed: {rep.get('elapsed_s')} s",
             f"- workdir: {rep.get('workdir')}"]
    if rep.get("workdir_check"):
        w = rep["workdir_check"]
        lines.append(f"- final workdir vs rollout: {w['files']} files, {w['identical']} byte-identical, "
                     f"re-rendered audio mel dB {json.dumps(w['audio_rerendered'])}")
    lines += ["", "## Coverage", ""]
    for k, c in rep["coverage"].items():
        lines.append(f"- `{k}`: {c['requested']}/{c['steps']} steps requested"
                     + (f", missing {c['missing']}" if c["missing"] else ""))
    lines += ["", "## Requests", ""]
    for r in rep["rows"]:
        lines.append(f"- #{r['seq']} `{r['route']}` step {r['step']} ({r['kind']}): **{r['status']}**")
        for d in r.get("diffs", []):
            det = d["detail"] if isinstance(d["detail"], str) else json.dumps(d["detail"], ensure_ascii=False)
            lines += [f"  - {d['kind']}:", "", "```diff", det, "```", ""]
        for w in r.get("whitelisted", []):
            lines.append(f"  - whitelisted {w['kind']}: {json.dumps(w, ensure_ascii=False)}")
    lines += ["", "## Whitelisted differences", ""] + [f"- {k}: {v}" for k, v in rep["whitelist"].items()]
    return "\n".join(lines) + "\n"
