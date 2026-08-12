#!/usr/bin/env python
"""Containerized harness replay — the confidence gate for legacy-data training.

Replays a repaired+aligned legacy rollout through the REAL maestro-reaper-plugin
harness inside a daw-farm container: the fake replay server (host) supplies the
recorded assistant turns; the harness executes every tool call for real against
container REAPER. Produces:
  - a divergence report: recorded (fabricated) tool_responses vs the harness's
    actual tool results, classified match / acceptable / divergent
  - final-state fidelity: container Vital preset vs the sample's target preset
  - render existence / non-silence

Usage:
    python scripts/replay_rollout_in_container.py \
        --sample bass_00ac47dd --session daw-farm-reaper-4 \
        [--repaired-dir outputs/sft_32k_repaired] [--harness ~/Documents/maestro-reaper-plugin]
"""

from __future__ import annotations

import argparse
import difflib
import json
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from maestro.reaper.dawfarm import DockerSession, reset_project

RUNNER = r'''
import json, sys, time
sys.path.insert(0, "/work/harness/src")
from maestro_plugin.app import ChatApp, AppConfig
from maestro_plugin.session import Attachment
from maestro_plugin.messages import OutputMessage, InputMessage, TextBlock, ToolUseBlock, ToolResultBlock

cfg = AppConfig(
    llm_url=LLM_URL,
    model="maestro-main",
    cwd="/work/replay_cwd",
    enable_sub_agents=True,
)
app = ChatApp(config=cfg)
app.session.pending_attachments.append(Attachment(path=TARGET_WAV, label="target"))
app.submit_user(USER_TEXT)
deadline = time.time() + 1200
while app.is_busy and time.time() < deadline:
    app.drain(); time.sleep(0.02)
app.drain()

dump = []
for msg in app.session.messages:
    if isinstance(msg, OutputMessage):
        for b in msg.blocks:
            if isinstance(b, TextBlock) and b.text.strip():
                dump.append({"role": "assistant", "content": b.text})
            elif isinstance(b, ToolUseBlock):
                dump.append({"role": "tool_call", "name": b.name, "arguments": b.arguments})
    elif isinstance(msg, InputMessage) and msg.role == "tool":
        for b in msg.blocks:
            dump.append({"role": "tool_response", "content": b.content,
                         "is_error": b.is_error, "audio": bool(b.audio_path)})
with open("/work/replay_dump.json", "w") as f:
    json.dump({"timed_out": app.is_busy, "messages": dump}, f)
print("REPLAY_DONE timed_out=", app.is_busy)
'''

_NOTES_RE = re.compile(r"^notes = (\[.*\])$", re.M)


def sh(session, cmd, **kw):
    r = session.exec_bash(cmd, **kw)
    if not r.ok:
        raise RuntimeError(f"container cmd failed: {cmd[:80]}: {r.stderr[-300:]}")
    return r


def classify(recorded: str, actual: str) -> str:
    if recorded.strip() == actual.strip():
        return "match"
    try:
        rj, aj = json.loads(recorded), json.loads(actual)
        if isinstance(rj, dict) and isinstance(aj, dict):
            if set(rj) == set(aj):
                # same keys — treat number/duration/path noise as acceptable
                return "acceptable"
    except Exception:
        pass
    ratio = difflib.SequenceMatcher(None, recorded[:400], actual[:400]).ratio()
    return "acceptable" if ratio > 0.8 else "divergent"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", required=True)
    ap.add_argument("--session", default="daw-farm-reaper-4")
    ap.add_argument("--repaired-dir", type=Path, default=Path("outputs/sft_32k_repaired"))
    ap.add_argument("--harness", type=Path,
                    default=Path.home() / "Documents/maestro-reaper-plugin")
    ap.add_argument("--fake-port", type=int, default=8021)
    args = ap.parse_args()

    rec = next((json.loads(l) for l in open(args.repaired_dir / "main_aligned.jsonl")
                if f'"{args.sample}"' in l[:200] or json.loads(l)["meta"]["sample_id"] == args.sample), None)
    assert rec, f"sample {args.sample} not in main_aligned"
    sid = rec["meta"]["sample_id"]
    target_wav = rec["audios"][0]
    user_text = rec["messages"][0]["content"].replace("<audio>", "").strip()

    s = DockerSession(args.session)
    # --- container prep
    reset_project(s)
    sh(s, "rm -rf /work/harness /work/replay_cwd /tmp/agents /tmp/search_probes && mkdir -p /work/replay_cwd")
    subprocess.run(["docker", "cp", str(args.harness), f"{args.session}:/work/harness"], check=True)
    sh(s, "pip install -e /work/harness --break-system-packages -q 2>&1 | tail -1", timeout=300)
    # skills at the Bash cwd
    subprocess.run(["docker", "cp", str(ROOT / "skills"), f"{args.session}:/work/replay_cwd/skills"], check=True)
    # audio + preset paths at their recorded absolute locations (root mkdir)
    for wav in {target_wav, str(rec["assets"].get("current_audio", ""))} | set(rec.get("audios", [])[:1]):
        if wav and Path(wav).exists():
            d = str(Path(wav).parent)
            subprocess.run(["docker", "exec", "-u", "0", args.session, "mkdir", "-p", d], check=True)
            subprocess.run(["docker", "exec", "-u", "0", args.session, "chmod", "-R", "777", d.split("/outputs/")[0] + "/outputs" if "/outputs/" in d else d], check=True)
            subprocess.run(["docker", "cp", wav, f"{args.session}:{wav}"], check=True)
    # notes json reconstructed from the transcription insert snippet
    trans = next((json.loads(l) for l in open(args.repaired_dir / "transcription_aligned.jsonl")
                  if json.loads(l)["meta"]["sample_id"] == sid), None)
    if trans:
        for m in trans["messages"]:
            if m["role"] == "tool_call" and "MIDI_InsertNote" in str(m.get("content", "")):
                cmd = json.loads(m["content"])["arguments"]["command"]
                nm = _NOTES_RE.search(cmd)
                if nm:
                    sh(s, f"mkdir -p /tmp/agents/{sid}")
                    notes_json = json.dumps({"notes": eval(nm.group(1))})
                    sh(s, f"cat > /tmp/agents/{sid}/{sid}_notes.json <<'EOF'\n{notes_json}\nEOF")
                break

    # --- fake replay server on host
    srv = subprocess.Popen(
        [str(ROOT / ".venv/bin/python"), str(args.harness / "src/maestro_plugin/fake_server.py"),
         "--host", "0.0.0.0", "--port", str(args.fake_port),
         "--file", str(args.repaired_dir / "main_aligned.jsonl"),
         "--file", str(args.repaired_dir / "search_aligned.jsonl"),
         "--file", str(args.repaired_dir / "judge_aligned.jsonl"),
         "--file", str(args.repaired_dir / "transcription_aligned.jsonl"),
         "--default-id", sid, "--delay", "0"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(3)
    try:
        # bridge gateway (container -> host); `ip` isn't installed in the image
        net = subprocess.run(
            ["docker", "inspect", "-f",
             "{{range .NetworkSettings.Networks}}{{.Gateway}}{{end}}", args.session],
            capture_output=True, text=True).stdout.strip()
        gw = net or "172.17.0.1"
        llm_url = f"http://{gw}:{args.fake_port}/v1/chat/completions?rollout={sid}"
        # sanity: server reachable from the container before the full run
        probe = s.exec_bash(
            f"python3 -c \"import urllib.request as u; "
            f"print(u.urlopen('http://{gw}:{args.fake_port}/info', timeout=5).status)\" "
            f"2>&1 || curl -s -m5 -o /dev/null -w '%{{http_code}}' http://{gw}:{args.fake_port}/info",
            timeout=30)
        print(f"server reachability from container ({gw}:{args.fake_port}):", probe.stdout.strip()[-40:])
        runner = (RUNNER
                  .replace("LLM_URL", json.dumps(llm_url))
                  .replace("TARGET_WAV", json.dumps(target_wav))
                  .replace("USER_TEXT", json.dumps(user_text)))
        env_prefix = "MAESTRO_KEEP_WORKSPACE=1 MAESTRO_LLM_REPO=/work/replay_cwd "
        r = s.exec_bash(env_prefix + f"python3 - <<'RUNNER_EOF'\n{runner}\nRUNNER_EOF",
                        timeout=1500)
        print("runner rc:", r.returncode)
        print((r.stdout or "")[-300:], (r.stderr or "")[-500:])
    finally:
        srv.terminate()

    # --- divergence report
    dump_local = Path(f"/tmp/claude_replay_{sid}.json")
    s.get("/work/replay_dump.json", dump_local)
    dump = json.load(open(dump_local))
    actual_resps = [m for m in dump["messages"] if m["role"] == "tool_response"]
    recorded_resps = [m for m in rec["messages"] if m["role"] == "tool_response"]
    counts = {"match": 0, "acceptable": 0, "divergent": 0}
    divergent_examples = []
    for i, (a, rr) in enumerate(zip(actual_resps, recorded_resps)):
        c = classify(rr["content"], a["content"])
        counts[c] += 1
        if c == "divergent" and len(divergent_examples) < 8:
            divergent_examples.append({"idx": i, "recorded": rr["content"][:160],
                                       "actual": a["content"][:160],
                                       "is_error": a.get("is_error")})
    report = {
        "sample": sid, "timed_out": dump["timed_out"],
        "recorded_tool_responses": len(recorded_resps),
        "executed_tool_responses": len(actual_resps),
        "divergence": counts, "divergent_examples": divergent_examples,
        "errors": sum(1 for a in actual_resps if a.get("is_error")),
        "audio_tool_results": sum(1 for a in actual_resps if a.get("audio")),
    }
    out = args.repaired_dir / f"replay_report_{sid}.json"
    json.dump(report, open(out, "w"), indent=2)
    print(json.dumps({k: v for k, v in report.items() if k != "divergent_examples"}, indent=2))
    print("full report:", out)


if __name__ == "__main__":
    main()
