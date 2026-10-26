#!/usr/bin/env python3
"""Normalize a capture.sh sandbox into the golden fixtures next to this dir.

    python docs/opencode_capture/goldens/capture/extract.py SANDBOX_DIR            # stock run
    python docs/opencode_capture/goldens/capture/extract.py SANDBOX_DIR with_plugin  # WITH_PLUGIN=1 run

A plugin run writes the request-level goldens into goldens/with_plugin/ and
leaves the stock goldens and the agent/skill snapshots alone. Base64 media is
checked against the file it came from and replaced by {B64:<file name>}.

Volatile values are replaced by placeholders (documented in meta.json):
  {WORKDIR} session cwd, {CONFIG_DIR} OPENCODE_CONFIG_DIR, {DATA_DIR}
  $XDG_DATA_HOME/opencode, {TOOL_OUTPUT_ID} / {SES_ID} random ids, {DATE}
  `Date.toDateString()`, {PORT} the mock port.
"""
import json
import os
import re
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SUBDIR = sys.argv[2] if len(sys.argv) > 2 else ""
GOLD = os.path.join(ROOT, SUBDIR)
sys.path.insert(0, HERE)
sys.dont_write_bytecode = True
from mock_scripted import SUB_FINAL, main_script  # noqa: E402

SB = os.path.abspath(sys.argv[1])
PORT = open(os.path.join(SB, "port")).read().strip()
WD = os.path.join(SB, "work")
INIT = os.path.join(SB, "work_initial")
SUBAGENTS = ("transcriber", "searcher", "judge")


def b64_file(data):
    """{B64:<name>} for media that is byte-identical to a work file."""
    import base64
    raw = base64.b64decode(data)
    for fn in sorted(os.listdir(INIT)):
        p = os.path.join(INIT, fn)
        if os.path.isfile(p) and open(p, "rb").read() == raw:
            return "{B64:" + fn + "}"
    raise AssertionError("media part matches no work file")


def norm(x):
    if isinstance(x, dict) and x.get("type") == "input_audio":
        return {**x, "input_audio": {**x["input_audio"], "data": b64_file(x["input_audio"]["data"])}}
    if isinstance(x, dict) and x.get("type") == "image_url":
        head, data = x["image_url"]["url"].split(",", 1)
        return {**x, "image_url": {**x["image_url"], "url": head + "," + b64_file(data)}}
    if isinstance(x, str):
        x = x.replace(os.path.join(SB, "home/.local/share/opencode"), "{DATA_DIR}")
        x = x.replace(os.path.join(SB, "cfg"), "{CONFIG_DIR}")
        x = x.replace(WD, "{WORKDIR}")
        x = re.sub(r"tool_[0-9a-f]{12}[0-9A-Za-z]{14}", "{TOOL_OUTPUT_ID}", x)
        x = re.sub(r"ses_[0-9a-f]{12}[0-9A-Za-z]{14}", "{SES_ID}", x)
        x = re.sub(r"Today's date: \w{3} \w{3} \d{2} \d{4}", "Today's date: {DATE}", x)
        return x.replace(f"127.0.0.1:{PORT}", "127.0.0.1:{PORT}")
    if isinstance(x, list):
        return [norm(v) for v in x]
    if isinstance(x, dict):
        return {k: norm(v) for k, v in x.items()}
    return x


def dump(name, obj):
    path = os.path.join(GOLD, name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        if isinstance(obj, str):
            f.write(obj)
        else:
            json.dump(obj, f, indent=1, ensure_ascii=False)
            f.write("\n")


recs = [json.loads(line) for line in open(os.path.join(SB, "captured.jsonl"))]
last = {}
first = {}
for r in recs:
    last[r["agent"]] = r["body"]
    first.setdefault(r["agent"], r["body"])
main = last["maestro"]

# -- request shape, tools, system prompts ------------------------------------
dump("request_params.json", {k: v for k, v in main.items() if k not in ("messages", "tools")})
dump("tools.json", main["tools"])
dump("tools_subagent.json", first["searcher"]["tools"])
for a in ("maestro",) + SUBAGENTS:
    assert all(r["body"].get("tools") == first[a]["tools"] for r in recs if r["agent"] == a)
    dump(f"system/{a}.txt", norm(first[a]["messages"][0]["content"]))
dump("conversation_maestro.json", norm(main["messages"][1:]))
dump("conversation_subagents.json", {a: norm(last[a]["messages"][1:]) for a in SUBAGENTS})
dump("title_request.json", norm(first["other"]))

# -- tool results ---------------------------------------------------------------
calls, results = {}, {}
for m in main["messages"]:
    for tc in m.get("tool_calls") or []:
        calls[tc["id"]] = tc["function"]
    if m["role"] == "tool":
        results[m["tool_call_id"]] = m["content"]


def initial(path):
    p = path.replace(WD, INIT)
    if os.path.isdir(p):
        return {"entries": sorted((e + "/" if os.path.isdir(os.path.join(p, e)) else e) for e in os.listdir(p))}
    if os.path.isfile(p):
        raw = open(p, "rb").read()
        try:
            return {"content": raw.decode()}
        except UnicodeDecodeError:
            return {"binary": True}
    return {"missing": True}


cases = []
for step, (case, step_calls) in enumerate(main_script(WD, PORT)):
    for i, (tool, args) in enumerate(step_calls):
        cid = f"call_maestro_{step}_{i}"
        fn = calls[cid]
        entry = {"case": case if len(step_calls) == 1 else f"{case}_{i}", "tool": tool,
                 "wire_call": {"name": fn["name"], "arguments": fn["arguments"]},
                 "output": results[cid]}
        if tool == "read" and "filePath" in args:
            entry["inputs"] = initial(args["filePath"])
        if tool == "skill" and args.get("name") == "vital":
            skill_md = os.path.join(SB, "cfg/skills/vital/SKILL.md")
            entry["inputs"] = {"skill_md": open(skill_md).read()}
        if tool == "bash" and "Full output saved to: " in results[cid]:
            saved = results[cid].split("Full output saved to: ", 1)[1].split("\n", 1)[0]
            entry["inputs"] = {"saved_output_bytes": os.path.getsize(saved)}
        cases.append(norm(entry))
dump("tool_results.json", cases)

# -- hoisted media: the step's tool messages + the "Attached media" user message --
media = []
names = [case for case, _ in main_script(WD, PORT)]
msgs = main["messages"]
for i, m in enumerate(msgs):
    if m["role"] == "user" and isinstance(m["content"], list) and \
            m["content"][0].get("text") == "Attached media from tool result:":
        j = i - 1
        while msgs[j]["role"] == "tool":
            j -= 1
        step = int(msgs[j]["tool_calls"][0]["id"].split("_")[2])
        media.append({"case": names[step], "assistant": norm(msgs[j]),
                      "tool_messages": norm(msgs[j + 1:i]), "user": norm(m)})
dump("media_messages.json", media)

# -- task round trip ------------------------------------------------------------
task = []
for c in cases:
    if c["tool"] != "task" or c["case"] == "task_unknown_agent":
        continue
    args = json.loads(c["wire_call"]["arguments"])
    sub = norm(first[args["subagent_type"]]["messages"])
    task.append({"subagent_type": args["subagent_type"], "description": args["description"],
                 "prompt": args["prompt"], "subagent_first_user": sub[1],
                 "subagent_system": f"system/{args['subagent_type']}.txt",
                 "subagent_final_text": SUB_FINAL[args["subagent_type"]], "parent_result": c["output"]})
dump("task_roundtrip.json", task)

version = subprocess.run([os.path.expanduser("~/.opencode/bin/opencode"), "--version"],
                         capture_output=True, text=True).stdout.strip()
skills_src = "opencode/skills" if os.path.exists(os.path.join(GOLD, "../../../opencode/skills/vital/SKILL.md")) else "skills (fallback)"
meta = {
    "opencode_version": version,
    "captured_from": "docs/opencode_capture/goldens/capture/capture.sh" + (" (WITH_PLUGIN=1)" if SUBDIR else " (stock OpenCode, no plugin)"),
    "skills_staged_from": skills_src,
    "placeholders": {
        "{WORKDIR}": "session cwd (`<env>` Working directory, tool paths)",
        "{CONFIG_DIR}": "OPENCODE_CONFIG_DIR (skill <location>, skill base directory)",
        "{DATA_DIR}": "$XDG_DATA_HOME/opencode (bash 'Full output saved to' lives in {DATA_DIR}/tool-output/)",
        "{TOOL_OUTPUT_ID}": "tool_<12 hex><14 base62>, ascending id",
        "{SES_ID}": "ses_<12 hex><14 base62>, descending id; task results carry the subagent's",
        "{DATE}": "JS Date.toDateString() in local TZ, e.g. 'Sun Oct 25 2026'",
        "{PORT}": "mock server port (webfetch case only)",
    },
    "order_dependent": "skill_files (ripgrep order, max 10) and glob (mtime order) vary run to run; "
                       "the contract takes the list as input",
    "tool_names": [t["function"]["name"] for t in main["tools"]],
    "tool_names_subagent": [t["function"]["name"] for t in first["searcher"]["tools"]],
}
dump("meta.json", meta)
if SUBDIR:
    print("wrote goldens to", GOLD)
    sys.exit(0)
shutil.rmtree(os.path.join(GOLD, "agent_snapshot"), ignore_errors=True)
os.makedirs(os.path.join(GOLD, "agent_snapshot"))
for fn in os.listdir(os.path.join(SB, "cfg/agent")):
    shutil.copy(os.path.join(SB, "cfg/agent", fn), os.path.join(GOLD, "agent_snapshot", fn))
snap = os.path.join(GOLD, "skills_snapshot")
shutil.rmtree(snap, ignore_errors=True)
for root, _, files in os.walk(os.path.join(SB, "cfg/skills")):
    if "SKILL.md" in files:
        rel = os.path.relpath(root, os.path.join(SB, "cfg/skills"))
        os.makedirs(os.path.join(snap, rel))
        shutil.copy(os.path.join(root, "SKILL.md"), os.path.join(snap, rel, "SKILL.md"))
print("wrote goldens to", GOLD)
