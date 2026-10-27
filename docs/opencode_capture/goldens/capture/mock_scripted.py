#!/usr/bin/env python3
"""Scripted capturing mock of an OpenAI-compatible endpoint, for golden capture.

Every POST body is appended to CAPTURE (jsonl). Chat requests are answered by a
per-agent script: the agent is recognised from its system prompt, the step is
the number of assistant turns already in the request, and the reply is either a
batch of tool calls or a final text. The main agent walks through every stock
tool and result shape; subagents answer with one tool call then a final text.

Usage: mock_scripted.py CAPTURE PORT WORKDIR CONFIG_DIR
(extract.py imports main_script() to name the captured cases.)
"""
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def call(_name, **args):
    return (_name, args)


def main_script(WD, PORT):
    """(case name, tool calls) per main-agent step."""
    return [
        ("bash_ok", [call("bash", command="echo hello", description="Say hello")]),
        ("bash_fail_stderr", [call("bash", command="echo out; echo err >&2; exit 3", description="Fail with stderr")]),
        ("bash_no_output", [call("bash", command="true", description="No output")]),
        ("bash_truncated_lines", [call("bash", command="seq 1 3000", description="Long output")]),
        ("bash_truncated_bytes", [call("bash", command="python3 -c \"print('x'*60000)\"", description="Long line")]),
        ("bash_timeout", [call("bash", command="echo start; sleep 5", timeout=1000, description="Time out")]),
        ("bash_workdir", [call("bash", command="pwd", workdir=f"{WD}/sub", description="Workdir")]),
        ("read_offset_limit", [call("read", filePath=f"{WD}/notes.txt", offset=3, limit=5)]),
        ("read_full", [call("read", filePath=f"{WD}/notes.txt")]),
        ("read_short", [call("read", filePath=f"{WD}/short.txt")]),
        ("read_default_limit", [call("read", filePath=f"{WD}/long.txt")]),
        ("read_long_line", [call("read", filePath=f"{WD}/longline.txt")]),
        ("read_empty", [call("read", filePath=f"{WD}/empty.txt")]),
        ("read_directory", [call("read", filePath=f"{WD}/sub")]),
        ("read_missing", [call("read", filePath=f"{WD}/missing.txt")]),
        ("read_missing_suggest", [call("read", filePath=f"{WD}/note")]),
        ("read_offset_out_of_range", [call("read", filePath=f"{WD}/short.txt", offset=10)]),
        ("read_byte_cap", [call("read", filePath=f"{WD}/big.txt")]),
        ("read_binary", [call("read", filePath=f"{WD}/noise.raw")]),
        ("read_image", [call("read", filePath=f"{WD}/img.png")]),
        ("read_image_parallel", [call("read", filePath=f"{WD}/img.png"),
                                 call("read", filePath=f"{WD}/img2.png")]),
        ("read_audio", [call("read", filePath=f"{WD}/target.wav")]),
        ("read_audio_parallel", [call("read", filePath=f"{WD}/target.wav"),
                                 call("read", filePath=f"{WD}/second.wav")]),
        ("read_invalid_args", [call("read")]),
        ("unavailable_tool", [call("nosuchtool", x=1)]),
        ("skill_vital", [call("skill", name="vital")]),
        ("skill_missing", [call("skill", name="nosuchskill")]),
        ("write_new", [call("write", filePath=f"{WD}/new.txt", content="alpha\nbeta\n")]),
        ("write_existing", [call("write", filePath=f"{WD}/short.txt", content="overwritten\n")]),
        ("edit_ok", [call("edit", filePath=f"{WD}/notes.txt", oldString="line 30", newString="LINE THIRTY")]),
        ("edit_multiple", [call("edit", filePath=f"{WD}/notes.txt", oldString="line 2", newString="LINE TWO")]),
        ("edit_not_found", [call("edit", filePath=f"{WD}/notes.txt", oldString="not present", newString="x")]),
        ("glob_ok", [call("glob", pattern="**/*.txt")]),
        ("glob_none", [call("glob", pattern="*.nomatch")]),
        ("grep_ok", [call("grep", pattern="line 1", include="*.txt")]),
        ("grep_none", [call("grep", pattern="zzznomatch")]),
        ("todowrite", [call("todowrite", todos=[
            {"content": "Transcribe melody", "status": "completed", "priority": "high"},
            {"content": "Search wavetables", "status": "in_progress", "priority": "medium"}])]),
        ("webfetch", [call("webfetch", url=f"http://127.0.0.1:{PORT}/page.html", format="markdown")]),
        ("task_unknown_agent", [call("task", description="Bad subagent", subagent_type="nosuchagent", prompt="x")]),
        ("task_parallel", [call("task", description="Transcribe target melody", subagent_type="transcriber",
              prompt=("Transcribe the melody in the target clip — I'll use the notes to audition wavetables, so "
                      "the pitches and the rhythm have to be right. Write them to notes.json in the format the "
                      f"`vital` skill documents.\n\nTarget: {WD}/target.wav\nWrite the melody to {WD}/notes.json.")),
         call("task", description="Search wavetables 0-47", subagent_type="searcher",
              prompt=("Audition wavetables for the target clip — I'm recreating its sound and need the oscillator "
                      "wavetables. Audition every table in your chunk: render each one with the session notes "
                      "and listen against the target one at a time. Then combine the closest ones into pairs — "
                      "both oscillator orders, since osc 1 leads — render and listen to those too. Reply with "
                      "your best 2-3 configurations for the target in this chunk: tables in oscillator order, "
                      "each with a few words on what it explains.\n\n"
                      f"Target: {WD}/target.wav\nNotes: {WD}/notes.json.\nChunk: wavetables 0-47.")),
         call("task", description="Judge wavetable pool", subagent_type="judge",
              prompt=("Pick the oscillator wavetables for the target clip. The pool is the best configurations "
                      "each search chunk found — a configuration is a list of table names in oscillator order. "
                      "Render the strongest configurations, including recombos (one table from one "
                      "configuration, another from another), and listen against the target; reply with JSON: "
                      "{\"tuple\": [names in oscillator order], \"reasoning\": \"one or two sentences\"}.\n\n"
                      f"Target: {WD}/target.wav\nNotes: {WD}/notes.json.\n"
                      f"Pool: [[\"Basic Shapes\", \"Harmonic Series\"], [\"Basic Shapes\"]]."))]),
    ]

SUB_FINAL = {
    "transcriber": "notes.json: 8 notes.",
    "searcher": "1. Basic Shapes (osc 1) + Harmonic Series (osc 2): closest to the target in this chunk\n2. Basic Shapes: also close",
    "judge": '{"tuple": ["Basic Shapes"], "reasoning": "Closest body."}',
}



def agent_markers(cfg):
    """agent name -> first line of its prompt body (the system prompt starts with it)."""
    out = {}
    for fn in sorted(os.listdir(os.path.join(cfg, "agent"))):
        text = open(os.path.join(cfg, "agent", fn)).read()
        body = text.split("---", 2)[2].strip() if text.startswith("---") else text.strip()
        out[fn[:-3]] = body.splitlines()[0]
    # no custom main agent: the main agent is OpenCode's default `build` with the stock prompt
    out.setdefault("maestro", "You are opencode, an interactive CLI tool")
    return out


def agent_of(body):
    sysc = "".join(m["content"] if isinstance(m["content"], str) else json.dumps(m["content"])
                   for m in body.get("messages", []) if m["role"] == "system")
    for name, marker in MARKERS.items():
        if sysc.startswith(marker):
            return name
    return "other"


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, data, ct):
        self.send_response(code)
        self.send_header("Content-Type", ct)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path.startswith("/page.html"):
            html = (b"<html><head><title>Page</title></head><body><h1>Heading</h1>"
                    b"<p>Some <b>bold</b> text.</p><ul><li>one</li><li>two</li></ul></body></html>")
            return self._send(200, html, "text/html; charset=utf-8")
        self._send(200, json.dumps({"object": "list", "data": [{"id": "maestro", "object": "model"}]}).encode(),
                   "application/json")

    def do_POST(self):
        raw = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        body = json.loads(raw)
        agent = agent_of(body)
        with open(CAP, "a") as f:
            f.write(json.dumps({"path": self.path, "agent": agent, "headers": dict(self.headers),
                                "body": body}) + "\n")
        step = sum(1 for m in body.get("messages", []) if m["role"] == "assistant")
        calls, text = None, None
        if agent == "maestro":
            calls = MAIN[step][1] if step < len(MAIN) else None
            text = None if calls else "Done."
        elif agent in SUB_FINAL:
            if step == 0:
                calls = [call("bash", command=f"echo {agent}", description="Probe")]
            else:
                text = SUB_FINAL[agent]
        else:
            text = "Capture session"
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        base = {"id": f"cmpl-{agent}-{step}", "object": "chat.completion.chunk", "created": 0, "model": "maestro"}

        def chunk(delta, finish=None):
            obj = {**base, "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
            self.wfile.write(b"data: " + json.dumps(obj).encode() + b"\n\n")

        if calls:
            tcs = [{"index": i, "id": f"call_{agent}_{step}_{i}", "type": "function",
                    "function": {"name": n, "arguments": json.dumps(a)}} for i, (n, a) in enumerate(calls)]
            chunk({"role": "assistant", "tool_calls": tcs})
            chunk({}, "tool_calls")
        else:
            chunk({"role": "assistant", "content": text})
            chunk({}, "stop")
        usage = {**base, "choices": [], "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}
        self.wfile.write(b"data: " + json.dumps(usage).encode() + b"\n\n")
        self.wfile.write(b"data: [DONE]\n\n")


if __name__ == "__main__":
    CAP, PORT, WD, CFG = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4]
    MAIN = main_script(WD, PORT)
    MARKERS = agent_markers(CFG)
    ThreadingHTTPServer(("127.0.0.1", PORT), H).serve_forever()
