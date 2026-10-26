#!/usr/bin/env python3
"""Replay mock of an OpenAI-compatible endpoint: serves recorded assistant steps verbatim to OpenCode.

    mock_server.py SCENARIO.json LOG.jsonl PORT

SCENARIO is records.build_scenario() output. A session's first POST is routed to a record by its system
prompt's first line + first user message (records.route); later POSTs of the same OpenCode session
(x-opencode-session-id header) stick to that record, the step is the number of assistant messages already
in the request, and the reply is that step's text and/or tool calls, streamed like the golden capture
mock (docs/opencode_capture/goldens/capture/mock_scripted.py; same call id scheme call_<agent>_<step>_<i>).
Every request is appended to LOG with base64 media replaced by sha256 digests; the decoded media is
saved once as <LOG dir>/media/<sha256>.<ext> (for audio_eq). A request without a session header whose
messages equal an earlier request's (a retry) gets that request's record. Title requests get a
fixed title. GET /page.html serves the capture mock's page (golden webfetch case). Stdlib only.
"""
import hashlib
import json
import os
import re
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import records as R  # noqa: E402

PAGE = (b"<html><head><title>Page</title></head><body><h1>Heading</h1>"
        b"<p>Some <b>bold</b> text.</p><ul><li>one</li><li>two</li></ul></body></html>")
TITLE = "Replay session"
_lock = threading.Lock()
_seq = [0]
_claimed: set = set()
_prefixes: dict = {}  # sha256 of the request messages -> (record key, kind)
_sessions: dict = {}  # x-opencode-session-id -> (record key, kind): a session sticks to its record


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
            return self._send(200, PAGE, "text/html; charset=utf-8")
        self._send(200, json.dumps({"object": "list", "data": [{"id": "maestro", "object": "model"}]}).encode(),
                   "application/json")

    def do_POST(self):
        raw = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        body = json.loads(raw)
        sid = self.headers.get("x-opencode-session-id")
        prefix = hashlib.sha256(json.dumps(body.get("messages"), sort_keys=True).encode()).hexdigest()
        with _lock:
            step = R.step_of(body)
            if R.route(SCENARIO, body)[1] == "title":
                key, kind = None, "title"
            elif sid and sid in _sessions:
                key, kind = _sessions[sid]
            elif not sid and prefix in _prefixes:  # a retry without a session header: same messages, same record
                key, kind = _prefixes[prefix]
            else:
                key, kind = R.route(SCENARIO, body, _claimed)
                if key is not None and step == 0:
                    _claimed.add(key)
            if key is not None:
                _prefixes[prefix] = (key, kind)
                if sid:
                    _sessions[sid] = (key, kind)
            media = R.media_payloads(body)
        for sha, raw, ext in media:
            path = os.path.join(MEDIA_DIR, f"{sha}.{ext}")
            if not os.path.exists(path):
                with open(path + ".part", "wb") as f:
                    f.write(raw)
                os.replace(path + ".part", path)
        text, calls, error = None, [], None
        if kind == "title":
            text = TITLE
        elif key is None:
            text, error = "REPLAY_UNROUTED", "no record matches this request"
        else:
            steps = SCENARIO["records"][key]["steps"]
            if step < len(steps):
                text, calls = steps[step]["text"], steps[step]["calls"]
            else:
                text, error = "REPLAY_EXHAUSTED", f"record has {len(steps)} steps, asked for step {step}"
        with _lock:
            _seq[0] += 1
            entry = {"seq": _seq[0], "path": self.path, "route": key, "kind": kind, "step": step,
                     "error": error,
                     "headers": {k.lower(): v for k, v in self.headers.items()
                                 if k.lower() not in ("authorization", "content-length")},
                     "body": R.strip_media(body)}
            with open(LOG, "a") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        agent = re.sub(r"[^A-Za-z0-9]", "_", key or kind)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        base = {"id": f"cmpl-{agent}-{step}", "object": "chat.completion.chunk", "created": 0, "model": "maestro"}

        def chunk(delta, finish=None):
            obj = {**base, "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
            self.wfile.write(b"data: " + json.dumps(obj).encode() + b"\n\n")

        if text:
            chunk({"role": "assistant", "content": text})
        if calls:
            tcs = [{"index": i, "id": f"call_{agent}_{step}_{i}", "type": "function",
                    "function": {"name": n, "arguments": a}} for i, (n, a) in enumerate(calls)]
            chunk({"role": "assistant", "tool_calls": tcs} if not text else {"tool_calls": tcs})
            chunk({}, "tool_calls")
        else:
            if not text:
                chunk({"role": "assistant", "content": ""})
            chunk({}, "stop")
        usage = {**base, "choices": [], "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}
        self.wfile.write(b"data: " + json.dumps(usage).encode() + b"\n\n")
        self.wfile.write(b"data: [DONE]\n\n")


if __name__ == "__main__":
    SCENARIO = json.load(open(sys.argv[1]))
    LOG, PORT = sys.argv[2], int(sys.argv[3])
    MEDIA_DIR = os.path.join(os.path.dirname(os.path.abspath(LOG)), "media")
    os.makedirs(MEDIA_DIR, exist_ok=True)
    ThreadingHTTPServer(("127.0.0.1", PORT), H).serve_forever()
