#!/usr/bin/env python3
"""Scripted capturing mock: if the last message is a user text containing READ:<path>, reply with a
read tool call on <path>; otherwise reply with plain text. Captures every POST body to CAP."""
import json, os, re, sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

CAP = sys.argv[1]; PORT = int(sys.argv[2])

def text_of(m):
    c = m.get('content')
    if isinstance(c, str): return c
    return ' '.join(p.get('text', '') for p in (c or []) if isinstance(p, dict))

class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def do_GET(self):
        b = b'{"object":"list","data":[{"id":"qwen3.5-4b","object":"model"}]}'
        self.send_response(200); self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(b))); self.end_headers(); self.wfile.write(b)
    def do_POST(self):
        raw = self.rfile.read(int(self.headers.get('Content-Length', 0)))
        body = json.loads(raw)
        with open(CAP, 'a') as f: f.write(json.dumps({"path": self.path, "body": body}) + "\n")
        msgs = body.get('messages', [])
        last = msgs[-1] if msgs else {}
        m = re.search(r'READ:(\S+)', text_of(last)) if last.get('role') == 'user' else None
        is_title = 'title' in text_of(msgs[0]).lower()[:400] if msgs else False
        self.send_response(200); self.send_header('Content-Type', 'text/event-stream'); self.end_headers()
        def chunk(o): self.wfile.write(b"data: " + json.dumps(o).encode() + b"\n\n")
        base = {"id": "cmpl-x", "object": "chat.completion.chunk"}
        if m and not is_title and body.get('tools'):
            paths = m.group(1).split(',')
            tcs = [{"index": i, "id": f"call_{i}", "type": "function",
                    "function": {"name": "read", "arguments": json.dumps({"filePath": p})}} for i, p in enumerate(paths)]
            chunk({**base, "choices": [{"index": 0, "delta": {"role": "assistant", "tool_calls": tcs}, "finish_reason": None}]})
            chunk({**base, "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]})
        else:
            chunk({**base, "choices": [{"index": 0, "delta": {"role": "assistant", "content": "Done listening."}, "finish_reason": None}]})
            chunk({**base, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]})
        self.wfile.write(b"data: [DONE]\n\n")

ThreadingHTTPServer(('127.0.0.1', PORT), H).serve_forever()
