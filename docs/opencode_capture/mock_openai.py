#!/usr/bin/env python3
"""Mock OpenAI server: captures requests, replies with a canned text stream."""
import json, sys, threading
from http.server import BaseHTTPRequestHandler, HTTPServer

CAP = sys.argv[1] if len(sys.argv) > 1 else '/tmp/opencode/captured.jsonl'

class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def _reply(self, code, obj, ct='application/json'):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header('Content-Type', ct)
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)
    def do_GET(self):
        if self.path.startswith('/v1/models'):
            self._reply(200, {"object":"list","data":[{"id":"qwen3.5-4b","object":"model"}]})
        else:
            self._reply(200, {"ok": True, "path": self.path})
    def do_POST(self):
        n = int(self.headers.get('Content-Length', 0))
        raw = self.rfile.read(n)
        with open(CAP, 'a') as f:
            f.write(json.dumps({"path": self.path, "headers": dict(self.headers), "body": json.loads(raw)}) + "\n")
        try:
            body = json.loads(raw)
        except Exception:
            body = {}
        if self.path.endswith('/chat/completions'):
            if body.get('stream'):
                self.send_response(200)
                self.send_header('Content-Type', 'text/event-stream')
                self.end_headers()
                delta = {"id":"cmpl-1","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"Mock reply."},"finish_reason":None}]}
                self.wfile.write(b"data: " + json.dumps(delta).encode() + b"\n\n")
                done = {"id":"cmpl-1","object":"chat.completion.chunk","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}
                self.wfile.write(b"data: " + json.dumps(done).encode() + b"\n\n")
                self.wfile.write(b"data: [DONE]\n\n")
            else:
                self._reply(200, {"id":"cmpl-1","object":"chat.completion","choices":[{"index":0,"message":{"role":"assistant","content":"Mock reply."},"finish_reason":"stop"}],"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15}})
        else:
            self._reply(200, {"ok": True})

HTTPServer(('127.0.0.1', 8899), H).serve_forever()
