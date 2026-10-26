#!/usr/bin/env python3
"""Mock v2: returns a task tool-call on first request, then plain text."""
import json, sys, threading
from http.server import BaseHTTPRequestHandler, HTTPServer

CAP = '/tmp/opencode/captured2.jsonl'
COUNT = {'n': 0}

class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def do_GET(self):
        body = b'{"object":"list","data":[{"id":"qwen3.5-4b","object":"model"}]}'
        self.send_response(200); self.send_header('Content-Type','application/json')
        self.send_header('Content-Length', str(len(body))); self.end_headers(); self.wfile.write(body)
    def do_POST(self):
        n = int(self.headers.get('Content-Length', 0))
        raw = self.rfile.read(n)
        with open(CAP, 'a') as f:
            f.write(json.dumps({"headers": dict(self.headers), "body": json.loads(raw)}) + "\n")
        body = json.loads(raw)
        COUNT['n'] += 1
        kind = COUNT['n']
        self.send_response(200)
        self.send_header('Content-Type', 'text/event-stream')
        self.end_headers()
        def chunk(obj): self.wfile.write(b"data: " + json.dumps(obj).encode() + b"\n\n")
        base = {"id":"cmpl-x","object":"chat.completion.chunk"}
        # decide response based on which session: parent asks for task call; subagent replies text
        msgs = body.get('messages', [])
        sysc = str([m for m in msgs if m['role']=='system'][0].get('content',''))[:400]
        if 'wavetable-search agent' in sysc:
            chunk({**base,"choices":[{"index":0,"delta":{"role":"assistant","content":"SEARCH-SUB-RESULT: 2 candidates."},"finish_reason":None}]})
        else:
            tc = {"index":0,"id":"call_abc123","type":"function","function":{"name":"task","arguments":json.dumps({
                "description":"Evaluate wavetables 0-47",
                "prompt":"Target: /work/rollouts/bass_2fa91425/x_gt.wav.\nEvaluate wavetables at indices 0-47.",
                "subagent_type":"wavetable_search"})}}
            chunk({**base,"choices":[{"index":0,"delta":{"role":"assistant","tool_calls":[tc]},"finish_reason":None}]})
        chunk({**base,"choices":[{"index":0,"delta":{},"finish_reason":"tool_calls" if 'wavetable-search agent' not in sysc else "stop"}]})
        self.wfile.write(b"data: [DONE]\n\n")

HTTPServer(('127.0.0.1', 8899), H).serve_forever()
