#!/usr/bin/env python3
"""Drive ONE live rollout through the harness inside a daw-farm container.

bench_headless.py cannot attach audio (it only submit_user()s a text prompt),
which is fine for fake-server replay but useless for a live audio-conditioned
model. This runner seeds the session with the target clip the way the real
plugin does, then lets the model drive: every tool call it emits executes for
real against the container's REAPER.

Runs INSIDE the container. Writes /work/live_dump.json with the full
conversation so the host can compare it against the memorized rollout.

Usage (inside container):
    python3 live_rollout_runner.py <llm_url> <target_wav> [user_text] [max_s]
"""
import json
import sys
import time

sys.path.insert(0, "/work/harness/src")
from maestro_plugin.app import ChatApp, AppConfig            # noqa: E402
from maestro_plugin.session import Attachment                # noqa: E402
from maestro_plugin.messages import (                        # noqa: E402
    OutputMessage, InputMessage, TextBlock, ToolUseBlock,
)

llm_url = sys.argv[1]
target = sys.argv[2]
user_text = sys.argv[3] if len(sys.argv) > 3 else "Recreate this sound in Vital."
budget = float(sys.argv[4]) if len(sys.argv) > 4 else 1800.0

cfg = AppConfig(
    llm_url=llm_url,
    model="maestro-main",
    cwd="/work/live_cwd",
    enable_sub_agents=True,
    # Greedy. AppConfig.temperature defaults to None, which omits the field and
    # lets vLLM apply its own default (0.7) — so every "deterministic" rollout
    # was actually sampling. For a memorized model driving a known sequence
    # that is the difference between reproducing the rollout and inventing a
    # plausible-looking one.
    temperature=0.0,
)
app = ChatApp(config=cfg)
# The tools block is inlined by the harness itself (build_system_prompt,
# include_tools=True) so the served prompt matches training byte-for-byte.
app.session.pending_attachments.append(Attachment(path=target, label="target"))
app.submit_user(user_text)

t0 = time.time()
while app.is_busy and time.time() - t0 < budget:
    app.drain()
    time.sleep(0.02)
app.drain()

dump = []
for msg in app.session.messages:
    if isinstance(msg, OutputMessage):
        for b in msg.blocks:
            if isinstance(b, TextBlock) and b.text.strip():
                dump.append({"role": "assistant", "content": b.text})
            elif isinstance(b, ToolUseBlock):
                dump.append({"role": "tool_call", "name": b.name,
                             "arguments": b.arguments})
    elif isinstance(msg, InputMessage):
        for b in msg.blocks:
            c = getattr(b, "content", None)
            if c is not None:
                dump.append({"role": msg.role, "content": str(c),
                             "is_error": bool(getattr(b, "is_error", False)),
                             "audio": bool(getattr(b, "audio_path", None))})

n_calls = sum(1 for d in dump if d["role"] == "tool_call")
by_tool = {}
for d in dump:
    if d["role"] == "tool_call":
        by_tool[d["name"]] = by_tool.get(d["name"], 0) + 1
errs = sum(1 for d in dump if d.get("is_error"))
out = {
    "timed_out": app.is_busy,
    "elapsed_s": round(time.time() - t0, 1),
    "n_messages": len(dump),
    "n_tool_calls": n_calls,
    "tools_by_type": by_tool,
    "tool_errors": errs,
    "messages": dump,
}
with open("/work/live_dump.json", "w") as f:
    json.dump(out, f, indent=2)
print(f"DONE timed_out={app.is_busy} elapsed={out['elapsed_s']}s "
      f"messages={len(dump)} tool_calls={n_calls} by_tool={by_tool} errors={errs}")
