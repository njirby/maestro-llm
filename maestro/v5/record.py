"""ms-swift record assembly ({"id", "messages", "audios", "tools"}) on top of the OpenCode contract.

One Record per agent session. The record grows step by step the way OpenCode builds the conversation:

  system  oc.system_message(agent, cwd, date)            (fixed date for reproducibility)
  user    the first user message (main: the user's request; subagent: the dispatch prompt verbatim)
  then per assistant step: optional assistant text, then the step's tool_call messages, then their tool_response
  messages in call order, then (if the step read audio) ONE user message "Attached media from tool result:<audio>..."
  (oc.audio_read_step). The record ends with assistant text.

`tools` is the JSON string of oc.tools_for(agent) (what OpenCode advertises to that agent).
"""
from __future__ import annotations

import json
from typing import Iterable

from scripts import opencode_contract as oc

# Fixed, reproducible env values (the <env> block of every system message).
DATE = "Sun Oct 25 2026"


class Record:
    def __init__(self, agent: str, rid: str, cwd: str, first_user: str, date: str = DATE):
        self.agent = agent
        self.id = rid
        self.cwd = cwd
        self.messages: list[dict[str, str]] = [
            {"role": "system", "content": oc.system_message(agent, cwd, date=date)},
            {"role": "user", "content": first_user},
        ]
        self.audios: list[str] = []
        self._text: str | None = None   # assistant text waiting for the next step (or the end)

    # -- assistant output ------------------------------------------------------

    def say(self, text: str) -> None:
        """Assistant text. It opens the next step (text + tool calls = one assistant message) or ends the record."""
        text = text.strip()
        if not text:
            return
        self._text = f"{self._text}\n\n{text}" if self._text else text

    def _flush_text(self) -> None:
        if self._text:
            self.messages.append({"role": "assistant", "content": self._text})
            self._text = None

    def step(self, calls: Iterable[str], responses: Iterable[str]) -> None:
        """One assistant step: N tool calls (oc.*_call strings) then their N results in call order."""
        calls, responses = list(calls), list(responses)
        assert calls and len(calls) == len(responses), (len(calls), len(responses))
        self._flush_text()
        self.messages += [{"role": "tool_call", "content": c} for c in calls]
        self.messages += [{"role": "tool_response", "content": r} for r in responses]

    def audio_step(self, file_paths: list[str], host_paths: list[str]) -> None:
        """One step of parallel reads of audio files: the reads, their AUDIO_READ_TEXT results, then the
        attachment user message; the clips join `audios` in call order."""
        assert len(file_paths) == len(host_paths) and file_paths
        self._flush_text()
        self.messages += [{"role": "tool_call", "content": oc.read_call(p)} for p in file_paths]
        responses, user = oc.audio_read_step(file_paths)
        self.messages += responses
        self.messages.append(user)
        self.audios += list(host_paths)

    def finish(self, text: str) -> dict:
        """Final assistant reply; returns the record."""
        self.say(text)
        self._flush_text()
        return self.to_dict()

    def to_dict(self) -> dict:
        return {"id": self.id, "messages": list(self.messages), "audios": list(self.audios),
                "tools": json.dumps(oc.tools_for(self.agent), ensure_ascii=False)}


def final_text(record: dict) -> str:
    """The record's last assistant message (a subagent's reply, which the parent gets in its task_result)."""
    m = record["messages"][-1]
    assert m["role"] == "assistant", m["role"]
    return m["content"]
