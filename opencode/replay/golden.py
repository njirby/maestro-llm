"""The golden capture as replayable records (self-test of the replay harness).

goldens/<variant>/conversation_maestro.json + conversation_subagents.json are the last request
OpenCode sent per agent in capture.sh's run; with the final texts the scripted mock answered
("Done." and task_roundtrip's subagent_final_text) they are complete conversations. Converted to
record form, with the capture's placeholders filled for the replay environment, they must replay
with zero diffs. make_workdir() recreates capture.sh's work files byte for byte.
"""
from __future__ import annotations

import json
import math
import os
import struct
import wave
import zlib
from pathlib import Path
from typing import Any

from . import records as R

REPO = Path(__file__).resolve().parents[2]
GOLDENS = REPO / "docs" / "opencode_capture" / "goldens"
SUBAGENTS = ("transcriber", "searcher", "judge")
MAIN_FINAL = "Done."
PROMPT = "Recreate the sound in target.wav."


def make_workdir(wd: str | Path) -> None:
    """capture.sh's work files (same bytes; mtimes are fresh, glob order is whitelisted)."""
    wd = Path(wd)
    (wd / "sub" / "nested").mkdir(parents=True, exist_ok=True)
    (wd / "notes.txt").write_text("".join(f"line {i}\n" for i in range(1, 31)))
    (wd / "short.txt").write_text("one\ntwo\nthree\n")
    (wd / "long.txt").write_text("".join(f"{i}\n" for i in range(1, 3001)))
    (wd / "big.txt").write_text("\n".join("%04d " % i + "z" * 95 for i in range(1000)) + "\n")
    (wd / "longline.txt").write_text("y" * 2500 + "\n")
    (wd / "empty.txt").write_text("")
    (wd / "sub" / "a.txt").write_text("a\n")
    (wd / "sub" / "b.txt").write_text("b\n")
    (wd / "sub" / "nested" / "c.txt").write_text("c\n")
    for name, hz in (("target", 220), ("second", 330)):
        with wave.open(str(wd / f"{name}.wav"), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(16000)
            w.writeframes(b"".join(struct.pack("<h", int(8000 * math.sin(2 * math.pi * hz * i / 16000)))
                                   for i in range(4000)))
    (wd / "noise.raw").write_bytes(bytes(range(256)) * 16)

    def chunk(t: bytes, d: bytes) -> bytes:
        return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xffffffff)
    raw = b"".join(b"\x00" + b"\xff\x00\x00" * 4 for _ in range(4))
    png = (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", 4, 4, 8, 2, 0, 0, 0))
           + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))
    (wd / "img.png").write_bytes(png)
    (wd / "img2.png").write_bytes(png)


def _fill(x: Any, subs: dict[str, str]) -> Any:
    if isinstance(x, str):
        for k, v in subs.items():
            x = x.replace(k, v)
        return x
    if isinstance(x, list):
        return [_fill(v, subs) for v in x]
    if isinstance(x, dict):
        return {k: _fill(v, subs) for k, v in x.items()}
    return x


def golden_records(workdir: str, init_dir: str | Path, port: int, variant: str = "with_plugin",
                   config_dir: str = "/tmp/maestro-oc/config",
                   data_dir: str = "/tmp/maestro-oc/data/opencode") -> dict[str, dict[str, Any]]:
    """{"maestro": record, "transcriber": ..., "searcher": ..., "judge": ...}. Media placeholders
    {B64:<file>} become `audios` entries pointing at <init_dir>/<file>."""
    g = GOLDENS / variant if variant else GOLDENS
    subs = {"{WORKDIR}": workdir, "{CONFIG_DIR}": config_dir, "{DATA_DIR}": data_dir, "{PORT}": str(port)}
    conv_main = json.loads((g / "conversation_maestro.json").read_text())
    conv_subs = json.loads((g / "conversation_subagents.json").read_text())
    finals = {t["subagent_type"]: t["subagent_final_text"]
              for t in json.loads((g / "task_roundtrip.json").read_text())}
    tools = json.loads((g / "tools.json").read_text())
    tools_sub = json.loads((g / "tools_subagent.json").read_text())
    out = {}
    for agent, conv, final, tl in [("maestro", conv_main, MAIN_FINAL, tools)] + \
            [(a, conv_subs[a], finals[a], tools_sub) for a in SUBAGENTS]:
        system = (g / "system" / f"{agent}.txt").read_text()
        msgs, media = R.wire_to_record([{"role": "system", "content": system}] + conv)
        msgs = _fill(msgs, subs) + [{"role": "assistant", "content": final}]
        audios = []
        for ph in media:
            assert ph.startswith("{B64:") and ph.endswith("}"), ph
            audios.append(str(Path(init_dir) / ph[5:-1]))
        out[agent] = {"id": f"golden-{agent}", "messages": msgs, "audios": audios, "tools": tl}
    return out
