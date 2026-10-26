"""Minimal OpenAI-compatible chat client for the narrators (vLLM servers: Qwen3-Omni on :8000, the 27B on :8001).

No sampling parameters are ever sent (server defaults only); extra request fields are limited to what the caller
passes (e.g. chat_template_kwargs for the 27B's reasoning effort). HTTP errors fail immediately; only connection
errors (server restarting) are retried.

    parts = [audio_part(y, sr), {"type": "text", "text": prompt}]
    text, reasoning = chat("http://localhost:8000/v1", parts)
"""
from __future__ import annotations

import base64
import io
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
import requests
import soundfile as sf


@lru_cache(maxsize=8)
def model_id(url: str) -> str:
    r = requests.get(url.rstrip("/") + "/models", timeout=30)
    r.raise_for_status()
    return r.json()["data"][0]["id"]


def load_mono(path_or_array, sr: int | None = None) -> tuple[np.ndarray, int]:
    if isinstance(path_or_array, (str, Path)):
        y, sr = sf.read(path_or_array, dtype="float32", always_2d=True)
    else:
        y = np.asarray(path_or_array, dtype="float32")
        y = y[:, None] if y.ndim == 1 else y
    return y.mean(1), int(sr or 44100)


def audio_part(y: np.ndarray, sr: int) -> dict:
    buf = io.BytesIO()
    sf.write(buf, np.clip(y, -1, 1), sr, format="WAV", subtype="PCM_16")
    return {"type": "input_audio", "input_audio": {"data": base64.b64encode(buf.getvalue()).decode(), "format": "wav"}}


def audio_parts(clips: list, peak: float = 0.9) -> list[dict]:
    """Mono clips normalised by ONE gain (the loudest clip peaks at `peak`), so level differences between clips stay
    audible; a single clip is simply peak-normalised."""
    ys = [load_mono(c) for c in clips]
    g = peak / max(max(float(np.abs(y).max()) for y, _ in ys), 1e-6)
    return [audio_part(y * g, sr) for y, sr in ys]


def chat(url: str, content, *, model: str | None = None, system: str | None = None, extra: dict | None = None,
         timeout: float = 900, max_wait_s: float = 600) -> tuple[str, str]:
    """One chat call -> (text, reasoning). `content` is a string or a list of content parts."""
    msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": content}]
    body = {"model": model or model_id(url), "messages": msgs, **(extra or {})}
    t0 = time.time()
    while True:
        try:
            r = requests.post(url.rstrip("/") + "/chat/completions", json=body, timeout=timeout)
            break
        except requests.ConnectionError:
            if time.time() - t0 > max_wait_s:
                raise
            time.sleep(10)
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code} from {url}: {r.text[:500]}")
    m = r.json()["choices"][0]["message"]
    txt = m.get("content") or ""
    rsn = m.get("reasoning") or m.get("reasoning_content") or ""
    if "</think>" in txt:  # no reasoning parser on the server
        rsn, txt = txt.split("</think>", 1)
    return txt.strip(), rsn.replace("<think>", "").strip()
