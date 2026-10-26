"""No sampling parameters (temperature/top_p/top_k/min_p/penalties/seed) in any request body we send: server defaults
only. Checked on the live clients (requests.post patched) and by a source scan of every client module."""
import re
from pathlib import Path

import pytest

SAMPLING = {"temperature", "top_p", "top_k", "min_p", "presence_penalty", "frequency_penalty", "repetition_penalty",
            "seed", "typical_p", "logit_bias"}
ROOT = Path(__file__).resolve().parents[1]


class _Resp:
    status_code = 200
    text = ""

    def json(self):
        return {"choices": [{"message": {"content": "A dry, plucky little line with clean gaps between the notes."}}],
                "data": [{"id": "m"}]}

    def raise_for_status(self):
        pass


@pytest.fixture
def bodies(monkeypatch):
    import requests
    seen = []

    def post(url, json=None, **kw):
        seen.append(json)
        return _Resp()
    monkeypatch.setattr(requests, "post", post)
    monkeypatch.setattr(requests, "get", lambda *a, **k: _Resp())
    return seen


def _flat_keys(d):
    for k, v in d.items():
        yield k
        if isinstance(v, dict) and k != "chat_template_kwargs":
            yield from _flat_keys(v)


def test_omni_chat(bodies):
    from maestro.sounddiff.omni import chat
    chat("http://x/v1", "hi", model="m")
    chat("http://x/v1", "hi", model="m", extra={"chat_template_kwargs": {"reasoning_effort": "xhigh"}})
    assert bodies and all(not SAMPLING & set(_flat_keys(b)) for b in bodies)


def test_narrator(bodies):
    from maestro.sounddiff.narrate import Narrator
    for think in (True, False):
        Narrator("http://x/v1", "m", think=think).call("hi")
    assert not hasattr(Narrator(), "temperature")
    assert bodies and all(not SAMPLING & set(_flat_keys(b)) for b in bodies)


def test_delta_narrators(bodies):
    from maestro.sounddiff.delta import narrate_self_edit, narrate_target_vs_current
    n = {"text": "Session notes", "check": {"mode": "self_edit", "overall": "none", "dirs": {}, "flags": {}}}
    narrate_self_edit(n, url="http://x/v1", model="m", arm="text")
    narrate_target_vs_current({**n, "check": {**n["check"], "mode": "target_vs_current"}}, url="http://x/v1",
                              model="m", arm="text")
    assert bodies and all(not SAMPLING & set(_flat_keys(b)) for b in bodies)


def test_source_scan():
    """No client under maestro/sounddiff writes a sampling key into a request body (scratch scripts included)."""
    pat = re.compile(r"""["'](%s)["']\s*:""" % "|".join(SAMPLING))
    bad = [f"{p.relative_to(ROOT)}:{i + 1}" for p in ROOT.rglob("*.py") if "tests" not in p.parts
           for i, line in enumerate(p.read_text().splitlines()) if pat.search(line)]
    assert bad == []
    kw = re.compile(r"\b(temperature|top_p|top_k|min_p)\s*=")
    bad = [f"{p.relative_to(ROOT)}:{i + 1}" for p in ROOT.rglob("*.py") if "tests" not in p.parts
           for i, line in enumerate(p.read_text().splitlines()) if kw.search(line)]
    assert bad == []
