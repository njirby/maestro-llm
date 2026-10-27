"""opencode/replay (conformance replay of records through real OpenCode 1.19.3).

Offline: normalizer whitelist, wire <-> record conversion, record -> expected request, routing, and the
per-request diff, all on the golden capture converted to record form (opencode/replay/golden.py).
Live (opt-in, needs Docker + daw-farm/reaper:latest + ~/.opencode/bin/opencode + rg 15.1.0):
    MAESTRO_REPLAY_LIVE=1 .venv/bin/python -m pytest tests/test_replay.py -q -k live
"""

import base64
import copy
import json
import os
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from opencode.replay import golden as G  # noqa: E402
from opencode.replay import records as R  # noqa: E402

WORKDIR = "/work/rollouts/golden"
PORT = 8911


@pytest.fixture(scope="module")
def golden(tmp_path_factory):
    init = tmp_path_factory.mktemp("golden_init")
    G.make_workdir(init)
    return G.golden_records(WORKDIR, init, PORT), init


def _b64(path):
    return base64.b64encode(Path(path).read_bytes()).decode()


def _request(rec, step):
    """The request OpenCode should send for `step` of `rec`, built from the record (what a perfect
    replay logs, media stripped to digests)."""
    prefix = R.expected_prefix(rec, step)
    n = R.count_media(prefix)
    media = [_b64(p) for p in rec["audios"][:n]]
    body = {"model": "maestro", "messages": R.record_to_wire(prefix, media), "tools": rec["tools"]}
    return R.strip_media(body)


def _digests(rec):
    return [R.file_digest(p) for p in rec["audios"]]


# -- normalizer -----------------------------------------------------------------------------------

def test_norm_text_volatile_fields():
    a = ("Today's date: Sun Oct 25 2026\n<task id=\"ses_e8a157ff0ffeUPCdsKuo4u8Ciu\" state=\"completed\">\n"
         "Full output saved to: /tmp/maestro-oc/data/opencode/tool-output/tool_e8a157ff0ffeUPCdsKuo4u8Ciu\n"
         "/tmp/tmpab12_xyz/out.wav")
    b = ("Today's date: Mon Jan 05 2027\n<task id=\"ses_0123456789abZZZZZZZZZZZZZZ\" state=\"completed\">\n"
         "Full output saved to: /tmp/maestro-oc/data/opencode/tool-output/tool_0123456789abZZZZZZZZZZZZZZ\n"
         "/tmp/tmpQQQQQQQQ/out.wav")
    assert a != b and R.norm_text(a) == R.norm_text(b)
    assert "{DATE}" in R.norm_text(a) and "{SES_ID}" in R.norm_text(a) and "{TOOL_OUTPUT_ID}" in R.norm_text(a)


def test_norm_text_keeps_real_differences():
    assert R.norm_text("Working directory: /work/rollouts/a") != R.norm_text("Working directory: /work/rollouts/b")
    assert R.norm_text("ses_short") == "ses_short"


def _skill(files, base="/tmp/maestro-oc/config/skills/vital"):
    return (f"<skill_content name=\"vital\">\nbody\n\nBase directory for this skill: {base}\n"
            "Relative paths in this skill (e.g., scripts/, reference/) are relative to this base directory.\n"
            "Note: file list is sampled.\n\n<skill_files>\n"
            + "\n".join(f"<file>{base}/{f}</file>" for f in files) + "\n</skill_files>\n</skill_content>")


def test_skill_files_sample_order_and_choice_whitelisted():
    assert R.norm_text(_skill(["a.py", "b.py"])) == R.norm_text(_skill(["scripts/c.py", "a.py"]))
    assert R.norm_text(_skill(["a.py", "b.py"])) != R.norm_text(_skill(["a.py"]))          # count matters


def test_skill_files_outside_base_not_whitelisted():
    good = _skill(["a.py"])
    bad = good.replace("<file>/tmp/maestro-oc/config/skills/vital/a.py", "<file>/home/nate/x/a.py")
    assert R.norm_text(good) != R.norm_text(bad)


def test_glob_order_whitelisted_only_for_glob():
    def conv(name, out):
        return [{"role": "tool_call", "content": json.dumps({"name": name, "arguments": {"pattern": "*"}})},
                {"role": "tool_response", "content": out}]
    assert R.normalize_messages(conv("glob", "/w/a\n/w/b")) == R.normalize_messages(conv("glob", "/w/b\n/w/a"))
    assert R.normalize_messages(conv("bash", "/w/a\n/w/b")) != R.normalize_messages(conv("bash", "/w/b\n/w/a"))


def test_tool_call_whitespace_insensitive_key_order_sensitive():
    a = [{"role": "tool_call", "content": '{"name": "read", "arguments": {"filePath": "x", "offset": 2}}'}]
    b = [{"role": "tool_call", "content": '{"name":"read","arguments":{"filePath":"x","offset":2}}'}]
    c = [{"role": "tool_call", "content": '{"name":"read","arguments":{"offset":2,"filePath":"x"}}'}]
    assert R.normalize_messages(a) == R.normalize_messages(b) != R.normalize_messages(c)


# -- conversion ----------------------------------------------------------------------------------

def test_golden_record_shape(golden):
    recs, _ = golden
    assert set(recs) == {"maestro", *G.SUBAGENTS}
    main = recs["maestro"]
    assert main["messages"][0]["role"] == "system" and main["messages"][1] == {"role": "user", "content": G.PROMPT}
    assert main["messages"][-1] == {"role": "assistant", "content": G.MAIN_FINAL}
    assert len(main["audios"]) == R.count_media(main["messages"]) == 3
    assert f"Working directory: {WORKDIR}" in main["messages"][0]["content"]
    for a in G.SUBAGENTS:
        assert recs[a]["messages"][-1]["role"] == "assistant"


def test_record_wire_roundtrip(golden):
    recs, _ = golden
    for rec in recs.values():
        media = [_b64(p) for p in rec["audios"]]
        wire = R.record_to_wire(rec["messages"], media)
        back, got_media = R.wire_to_record(wire)
        assert R.normalize_messages(back) == R.normalize_messages(rec["messages"])
        assert got_media == media


def test_media_message_wire_form():
    msgs = [{"role": "user", "content": R.MEDIA_ATTACHED_TEXT + "<audio><audio>"}]
    wire = R.record_to_wire(msgs, ["QUFB", "QkJC"])
    assert wire == [{"role": "user", "content": [
        {"type": "text", "text": R.MEDIA_ATTACHED_TEXT},
        {"type": "input_audio", "input_audio": {"data": "QUFB", "format": "wav"}},
        {"type": "input_audio", "input_audio": {"data": "QkJC", "format": "wav"}}]}]


def test_expected_prefix_matches_captured_conversation(golden):
    """The capture's last main request is the record's prefix for its last step."""
    recs, init = golden
    main = recs["maestro"]
    last = len(R.record_steps(main["messages"])) - 1
    conv = json.loads((G.GOLDENS / "with_plugin" / "conversation_maestro.json").read_text())
    system = (G.GOLDENS / "with_plugin" / "system" / "maestro.txt").read_text()
    got, media = R.wire_to_record([{"role": "system", "content": system}] + conv)
    got = G._fill(got, {"{WORKDIR}": WORKDIR, "{CONFIG_DIR}": "/tmp/maestro-oc/config",
                        "{DATA_DIR}": "/tmp/maestro-oc/data/opencode", "{PORT}": str(PORT)})
    assert R.normalize_messages(got) == R.normalize_messages(R.expected_prefix(main, last))
    assert len(media) == len(main["audios"])


def test_step_reply_and_invalid_tool():
    msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"},
            {"role": "assistant", "content": "thinking"},
            {"role": "tool_call", "content": json.dumps({"name": "read", "arguments": {"filePath": "/a"}})},
            {"role": "tool_call", "content": json.dumps({"name": "invalid",
                                                         "arguments": {"tool": "nope", "error": "x"}})},
            {"role": "tool_response", "content": "r1"}, {"role": "tool_response", "content": "r2"},
            {"role": "assistant", "content": "done"}]
    steps = R.record_steps(msgs)
    assert steps == [(2, 5), (7, 8)]
    assert R.step_reply(msgs, *steps[0]) == {"text": "thinking",
                                             "calls": [["read", '{"filePath": "/a"}'], ["nope", "{}"]]}
    assert R.step_reply(msgs, *steps[1]) == {"text": "done", "calls": []}


# -- diff ------------------------------------------------------------------------------------------

def test_compare_perfect_requests_match(golden):
    recs, _ = golden
    for rec in recs.values():
        dig = _digests(rec)
        for step in range(len(R.record_steps(rec["messages"]))):
            res = R.compare_request(_request(rec, step), rec, step, dig)
            assert res["status"] == "match", (rec["id"], step, res["diffs"])


def test_compare_whitelisted_change_still_matches(golden):
    """Record built on one date, replayed on another (the golden system keeps a {DATE} placeholder)."""
    recs, _ = golden
    rec = copy.deepcopy(recs["maestro"])
    sysm = rec["messages"][0]["content"]
    assert "Today's date: {DATE}" in sysm
    rec["messages"][0]["content"] = sysm.replace("{DATE}", "Sun Oct 25 2026")
    body = _request(rec, 3)
    body["messages"][0]["content"] = sysm.replace("{DATE}", "Mon Jan 05 2027")
    assert R.compare_request(body, rec, 3, _digests(rec))["status"] == "match"
    body["messages"][0]["content"] = sysm.replace("{DATE}", "Mon Jan 05 2027 ")
    assert R.compare_request(body, rec, 3, _digests(rec))["status"] == "diff"


@pytest.mark.parametrize("what", ["system", "tool_response", "tools", "media", "params"])
def test_compare_detects(golden, what):
    recs, _ = golden
    rec = recs["maestro"]
    step = len(R.record_steps(rec["messages"])) - 1
    body = _request(rec, step)
    params = None
    if what == "system":
        body["messages"][0]["content"] += " "
    elif what == "tool_response":
        tm = next(m for m in body["messages"] if m["role"] == "tool")
        tm["content"] += "X"
    elif what == "tools":
        body["tools"] = copy.deepcopy(body["tools"])
        body["tools"][0]["function"]["description"] += "."
    elif what == "media":
        part = next(p for m in body["messages"] if isinstance(m["content"], list)
                    for p in m["content"] if p["type"] == "input_audio")
        part["input_audio"]["data"] = R.media_digest(base64.b64encode(b"other").decode())
    elif what == "params":
        params = {"model": "maestro", "temperature": 0.5}
    res = R.compare_request(body, rec, step, _digests(rec), params)
    assert res["status"] == "diff"
    assert what.replace("tool_response", "messages") in [d["kind"] for d in res["diffs"]]


# -- routing ---------------------------------------------------------------------------------------

def test_route_golden(golden):
    recs, _ = golden
    sc = R.build_scenario(recs)
    for key, rec in recs.items():
        for step in range(len(R.record_steps(rec["messages"]))):
            body = R.record_to_wire(R.expected_prefix(rec, step), [_b64(p) for p in rec["audios"]])
            assert R.route(sc, {"messages": body}) == (key, "record")
            assert R.step_of({"messages": body}) == step
    title = json.loads((G.GOLDENS / "title_request.json").read_text())
    assert R.route(sc, title)[1] == "title"


def test_route_same_prompt_disambiguated_by_history(golden):
    recs, _ = golden
    a = copy.deepcopy(recs["transcriber"])
    b = copy.deepcopy(recs["transcriber"])
    a["id"], b["id"] = "t-a", "t-b"
    b["messages"][-1]["content"] = "a different final answer"
    i = next(i for i, m in enumerate(b["messages"]) if m["role"] == "tool_call")
    d = json.loads(b["messages"][i]["content"])
    d["arguments"] = {**d["arguments"], "description": "other"} if "description" in d["arguments"] else \
        {**d["arguments"], "zz": 1}
    b["messages"][i]["content"] = json.dumps(d)
    sc = R.build_scenario({"t-a": a, "t-b": b})
    claimed = set()
    first = {"messages": R.record_to_wire(R.expected_prefix(a, 0), [])}
    assert R.route(sc, first, claimed)[0] == "t-a"
    claimed.add("t-a")
    assert R.route(sc, first, claimed)[0] == "t-b"
    for rec in (a, b):
        body = {"messages": R.record_to_wire(R.expected_prefix(rec, 1), [])}
        assert R.route(sc, body, claimed)[0] == rec["id"]


# -- live ------------------------------------------------------------------------------------------

@pytest.mark.skipif(os.environ.get("MAESTRO_REPLAY_LIVE") != "1", reason="set MAESTRO_REPLAY_LIVE=1 (Docker)")
def test_live_golden_replay_zero_diffs(tmp_path):
    from opencode.replay import harness as H
    init = tmp_path / "init"
    G.make_workdir(init)
    recs = G.golden_records(WORKDIR, init, PORT)
    rep = H.replay(recs["maestro"], [recs[s] for s in G.SUBAGENTS], tmp_path / "run", init_dir=init,
                   prompt=G.PROMPT, port=PORT)
    assert rep["opencode_exit"] == 0
    # 47 record requests + the title request
    assert rep["compared"] == 48 and rep["diffs"] == 0, (tmp_path / "run" / "report.md").read_text()


# -- hardening: numeric class, audio equivalence, title, retries -----------------------------------

def test_numeric_only_diff_is_own_class(golden):
    recs, _ = golden
    rec = copy.deepcopy(recs["maestro"])
    i = next(i for i, m in enumerate(rec["messages"]) if m["role"] == "tool_response" and any(c.isdigit() for c in m["content"]))
    step = next(k for k, (a, _) in enumerate(R.record_steps(rec["messages"])) if a > i)
    body = _request(rec, step)
    tm = [m for m in body["messages"] if m["role"] == "tool"]
    n_before = sum(1 for m in rec["messages"][:i] if m["role"] == "tool_response")
    import re
    tm[n_before]["content"] = re.sub(r"\d", "7", tm[n_before]["content"])
    kinds = [d["kind"] for d in R.compare_request(body, rec, step, _digests(rec))["diffs"]]
    assert kinds == ["numeric"]


def _wav(path, hz, phase=0.0, n=44100, sr=44100, noise=0.0, seed=0):
    import numpy as np
    import soundfile as sf
    t = np.arange(n) / sr
    # band-limited saw with an attack (a pure sine is pathological for the >-80 dB mask: sidelobes move with phase)
    x = 0.2 * sum(np.sin(2 * np.pi * hz * k * t + phase * k) / k for k in range(1, int(8000 / hz)))
    x = x * np.minimum(1, t * 20) + noise * np.random.default_rng(seed).standard_normal(n)
    sf.write(str(path), np.stack([x, x], 1), sr, subtype="PCM_16")
    return path


def _scaled(src, dst, gain):
    import soundfile as sf
    x, sr = sf.read(str(src), always_2d=True)
    sf.write(str(dst), x * gain, sr, subtype="FLOAT")
    return dst


def test_audio_eq(tmp_path):
    from opencode.replay import audio_eq
    assert audio_eq.DEFAULT_MAX_DB == 10.0 and audio_eq.SILENT_PEAK_DBFS == -60.0
    a = _wav(tmp_path / "a.wav", 110)
    b = _wav(tmp_path / "b.wav", 110, phase=1.3)      # re-render: other oscillator phase
    r = audio_eq.compare(a, b)
    assert r["equivalent"] and r["mel_db"] < 1, r
    # mel_db is informational: another pitch passes the loose default cap, fails a tight one
    c = _wav(tmp_path / "c.wav", 165)
    assert audio_eq.compare(a, c)["equivalent"]
    r = audio_eq.compare(a, c, max_db=1.5)
    assert not r["equivalent"] and "cap" in r["reason"]
    # silence mismatch, both directions; silent vs silent is fine
    silent = _scaled(a, tmp_path / "s.wav", 0.0)
    quiet = _scaled(a, tmp_path / "q.wav", 10 ** (-70 / 20))   # peak ~ -80 dBFS: silent
    for x, y in ((a, silent), (silent, a), (a, quiet)):
        r = audio_eq.compare(x, y)
        assert not r["equivalent"] and r["reason"].startswith("silence mismatch"), r
    assert audio_eq.compare(silent, quiet)["equivalent"]
    # length: beyond one block fails, within passes
    r = audio_eq.compare(a, _wav(tmp_path / "d.wav", 110, n=44100 + 2000))
    assert not r["equivalent"] and r["reason"].startswith("length"), r
    assert audio_eq.compare(a, _wav(tmp_path / "e.wav", 110, n=44100 + 300))["equivalent"]
    (tmp_path / "x.wav").write_bytes(b"junk")
    assert not audio_eq.compare(a, tmp_path / "x.wav")["equivalent"]


def test_media_eq_hook(golden):
    recs, _ = golden
    rec = recs["maestro"]
    step = len(R.record_steps(rec["messages"])) - 1
    body = _request(rec, step)
    part = next(p for m in body["messages"] if isinstance(m["content"], list)
                for p in m["content"] if p["type"] == "input_audio")
    part["input_audio"]["data"] = R.media_digest(base64.b64encode(b"rerender").decode())
    ok = R.compare_request(body, rec, step, _digests(rec), media_eq=lambda i, s: {"equivalent": True, "mel_db": 0.3})
    assert ok["status"] == "match" and ok["whitelisted"][0]["kind"] == "media_render"
    bad = R.compare_request(body, rec, step, _digests(rec), media_eq=lambda i, s: {"equivalent": False})
    assert [d["kind"] for d in bad["diffs"]] == ["media"]


def test_media_payloads():
    body = {"messages": [{"role": "user", "content": [
        {"type": "input_audio", "input_audio": {"data": base64.b64encode(b"abc").decode(), "format": "wav"}},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(b"png").decode()}}]}]}
    got = R.media_payloads(body)
    assert [(raw, ext) for _, raw, ext in got] == [(b"abc", "wav"), (b"png", "png")]


def test_title_request_diff():
    from opencode.replay import harness as H
    exp = H.expected_title_request(G.PROMPT, "maestro")
    assert exp == json.loads((G.GOLDENS / "title_request.json").read_text())
    assert H._title_row(copy.deepcopy(exp), exp)["status"] == "match"
    other = copy.deepcopy(exp)
    other["max_tokens"] = 100
    assert H._title_row(other, exp)["status"] == "diff"


def test_mock_routes_retries_and_sessions(golden, tmp_path):
    """Local mock process (no Docker): a header-less retry of a step-0 request keeps its record; a
    second session with the same prompt gets the other record; sessions stick by header."""
    import socket
    import subprocess
    import time
    import urllib.request
    recs, _ = golden
    a, b = copy.deepcopy(recs["transcriber"]), copy.deepcopy(recs["transcriber"])
    a["id"], b["id"] = "t-a", "t-b"
    (tmp_path / "sc.json").write_text(json.dumps(R.build_scenario({"t-a": a, "t-b": b})))
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    log = tmp_path / "io" / "requests.jsonl"
    log.parent.mkdir()
    p = subprocess.Popen([sys.executable, "-B", str(REPO / "opencode/replay/mock_server.py"), str(tmp_path / "sc.json"),
                          str(log), str(port)])
    try:
        body0 = json.dumps({"messages": R.record_to_wire(R.expected_prefix(a, 0), [])}).encode()

        def post(data, sid=None):
            h = {"Content-Type": "application/json", **({"x-opencode-session-id": sid} if sid else {})}
            for _ in range(50):
                try:
                    return urllib.request.urlopen(urllib.request.Request(
                        f"http://127.0.0.1:{port}/v1/chat/completions", data=data, headers=h), timeout=5).read()
                except OSError:
                    time.sleep(0.1)
            raise RuntimeError("mock not reachable")
        post(body0)
        post(body0)                      # retry, no header
        post(body0, "ses_one")           # a new session: claims the next record
        post(body0, "ses_two")           # all claimed: falls back to the first
        post(body0, "ses_one")           # retry of ses_one: sticks
        routes = [json.loads(line)["route"] for line in log.open()]
        assert routes == ["t-a", "t-a", "t-b", "t-a", "t-b"]
    finally:
        p.terminate()
        p.wait(timeout=10)


def test_compare_workdirs(tmp_path):
    from opencode.replay import harness as H
    a, b = tmp_path / "a", tmp_path / "b"
    for d in (a, b):
        (d / "probes").mkdir(parents=True)
        (d / "notes.json").write_text('{"notes": []}')
    _wav(a / "r01.wav", 110)
    _wav(b / "r01.wav", 110, phase=1.3)
    w = H.compare_workdirs(a, b)
    assert w["diffs"] == [] and w["identical"] == 1 and list(w["audio_rerendered"]) == ["r01.wav"]
    (b / "preset.json").write_text("{}")
    (b / "notes.json").write_text('{"notes": [1]}')
    kinds = [d["kind"] for d in H.compare_workdirs(a, b)["diffs"]]
    assert kinds == ["workdir_files", "workdir_content"]


def test_stage_root_stages_all_config(tmp_path):
    from opencode.replay import harness as H
    H.stage_root(tmp_path / "r")
    staged = sorted(p.name for p in (tmp_path / "r" / "config").iterdir())
    expect = sorted(p.name for p in (REPO / "opencode").iterdir()
                    if p.name not in H.STAGE_SKIP and not p.name.startswith("."))
    assert staged == expect and "opencode.json" in staged and "plugin" in staged
    assert all((tmp_path / "r" / "config" / n).is_symlink() for n in staged)
    H.stage_root(tmp_path / "s", with_plugin=False)
    assert "plugin" not in [p.name for p in (tmp_path / "s" / "config").iterdir()]


@pytest.mark.parametrize("shape", ["list", "files", "dict"])
def test_compare_workdirs_manifest(tmp_path, shape):
    import hashlib
    from opencode.replay import harness as H
    a, b = tmp_path / "a", tmp_path / "b"
    for d in (a, b):
        d.mkdir()
        (d / "notes.json").write_text('{"notes": []}')
    _wav(a / "r01.wav", 110)
    _wav(b / "r01.wav", 110, phase=1.3)
    ents = [{"path": str(p.relative_to(a)), "size": p.stat().st_size,
             "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(a.rglob("*")) if p.is_file()]
    m = {"list": ents, "files": {"files": ents}, "dict": {e["path"]: e for e in ents}}[shape]
    (tmp_path / "m.json").write_text(json.dumps(m))
    man = H.load_manifest(tmp_path / "m.json")
    w = H.compare_workdirs(man, b)
    assert w["source"] == "manifest" and w["diffs"] == [] and w["identical"] == 1
    assert w["audio_rerendered"] == {"r01.wav": 0}                      # size difference, no samples
    (b / "notes.json").write_text('{"notes": [1]}')
    (b / "extra.json").write_text("{}")
    assert [d["kind"] for d in H.compare_workdirs(man, b)["diffs"]] == ["workdir_files", "workdir_content"]
    w = H.compare_workdirs(man, b, copies=a)                           # non-audio copies: unified diff
    assert "+{\"notes\": [1]}" in w["diffs"][1]["detail"]["notes.json"]


def test_stream_interleave_whitelisted_only_for_moved_stderr_lines(golden):
    recs, _ = golden
    rec = copy.deepcopy(recs["maestro"])
    i = next(i for i, m in enumerate(rec["messages"]) if m["role"] == "tool_response")
    rec["messages"][i]["content"] = "slot 1 a\nslot 2 b\nerror: amount 1.38 out of range\nslot 3 c\nslot 4 d"
    step = next(k for k, (a, _) in enumerate(R.record_steps(rec["messages"])) if a > i)

    def run(out):
        body = _request(rec, step)
        next(m for m in body["messages"] if m["role"] == "tool")["content"] = out
        return R.compare_request(body, rec, step, _digests(rec))
    moved = run("slot 1 a\nslot 2 b\nslot 3 c\nerror: amount 1.38 out of range\nslot 4 d")
    assert moved["status"] == "match" and [w["kind"] for w in moved["whitelisted"]] == ["stream_interleave"]
    assert run("slot 1 a\nslot 3 c\nslot 2 b\nerror: amount 1.38 out of range\nslot 4 d")["status"] == "diff"  # stdout reorder
    assert run("slot 1 a\nslot 2 b\nslot 3 c\nerror: amount 1.39 out of range\nslot 4 d")["status"] == "diff"  # other text
    assert run("slot 1 a\nslot 2 b\nslot 3 c\nslot 4 d")["status"] == "diff"                                   # line lost


# -- reasoning echo (goldens/reasoning, captured by opencode/replay/reasoning_probe.py) -------------

REASONING_GOLDENS = G.GOLDENS / "reasoning"


@pytest.mark.parametrize("mode", ["reasoning_content", "reasoning", "think_tag"])
def test_reasoning_echo_rule(mode):
    """OpenCode 1.19.3 + @ai-sdk/openai-compatible: every earlier assistant turn of the same session is
    re-sent with its reasoning verbatim as `reasoning_content` (whichever delta field streamed it);
    content stays the visible text ("" when none). Reasoning streamed as a <think> block in content is
    kept verbatim in content. Sessions don't share reasoning; a task result carries only the
    subagent's final content (so a <think> block in content leaks into it)."""
    g = json.loads((REASONING_GOLDENS / f"{mode}.json").read_text())
    assert g["opencode_exit"] == 0
    from opencode.replay.reasoning_probe import MARK, reasoning_text
    for q in g["requests"]:
        turns = q["assistant_turns"]
        assert len(turns) == q["step"]                          # every earlier turn, none dropped
        agent = "main" if q["route"] == "main" else "sub"
        for j, t in enumerate(turns):
            if mode == "think_tag":
                assert t["reasoning_fields"] == [] and "reasoning_content" not in t["keys"]
                assert t["content"].startswith(f"<think>\n{reasoning_text(agent, j)}\n</think>\n\n")
            else:
                assert t["reasoning_fields"] == ["reasoning_content"] and t["reasoning_content_verbatim"]
                assert not t["content_has_reasoning"] and not t["content_has_think_tag"]
                assert isinstance(t["content"], str)
        for tr in q["task_results"]:
            assert (MARK in tr) == (mode == "think_tag")
    assert any(q["task_results"] for q in g["requests"])        # the task boundary was crossed
