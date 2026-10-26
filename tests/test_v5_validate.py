"""maestro/v5/validate.py: the real record contract, good and broken records."""
import copy
import json

import pytest

from maestro.v5.record import Record
from maestro.v5.validate import validate_record, validate_rollout
from scripts import opencode_contract as oc


@pytest.fixture
def wav(tmp_path):
    p = tmp_path / "a.wav"
    p.write_bytes(b"RIFF")
    return str(p)


def sub_record(wav, agent="transcriber", prompt="Target: /w/target.wav. Write the melody to /w/notes.json."):
    rec = Record(agent, f"r.{agent}", "/w", prompt)
    rec.audio_step(["/w/target.wav"], [wav])
    rec.step([oc.write_call("/w/notes.json", "{}")], [oc.WRITE_OK])
    return rec.finish("/w/notes.json: 3 notes.")


def main_record(wav, subs):
    rec = Record("maestro", "r.main", "/w", "Recreate the sound in target.wav in Vital.")
    rec.step([oc.skill_call("vital")], ["<skill_content name=\"vital\">...</skill_content>"])
    rec.audio_step(["/w/target.wav", "/w/a.wav"], [wav, wav])          # parallel reads: "<audio><audio>"
    rec.say("Notes first.")
    calls, resps, links = [], [], []
    for k, s in enumerate(subs):
        sid = oc.session_id(seed=f"r/{k}")
        calls.append(oc.task_call(f"task {k}", s["messages"][1]["content"], "transcriber"))
        resps.append(oc.task_result(sid, s["messages"][-1]["content"]))
        links.append({"agent": "transcriber", "record_id": s["id"], "session_id": sid})
    rec.step(calls, resps)
    rec.step([oc.bash_call("python3 $VITAL_SKILL/scripts/render.py preset.json notes.json a.wav")],
             ["a.wav  (notes 1.00 s, 120 bpm)\n"])
    return rec.finish("Done."), links


def test_good_records(wav):
    sub = sub_record(wav)
    main, links = main_record(wav, [sub])
    assert validate_record(sub) == []
    assert validate_record(main) == []
    assert validate_rollout(main, [sub], links) == []


def test_parallel_subagents(wav):
    subs = [sub_record(wav, prompt=f"Target: /w/target.wav. Write the melody to /w/n{k}.json.") for k in range(2)]
    subs[1]["id"] = "r.transcriber2"
    main, links = main_record(wav, subs)
    assert validate_rollout(main, subs, links) == []


def broken(rec, fn):
    r = copy.deepcopy(rec)
    fn(r)
    return r


@pytest.mark.parametrize("mutate, needle", [
    (lambda r: r["messages"].pop(8), "attachment"),                              # attachment user message dropped
    (lambda r: r["messages"][8].update(content=oc.MEDIA_ATTACHED_TEXT + "<audio>"), "attachment"),  # 1 of 2
    (lambda r: r["audios"].pop(), "audios"),
    (lambda r: r["messages"][6].update(content="<audio>"), "AUDIO_READ_TEXT"),  # audio in a tool response
    (lambda r: r["messages"][2].update(content=json.dumps({"name": "nosuch", "arguments": {}})), "not offered"),
    (lambda r: r["messages"][2].update(content=oc.tool_call("skill", {"name": "vital", "x": 1})), "unknown args"),
    (lambda r: r["messages"].pop(), "end with assistant"),
    (lambda r: r["messages"].insert(-1, {"role": "assistant", "content": "x"}), "in a row"),
    (lambda r: r["messages"].insert(3, {"role": "tool_response", "content": "x"}), "responses"),
    (lambda r: r["messages"][0].update(content=r["messages"][0]["content"].replace("Oct 25", "Oct 26")), "system"),
    (lambda r: r.update(tools="[]"), "tools"),
    (lambda r: r["messages"][1].update(content="<audio>hi"), "first user"),
    (lambda r: r.update(extra=1), "keys"),
])
def test_broken_main_is_rejected(wav, mutate, needle):
    sub = sub_record(wav)
    main, _ = main_record(wav, [sub])
    errs = validate_record(broken(main, mutate))
    assert errs and any(needle in e for e in errs), errs


def test_missing_audio_file(wav, tmp_path):
    sub = sub_record(wav)
    sub["audios"] = [str(tmp_path / "gone.wav")]
    assert any("missing audio" in e for e in validate_record(sub))
    assert validate_record(sub, check_files=False) == []


def test_rollout_links(wav):
    sub = sub_record(wav)
    main, links = main_record(wav, [sub])
    # dispatch prompt != subagent's first user message
    bad = broken(sub, lambda r: r["messages"][1].update(content="Target: elsewhere."))
    assert any("matching subagent" in e for e in validate_rollout(main, [bad], links))
    # task result text != subagent's final text
    bad = broken(sub, lambda r: r["messages"][-1].update(content="other"))
    assert any("final text" in e for e in validate_rollout(main, [bad], links))
    # session id not linked to the record
    assert any("not linked" in e for e in validate_rollout(main, [sub], [{**links[0], "record_id": "zzz"}]))
    # orphan subagent record
    extra = broken(sub, lambda r: r.update(id="orphan"))
    extra["messages"][1]["content"] = "Target: /w/target.wav. Write the melody to /w/other.json."
    assert any("no task call" in e for e in validate_rollout(main, [sub, extra], links))
