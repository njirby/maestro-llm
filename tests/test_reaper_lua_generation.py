import pretty_midi
import re

from maestro.render.reaper import generate_lua


def _note(pitch: int, start: float, end: float, velocity: int) -> pretty_midi.Note:
    return pretty_midi.Note(pitch=pitch, start=start, end=end, velocity=velocity)


def test_generate_lua_preserves_velocity_and_timing_precision():
    notes = [
        _note(60, 0.0, 0.25, 33),
        _note(62, 1 / 3, 2 / 3, 70),
        _note(64, 1.125, 1.5, 111),
    ]
    lua = generate_lua(
        notes=notes,
        bpm=123.456789,
        wav_path="/tmp/out.wav",
        rpp_path="/tmp/out.rpp",
        track_name="Lead",
        tail_s=2.0,
    )

    assert "local bpm,ppb,ts = 123.456789,960,4" in lua
    assert 'n("C4",m(1,1,"0"),"e",33)' in lua
    assert 'n("D4",m(1,1,"qt"),"qt",70)' in lua
    m = re.search(r'n\("E4",m\(1,3,"8t"\),"([^"]+)",111\)', lua)
    assert m is not None
    dur_expr = m.group(1)
    tok = {
        "16t": 1.0 / 6.0,
        "t": 1.0 / 8.0,
        "s": 1.0 / 4.0,
        "8t": 1.0 / 3.0,
        "e": 1.0 / 2.0,
        "qt": 2.0 / 3.0,
        "e.": 3.0 / 4.0,
        "q": 1.0,
        "q.": 1.5,
        "h": 2.0,
        "h.": 3.0,
        "w": 4.0,
    }
    beats = sum(tok[p] for p in dur_expr.split("+"))
    assert abs(beats - 0.75) < 0.2
    assert "CreateNewMIDIItemInProj(track,0.0,3.5000,false)" in lua
    assert "local function m(bar,beat,offset_tok)" in lua
    assert "local function B(tok)" in lua


def test_generate_lua_escapes_track_name_quotes_and_backslashes():
    lua = generate_lua(
        notes=[_note(60, 0.0, 0.5, 90)],
        bpm=120.0,
        wav_path=r'/tmp/path with "quotes"/x.wav',
        rpp_path=r'/tmp/path with "quotes"/x.rpp',
        track_name='Lead "A" \\B',
    )

    assert '"P_NAME","Lead \\"A\\" \\\\B",true' in lua
    assert 'RENDER_FILE","/tmp/path with \\"quotes\\"/",true' in lua
    assert 'Main_SaveProjectEx(0,"/tmp/path with \\"quotes\\"/x.rpp",0)' in lua


def test_generate_lua_respects_min_duration_override():
    lua = generate_lua(
        notes=[_note(60, 0.0, 0.4, 90)],
        bpm=120.0,
        wav_path="/tmp/out.wav",
        rpp_path="/tmp/out.rpp",
        tail_s=0.5,
        min_duration_s=7.25,
    )

    assert "CreateNewMIDIItemInProj(track,0.0,7.2500,false)" in lua


def test_generate_lua_start_quantization_can_carry_to_next_beat():
    # At 120 BPM, 0.495s is ~0.99 beat and should quantize to beat 2 with offset "0".
    lua = generate_lua(
        notes=[_note(60, 0.495, 0.745, 90)],
        bpm=120.0,
        wav_path="/tmp/out.wav",
        rpp_path="/tmp/out.rpp",
    )

    assert 'n("C4",m(1,2,"0"),"e",90)' in lua


def test_generate_lua_long_duration_uses_composite_tokens():
    # At 120 BPM, 2.8s == 5.6 beats; nearest composite should be w+q. (5.5 beats).
    lua = generate_lua(
        notes=[_note(60, 0.0, 2.8, 100)],
        bpm=120.0,
        wav_path="/tmp/out.wav",
        rpp_path="/tmp/out.rpp",
    )

    m = re.search(r'n\("C4",m\(1,1,"0"\),"([^"]+)",100\)', lua)
    assert m is not None
    dur_expr = m.group(1)
    assert "w" in dur_expr

    tok = {
        "16t": 1.0 / 6.0, "t": 1.0 / 8.0, "s": 1.0 / 4.0, "8t": 1.0 / 3.0,
        "e": 1.0 / 2.0, "qt": 2.0 / 3.0, "e.": 3.0 / 4.0, "q": 1.0,
        "q.": 1.5, "h": 2.0, "h.": 3.0, "w": 4.0,
    }
    beats = sum(tok[p] for p in dur_expr.split("+"))
    assert abs(beats - 5.6) < 0.2
