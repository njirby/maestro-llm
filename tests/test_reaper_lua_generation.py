import pretty_midi

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
    assert 'n("E4",m(1,3,"8t"),"e.",111)' in lua
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
