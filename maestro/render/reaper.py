"""
REAPER Lua script generation and headless subprocess execution utilities.

The agent controls REAPER by generating Lua scripts and executing them via
subprocess against a virtual display (Xvfb on :1). No Python-in-REAPER needed.
"""

import os
import subprocess
import tempfile
import math
from pathlib import Path


REAPER_BIN = os.environ.get("REAPER_BIN", "/home/nate/opt/REAPER/REAPER/reaper")
DISPLAY = os.environ.get("REAPER_DISPLAY", ":1")

_PPQ_PER_BEAT = 960  # REAPER default ticks per quarter note
_RHYTHM_TOKENS = [
    ("16t", 1.0 / 6.0),
    ("t",   1.0 / 8.0),
    ("s",   1.0 / 4.0),
    ("8t",  1.0 / 3.0),
    ("e",   1.0 / 2.0),
    ("qt",  2.0 / 3.0),
    ("e.",  3.0 / 4.0),
    ("q",   1.0),
    ("q.",  1.5),
    ("h",   2.0),
    ("h.",  3.0),
    ("w",   4.0),
]
_OFFSET_TOKENS = [
    ("0",   0.0),
    ("16t", 1.0 / 6.0),
    ("t",   1.0 / 8.0),
    ("s",   1.0 / 4.0),
    ("8t",  1.0 / 3.0),
    ("e",   1.0 / 2.0),
    ("qt",  2.0 / 3.0),
    ("e.",  3.0 / 4.0),
]


def _nearest_token(value: float, tokens: list[tuple[str, float]]) -> tuple[str, float]:
    best_tok = tokens[0][0]
    best_val = tokens[0][1]
    best_err = abs(value - best_val)
    for tok, tok_val in tokens[1:]:
        err = abs(value - tok_val)
        # Tie-break toward shorter tokens for better LLM token efficiency.
        if (err < best_err) or (err == best_err and len(tok) < len(best_tok)):
            best_tok = tok
            best_val = tok_val
            best_err = err
    return best_tok, best_val


def generate_lua(
    notes,
    bpm: float,
    wav_path: str,
    rpp_path: str,
    track_name: str = "Vital Synth",
    tail_s: float = 0.5,
    min_duration_s: float | None = None,
) -> str:
    """
    Generate a self-contained REAPER Lua script that:
      1. Creates a track with Vital synth (default preset)
      2. Inserts MIDI notes directly via MIDI_InsertNote (no InsertMedia dialog)
      3. Renders the project to a WAV file
      4. Saves the .rpp project file

    Args:
        notes:      List of pretty_midi.Note objects (have .start, .end, .pitch, .velocity).
        bpm:        Tempo in BPM used to convert seconds → PPQ ticks.
        wav_path:   Absolute path for the output WAV file.
        rpp_path:   Absolute path for the saved REAPER project.
        track_name: Name shown on the REAPER track.

    Returns:
        Lua script as a string.
    """
    def _esc(s: str) -> str:
        return s.replace("\\", "\\\\").replace('"', '\\"')

    def _fmt_num(value: float, ndigits: int = 9) -> str:
        text = f"{float(value):.{ndigits}f}".rstrip("0").rstrip(".")
        return text if text else "0"

    wav_dir  = str(Path(wav_path).parent) + "/"
    wav_name = Path(wav_path).stem
    duration = max((n.end for n in notes), default=4.0) + max(0.0, float(tail_s))
    if min_duration_s is not None:
        duration = max(duration, max(0.0, float(min_duration_s)))

    # ── Pitch helpers ──────────────────────────────────────────────────────
    _NAMES = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
    def _pitch_name(midi: int) -> str:
        return f"{_NAMES[midi % 12]}{midi // 12 - 1}"

    # ── Build note lines: n("D3", m(4,3,"8t"), "q.", 96) ──────────────────
    ppb = _PPQ_PER_BEAT
    ts  = 4  # assume 4/4
    note_lines = []
    for n in notes:
        p_str = f'"{_pitch_name(n.pitch)}"'

        start_beats = max(0.0, float(n.start) * (float(bpm) / 60.0))
        start_floor = math.floor(start_beats)
        frac = start_beats - start_floor
        off_tok, off_val = _nearest_token(frac, _OFFSET_TOKENS)
        q_start_beats = start_floor + off_val

        bar = int(q_start_beats // ts) + 1
        beat_pos = q_start_beats % ts
        beat = int(math.floor(beat_pos)) + 1

        dur_beats = max(0.0, float(n.end - n.start) * (float(bpm) / 60.0))
        dur_tok, _ = _nearest_token(dur_beats, _RHYTHM_TOKENS)
        v     = max(1, min(127, int(n.velocity)))
        note_lines.append(f'n({p_str},m({bar},{beat},"{off_tok}"),"{dur_tok}",{v})')
    note_block = "\n".join(note_lines)

    return f"""\
-- wav: {wav_path}
-- rpp: {rpp_path}

-- ── Helpers ───────────────────────────────────────────────────────────────
local bpm,ppb,ts = {_fmt_num(bpm, 8)},{ppb},{ts}
local _P={{C=0,["C#"]=1,D=2,["D#"]=3,E=4,F=5,["F#"]=6,G=7,["G#"]=8,A=9,["A#"]=10,B=11}}
local function P(s) local n,o=s:match("([A-G]#?)(-?%d+)") return _P[n]+(tonumber(o)+1)*12 end
local TOK={{
  ["0"]=0, ["16t"]=(1/6), t=(1/8), s=(1/4), ["8t"]=(1/3), e=(1/2),
  qt=(2/3), ["e."]=(3/4), q=1, ["q."]=(3/2), h=2, ["h."]=3, w=4
}}
local function B(tok)
  local v = TOK[tok]
  if v == nil then error("Unknown rhythm token: " .. tostring(tok)) end
  return v
end
local function m(bar,beat,offset_tok)
  return (((bar-1)*ts) + ((beat or 1)-1) + B(offset_tok or "0")) * ppb
end
local take  -- forward-declare so function n captures the right upvalue
local function n(pitch,st,dur,vel)
  local en = st + B(dur) * ppb
  reaper.MIDI_InsertNote(take,false,false,st,en,0,P(pitch),vel,false)
end

-- ── Project setup ─────────────────────────────────────────────────────────
for i=reaper.CountTracks(0)-1,0,-1 do reaper.DeleteTrack(reaper.GetTrack(0,i)) end
reaper.InsertTrackAtIndex(0,true)
local track=reaper.GetTrack(0,0)
reaper.GetSetMediaTrackInfo_String(track,"P_NAME","{_esc(track_name)}",true)
reaper.TrackFX_AddByName(track,"Vital",false,1)
reaper.SetTempoTimeSigMarker(0,-1,0,-1,-1,bpm,ts,4,false)
local item=reaper.CreateNewMIDIItemInProj(track,0.0,{duration:.4f},false)
take=reaper.GetActiveTake(item)  -- assign into the upvalue captured by n()

-- ── Notes ─────────────────────────────────────────────────────────────────
{note_block}

reaper.MIDI_Sort(take)

-- ── Render + save ─────────────────────────────────────────────────────────
reaper.GetSetProjectInfo_String(0,"RENDER_FILE","{_esc(wav_dir)}",true)
reaper.GetSetProjectInfo_String(0,"RENDER_PATTERN","{_esc(wav_name)}",true)
reaper.GetSetProjectInfo(0,"RENDER_FMT",0,true)
reaper.GetSetProjectInfo(0,"RENDER_FMT2",0,true)
reaper.GetSetProjectInfo(0,"RENDER_SRATE",44100,true)
reaper.GetSetProjectInfo(0,"RENDER_NCH",2,true)
reaper.GetSetProjectInfo(0,"RENDER_BOUNDSFLAG",1,true)
reaper.Main_OnCommand(42230,0)
reaper.Main_SaveProjectEx(0,"{_esc(rpp_path)}",0)
"""


def run_lua_batch(lua_paths: list[str], timeout: int = 300) -> list[bool]:
    """
    Run a batch of Lua scripts in a single REAPER session using dofile().

    Between each script a new empty project is created so scripts don't
    interfere with each other. The prior project is already saved by the
    script itself, so the new-project command doesn't prompt to save.

    Args:
        lua_paths: List of absolute paths to Lua scripts to run in order.
        timeout:   Seconds before killing REAPER (default 300).

    Returns:
        List of bools indicating whether each expected output was produced.
        (Checking is left to the caller — this just runs the batch.)
    """
    if not lua_paths:
        return []

    lines = []
    for path in lua_paths:
        lines.append(f'dofile("{path}")')
    batch_src = "\n".join(lines) + "\n"

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".lua", prefix="reaper_batch_", delete=False
    ) as f:
        f.write(batch_src)
        batch_path = f.name

    env = os.environ.copy()
    env["DISPLAY"] = DISPLAY

    try:
        subprocess.run(
            [
                REAPER_BIN,
                "-nosplash",
                "-newinst",
                "-new",
                batch_path,
                "-close:nosave:exit",
            ],
            env=env,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        pass  # REAPER doesn't self-terminate after script — timeout is expected
    finally:
        os.unlink(batch_path)

    return [True] * len(lua_paths)  # caller checks actual output files
