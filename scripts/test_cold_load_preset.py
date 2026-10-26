"""Test: build .rpp with Vital + large preset, cold-open it, verify."""
import sys, json, os, re, time
sys.path.insert(0, '/home/nate/Documents/maestro-llm')
import reapy
from maestro.reaper.vital_tools import VitalController

# Step 1: in running REAPER, get default track chunk and default preset
with reapy.inside_reaper():
    api = reapy.reascript_api
    if int(api.CountTracks(0)) == 0:
        api.InsertTrackAtIndex(0, True)
    track = api.GetTrack(0, 0)
    if int(api.TrackFX_GetCount(track)) == 0:
        api.TrackFX_AddByName(track, 'Vital', False, 1)
        time.sleep(2.0)
    rpr = {f'RPR_{fn}': getattr(api, fn) for fn in dir(api) if not fn.startswith('_')}
    vc = VitalController(_rpr=rpr)
    vc.discover()
    preset = vc.get_preset()

    wt_lib = json.load(open('data/wavetable_lib.json'))
    target = next(w for w in wt_lib if w.get('name') == 'hola')
    preset['settings']['wavetables'][0] = target
    new_encoded = vc._encode_chunk(vc._chunk_prefix, preset, vc._chunk_suffix)
    new_lines = '\n'.join(new_encoded[i:i+128] for i in range(0, len(new_encoded), 128))
    print(f'encoded size: {len(new_encoded)}')

    res = api.GetTrackStateChunk(track, '', 20*1024*1024, False)
    track_chunk = res[2] if isinstance(res,(tuple,list)) and len(res)>2 else ''
    pat = re.compile(r'(<VST "VST3i: Vital[^\n]*\n[^\n]+\n)([A-Za-z0-9+/=\n]+?)(\n>)', re.DOTALL)
    modified = pat.sub(lambda m: m.group(1) + new_lines + m.group(3), track_chunk, count=1)
    print(f'modified track chunk size: {len(modified)}')

rpp = (
    '<REAPER_PROJECT 0.1 "7.31" 1742311700\n'
    '  SAMPLERATE 48000 0 0\n'
    '  TEMPO 120 4 4\n'
    + modified +
    '>\n'
)
out = '/tmp/vital_cold_load.rpp'
with open(out, 'w') as f:
    f.write(rpp)
print(f'wrote {out} ({os.path.getsize(out)} bytes)')
