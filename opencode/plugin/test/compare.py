"""Compares stock vs plugin captures written by run_matrix.sh (usage: compare.py <workdir>)."""
import json, glob, os, base64, sys
W = sys.argv[1]
def reqs(f): return [json.loads(l)['body'] for l in open(f) if json.loads(l)['body'].get('tools')]
ref = json.loads(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../../docs/opencode_capture/captured.jsonl')).readline())['body']['tools']
ref_read = [t for t in ref if t['function']['name'] == 'read'][0]
cases = sorted({os.path.basename(f).split('__')[1][:-6] for f in glob.glob(f'{W}/cap/cfg_*__*.jsonl')})
for c in cases:
    s, p = reqs(f'{W}/cap/cfg_stock__{c}.jsonl'), reqs(f'{W}/cap/cfg_plugin__{c}.jsonl')
    tools_same = all(json.dumps(r['tools']) == json.dumps(s[0]['tools']) for r in s + p)
    read_ref = all(json.dumps([t for t in r['tools'] if t['function']['name'] == 'read'][0]) == json.dumps(ref_read) for r in p)
    same = json.dumps(s) == json.dumps(p)
    audio = c in ('wav', 'rel_wav', 'mp3', 'flac', 'ogg', 'par')
    assert tools_same and read_ref and (same or audio), c
    print(f'{c:12s} nreq stock={len(s)} plugin={len(p)} tools_identical={tools_same} read_eq_captured.jsonl={read_ref} full_body_identical={same}')
    if not same:
        last = p[-1]['messages']
        sl = s[-1]['messages']
        for i, (a, b) in enumerate(zip(sl, last)):
            if a != b: print('   first diff at msg', i)
            if a != b: break
        for m in last[1:]:
            cont = m.get('content')
            if isinstance(cont, list):
                for part in cont:
                    if part.get('type') == 'input_audio':
                        raw = base64.b64decode(part['input_audio']['data'])
                        print(f'   input_audio format={part["input_audio"]["format"]} bytes={len(raw)} head={raw[:4]}')
                        if part['input_audio']['format'] == 'mp3': assert raw == open(f'{W}/data/tone.mp3', 'rb').read()
                        elif c == 'ogg': assert raw[:4] == b'RIFF' and raw[8:12] == b'WAVE'  # lossy source: header only
                        else: assert raw == open(f'{W}/data/tone.wav', 'rb').read(), 'audio bytes differ from source'
            print('   ', m['role'], json.dumps(cont)[:160] if cont is not None else m.get('tool_calls'))
