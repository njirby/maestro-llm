# written by the agent, first render of the session
import json, struct
_B64 = ".ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+"
def _b64(b):
    out = []
    for i in range(((len(b)*8)+5)//6):
        bit = i*6; y, o = bit >> 3, bit & 7
        w = b[y] | (b[y+1]<<8) if y+1 < len(b) else (b[y] if y < len(b) else 0)
        out.append(_B64[(w>>o)&0x3F])
    return str(len(b)) + "." + "".join(out)
def build_state(preset):
    jb = json.dumps(preset, separators=(",",":")).encode() + b"\x00"
    chunk = jb + b"\x00"*16 + b"JUCEPrivateData"
    body = 4+4+4+4+4+128+4+len(chunk)
    icomp = (b"VstW" + struct.pack(">III",8,1,0) + b"CcnK" + struct.pack(">i",body)
             + b"FBCh" + struct.pack(">i",2) + b"Vita" + struct.pack(">i",0x00010600)
             + struct.pack(">i",0) + b"\x00"*128 + struct.pack(">i",len(chunk)) + chunk)
    xml = ('<?xml version="1.0" encoding="UTF-8"?>\n<VST3PluginState><IComponent>'
           + _b64(icomp) + "</IComponent></VST3PluginState>").encode() + b"\x00"
    return struct.pack("<II", 0x21324356, len(xml)) + xml
