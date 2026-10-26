"""Extract downloaded HF parquet audio sources (data/ear_corpus/raw/<name>) into clip files + manifest rows."""
import glob
import io
import os

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf

from .build_manifest import CORPUS, MAX_S, segment

CAPS = {"esc50": 2000, "librispeech": 3000, "nsynth": 6000, "clotho": 3000, "fma_small": 8000, "audioset_bal": 25000}


def _kind_and_meta(name, row):
    if name == "esc50":
        return "sound", {"labels": [row["category"].replace("_", " ")]}
    if name == "librispeech":
        return "speech", {"text": row["text"], "speaker_id": row["speaker_id"]}
    if name == "fma_small":
        return "music", {"genres": row.get("genres"), "title": row.get("title")}
    if name == "clotho":
        return "sound", {"captions": row.get("raw_text")}
    if name == "audioset_bal":
        labels = list(row.get("human_labels") or [])
        kind = "speech" if "Speech" in labels and "Music" not in labels else "music" if "Music" in labels else "sound"
        return kind, {"labels": labels, "video_id": row.get("video_id")}
    if name == "nsynth":
        keep = ("instrument_family_str", "instrument_source_str", "pitch", "velocity", "qualities_str", "note_str")
        return "note", {k: row[k] for k in keep if k in row}
    raise KeyError(name)


def parquet_rows(name, rng):
    files = sorted(glob.glob(str(CORPUS / "raw" / name / "**/*.parquet"), recursive=True))
    out_dir = CORPUS / "clips" / name
    out_dir.mkdir(parents=True, exist_ok=True)
    rows, cap = [], CAPS[name]
    for fpath in files:
        for batch in pq.ParquetFile(fpath).iter_batches(batch_size=64):
            for row in batch.to_pylist():
                if len(rows) >= cap:
                    return rows
                audio = row.get("audio") or {}
                data = audio.get("bytes")
                if not data and audio.get("array") is not None:  # NSynth: decoded float array, not file bytes
                    buf = io.BytesIO()
                    sf.write(buf, np.asarray(audio["array"], dtype=np.float32), audio["sampling_rate"], format="FLAC")
                    data, audio = buf.getvalue(), {"path": "x.flac"}
                if not data:
                    continue
                try:
                    info = sf.info(io.BytesIO(data))
                except Exception:
                    continue
                if info.duration < 1.0:
                    continue
                kind, meta = _kind_and_meta(name, row)
                idx = len(rows)
                if info.duration <= MAX_S + 0.05:
                    ext = os.path.splitext(audio.get("path") or "")[1].lower() or ".flac"
                    if ext not in (".wav", ".flac", ".mp3", ".ogg"):
                        ext = "." + info.format.lower()
                    path = out_dir / f"{idx:06d}{ext}"
                    path.write_bytes(data)
                    dur, sr, ch = info.duration, info.samplerate, info.channels
                else:
                    tmp = out_dir / f"{idx:06d}.tmp"
                    tmp.write_bytes(data)
                    seg = segment(str(tmp), rng)
                    tmp.unlink()
                    if seg is None:
                        continue
                    y, sr, off = seg
                    path = out_dir / f"{idx:06d}.flac"
                    sf.write(str(path), y, sr, format="FLAC")
                    dur, ch, meta["offset_s"] = len(y) / sr, y.shape[1], round(off, 2)
                rows.append({"id": f"{name}/{idx:06d}", "source": name, "kind": kind, "path": str(path),
                             "dur": round(dur, 2), "sr": sr, "channels": ch, "meta": meta})
    return rows
