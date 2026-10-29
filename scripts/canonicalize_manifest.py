"""Canonicalize a v5 manifest's targets: a key the target changes from init but whose change provably has no effect on
the target's sound (its one-key revert renders bit-identical through the sample's own notes, and all such reverts
together do too: maestro/v5/wt_rank.audible_changes) is reset to its init value in the stored preset. Then the target
wav is re-rendered from the stored preset, and the row gets {"canonical": true, "canonical_reset": [...]}.

After this no rollout has to decide what is inaudible: every remaining change from init is audible, the agent edits
exactly those, and the GT gate checks all of them.

  .venv/bin/python scripts/canonicalize_manifest.py outputs/v5_manifest/<name>      (rewrites manifest.jsonl in place;
                                                                                      the original is kept as .orig)
  .venv/bin/python scripts/canonicalize_manifest.py outputs/v5_manifest/<name> --check   (proof: 0 no-effect keys)
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from maestro import notation  # noqa: E402
from maestro.sounddiff.renderer import ContainerRenderer  # noqa: E402
from maestro.v5 import planner as pl, wt_rank  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("manifest_dir", type=Path)
    ap.add_argument("--check", action="store_true", help="only verify: every changed key must be audible")
    ap.add_argument("--jobs", type=int, default=12)
    a = ap.parse_args()
    mf = a.manifest_dir / "manifest.jsonl"
    rows = [json.loads(l) for l in mf.read_text().splitlines() if l.strip()]
    init = json.loads(Path(rows[0]["init_preset_path"]).read_text())
    bad, out = 0, []
    with ContainerRenderer(a.manifest_dir / "canonical_cache", jobs=a.jobs) as r:
        r._session()
        for row in rows:
            gt = json.loads(Path(row["gt_preset_path"]).read_text())
            notes = {"bpm": row["bpm"], "notes": notation.to_seconds(Path(row["notes_path"]).read_text())[0]}
            keys = pl.live_changed_keys(gt, init)
            res = wt_rank.audible_changes(r, gt, init, keys, notes)
            if a.check:
                bad += bool(res["inaudible"])
                print(f"{row['id']}: {len(keys)} changed, {len(res['inaudible'])} no-effect "
                      f"{res['inaudible'][:6] if res['inaudible'] else ''}", flush=True)
                continue
            for k in res["inaudible"]:
                gt["settings"][k] = init["settings"].get(k, pl.params()[k]["default"])
            Path(row["gt_preset_path"]).write_text(json.dumps(gt, separators=(",", ":")))
            # the target stays the render of exactly the stored preset (same take convention as the manifest:
            # the GT's own random phase, one render)
            wav = r.render({"t": gt}, notes)["t"]
            import soundfile as sf
            sf.write(row["target_wav_path"], wav, 44100, subtype="FLOAT")
            # each remaining key's audible impact (its one-key revert distance): the planner's within-family
            # tie-break and coarse/fine split
            impact = {k: v for k, v in res["scores"].items() if k not in res["inaudible"]}
            reset = sorted(set(row.get("canonical_reset", [])) | set(res["inaudible"]))   # re-runs keep earlier resets
            out.append({**row, "canonical": True, "canonical_reset": reset, "impact": impact})
            print(f"{row['id']}: {len(keys)} changed, reset {len(res['inaudible'])} no-effect keys", flush=True)
    if a.check:
        print(f"{'PASS' if not bad else 'FAIL'}: {bad}/{len(rows)} targets still have no-effect changes")
        sys.exit(1 if bad else 0)
    shutil.copy(mf, mf.with_suffix(".jsonl.orig"))
    mf.write_text("".join(json.dumps(r) + "\n" for r in out))


if __name__ == "__main__":
    main()
