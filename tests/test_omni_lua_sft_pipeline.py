from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _touch_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def test_build_omni_lua_sft_dataset_smoke(tmp_path: Path):
    audio_dir = tmp_path / "wavs"
    lua_dir = tmp_path / "luas"
    out_dir = tmp_path / "prepared"
    prompt_file = tmp_path / "prompt.txt"

    _write(prompt_file, "Listen to this mp3 and write a REAPER Lua script using Vital default preset.")

    _touch_bytes(audio_dir / "song_a.mp3", b"ID3dummy-a")
    _touch_bytes(audio_dir / "song_b.mp3", b"ID3dummy-b")
    _write(
        lua_dir / "song_a.lua",
        'reaper.TrackFX_AddByName(track,"Vital",false,1)\nreaper.MIDI_InsertNote(take,false,false,0,120,0,60,90,false)\n',
    )
    _write(
        lua_dir / "song_b.lua",
        'reaper.TrackFX_AddByName(track,"Vital",false,1)\nreaper.MIDI_InsertNote(take,false,false,0,240,0,64,100,false)\n',
    )

    cmd = [
        sys.executable,
        "scripts/build_omni_lua_sft_dataset.py",
        "--audio-dir",
        str(audio_dir),
        "--lua-dir",
        str(lua_dir),
        "--prompt-file",
        str(prompt_file),
        "--out-dir",
        str(out_dir),
        "--val-ratio",
        "0.5",
        "--seed",
        "7",
        "--require-lua-markers",
    ]
    subprocess.run(cmd, check=True)

    all_rows = [json.loads(x) for x in (out_dir / "all.jsonl").read_text(encoding="utf-8").splitlines()]
    train_rows = [json.loads(x) for x in (out_dir / "train.jsonl").read_text(encoding="utf-8").splitlines()]
    val_rows = [json.loads(x) for x in (out_dir / "val.jsonl").read_text(encoding="utf-8").splitlines()]
    stats = json.loads((out_dir / "stats.json").read_text(encoding="utf-8"))

    assert len(all_rows) == 2
    assert len(train_rows) + len(val_rows) == 2
    assert stats["written"]["all"] == 2
    assert stats["written"]["train"] + stats["written"]["val"] == 2

    row = all_rows[0]
    assert row["messages"][0]["role"] == "user"
    assert row["messages"][1]["role"] == "assistant"
    assert "<audio>" in row["messages"][0]["content"]
    assert "MIDI_InsertNote" in row["messages"][1]["content"]
    assert len(row["audios"]) == 1
    assert row["audios"][0].endswith(".mp3")

    smoke_cmd = [
        sys.executable,
        "scripts/smoke_test_omni_lua_dataset.py",
        "--dataset-dir",
        str(out_dir),
        "--split",
        "all",
        "--max-rows",
        "32",
        "--require-lua-markers",
    ]
    subprocess.run(smoke_cmd, check=True)


def test_training_launcher_and_setup_smoke(tmp_path: Path):
    dataset_dir = tmp_path / "dataset"
    _write(dataset_dir / "train.jsonl", json.dumps({"id": "x", "messages": [], "audios": []}) + "\n")
    _write(dataset_dir / "val.jsonl", json.dumps({"id": "y", "messages": [], "audios": []}) + "\n")

    fake_bin = tmp_path / "bin"
    fake_swift = fake_bin / "swift"
    fake_bin.mkdir(parents=True, exist_ok=True)
    fake_swift.write_text("#!/usr/bin/env bash\necho \"fake swift $*\"\n", encoding="utf-8")
    fake_swift.chmod(0o755)

    env = os.environ.copy()
    env["PATH"] = f"{fake_bin}:{env.get('PATH', '')}"

    dry_run_cmd = [
        sys.executable,
        "scripts/train_qwen25_omni_lora.py",
        "--profile",
        "smoke",
        "--dataset-dir",
        str(dataset_dir),
        "--dry-run",
    ]
    subprocess.run(dry_run_cmd, check=True, env=env)

    setup_smoke_cmd = [
        sys.executable,
        "scripts/smoke_test_qwen25_omni_lora_setup.py",
        "--dataset-dir",
        str(dataset_dir),
        "--max-rows",
        "16",
    ]
    subprocess.run(setup_smoke_cmd, check=True, env=env)
