"""
Smoke tests — fast, no GPU or model downloads required.

Run:
    python -m pytest tests/ -v
    # or just:
    python tests/test_smoke.py
"""

import math
import os
import sys
import tempfile
import pathlib
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---------------------------------------------------------------------------
# Audio processing
# ---------------------------------------------------------------------------

def test_load_and_chunk_exact_length():
    """Audio that is exactly one chunk long → one chunk returned."""
    from maestro.audio.processing import load_and_chunk, CHUNK_SAMPLES, CLAP_SAMPLE_RATE

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        tmp_path = f.name

    try:
        import soundfile as sf
        audio = np.zeros(CHUNK_SAMPLES, dtype=np.float32)
        sf.write(tmp_path, audio, CLAP_SAMPLE_RATE)
        chunks = load_and_chunk(tmp_path)
        assert len(chunks) == 1
        assert len(chunks[0]) == CHUNK_SAMPLES
    finally:
        os.unlink(tmp_path)


def test_load_and_chunk_multi():
    """25-second audio → 3 chunks (10s, 10s, 5s padded to 10s)."""
    from maestro.audio.processing import load_and_chunk, CHUNK_SAMPLES, CLAP_SAMPLE_RATE

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        tmp_path = f.name

    try:
        import soundfile as sf
        duration_s = 25
        audio = np.random.randn(duration_s * CLAP_SAMPLE_RATE).astype(np.float32)
        sf.write(tmp_path, audio, CLAP_SAMPLE_RATE)
        chunks = load_and_chunk(tmp_path)
        expected = math.ceil(duration_s / 10)  # 3
        assert len(chunks) == expected
        for chunk in chunks:
            assert len(chunk) == CHUNK_SAMPLES
    finally:
        os.unlink(tmp_path)


def test_load_and_chunk_short():
    """Audio shorter than one chunk → one zero-padded chunk."""
    from maestro.audio.processing import load_and_chunk, CHUNK_SAMPLES, CLAP_SAMPLE_RATE

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        tmp_path = f.name

    try:
        import soundfile as sf
        audio = np.random.randn(CLAP_SAMPLE_RATE * 3).astype(np.float32)  # 3 seconds
        sf.write(tmp_path, audio, CLAP_SAMPLE_RATE)
        chunks = load_and_chunk(tmp_path)
        assert len(chunks) == 1
        assert len(chunks[0]) == CHUNK_SAMPLES
    finally:
        os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def test_config_defaults():
    """ExperimentConfig loads sensible defaults."""
    from experiments.audio_caption.config import ExperimentConfig
    cfg = ExperimentConfig()
    assert cfg.data.audio_dir == "./data/audio"
    assert cfg.data.prepared_dir == "./data/prepared/musiccaps_v1"
    assert cfg.data.use_prepared is True
    assert cfg.data.require_prepared is True
    assert cfg.model.clap_dim == 512
    assert cfg.training.learning_rate == 1.5e-3


def test_config_from_yaml(tmp_path):
    """ExperimentConfig.from_yaml correctly overrides defaults."""
    import yaml
    from experiments.audio_caption.config import ExperimentConfig

    cfg_file = tmp_path / "test_cfg.yaml"
    cfg_file.write_text(yaml.dump({
        "data": {"audio_dir": "/custom/audio", "dataset_limit": 32},
        "training": {"num_train_epochs": 2, "learning_rate": 5e-4},
    }))

    cfg = ExperimentConfig.from_yaml(str(cfg_file))
    assert cfg.data.audio_dir == "/custom/audio"
    assert cfg.data.dataset_limit == 32
    assert cfg.training.num_train_epochs == 2
    assert cfg.training.learning_rate == 5e-4
    # Unspecified values stay as defaults
    assert cfg.model.clap_dim == 512


# ---------------------------------------------------------------------------
# Dataset builder (no download needed — just tests filtering logic)
# ---------------------------------------------------------------------------

def test_build_dataset_filters_missing_files(tmp_path):
    """build_dataset only includes clips whose .mp3 exists on disk."""
    from unittest.mock import patch, MagicMock
    from maestro.data.musiccaps import build_dataset

    # Create one fake mp3 file
    (tmp_path / "abc123.mp3").write_bytes(b"fake")

    fake_records = [
        {"ytid": "abc123", "caption": "A funky groove", "start_s": 0, "end_s": 10},
        {"ytid": "xyz999", "caption": "Missing clip",   "start_s": 0, "end_s": 10},
    ]

    with patch("maestro.data.musiccaps.load_dataset", return_value=fake_records):
        ds = build_dataset(audio_dir=str(tmp_path), limit=None)

    assert len(ds) == 1
    assert ds[0]["ytid"] == "abc123"


def test_build_dataset_limit(tmp_path):
    """dataset_limit caps the number of returned records."""
    from unittest.mock import patch
    from maestro.data.musiccaps import build_dataset

    for i in range(5):
        (tmp_path / f"clip{i:02d}.mp3").write_bytes(b"fake")

    fake_records = [
        {"ytid": f"clip{i:02d}", "caption": f"clip {i}", "start_s": 0, "end_s": 10}
        for i in range(5)
    ]

    with patch("maestro.data.musiccaps.load_dataset", return_value=fake_records):
        ds = build_dataset(audio_dir=str(tmp_path), limit=3)

    assert len(ds) == 3


def test_build_prepared_dataset_reads_manifest(tmp_path):
    """build_prepared_dataset loads only status=ok rows and resolves clip paths."""
    from maestro.data.musiccaps import build_prepared_dataset
    import json

    prepared_dir = tmp_path / "prepared"
    clips_dir = prepared_dir / "clips"
    clips_dir.mkdir(parents=True)

    good_clip = clips_dir / "00000_a.wav"
    bad_clip = clips_dir / "00001_b.wav"
    good_clip.write_bytes(b"ok")
    # bad_clip intentionally missing

    records = [
        {
            "row_id": 0,
            "ytid": "a",
            "caption": "good",
            "start_s": 30.0,
            "end_s": 40.0,
            "clip_path": "clips/00000_a.wav",
            "duration_s": 10.0,
            "sample_rate": 48000,
            "channels": 1,
            "status": "ok",
            "error": None,
        },
        {
            "row_id": 1,
            "ytid": "b",
            "caption": "missing",
            "start_s": 30.0,
            "end_s": 40.0,
            "clip_path": "clips/00001_b.wav",
            "duration_s": 10.0,
            "sample_rate": 48000,
            "channels": 1,
            "status": "ok",
            "error": None,
        },
        {
            "row_id": 2,
            "ytid": "c",
            "caption": "failed",
            "start_s": 30.0,
            "end_s": 40.0,
            "clip_path": "clips/00002_c.wav",
            "duration_s": 0.0,
            "sample_rate": 0,
            "channels": 0,
            "status": "failed",
            "error": "missing_source",
        },
    ]
    manifest = prepared_dir / "manifest.jsonl"
    with manifest.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    ds = build_prepared_dataset(str(prepared_dir))
    assert len(ds) == 1
    assert ds[0]["ytid"] == "a"


def test_validate_prepared_manifest_success(tmp_path):
    """validate_prepared_manifest keeps valid clips as status=ok."""
    import json
    import soundfile as sf
    from maestro.audio.processing import CLAP_SAMPLE_RATE
    from maestro.data.prepare_musiccaps import validate_prepared_manifest

    prepared_dir = tmp_path / "prepared"
    clips_dir = prepared_dir / "clips"
    clips_dir.mkdir(parents=True)

    clip = clips_dir / "00000_a.wav"
    sf.write(str(clip), np.zeros(CLAP_SAMPLE_RATE * 10, dtype=np.float32), CLAP_SAMPLE_RATE)

    manifest = prepared_dir / "manifest.jsonl"
    with manifest.open("w", encoding="utf-8") as f:
        f.write(
            json.dumps(
                {
                    "row_id": 0,
                    "ytid": "a",
                    "caption": "ok",
                    "start_s": 30.0,
                    "end_s": 40.0,
                    "clip_path": "clips/00000_a.wav",
                    "duration_s": 0.0,
                    "sample_rate": 0,
                    "channels": 0,
                    "status": "unknown",
                    "error": None,
                }
            )
            + "\n"
        )

    summary = validate_prepared_manifest(str(prepared_dir))
    assert summary.total_rows == 1
    assert summary.ok_rows == 1
    assert summary.failed_rows == 0


# ---------------------------------------------------------------------------
# Collator shape check (no real models — uses mock CLAP)
# ---------------------------------------------------------------------------

def test_collator_output_shapes(tmp_path):
    """AudioCaptionCollator returns tensors with expected shapes."""
    import torch
    from unittest.mock import MagicMock, patch
    from experiments.audio_caption.collator import AudioCaptionCollator
    from maestro.audio.processing import CHUNK_SAMPLES, CLAP_SAMPLE_RATE

    # Create a fake 10s WAV file
    import soundfile as sf
    audio_path = str(tmp_path / "test.wav")
    sf.write(audio_path, np.zeros(CHUNK_SAMPLES, dtype=np.float32), CLAP_SAMPLE_RATE)

    # Mock tokenizer
    mock_tok = MagicMock()
    mock_tok.encode.side_effect = lambda text, **kw: [1, 2, 3]
    mock_tok.pad_token_id = 0

    # Mock CLAP processor and model
    mock_proc = MagicMock()
    mock_proc.return_value = {"input_features": torch.zeros(1, 64, 1001)}

    mock_clap = MagicMock()
    mock_clap.get_audio_features.return_value = torch.zeros(1, 512)  # 1 chunk, 512 dim

    audio_token_id = 99
    prefix_ids = [10, 11]
    suffix_ids = [12]
    mock_tok.encode.side_effect = lambda text, **kw: (
        prefix_ids if "im_start" in text else (suffix_ids if "im_end" in text else [1, 2, 3])
    )

    collator = AudioCaptionCollator(
        tokenizer=mock_tok,
        clap_processor=mock_proc,
        clap_model=mock_clap,
        audio_token_id=audio_token_id,
        device="cpu",
        max_length=128,
    )

    batch = collator([{"audio_path": audio_path, "caption": "test caption", "ytid": "x"}])

    assert "input_ids" in batch
    assert "attention_mask" in batch
    assert "audio_features" in batch
    assert "audio_token_counts" in batch
    assert "labels" in batch
    assert batch["audio_features"].shape == (1, 512)
    assert batch["audio_token_counts"].tolist() == [1]
    assert batch["input_ids"].shape[0] == 1  # batch size 1


# ---------------------------------------------------------------------------
def test_collator_truncation_keeps_audio_alignment():
    """Truncation must keep audio feature rows aligned with retained <audio> tokens."""
    import torch
    from unittest.mock import MagicMock, patch
    from experiments.audio_caption.collator import AudioCaptionCollator

    audio_token_id = 99
    mock_tok = MagicMock()
    prefix_ids = [10, 11]
    suffix_ids = [12]
    mock_tok.encode.side_effect = lambda text, **kw: (
        prefix_ids if "im_start" in text else (suffix_ids if "im_end" in text else [1, 2, 3, 4])
    )
    mock_tok.pad_token_id = 0

    mock_proc = MagicMock()
    mock_proc.return_value = {"input_features": torch.zeros(3, 64, 1001)}

    # 3 chunks x 4 dims
    mock_clap = MagicMock()
    mock_clap.get_audio_features.return_value = torch.arange(12, dtype=torch.float32).view(3, 4)

    collator = AudioCaptionCollator(
        tokenizer=mock_tok,
        clap_processor=mock_proc,
        clap_model=mock_clap,
        audio_token_id=audio_token_id,
        device="cpu",
        max_length=2,  # force truncation inside the leading audio token span
        use_dynamic_chunking=True,
    )

    with patch("experiments.audio_caption.collator.load_and_chunk", return_value=[np.zeros(10)] * 3):
        batch = collator([{"audio_path": "fake.mp3", "caption": "cap", "ytid": "x"}])

    assert batch["audio_token_counts"].tolist() == [2]
    assert batch["audio_features"].shape == (2, 4)
    assert int(batch["audio_token_counts"].sum().item()) == batch["audio_features"].shape[0]


def _build_dummy_audio_language_model(audio_token_id: int, clap_dim: int, llm_dim: int):
    import torch
    import torch.nn as nn
    from unittest.mock import patch
    from transformers.modeling_outputs import CausalLMOutputWithPast
    from experiments.audio_caption.model import AudioLanguageModel

    class _DummyInnerModel(nn.Module):
        def __init__(self, vocab_size: int, hidden_size: int):
            super().__init__()
            self.embed_tokens = nn.Embedding(vocab_size, hidden_size)

    class _DummyLLM(nn.Module):
        def __init__(self, vocab_size: int, hidden_size: int):
            super().__init__()
            self.model = _DummyInnerModel(vocab_size, hidden_size)
            self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
            self.config = SimpleNamespace()
            self.generation_config = SimpleNamespace()
            self.last_inputs_embeds = None

        def get_input_embeddings(self):
            return self.model.embed_tokens

        def set_input_embeddings(self, value):
            self.model.embed_tokens = value

        def resize_token_embeddings(self, new_num_tokens: int, pad_to_multiple_of: int = None):
            hidden = self.model.embed_tokens.embedding_dim
            self.model.embed_tokens = nn.Embedding(new_num_tokens, hidden)
            self.lm_head = nn.Linear(hidden, new_num_tokens, bias=False)
            return self.model.embed_tokens

        def add_model_tags(self, tags):
            return None

        def forward(self, input_ids=None, inputs_embeds=None, attention_mask=None, labels=None, return_dict=True, **kwargs):
            if inputs_embeds is None:
                inputs_embeds = self.model.embed_tokens(input_ids)
            self.last_inputs_embeds = inputs_embeds.detach().clone()
            logits = self.lm_head(inputs_embeds)
            loss = logits.float().mean()
            if return_dict:
                return CausalLMOutputWithPast(loss=loss, logits=logits)
            return (loss, logits)

    dummy_llm = _DummyLLM(vocab_size=300, hidden_size=llm_dim)
    with patch(
        "experiments.audio_caption.model.AutoModelForCausalLM.from_pretrained",
        return_value=dummy_llm,
    ):
        model = AudioLanguageModel(
            audio_token_id=audio_token_id,
            qwen_model_id="dummy",
            clap_dim=clap_dim,
            llm_dim=llm_dim,
            torch_dtype=torch.float32,
        )
    return model, dummy_llm


def test_model_forward_raises_on_alignment_mismatch():
    """Model forward should fail fast if audio feature rows != number of <audio> tokens."""
    import torch

    model, _ = _build_dummy_audio_language_model(audio_token_id=99, clap_dim=4, llm_dim=8)
    input_ids = torch.tensor([[99, 5, 99, 7]], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    audio_features = torch.randn(1, 4)  # should be 2 rows
    labels = torch.full_like(input_ids, -100)

    with pytest.raises(ValueError, match="Audio feature count mismatch"):
        model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            audio_features=audio_features,
            labels=labels,
        )


def test_model_forward_inputs_embeds_replaces_exact_positions():
    """Projected rows should replace exactly the flattened <audio> positions."""
    import torch

    audio_token_id = 99
    model, dummy_llm = _build_dummy_audio_language_model(audio_token_id=audio_token_id, clap_dim=4, llm_dim=8)

    with torch.no_grad():
        model.projection.weight.zero_()
        model.projection.bias.zero_()
        model.projection.weight[:4, :4] = torch.eye(4)

    input_ids = torch.tensor(
        [
            [audio_token_id, 10, 11, audio_token_id],
            [20, audio_token_id, 21, 22],
        ],
        dtype=torch.long,
    )
    attention_mask = torch.ones_like(input_ids)
    labels = torch.full_like(input_ids, -100)
    audio_features = torch.tensor(
        [
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0, 12.0],
        ],
        dtype=torch.float32,
    )

    projected = model.projection(audio_features).to(model.projection.weight.dtype)
    model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        audio_features=audio_features,
        audio_token_counts=torch.tensor([2, 1], dtype=torch.long),
        labels=labels,
    )

    replaced = dummy_llm.last_inputs_embeds[input_ids == audio_token_id]
    torch.testing.assert_close(replaced, projected.to(replaced.dtype))


def test_resolve_qwen_hidden_size_prefers_text_config():
    """Hidden size resolver should read text_config.hidden_size when top-level is missing."""
    from unittest.mock import patch
    from experiments.audio_caption.train import resolve_qwen_hidden_size

    class _Cfg:
        hidden_size = None
        text_config = SimpleNamespace(hidden_size=2560)

        def to_dict(self):
            return {"text_config": {"hidden_size": 2560}}

    with patch("experiments.audio_caption.train.AutoConfig.from_pretrained", return_value=_Cfg()):
        assert resolve_qwen_hidden_size("Qwen/Qwen3.5-4B") == 2560


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
