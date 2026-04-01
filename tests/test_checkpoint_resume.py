import importlib.util
import os
import sys
import types

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture(autouse=True)
def _force_cpu_training_args(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")


def _load_run_experiment_module():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(repo_root, "scripts", "run_experiment.py")
    spec = importlib.util.spec_from_file_location("run_experiment_for_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_build_training_args_uses_checkpoint_fields():
    from experiments.audio_caption.config import TrainingConfig
    from experiments.audio_caption.train import build_training_args

    cfg = TrainingConfig(
        output_dir="./outputs/test",
        save_strategy="epoch",
        save_steps=None,
        save_total_limit=7,
        save_safetensors=True,
        save_only_model=False,
        overwrite_output_dir=True,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        weight_decay=0.01,
    )
    args = build_training_args(cfg, has_eval_dataset=True)

    assert args.save_strategy.value == "epoch"
    assert args.eval_strategy.value == "epoch"
    assert args.save_total_limit == 7
    assert args.save_only_model is False
    assert args.lr_scheduler_type.value == "cosine"
    assert args.warmup_ratio == 0.1
    assert args.weight_decay == 0.01


def test_build_training_args_requires_steps_value():
    from experiments.audio_caption.config import TrainingConfig
    from experiments.audio_caption.train import build_training_args

    cfg = TrainingConfig(save_strategy="steps", save_steps=None)
    with pytest.raises(ValueError, match="save_steps"):
        build_training_args(cfg, has_eval_dataset=False)


def test_build_training_args_rejects_save_safetensors_false_if_not_supported():
    from experiments.audio_caption.config import TrainingConfig
    from experiments.audio_caption.train import build_training_args

    cfg = TrainingConfig(save_safetensors=False)
    with pytest.raises(ValueError, match="save_safetensors"):
        build_training_args(cfg, has_eval_dataset=False)


def test_build_training_args_requires_eval_steps_when_eval_is_steps():
    from experiments.audio_caption.config import TrainingConfig
    from experiments.audio_caption.train import build_training_args

    cfg = TrainingConfig(eval_strategy="steps", eval_steps=None)
    with pytest.raises(ValueError, match="eval_steps"):
        build_training_args(cfg, has_eval_dataset=True)


def test_build_training_args_disables_eval_without_eval_dataset():
    from experiments.audio_caption.config import TrainingConfig
    from experiments.audio_caption.train import build_training_args

    cfg = TrainingConfig(eval_strategy="epoch")
    args = build_training_args(cfg, has_eval_dataset=False)
    assert args.eval_strategy.value == "no"


def test_split_train_eval_dataset(tmp_path):
    from datasets import Dataset
    from experiments.audio_caption.train import split_train_eval_dataset

    ds = Dataset.from_list([{"x": i} for i in range(10)])
    train_ds, eval_ds = split_train_eval_dataset(ds, validation_split_ratio=0.2, seed=123)

    assert len(train_ds) == 8
    assert len(eval_ds) == 2


def test_split_train_eval_dataset_zero_ratio_keeps_all_train():
    from datasets import Dataset
    from experiments.audio_caption.train import split_train_eval_dataset

    ds = Dataset.from_list([{"x": i} for i in range(4)])
    train_ds, eval_ds = split_train_eval_dataset(ds, validation_split_ratio=0.0, seed=42)

    assert len(train_ds) == 4
    assert eval_ds is None


def test_split_train_eval_dataset_rejects_invalid_ratio():
    from datasets import Dataset
    from experiments.audio_caption.train import split_train_eval_dataset

    ds = Dataset.from_list([{"x": i} for i in range(4)])
    with pytest.raises(ValueError, match="validation_split_ratio"):
        split_train_eval_dataset(ds, validation_split_ratio=1.0, seed=42)


def test_resolve_resume_checkpoint_latest(tmp_path):
    from experiments.audio_caption.train import resolve_resume_checkpoint

    out = tmp_path / "outputs"
    (out / "checkpoint-10").mkdir(parents=True)
    (out / "checkpoint-25").mkdir(parents=True)

    resolved = resolve_resume_checkpoint(
        output_dir=str(out),
        resume_from_checkpoint="latest",
        overwrite_output_dir=False,
    )
    assert resolved.endswith("checkpoint-25")


def test_resolve_resume_checkpoint_requires_explicit_when_checkpoint_exists(tmp_path):
    from experiments.audio_caption.train import resolve_resume_checkpoint

    out = tmp_path / "outputs"
    (out / "checkpoint-3").mkdir(parents=True)

    with pytest.raises(ValueError, match="Found existing checkpoint"):
        resolve_resume_checkpoint(
            output_dir=str(out),
            resume_from_checkpoint=None,
            overwrite_output_dir=False,
        )


def test_resolve_resume_checkpoint_overwrite_bypasses_guard(tmp_path):
    from experiments.audio_caption.train import resolve_resume_checkpoint

    out = tmp_path / "outputs"
    (out / "checkpoint-4").mkdir(parents=True)

    resolved = resolve_resume_checkpoint(
        output_dir=str(out),
        resume_from_checkpoint=None,
        overwrite_output_dir=True,
    )
    assert resolved is None


def test_resolve_resume_checkpoint_explicit_path_must_exist(tmp_path):
    from experiments.audio_caption.train import resolve_resume_checkpoint

    with pytest.raises(ValueError, match="does not exist"):
        resolve_resume_checkpoint(
            output_dir=str(tmp_path / "outputs"),
            resume_from_checkpoint=str(tmp_path / "missing-checkpoint"),
            overwrite_output_dir=False,
        )


def test_resolve_resume_checkpoint_rejects_overwrite_plus_resume(tmp_path):
    from experiments.audio_caption.train import resolve_resume_checkpoint

    with pytest.raises(ValueError, match="cannot be combined"):
        resolve_resume_checkpoint(
            output_dir=str(tmp_path / "outputs"),
            resume_from_checkpoint="latest",
            overwrite_output_dir=True,
        )


def test_purge_checkpoint_dirs_removes_only_checkpoint_dirs(tmp_path):
    from experiments.audio_caption.train import purge_checkpoint_dirs

    out = tmp_path / "outputs"
    (out / "checkpoint-1").mkdir(parents=True)
    (out / "checkpoint-2").mkdir()
    keep_file = out / "projection_epoch1.pt"
    keep_file.write_bytes(b"ok")

    removed = purge_checkpoint_dirs(str(out))

    assert removed == 2
    assert not (out / "checkpoint-1").exists()
    assert not (out / "checkpoint-2").exists()
    assert keep_file.exists()


def test_run_experiment_resume_flag_sets_latest(tmp_path, monkeypatch):
    module = _load_run_experiment_module()

    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(
        """
data: {}
model: {}
training:
  output_dir: ./outputs/test
"""
    )

    captured = {}
    fake_module = types.ModuleType("fake_experiment")

    def fake_train(cfg):
        captured["cfg"] = cfg

    fake_module.train = fake_train
    sys.modules["fake_experiment"] = fake_module
    module.EXPERIMENTS["audio_caption"] = "fake_experiment"

    monkeypatch.setattr(
        sys,
        "argv",
        ["run_experiment.py", "audio_caption", "--config", str(cfg_path), "--resume"],
    )

    try:
        module.main()
    finally:
        sys.modules.pop("fake_experiment", None)

    assert captured["cfg"].training.resume_from_checkpoint == "latest"


def test_run_experiment_validation_split_override(tmp_path, monkeypatch):
    module = _load_run_experiment_module()

    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(
        """
data: {}
model: {}
training:
  output_dir: ./outputs/test
"""
    )

    captured = {}
    fake_module = types.ModuleType("fake_experiment")

    def fake_train(cfg):
        captured["cfg"] = cfg

    fake_module.train = fake_train
    sys.modules["fake_experiment"] = fake_module
    module.EXPERIMENTS["audio_caption"] = "fake_experiment"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_experiment.py",
            "audio_caption",
            "--config",
            str(cfg_path),
            "--validation_split_ratio",
            "0.25",
            "--eval_strategy",
            "steps",
            "--eval_steps",
            "100",
        ],
    )

    try:
        module.main()
    finally:
        sys.modules.pop("fake_experiment", None)

    assert captured["cfg"].training.validation_split_ratio == 0.25
    assert captured["cfg"].training.eval_strategy == "steps"
    assert captured["cfg"].training.eval_steps == 100
