"""
Configuration for the audio_caption experiment.

Load from YAML:
    cfg = ExperimentConfig.from_yaml("configs/audio_caption.yaml")

Override fields at runtime via CLI:
    python scripts/run_experiment.py audio_caption --learning_rate 1e-3
"""

from __future__ import annotations

import yaml
from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class DataConfig:
    audio_dir: str = "./data/audio"
    prepared_dir: str = "./data/prepared/musiccaps_v1"
    use_prepared: bool = True
    require_prepared: bool = True
    dataset_limit: Optional[int] = None


@dataclass
class ModelConfig:
    clap_model_id: str = "laion/larger_clap_general"
    qwen_model_id: str = "Qwen/Qwen3.5-9B"
    clap_dim: int = 512
    qwen_hidden_dim: Optional[int] = None


@dataclass
class TrainingConfig:
    output_dir: str = "./outputs/audio_caption"
    num_train_epochs: int = 5
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 1.5e-3
    max_length: int = 512
    logging_steps: int = 10
    dataloader_num_workers: int = 0  # collator does heavy CPU work; keep simple and deterministic
    gradient_checkpointing: bool = False


@dataclass
class ExperimentConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    @classmethod
    def from_yaml(cls, path: str) -> ExperimentConfig:
        with open(path) as f:
            d = yaml.safe_load(f) or {}
        return cls(
            data=DataConfig(**d.get("data", {})),
            model=ModelConfig(**d.get("model", {})),
            training=TrainingConfig(**d.get("training", {})),
        )

    def to_yaml(self, path: str):
        with open(path, "w") as f:
            yaml.dump(asdict(self), f, default_flow_style=False)
