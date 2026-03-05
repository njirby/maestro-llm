#!/usr/bin/env python3
"""
Unified experiment launcher.

Usage:
    python scripts/prepare_musiccaps.py --limit 16
    python scripts/run_experiment.py audio_caption
    python scripts/run_experiment.py audio_caption --dataset_limit 16 --num_train_epochs 1
    python scripts/run_experiment.py audio_caption --config configs/audio_caption.yaml

Multi-GPU (via accelerate):
    accelerate launch --num_processes 2 --mixed_precision bf16 \\
        scripts/run_experiment.py audio_caption

Adding a new experiment:
    1. Create experiments/<name>/ with train.py exporting train(cfg)
    2. Create configs/<name>.yaml
    3. Register it in EXPERIMENTS below
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

EXPERIMENTS = {
    "audio_caption": "experiments.audio_caption.train",
}


def main():
    parser = argparse.ArgumentParser(description="Run a maestro experiment")
    parser.add_argument("experiment", choices=list(EXPERIMENTS), help="Experiment to run")
    parser.add_argument("--config", type=str, help="Path to YAML config (defaults to configs/<experiment>.yaml)")

    # Common overrides (forwarded to the experiment)
    parser.add_argument("--audio_dir", type=str)
    parser.add_argument("--prepared_dir", type=str)
    parser.add_argument("--use_prepared", dest="use_prepared", action="store_true")
    parser.add_argument("--no_use_prepared", dest="use_prepared", action="store_false")
    parser.set_defaults(use_prepared=None)
    parser.add_argument("--require_prepared", dest="require_prepared", action="store_true")
    parser.add_argument("--no_require_prepared", dest="require_prepared", action="store_false")
    parser.set_defaults(require_prepared=None)
    parser.add_argument("--dataset_limit", type=int)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--num_train_epochs", type=int)
    parser.add_argument("--per_device_train_batch_size", type=int)
    parser.add_argument("--gradient_checkpointing", dest="gradient_checkpointing", action="store_true")
    parser.add_argument("--no_gradient_checkpointing", dest="gradient_checkpointing", action="store_false")
    parser.set_defaults(gradient_checkpointing=None)
    parser.add_argument("--learning_rate", type=float)
    parser.add_argument("--max_length", type=int)

    args = parser.parse_args()

    # Default config path
    if args.config is None:
        args.config = f"configs/{args.experiment}.yaml"

    # Dynamically import and run the experiment's train() function
    import importlib
    module = importlib.import_module(EXPERIMENTS[args.experiment])

    # Build config from YAML then apply CLI overrides
    from experiments.audio_caption.config import ExperimentConfig
    cfg = ExperimentConfig.from_yaml(args.config)

    if args.audio_dir:
        cfg.data.audio_dir = args.audio_dir
    if args.prepared_dir:
        cfg.data.prepared_dir = args.prepared_dir
    if args.use_prepared is not None:
        cfg.data.use_prepared = args.use_prepared
    if args.require_prepared is not None:
        cfg.data.require_prepared = args.require_prepared
    if args.dataset_limit is not None:
        cfg.data.dataset_limit = args.dataset_limit
    if args.output_dir:
        cfg.training.output_dir = args.output_dir
    if args.num_train_epochs is not None:
        cfg.training.num_train_epochs = args.num_train_epochs
    if args.per_device_train_batch_size is not None:
        cfg.training.per_device_train_batch_size = args.per_device_train_batch_size
    if args.gradient_checkpointing is not None:
        cfg.training.gradient_checkpointing = args.gradient_checkpointing
    if args.learning_rate is not None:
        cfg.training.learning_rate = args.learning_rate
    if args.max_length is not None:
        cfg.training.max_length = args.max_length

    print(f"\n=== Running experiment: {args.experiment} ===")
    print(f"Config: {args.config}")
    print(f"Data dir: {cfg.data.audio_dir}")
    print(f"Prepared dir: {cfg.data.prepared_dir}")
    print(f"Use prepared: {cfg.data.use_prepared}")
    print(f"Dataset limit: {cfg.data.dataset_limit}")
    print(f"Output dir: {cfg.training.output_dir}")
    print(f"Epochs: {cfg.training.num_train_epochs}")
    print()

    module.train(cfg)


if __name__ == "__main__":
    main()
