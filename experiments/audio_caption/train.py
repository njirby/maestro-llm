"""
Training entry point for the audio_caption experiment.

Usage:
    # 1) Download source audio:
    python scripts/download_data.py --limit 16

    # 2) Prepare canonical 10s clips:
    python scripts/prepare_musiccaps.py --limit 16

    # 3) Train:
    # From repo root (after pip install -e .):
    python scripts/run_experiment.py audio_caption

    # Or directly with a config override:
    python -m experiments.audio_caption.train --config configs/audio_caption.yaml

    # Multi-GPU ZeRO-3:
    bash scripts/train_2gpu_zero3.sh --config configs/audio_caption_qwen4b_stable.yaml
"""

import argparse
import os

import torch
from transformers import AutoConfig, AutoTokenizer, ClapModel, ClapProcessor, TrainerCallback
from transformers.integrations import is_deepspeed_zero3_enabled
from trl import SFTConfig, SFTTrainer

from experiments.audio_caption.collator import AudioCaptionCollator
from experiments.audio_caption.config import ExperimentConfig
from experiments.audio_caption.model import AUDIO_TOKEN, AudioLanguageModel
from maestro.data.musiccaps import build_dataset, build_prepared_dataset


class ProjectionCheckpointCallback(TrainerCallback):
    """Save only the projection layer after each epoch (avoids OOM from saving the full 9B model)."""

    def __init__(self, output_dir: str):
        self.output_dir = output_dir

    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        epoch = int(state.epoch)
        os.makedirs(self.output_dir, exist_ok=True)
        save_path = os.path.join(self.output_dir, f"projection_epoch{epoch}.pt")

        actual = model.module if hasattr(model, "module") else model
        proj = actual.projection

        if is_deepspeed_zero3_enabled():
            import deepspeed
            with deepspeed.zero.GatheredParameters(list(proj.parameters()), modifier_rank=0):
                if args.process_index == 0:
                    torch.save(
                        {"projection.weight": proj.weight.data.cpu(),
                         "projection.bias":   proj.bias.data.cpu()},
                        save_path,
                    )
                    print(f"  [checkpoint] projection saved → {save_path}")
        else:
            if args.process_index == 0:
                torch.save(
                    {k: v for k, v in actual.state_dict().items() if "projection" in k},
                    save_path,
                )
                print(f"  [checkpoint] projection saved → {save_path}")


def setup_tokenizer(qwen_model_id: str):
    tokenizer = AutoTokenizer.from_pretrained(qwen_model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_special_tokens({"additional_special_tokens": [AUDIO_TOKEN]})
    audio_token_id = tokenizer.convert_tokens_to_ids(AUDIO_TOKEN)
    print(f"<audio> token ID: {audio_token_id}, vocab size: {len(tokenizer)}")
    return tokenizer, audio_token_id


def resolve_qwen_hidden_size(qwen_model_id: str) -> int:
    config = AutoConfig.from_pretrained(qwen_model_id, trust_remote_code=True)

    hidden_size = getattr(config, "hidden_size", None)
    if hidden_size is not None:
        return int(hidden_size)

    text_config = getattr(config, "text_config", None)
    hidden_size = getattr(text_config, "hidden_size", None)
    if hidden_size is not None:
        return int(hidden_size)

    config_dict = config.to_dict()
    text_config_dict = config_dict.get("text_config", {})
    hidden_size = text_config_dict.get("hidden_size")
    if hidden_size is not None:
        return int(hidden_size)

    raise ValueError(
        "Could not resolve hidden size from model config. "
        f"qwen_model_id={qwen_model_id}, config_class={type(config).__name__}."
    )


def train(cfg: ExperimentConfig):
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"

    tokenizer, audio_token_id = setup_tokenizer(cfg.model.qwen_model_id)

    clap_processor = ClapProcessor.from_pretrained(cfg.model.clap_model_id)
    # CLAP runs on CPU: ZeRO-3 communication buffers consume the freed GPU memory,
    # so CLAP's forward-pass activations (~3 GB peak) would still OOM on GPU.
    # CPU inference is fast enough for 10s clips.
    clap_model = ClapModel.from_pretrained(cfg.model.clap_model_id).eval()
    for p in clap_model.parameters():
        p.requires_grad = False

    resolved_llm_dim = resolve_qwen_hidden_size(cfg.model.qwen_model_id)
    if cfg.model.qwen_hidden_dim is not None and cfg.model.qwen_hidden_dim != resolved_llm_dim:
        raise ValueError(
            "Configured qwen_hidden_dim does not match model config hidden size. "
            f"configured={cfg.model.qwen_hidden_dim}, resolved={resolved_llm_dim}, "
            f"qwen_model_id={cfg.model.qwen_model_id}."
        )
    llm_dim = cfg.model.qwen_hidden_dim or resolved_llm_dim
    print(f"Resolved LLM hidden size: {llm_dim}")

    model = AudioLanguageModel(
        audio_token_id=audio_token_id,
        qwen_model_id=cfg.model.qwen_model_id,
        clap_dim=cfg.model.clap_dim,
        llm_dim=llm_dim,
    )
    model.resize_token_embeddings(len(tokenizer))
    # Disable KV cache during training to reduce memory pressure.
    model.config.use_cache = False
    model.llm.config.use_cache = False
    if hasattr(model.llm, "generation_config") and model.llm.generation_config is not None:
        model.llm.generation_config.use_cache = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable:,} / {total:,} params")

    if cfg.data.use_prepared:
        try:
            dataset = build_prepared_dataset(
                prepared_dir=cfg.data.prepared_dir,
                limit=cfg.data.dataset_limit,
            )
        except Exception:
            if cfg.data.require_prepared:
                raise
            print(
                "WARNING: failed to load prepared dataset; "
                "falling back to legacy dynamic chunking path."
            )
            dataset = build_dataset(
                audio_dir=cfg.data.audio_dir,
                limit=cfg.data.dataset_limit,
            )
    else:
        print(
            "WARNING: using legacy dynamic chunking path. "
            "For stable multi-GPU training, prefer prepared clips."
        )
        dataset = build_dataset(
            audio_dir=cfg.data.audio_dir,
            limit=cfg.data.dataset_limit,
        )

    collator = AudioCaptionCollator(
        tokenizer=tokenizer,
        clap_processor=clap_processor,
        clap_model=clap_model,
        audio_token_id=audio_token_id,
        device=device,
        max_length=cfg.training.max_length,
        use_dynamic_chunking=not cfg.data.use_prepared,
    )

    t = cfg.training
    if t.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    training_args = SFTConfig(
        output_dir=t.output_dir,
        num_train_epochs=t.num_train_epochs,
        per_device_train_batch_size=t.per_device_train_batch_size,
        gradient_accumulation_steps=t.gradient_accumulation_steps,
        learning_rate=t.learning_rate,
        lr_scheduler_type="constant",
        warmup_steps=100,
        bf16=True,
        fp16=False,
        logging_steps=t.logging_steps,
        report_to="tensorboard",
        save_strategy="no",  # projection saved per-epoch via ProjectionCheckpointCallback
        dataloader_num_workers=t.dataloader_num_workers,
        remove_unused_columns=False,
        dataset_kwargs={"skip_prepare_dataset": True},
        max_length=t.max_length,
        deepspeed="configs/deepspeed_zero3.json",
        gradient_checkpointing=t.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False} if t.gradient_checkpointing else None,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
        processing_class=tokenizer,
        callbacks=[ProjectionCheckpointCallback(t.output_dir)],
    )

    # Set TENSORBOARD_LOGGING_DIR so tensorboard logs land in a predictable place
    os.environ["TENSORBOARD_LOGGING_DIR"] = os.path.join(t.output_dir, "tensorboard")

    trainer.train()

    # Save only the trainable projection layer (~8MB)
    save_path = os.path.join(t.output_dir, "projection_final.pt")
    actual = trainer.model.module if hasattr(trainer.model, "module") else trainer.model
    proj = actual.projection
    if is_deepspeed_zero3_enabled():
        import deepspeed
        with deepspeed.zero.GatheredParameters(list(proj.parameters()), modifier_rank=0):
            if training_args.process_index == 0:
                torch.save(
                    {"projection.weight": proj.weight.data.cpu(),
                     "projection.bias":   proj.bias.data.cpu()},
                    save_path,
                )
    else:
        if training_args.process_index == 0:
            torch.save(
                {k: v for k, v in actual.state_dict().items() if "projection" in k},
                save_path,
            )
    if training_args.process_index == 0:
        print(f"Projection layer saved to {save_path}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/audio_caption.yaml")
    # Per-field overrides (flat, for quick CLI tweaks)
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
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = ExperimentConfig.from_yaml(args.config)

    # Apply any CLI overrides
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

    train(cfg)
