"""Qwen3-Omni audio tower grafted onto Qwen3.5-4B. No copied model code: both HF classes are used unmodified
(transformers 5.16.1, pinned by the ear-graft image — 5.8's audio encoder ignores its window mask under sdpa).

  wav -> mel (128, T) -> frozen Omni audio tower -> (n, 2048) -> projector -> (n, 2560)
      -> scattered over <|audio_pad|> positions in the 4B's input embeddings -> frozen 4B (via inputs_embeds)

Trainable: the projector (linear, or a 2-layer MLP), plus input-side embeddings for <|audio_start|>/<|audio_end|> (never
trained in the 4B checkpoint: their rows equal every unused vocab row). They live outside the tied embedding table so
lm_head is untouched. Optionally LoRA (lora_r > 0) on every linear layer of the 4B's decoder.
"""
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from safetensors import safe_open
from torch import nn
from torch.utils.checkpoint import checkpoint
from transformers import Qwen3_5ForConditionalGeneration
from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import Qwen3OmniMoeAudioEncoderConfig
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import Qwen3OmniMoeAudioEncoder

OMNI_PATH = "/home/nate/.cache/huggingface/hub/models--Qwen--Qwen3-Omni-30B-A3B-Instruct/snapshots/26291f793822fb6be9555850f06dfe95f2d7e695"
QWEN4B_PATH = "/home/nate/.cache/huggingface/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"

AUDIO_START_ID, AUDIO_END_ID, AUDIO_PAD_ID = 248070, 248071, 248076
VISION_START_ID, VISION_END_ID = 248053, 248054  # trained rows; used to initialise the audio markers
LORA_TARGETS = r".*language_model\.layers\.\d+\.(self_attn\.(q|k|v|o)_proj|linear_attn\.(in_proj_qkv|in_proj_z|out_proj)|mlp\.(gate|up|down)_proj)"
CE_CHUNK = 1024  # labelled positions per lm_head chunk (1024 x 248k fp32 logits = 1 GB)


def _chunk_ce(w, h, t):
    return nn.functional.cross_entropy((h @ w.T).float(), t, reduction="none")


def load_audio_tower(omni_path=OMNI_PATH, attn_implementation="sdpa"):
    cfg = json.loads((Path(omni_path) / "config.json").read_text())["thinker_config"]["audio_config"]
    cfg = Qwen3OmniMoeAudioEncoderConfig(**cfg)
    cfg._attn_implementation = attn_implementation
    tower = Qwen3OmniMoeAudioEncoder(cfg)
    prefix = "thinker.audio_tower."
    index = json.loads((Path(omni_path) / "model.safetensors.index.json").read_text())["weight_map"]
    state = {}
    for shard in sorted({v for k, v in index.items() if k.startswith(prefix)}):
        with safe_open(str(Path(omni_path) / shard), "pt") as f:
            state.update({k[len(prefix):]: f.get_tensor(k) for k in f.keys() if k.startswith(prefix)})
    missing, unexpected = tower.load_state_dict(state, strict=False)
    # positional_embedding is a non-persistent sinusoid buffer, rebuilt in __init__
    assert not unexpected and all("positional_embedding" in k for k in missing), (missing, unexpected)
    assert len(state) == 525, len(state)
    return tower


class EarGraft(nn.Module):
    def __init__(self, qwen_path=QWEN4B_PATH, omni_path=OMNI_PATH, dtype=torch.bfloat16, attn_implementation="sdpa",
                 device=None, projector="linear", lora_r=0, lora_alpha=None):
        super().__init__()
        # device: load the 4B straight onto that GPU (DDP ranks would otherwise each stage ~9 GB in the box's 30 GB RAM)
        self.lm = Qwen3_5ForConditionalGeneration.from_pretrained(
            qwen_path, dtype=dtype, attn_implementation=attn_implementation,
            **({"device_map": {"": str(device)}} if device is not None else {}))
        self.tower = load_audio_tower(omni_path, attn_implementation).to(dtype)
        self.lm.requires_grad_(False)
        self.tower.requires_grad_(False)

        hidden = self.lm.config.text_config.hidden_size
        d_in = self.tower.config.output_dim
        if projector == "linear":  # fp32 master weights
            self.projector = nn.Linear(d_in, hidden)
        elif projector == "mlp":  # hidden width 2x so a trained linear projector can be lifted into it exactly
            self.projector = nn.Sequential(nn.Linear(d_in, 2 * hidden), nn.GELU(), nn.Linear(2 * hidden, hidden))
        else:
            raise ValueError(projector)
        if lora_r:
            from peft import LoraConfig, inject_adapter_in_model
            inject_adapter_in_model(LoraConfig(r=lora_r, lora_alpha=lora_alpha or 2 * lora_r, lora_dropout=0.0,
                                               target_modules=LORA_TARGETS), self.lm)
            for n, p in self.lm.named_parameters():
                p.requires_grad_("lora_" in n)
        emb = self.lm.get_input_embeddings().weight
        self.audio_start = nn.Parameter(emb[VISION_START_ID].detach().float().clone())
        self.audio_end = nn.Parameter(emb[VISION_END_ID].detach().float().clone())

    def trainable_state_dict(self):
        return {k: v for k, v in self.state_dict().items() if not k.startswith(("lm.", "tower.")) or "lora_" in k}

    @torch.no_grad()
    def lift_linear_projector(self, weight, bias):
        """Set the MLP to compute exactly x @ W.T + b: GELU(z) - GELU(-z) = z, so hidden = [Wx+b, -(Wx+b)], out = h1 - h2."""
        l1, l2 = self.projector[0], self.projector[2]
        n = weight.shape[0]
        l1.weight.copy_(torch.cat([weight, -weight]))
        l1.bias.copy_(torch.cat([bias, -bias]))
        eye = torch.eye(n, dtype=l2.weight.dtype, device=l2.weight.device)
        l2.weight.copy_(torch.cat([eye, -eye], dim=1))
        l2.bias.zero_()

    def load_trainable(self, path, only=None):
        """Load a trainable checkpoint, possibly from an earlier setup: a linear projector loaded into an MLP model is
        lifted exactly, and a checkpoint without LoRA leaves the LoRA at its zero init. only: key prefixes to take."""
        state = torch.load(path, map_location="cpu")
        lifted = isinstance(self.projector, nn.Sequential) and "projector.weight" in state
        if lifted:
            self.lift_linear_projector(state.pop("projector.weight"), state.pop("projector.bias"))
        if only:
            state = {k: v for k, v in state.items() if k.startswith(tuple(only))}
        missing, unexpected = self.load_state_dict(state, strict=False)
        assert not unexpected, unexpected
        if not only:
            file_has_lora = any("lora_" in k for k in state)
            bad = [k for k in missing if not (k.startswith(("lm.", "tower.")) and ("lora_" not in k or not file_has_lora))
                   and not (lifted and k.startswith("projector."))]
            assert not bad, bad

    def audio_embeds(self, input_features, feature_attention_mask):
        """(B, 128, T) padded mel + (B, T) mask -> (sum of per-clip token counts, hidden), clips in batch order."""
        lens = feature_attention_mask.sum(-1)
        packed = input_features.permute(0, 2, 1)[feature_attention_mask.bool()].permute(1, 0)
        with torch.no_grad():  # nothing upstream of the tower trains
            out = self.tower(packed.to(self.tower.dtype), feature_lens=lens).last_hidden_state
        return self.projector(out.float())

    def build_inputs_embeds(self, input_ids, input_features=None, feature_attention_mask=None):
        embeds = self.lm.get_input_embeddings()(input_ids)
        dt = embeds.dtype
        embeds = torch.where((input_ids == AUDIO_START_ID)[..., None], self.audio_start.to(dt), embeds)
        embeds = torch.where((input_ids == AUDIO_END_ID)[..., None], self.audio_end.to(dt), embeds)
        pad = input_ids == AUDIO_PAD_ID
        if input_features is not None:
            audio = self.audio_embeds(input_features, feature_attention_mask).to(dt)
            if pad.sum() != audio.shape[0]:
                raise ValueError(f"{int(pad.sum())} <|audio_pad|> tokens but {audio.shape[0]} audio embeddings")
            embeds = embeds.masked_scatter(pad[..., None].expand_as(embeds), audio)
        elif pad.any():
            raise ValueError("<|audio_pad|> tokens present but no input_features")
        return embeds

    def forward(self, input_ids, attention_mask=None, labels=None, input_features=None, feature_attention_mask=None,
                audio_scale=1.0, **kwargs):
        embeds = self.build_inputs_embeds(input_ids, input_features, feature_attention_mask)
        if audio_scale != 1.0:  # negative control: audio_scale=0 zeroes the audio embeddings, keeps the tokens
            pad = (input_ids == AUDIO_PAD_ID)[..., None]
            embeds = torch.where(pad, embeds * audio_scale, embeds)
        # Audio counts as text for M-RoPE (same as Omni for audio-only input): with inputs_embeds and no image grids the
        # 4B falls back to sequential positions — but only if no stale rope_deltas from an earlier generate() remain.
        self.lm.model.rope_deltas = None
        if labels is None:
            return self.lm(inputs_embeds=embeds, attention_mask=attention_mask, **kwargs)
        # Loss on labelled positions only, lm_head in checkpointed chunks: full (B, T, 248k) fp32 logits are ~4.5 GB
        # per microbatch, and most positions (audio, prompt) carry no label.
        hidden = self.lm.model(inputs_embeds=embeds, attention_mask=attention_mask, **kwargs)[0][:, :-1]
        target = labels[:, 1:]
        keep = target != -100
        h, t = hidden[keep], target[keep]
        tok_loss = torch.cat([
            checkpoint(_chunk_ce, self.lm.lm_head.weight, h[s:s + CE_CHUNK], t[s:s + CE_CHUNK], use_reentrant=False)
            for s in range(0, h.shape[0], CE_CHUNK)])
        per_pos = torch.zeros(target.shape, dtype=tok_loss.dtype, device=tok_loss.device).masked_scatter(keep, tok_loss)
        return SimpleNamespace(loss=tok_loss.mean(), loss_sum=tok_loss.sum(), n_tok=keep.sum(), tok_loss=per_pos, mask=keep)

    @torch.no_grad()
    def generate(self, input_ids, attention_mask=None, input_features=None, feature_attention_mask=None, **gen_kwargs):
        embeds = self.build_inputs_embeds(input_ids, input_features, feature_attention_mask)
        self.lm.model.rope_deltas = None
        return self.lm.generate(inputs_embeds=embeds, attention_mask=attention_mask, **gen_kwargs)
