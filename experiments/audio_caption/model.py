"""
AudioLanguageModel: frozen CLAP + trainable linear projection + frozen Qwen3.5-xB.

Architecture:
    Audio chunks (N × 10s)
    → ClapModel.get_audio_features()              [frozen]    → (N, 512)
    → nn.Linear(512, qwen_hidden_size)            [trainable] → (N, qwen_hidden_size)
    → inject into <audio> token positions
    → Qwen3.5 LM                                  [frozen]    → text logits

Only the projection layer is trainable.
"""

from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast

AUDIO_TOKEN = "<audio>"


class AudioLanguageModel(nn.Module):
    """
    Multimodal audio-language model.

    The model expects input_ids containing one <audio> token per 10s audio chunk.
    During forward(), each <audio> token's embedding is replaced with the corresponding
    projected CLAP embedding before Qwen processes the sequence.
    """

    def __init__(
        self,
        audio_token_id: int,
        qwen_model_id: str = "Qwen/Qwen3.5-9B",
        clap_dim: int = 512,
        llm_dim: int = 4096,
        torch_dtype=torch.bfloat16,
    ):
        super().__init__()
        self.audio_token_id = audio_token_id

        # Trainable projection: CLAP dim → LLM hidden dim
        # (CLAP inference is handled by the DataCollator, not this model)
        self.projection = nn.Linear(clap_dim, llm_dim, bias=True)
        nn.init.normal_(self.projection.weight, std=0.02)
        nn.init.zeros_(self.projection.bias)
        self.projection = self.projection.to(torch_dtype)

        # Frozen Qwen LLM
        self.llm = AutoModelForCausalLM.from_pretrained(
            qwen_model_id,
            torch_dtype=torch_dtype,
            attn_implementation="sdpa",
        )
        for param in self.llm.parameters():
            param.requires_grad = False

        # Expose LLM config so HF Trainer recognises this as a LM
        self.config = self.llm.config
        self.generation_config = self.llm.generation_config

    def get_input_embeddings(self):
        return self.llm.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.llm.set_input_embeddings(value)

    def resize_token_embeddings(self, new_num_tokens: int, pad_to_multiple_of: int = None):
        return self.llm.resize_token_embeddings(new_num_tokens, pad_to_multiple_of)

    def add_model_tags(self, tags):
        self.llm.add_model_tags(tags)

    def __getattr__(self, name: str):
        # Delegate unknown HF Trainer compatibility calls (e.g. gradient_checkpointing_enable)
        # to the inner LLM. nn.Module.__getattr__ raises AttributeError for missing attrs,
        # so we only reach here for things not set on this module itself.
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.llm, name)

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
        audio_features: torch.FloatTensor,
        labels: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        """
        Args:
            input_ids:       (B, seq_len)             — contains <audio> token(s) + text
            attention_mask:  (B, seq_len)
            audio_features:  (total_chunks, clap_dim)  — flat across the batch
            labels:          (B, seq_len)              — -100 for audio + prompt tokens
        """
        projected = self.projection(audio_features.to(self.projection.weight.dtype))
        audio_mask = (input_ids == self.audio_token_id)

        num_audio_tokens = int(audio_mask.sum().item())
        num_audio_features = int(projected.shape[0])
        if num_audio_tokens != num_audio_features:
            raise ValueError(
                "Audio feature count mismatch in model forward. "
                f"audio_tokens={num_audio_tokens}, audio_features={num_audio_features}."
            )

        audio_token_counts = kwargs.pop("audio_token_counts", None)
        if audio_token_counts is not None:
            expected_tokens = int(audio_token_counts.sum().item())
            if expected_tokens != num_audio_tokens:
                raise ValueError(
                    "Batch audio token counts mismatch in model forward. "
                    f"sum(audio_token_counts)={expected_tokens}, audio_tokens={num_audio_tokens}."
                )

        inputs_embeds = self.llm.model.embed_tokens(input_ids)
        if inputs_embeds.shape[-1] != projected.shape[-1]:
            raise ValueError(
                "Projection output dimension does not match model embedding dimension. "
                f"projection_dim={projected.shape[-1]}, embed_dim={inputs_embeds.shape[-1]}."
            )

        if num_audio_tokens > 0:
            inputs_embeds = inputs_embeds.clone()
            flat_inputs_embeds = inputs_embeds.view(-1, inputs_embeds.shape[-1])
            flat_audio_mask = audio_mask.view(-1)
            flat_inputs_embeds[flat_audio_mask] = projected.to(flat_inputs_embeds.dtype)

        kwargs.pop("input_ids", None)
        kwargs.setdefault("return_dict", True)
        return self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs,
        )
