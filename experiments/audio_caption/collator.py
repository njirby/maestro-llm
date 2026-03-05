"""
DataCollator for the audio_caption experiment.

Default mode expects preprocessed fixed 10s clips and inserts one <audio> token
per sample. Legacy mode can still perform dynamic chunking of full tracks.
"""

from dataclasses import dataclass
from typing import Any, Dict, List

import torch

from maestro.audio.processing import CLAP_SAMPLE_RATE, load_and_chunk, load_audio_clip


@dataclass
class AudioCaptionCollator:
    tokenizer: Any
    clap_processor: Any
    clap_model: Any
    audio_token_id: int
    device: str
    max_length: int = 512
    use_dynamic_chunking: bool = False

    def __post_init__(self):
        self._prefix_ids = self.tokenizer.encode(
            "<|im_start|>assistant\n", add_special_tokens=False
        )
        self._suffix_ids = self.tokenizer.encode(
            "<|im_end|>", add_special_tokens=False
        )

    def _extract_features(self, examples: List[Dict[str, Any]]) -> tuple[torch.Tensor, List[int]]:
        if self.use_dynamic_chunking:
            all_audio = []
            audio_per_example = []
            for ex in examples:
                chunks = load_and_chunk(ex["audio_path"])
                all_audio.extend(chunks)
                audio_per_example.append(len(chunks))
        else:
            all_audio = [load_audio_clip(ex["audio_path"]) for ex in examples]
            audio_per_example = [1] * len(examples)

        clap_inputs = self.clap_processor(
            audio=all_audio,
            sampling_rate=CLAP_SAMPLE_RATE,
            return_tensors="pt",
        )
        with torch.no_grad():
            out = self.clap_model.get_audio_features(**clap_inputs)

        audio_features = (out.pooler_output if hasattr(out, "pooler_output") else out).float().cpu()
        expected = sum(audio_per_example)
        if audio_features.ndim != 2 or audio_features.shape[0] != expected:
            raise ValueError(
                "CLAP feature extraction returned unexpected shape. "
                f"Expected ({expected}, D), got {tuple(audio_features.shape)}."
            )
        return audio_features, audio_per_example

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        audio_features, audio_per_example = self._extract_features(examples)
        features_per_example = audio_features.split(audio_per_example, dim=0)

        all_input_ids, all_labels = [], []
        all_audio_features = []
        audio_token_counts = []

        for sample_idx, (ex, n_audio, sample_features) in enumerate(
            zip(examples, audio_per_example, features_per_example)
        ):
            caption_ids = self.tokenizer.encode(ex["caption"], add_special_tokens=False)
            audio_tok_ids = [self.audio_token_id] * n_audio
            full_input_ids = audio_tok_ids + self._prefix_ids + caption_ids + self._suffix_ids
            n_masked = n_audio + len(self._prefix_ids)
            full_labels = [-100] * n_masked + caption_ids + self._suffix_ids

            input_ids = full_input_ids[: self.max_length]
            labels = full_labels[: self.max_length]
            retained_audio = sum(1 for token_id in input_ids if token_id == self.audio_token_id)

            if retained_audio > sample_features.shape[0]:
                raise ValueError(
                    "Retained audio token count exceeds available CLAP features. "
                    f"sample_idx={sample_idx}, retained_audio={retained_audio}, "
                    f"available_features={sample_features.shape[0]}, max_length={self.max_length}."
                )

            all_input_ids.append(input_ids)
            all_labels.append(labels)
            if retained_audio > 0:
                all_audio_features.append(sample_features[:retained_audio])
            audio_token_counts.append(retained_audio)

        batch_audio_features = (
            torch.cat(all_audio_features, dim=0)
            if all_audio_features
            else torch.empty((0, audio_features.shape[1]), dtype=audio_features.dtype)
        )
        total_audio_tokens = sum(audio_token_counts)
        if batch_audio_features.shape[0] != total_audio_tokens:
            raise ValueError(
                "Audio alignment invariant failed in collator. "
                f"sum(audio_token_counts)={total_audio_tokens}, "
                f"audio_features_rows={batch_audio_features.shape[0]}."
            )

        max_len = max(len(ids) for ids in all_input_ids)
        pad_id = self.tokenizer.pad_token_id
        padded_ids, padded_masks, padded_labels = [], [], []
        for input_ids, labels in zip(all_input_ids, all_labels):
            pad_len = max_len - len(input_ids)
            padded_ids.append(input_ids + [pad_id] * pad_len)
            padded_masks.append([1] * len(input_ids) + [0] * pad_len)
            padded_labels.append(labels + [-100] * pad_len)

        return {
            "input_ids": torch.tensor(padded_ids, dtype=torch.long),
            "attention_mask": torch.tensor(padded_masks, dtype=torch.long),
            "audio_features": batch_audio_features,
            "audio_token_counts": torch.tensor(audio_token_counts, dtype=torch.long),
            "labels": torch.tensor(padded_labels, dtype=torch.long),
        }
