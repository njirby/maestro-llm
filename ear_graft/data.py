"""wav -> 16 kHz mono -> Whisper mel (the Omni tower's front-end), and chat-formatted samples for the 4B."""
import numpy as np
import soundfile as sf
import soxr
import torch
from transformers import AutoTokenizer, WhisperFeatureExtractor
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import _get_feat_extract_output_lengths

from .model import OMNI_PATH, QWEN4B_PATH

SR = 16000
AUDIO_SPAN = "<|audio_start|>{}<|audio_end|>"
# 4B chat template (verified against apply_chat_template). The user turn mirrors what the teacher saw: one clip is
# just the audio then the prompt; two clips are "Audio 1:" <audio> "Audio 2:" <audio> then the prompt.
USER = "<|im_start|>user\n{content}<|im_end|>\n<|im_start|>assistant\n"
THINK_OPEN = "<think>\n"                  # enable_thinking=True generation prompt; target = reasoning + close + answer
NO_THINK = "<think>\n\n</think>\n\n"      # enable_thinking=False generation prompt; target = answer


def load_audio(path, max_seconds=30.0):
    y, sr = sf.read(path, dtype="float32", always_2d=True)
    y = y.mean(axis=1)
    if sr != SR:
        y = soxr.resample(y, sr, SR)
    return y[: int(max_seconds * SR)]


def n_audio_tokens(n_mel_frames):
    return int(_get_feat_extract_output_lengths(torch.tensor(n_mel_frames)))


def render(prompt, n_tokens_per_clip, think, reasoning=None, answer=None):
    """-> (prompt_text, target_text). target_text is "" when answer is None (generation)."""
    spans = [AUDIO_SPAN.format("<|audio_pad|>" * n) for n in n_tokens_per_clip]
    if len(spans) <= 1:  # one clip, or none (a text-only turn)
        content = "".join(spans) + prompt
    else:
        content = "".join(f"Audio {k + 1}:{s}" for k, s in enumerate(spans)) + prompt
    head = USER.format(content=content) + (THINK_OPEN if think else NO_THINK)
    if answer is None:
        return head, ""
    target = (f"{reasoning.strip()}\n</think>\n\n" if think else "") + answer.strip() + "<|im_end|>\n"
    return head, target


class Collator:
    """Batches of {"audios": [np.ndarray (16 kHz) | path, ...], "prompt": str, "answer": str | None,
    "reasoning": str | None, "think": bool}. A single "audio" key and "target" (answer) are also accepted."""

    def __init__(self, qwen_path=QWEN4B_PATH, omni_path=OMNI_PATH, max_seconds=30.0):
        self.tok = AutoTokenizer.from_pretrained(qwen_path)
        self.fe = WhisperFeatureExtractor.from_pretrained(omni_path)
        self.max_seconds = max_seconds

    def _wav(self, a):
        return a if isinstance(a, np.ndarray) else load_audio(a, self.max_seconds)

    def __call__(self, batch):
        clips = [[self._wav(a) for a in (b["audios"] if "audios" in b else [b["audio"]])] for b in batch]
        flat = [w for c in clips for w in c]  # clip order = order of audio spans in the batch's input_ids
        feats = (self.fe(flat, sampling_rate=SR, padding=True, truncation=False, return_attention_mask=True, return_tensors="pt")
                 if flat else None)  # text-only batch
        counts = [n_audio_tokens(n) for n in feats["attention_mask"].sum(-1).tolist()] if flat else []
        ids, labels, k = [], [], 0
        for b, c in zip(batch, clips):
            answer = b.get("answer", b.get("target"))
            head, target = render(b["prompt"], counts[k:k + len(c)], b.get("think", False), b.get("reasoning"), answer)
            k += len(c)
            p = self.tok(head, add_special_tokens=False).input_ids
            t = self.tok(target, add_special_tokens=False).input_ids if target else []
            ids.append(p + t)
            labels.append([-100] * len(p) + t)
        width = max(map(len, ids))
        pad_id = self.tok.pad_token_id
        out = {
            "input_ids": torch.tensor([x + [pad_id] * (width - len(x)) for x in ids]),
            "attention_mask": torch.tensor([[1] * len(x) + [0] * (width - len(x)) for x in ids]),
            "labels": torch.tensor([x + [-100] * (width - len(x)) for x in labels]),
        }
        if feats is not None:
            out.update(input_features=feats["input_features"], feature_attention_mask=feats["attention_mask"])
        return out
