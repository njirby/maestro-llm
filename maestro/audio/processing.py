"""
Audio loading and chunking utilities shared across experiments.
"""

import math
from typing import List

import librosa
import numpy as np
import torch

CLAP_SAMPLE_RATE = 48_000
CHUNK_DURATION_S = 10
CHUNK_SAMPLES = CLAP_SAMPLE_RATE * CHUNK_DURATION_S  # 480_000


def load_audio_clip(audio_path: str, sample_rate: int = CLAP_SAMPLE_RATE) -> np.ndarray:
    """
    Load audio as a mono float32 waveform at the target sample rate.

    Returns:
        1-D numpy array of shape (num_samples,), dtype float32.
    """
    audio, _ = librosa.load(audio_path, sr=sample_rate, mono=True)
    return audio.astype(np.float32, copy=False)


def load_and_chunk(
    audio_path: str,
    sample_rate: int = CLAP_SAMPLE_RATE,
    chunk_samples: int = CHUNK_SAMPLES,
) -> List[np.ndarray]:
    """
    Load an audio file and split it into fixed-length chunks.

    Args:
        audio_path: Path to audio file (any format librosa supports).
        sample_rate: Target sample rate in Hz.
        chunk_samples: Samples per chunk. Last chunk is zero-padded if shorter.

    Returns:
        List of numpy arrays, each of shape (chunk_samples,), dtype float32.
    """
    audio, _ = librosa.load(audio_path, sr=sample_rate, mono=True)
    num_chunks = max(1, math.ceil(len(audio) / chunk_samples))
    chunks = []
    for i in range(num_chunks):
        chunk = audio[i * chunk_samples : (i + 1) * chunk_samples]
        if len(chunk) < chunk_samples:
            chunk = torch.nn.functional.pad(
                torch.tensor(chunk), (0, chunk_samples - len(chunk))
            ).numpy()
        chunks.append(chunk)
    return chunks
