#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：wavlm 
@File    ：utils.py
@Author  ：Neil
@Date    ：2025/12/18 20:46 
"""
import torch
import torchaudio
from WavLM import WavLM, WavLMConfig
import random
import numpy as np
from typing import Dict, List, Tuple
import json
from torch.utils.data import Dataset, DataLoader
import os


def load_wav_16k_mono(wav_path: str, target_sr: int = 16000) -> torch.Tensor:
    wav, sr = torchaudio.load(wav_path, backend="soundfile")  # (C, T)
    wav = wav.mean(dim=0, keepdim=True)  # (1, T)
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    return wav  # (1, T)

def load_wav(wav_path, target_sr=16000):
    """
    Load wav -> mono -> resample to target_sr (official demo style).
    Return: Tensor (1, T)
    """
    speech, sample_rate = torchaudio.load(wav_path, backend="soundfile")  # (C, T)
    speech = speech.mean(dim=0, keepdim=True)  # -> (1, T)
    # allow both upsample and downsample
    if sample_rate != target_sr:
        speech = torchaudio.functional.resample(speech, sample_rate, target_sr)
    return speech

def maybe_normalize(wav, cfg):
    """
    Follow the official demo: if cfg.normalize, apply layer_norm over full shape.
    wav: (1, T) or (B, T)
    """
    if cfg.normalize:
        wav = torch.nn.functional.layer_norm(wav, wav.shape)
    return wav

def print_out_info(out, tag="out"):
    """
    out is the return of model.extract_features(...)
    In your case: out is a tuple(len=2), out[0] is Tensor (B,T,D)
    """
    print(f"{tag} type:", type(out))
    if isinstance(out, tuple):
        print(f"len({tag}):", len(out))
        print(f"{tag}[0] type:", type(out[0]))
        if hasattr(out[0], "shape"):
            print(f"{tag}[0] shape:", out[0].shape)
        print(f"{tag}[1] type:", type(out[1]))
    else:
        print(f"{tag} is not tuple, value:", out)

def mean_std_pooling(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Mean + Std pooling over time dimension.

    Args:
        x: Tensor of shape (B, T, D)
        eps: small value to avoid numerical issues

    Returns:
        u: Tensor of shape (B, 2D)  where u = [mean, std]
    """
    # mean over time (T)
    mean = x.mean(dim=1)                  # (B, D)

    # std over time (T)
    std = x.std(dim=1).clamp_min(eps)     # (B, D)

    # concat -> (B, 2D)
    u = torch.cat([mean, std], dim=-1)
    return u


def masked_mean_std_pooling(
    x: torch.Tensor,
    padding_mask: torch.Tensor,
    eps: float = 1e-6
) -> torch.Tensor:
    """
    Masked Mean + Std pooling over time dimension.

    Args:
        x: Tensor (B, T, D)
        padding_mask: Bool Tensor (B, T), True means PAD
        eps: numerical stability

    Returns:
        u: Tensor (B, 2D)
    """
    valid = (~padding_mask).unsqueeze(-1).to(x.dtype)  # (B, T, 1) with 1 for valid frames
    denom = valid.sum(dim=1).clamp_min(1.0)            # (B, 1)

    mean = (x * valid).sum(dim=1) / denom              # (B, D)

    var = ((x - mean.unsqueeze(1)) ** 2) * valid
    std = torch.sqrt(var.sum(dim=1) / denom).clamp_min(eps)  # (B, D)

    return torch.cat([mean, std], dim=-1)              # (B, 2D)

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# Kaldi-style file readers
def read_wav_scp(path: str) -> Dict[str, str]:
    """
    Read Kaldi wav.scp:
      utt_id <path>
    Return dict: utt_id -> wav_path
    """
    utt2wav = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            # Some wav.scp may have more than 2 parts (e.g., pipes).
            # SpeechOcean uses plain paths.
            utt_id = parts[0]
            wav_path = parts[1]
            utt2wav[utt_id] = wav_path
    return utt2wav

def load_scores_json(path: str) -> Dict[str, dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


class SpeechOcean762Dataset(Dataset):
    """
    SpeechOcean762 structure (per README):
      root/
        resource/scores.json
        train/wav.scp
        test/wav.scp
        WAVE/...

    We use sentence-level target score: scores[utt_id][target_key]
    Recommended: target_key="total"
    :contentReference[oaicite:1]{index=1}
    """
    def __init__(
            self, root_dir: str,
            resource: str = "resource",
            split: str = "train", target_key: str = "total"
    ):
        assert split in ("train", "test"), "split must be 'train' or 'test'"
        self.root_dir = root_dir
        self.resource = resource
        self.split = split
        self.target_key = target_key

        scores_path = os.path.join(root_dir, resource, "scores.json")
        wav_scp_path = os.path.join(root_dir, split, "wav.scp")

        if not os.path.isfile(scores_path):
            raise FileNotFoundError(f"Missing scores.json at: {scores_path}")
        if not os.path.isfile(wav_scp_path):
            raise FileNotFoundError(f"Missing {split}/wav.scp at: {wav_scp_path}")

        self.scores = load_scores_json(scores_path)         # utt_id -> dict
        utt2wav = read_wav_scp(wav_scp_path)                # utt_id -> wav_path (may be relative)

        items: List[Tuple[str, str, float]] = []
        missing_score = 0
        missing_wav = 0
        for utt_id, wav_path in utt2wav.items():
            if utt_id not in self.scores:
                missing_score += 1
                continue
            if self.target_key not in self.scores[utt_id]:
                missing_score += 1
                continue

            # Resolve path
            if not os.path.isabs(wav_path):
                wav_path = os.path.join(root_dir, wav_path)

            if not os.path.isfile(wav_path):
                missing_wav += 1
                continue

            y = float(self.scores[utt_id][self.target_key])
            items.append((utt_id, wav_path, y))

        if len(items) == 0:
            raise RuntimeError(
                f"No usable items found. missing_score={missing_score}, missing_wav={missing_wav}. "
                f"Check root_dir and wav.scp paths."
            )

        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        utt_id, wav_path, y = self.items[idx]
        return utt_id, wav_path, torch.tensor(y, dtype=torch.float32)


def collate_bs1(batch):
    # batch_size=1 for simplicity (no padding yet)
    utt_id, wav_path, y = batch[0]
    return utt_id, wav_path, y


