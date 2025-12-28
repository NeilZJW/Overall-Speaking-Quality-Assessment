#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：wavlm 
@File    ：scoremodel.py
@Author  ：Neil
@Date    ：2025/12/18 21:52 
"""
import os
import json
import math
import random
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchaudio

from WavLM import WavLM, WavLMConfig
from utils import *


# Model: WavLM + pooling + regressor
class WavLMScoreModel(nn.Module):
    def __init__(self, wavlm: WavLM, cfg, layer_index: int = -1, target_key: str = "total"):
        """
        layer_index:
          -1: use last layer output (rep_last)
          1..24: use saved layer_reps[i] with ret_layer_results=True
        """
        super().__init__()
        self.wavlm = wavlm
        self.cfg = cfg
        self.layer_index = layer_index

        # WavLM-Large: D=1024, mean+std -> 2048
        self.regressor = nn.Sequential(
            nn.Linear(2048, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    @torch.no_grad()
    def extract_u(self, wav_1xT: torch.Tensor) -> torch.Tensor:
        wav_1xT = maybe_normalize(wav_1xT, self.cfg)

        if self.layer_index == -1:
            rep_last = self.wavlm.extract_features(wav_1xT)[0]  # (B,T,D)
            return mean_std_pooling(rep_last)  # (B,2048)

        rep_last, layer_results = self.wavlm.extract_features(
            wav_1xT,
            output_layer=self.wavlm.cfg.encoder_layers,
            ret_layer_results=True
        )[0]
        layer_reps = [x.transpose(0, 1).contiguous() for x, _ in layer_results]  # each (B,T,D)
        x = layer_reps[self.layer_index]
        return mean_std_pooling(x)

    def forward(self, wav_1xT: torch.Tensor) -> torch.Tensor:
        # freeze WavLM by default; train regressor
        with torch.no_grad():
            u = self.extract_u(wav_1xT)  # (B,2048)
        y = self.regressor(u).squeeze(-1)  # (B,)
        return y


# Metrics
def rmse(pred: np.ndarray, gold: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred - gold) ** 2)))

def pearson(pred: np.ndarray, gold: np.ndarray) -> float:
    if len(pred) < 2:
        return 0.0
    return float(np.corrcoef(pred, gold)[0, 1])


# Train / Eval
def train_one_epoch(model, loader, optimizer, device):
    model.train()
    mse = nn.MSELoss()
    losses = []

    for utt_id, wav_path, y in loader:
        wav = load_wav_16k_mono(wav_path).to(device)  # (1,T)
        y = y.to(device).view(1)                      # (1,)

        pred = model(wav)                             # (1,)
        loss = mse(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
    # for step, (utt_id, wav_path, y) in enumerate(loader, start=1):
    #     wav = load_wav_16k_mono(wav_path).to(device)  # (1,T)
    #     y = y.to(device).view(1)                      # (1,)
    #
    #     pred = model(wav)                             # (1,)
    #     loss = mse(pred, y)
    #
    #     optimizer.zero_grad()
    #     loss.backward()
    #     optimizer.step()
    #
    #     loss_val = float(loss.item())
    #     losses.append(loss_val)
    #     running += loss_val
    #
    #     # if step == 1 or step % log_every == 0:
    #     avg_loss = running / step
    #     lr = optimizer.param_groups[0]["lr"]
    #     print(
    #         f"step {step}/{total_steps} | "
    #         f"loss={loss_val:.4f} | "
    #         f"avg_loss={avg_loss:.4f} | "
    #         f"lr={lr:.2e}"
    #     )
    return float(np.mean(losses)) if losses else 0.0

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    preds, golds = [], []

    for utt_id, wav_path, y in loader:
        wav = load_wav_16k_mono(wav_path).to(device)
        pred = float(model(wav).item())
        preds.append(pred)
        golds.append(float(y.item()))

    preds = np.array(preds, dtype=np.float32)
    golds = np.array(golds, dtype=np.float32)
    return {
        "rmse": rmse(preds, golds),
        "pearson": pearson(preds, golds),
        "n": int(len(golds))
    }

