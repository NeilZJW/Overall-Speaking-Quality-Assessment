#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：wavlm 
@File    ：train.py
@Author  ：Neil
@Date    ：2025/12/18 21:38 
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
import csv
import matplotlib.pyplot as plt

from WavLM import WavLM, WavLMConfig
from utils import *
from scoremodel import *
import warnings
warnings.filterwarnings("ignore")


def start_train():
    root_dir = r"Datasets/speechocean762"
    ckpt_path = r"pre_trained/WavLM-Large.pt"
    train_logs = []

    seed = 1314
    set_seed(seed)
    best_pearson = -1e9
    best_epoch = -1

    save_dir = f"outputs/{seed}"
    os.makedirs(save_dir, exist_ok=True)

    log_csv_path = os.path.join(save_dir, "train_log.csv")
    best_ckpt_path = os.path.join(save_dir, "best_model.pth")
    last_ckpt_path = os.path.join(save_dir, "last_model.pth")

    # sentence-level: "total" (recommended), or "accuracy"/"fluency"/"prosodic"/"completeness"
    target_key = "total"
    layer_index = -1

    epochs = 20
    lr = 1e-3

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load dataset
    train_ds = SpeechOcean762Dataset(root_dir, split="train", target_key=target_key)
    test_ds = SpeechOcean762Dataset(root_dir, split="test", target_key=target_key)
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, collate_fn=collate_bs1)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, collate_fn=collate_bs1)

    print(f"Train items: {len(train_ds)} | Test items: {len(test_ds)} | target={target_key}")

    # Load WavLM
    checkpoint = torch.load(ckpt_path)
    cfg = WavLMConfig(checkpoint["cfg"])
    wavlm = WavLM(cfg)
    wavlm.load_state_dict(checkpoint["model"])
    wavlm.eval()
    wavlm.to(device)

    # Model
    model = WavLMScoreModel(wavlm, cfg, layer_index=layer_index).to(device)

    # Train only the regressor head
    optimizer = torch.optim.Adam(model.regressor.parameters(), lr=lr)
    for ep in range(1, epochs + 1):
        print(f"Epoch {ep:02d}, training...", end="")
        tr_mse = train_one_epoch(model, train_loader, optimizer, device)
        metrics = evaluate(model, test_loader, device)
        print(
            f"\rEpoch {ep:02d} | train_mse={tr_mse:.4f} | "
            f"test_rmse={metrics['rmse']:.4f} | test_pearson={metrics['pearson']:.4f} | n={metrics['n']}"
        )
        # ---- record log ----
        train_logs.append({
            "epoch": ep,
            "train_mse": tr_mse,
            "test_rmse": metrics["rmse"],
            "test_pearson": metrics["pearson"]
        })

        # ---- save last model ----
        torch.save({
            "epoch": ep,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "test_pearson": metrics["pearson"]
        }, last_ckpt_path)

        # ---- save best model (by Pearson) ----
        if metrics["pearson"] > best_pearson:
            best_pearson = metrics["pearson"]
            best_epoch = ep
            torch.save({
                "epoch": ep,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "test_pearson": metrics["pearson"]
            }, best_ckpt_path)
            print(f"✔ New best model at epoch {ep}, pearson={best_pearson:.4f}")


    with open(log_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["epoch", "train_mse", "test_rmse", "test_pearson"]
        )
        writer.writeheader()
        for row in train_logs:
            writer.writerow(row)

    print(f"Training log saved to: {log_csv_path}")
    print(f"Best epoch: {best_epoch}, best pearson: {best_pearson:.4f}")

    epochs = [x["epoch"] for x in train_logs]
    train_mse = [x["train_mse"] for x in train_logs]
    test_rmse = [x["test_rmse"] for x in train_logs]
    test_pearson = [x["test_pearson"] for x in train_logs]

    # ---- Loss curve ----
    plt.figure()
    plt.plot(epochs, train_mse, label="Train MSE")
    plt.plot(epochs, test_rmse, label="Test RMSE")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, "loss_curve.png"))
    plt.close()

    # ---- Pearson curve ----
    plt.figure()
    plt.plot(epochs, test_pearson, label="Test Pearson")
    plt.xlabel("Epoch")
    plt.ylabel("Pearson Correlation")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, "pearson_curve.png"))
    plt.close()

    print("Training curves saved:")
    print(" - loss_curve.png")
    print(" - pearson_curve.png")


if __name__ == '__main__':
    start_train()


