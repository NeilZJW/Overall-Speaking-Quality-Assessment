# !/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：wavlm 
@File    ：main.py
@Author  ：Neil
@Date    ：2025/12/18 20:45 
"""
import torch
import torchaudio
from WavLM import WavLM, WavLMConfig
from utils import *


def main():
    wav_path = "Haochen.wav"
    ckpt_path = "pre_trained/WavLM-Large.pt"

    # load the pre-trained checkpoints
    checkpoint = torch.load(ckpt_path)
    cfg = WavLMConfig(checkpoint["cfg"])
    model = WavLM(cfg)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    # # extract the representation of last layer (official demo style)
    # wav_input_16khz = load_wav(wav_path, 16000)
    # wav_input_16khz = maybe_normalize(wav_input_16khz, cfg)
    #
    # out = model.extract_features(wav_input_16khz)  # tuple(len=2)
    # rep_last = out[0]  # Tensor (B, T, D)
    # print_out_info(out, tag="out(last)")
    #
    # # utterance-level mean pooling (B, D)
    # utt_mean = rep_last.mean(dim=1)
    # print("utt_mean shape:", utt_mean.shape)  # e.g. (1, 1024)


    # extract the representation of each layer (official demo style)
    wav_input_16khz = load_wav(wav_path, 16000)
    wav_input_16khz = maybe_normalize(wav_input_16khz, cfg)

    out2 = model.extract_features(
        wav_input_16khz,
        output_layer=model.cfg.encoder_layers,
        ret_layer_results=True
    )

    # out2[0] is a tuple: (rep, layer_results)
    print("out2[0] type:", type(out2[0]))
    print("out2[0] =", out2[0] if not isinstance(out2[0], tuple) else ("tuple_len", len(out2[0])))

    rep_last2, layer_results = out2[0]  # rep_last2: (B,T,D)
    print("rep_last2 shape:", rep_last2.shape)

    layer_reps = []
    for i, (x, _) in enumerate(layer_results):
        if x.dim() == 3:
            # typically (T,B,D)
            x_bt_d = x.transpose(0, 1).contiguous()
            layer_reps.append(x_bt_d)
            print(f"layer_results[{i}] x shape: {tuple(x.shape)} -> layer_reps[{i}] shape: {tuple(x_bt_d.shape)}")
        else:
            # unexpected, still store it and print
            layer_reps.append(x)
            print(f"layer_results[{i}] x shape (unexpected): {tuple(x.shape)}")

    print("num layer_reps:", len(layer_reps))
    # pick one layer and do utterance mean pooling
    if len(layer_reps) > 0 and hasattr(layer_reps[-1], "mean"):
        x = layer_reps[-1]
        utt_mean_last_saved = mean_std_pooling(x)
        print("utt_mean (last saved layer) shape:", utt_mean_last_saved.shape)

if __name__ == "__main__":
    main()
