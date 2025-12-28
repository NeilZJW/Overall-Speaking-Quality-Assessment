#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：wavlm 
@File    ：test.py
@Author  ：Neil
@Date    ：2025/12/18 23:38 
"""
from scoremodel import *


def results_checking():
    wavlm_ckpt = "pre_trained/WavLM-Large.pt"
    best_ckpt = "outputs/0/best_model.pth"
    audio_dir = "Audios"
    audio_neil_dir = "Audios/Neil"
    results = []

    audio_path = [audio_dir + "/" + a for a in os.listdir(audio_dir) if a.endswith(".wav")]
    audio_neil_path = [audio_neil_dir + "/" + a for a in os.listdir(audio_neil_dir) if a.endswith(".wav")]
    layer_index = -1


    device = "cuda" if torch.cuda.is_available() else "cpu"

    # load wavlm
    ckpt = torch.load(wavlm_ckpt)
    cfg = WavLMConfig(ckpt["cfg"])
    wavlm = WavLM(cfg)
    wavlm.load_state_dict(ckpt["model"])
    wavlm.eval().to(device)

    # build model
    model = WavLMScoreModel(wavlm, cfg, layer_index=layer_index).to(device)
    # infer with BEST
    best_state = torch.load(best_ckpt)
    model.load_state_dict(best_state["model_state"])
    model.eval()

    # for audio in audio_path:
    for audio in audio_neil_path:
        # load audio
        wav = load_wav_16k_mono(audio).to(device)
        with torch.no_grad():
            pred_best = float(model(wav).item())
        results.append({
            "Audio": audio,
            "Score(BEST)": pred_best
        })
    results = sorted(results, key=lambda x: x["Score(BEST)"], reverse=True)
    print(" Inference Results (sorted by score) ".center(50, "*"))
    for r in results:
        print(f"Audio: {r['Audio']}")
        print(f"Score (BEST): {r['Score(BEST)']:.4f}")



if __name__ == "__main__":
    results_checking()














