"""
Infer script for AASIST model - run detection on single or directory of audio files.

Usage:
  python tools/infer.py --config ./config/AASIST.conf --weights ./models/weights/AASIST.pth --input test.wav --out results.csv

This script will:
 - Load configuration and model
 - For each audio input, resample (if needed), pad/truncate to 64600 samples
 - Run the model and store score/probabilities
 - Optionally threshold at 0.5 to classify as bonafide/spoof
"""

import argparse
import json
import os
from pathlib import Path
from typing import List

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

from data_utils import pad
from importlib import import_module


def find_audio_files(path: Path) -> List[Path]:
    files = []
    if path.is_file():
        files.append(path)
    elif path.is_dir():
        for ext in (".wav", ".flac", ".mp3", ".ogg"):
            files.extend(list(path.rglob(f"*{ext}")))
    return sorted(files)


def resample_audio(x, orig_sr, target_sr=16000):
    if orig_sr == target_sr:
        return x
    # naive resampling using linear interpolation
    duration = x.shape[0] / orig_sr
    num_samples = int(duration * target_sr)
    if num_samples <= 0:
        return np.zeros(target_sr)
    x_old_idx = np.linspace(0, duration, num=x.shape[0], endpoint=False)
    x_new_idx = np.linspace(0, duration, num=num_samples, endpoint=False)
    x_resampled = np.interp(x_new_idx, x_old_idx, x)
    return x_resampled


def load_model_from_config(config_file, model_weights, device='cpu'):
    with open(config_file, 'r') as f:
        config = json.load(f)
    model_config = config['model_config']
    module = import_module(f"models.{model_config['architecture']}")
    _model = getattr(module, 'Model')
    model = _model(model_config).to(device)
    state = torch.load(model_weights, map_location=device)
    try:
        model.load_state_dict(state)
    except RuntimeError:
        # some models saved with a 'module.' prefix (DataParallel)
        try:
            from collections import OrderedDict
            new_state = OrderedDict()
            for k, v in state.items():
                name = k[7:] if k.startswith('module.') else k
                new_state[name] = v
            model.load_state_dict(new_state)
        except Exception as e:
            raise e
    model.eval()
    return model


def infer_file(model, file_path: Path, device='cpu', nb_samp=64600):
    audio, sr = sf.read(str(file_path))
    # if stereo, convert to mono
    if len(audio.shape) == 2:
        audio = np.mean(audio, axis=1)
    audio_rs = resample_audio(audio, orig_sr=sr, target_sr=16000)
    audio_padded = pad(audio_rs, max_len=nb_samp)
    x = torch.tensor(audio_padded, dtype=torch.float32).to(device)
    x = x.unsqueeze(0)  # batch
    with torch.no_grad():
        _, out = model(x)
        probs = F.softmax(out, dim=1).cpu().numpy()[0]
    # probs: [spoof, bonafide]? In net it's likely index 1 = bonafide
    # We'll return both
    return probs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='Path to config json', required=True)
    parser.add_argument('--weights', type=str, help='Path to model weights .pth', required=True)
    parser.add_argument('--input', type=str, help='Input file or directory', required=True)
    parser.add_argument('--out', type=str, help='Output csv path', default='infer_results.csv')
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--threshold', type=float, default=0.5, help='Probability threshold for bonafide')
    args = parser.parse_args()

    inp = Path(args.input)
    files = find_audio_files(inp)
    if len(files) == 0:
        print('No audio files found for', inp)
        return

    # load config to get model nb_samp
    with open(args.config, 'r') as f:
        config = json.load(f)
    nb_samp = config['model_config'].get('nb_samp', 64600)

    model = load_model_from_config(args.config, args.weights, device=args.device)

    out_lines = ['file,prob_spoof,prob_bonafide,prediction']
    for file in files:
        probs = infer_file(model, file, device=args.device, nb_samp=nb_samp)
        # assume index 1 is bonafide
        prob_spoof = float(probs[0])
        prob_bonafide = float(probs[1])
        pred = 'bonafide' if prob_bonafide >= args.threshold else 'spoof'
        print(f'{file}: bonafide_prob={prob_bonafide:.4f}, spoof_prob={prob_spoof:.4f} -> {pred}')
        out_lines.append(f'{file},{prob_spoof},{prob_bonafide},{pred}')

    with open(args.out, 'w') as f:
        f.write('\n'.join(out_lines))
    print('Results saved to', args.out)


if __name__ == '__main__':
    main()
