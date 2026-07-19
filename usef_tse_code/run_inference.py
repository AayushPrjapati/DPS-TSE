import os
import sys
# Add current directory to path to resolve local imports (models, utils)
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.append(script_dir)

import argparse
import torch
import numpy as np
import librosa
import soundfile as sf
from hyperpyyaml import load_hyperpyyaml
from collections import OrderedDict

def load_pretrained_modules(model, ckpt_path):
    model_info = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    state_dict = OrderedDict()
    
    # Check if checkpoint contains 'model_state_dict' key
    if 'model_state_dict' in model_info:
        weights = model_info['model_state_dict']
    else:
        weights = model_info

    for k, v in weights.items():
        name = k.replace("module.", "").replace("convolution_", "convolution_module.")
        state_dict[name] = v
        
    model.load_state_dict(state_dict)
    return model

def main():
    parser = argparse.ArgumentParser(description='USEF-TSE Custom File Inference')
    parser.add_argument('-c', '--config', type=str, required=True, help='Path to config yaml file')
    parser.add_argument('-p', '--checkpoint', type=str, required=True, help='Path to pretrained checkpoint file')
    parser.add_argument('-m', '--mixture', type=str, required=True, help='Path to input mixture wav file')
    parser.add_argument('-a', '--aux', type=str, required=True, help='Path to auxiliary speaker cue wav file')
    parser.add_argument('-o', '--output', type=str, default='output.wav', help='Path to save output wav file')
    args = parser.parse_args()

    # Verify input paths
    for path in [args.config, args.checkpoint, args.mixture, args.aux]:
        assert os.path.exists(path), f"File not found: {path}"

    # Load configuration
    with open(args.config, 'r') as f:
        config_strings = f.read()
    config = load_hyperpyyaml(config_strings)
    fs = config['sample_rate']
    print(f"INFO: Loaded hparams from: {args.config} (Target Sample Rate: {fs} Hz)")

    # Load audio waveforms
    print("Loading mixture and aux waveforms...")
    import soundfile as sf
    import soxr
    
    def load_and_resample(path, target_sr, max_duration=None):
        wav, sr = sf.read(path)
        if len(wav.shape) > 1:
            wav = wav[:, 0]  # Take first channel if stereo
            
        if sr != target_sr:
            wav = soxr.resample(wav, sr, target_sr)
            
        if max_duration is not None:
            max_samples = int(max_duration * target_sr)
            if len(wav) > max_samples:
                wav = wav[:max_samples]
        return wav

    # WSJ0-2mix models are trained on 4s mixtures and 3s cues. 
    # Long sequences dilute the cross-attention. We crop to 5s mix and 3s aux for stable inference.
    mix_wav = load_and_resample(args.mixture, fs)
    aux_wav = load_and_resample(args.aux, fs, max_duration=3.0)

    # Initialize model
    print("Initializing model...")
    model = config['modules']['masknet']
    model = load_pretrained_modules(model, args.checkpoint)

    # Use GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running inference on device: {device}")
    model = model.to(device)

    # Convert aux to tensor once
    aux_tensor = torch.from_numpy(aux_wav).float().unsqueeze(0).to(device)

    model.eval()
    
    print("Extracting target speech in chunks...")
    chunk_size = int(4.0 * fs) # 4 second chunks
    extracted_chunks = []
    
    with torch.no_grad():
        for start_idx in range(0, len(mix_wav), chunk_size):
            end_idx = min(start_idx + chunk_size, len(mix_wav))
            mix_chunk = mix_wav[start_idx:end_idx]
            
            # Pad the last chunk if it's too short (less than 1 sec), though the model can usually handle variable length
            # We'll just pass it as is
            mix_tensor = torch.from_numpy(mix_chunk).float().unsqueeze(0).to(device)
            
            est_source = model(mix_tensor, aux_tensor)
            est_source = est_source.squeeze(0).cpu().numpy()
            extracted_chunks.append(est_source)

    final_est_source = np.concatenate(extracted_chunks)

    sf.write(args.output, final_est_source, fs)
    print(f"SUCCESS: Extracted full target speech saved to: {args.output}")

if __name__ == '__main__':
    main()
