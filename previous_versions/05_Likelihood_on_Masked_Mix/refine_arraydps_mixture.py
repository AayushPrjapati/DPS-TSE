import os
import sys
import torch
import soundfile as sf
import yaml
import numpy as np
from dotmap import DotMap

# Add ArrayDPS to path so we can import its modules
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(script_dir, '../../ArrayDPS')))

from src.utils.setup import load_ema_weights
from src.models.unet_wav import UNet1d
from src.sde import VE_Sde_Elucidating

def get_edm_schedule(num_steps, sigma_min, sigma_max, rho):
    step_indices = torch.arange(num_steps, dtype=torch.float32)
    sigma_min, sigma_max, rho = float(sigma_min), float(sigma_max), float(rho)
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])])
    return t_steps

def run_storm_refinement(input_path, mixture_path, output_path, config_path, checkpoint_path, sigma_start=0.5, num_steps=100, xi=1.0):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running on device: {device}")

    # Load configuration
    with open(config_path, 'r') as f:
        args = yaml.safe_load(f)
    args = DotMap(args)

    # Initialize Model
    print("Loading ArrayDPS Model...")
    model = UNet1d(args.unet_wav, device).to(device)
    model = load_ema_weights(model, checkpoint_path)
    model.eval()

    diff_parameters = VE_Sde_Elucidating(args.diffusion_parameters, args.diffusion_parameters.sigma_data)

    # Load Discriminative Output (y)
    print(f"Loading extracted audio from {input_path}")
    wav_y, sr = sf.read(input_path)
    if len(wav_y.shape) > 1: wav_y = wav_y[:, 0]
    
    # Load Original Mixture (y_mix)
    print(f"Loading original mixture from {mixture_path}")
    wav_mix, sr_mix = sf.read(mixture_path)
    if len(wav_mix.shape) > 1: wav_mix = wav_mix[:, 0]

    # Ensure same length
    min_len = min(wav_y.shape[-1], wav_mix.shape[-1])
    wav_y = wav_y[:min_len]
    wav_mix = wav_mix[:min_len]
    
    sig_len = wav_y.shape[-1]
    target_len = int(2**np.ceil(np.log2(sig_len)))
    
    y = torch.nn.functional.pad(torch.from_numpy(wav_y).float().unsqueeze(0).to(device), (0, target_len - sig_len))
    y_mix = torch.nn.functional.pad(torch.from_numpy(wav_mix).float().unsqueeze(0).to(device), (0, target_len - sig_len))

    # --- 1. MASK GENERATION (Energy Thresholding of y) ---
    print("Calculating energy mask based on extracted audio...")
    y_np = y.squeeze().cpu().numpy()
    window_size = 800 # 100ms at 8kHz
    squared = y_np ** 2
    energy = np.convolve(squared, np.ones(window_size)/window_size, mode='same')
    
    threshold = 0.005 * np.max(energy) 
    M_np = (energy > threshold).astype(np.float32)
    M = torch.from_numpy(M_np).unsqueeze(0).to(device)
    print(f"Mask created. Clean Speech Ratio: {(M.sum() / M.numel() * 100):.2f}%")

    # Generate Schedule
    print(f"Starting refinement from sigma = {sigma_start}")
    t_steps = get_edm_schedule(num_steps, args.diffusion_parameters.sigma_min, args.diffusion_parameters.sigma_max, args.diffusion_parameters.ro).to(device)
    start_idx = (torch.abs(t_steps - sigma_start)).argmin()
    t_steps = t_steps[start_idx:]

    # Inject initial noise
    x = y + t_steps[0] * torch.randn_like(y)

    # --- 2. REVERSE SDE LOOP (WITH JOINT MIXTURE GUIDANCE) ---
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
        x = x.detach().requires_grad_(True)
        t_tensor_cur = t_cur.unsqueeze(0)
        
        # Forward pass (Clean speech estimation)
        denoised = diff_parameters.denoiser(x, model, t_tensor_cur)
        
        # Joint Likelihood (The New Core Innovation)
        # 1. Pull towards the extracted audio where it's clean
        loss_clean = torch.linalg.norm(M * (denoised - y))
        
        # 2. Bound inside the original mixture where it's an overlapping gap
        loss_mix = torch.linalg.norm((1 - M) * torch.relu(denoised.abs() - y_mix.abs()))
        
        # Total Joint Loss
        loss = loss_clean + loss_mix
        
        # Calculate gradients
        rec_grads = torch.autograd.grad(outputs=loss, inputs=x)[0]
        
        # Scale guidance
        normguide = torch.linalg.norm(rec_grads) / (x.shape[-1] ** 0.5)
        s = xi / (normguide * t_cur + 1e-6)
        
        # Combine unconditional score + likelihood gradient
        uncond_score = (denoised.detach() - x) / (t_cur ** 2)
        score = uncond_score - s * rec_grads
        
        # 1st order Euler step
        d_cur = -t_cur * score
        x_next = x.detach() + (t_next - t_cur) * d_cur

        # 2nd order Heun correction
        if i < len(t_steps) - 2:
            with torch.no_grad():
                t_tensor_next = t_next.unsqueeze(0)
                denoised_next = diff_parameters.denoiser(x_next, model, t_tensor_next)
                d_prime = (x_next - denoised_next) / t_next
                x_next = x.detach() + (t_next - t_cur) * (0.5 * d_cur.detach() + 0.5 * d_prime)

        x = x_next
        
        if i % 10 == 0:
            print(f"Step {i}/{len(t_steps)-1}, sigma={t_cur.item():.4f}, LossClean={loss_clean.item():.4f}, LossMix={loss_mix.item():.4f}")

    print("Refinement complete!")
    refined_wav = x.detach().squeeze(0).cpu().numpy()[:sig_len]
    sf.write(output_path, refined_wav, sr)
    print(f"Saved refined audio to: {output_path}")

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    run_storm_refinement(
        input_path=os.path.join(script_dir, '../../test_samples/extracted_real_speech_whamr.wav'),
        mixture_path=os.path.join(script_dir, '../../test_samples/S01_P01_01_mix_mix.wav'),
        output_path=os.path.join(script_dir, '../../test_samples/refined_real_speech_mixture_guided.wav'),
        config_path=os.path.join(script_dir, '../../ArrayDPS/conf/conf_libritts_unet1d_attention_8k.yaml'),
        checkpoint_path=os.path.join(script_dir, '../../ArrayDPS/model_ckpt.pt'),
        sigma_start=0.5, 
        num_steps=100,
        xi=1.0 
    )
