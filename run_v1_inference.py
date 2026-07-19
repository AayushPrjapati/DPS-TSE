import argparse
import subprocess
import os
import sys
import shutil
import warnings
import torch
import soundfile as sf
import numpy as np

warnings.filterwarnings("ignore")

def check_and_download_checkpoints():
    print("Checking model checkpoints...")
    p1_path = "usef_tse_code/chkpt/chkpt/USEF-TFGridNet/whamr!/temp_best.pth.tar"
    p2_path = "ArrayDPS/model_ckpt.pt"
    
    # 1. Download Phase 1 weights if missing
    if not os.path.exists(p1_path):
        print("Phase 1 checkpoint (TFGridNet) is missing. Downloading from HF Hub...")
        try:
            from huggingface_hub import hf_hub_download
            hf_hub_download(
                repo_id='ZBang/USEF-TSE', 
                filename='chkpt/USEF-TFGridNet/whamr!/temp_best.pth.tar', 
                local_dir='usef_tse_code/chkpt'
            )
            print("Successfully downloaded Phase 1 weights.")
        except Exception as e:
            print(f"Error downloading Phase 1 weights: {e}")
            sys.exit(1)
            
    # 2. Download Phase 2 weights if missing
    if not os.path.exists(p2_path):
        print("Phase 2 checkpoint (ArrayDPS) is missing. Downloading from Box Server...")
        os.makedirs("ArrayDPS", exist_ok=True)
        try:
            import urllib.request
            urllib.request.urlretrieve(
                'https://uofi.box.com/shared/static/eent06t4b4hdkjf0vgjzsqw8defa3xbn.pt', 
                p2_path
            )
            print("Successfully downloaded Phase 2 weights.")
        except Exception as e:
            print(f"Error downloading Phase 2 weights: {e}")
            sys.exit(1)

def get_edm_schedule(num_steps, sigma_min, sigma_max, rho):
    sigma_min = float(sigma_min)
    sigma_max = float(sigma_max)
    rho = float(rho)
    steps = torch.arange(num_steps, dtype=torch.float32)
    t_steps = (sigma_max ** (1 / rho) + steps / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    return torch.cat([t_steps, torch.zeros_like(t_steps[:1])])

def main():
    parser = argparse.ArgumentParser(description="Run V1 End-to-End Inference (TSE + Joint Mixture Guidance)")
    parser.add_argument("--mixture", type=str, required=True, help="Path to input mixture wave file")
    parser.add_argument("--enrollment", type=str, required=True, help="Path to clean speaker enrollment wave file")
    parser.add_argument("--output", type=str, required=True, help="Path to save the final refined target speech")
    args = parser.parse_args()

    # Verify input files
    if not os.path.exists(args.mixture):
        raise FileNotFoundError(f"Mixture file not found: {args.mixture}")
    if not os.path.exists(args.enrollment):
        raise FileNotFoundError(f"Enrollment file not found: {args.enrollment}")

    # Check and download weights
    check_and_download_checkpoints()

    # Create temporary directory for intermediate SSD-buffered operations
    temp_dir = "./v1_temp_run"
    os.makedirs(temp_dir, exist_ok=True)
    
    local_mix = os.path.join(temp_dir, "mixture.wav")
    local_aux = os.path.join(temp_dir, "enrollment.wav")
    local_disc = os.path.join(temp_dir, "disc_out.wav")
    local_disc_tmp = local_disc + ".tmp.wav"
    
    # Copy files locally
    shutil.copy(args.mixture, local_mix)
    shutil.copy(args.enrollment, local_aux)

    # ── PHASE 1: DISCRIMINATIVE EXTRACTION ──
    print("\n[Phase 1/2] Running Discriminative Target Speech Extraction (TFGridNet)...")
    result = subprocess.run([
        "python", "usef_tse_code/run_inference.py",
        "--config", "usef_tse_code/config/config-USEF-TFGridNet.yaml",
        "--checkpoint", "usef_tse_code/chkpt/chkpt/USEF-TFGridNet/whamr!/temp_best.pth.tar",
        "--mixture", local_mix,
        "--aux", local_aux,
        "--output", local_disc_tmp
    ], capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Error in Phase 1: {result.stderr}")
        shutil.rmtree(temp_dir)
        sys.exit(1)
        
    os.rename(local_disc_tmp, local_disc)
    print("Phase 1 Complete! Output saved to buffer.")

    # ── PHASE 2: GENERATIVE REFINEMENT ──
    print("\n[Phase 2/2] Running Generative Refinement with Joint Mixture Guidance (ArrayDPS)...")
    
    # Setup paths for ArrayDPS imports
    sys.path.append(os.path.join(os.getcwd(), 'ArrayDPS'))
    import yaml
    from dotmap import DotMap
    from src.utils.setup import load_ema_weights
    from src.models.unet_wav import UNet1d
    from src.sde import VE_Sde_Elucidating

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running Phase 2 SDE solver on: {device}")

    # Load configuration
    config_path = "ArrayDPS/conf/conf_libritts_unet1d_attention_8k.yaml"
    checkpoint_path = "ArrayDPS/model_ckpt.pt"
    
    with open(config_path, 'r') as f:
        gen_args = yaml.safe_load(f)
    gen_args = DotMap(gen_args)

    # Force all diffusion parameters to float to bypass scientific notation string-parsing differences
    for key in gen_args.diffusion_parameters.keys():
        val = gen_args.diffusion_parameters[key]
        if val is not None and val != 'None':
            try:
                gen_args.diffusion_parameters[key] = float(val)
            except:
                pass
    if hasattr(gen_args.diffusion_parameters, 'sigma_data'):
        gen_args.diffusion_parameters.sigma_data = float(gen_args.diffusion_parameters.sigma_data)

    # Load models
    gen_model = UNet1d(gen_args.unet_wav, device).to(device)
    gen_model = load_ema_weights(gen_model, checkpoint_path)
    gen_model.eval()
    diff_parameters = VE_Sde_Elucidating(gen_args.diffusion_parameters, gen_args.diffusion_parameters.sigma_data)

    # Read audio files
    wav_y, sr = sf.read(local_disc)
    if len(wav_y.shape) > 1: wav_y = wav_y[:, 0]
    wav_mix, sr_mix = sf.read(local_mix)
    if len(wav_mix.shape) > 1: wav_mix = wav_mix[:, 0]
    
    # Check sampling rates
    if sr != 8000 or sr_mix != 8000:
        print("Warning: Input files are not 8kHz. Sampling rates must be 8kHz for the V1 model.")
    
    # Match lengths
    min_len = min(wav_y.shape[-1], wav_mix.shape[-1])
    wav_y, wav_mix = wav_y[:min_len], wav_mix[:min_len]
    sig_len = wav_y.shape[-1]
    
    # Pad to power of 2 for U-Net compatibility
    target_len = int(2**np.ceil(np.log2(sig_len)))
    y = torch.nn.functional.pad(torch.from_numpy(wav_y).float().unsqueeze(0).to(device), (0, target_len - sig_len))
    y_mix = torch.nn.functional.pad(torch.from_numpy(wav_mix).float().unsqueeze(0).to(device), (0, target_len - sig_len))
    
    # Calculate target speech energy mask
    y_np = y.squeeze().cpu().numpy()
    window_size = 800
    squared = y_np ** 2
    energy = np.convolve(squared, np.ones(window_size)/window_size, mode='same')
    threshold = 0.005 * np.max(energy)
    M = torch.from_numpy((energy > threshold).astype(np.float32)).unsqueeze(0).to(device)
    
    # Setup diffusion schedule steps
    t_steps = get_edm_schedule(100, gen_args.diffusion_parameters.sigma_min, gen_args.diffusion_parameters.sigma_max, gen_args.diffusion_parameters.ro).to(device)
    start_idx = (torch.abs(t_steps - 0.5)).argmin()
    t_steps = t_steps[start_idx:]
    
    # Refinement Loop
    x = y + t_steps[0] * torch.randn_like(y)
    
    with tqdm(total=len(t_steps)-1, desc="Diffusion Denoising steps") as pbar:
        for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
            x = x.detach().requires_grad_(True)
            t_tensor_cur = t_cur.unsqueeze(0)
            
            # Predict clean speech
            denoised = diff_parameters.denoiser(x, gen_model, t_tensor_cur)
            
            # Dual-Guidance loss function
            loss_clean = torch.linalg.norm(M * (denoised - y))
            loss_mix = torch.linalg.norm((1 - M) * torch.relu(denoised.abs() - y_mix.abs()))
            loss = loss_clean + loss_mix
            
            # Compute score gradients
            rec_grads = torch.autograd.grad(outputs=loss, inputs=x)[0]
            normguide = torch.linalg.norm(rec_grads) / (x.shape[-1] ** 0.5)
            s = 1.0 / (normguide * t_cur + 1e-6)
            
            # Solve score matching
            uncond_score = (denoised.detach() - x) / (t_cur ** 2)
            score = uncond_score - s * rec_grads
            d_cur = -t_cur * score
            x_next = x.detach() + (t_next - t_cur) * d_cur
            
            if i < len(t_steps) - 2:
                with torch.no_grad():
                    denoised_next = diff_parameters.denoiser(x_next, gen_model, t_next.unsqueeze(0))
                    d_prime = (x_next - denoised_next) / t_next
                    x_next = x.detach() + (t_next - t_cur) * (0.5 * d_cur.detach() + 0.5 * d_prime)
            x = x_next
            pbar.update(1)
            
    # Save final refined audio
    refined_wav = x.detach().squeeze(0).cpu().numpy()[:sig_len]
    sf.write(args.output, refined_wav, sr)
    print(f"Phase 2 Complete! Output saved to: {args.output}")

    # Cleanup temp buffer
    shutil.rmtree(temp_dir)
    print("\n🎉 SUCCESS! Target speech extracted and refined cleanly!")

if __name__ == "__main__":
    main()
