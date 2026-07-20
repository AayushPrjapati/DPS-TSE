import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
import shutil
import subprocess
import warnings
import torch
import torchaudio
import soundfile as sf
import numpy as np
from tqdm import tqdm

warnings.filterwarnings("ignore")

# Add paths for ArrayDPS imports
sys.path.append(os.path.join(os.getcwd(), 'ArrayDPS'))
import yaml
from dotmap import DotMap
from src.utils.setup import load_ema_weights
from src.models.unet_wav import UNet1d
from src.sde import VE_Sde_Elucidating

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

def run_generative_refinement(mix_path, disc_path, gen_model, diff_parameters, gen_args, out_path, device):
    # Read audio files using torchaudio and resample to 8000Hz
    wav_y, sr = torchaudio.load(disc_path)
    if sr != 8000:
        wav_y = torchaudio.transforms.Resample(sr, 8000)(wav_y)
    y_full = wav_y.mean(dim=0).numpy()
    
    wav_mix, sr_mix = torchaudio.load(mix_path)
    if sr_mix != 8000:
        wav_mix = torchaudio.transforms.Resample(sr_mix, 8000)(wav_mix)
    mix_full = wav_mix.mean(dim=0).numpy()
    
    # Match lengths
    min_len = min(len(y_full), len(mix_full))
    y_full = y_full[:min_len]
    mix_full = mix_full[:min_len]
    
    # Run in 65536-sample chunks (trained length, multiple of 8192) to match model downsampling depth
    chunk_size = 65536
    target_len = 65536
    refined_chunks = []
    
    # Setup diffusion schedule steps once
    t_steps = get_edm_schedule(100, gen_args.diffusion_parameters.sigma_min, gen_args.diffusion_parameters.sigma_max, gen_args.diffusion_parameters.ro).to(device)
    start_idx = (torch.abs(t_steps - 0.5)).argmin()
    t_steps = t_steps[start_idx:]
    
    num_chunks = int(np.ceil(min_len / chunk_size))
    
    for chunk_idx, start_idx in enumerate(range(0, min_len, chunk_size)):
        end_idx = min(start_idx + chunk_size, min_len)
        y_chunk = y_full[start_idx:end_idx]
        mix_chunk = mix_full[start_idx:end_idx]
        sig_len = len(y_chunk)
        
        # Pad chunk to exactly 65536 samples
        y = torch.nn.functional.pad(torch.from_numpy(y_chunk).float().unsqueeze(0).to(device), (0, target_len - sig_len))
        y_mix = torch.nn.functional.pad(torch.from_numpy(mix_chunk).float().unsqueeze(0).to(device), (0, target_len - sig_len))
        
        # Calculate target speech energy mask
        y_np = y.squeeze().cpu().numpy()
        window_size = min(800, sig_len // 2)
        window_size = max(10, window_size)
        squared = y_np ** 2
        energy = np.convolve(squared, np.ones(window_size)/window_size, mode='same')
        threshold = 0.005 * np.max(energy) if np.max(energy) > 1e-6 else 1e-6
        M = torch.from_numpy((energy > threshold).astype(np.float32)).unsqueeze(0).to(device)
        
        # Refinement Loop for this chunk
        x = y + t_steps[0] * torch.randn_like(y)
        
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
            
        # Save refined chunk
        refined_chunk = x.detach().squeeze(0).cpu().numpy()[:sig_len]
        refined_chunks.append(refined_chunk)
        print(f"  Processed chunk {chunk_idx+1}/{num_chunks}...")
        
    refined_full = np.concatenate(refined_chunks)
    sf.write(out_path, refined_full, 8000)
    print(f"Saved Generative: {out_path}")

def main():
    check_and_download_checkpoints()
    
    # Define the 5 cases
    cases = [
        {
            "id": "case1",
            "mix": "SITE_samples/LIBRI2MIX_mixture.wav",
            "cue": "SITE_samples/LIBRI2MIX_s1.wav"
        },
        {
            "id": "case2",
            "mix": "SITE_samples/S01_P01_01_mix_mix.wav",
            "cue": "SITE_samples/S01_P01_01_mix_enrollment.wav"
        },
        {
            "id": "case3",
            "mix": "SITE_samples/S21_P48_02_mix_mix.wav",
            "cue": "SITE_samples/S21_P48_02_mix_enrollment.wav"
        },
        {
            "id": "case4",
            "mix": "SITE_samples/M22_Hindi.wav",
            "cue": "SITE_samples/M22_enroll.wav"
        },
        {
            "id": "case5",
            "mix": "SITE_samples/S_Mixture_Trimmed (1).wav",
            "cue": "SITE_samples/S_M3 (1).wav"
        }
    ]
    
    os.makedirs("docs/audio", exist_ok=True)
    
    # Load ArrayDPS (Phase 2) config once to save overhead
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Loading ArrayDPS model on device: {device}...")
    config_path = "ArrayDPS/conf/conf_libritts_unet1d_attention_8k.yaml"
    checkpoint_path = "ArrayDPS/model_ckpt.pt"
    
    with open(config_path, 'r') as f:
        gen_args = yaml.safe_load(f)
    gen_args = DotMap(gen_args)
    
    for key in gen_args.diffusion_parameters.keys():
        val = gen_args.diffusion_parameters[key]
        if val is not None and val != 'None':
            try:
                gen_args.diffusion_parameters[key] = float(val)
            except:
                pass
    if hasattr(gen_args.diffusion_parameters, 'sigma_data'):
        gen_args.diffusion_parameters.sigma_data = float(gen_args.diffusion_parameters.sigma_data)
        
    gen_model = UNet1d(gen_args.unet_wav, device).to(device)
    gen_model = load_ema_weights(gen_model, checkpoint_path)
    gen_model.eval()
    diff_parameters = VE_Sde_Elucidating(gen_args.diffusion_parameters, gen_args.diffusion_parameters.sigma_data)
    
    print("\nStarting end-to-end run for all 5 cases...")
    for idx, c in enumerate(cases):
        print(f"\n==================== PROCESSING {c['id'].upper()} ({idx+1}/{len(cases)}) ====================")
        
        mix_out = f"docs/audio/{c['id']}_mix.wav"
        cue_out = f"docs/audio/{c['id']}_cue.wav"
        disc_out = f"docs/audio/{c['id']}_disc.wav"
        gen_out = f"docs/audio/{c['id']}_gen.wav"
        
        # Copy input and reference directly
        print(f"Copying input and cue files...")
        shutil.copy(c["mix"], mix_out)
        shutil.copy(c["cue"], cue_out)
        
        # ── PHASE 1: DISCRIMINATIVE EXTRACTION (USEF-TSE) ──
        print("Running Phase 1 (USEF-TSE Discriminative Extraction)...")
        disc_tmp = disc_out + ".tmp.wav"
        result = subprocess.run([
            "python", "usef_tse_code/run_inference.py",
            "--config", "usef_tse_code/config/config-USEF-TFGridNet.yaml",
            "--checkpoint", "usef_tse_code/chkpt/chkpt/USEF-TFGridNet/whamr!/temp_best.pth.tar",
            "--mixture", mix_out,
            "--aux", cue_out,
            "--output", disc_tmp
        ], capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"Error running Phase 1 on {c['id']}: {result.stderr}")
            continue
            
        os.replace(disc_tmp, disc_out)
        print(f"Saved Discriminative: {disc_out}")
        
        # ── PHASE 2: GENERATIVE REFINEMENT (ArrayDPS) ──
        print("Running Phase 2 (ArrayDPS Generative Refinement)...")
        try:
            run_generative_refinement(mix_out, disc_out, gen_model, diff_parameters, gen_args, gen_out, device)
        except Exception as e:
            print(f"Error running Phase 2 on {c['id']}: {e}")
            
    print("\nInference complete! Regenerating spectrograms...")
    try:
        # Run generate_spectrograms.py
        subprocess.run(["python", "generate_spectrograms.py"], check=True)
        print("Spectrograms regenerated successfully!")
    except Exception as e:
        print(f"Error regenerating spectrograms: {e}")

if __name__ == "__main__":
    main()
