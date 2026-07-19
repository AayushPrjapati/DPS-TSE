import argparse
import subprocess
import os

def main():
    parser = argparse.ArgumentParser(description="Full TSE Pipeline: Discriminative Extraction + Generative Refinement")
    parser.add_argument("--mixture", type=str, required=True, help="Path to the input mixture audio file")
    parser.add_argument("--enrollment", type=str, required=True, help="Path to the enrollment (target speaker) audio file")
    parser.add_argument("--output", type=str, required=True, help="Path to save the final refined audio")
    
    args = parser.parse_args()

    intermediate_output = "temp_extracted.wav"

    print("="*50)
    print("PHASE 1: Discriminative Extraction (USEF-TSE)")
    print("="*50)
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Run Phase 1
    phase1_cmd = [
        "python", os.path.join(script_dir, "../../usef_tse_code/run_inference.py"),
        "--config", os.path.join(script_dir, "../../usef_tse_code/config/config-USEF-TFGridNet.yaml"),
        "--checkpoint", os.path.join(script_dir, "../../usef_tse_code/chkpt/chkpt/USEF-TFGridNet/whamr!/temp_best.pth.tar"),
        "--mixture", args.mixture,
        "--aux", args.enrollment,
        "--output", os.path.join(script_dir, intermediate_output)
    ]
    
    subprocess.run(phase1_cmd, check=True)
    
    print("\n" + "="*50)
    print("PHASE 2: Generative Refinement (ArrayDPS / Masked DPS)")
    print("="*50)
    
    # Since refine_arraydps.py is hardcoded in the __main__ block, we will import it and call the function directly.
    import refine_arraydps
    
    refine_arraydps.run_storm_refinement(
        input_path=os.path.join(script_dir, intermediate_output),
        output_path=args.output,
        config_path=os.path.join(script_dir, "../../ArrayDPS/conf/conf_libritts_unet1d_attention_8k.yaml"),
        checkpoint_path=os.path.join(script_dir, "../../ArrayDPS/model_ckpt.pt"),
        sigma_start=0.5,
        num_steps=100
    )
    
    # Clean up intermediate file
    temp_path = os.path.join(script_dir, intermediate_output)
    if os.path.exists(temp_path):
        os.remove(temp_path)
        
    print(f"\nSUCCESS! Full pipeline complete. Final audio saved to: {args.output}")

if __name__ == "__main__":
    main()
