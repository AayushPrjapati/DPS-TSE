import argparse
import subprocess
import os

def main():
    parser = argparse.ArgumentParser(description="Stage 1: Discriminative Target Speech Extraction only (TFGridNet)")
    parser.add_argument("--mixture", type=str, required=True, help="Path to the input mixture audio file")
    parser.add_argument("--enrollment", type=str, required=True, help="Path to the enrollment (target speaker) audio file")
    parser.add_argument("--output", type=str, required=True, help="Path to save the extracted audio")
    
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    print("="*60)
    print("STAGE 1: Discriminative Target Speech Extraction Only (TFGridNet)")
    print("="*60)
    
    cmd = [
        "python", os.path.join(script_dir, "../../usef_tse_code/run_inference.py"),
        "--config", os.path.join(script_dir, "../../usef_tse_code/config/config-USEF-TFGridNet.yaml"),
        "--checkpoint", os.path.join(script_dir, "../../usef_tse_code/chkpt/chkpt/USEF-TFGridNet/whamr!/temp_best.pth.tar"),
        "--mixture", args.mixture,
        "--aux", args.enrollment,
        "--output", args.output
    ]
    
    subprocess.run(cmd, check=True)
    print(f"\nSUCCESS! Discriminative target speech extraction complete. Saved to: {args.output}")

if __name__ == "__main__":
    main()
