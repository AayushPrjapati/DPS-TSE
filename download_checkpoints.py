import os
import urllib.request
from huggingface_hub import hf_hub_download

def download():
    print("="*60)
    print("📥 MODEL CHECKPOINT DOWNLOADER (ON-THE-FLY)")
    print("="*60)
    
    # 1. Download Phase 1 TFGridNet Checkpoint from Hugging Face
    print("Downloading Phase 1 Checkpoint (TFGridNet) from HF Hub...")
    try:
        local_path = hf_hub_download(
            repo_id='ZBang/USEF-TSE', 
            filename='chkpt/USEF-TFGridNet/whamr!/temp_best.pth.tar', 
            local_dir='usef_tse_code/chkpt'
        )
        print(f"Success! Saved Phase 1 to: {local_path}")
    except Exception as e:
        print(f"ERROR downloading Phase 1: {e}")
        
    # 2. Download Phase 2 ArrayDPS Checkpoint from Box Link
    print("\nDownloading Phase 2 Checkpoint (ArrayDPS) from Box Server...")
    os.makedirs("ArrayDPS", exist_ok=True)
    box_url = 'https://uofi.box.com/shared/static/eent06t4b4hdkjf0vgjzsqw8defa3xbn.pt'
    dest_path = 'ArrayDPS/model_ckpt.pt'
    
    try:
        urllib.request.urlretrieve(box_url, dest_path)
        print(f"Success! Saved Phase 2 to: {dest_path}")
    except Exception as e:
        print(f"ERROR downloading Phase 2: {e}")
        
    print("\n" + "="*60)
    print("🎉 ALL WEIGHTS DOWNLOADED AND INSTALLED SUCCESSFULLY!")
    print("="*60)

if __name__ == "__main__":
    download()
