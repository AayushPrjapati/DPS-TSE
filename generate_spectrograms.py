import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
import torchaudio
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

def generate_spectrogram(audio_path, output_image_path):
    print(f"Generating spectrogram for: {audio_path}")
    try:
        # Load audio using torchaudio
        wav, sr = torchaudio.load(audio_path)
        
        # Convert to mono numpy array
        y = wav.mean(dim=0).numpy()
        
        # Create figure
        fig, ax = plt.subplots(figsize=(4.5, 1.8), dpi=150)
        
        # Plot spectrogram
        # NFFT=512, noverlap=384 (hop_length=128)
        ax.specgram(y, NFFT=512, Fs=sr, noverlap=384, cmap='magma', scale='dB')
        
        # Clean up axes for a formal scientific look
        ax.set_xlabel('Time (s)', fontsize=8)
        ax.set_ylabel('Freq (Hz)', fontsize=8)
        ax.tick_params(axis='both', which='major', labelsize=7)
        
        # Limit frequency display
        if sr <= 8000:
            ax.set_ylim(0, 4000)
        else:
            ax.set_ylim(0, min(8000, sr // 2))
            
        plt.tight_layout()
        plt.savefig(output_image_path, bbox_inches='tight', pad_inches=0.05)
        plt.close(fig)
        print(f"Saved: {output_image_path}")
    except Exception as e:
        print(f"Error generating spectrogram for {audio_path}: {e}")

def main():
    audio_dir = 'docs/audio'
    image_dir = 'docs/images'
    os.makedirs(image_dir, exist_ok=True)
    
    # Process all wav files
    for file in os.listdir(audio_dir):
        if file.endswith('.wav'):
            audio_path = os.path.join(audio_dir, file)
            # Create a corresponding png filename
            # Note: handle cases like .wav.wav nicely
            clean_name = file
            if clean_name.endswith('.wav.wav'):
                clean_name = clean_name[:-4]
            image_name = clean_name.replace('.wav', '.png')
            output_image_path = os.path.join(image_dir, image_name)
            generate_spectrogram(audio_path, output_image_path)

if __name__ == "__main__":
    main()
