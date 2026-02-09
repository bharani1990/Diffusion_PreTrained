import sys
from pathlib import Path
import torch
import torchaudio
import soundfile as sf
import numpy as np
import random

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.trainer import AudioCodecModule
from src import config as cfg


def mel_to_audio(mel, sr=16000, n_fft=1024, hop_length=256):
    if isinstance(mel, torch.Tensor):
        mel = mel.squeeze().cpu().numpy()
    
    mel_db = mel * 80.0 - 80.0
    mel_spec = 10 ** (mel_db / 20.0)
    
    mel_basis = torchaudio.transforms.InverseMelScale(
        n_stft=n_fft // 2 + 1,
        n_mels=80,
        sample_rate=sr,
        f_min=0,
        f_max=8000
    )
    
    stft = mel_basis(torch.from_numpy(mel_spec))
    
    audio = torchaudio.transforms.GriffinLim(
        n_fft=n_fft,
        hop_length=hop_length,
        n_iter=32
    )(stft)
    
    return audio.numpy()


def run_inference(checkpoint_path, test_file, output_path, original_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    print(f"Loading checkpoint: {checkpoint_path}")
    model = AudioCodecModule.load_from_checkpoint(checkpoint_path, map_location=device)
    model.eval()
    model.to(device)
    
    print(f"Loading test file: {test_file}")
    mel_original = torch.load(test_file).float()
    if mel_original.dim() == 2:
        mel_original = mel_original.unsqueeze(0)
    if mel_original.dim() == 3:
        mel_original = mel_original.unsqueeze(1)
    
    print(f"Input shape: {mel_original.shape}")
    
    print("Converting original to audio...")
    audio_original = mel_to_audio(mel_original[0, 0])
    sf.write(original_path, audio_original, cfg.SAMPLE_RATE)
    print(f"Original saved: {original_path}")
    
    mel = mel_original.to(device)
    
    print("Running reconstruction...")
    with torch.no_grad():
        t = torch.zeros(mel.size(0), dtype=torch.long, device=device)
        reconstructed, _ = model.model(mel, t)
    
    print(f"Output shape: {reconstructed.shape}")
    
    print("Converting reconstructed to audio...")
    audio_reconstructed = mel_to_audio(reconstructed[0, 0])
    
    sf.write(output_path, audio_reconstructed, cfg.SAMPLE_RATE)
    print(f"Reconstructed saved: {output_path}")
    print(f"Duration: {len(audio_reconstructed) / cfg.SAMPLE_RATE:.2f}s")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoints/last.ckpt")
    parser.add_argument("--test_file", type=str, default=None)
    parser.add_argument("--output", type=str, default="reconstructed.wav")
    parser.add_argument("--original", type=str, default="original.wav")
    args = parser.parse_args()
    
    if args.test_file is None:
        test_files = list(Path("data/processed_norm/test-clean").rglob("*.pt"))
        args.test_file = str(random.choice(test_files))
        print(f"Randomly selected: {args.test_file}")
    
    run_inference(args.checkpoint, args.test_file, args.output, args.original)
