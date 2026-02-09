import sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.dataset import SpectrogramDataset
from src.model import AudioCodec
from src import config as cfg

print("Testing pretrained model setup...")
print(f"Pretrained model: {cfg.PRETRAINED_MODEL_ID}")

print("\nLoading dataset...")
dataset = SpectrogramDataset(cfg.TRAIN_MANIFEST, target_frames=120)
print(f"Dataset size: {len(dataset)}")

print("\nCreating pretrained model...")
model = AudioCodec.from_pretrained(cfg.PRETRAINED_MODEL_ID)
print(f"Model created")

print("\nTesting forward pass...")
x = dataset[0].unsqueeze(0)
t = torch.randint(0, 1000, (1,))
print(f"Input shape: {x.shape}")

mel, _ = model(x, t)
print(f"Output shape: {mel.shape}")

print("\nAll tests passed!")
