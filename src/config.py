from pathlib import Path

SAMPLE_RATE = 16000
N_MELS = 80

LR = 1e-5
TRAIN_EPOCHS = 10
TRAIN_BATCH_SIZE = 2

DATA_DIR = Path("data")
TRAIN_MANIFEST = DATA_DIR / "processed" / "train_manifest.jsonl"
DEV_MANIFEST = DATA_DIR / "processed" / "dev_manifest.jsonl"
TEST_MANIFEST = DATA_DIR / "processed" / "test_manifest.jsonl"

CHECKPOINT_DIR = Path("checkpoints")
PRETRAINED_MODEL_ID = "google/ddpm-ema-celebahq-256"

USE_GPU = True
NUM_WORKERS = 2
LOG_INTERVAL = 50
SAVE_TOP_K = 2
MONITOR_METRIC = "val_loss"
MONITOR_MODE = "min"

PRECISION = "16-mixed"
GRADIENT_CLIP_VAL = 1.0
GRADIENT_CLIP_ALGORITHM = "norm"
SEED = 42
