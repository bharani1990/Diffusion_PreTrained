import sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset import SpectrogramDataset
from src.trainer import AudioCodecModule
from src.utils import collate_fn
from src import config as cfg


def main():
    torch.manual_seed(cfg.SEED)
    
    device = torch.device('cuda' if torch.cuda.is_available() and cfg.USE_GPU else 'cpu')
    print(f"Device: {device}")
    
    if torch.cuda.is_available() and cfg.USE_GPU:
        torch.set_float32_matmul_precision('medium')
    
    train_dataset = SpectrogramDataset(cfg.TRAIN_MANIFEST, target_frames=120)
    val_dataset = SpectrogramDataset(cfg.DEV_MANIFEST, target_frames=120)
    
    print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)}")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.TRAIN_BATCH_SIZE,
        shuffle=True,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=cfg.USE_GPU,
        collate_fn=collate_fn,
        persistent_workers=True if cfg.NUM_WORKERS > 0 else False,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.TRAIN_BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=cfg.USE_GPU,
        collate_fn=collate_fn,
        persistent_workers=True if cfg.NUM_WORKERS > 0 else False,
    )
    
    use_pretrained = getattr(cfg, 'USE_PRETRAINED', False)
    pretrained_id = getattr(cfg, 'PRETRAINED_MODEL_ID', None)
    
    model = AudioCodecModule(
        lr=cfg.LR,
        pretrained_model_id=cfg.PRETRAINED_MODEL_ID
    )
    
    cfg.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=cfg.CHECKPOINT_DIR,
        filename="codec-{epoch:02d}-{val_loss:.4f}",
        monitor=cfg.MONITOR_METRIC,
        mode=cfg.MONITOR_MODE,
        save_top_k=cfg.SAVE_TOP_K,
        save_last=True,
    )

    callbacks: list = [checkpoint_callback]

    accelerator = 'gpu' if torch.cuda.is_available() and cfg.USE_GPU else 'cpu'
    devices = 1 if accelerator == 'gpu' else 'auto'

    trainer = L.Trainer(
        max_epochs=cfg.TRAIN_EPOCHS,
        accelerator=accelerator,
        devices=devices,
        callbacks=callbacks,
        logger=CSVLogger(save_dir=str(cfg.CHECKPOINT_DIR), name="logs"),
        precision=cfg.PRECISION,
        gradient_clip_val=cfg.GRADIENT_CLIP_VAL,
        gradient_clip_algorithm=cfg.GRADIENT_CLIP_ALGORITHM,
        log_every_n_steps=cfg.LOG_INTERVAL,
    )
    
    print(f"Starting training for {cfg.TRAIN_EPOCHS} epochs")
    trainer.fit(model, train_loader, val_loader)
    
    print(f"Training complete. Checkpoints: {cfg.CHECKPOINT_DIR}")


if __name__ == "__main__":
    main()
