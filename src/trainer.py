import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from pathlib import Path
from src.model import AudioCodec
from src import config as cfg


class PerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, target, recon):
        if target.dim() == 4 and recon.dim() == 4 and target.size(1) != recon.size(1):
            if target.size(1) > recon.size(1):
                target = target.mean(dim=1, keepdim=True)
            else:
                recon = recon.mean(dim=1, keepdim=True)
        loss = F.l1_loss(target, recon)
        if target.size(1) > 1:
            for scale in [1, 2, 4]:
                if scale > 1:
                    t_down = F.avg_pool2d(target, kernel_size=scale, stride=scale)
                    r_down = F.avg_pool2d(recon, kernel_size=scale, stride=scale)
                else:
                    t_down = target
                    r_down = recon
                loss = loss + 0.1 * F.l1_loss(t_down, r_down)
        return loss


class AudioCodecModule(L.LightningModule):
    def __init__(self, lr=1e-5, pretrained_model_id="google/ddpm-ema-celebahq-256"):
        super().__init__()
        self.save_hyperparameters()
        
        self.model = AudioCodec.from_pretrained(pretrained_model_id)
        self.lr = lr
        self.recon_fn = nn.L1Loss()
        self.perc_fn = PerceptualLoss()

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.model.parameters(), lr=self.lr, betas=(0.9, 0.999), weight_decay=1e-2)
        max_epochs = self.trainer.max_epochs if self.trainer.max_epochs is not None else 1
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, max_epochs))
        return [opt], [sched]

    def training_step(self, batch, batch_idx):
        x = batch
        t = torch.randint(0, 1000, (x.size(0),), device=x.device)
        mel, _ = self.model(x, t)

        if mel.dim() == 4 and x.dim() == 4 and mel.size(1) != x.size(1):
            mel = mel.mean(dim=1, keepdim=True)

        recon_loss = self.recon_fn(mel, x)
        perc_loss = self.perc_fn(x, mel)
        total_loss = recon_loss + 0.5 * perc_loss

        self.log("train_loss", total_loss, prog_bar=True)
        return total_loss

    def validation_step(self, batch, batch_idx):
        x = batch
        t = torch.randint(0, 1000, (x.size(0),), device=x.device)
        mel, _ = self.model(x, t)

        if mel.dim() == 4 and x.dim() == 4 and mel.size(1) != x.size(1):
            mel = mel.mean(dim=1, keepdim=True)

        recon_loss = self.recon_fn(mel, x)
        perc_loss = self.perc_fn(x, mel)
        total_loss = recon_loss + 0.5 * perc_loss

        self.log("val_loss", total_loss, prog_bar=True)
        return total_loss

    def on_train_epoch_start(self):
        epoch = int(self.current_epoch) + 1
        max_epochs = self.trainer.max_epochs if self.trainer.max_epochs is not None else 1
        print(f"Epoch {epoch}/{max_epochs}")
