import torch
import torch.nn as nn
from diffusers.models.unets.unet_2d import UNet2DModel
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler


class AudioCodec(nn.Module):
    def __init__(self, pretrained_model_id="google/ddpm-ema-celebahq-256"):
        super().__init__()
        self.unet = UNet2DModel.from_pretrained(pretrained_model_id)
        self.scheduler = DDPMScheduler.from_pretrained(pretrained_model_id)
        
        in_channels = self.unet.config["in_channels"]
        sample_size = self.unet.config["sample_size"]
        
        self.input_adapter = nn.Sequential(
            nn.Conv2d(1, in_channels, 3, padding=1),
            nn.SiLU()
        )
        
        out_channels = self.unet.config["out_channels"]
        self.output_adapter = nn.Sequential(
            nn.Conv2d(out_channels, 80, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(80, 80, 1)
        )
        
        print(f"Loaded pretrained UNet from {pretrained_model_id}")
        print(f"UNet expects: {in_channels} channels, {sample_size}x{sample_size} size")

    def forward(self, x, t):
        B, C, H, W = x.shape
        
        x_adapted = self.input_adapter(x)
        
        sample_size = self.unet.config["sample_size"]
        x_resized = torch.nn.functional.interpolate(x_adapted, size=(sample_size, sample_size), mode='bilinear', align_corners=False)
        
        noise_pred = self.unet(x_resized, t).sample
        
        mel = self.output_adapter(noise_pred)
        mel = torch.nn.functional.interpolate(mel, size=(H, W), mode='bilinear', align_corners=False)
        
        return mel, torch.tensor(0.0, device=x.device)
    
    @classmethod
    def from_pretrained(cls, pretrained_model_id):
        return cls(pretrained_model_id=pretrained_model_id)

