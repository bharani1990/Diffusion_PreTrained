import torch
import torch.nn.functional as F


def collate_fn(batch):
    max_time = max(mel.shape[-1] for mel in batch)
    padded = []
    for mel in batch:
        if mel.dim() == 2:
            mel = mel.unsqueeze(0)
        if mel.shape[-1] < max_time:
            pad = max_time - mel.shape[-1]
            mel = F.pad(mel, (0, pad), mode='constant', value=0)
        padded.append(mel)
    return torch.stack(padded, dim=0)
