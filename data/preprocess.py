import torch
import torch.nn as nn
import torch.nn.functional as F


class SignalSTFT(nn.Module):
    def __init__(
        self,
        n_fft: int = 256,
        hop_length: int = 32,
        win_length: int = 256,
        power: float = 1.0,
        log: bool = True,
    ) -> None:
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.power = power
        self.log = log
        self.register_buffer("window", torch.hann_window(win_length), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.squeeze(1)
        spec = torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=True,
            return_complex=True,
            pad_mode="reflect",
        )
        mag = spec.abs() ** self.power
        if self.log:
            mag = torch.log1p(mag)
        return mag.unsqueeze(1)


def zscore_normalize(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    mu = x.mean(dim=dim, keepdim=True)
    sd = x.std(dim=dim, keepdim=True) + eps
    return (x - mu) / sd


def random_band_mask(
    spec: torch.Tensor,
    mask_ratio: float = 0.3,
    weight: torch.Tensor = None,
) -> torch.Tensor:
    B, C, F_, T_ = spec.shape
    n_mask = max(1, int(F_ * mask_ratio))
    out = spec.clone()
    for b in range(B):
        if weight is not None:
            p = weight[b].clamp_min(1e-6)
            p = p / p.sum()
            bins = torch.multinomial(p, n_mask, replacement=False)
        else:
            bins = torch.randperm(F_, device=spec.device)[:n_mask]
        out[b, :, bins, :] = 0.0
    return out


def random_time_dropout(
    spec: torch.Tensor,
    drop_ratio: float = 0.2,
    weight: torch.Tensor = None,
) -> torch.Tensor:
    B, C, F_, T_ = spec.shape
    n_mask = max(1, int(T_ * drop_ratio))
    out = spec.clone()
    for b in range(B):
        if weight is not None:
            p = weight[b].clamp_min(1e-6)
            p = p / p.sum()
            cols = torch.multinomial(p, n_mask, replacement=False)
        else:
            cols = torch.randperm(T_, device=spec.device)[:n_mask]
        out[b, :, :, cols] = 0.0
    return out


def random_spectrum_perturb(
    spec: torch.Tensor,
    std: float = 0.15,
    weight: torch.Tensor = None,
) -> torch.Tensor:
    noise = torch.randn_like(spec) * std
    if weight is not None:
        w = weight.to(spec.dtype)
        noise = noise * w
    return spec * (1.0 + noise)


def signal_time_mask(x: torch.Tensor, ratio: float = 0.1) -> torch.Tensor:
    B, C, L = x.shape
    n_mask = max(1, int(L * ratio))
    out = x.clone()
    for b in range(B):
        start = torch.randint(0, L - n_mask + 1, (1,), device=x.device).item()
        out[b, :, start : start + n_mask] = 0.0
    return out


def signal_jitter(x: torch.Tensor, std: float = 0.02) -> torch.Tensor:
    return x + torch.randn_like(x) * std
