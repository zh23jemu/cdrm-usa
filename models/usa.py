from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class _PatchEmbed2D(nn.Module):
    def __init__(self, in_channels: int = 1, embed_dim: int = 64, patch: Tuple[int, int] = (4, 4)) -> None:
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch, stride=patch)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor):
        x = self.proj(x)
        B, C, H, W = x.shape
        tokens = x.flatten(2).transpose(1, 2)
        tokens = self.norm(tokens)
        return tokens, (H, W)


class _SparseMHSA(nn.Module):
    def __init__(self, dim: int, num_heads: int = 4, topk_ratio: float = 0.5, attn_drop: float = 0.1, proj_drop: float = 0.1) -> None:
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, 3 * dim, bias=True)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.topk_ratio = float(topk_ratio)

    def forward(self, x: torch.Tensor, return_attn: bool = False):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn_logits = (q @ k.transpose(-2, -1)) * self.scale

        k_keep = max(1, int(N * self.topk_ratio))
        if k_keep < N:
            topk = attn_logits.topk(k_keep, dim=-1)
            mask = torch.full_like(attn_logits, float("-inf"))
            mask.scatter_(-1, topk.indices, topk.values)
            attn_logits = mask
        attn = attn_logits.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        out = self.proj_drop(out)
        if return_attn:
            return out, attn
        return out, None


class _SparseAttnBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 2.0, topk_ratio: float = 0.5, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = _SparseMHSA(dim, num_heads=num_heads, topk_ratio=topk_ratio, attn_drop=dropout, proj_drop=dropout)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, return_attn: bool = False):
        h, attn = self.attn(self.norm1(x), return_attn=return_attn)
        x = x + h
        x = x + self.mlp(self.norm2(x))
        return x, attn


class SparseAttentionEncoder(nn.Module):
    """
    Sparse-attention time-frequency encoder.

    Input  : log-magnitude STFT (B, 1, F, T)
    Output : token features (B, N, D) plus the last-layer attention
             (B, H, N, N) used to compute attention entropy per token.
    """

    def __init__(
        self,
        in_channels: int = 1,
        embed_dim: int = 64,
        depth: int = 2,
        num_heads: int = 4,
        topk_ratio: float = 0.5,
        patch: Tuple[int, int] = (4, 4),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.patch_embed = _PatchEmbed2D(in_channels, embed_dim, patch=patch)
        self.blocks = nn.ModuleList(
            [_SparseAttnBlock(embed_dim, num_heads=num_heads, topk_ratio=topk_ratio, dropout=dropout) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.out_dim = embed_dim
        self.patch = patch

    def forward(self, spec: torch.Tensor):
        tokens, (Hp, Wp) = self.patch_embed(spec)
        last_attn = None
        for i, blk in enumerate(self.blocks):
            return_attn = i == (len(self.blocks) - 1)
            tokens, attn = blk(tokens, return_attn=return_attn)
            if return_attn:
                last_attn = attn
        tokens = self.norm(tokens)
        return tokens, last_attn, (Hp, Wp)


class USA(nn.Module):
    """
    Uncertainty-aware Structural Augmentation module.

    Pipeline:
      1) Compute log-STFT spec from the 1D vibration signal.
      2) Encode with a sparse-attention encoder.
      3) From the last attention head, compute per-token entropy
         u_n = -sum_m a_{n,m} log a_{n,m}, then split it back to a
         (B, F', T') uncertainty map u(f, t).
      4) Use u to derive band weights (mean over T') and time weights
         (mean over F'); both are used as multinomial probabilities
         for adaptive structural perturbations:
            - frequency-band masking (band_mask_ratio)
            - time-slice dropout (time_drop_ratio)
            - local spectrum perturbation (spectrum_perturb_std)
         in high-uncertainty regions.

    The module returns: pooled features for the original spec, the
    perturbed spec, the per-sample uncertainty mean (for the
    consistency loss), and the patch attention entropy (sparsity loss).
    """

    def __init__(
        self,
        stft_module: nn.Module,
        tf_channels: int = 64,
        depth: int = 2,
        attn_heads: int = 4,
        attn_topk_ratio: float = 0.5,
        feat_dim: int = 256,
        patch: Tuple[int, int] = (4, 4),
        high_unc_quantile: float = 0.7,
        perturb_cfg: dict = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.stft = stft_module
        self.encoder = SparseAttentionEncoder(
            in_channels=1,
            embed_dim=tf_channels,
            depth=depth,
            num_heads=attn_heads,
            topk_ratio=attn_topk_ratio,
            patch=patch,
            dropout=dropout,
        )
        self.token_pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Linear(tf_channels, feat_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.out_dim = feat_dim
        self.high_unc_quantile = float(high_unc_quantile)
        cfg = dict(perturb_cfg or {})
        self.band_mask_ratio = float(cfg.get("band_mask_ratio", 0.3))
        self.time_drop_ratio = float(cfg.get("time_drop_ratio", 0.2))
        self.spectrum_perturb_std = float(cfg.get("spectrum_perturb_std", 0.15))

    def _entropy_from_attn(self, attn: torch.Tensor) -> torch.Tensor:
        if attn is None:
            return None
        a = attn.mean(dim=1)
        p = a.clamp_min(1e-8)
        ent = -(p * p.log()).sum(dim=-1)
        return ent

    def _apply_adaptive_perturb(
        self,
        spec: torch.Tensor,
        u_map: torch.Tensor,
    ) -> torch.Tensor:
        B, C, F_, T_ = spec.shape
        u_resized = F.interpolate(u_map.unsqueeze(1), size=(F_, T_), mode="bilinear", align_corners=False).squeeze(1)
        q = torch.quantile(u_resized.flatten(1), self.high_unc_quantile, dim=1).clamp_min(1e-8)
        mask_high = (u_resized >= q.view(-1, 1, 1)).float()
        m = mask_high.unsqueeze(1)

        f_weight = mask_high.mean(dim=-1).clamp_min(1e-6)
        t_weight = mask_high.mean(dim=-2).clamp_min(1e-6)

        n_band = max(1, int(F_ * self.band_mask_ratio))
        n_time = max(1, int(T_ * self.time_drop_ratio))

        band_indicator = torch.zeros(B, F_, device=spec.device)
        time_indicator = torch.zeros(B, T_, device=spec.device)
        for b in range(B):
            p_f = f_weight[b] / f_weight[b].sum()
            bands = torch.multinomial(p_f, n_band, replacement=False)
            band_indicator[b, bands] = 1.0
            p_t = t_weight[b] / t_weight[b].sum()
            cols = torch.multinomial(p_t, n_time, replacement=False)
            time_indicator[b, cols] = 1.0

        band_mask_pix = band_indicator.view(B, 1, F_, 1) * m
        time_mask_pix = time_indicator.view(B, 1, 1, T_) * m
        kill = (band_mask_pix + time_mask_pix).clamp(max=1.0)

        out = spec * (1.0 - kill)

        noise = torch.randn_like(spec) * self.spectrum_perturb_std
        out = out * (1.0 + noise * m)
        return out

    def forward(self, signal: torch.Tensor):
        spec = self.stft(signal)
        tokens, attn, (Hp, Wp) = self.encoder(spec)
        ent_tokens = self._entropy_from_attn(attn)
        if ent_tokens is None:
            ent_tokens = torch.zeros(spec.size(0), Hp * Wp, device=spec.device)
        u_map = ent_tokens.view(-1, Hp, Wp)
        feat = self.head(tokens.mean(dim=1))
        u_mean = u_map.flatten(1).mean(dim=1)
        sparsity = ent_tokens.mean()

        if self.training:
            perturbed_spec = self._apply_adaptive_perturb(spec, u_map)
            tokens_p, _, _ = self.encoder(perturbed_spec)
            feat_p = self.head(tokens_p.mean(dim=1))
        else:
            perturbed_spec = spec
            feat_p = feat

        return {
            "feat": feat,
            "feat_perturbed": feat_p,
            "spec": spec,
            "spec_perturbed": perturbed_spec,
            "u_map": u_map,
            "u_mean": u_mean,
            "attn_entropy": sparsity,
            "patch_grid": (Hp, Wp),
        }
