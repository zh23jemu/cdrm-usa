from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbones import build_backbone
from .cdrm import CDRM
from .usa import USA
from data.preprocess import SignalSTFT
from utils.losses import consistency_loss


class CDRM_USA_Model(nn.Module):
    """
    Full model that wires together:
      - 1D backbone for raw signal (time-domain stream)
      - USA module for TF-domain stream with uncertainty-aware augmentation
      - CDRM module that disentangles fused features into z_f and z_c

    Two streams are fused with concatenation followed by a linear gate,
    producing a single feature vector h passed to CDRM. The classification
    head of CDRM operates on z_f and is used at inference.
    """

    def __init__(
        self,
        num_classes: int,
        num_conditions: int,
        feat_dim: int = 256,
        backbone: str = "resnet1d",
        stft_cfg: Optional[dict] = None,
        usa_cfg: Optional[dict] = None,
        cdrm_cfg: Optional[dict] = None,
        backbone_kwargs: Optional[dict] = None,
    ) -> None:
        super().__init__()
        backbone_kwargs = backbone_kwargs or {}
        self.backbone = build_backbone(backbone, in_channels=1, feat_dim=feat_dim, **backbone_kwargs)

        stft_cfg = stft_cfg or {}
        self.stft = SignalSTFT(
            n_fft=stft_cfg.get("n_fft", 256),
            hop_length=stft_cfg.get("hop_length", 32),
            win_length=stft_cfg.get("win_length", 256),
            power=stft_cfg.get("power", 1.0),
            log=stft_cfg.get("log", True),
        )

        usa_cfg = usa_cfg or {}
        self.usa = USA(
            stft_module=self.stft,
            tf_channels=usa_cfg.get("tf_channels", 64),
            depth=usa_cfg.get("depth", 2),
            attn_heads=usa_cfg.get("attn_heads", 4),
            attn_topk_ratio=usa_cfg.get("attn_topk_ratio", 0.5),
            feat_dim=feat_dim,
            patch=tuple(usa_cfg.get("patch", (4, 4))),
            high_unc_quantile=usa_cfg.get("high_unc_quantile", 0.7),
            perturb_cfg=usa_cfg.get("perturb", {}),
        )

        self.fuse = nn.Sequential(
            nn.Linear(feat_dim * 2, feat_dim),
            nn.LayerNorm(feat_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )

        cdrm_cfg = cdrm_cfg or {}
        self.cdrm = CDRM(
            feat_dim=feat_dim,
            proj_dim=cdrm_cfg.get("proj_dim", 128),
            num_classes=num_classes,
            num_conditions=num_conditions,
            lambda_grl=cdrm_cfg.get("lambda_grl", 1.0),
            lambda_cond=cdrm_cfg.get("lambda_cond", 0.5),
        )

        self.lambda_consistency = float(usa_cfg.get("lambda_consistency", 1.0))
        self.lambda_sparsity = float(usa_cfg.get("lambda_sparsity", 0.01))
        self.lambda_grl = float(cdrm_cfg.get("lambda_grl", 1.0))
        self.lambda_cond = float(cdrm_cfg.get("lambda_cond", 0.5))
        self.lambda_cond_adv = float(cdrm_cfg.get("lambda_cond_adv", 1.0))
        self.lambda_ortho = float(cdrm_cfg.get("lambda_ortho", 0.1))

    def set_grl_lambda(self, lambd: float) -> None:
        self.cdrm.set_lambda(lambd)

    def encode(self, x: torch.Tensor):
        z_time = self.backbone(x)
        usa_out = self.usa(x)
        z_tf = usa_out["feat"]
        z_tf_p = usa_out["feat_perturbed"]
        h = self.fuse(torch.cat([z_time, z_tf], dim=-1))
        h_p = self.fuse(torch.cat([z_time, z_tf_p], dim=-1))
        return h, h_p, usa_out

    def forward(self, x: torch.Tensor):
        h, h_p, usa_out = self.encode(x)
        out = self.cdrm(h)
        out_p = self.cdrm(h_p)
        out["logits_cls_perturbed"] = out_p["logits_cls"]
        out["u_mean"] = usa_out["u_mean"]
        out["attn_entropy"] = usa_out["attn_entropy"]
        return out

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        h, _, _ = self.encode(x)
        return self.cdrm.fault_classifier(self.cdrm.proj_f(h))

    def compute_total_loss(
        self,
        out: dict,
        y: torch.Tensor,
        cond: torch.Tensor,
    ):
        losses = self.cdrm.compute_losses(out, y, cond)

        unc = out["u_mean"]
        unc_norm = (unc - unc.min().detach()) / (unc.max().detach() - unc.min().detach() + 1e-6)
        weight = 0.5 + unc_norm.detach()
        cons = consistency_loss(
            out["logits_cls_perturbed"],
            out["logits_cls"],
            weight=weight,
            mode="kl",
        )
        sparsity = out["attn_entropy"]

        total = (
            losses["loss_cls"]
            + self.lambda_cond_adv * losses["loss_cond_adv"]
            + self.lambda_cond * losses["loss_cond"]
            + self.lambda_ortho * losses["loss_ortho"]
            + self.lambda_consistency * cons
            + self.lambda_sparsity * sparsity
        )
        losses.update({
            "loss_consistency": cons,
            "loss_sparsity": sparsity,
            "loss_total": total,
        })
        return total, losses
