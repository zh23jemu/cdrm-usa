from typing import Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbones import build_backbone, WDCNN
from utils.losses import grad_reverse


class _BaseClassifier(nn.Module):
    def __init__(self, backbone: str, num_classes: int, feat_dim: int = 256, dropout: float = 0.1, backbone_kwargs: Optional[dict] = None) -> None:
        super().__init__()
        backbone_kwargs = backbone_kwargs or {}
        self.backbone = build_backbone(backbone, in_channels=1, feat_dim=feat_dim, **backbone_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(feat_dim, num_classes)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def forward(self, x: torch.Tensor):
        h = self.encode(x)
        return self.classifier(self.dropout(h))


class ERMModel(_BaseClassifier):
    def __init__(self, backbone: str, num_classes: int, feat_dim: int = 256, **kw):
        super().__init__(backbone, num_classes, feat_dim, **kw)

    def compute_loss(self, x, y, **kwargs):
        logits = self(x)
        return F.cross_entropy(logits, y), {"loss_cls": F.cross_entropy(logits, y).detach()}


class DANNModel(nn.Module):
    """
    Domain-Adversarial Neural Network adapted for single-source DG by using
    pseudo-condition labels as surrogate domain labels.
    """

    def __init__(self, backbone: str, num_classes: int, num_conditions: int, feat_dim: int = 256, dropout: float = 0.1, backbone_kwargs: Optional[dict] = None) -> None:
        super().__init__()
        backbone_kwargs = backbone_kwargs or {}
        self.backbone = build_backbone(backbone, in_channels=1, feat_dim=feat_dim, **backbone_kwargs)
        self.classifier = nn.Linear(feat_dim, num_classes)
        self.dom_classifier = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feat_dim, num_conditions),
        )
        self.lambd = 1.0

    def set_lambda(self, lambd: float) -> None:
        self.lambd = float(lambd)

    def forward(self, x: torch.Tensor):
        h = self.backbone(x)
        cls_logits = self.classifier(h)
        dom_logits = self.dom_classifier(grad_reverse(h, self.lambd))
        return {"logits": cls_logits, "dom_logits": dom_logits, "feat": h}

    def compute_loss(self, x, y, cond, **kwargs):
        out = self(x)
        loss_cls = F.cross_entropy(out["logits"], y)
        loss_dom = F.cross_entropy(out["dom_logits"], cond)
        total = loss_cls + loss_dom
        return total, {"loss_cls": loss_cls.detach(), "loss_dom": loss_dom.detach()}


class _MixStyle1D(nn.Module):
    def __init__(self, p: float = 0.5, alpha: float = 0.1, eps: float = 1e-6) -> None:
        super().__init__()
        self.p = p
        self.alpha = alpha
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or torch.rand(1).item() > self.p:
            return x
        B = x.size(0)
        mu = x.mean(dim=-1, keepdim=True)
        sd = x.std(dim=-1, keepdim=True) + self.eps
        x_norm = (x - mu) / sd
        lmda = torch.distributions.Beta(self.alpha, self.alpha).sample((B, 1, 1)).to(x.device)
        perm = torch.randperm(B, device=x.device)
        mu_mix = mu * lmda + mu[perm] * (1.0 - lmda)
        sd_mix = sd * lmda + sd[perm] * (1.0 - lmda)
        return x_norm * sd_mix + mu_mix


class _DSU1D(nn.Module):
    """
    DSU (Uncertainty Modeling for OOD Generalization, ICLR 2022).
    Perturbs per-channel feature statistics with Gaussian noise whose
    scale is the batch-statistics standard deviation.
    """

    def __init__(self, p: float = 0.5, eps: float = 1e-6) -> None:
        super().__init__()
        self.p = p
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or torch.rand(1).item() > self.p:
            return x
        mu = x.mean(dim=-1, keepdim=True)
        sd = x.std(dim=-1, keepdim=True) + self.eps
        sigma_mu = mu.std(dim=0, keepdim=True) + self.eps
        sigma_sd = sd.std(dim=0, keepdim=True) + self.eps
        beta = mu + torch.randn_like(mu) * sigma_mu
        gamma = sd + torch.randn_like(sd) * sigma_sd
        x_norm = (x - mu) / sd
        return x_norm * gamma + beta


class _EFDMix1D(nn.Module):
    """
    Exact Feature Distribution Mixing (CVPR 2022). Operates on per-channel
    features over the temporal axis: sorts both samples, mixes by reordering
    to match the partner's sort order at ratio lambda.
    """

    def __init__(self, p: float = 0.5, alpha: float = 0.1) -> None:
        super().__init__()
        self.p = p
        self.alpha = alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or torch.rand(1).item() > self.p:
            return x
        B, C, L = x.shape
        perm = torch.randperm(B, device=x.device)
        sort_idx = x.argsort(dim=-1)
        inverse_idx = sort_idx.argsort(dim=-1)
        x_sorted = torch.gather(x, -1, sort_idx)
        x_partner_sorted = torch.gather(x[perm], -1, x[perm].argsort(dim=-1))
        lmda = torch.distributions.Beta(self.alpha, self.alpha).sample((B, 1, 1)).to(x.device)
        mixed_sorted = lmda * x_sorted + (1.0 - lmda) * x_partner_sorted
        return torch.gather(mixed_sorted, -1, inverse_idx)


class _StyleModulatedBackbone(nn.Module):
    """
    Wraps a ResNet1D-style backbone and injects a feature-statistics
    augmentor after the stem. Used by MixStyle, DSU, and EFDMix.
    """

    def __init__(self, backbone: nn.Module, augmentor: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone
        self.augmentor = augmentor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(self.backbone, "stem") and hasattr(self.backbone, "stages"):
            h = self.backbone.stem(x)
            h = self.augmentor(h)
            for stage in self.backbone.stages:
                h = stage(h)
            z = self.backbone.pool(h).squeeze(-1)
            return self.backbone.proj(z)
        return self.backbone(x)


class MixStyleModel(nn.Module):
    """
    MixStyle (Zhou et al. ICLR 2021) adapted for 1D signals. Style is
    computed per-channel statistics (mean, std) along the temporal axis
    after the backbone stem, then mixed across batch samples.
    """

    def __init__(self, backbone: str, num_classes: int, feat_dim: int = 256, p: float = 0.5, alpha: float = 0.1, backbone_kwargs: Optional[dict] = None) -> None:
        super().__init__()
        backbone_kwargs = backbone_kwargs or {}
        bb = build_backbone(backbone, in_channels=1, feat_dim=feat_dim, **backbone_kwargs)
        self.backbone = _StyleModulatedBackbone(bb, _MixStyle1D(p=p, alpha=alpha))
        self.classifier = nn.Linear(feat_dim, num_classes)

    def forward(self, x: torch.Tensor):
        return self.classifier(self.backbone(x))

    def compute_loss(self, x, y, **kwargs):
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        return loss, {"loss_cls": loss.detach()}


class DSUModel(nn.Module):
    """
    Domain Statistics Uncertainty (Li et al. ICLR 2022).
    """

    def __init__(self, backbone: str, num_classes: int, feat_dim: int = 256, p: float = 0.5, backbone_kwargs: Optional[dict] = None) -> None:
        super().__init__()
        backbone_kwargs = backbone_kwargs or {}
        bb = build_backbone(backbone, in_channels=1, feat_dim=feat_dim, **backbone_kwargs)
        self.backbone = _StyleModulatedBackbone(bb, _DSU1D(p=p))
        self.classifier = nn.Linear(feat_dim, num_classes)

    def forward(self, x: torch.Tensor):
        return self.classifier(self.backbone(x))

    def compute_loss(self, x, y, **kwargs):
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        return loss, {"loss_cls": loss.detach()}


class EFDMixModel(nn.Module):
    """
    Exact Feature Distribution Mixing (Zhang et al. CVPR 2022) for 1D signals.
    """

    def __init__(self, backbone: str, num_classes: int, feat_dim: int = 256, p: float = 0.5, alpha: float = 0.1, backbone_kwargs: Optional[dict] = None) -> None:
        super().__init__()
        backbone_kwargs = backbone_kwargs or {}
        bb = build_backbone(backbone, in_channels=1, feat_dim=feat_dim, **backbone_kwargs)
        self.backbone = _StyleModulatedBackbone(bb, _EFDMix1D(p=p, alpha=alpha))
        self.classifier = nn.Linear(feat_dim, num_classes)

    def forward(self, x: torch.Tensor):
        return self.classifier(self.backbone(x))

    def compute_loss(self, x, y, **kwargs):
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        return loss, {"loss_cls": loss.detach()}


class RSCModel(nn.Module):
    """
    Representation Self-Challenging (Huang et al. ECCV 2020).
    """

    def __init__(self, backbone: str, num_classes: int, feat_dim: int = 256, drop_f: float = 0.33, drop_b: float = 0.33, backbone_kwargs: Optional[dict] = None) -> None:
        super().__init__()
        backbone_kwargs = backbone_kwargs or {}
        self.backbone = build_backbone(backbone, in_channels=1, feat_dim=feat_dim, **backbone_kwargs)
        self.classifier = nn.Linear(feat_dim, num_classes)
        self.drop_f = float(drop_f)
        self.drop_b = float(drop_b)

    def forward(self, x: torch.Tensor):
        h = self.backbone(x)
        return self.classifier(h), h

    def compute_loss(self, x, y, **kwargs):
        logits, h = self(x)
        if not self.training:
            return F.cross_entropy(logits, y), {"loss_cls": F.cross_entropy(logits, y).detach()}
        h = h.clone().detach().requires_grad_(True)
        logits_h = self.classifier(h)
        one_hot = F.one_hot(y, num_classes=logits_h.size(1)).float()
        g = torch.autograd.grad((logits_h * one_hot).sum(), h, retain_graph=False)[0]
        feat_thresh = torch.quantile(g.abs(), 1.0 - self.drop_f, dim=1, keepdim=True)
        mask_f = (g.abs() < feat_thresh).float()

        h_orig = self.backbone(x)
        masked = h_orig * mask_f
        logits_masked = self.classifier(masked)
        loss_per_sample = F.cross_entropy(logits_masked, y, reduction="none")
        sample_thresh = torch.quantile(loss_per_sample, 1.0 - self.drop_b)
        sample_mask = (loss_per_sample >= sample_thresh).float()
        if sample_mask.sum() < 1:
            sample_mask = torch.ones_like(sample_mask)
        loss_rsc = (loss_per_sample * sample_mask).sum() / sample_mask.sum().clamp_min(1.0)

        loss_clean = F.cross_entropy(self.classifier(h_orig), y)
        loss = 0.5 * (loss_clean + loss_rsc)
        return loss, {"loss_cls": loss_clean.detach(), "loss_rsc": loss_rsc.detach()}


class WDCNNModel(nn.Module):
    def __init__(self, num_classes: int, feat_dim: int = 256, **kw) -> None:
        super().__init__()
        self.backbone = WDCNN(in_channels=1, feat_dim=feat_dim)
        self.classifier = nn.Linear(feat_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.backbone(x))

    def compute_loss(self, x, y, **kwargs):
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        return loss, {"loss_cls": loss.detach()}


class PCLModel(nn.Module):
    """
    Proxy-based Contrastive Learning (CVPR 2022) adapted for single-source DG.
    Class proxies are learned, and a contrastive loss pulls features toward
    their class proxy while pushing them away from other class proxies.
    """

    def __init__(self, backbone: str, num_classes: int, feat_dim: int = 256, proj_dim: int = 128, temperature: float = 0.1, backbone_kwargs: Optional[dict] = None) -> None:
        super().__init__()
        backbone_kwargs = backbone_kwargs or {}
        self.backbone = build_backbone(backbone, in_channels=1, feat_dim=feat_dim, **backbone_kwargs)
        self.classifier = nn.Linear(feat_dim, num_classes)
        self.proj = nn.Sequential(
            nn.Linear(feat_dim, proj_dim),
            nn.GELU(),
            nn.Linear(proj_dim, proj_dim),
        )
        self.proxies = nn.Parameter(torch.randn(num_classes, proj_dim) * 0.02)
        self.temperature = float(temperature)

    def forward(self, x: torch.Tensor):
        h = self.backbone(x)
        return self.classifier(h), self.proj(h)

    def compute_loss(self, x, y, **kwargs):
        logits, z = self(x)
        z = F.normalize(z, dim=-1)
        p = F.normalize(self.proxies, dim=-1)
        sim = z @ p.t() / self.temperature
        loss_proxy = F.cross_entropy(sim, y)
        loss_cls = F.cross_entropy(logits, y)
        total = loss_cls + 0.5 * loss_proxy
        return total, {"loss_cls": loss_cls.detach(), "loss_proxy": loss_proxy.detach()}


class IRMTrainer(nn.Module):
    """
    Invariant Risk Minimization (Arjovsky 2019).
    Uses pseudo-condition labels as surrogate environments inside the source.
    """

    def __init__(self, backbone: str, num_classes: int, num_conditions: int, feat_dim: int = 256, penalty: float = 1.0, backbone_kwargs: Optional[dict] = None) -> None:
        super().__init__()
        backbone_kwargs = backbone_kwargs or {}
        self.backbone = build_backbone(backbone, in_channels=1, feat_dim=feat_dim, **backbone_kwargs)
        self.classifier = nn.Linear(feat_dim, num_classes)
        self.num_conditions = num_conditions
        self.penalty = float(penalty)

    def forward(self, x: torch.Tensor):
        return self.classifier(self.backbone(x))

    @staticmethod
    def _irm_penalty(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        scale = torch.tensor(1.0, requires_grad=True, device=logits.device)
        loss = F.cross_entropy(logits * scale, y)
        grad = torch.autograd.grad(loss, [scale], create_graph=True)[0]
        return grad.pow(2).mean()

    def compute_loss(self, x, y, cond, **kwargs):
        logits = self(x)
        loss_total = 0.0
        penalty_total = 0.0
        n_env = 0
        for e in range(self.num_conditions):
            mask = (cond == e)
            if mask.sum() < 2:
                continue
            le = logits[mask]
            ye = y[mask]
            loss_e = F.cross_entropy(le, ye)
            pen_e = self._irm_penalty(le, ye)
            loss_total = loss_total + loss_e
            penalty_total = penalty_total + pen_e
            n_env += 1
        if n_env == 0:
            loss_total = F.cross_entropy(logits, y)
        else:
            loss_total = loss_total / n_env
            penalty_total = penalty_total / n_env
        total = loss_total + self.penalty * penalty_total
        if isinstance(penalty_total, torch.Tensor):
            return total, {"loss_cls": loss_total.detach(), "loss_irm": penalty_total.detach()}
        return total, {"loss_cls": loss_total.detach()}


class FishrTrainer(nn.Module):
    """
    Fishr (Rame et al. ICML 2022) variance penalty of per-environment
    classifier gradients; adapted to single-source via pseudo-conditions.
    """

    def __init__(self, backbone: str, num_classes: int, num_conditions: int, feat_dim: int = 256, penalty: float = 1000.0, backbone_kwargs: Optional[dict] = None) -> None:
        super().__init__()
        backbone_kwargs = backbone_kwargs or {}
        self.backbone = build_backbone(backbone, in_channels=1, feat_dim=feat_dim, **backbone_kwargs)
        self.classifier = nn.Linear(feat_dim, num_classes)
        self.num_conditions = num_conditions
        self.penalty = float(penalty)

    def forward(self, x: torch.Tensor):
        h = self.backbone(x)
        return self.classifier(h), h

    def compute_loss(self, x, y, cond, **kwargs):
        logits, h = self(x)
        loss_cls = F.cross_entropy(logits, y)
        grads_per_env: List[torch.Tensor] = []
        n_env = 0
        for e in range(self.num_conditions):
            mask = (cond == e)
            if mask.sum() < 2:
                continue
            he = h[mask].detach().clone().requires_grad_(True)
            le = self.classifier(he)
            ye = y[mask]
            loss_e = F.cross_entropy(le, ye)
            grad_w = torch.autograd.grad(loss_e, self.classifier.weight, create_graph=True, retain_graph=True)[0]
            grads_per_env.append(grad_w.reshape(-1))
            n_env += 1
        if n_env < 2:
            return loss_cls, {"loss_cls": loss_cls.detach()}
        g = torch.stack(grads_per_env, dim=0)
        mean_g = g.mean(dim=0, keepdim=True)
        var_g = (g - mean_g).pow(2).mean()
        total = loss_cls + self.penalty * var_g
        return total, {"loss_cls": loss_cls.detach(), "loss_fishr": var_g.detach()}


class SAGMTrainer(nn.Module):
    """
    Sharpness-Aware Gradient Matching (Wang et al. CVPR 2023).

    Implementation uses torch.func.functional_call so that the perturbed
    forward pass does not modify the parameter buffers in place, keeping
    the outer .backward() in the training loop safe.
    """

    def __init__(self, backbone: str, num_classes: int, feat_dim: int = 256, rho: float = 0.05, alpha: float = 0.0005, backbone_kwargs: Optional[dict] = None) -> None:
        super().__init__()
        backbone_kwargs = backbone_kwargs or {}
        self.backbone = build_backbone(backbone, in_channels=1, feat_dim=feat_dim, **backbone_kwargs)
        self.classifier = nn.Linear(feat_dim, num_classes)
        self.rho = float(rho)
        self.alpha = float(alpha)

    def forward(self, x: torch.Tensor):
        return self.classifier(self.backbone(x))

    def compute_loss(self, x, y, **kwargs):
        named_params = {n: p for n, p in self.named_parameters() if p.requires_grad}
        param_names = list(named_params.keys())
        params = [named_params[n] for n in param_names]

        logits = self(x)
        loss = F.cross_entropy(logits, y)
        grads = torch.autograd.grad(loss, params, retain_graph=True, create_graph=False)
        with torch.no_grad():
            norm = torch.sqrt(sum(g.detach().pow(2).sum() for g in grads) + 1e-12)
            scale = (self.rho / norm).detach()

        perturbed = {n: p + g.detach() * scale for n, p, g in zip(param_names, params, grads)}
        buffers = {n: b for n, b in self.named_buffers()}
        logits_pert = torch.func.functional_call(self, {**perturbed, **buffers}, (x,))
        loss_pert = F.cross_entropy(logits_pert, y)

        total = loss + self.alpha * loss_pert
        return total, {"loss_cls": loss.detach(), "loss_sagm": loss_pert.detach()}
