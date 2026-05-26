import torch
import torch.nn as nn
import torch.nn.functional as F


class _GradReverseFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = float(lambd)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambd * grad_output, None


def grad_reverse(x: torch.Tensor, lambd: float = 1.0) -> torch.Tensor:
    return _GradReverseFn.apply(x, lambd)


class GradReverse(nn.Module):
    def __init__(self, lambd: float = 1.0) -> None:
        super().__init__()
        self.lambd = lambd

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return grad_reverse(x, self.lambd)


def consistency_loss(
    logits_a: torch.Tensor,
    logits_b: torch.Tensor,
    weight: torch.Tensor = None,
    mode: str = "kl",
) -> torch.Tensor:
    if mode == "kl":
        log_p = F.log_softmax(logits_a, dim=-1)
        q = F.softmax(logits_b.detach(), dim=-1)
        kl = F.kl_div(log_p, q, reduction="none").sum(dim=-1)
        if weight is None:
            return kl.mean()
        return (kl * weight).mean()
    if mode == "sym_kl":
        log_p = F.log_softmax(logits_a, dim=-1)
        log_q = F.log_softmax(logits_b, dim=-1)
        p = log_p.exp()
        q = log_q.exp()
        kl_pq = (p * (log_p - log_q)).sum(dim=-1)
        kl_qp = (q * (log_q - log_p)).sum(dim=-1)
        out = 0.5 * (kl_pq + kl_qp)
        if weight is None:
            return out.mean()
        return (out * weight).mean()
    if mode == "mse":
        diff = (logits_a - logits_b).pow(2).mean(dim=-1)
        if weight is None:
            return diff.mean()
        return (diff * weight).mean()
    raise ValueError(mode)


def entropy_loss(logits: torch.Tensor) -> torch.Tensor:
    p = F.softmax(logits, dim=-1)
    log_p = F.log_softmax(logits, dim=-1)
    return -(p * log_p).sum(dim=-1).mean()


def mmd_loss(x: torch.Tensor, y: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
    def k(a, b):
        d = (a.unsqueeze(1) - b.unsqueeze(0)).pow(2).sum(-1)
        return torch.exp(-d / (2.0 * sigma * sigma))

    return k(x, x).mean() + k(y, y).mean() - 2 * k(x, y).mean()


def coral_loss(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    d = source.size(1)
    s_mean = source.mean(0, keepdim=True)
    t_mean = target.mean(0, keepdim=True)
    sc = source - s_mean
    tc = target - t_mean
    cs = sc.t() @ sc / max(source.size(0) - 1, 1)
    ct = tc.t() @ tc / max(target.size(0) - 1, 1)
    return (cs - ct).pow(2).sum() / (4 * d * d)
