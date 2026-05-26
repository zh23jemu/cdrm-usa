import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.losses import grad_reverse


class _ProjectionHead(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CDRM(nn.Module):
    """
    Causal Disentangled Representation Module.

    Splits the shared backbone feature h into:
      z_f : fault-invariant representation (independent of working condition)
      z_c : working-condition-related representation

    Disentanglement is enforced by:
      (i) Gradient Reversal Layer on z_f when feeding the condition classifier
          c_grl: makes z_f indistinguishable across working conditions.
      (ii) Direct condition prediction on z_c: makes z_c maximally informative
           about the working condition.

    Pseudo working-condition labels are produced by data.cwru._spectral_features
    + KMeans inside one source domain (single-source DG).
    """

    def __init__(
        self,
        feat_dim: int,
        proj_dim: int,
        num_classes: int,
        num_conditions: int,
        lambda_grl: float = 1.0,
        lambda_cond: float = 0.5,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.feat_dim = feat_dim
        self.proj_dim = proj_dim
        self.num_classes = num_classes
        self.num_conditions = num_conditions
        self.lambda_grl = lambda_grl
        self.lambda_cond = lambda_cond

        self.proj_f = _ProjectionHead(feat_dim, feat_dim, proj_dim, dropout)
        self.proj_c = _ProjectionHead(feat_dim, feat_dim, proj_dim, dropout)

        self.fault_classifier = nn.Linear(proj_dim, num_classes)
        self.cond_classifier = nn.Linear(proj_dim, num_conditions)
        self.cond_classifier_grl = nn.Linear(proj_dim, num_conditions)

    def set_lambda(self, lambd: float) -> None:
        self.lambda_grl = float(lambd)

    def forward(self, h: torch.Tensor):
        z_f = self.proj_f(h)
        z_c = self.proj_c(h)

        logits_cls = self.fault_classifier(z_f)

        z_f_rev = grad_reverse(z_f, self.lambda_grl)
        logits_cond_adv = self.cond_classifier_grl(z_f_rev)

        logits_cond = self.cond_classifier(z_c)

        return {
            "z_f": z_f,
            "z_c": z_c,
            "logits_cls": logits_cls,
            "logits_cond_adv": logits_cond_adv,
            "logits_cond": logits_cond,
        }

    def compute_losses(
        self,
        out: dict,
        y: torch.Tensor,
        cond: torch.Tensor,
    ):
        loss_cls = F.cross_entropy(out["logits_cls"], y)
        loss_cond_adv = F.cross_entropy(out["logits_cond_adv"], cond)
        loss_cond = F.cross_entropy(out["logits_cond"], cond)
        ortho = self._orthogonality(out["z_f"], out["z_c"])
        return {
            "loss_cls": loss_cls,
            "loss_cond_adv": loss_cond_adv,
            "loss_cond": loss_cond,
            "loss_ortho": ortho,
        }

    @staticmethod
    def _orthogonality(z_f: torch.Tensor, z_c: torch.Tensor) -> torch.Tensor:
        zf = F.normalize(z_f, dim=-1)
        zc = F.normalize(z_c, dim=-1)
        cross = (zf * zc).sum(dim=-1)
        return cross.pow(2).mean()
