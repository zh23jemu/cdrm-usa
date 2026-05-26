import numpy as np
import torch
from sklearn.metrics import f1_score, confusion_matrix


def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=-1)
    return (preds == targets).float().mean().item()


def macro_f1(preds: np.ndarray, targets: np.ndarray) -> float:
    return float(f1_score(targets, preds, average="macro", zero_division=0))


def per_class_accuracy(preds: np.ndarray, targets: np.ndarray, num_classes: int) -> np.ndarray:
    cm = confusion_matrix(targets, preds, labels=list(range(num_classes)))
    row_sum = cm.sum(axis=1).clip(min=1)
    return cm.diagonal().astype(np.float32) / row_sum.astype(np.float32)


@torch.no_grad()
def evaluate_classifier(model, loader, device, forward_fn=None):
    model.eval()
    all_preds, all_targets = [], []
    for batch in loader:
        x = batch["x"].to(device, non_blocking=True)
        y = batch["y"].to(device, non_blocking=True) if isinstance(batch["y"], torch.Tensor) else torch.as_tensor(batch["y"], device=device)
        if forward_fn is None:
            logits = model(x)
            if isinstance(logits, (tuple, list)):
                logits = logits[0]
            if isinstance(logits, dict):
                logits = logits.get("logits", next(iter(logits.values())))
        else:
            logits = forward_fn(model, x)
        all_preds.append(logits.argmax(dim=-1).cpu().numpy())
        all_targets.append(y.cpu().numpy())
    preds = np.concatenate(all_preds)
    targets = np.concatenate(all_targets)
    acc = float((preds == targets).mean())
    mf1 = macro_f1(preds, targets)
    return {"acc": acc, "macro_f1": mf1, "preds": preds, "targets": targets}
