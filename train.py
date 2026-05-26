import argparse
import math
import os
import json
import time
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from tqdm import tqdm

from data import build_dataloaders, NUM_CLASSES
from models import (
    CDRM_USA_Model,
    ERMModel,
    DANNModel,
    MixStyleModel,
    DSUModel,
    EFDMixModel,
    RSCModel,
    IRMTrainer,
    FishrTrainer,
    SAGMTrainer,
    PCLModel,
    WDCNNModel,
)
from utils import set_seed, Logger, macro_f1, per_class_accuracy
from sklearn.metrics import confusion_matrix


def _load_cfg(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _device(s: str) -> torch.device:
    if s == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(s)


def _build_optimizer(model: nn.Module, cfg: dict):
    name = cfg["train"]["optimizer"].lower()
    lr = float(cfg["train"]["lr"])
    wd = float(cfg["train"]["weight_decay"])
    params = [p for p in model.parameters() if p.requires_grad]
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=wd)
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=wd, nesterov=True)
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=wd)
    raise ValueError(name)


def _build_scheduler(optimizer, cfg: dict, steps_per_epoch: int):
    epochs = cfg["train"]["epochs"]
    sname = cfg["train"]["scheduler"]
    warmup = cfg["train"]["warmup_epochs"]
    total_steps = epochs * steps_per_epoch
    warmup_steps = warmup * steps_per_epoch

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step + 1) / max(1, warmup_steps)
        if sname == "cosine":
            t = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return 0.5 * (1.0 + math.cos(math.pi * t))
        if sname == "constant":
            return 1.0
        return 1.0

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def _build_model(cfg: dict, method: str, num_conditions: int):
    num_classes = cfg["model"]["num_classes"]
    feat_dim = cfg["model"]["feat_dim"]
    backbone = cfg["model"]["backbone"]

    if method == "cdrm_usa":
        return CDRM_USA_Model(
            num_classes=num_classes,
            num_conditions=num_conditions,
            feat_dim=feat_dim,
            backbone=backbone,
            stft_cfg=cfg["stft"],
            usa_cfg=cfg["usa"],
            cdrm_cfg=cfg["cdrm"],
        )
    if method == "erm":
        return ERMModel(backbone=backbone, num_classes=num_classes, feat_dim=feat_dim)
    if method == "dann":
        return DANNModel(backbone=backbone, num_classes=num_classes, num_conditions=num_conditions, feat_dim=feat_dim)
    if method == "mixstyle":
        return MixStyleModel(backbone=backbone, num_classes=num_classes, feat_dim=feat_dim, p=cfg["baseline"]["mixstyle_p"])
    if method == "dsu":
        return DSUModel(backbone=backbone, num_classes=num_classes, feat_dim=feat_dim, p=cfg["baseline"].get("dsu_p", 0.5))
    if method == "efdmix":
        return EFDMixModel(backbone=backbone, num_classes=num_classes, feat_dim=feat_dim, p=cfg["baseline"].get("efdmix_p", 0.5), alpha=cfg["baseline"].get("efdmix_alpha", 0.1))
    if method == "rsc":
        return RSCModel(backbone=backbone, num_classes=num_classes, feat_dim=feat_dim, drop_f=cfg["baseline"]["rsc_drop_f"], drop_b=cfg["baseline"]["rsc_drop_b"])
    if method == "irm":
        return IRMTrainer(backbone=backbone, num_classes=num_classes, num_conditions=num_conditions, feat_dim=feat_dim, penalty=cfg["baseline"]["irm_penalty"])
    if method == "fishr":
        return FishrTrainer(backbone=backbone, num_classes=num_classes, num_conditions=num_conditions, feat_dim=feat_dim, penalty=cfg["baseline"]["fishr_penalty"])
    if method == "sagm":
        return SAGMTrainer(backbone=backbone, num_classes=num_classes, feat_dim=feat_dim, rho=cfg["baseline"]["sagm_rho"], alpha=cfg["baseline"]["sagm_alpha"])
    if method == "pcl":
        return PCLModel(backbone=backbone, num_classes=num_classes, feat_dim=feat_dim, temperature=cfg["baseline"]["pcl_temperature"])
    if method == "wdcnn":
        return WDCNNModel(num_classes=num_classes, feat_dim=feat_dim)
    raise ValueError(method)


def _step(model: nn.Module, batch, method: str, device, epoch: int, total_epochs: int):
    x = batch["x"].to(device, non_blocking=True)
    y = batch["y"]
    if not isinstance(y, torch.Tensor):
        y = torch.as_tensor(y)
    y = y.to(device, non_blocking=True)
    cond = batch["pseudo"]
    if not isinstance(cond, torch.Tensor):
        cond = torch.as_tensor(cond)
    cond = cond.to(device, non_blocking=True)

    if method == "cdrm_usa":
        grl_lambd = _grl_schedule(epoch, total_epochs)
        model.set_grl_lambda(grl_lambd)
        out = model(x)
        loss, log = model.compute_total_loss(out, y, cond)
        return loss, log, out["logits_cls"], y
    if method == "dann":
        model.set_lambda(_grl_schedule(epoch, total_epochs))
        loss, log = model.compute_loss(x, y, cond=cond)
        with torch.no_grad():
            logits = model(x)["logits"]
        return loss, log, logits, y
    if method in ("irm", "fishr"):
        loss, log = model.compute_loss(x, y, cond=cond)
        with torch.no_grad():
            logits = model(x)
            if isinstance(logits, tuple):
                logits = logits[0]
        return loss, log, logits, y
    loss, log = model.compute_loss(x, y)
    with torch.no_grad():
        logits = model(x)
        if isinstance(logits, tuple):
            logits = logits[0]
    return loss, log, logits, y


def _grl_schedule(epoch: int, total_epochs: int) -> float:
    p = float(epoch) / max(1, total_epochs)
    return 2.0 / (1.0 + math.exp(-10 * p)) - 1.0


@torch.no_grad()
def evaluate(model: nn.Module, loader, device, method: str):
    model.eval()
    all_p, all_y = [], []
    for batch in loader:
        x = batch["x"].to(device, non_blocking=True)
        y = batch["y"]
        if not isinstance(y, torch.Tensor):
            y = torch.as_tensor(y)
        if method == "cdrm_usa":
            logits = model.predict(x)
        elif method == "dann":
            logits = model(x)["logits"]
        elif method == "rsc":
            logits, _ = model(x)
        elif method == "pcl":
            logits, _ = model(x)
        else:
            logits = model(x)
            if isinstance(logits, tuple):
                logits = logits[0]
        all_p.append(logits.argmax(dim=-1).cpu().numpy())
        all_y.append(y.numpy())
    preds = np.concatenate(all_p)
    targets = np.concatenate(all_y)
    acc = float((preds == targets).mean())
    mf1 = macro_f1(preds, targets)
    return {"acc": acc, "macro_f1": mf1, "preds": preds, "targets": targets}


def train(cfg: dict, method: str, source: int, tag: str = None) -> Dict:
    device = _device(cfg["train"]["device"])
    set_seed(cfg["train"]["seed"])
    n_cond = cfg["data"]["pseudo_condition"]["n_clusters"]

    loaders = build_dataloaders(cfg, source_load=source, seed=cfg["train"]["seed"])
    train_loader = loaders["train"]
    val_loader = loaders["val"]
    target_loaders = loaders["targets"]

    model = _build_model(cfg, method, num_conditions=n_cond).to(device)
    optimizer = _build_optimizer(model, cfg)
    scheduler = _build_scheduler(optimizer, cfg, steps_per_epoch=len(train_loader))

    tag = tag or f"{method}_src{source}"
    logger = Logger(cfg["train"]["log_dir"], tag)
    logger.info(f"device={device} train={len(train_loader.dataset)} val={len(val_loader.dataset)}")
    logger.info(f"targets={ {ld: len(dl.dataset) for ld, dl in target_loaders.items()} }")

    best_val = -1.0
    best_target = {}
    os.makedirs(cfg["train"]["ckpt_dir"], exist_ok=True)
    ckpt_path = os.path.join(cfg["train"]["ckpt_dir"], f"{tag}.pt")

    total_epochs = cfg["train"]["epochs"]
    for epoch in range(total_epochs):
        model.train()
        running_loss = 0.0
        running_acc = 0.0
        n_batches = 0
        t0 = time.time()
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss, log, logits, y = _step(model, batch, method, device, epoch, total_epochs)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            scheduler.step()
            running_loss += float(loss.item())
            running_acc += float((logits.argmax(dim=-1) == y).float().mean().item())
            n_batches += 1
        train_loss = running_loss / max(1, n_batches)
        train_acc = running_acc / max(1, n_batches)

        val_metrics = evaluate(model, val_loader, device, method)
        payload = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_acc": val_metrics["acc"],
            "val_macro_f1": val_metrics["macro_f1"],
            "lr": optimizer.param_groups[0]["lr"],
            "dt": round(time.time() - t0, 2),
        }
        logger.log(payload)
        logger.info(
            f"ep{epoch:02d} loss={train_loss:.4f} tr_acc={train_acc:.4f} val_acc={val_metrics['acc']:.4f} val_f1={val_metrics['macro_f1']:.4f}"
        )

        if val_metrics["macro_f1"] > best_val:
            best_val = val_metrics["macro_f1"]
            tgt_results = {}
            for ld, dl in target_loaders.items():
                m = evaluate(model, dl, device, method)
                pcc = per_class_accuracy(m["preds"], m["targets"], cfg["model"]["num_classes"]).tolist()
                cm = confusion_matrix(m["targets"], m["preds"], labels=list(range(cfg["model"]["num_classes"]))).tolist()
                tgt_results[str(ld)] = {
                    "acc": m["acc"],
                    "macro_f1": m["macro_f1"],
                    "per_class_acc": pcc,
                    "confmat": cm,
                }
            best_target = tgt_results
            torch.save({"state_dict": model.state_dict(), "cfg": cfg, "method": method, "source": source}, ckpt_path)
            logger.log({"event": "save_best", "epoch": epoch, "val_f1": best_val, "targets": {k: {"acc": v["acc"], "macro_f1": v["macro_f1"]} for k, v in best_target.items()}})

    summary = {
        "method": method,
        "source": source,
        "best_val_macro_f1": best_val,
        "targets": best_target,
        "ckpt": ckpt_path,
    }
    out_path = os.path.join(cfg["train"]["log_dir"], f"{tag}.summary.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"summary -> {out_path}")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument(
        "--method",
        type=str,
        default="cdrm_usa",
        choices=["cdrm_usa", "erm", "dann", "mixstyle", "dsu", "efdmix", "rsc", "irm", "fishr", "sagm", "pcl", "wdcnn"],
    )
    parser.add_argument("--source", type=int, default=0)
    parser.add_argument("--tag", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    args = parser.parse_args()

    cfg = _load_cfg(args.config)
    if args.epochs is not None:
        cfg["train"]["epochs"] = args.epochs
    if args.batch_size is not None:
        cfg["train"]["batch_size"] = args.batch_size
    summary = train(cfg, args.method, args.source, args.tag)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
