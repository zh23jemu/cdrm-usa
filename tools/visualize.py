import argparse
import os
import json

import numpy as np
import torch
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from data import build_dataloaders
from train import _build_model, _device


@torch.no_grad()
def extract_features(model, loader, device, method: str):
    model.eval()
    feats, labels, loads = [], [], []
    for batch in loader:
        x = batch["x"].to(device, non_blocking=True)
        if method == "cdrm_usa":
            h, _, _ = model.encode(x)
            z = model.cdrm.proj_f(h)
        elif method == "dann":
            z = model.backbone(x)
        elif method in ("rsc", "pcl"):
            z = model.backbone(x)
        else:
            z = model.backbone(x) if hasattr(model, "backbone") else model(x)
        feats.append(z.cpu().numpy())
        labels.append(batch["y"].numpy() if isinstance(batch["y"], torch.Tensor) else np.asarray(batch["y"]))
        loads.append(batch["load"].numpy() if isinstance(batch["load"], torch.Tensor) else np.asarray(batch["load"]))
    return np.concatenate(feats), np.concatenate(labels), np.concatenate(loads)


def plot_tsne(feats, labels, loads, out_path: str):
    tsne = TSNE(n_components=2, perplexity=30, init="pca", random_state=0)
    proj = tsne.fit_transform(feats)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sc0 = axes[0].scatter(proj[:, 0], proj[:, 1], c=labels, cmap="tab10", s=8, alpha=0.8)
    axes[0].set_title("by class")
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    plt.colorbar(sc0, ax=axes[0])
    sc1 = axes[1].scatter(proj[:, 0], proj[:, 1], c=loads, cmap="viridis", s=8, alpha=0.8)
    axes[1].set_title("by working condition (load)")
    axes[1].set_xticks([])
    axes[1].set_yticks([])
    plt.colorbar(sc1, ax=axes[1])
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_confmat(cm, out_path: str, title: str = ""):
    cm = np.asarray(cm, dtype=float)
    norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(norm, cmap="Blues", vmin=0.0, vmax=1.0)
    ax.set_title(title)
    ax.set_xlabel("Pred")
    ax.set_ylabel("True")
    plt.colorbar(im, ax=ax)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--source", type=int, required=True)
    parser.add_argument("--out-dir", type=str, default="results/figs")
    args = parser.parse_args()

    state = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = state["cfg"]
    method = state["method"]
    device = _device(cfg["train"]["device"])

    loaders = build_dataloaders(cfg, source_load=args.source, seed=cfg["train"]["seed"])
    model = _build_model(cfg, method, num_conditions=cfg["data"]["pseudo_condition"]["n_clusters"]).to(device)
    model.load_state_dict(state["state_dict"])

    src_feats, src_labels, src_loads = extract_features(model, loaders["val"], device, method)
    all_feats = [src_feats]
    all_labels = [src_labels]
    all_loads = [src_loads]
    for ld, dl in loaders["targets"].items():
        f, y, l = extract_features(model, dl, device, method)
        all_feats.append(f)
        all_labels.append(y)
        all_loads.append(l)

    feats = np.concatenate(all_feats, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    loads = np.concatenate(all_loads, axis=0)

    tag = os.path.splitext(os.path.basename(args.ckpt))[0]
    plot_tsne(feats, labels, loads, os.path.join(args.out_dir, f"{tag}_tsne.png"))

    summary_path = os.path.join(cfg["train"]["log_dir"], f"{tag}.summary.json")
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            summary = json.load(f)
        for ld, info in summary.get("targets", {}).items():
            cm = info.get("confmat")
            if cm is not None:
                plot_confmat(cm, os.path.join(args.out_dir, f"{tag}_confmat_tgt{ld}.png"), title=f"target load {ld}")
    print(f"saved to {args.out_dir}")


if __name__ == "__main__":
    main()
