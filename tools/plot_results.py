import argparse
import json
import os
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


METHOD_LABELS = {
    "wdcnn": "WDCNN",
    "erm": "ERM",
    "dann": "DANN",
    "mixstyle": "MixStyle",
    "dsu": "DSU",
    "efdmix": "EFDMix",
    "rsc": "RSC",
    "irm": "IRM",
    "fishr": "Fishr",
    "sagm": "SAGM",
    "pcl": "PCL",
    "cdrm_usa": "CDRM-USA",
}


def _load_results(path: str) -> dict:
    """读取 run_all.py 生成的聚合 JSON，并检查必要字段是否存在。"""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if "summary" not in data:
        raise ValueError(f"{path} 缺少 summary 字段，无法绘制聚合结果图。")
    return data


def _method_label(method: str) -> str:
    """把内部方法名转换为图中更易读的显示名称。"""
    return METHOD_LABELS.get(method, method)


def _ordered_methods(summary: Dict[str, dict]) -> List[str]:
    """按 mean_f1 从高到低排序，便于图中突出整体表现更好的方法。"""
    return sorted(summary.keys(), key=lambda m: summary[m].get("mean_f1", 0.0), reverse=True)


def _collect_method_scores(summary: Dict[str, dict], methods: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    """提取每个方法的跨源负载平均 Accuracy 与 Macro-F1。"""
    mean_acc = np.asarray([summary[m].get("mean_acc", np.nan) for m in methods], dtype=float)
    mean_f1 = np.asarray([summary[m].get("mean_f1", np.nan) for m in methods], dtype=float)
    return mean_acc, mean_f1


def _source_mean_metric(method_info: dict, metric: str) -> Dict[str, float]:
    """计算单个方法在每个源负载下，对所有目标负载的平均指标。"""
    out: Dict[str, float] = {}
    for src, src_info in method_info.get("sources", {}).items():
        vals = [float(tgt_info[metric]) for tgt_info in src_info.get("targets", {}).values()]
        out[str(src)] = float(np.mean(vals)) if vals else np.nan
    return out


def _target_matrix(method_info: dict, metric: str) -> Tuple[List[str], List[str], np.ndarray]:
    """构建 source -> target 的矩阵；对角线没有目标域评估，使用 NaN 留白。"""
    sources = sorted(method_info.get("sources", {}).keys(), key=lambda x: int(x))
    targets = sorted({t for s in method_info.get("sources", {}).values() for t in s.get("targets", {})}, key=lambda x: int(x))
    mat = np.full((len(sources), len(targets)), np.nan, dtype=float)
    for i, src in enumerate(sources):
        for j, tgt in enumerate(targets):
            tgt_info = method_info["sources"].get(src, {}).get("targets", {}).get(tgt)
            if tgt_info is not None:
                mat[i, j] = float(tgt_info[metric])
    return sources, targets, mat


def _annotate_bars(ax, bars, values: np.ndarray) -> None:
    """给柱状图标注数值，避免读图时需要反复对照坐标轴。"""
    for bar, value in zip(bars, values):
        if np.isnan(value):
            continue
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 0.003,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
            rotation=90,
        )


def _save(fig, out_path: str) -> None:
    """统一保存图片，确保目录存在并释放 Matplotlib 资源。"""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_overall_bars(summary: Dict[str, dict], out_dir: str) -> str:
    """绘制各方法跨源负载平均 Accuracy 与 Macro-F1 对比柱状图。"""
    methods = _ordered_methods(summary)
    labels = [_method_label(m) for m in methods]
    mean_acc, mean_f1 = _collect_method_scores(summary, methods)
    x = np.arange(len(methods))
    width = 0.38

    fig, ax = plt.subplots(figsize=(12, 5.8))
    bars_f1 = ax.bar(x - width / 2, mean_f1, width, label="Macro-F1", color="#3b82f6")
    bars_acc = ax.bar(x + width / 2, mean_acc, width, label="Accuracy", color="#10b981")
    _annotate_bars(ax, bars_f1, mean_f1)
    _annotate_bars(ax, bars_acc, mean_acc)
    ax.set_ylabel("Score")
    ax.set_ylim(max(0.0, float(np.nanmin([mean_acc, mean_f1])) - 0.04), 1.02)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_title("Overall Target-Domain Performance")
    ax.grid(axis="y", linestyle="--", alpha=0.28)
    ax.legend(loc="lower right")

    out_path = os.path.join(out_dir, "fig21_overall_method_scores.png")
    _save(fig, out_path)
    return out_path


def plot_method_source_heatmap(summary: Dict[str, dict], out_dir: str, metric: str = "macro_f1") -> str:
    """绘制方法 x 源负载热力图，单元格为该源负载下所有目标负载的平均指标。"""
    methods = _ordered_methods(summary)
    sources = sorted({s for m in methods for s in summary[m].get("sources", {})}, key=lambda x: int(x))
    mat = np.full((len(methods), len(sources)), np.nan, dtype=float)
    for i, method in enumerate(methods):
        src_scores = _source_mean_metric(summary[method], metric)
        for j, src in enumerate(sources):
            mat[i, j] = src_scores.get(src, np.nan)

    fig, ax = plt.subplots(figsize=(7.2, 7.4))
    im = ax.imshow(mat, cmap="YlGnBu", vmin=max(0.0, np.nanmin(mat) - 0.02), vmax=1.0)
    ax.set_xticks(np.arange(len(sources)))
    ax.set_xticklabels([f"src {s}" for s in sources])
    ax.set_yticks(np.arange(len(methods)))
    ax.set_yticklabels([_method_label(m) for m in methods])
    ax.set_title(f"Mean Target {metric.replace('_', '-')} by Source Load")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if not np.isnan(mat[i, j]):
                ax.text(j, i, f"{mat[i, j]:.3f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    out_path = os.path.join(out_dir, f"fig22_method_source_{metric}.png")
    _save(fig, out_path)
    return out_path


def plot_best_target_matrix(summary: Dict[str, dict], out_dir: str, metric: str = "macro_f1") -> str:
    """绘制整体 Macro-F1 最高方法的 source -> target 指标矩阵。"""
    best_method = _ordered_methods(summary)[0]
    sources, targets, mat = _target_matrix(summary[best_method], metric)

    fig, ax = plt.subplots(figsize=(6.6, 5.6))
    im = ax.imshow(mat, cmap="magma", vmin=max(0.0, np.nanmin(mat) - 0.02), vmax=1.0)
    ax.set_xticks(np.arange(len(targets)))
    ax.set_xticklabels([f"tgt {t}" for t in targets])
    ax.set_yticks(np.arange(len(sources)))
    ax.set_yticklabels([f"src {s}" for s in sources])
    ax.set_xlabel("Target load")
    ax.set_ylabel("Source load")
    ax.set_title(f"{_method_label(best_method)} Source-to-Target {metric.replace('_', '-')}")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if not np.isnan(mat[i, j]):
                ax.text(j, i, f"{mat[i, j]:.3f}", ha="center", va="center", fontsize=9, color="white")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    out_path = os.path.join(out_dir, f"fig23_best_method_target_{metric}.png")
    _save(fig, out_path)
    return out_path


def plot_per_class_accuracy(summary: Dict[str, dict], out_dir: str) -> str:
    """绘制最佳方法在所有 source-target 组合上的平均逐类准确率。"""
    best_method = _ordered_methods(summary)[0]
    vectors = []
    for src_info in summary[best_method].get("sources", {}).values():
        for tgt_info in src_info.get("targets", {}).values():
            vectors.append(np.asarray(tgt_info.get("per_class_acc", []), dtype=float))
    if not vectors:
        raise ValueError(f"{best_method} 缺少 per_class_acc，无法绘制逐类准确率图。")
    per_class = np.nanmean(np.vstack(vectors), axis=0)
    classes = [f"C{i}" for i in range(len(per_class))]

    fig, ax = plt.subplots(figsize=(9, 4.8))
    bars = ax.bar(np.arange(len(per_class)), per_class, color="#f59e0b")
    _annotate_bars(ax, bars, per_class)
    ax.set_xticks(np.arange(len(per_class)))
    ax.set_xticklabels(classes)
    ax.set_ylim(max(0.0, float(np.nanmin(per_class)) - 0.06), 1.04)
    ax.set_ylabel("Accuracy")
    ax.set_title(f"{_method_label(best_method)} Mean Per-Class Accuracy")
    ax.grid(axis="y", linestyle="--", alpha=0.28)

    out_path = os.path.join(out_dir, "fig24_best_method_per_class_accuracy.png")
    _save(fig, out_path)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="results/all_methods.json")
    parser.add_argument("--out-dir", type=str, default="results/figs")
    args = parser.parse_args()

    data = _load_results(args.input)
    summary = data["summary"]
    saved = [
        plot_overall_bars(summary, args.out_dir),
        plot_method_source_heatmap(summary, args.out_dir, metric="macro_f1"),
        plot_method_source_heatmap(summary, args.out_dir, metric="acc"),
        plot_best_target_matrix(summary, args.out_dir, metric="macro_f1"),
        plot_per_class_accuracy(summary, args.out_dir),
    ]
    print("saved figures:")
    for path in saved:
        print(path)


if __name__ == "__main__":
    main()
