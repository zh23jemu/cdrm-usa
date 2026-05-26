import argparse
import json
import os
from typing import List

import yaml

from train import train, _load_cfg


METHODS_DEFAULT = [
    "wdcnn",
    "erm",
    "dann",
    "mixstyle",
    "dsu",
    "efdmix",
    "rsc",
    "irm",
    "fishr",
    "sagm",
    "pcl",
    "cdrm_usa",
]


def _aggregate(results: List[dict]) -> dict:
    agg = {}
    for r in results:
        method = r["method"]
        agg.setdefault(method, {"sources": {}, "mean_acc": 0.0, "mean_f1": 0.0, "n": 0})
        per_src = {"val_macro_f1": r["best_val_macro_f1"], "targets": r["targets"]}
        agg[method]["sources"][str(r["source"])] = per_src
        tgt_accs = [v["acc"] for v in r["targets"].values()]
        tgt_f1s = [v["macro_f1"] for v in r["targets"].values()]
        agg[method]["mean_acc"] += sum(tgt_accs) / max(1, len(tgt_accs))
        agg[method]["mean_f1"] += sum(tgt_f1s) / max(1, len(tgt_f1s))
        agg[method]["n"] += 1
    for m, v in agg.items():
        n = v["n"]
        v["mean_acc"] = v["mean_acc"] / max(1, n)
        v["mean_f1"] = v["mean_f1"] / max(1, n)
    return agg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--methods", type=str, nargs="+", default=METHODS_DEFAULT)
    parser.add_argument("--sources", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument("--seeds", type=int, nargs="+", default=[2026])
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--out", type=str, default="results/all_methods.json")
    args = parser.parse_args()

    cfg = _load_cfg(args.config)
    if args.epochs is not None:
        cfg["train"]["epochs"] = args.epochs

    results = []
    for seed in args.seeds:
        for method in args.methods:
            for src in args.sources:
                cfg["train"]["seed"] = int(seed)
                tag = f"{method}_src{src}_seed{seed}"
                print(f"\n=== {tag} ===")
                summary = train(cfg, method, src, tag=tag)
                summary["seed"] = int(seed)
                results.append(summary)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    agg = _aggregate(results)
    out = {"runs": results, "summary": agg}
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(agg, indent=2))


if __name__ == "__main__":
    main()
