"""
experiments/multiseed.py — Multi-seed reproducibility benchmark for GeoFlow-GFN.

Runs N_SEEDS training seeds × 3 landscape seeds × 2 conditions (GeoFlow / GFN-std)
× 3 length scales, then writes results/multiseed.json and prints a summary table.

Usage:
    python experiments/multiseed.py                  # default: 5 train seeds
    python experiments/multiseed.py --seeds 10       # more seeds
    python experiments/multiseed.py --quick          # 3 seeds, smaller budget
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from geoflow import GeoFlowAgent, GridWorld

# ── Configuration ─────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument("--seeds", type=int, default=5, help="Number of training seeds")
parser.add_argument("--quick", action="store_true", help="Smaller budget for fast testing")
parser.add_argument("--out", type=str, default=None, help="Output JSON path")
args = parser.parse_args()

D, K = 4, 4                          # 4^4 = 256 states
LS_VALUES = [0.4, 1.2, 4.0]
LAND_SEEDS = [0, 1, 2]              # 3 reward landscapes per ls
TRAIN_SEEDS = list(range(args.seeds))
N_STEPS = 200 if args.quick else 500
BATCH = 50

OUT = args.out or os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "results", "multiseed.json"
)


# ── Runner ────────────────────────────────────────────────────────────────────

def run_one(env, use_metric: bool, train_seed: int) -> dict:
    torch.manual_seed(train_seed)
    np.random.seed(train_seed)
    agent = GeoFlowAgent(
        env=env, hidden=64, lr=5e-3, lr_metric=5e-3,
        lambda_kl=0.2, beta=0.01, batch_size=BATCH,
        use_metric=use_metric, seed=train_seed,
    )
    agent.fit(n_steps=N_STEPS, log_every=N_STEPS + 1, verbose=False)
    return agent.evaluate(threshold=0.5)


def run_condition(ls: float, ls_seed: int, use_metric: bool,
                  train_seeds: list[int]) -> dict:
    env = GridWorld(D=D, K=K, length_scale=ls, seed=ls_seed)
    tvs, coverages = [], []
    for ts in train_seeds:
        m = run_one(env, use_metric, ts)
        tvs.append(m["tv"])
        coverages.append(m["coverage"])
    return {
        "tv_mean": float(np.mean(tvs)),
        "tv_std": float(np.std(tvs)),
        "tv_values": [float(v) for v in tvs],
        "coverage_mean": float(np.mean(coverages)),
        "coverage_std": float(np.std(coverages)),
        "coverage_values": [float(v) for v in coverages],
    }


# ── Main ──────────────────────────────────────────────────────────────────────

n_runs = len(LS_VALUES) * len(LAND_SEEDS) * 2  # 2 conditions
print(f"\nGeoFlow-GFN — Multi-seed benchmark")
print(f"  D={D}, K={K}, N={K**D} states | n_steps={N_STEPS} × batch={BATCH}")
print(f"  {args.seeds} train seeds × {len(LAND_SEEDS)} landscape seeds × {len(LS_VALUES)} ls × 2 conditions")
print(f"  Total runs: {n_runs * args.seeds} | Est. time: ~{n_runs * args.seeds * 2}s")
print()

t0 = time.time()
results: dict = {"config": {
    "D": D, "K": K, "n_states": K**D,
    "n_steps": N_STEPS, "batch_size": BATCH,
    "train_seeds": TRAIN_SEEDS, "land_seeds": LAND_SEEDS,
    "ls_values": LS_VALUES,
}, "by_ls": {}}

print(f"{'ls':>5} | {'Condition':>10} | {'TV mean':>9} | {'TV std':>8} | {'Coverage':>9} | {'Cov std':>8}")
print("─" * 65)

for ls in LS_VALUES:
    results["by_ls"][str(ls)] = {}
    for label, use_metric in [("GFN-std", False), ("GeoFlow", True)]:
        land_tvs, land_covs = [], []
        for ls_seed in LAND_SEEDS:
            r = run_condition(ls, ls_seed, use_metric, TRAIN_SEEDS)
            land_tvs.extend(r["tv_values"])
            land_covs.extend(r["coverage_values"])
        agg = {
            "tv_mean": float(np.mean(land_tvs)),
            "tv_std": float(np.std(land_tvs)),
            "tv_values": land_tvs,
            "coverage_mean": float(np.mean(land_covs)),
            "coverage_std": float(np.std(land_covs)),
            "coverage_values": land_covs,
        }
        results["by_ls"][str(ls)][label] = agg
        print(
            f"{ls:>5.1f} | {label:>10} | "
            f"{agg['tv_mean']:>9.4f} | {agg['tv_std']:>8.4f} | "
            f"{agg['coverage_mean']:>9.3f} | {agg['coverage_std']:>8.3f}"
        )

elapsed = time.time() - t0
print(f"\nDone in {elapsed:.1f}s")

# ── Acceleration summary ──────────────────────────────────────────────────────
print(f"\n{'ls':>5} | {'TV GFN-std':>12} | {'TV GeoFlow':>12} | {'Δ TV':>8}")
print("─" * 50)
for ls in LS_VALUES:
    std = results["by_ls"][str(ls)]["GFN-std"]
    geo = results["by_ls"][str(ls)]["GeoFlow"]
    delta = std["tv_mean"] - geo["tv_mean"]
    print(
        f"{ls:>5.1f} | "
        f"{std['tv_mean']:.4f} ± {std['tv_std']:.4f} | "
        f"{geo['tv_mean']:.4f} ± {geo['tv_std']:.4f} | "
        f"{delta:>+8.4f}"
    )

# ── Save ─────────────────────────────────────────────────────────────────────
results["elapsed_s"] = round(elapsed, 1)
os.makedirs(os.path.dirname(os.path.abspath(OUT)), exist_ok=True)
with open(OUT, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nResults → {OUT}")
