"""
run.py — Démonstration rapide de GeoFlow-GFN.

Usage :
  python run.py                # GeoFlow-GFN (avec metric kernel)
  python run.py --no-metric    # GFN standard (ablation)
  python run.py --steps 2000   # plus de steps
"""
import argparse
import sys

import numpy as np
import torch

from geoflow import GeoFlowAgent, GridWorld
from geoflow.utils.metrics import evaluate

parser = argparse.ArgumentParser()
parser.add_argument("--steps", type=int, default=500, help="Gradient steps")
parser.add_argument("--no-metric", action="store_true", help="Disable metric kernel")
parser.add_argument("--D", type=int, default=4, help="State space depth")
parser.add_argument("--K", type=int, default=4, help="Actions per step")
parser.add_argument("--ls", type=float, default=1.2, help="GP-RBF length scale")
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()

use_metric = not args.no_metric
label = "GeoFlow-GFN" if use_metric else "GFN-standard"
print(f"\n{'='*55}")
print(f"  {label}")
print(f"  Espace : {args.K}^{args.D} = {args.K**args.D} états")
print(f"  length_scale = {args.ls} | steps = {args.steps} | seed = {args.seed}")
print(f"{'='*55}\n")

env = GridWorld(D=args.D, K=args.K, length_scale=args.ls, seed=args.seed)

agent = GeoFlowAgent(
    env=env,
    hidden=128,
    lr=5e-3,
    lr_metric=1e-3,
    lambda_kl=0.1,
    beta=0.01,
    batch_size=50,
    use_metric=use_metric,
    seed=args.seed,
)

history = agent.fit(
    n_steps=args.steps,
    log_every=max(1, args.steps // 10),
    verbose=True,
)

print(f"\n{'─'*55}")
print("Évaluation finale (distribution exacte):")
m = agent.evaluate(threshold=0.5)
print(f"  TV(p_θ, p_R)  = {m['tv']:.4f}")
print(f"  Mode coverage = {m['coverage']:.1%}  ({m['n_covered']}/{m['n_modes']} modes)")
print(f"  P(modes)      = {m['p_on_modes']:.4f}")
print(f"{'─'*55}\n")
