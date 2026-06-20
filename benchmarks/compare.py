"""
benchmarks/compare.py — GeoFlow vs GFN-standard : convergence speed.

Question : GeoFlow converge-t-il PLUS VITE vers TV < seuil ?
Espace : GridWorld 4^4 = 256 états, budget faible (100 steps × 50 = 5000 traj ≈ 20×N).
Métriques : TV à plusieurs checkpoints + "steps to TV < 0.10".
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from geoflow import GeoFlowAgent, GridWorld
from geoflow.utils.metrics import evaluate

D, K = 4, 6                # N=1296 (harder exploration)
LS_VALUES   = [0.4, 1.2, 4.0]
LAND_SEEDS  = [0, 1, 2]
TRAIN_SEEDS = [10, 11]
N_STEPS     = 60          # 60 × 50 = 3000 traj ≈ 2.3×N (tight budget)
LOG_EVERY   = 5
BATCH       = 50
TV_THRESH   = 0.30        # looser threshold given small budget


def run_condition(env, use_metric, train_seed) -> tuple[list, list]:
    """Returns (steps_list, tv_list) at each checkpoint."""
    torch.manual_seed(train_seed); np.random.seed(train_seed)
    agent = GeoFlowAgent(
        env=env, hidden=128, lr=5e-3, lr_metric=5e-3,
        lambda_kl=0.2, beta=0.01,
        batch_size=BATCH, use_metric=use_metric, seed=train_seed,
    )
    agent.trainer._total_steps = N_STEPS   # needed for _epsilon() schedule
    steps_list, tv_list = [], []

    for step in range(N_STEPS):
        agent.trainer.step()
        if step % LOG_EVERY == 0 or step == N_STEPS - 1:
            m = agent.evaluate(threshold=0.5)
            steps_list.append(step)
            tv_list.append(m["tv"])

    return steps_list, tv_list


def steps_to_threshold(steps, tvs, thresh):
    """Return first step where TV < thresh, or N_STEPS if never reached."""
    for s, tv in zip(steps, tvs):
        if tv < thresh:
            return s
    return N_STEPS


print(f"\nBudget : {N_STEPS} steps × {BATCH} = {N_STEPS*BATCH} traj | "
      f"N={K**D} states | TV seuil = {TV_THRESH}")
print(f"\n{'l':>5} | {'Steps-to-{:.2f} GFN'.format(TV_THRESH):>20} | "
      f"{'Steps-to-{:.2f} GeoFlow'.format(TV_THRESH):>22} | {'Accélération':>13}")
print("-" * 75)

all_curves = {}
for ls in LS_VALUES:
    std_sts, geo_sts = [], []
    std_curves, geo_curves = [], []

    for ls_seed in LAND_SEEDS:
        env = GridWorld(D=D, K=K, length_scale=ls, seed=ls_seed)
        for ts in TRAIN_SEEDS:
            s_std, tv_std = run_condition(env, use_metric=False, train_seed=ts)
            s_geo, tv_geo = run_condition(env, use_metric=True,  train_seed=ts)
            std_sts.append(steps_to_threshold(s_std, tv_std, TV_THRESH))
            geo_sts.append(steps_to_threshold(s_geo, tv_geo, TV_THRESH))
            std_curves.append(tv_std)
            geo_curves.append(tv_geo)

    m_std = np.mean(std_sts); m_geo = np.mean(geo_sts)
    ratio = m_std / max(m_geo, 1)
    print(f"{ls:5.1f} | {m_std:>20.1f} ± {np.std(std_sts):.1f} | "
          f"{m_geo:>22.1f} ± {np.std(geo_sts):.1f} | {ratio:>10.2f}×")
    all_curves[ls] = dict(std=np.array(std_curves),
                          geo=np.array(geo_curves),
                          steps=s_std)

# ── Figure ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
colors = {"std": "#7aa2ff", "geo": "#e74c3c"}

for ax, ls in zip(axes, LS_VALUES):
    d = all_curves[ls]
    steps = d["steps"]
    std_m = d["std"].mean(0); std_s = d["std"].std(0)
    geo_m = d["geo"].mean(0); geo_s = d["geo"].std(0)

    ax.fill_between(steps, std_m - std_s, std_m + std_s,
                    alpha=0.15, color=colors["std"])
    ax.fill_between(steps, geo_m - geo_s, geo_m + geo_s,
                    alpha=0.15, color=colors["geo"])
    ax.plot(steps, std_m, "s--", color=colors["std"], lw=2, ms=5,
            label="GFN standard")
    ax.plot(steps, geo_m, "o-",  color=colors["geo"], lw=2, ms=5,
            label="GeoFlow-GFN")
    ax.axhline(TV_THRESH, c="gray", ls=":", lw=1, label=f"TV={TV_THRESH}")
    ax.set_xlabel("Gradient steps")
    ax.set_ylabel("TV(p_θ, p_R)")
    ax.set_title(f"l = {ls}", fontsize=10)
    ax.legend(fontsize=8)
    ax.set_ylim(bottom=0)

plt.suptitle(
    f"GeoFlow-GFN : vitesse de convergence vs GFN standard\n"
    f"GridWorld {K}^{D}={K**D} états | budget {N_STEPS}×{BATCH}={N_STEPS*BATCH} traj "
    f"| {len(LAND_SEEDS)*len(TRAIN_SEEDS)} runs",
    fontsize=10, y=1.02,
)
plt.tight_layout()
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "compare.png")
plt.savefig(out, dpi=130, bbox_inches="tight")
print(f"\nFigure → {out}")
