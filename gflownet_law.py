"""
Loi de régime quantitative : Gain_GFN = f(ρ(R))

Hypothèse : le gain de couverture d'un GFlowNet par rapport à une recherche
uniforme est une fonction monotone croissante de la corrélation locale ρ(R),
définie comme la corrélation de Pearson entre R(x) et la moyenne de R sur
les voisins de x.

Protocole :
  - Espace : {0..4}^6 = 15625 séquences (distance de Hamming comme topologie)
  - Famille de récompenses : R_α = α * R_lisse + (1-α) * R_bruit, α ∈ {0, 0.1, ..., 1.0}
  - Pour chaque α :
      1. Calculer ρ(R_α) sur tout l'espace
      2. Lancer GFN et Uniforme (même budget)
      3. Mesurer Coverage(GFN) / Coverage(Uniforme) = Gain
  - Tracer Gain vs ρ et ajuster une droite
"""
import math, torch, torch.nn as nn, torch.optim as optim
import numpy as np
from itertools import product as iproduct

torch.manual_seed(0); np.random.seed(0)

# ── Espace : {0..4}^6 ─────────────────────────────────────────────────────────
V_TOK = 5; SEQ_L = 6
ALL_SEQS = list(iproduct(range(V_TOK), repeat=SEQ_L))   # 15625 séquences
IDX = {s: i for i, s in enumerate(ALL_SEQS)}
N   = len(ALL_SEQS)
print(f"Espace : {N} séquences")

# Voisins (distance Hamming = 1) — précalculés une fois
def neighbors_of(s):
    nb = []
    for pos in range(SEQ_L):
        for v in range(V_TOK):
            if v != s[pos]:
                ns = list(s); ns[pos] = v; nb.append(tuple(ns))
    return nb

print("Pré-calcul des voisins...", end=" ", flush=True)
NEIGHBORS = [neighbors_of(s) for s in ALL_SEQS]
print("ok")

# ── Récompenses de base ───────────────────────────────────────────────────────
TARGET_SMOOTH = (2, 4, 1, 3, 0, 2)

def r_smooth(s):
    d = sum(a != b for a, b in zip(s, TARGET_SMOOTH))
    return math.exp(-d)

# Bruit fixe (même pour tous les α → reproductible)
rng_noise = np.random.default_rng(999)
R_NOISE = rng_noise.uniform(0.01, 1.0, N).astype(np.float32)

R_SMOOTH = np.array([r_smooth(s) for s in ALL_SEQS], dtype=np.float32)

# ── Corrélation locale ρ(R) ───────────────────────────────────────────────────
def local_corr(R_vec):
    """Pearson(R(x), mean_{x' ∈ N(x)} R(x')) sur tout l'espace."""
    nb_mean = np.array([R_vec[[IDX[nb] for nb in NEIGHBORS[i]]].mean()
                        for i in range(N)], dtype=np.float64)
    rho = np.corrcoef(R_vec.astype(np.float64), nb_mean)[0, 1]
    return float(rho)

print("Calcul de ρ(R_lisse) et ρ(R_bruit)...", end=" ", flush=True)
rho_smooth = local_corr(R_SMOOTH)
rho_noise  = local_corr(R_NOISE)
print(f"ρ_lisse={rho_smooth:.3f}  ρ_bruit={rho_noise:.3f}")

# ── Architecture GFN (identique à gflownet_regime.py) ────────────────────────
DIM_IN = SEQ_L + SEQ_L * V_TOK

class Policy(nn.Module):
    def __init__(self, h=64):
        super().__init__()
        self.net   = nn.Sequential(nn.Linear(DIM_IN, h), nn.ReLU(),
                                   nn.Linear(h, h),      nn.ReLU(),
                                   nn.Linear(h, V_TOK))
        self.log_Z = nn.Parameter(torch.zeros(()))
    def forward(self, x):
        return self.net(x)

def encode(partial):
    v = np.zeros(DIM_IN, dtype=np.float32)
    v[len(partial)] = 1.0
    for i, tok in enumerate(partial):
        v[SEQ_L + i * V_TOK + tok] = 1.0
    return v

def rollout(pol, B, eps, R_vec):
    partials = [[] for _ in range(B)]
    lpf      = torch.zeros(B)
    for t in range(SEQ_L):
        states = torch.tensor(np.stack([encode(p) for p in partials]))
        lp     = torch.log_softmax(pol(states), dim=-1)
        if np.random.rand() < eps:
            a = torch.randint(0, V_TOK, (B,))
        else:
            a = torch.multinomial(lp.exp(), 1).squeeze(1)
        lpf = lpf + lp.gather(1, a[:, None]).squeeze(1)
        for b in range(B):
            partials[b].append(int(a[b]))
    seqs = [tuple(p) for p in partials]
    R    = torch.tensor([float(R_vec[IDX[s]]) for s in seqs],
                        dtype=torch.float32).clamp(min=1e-6)
    return seqs, lpf, R

N_EVAL  = 4000
B_TRAIN = 64
N_SEEDS = 3

def run_gfn(seed, R_vec, n_eval=N_EVAL):
    torch.manual_seed(seed); np.random.seed(seed)
    pol = Policy(); opt = optim.Adam(pol.parameters(), lr=5e-3)
    found, used = set(), 0
    while used < n_eval:
        eps = max(0.05, 0.4 * (1 - used / (n_eval * 0.5)))
        seqs, lpf, R = rollout(pol, B_TRAIN, eps, R_vec)
        loss = ((pol.log_Z + lpf - torch.log(R)) ** 2).mean()
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(pol.parameters(), 1.0)
        opt.step()
        for s in seqs:
            found.add(s); used += 1
            if used >= n_eval: break
    return found

def run_uni(seed, n_eval=N_EVAL):
    rng = np.random.default_rng(seed)
    found = set()
    for _ in range(n_eval):
        s = tuple(int(rng.integers(V_TOK)) for _ in range(SEQ_L))
        found.add(s)
    return found

def coverage(found, modes):
    return sum(1 for s in modes if s in found) / max(len(modes), 1)

# ── Campagne sur 10 niveaux de α ──────────────────────────────────────────────
ALPHAS = np.linspace(0.0, 1.0, 11)   # 0.0, 0.1, ..., 1.0

rho_list, gain_list, gain_std_list = [], [], []
cov_G_list, cov_U_list = [], []

print(f"\n{'α':>5}  {'ρ(R_α)':>8}  {'GFN cov':>9}  {'UNI cov':>9}  {'Gain':>7}")
print("-" * 48)

for alpha in ALPHAS:
    # Récompense mixte
    R_alpha = (alpha * R_SMOOTH + (1 - alpha) * R_NOISE).astype(np.float32)
    # Normalise dans [eps, 1]
    R_alpha = R_alpha / (R_alpha.max() + 1e-9)

    # Modes = top 1%
    thr   = np.quantile(R_alpha, 0.99)
    modes = {ALL_SEQS[i] for i, r in enumerate(R_alpha) if r >= thr}

    rho = local_corr(R_alpha)

    cov_G, cov_U = [], []
    for seed in range(N_SEEDS):
        found_G = run_gfn(seed, R_alpha)
        found_U = run_uni(seed)
        cov_G.append(coverage(found_G, modes))
        cov_U.append(coverage(found_U, modes))

    gain = np.array(cov_G).mean() / max(np.array(cov_U).mean(), 0.01)
    gain_se = np.array(cov_G).std() / max(np.array(cov_U).mean(), 0.01) / np.sqrt(N_SEEDS)

    rho_list.append(rho)
    gain_list.append(gain)
    gain_std_list.append(gain_se)
    cov_G_list.append(np.array(cov_G))
    cov_U_list.append(np.array(cov_U))

    print(f"{alpha:5.2f}  {rho:8.3f}  {np.array(cov_G).mean():9.3f}  "
          f"{np.array(cov_U).mean():9.3f}  {gain:7.3f}")

rho_arr  = np.array(rho_list)
gain_arr = np.array(gain_list)
gain_std = np.array(gain_std_list)

# ── Ajustement linéaire Gain = a*ρ + b ───────────────────────────────────────
A = np.column_stack([rho_arr, np.ones_like(rho_arr)])
coeffs, residuals, _, _ = np.linalg.lstsq(A, gain_arr, rcond=None)
a, b = coeffs
gain_pred = a * rho_arr + b
ss_tot = ((gain_arr - gain_arr.mean()) ** 2).sum()
ss_res = ((gain_arr - gain_pred) ** 2).sum()
R2 = 1 - ss_res / (ss_tot + 1e-12)

print(f"\nAjustement linéaire Gain = {a:.3f}·ρ + {b:.3f}  (R² = {R2:.3f})")
print(f"Interprétation : +0.1 de ρ → ×{1 + a * 0.1 / max(b, 0.01):.2f} de gain")

# ── Figure ────────────────────────────────────────────────────────────────────
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Scatter Gain vs ρ
ax = axes[0]
ax.errorbar(rho_arr, gain_arr, yerr=gain_std,
            fmt="o", color="#4caf84", ms=7, lw=1.5, capsize=4,
            label="GFN/Uniforme (mesure)")
rho_line = np.linspace(rho_arr.min(), rho_arr.max(), 100)
ax.plot(rho_line, a * rho_line + b, "--", color="#ff8c42", lw=2,
        label=f"Gain = {a:.2f}·ρ + {b:.2f}  (R²={R2:.2f})")
ax.axhline(1.0, c="gray", ls=":", lw=1, label="Gain=1 (parité)")
ax.set_xlabel("Corrélation locale ρ(R)", fontsize=11)
ax.set_ylabel("Gain = Coverage_GFN / Coverage_Uniforme", fontsize=11)
ax.set_title(f"Loi de régime quantitative\nGain ∝ ρ(R)", fontsize=11)
ax.legend(fontsize=9)

# Courbes de coverage pour α ∈ {0, 0.5, 1.0}
ax = axes[1]
shown = [0, 5, 10]  # indices dans ALPHAS
colors = ["#7aa2ff", "#4caf84", "#ff8c42"]
for i, (idx, col) in enumerate(zip(shown, colors)):
    alpha = ALPHAS[idx]
    rho   = rho_list[idx]
    cG    = cov_G_list[idx].mean()
    cU    = cov_U_list[idx].mean()
    ax.bar([3*i + 0], [cG], color=col, alpha=0.85, width=0.8,
           label=f"α={alpha:.1f}  ρ={rho:.2f}")
    ax.bar([3*i + 1], [cU], color=col, alpha=0.35, width=0.8,
           label=f"_Uniforme")
    ax.text(3*i + 0, cG + 0.005, f"GFN\n{cG:.2f}", ha="center", fontsize=7)
    ax.text(3*i + 1, cU + 0.005, f"UNI\n{cU:.2f}", ha="center", fontsize=7)

ax.set_xticks([0.5, 3.5, 6.5])
ax.set_xticklabels([f"α={ALPHAS[i]:.1f}\nρ={rho_list[i]:.2f}" for i in shown])
ax.set_ylabel("Coverage (top 1%)")
ax.set_title("Coverage GFN (foncé) vs Uniforme (clair)\npour 3 niveaux de ρ", fontsize=11)

plt.suptitle(
    "Loi de régime : Gain_GFN = f(ρ(R))  —  preuve expérimentale\n"
    f"Espace {V_TOK}^{SEQ_L}={N} séquences, {N_SEEDS} seeds × {N_EVAL} éval., "
    f"11 niveaux de α",
    fontsize=10, y=1.02
)
plt.tight_layout()
out = "/home/user/curly-enigma/gflownet_regime_law.png"
plt.savefig(out, dpi=130, bbox_inches="tight")
print(f"\nFigure → {out}")
