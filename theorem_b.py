"""
theorem_b.py — Validation expérimentale du théorème TB + mesure de base.

THÉORÈME (TB + base measure q) :
  Soit un GFlowNet autorégressif sur {0..K-1}^D, entraîné avec la loss :
    L(τ) = (log Z + log P_F(τ) - log q(x) - log R(x))²
  où q(x) > 0 est une mesure de base normalisée (Σ q = 1).
  À l'optimum : P_F(x) = q(x) · R(x) / Z_q   où Z_q = Σ_x q(x)·R(x).

PREUVE (TB flow-matching) :
  À l'optimum, L = 0 pour tout τ :
    log Z + log P_F(τ) = log q(x) + log R(x)
  Marginalisons sur les D! ordres possibles pour atteindre x.
  P_F(x) · Z = q(x) · R(x)  →  P_F(x) = q(x) · R(x) / Z_q.  □

CONSÉQUENCE :
  q = uniforme  → P_F(x) ∝ R(x)          (GFlowNet standard)
  q = LLM prior → P_F(x) ∝ q_LLM(x)·R(x) (bias multiplicatif garanti)

NOTE D'IMPLÉMENTATION :
  Avec des probabilités normalisées (Σ pR = Σ q = 1), la valeur optimale
  de log_Z est log(Σ q·pR) ≈ -log(N) ≈ -6.4, loin de l'init 0.
  On décale la target de +log(N) → log_Z_opt ≈ 0, convergence rapide.
"""
import numpy as np, torch, torch.nn as nn, itertools, math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

D, K = 4, 5                               # 5^4 = 625 états
STATES = list(itertools.product(range(K), repeat=D))
COORDS = np.array(STATES, dtype=float)
N      = len(STATES)
IDX    = {s: i for i, s in enumerate(STATES)}
LOG_N  = float(np.log(N))                 # ≈ 6.44 : décalage de normalisation

BUDGET = 2500
BATCH  = 40
A_LEVELS = [0.0, 0.5, 1.0]
L_SCALE  = 1.2
SEEDS    = [0, 1, 2]

print(f"Espace : {N} états  |  budget = {BUDGET}  |  l = {L_SCALE}")
print(f"Théorème : P_F(x) ∝ q(x)·R(x)/Z_q\n")

# ── GP-RBF ────────────────────────────────────────────────────────────────────
def gp_sample(l, seed):
    rng = np.random.default_rng(seed)
    sq  = ((COORDS[:, None, :] - COORDS[None, :, :]) ** 2).sum(-1)
    Cov = np.exp(-sq / (2 * l**2)) + 1e-4 * np.eye(N)
    return np.linalg.cholesky(Cov) @ rng.standard_normal(N)

def to_prob(g, scale=1.5):
    v = np.exp(scale * (g - g.mean()) / (g.std() + 1e-9))
    return v / v.sum()

def orthogonalize(g_n, g_r):
    g_r = g_r - g_r.mean(); g_n = g_n - g_n.mean()
    g_n -= (g_n @ g_r / (g_r @ g_r + 1e-9)) * g_r
    g_n /= np.linalg.norm(g_n) + 1e-9
    g_r /= np.linalg.norm(g_r) + 1e-9
    return g_n, g_r

def make_prior(g_r_unit, g_n, A):
    g_p = A * g_r_unit + (1 - A) * g_n
    g_p /= np.linalg.norm(g_p) + 1e-9
    # log q(x) = log-normalized prior
    lv = 1.5 * g_p
    lv -= float(np.log(np.exp(lv).sum()))
    return lv   # log q(x), Σ exp(lv) = 1

# ── Architecture ──────────────────────────────────────────────────────────────
def enc(partial):
    v = np.zeros(D + D * K, dtype=np.float32)
    v[len(partial)] = 1.0
    for i, c in enumerate(partial):
        v[D + i * K + c] = 1.0
    return v

class Net(nn.Module):
    def __init__(self, h=128):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(D + D * K, h), nn.ReLU(),
                                  nn.Linear(h, h), nn.ReLU(), nn.Linear(h, K))
        self.log_Z = nn.Parameter(torch.zeros(()))
    def forward(self, x): return self.body(x)

def logprobs(net, partials):
    return torch.log_softmax(
        net(torch.tensor(np.stack([enc(p) for p in partials]))), -1)

def exact_dist(net):
    logp = torch.zeros(N)
    with torch.no_grad():
        for t in range(D):
            prefixes = [list(s[:t]) for s in STATES]
            lp  = logprobs(net, prefixes)
            nxt = torch.tensor([s[t] for s in STATES])
            logp += lp.gather(1, nxt[:, None]).squeeze(1)
    p = logp.exp().numpy(); return p / p.sum()

# ── Entraînement ──────────────────────────────────────────────────────────────
def train(pR, log_q, train_seed):
    """
    log_q : log q(x) pour chaque état (Σ exp = 1).
             None → GFN standard (q uniforme, log q = -log N partout).

    Loss TB + base measure :
      L = (log Z + log P_F(τ) − [log q(x) + log R(x)])²

    Décalage de normalisation : on ajoute LOG_N au target pour que
    log_Z_opt ≈ 0 (même échelle que l'initialisation).
    """
    torch.manual_seed(train_seed); np.random.seed(train_seed)
    net = Net(); opt = torch.optim.Adam(net.parameters(), lr=5e-3)
    pR_lu = {s: float(pR[i]) for i, s in enumerate(STATES)}
    lq_lu = ({s: float(log_q[i]) for i, s in enumerate(STATES)}
             if log_q is not None else None)
    used = 0
    while used < BUDGET:
        eps  = max(0.05, 0.4 * (1 - used / (BUDGET * 0.6)))
        partials = [[] for _ in range(BATCH)]
        lpf = torch.zeros(BATCH)
        for t in range(D):
            lp = logprobs(net, partials)
            a  = (torch.randint(0, K, (BATCH,)) if np.random.rand() < eps
                  else torch.multinomial(lp.exp(), 1).squeeze(1))
            lpf = lpf + lp.gather(1, a[:, None]).squeeze(1)
            for b in range(BATCH): partials[b].append(int(a[b]))

        log_R = torch.tensor([math.log(pR_lu[tuple(s)] + 1e-30)
                               for s in partials], dtype=torch.float32)

        if lq_lu is not None:
            # TB + base measure : target = log q(x) + log R(x) + LOG_N
            # Le +LOG_N ré-centre l'échelle : log_Z_opt ≈ log(N * Σ q·R) ≈ 0
            log_q_b = torch.tensor([lq_lu[tuple(s)] for s in partials],
                                   dtype=torch.float32)
            target = log_q_b + log_R + LOG_N
        else:
            # Standard : target = log R(x)  →  log_Z_opt = log(Σ pR) = 0
            target = log_R

        loss = ((net.log_Z + lpf - target) ** 2).mean()
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        used += BATCH
    return net

# ── Campagne ──────────────────────────────────────────────────────────────────
# Pour chaque A, on compare :
#   TV(GFN_std, pR)   — GFN standard vs sa cible
#   TV(GFN_q,  pqR)   — GFN+q vs sa cible théorique q·R/Z_q
# Les deux devraient être petits (théorème validé).

print(f"{'A':>5} | {'TV(GFN_std, pR)':>17} | {'TV(GFN_q, q·R/Z)':>18}")
print("-" * 47)

results = {}

for A in A_LEVELS:
    tvs_std, tvs_q = [], []
    for seed in SEEDS:
        g_r = gp_sample(L_SCALE, seed)
        g_n = gp_sample(L_SCALE, seed + 50)
        g_n_orth, g_r_unit = orthogonalize(g_n, g_r.copy())

        pR    = to_prob(g_r)
        log_q = make_prior(g_r_unit, g_n_orth, A)

        # Cible théorique : q(x)·R(x)/Z_q
        pqR = np.exp(log_q) * pR
        pqR /= pqR.sum()

        net_std = train(pR, log_q=None,  train_seed=seed)
        pth_std = exact_dist(net_std)
        tv_std  = 0.5 * np.abs(pth_std - pR).sum()

        net_q   = train(pR, log_q=log_q, train_seed=seed)
        pth_q   = exact_dist(net_q)
        tv_q    = 0.5 * np.abs(pth_q - pqR).sum()

        tvs_std.append(tv_std)
        tvs_q.append(tv_q)

    results[(A, "std")] = tvs_std
    results[(A, "q")]   = tvs_q
    ms = np.mean(tvs_std); mq = np.mean(tvs_q)
    ss = np.std(tvs_std);  sq = np.std(tvs_q)
    print(f"{A:5.1f} | {ms:.4f} ± {ss:.4f}        | {mq:.4f} ± {sq:.4f}")

# ── Résumé du théorème ────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("CONCLUSION DU THÉORÈME :")
print("  Les deux TV doivent être petits et comparables.")
print("  TV(GFN+q, q·R) ≈ TV(GFN_std, R) → théorème validé.")
print()

all_std = [v for A in A_LEVELS for v in results[(A, "std")]]
all_q   = [v for A in A_LEVELS for v in results[(A, "q")]]
print(f"  TV_std (moyen sur tous A,seeds) : {np.mean(all_std):.4f} ± {np.std(all_std):.4f}")
print(f"  TV_q   (moyen sur tous A,seeds) : {np.mean(all_q):.4f} ± {np.std(all_q):.4f}")
ratio = np.mean(all_q) / (np.mean(all_std) + 1e-9)
print(f"  Ratio TV_q / TV_std : {ratio:.3f}  (proche de 1 = théorème validé)")

# ── Démonstration visuelle pour A=1.0, seed 0 ────────────────────────────────
g_r   = gp_sample(L_SCALE, 0)
g_n   = gp_sample(L_SCALE, 50)
g_n_o, g_r_u = orthogonalize(g_n, g_r.copy())
pR    = to_prob(g_r)
log_q = make_prior(g_r_u, g_n_o, A=1.0)
pqR   = np.exp(log_q) * pR; pqR /= pqR.sum()
q_arr = np.exp(log_q)

net_std = train(pR, log_q=None,  train_seed=0)
net_q   = train(pR, log_q=log_q, train_seed=0)
pth_std = exact_dist(net_std)
pth_q   = exact_dist(net_q)

# ── Figure ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Gauche : TV par agent (barres groupées par A)
ax = axes[0]
x = np.arange(len(A_LEVELS)); w = 0.3
for j, (agent, color, lbl) in enumerate([
        ("std", "#7aa2ff", "GFN standard → cible R/Z"),
        ("q",   "#4caf84", "GFN+q → cible q·R/Z_q  [théorème]")]):
    means = [np.mean(results[(A, agent)]) for A in A_LEVELS]
    stds  = [np.std(results[(A, agent)])  for A in A_LEVELS]
    bars  = ax.bar(x + (j - 0.5) * w, means, w, yerr=stds,
                   capsize=4, color=color, alpha=0.85, label=lbl)
    for xi, (m, s) in zip(x + (j - 0.5) * w, zip(means, stds)):
        ax.text(xi, m + s + 0.003, f"{m:.3f}", ha="center", fontsize=7)
ax.set_xticks(x)
ax.set_xticklabels([f"A={a:.1f}" for a in A_LEVELS])
ax.set_ylabel("TV(p_θ, cible_théorique)")
ax.set_ylim(0, None)
ax.legend(fontsize=8.5)
ax.set_title("Validation du théorème\n"
             "TV vers la cible du théorème (bas = validé)", fontsize=10)

# Centre : distributions empilées (A=1.0, seed 0)
ax = axes[1]
idx_sorted = np.argsort(pqR)[-80:]   # top 80 états
xs = np.arange(len(idx_sorted))
ax.bar(xs - 0.2, pR[idx_sorted],    width=0.18, color="#e74c3c",  alpha=0.7, label="R/Z (reward)")
ax.bar(xs,       q_arr[idx_sorted], width=0.18, color="#9b59b6",  alpha=0.7, label="q (prior, ×5 pour vis.)")
ax.bar(xs + 0.2, pqR[idx_sorted],   width=0.18, color="#2ecc71",  alpha=0.9, label="q·R/Z_q (cible)")
ax.bar(xs + 0.4, pth_q[idx_sorted], width=0.18, color="#f39c12",  alpha=0.9, label="GFN+q appris")
ax.set_xlabel("États (triés par q·R/Z_q)")
ax.set_ylabel("Probabilité")
ax.set_title(f"A=1.0 : distributions (top 80 états)\n"
             f"TV(GFN+q, q·R) = {0.5*np.abs(pth_q-pqR).sum():.4f}", fontsize=10)
ax.legend(fontsize=7.5)

# Droite : scatter GFN+q vs cible (A=1.0, seed 0)
ax = axes[2]
top_idx = np.argsort(pqR)[-30:]
ax.scatter(pqR, pth_q, s=6, alpha=0.3, c="#4caf84")
ax.scatter(pqR[top_idx], pth_q[top_idx], s=40, c="red", zorder=5, label="top 30 états")
mn = min(pqR.min(), pth_q[pth_q > 0].min()); mx = max(pqR.max(), pth_q.max())
ax.plot([mn, mx], [mn, mx], "k--", lw=1, label="parfait")
corr = np.corrcoef(pqR, pth_q)[0, 1]
ax.set_xlabel("Cible q(x)·R(x)/Z_q", fontsize=10)
ax.set_ylabel("Appris P_F(x)", fontsize=10)
ax.set_title(f"GFN+q vs cible théorique\nTV={0.5*np.abs(pth_q-pqR).sum():.4f}  "
             f"corr={corr:.4f}", fontsize=10)
ax.legend(fontsize=9)

plt.suptitle(
    "Théorème TB + base measure  :  P_F(x) ∝ q(x)·R(x)  — validation expérimentale\n"
    f"Espace {K}^{D}={N} états, l={L_SCALE}, budget={BUDGET}, "
    f"{len(SEEDS)} seeds × {len(A_LEVELS)} niveaux d'alignement",
    fontsize=10, y=1.02
)
plt.tight_layout()
out = "/home/user/curly-enigma/theorem_b.png"
plt.savefig(out, dpi=130, bbox_inches="tight")
print(f"\nFigure → {out}")
