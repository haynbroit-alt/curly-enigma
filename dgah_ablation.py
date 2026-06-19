"""
DGAH — Ablation sur LFR Benchmark
==================================
Question : le terme de courbure Forman + rewiring apporte-t-il
           un gain mesurable par rapport à la boucle sans géométrie ?

Protocole :
  - Dataset : LFR benchmark (graphe communautaire réaliste,
    distribution de degrés en loi de puissance — standard NeurIPS/ICLR)
  - Conditions (3 runs × 3 seeds chacune) :
      A) DGAH complet   (prédiction + courbure + MDL)
      B) Sans courbure  (prédiction + MDL, pas de rewiring)
      C) GCN-like baseline  (diffusion laplacienne pure, dim fixe)
  - Métriques :
      · NMI (Normalized Mutual Information) avec les vraies communautés
      · Erreur de prédiction BCE finale
      · Dimensions actives finales
      · Pas pour atteindre 80 % du NMI max (vitesse de convergence)
"""
import numpy as np
import time
from itertools import permutations
import warnings
warnings.filterwarnings("ignore")

rng_global = np.random.default_rng(42)

# -----------------------------------------------------------------------
# 1. LFR Benchmark (via networkx)
# -----------------------------------------------------------------------
def build_lfr(seed=0):
    """
    LFR : degree power-law + community structure.
    Retourne A (matrice d'adj), labels vrais, N.
    """
    import networkx as nx
    from networkx.generators.community import LFR_benchmark_graph
    G = LFR_benchmark_graph(
        n=200, tau1=2.5, tau2=1.5,
        mu=0.2,                          # mixing parameter (0=pure, 1=aléatoire)
        min_degree=5, max_degree=25,
        min_community=20, max_community=60,
        seed=seed
    )
    nodes = sorted(G.nodes())
    N = len(nodes)
    idx = {v: i for i, v in enumerate(nodes)}
    A = np.zeros((N, N))
    for u, v in G.edges():
        A[idx[u], idx[v]] = A[idx[v], idx[u]] = 1.0
    # communautés vraies (frozensets dans les attributs)
    raw = [G.nodes[v]["community"] for v in nodes]
    unique = sorted({frozenset(c) for c in raw})
    comm_map = {c: i for i, c in enumerate(unique)}
    labels = np.array([comm_map[frozenset(raw[i])] for i in range(N)])
    return A, labels, N

# -----------------------------------------------------------------------
# 2. Métriques
# -----------------------------------------------------------------------
def nmi(labels_true, labels_pred):
    from sklearn.metrics import normalized_mutual_info_score
    return normalized_mutual_info_score(labels_true, labels_pred, average_method="arithmetic")

def kmeans_labels(Z, k, rng, n_restarts=5):
    best_inertia, best_lab = np.inf, None
    for _ in range(n_restarts):
        c = Z[rng.choice(len(Z), k, replace=False)]
        lab = np.zeros(len(Z), int)
        for _ in range(30):
            d = ((Z[:, None] - c[None]) ** 2).sum(2)
            lab = d.argmin(1)
            for j in range(k):
                if (lab == j).any(): c[j] = Z[lab == j].mean(0)
        inertia = sum(((Z[lab == j] - c[j]) ** 2).sum() for j in range(k) if (lab == j).any())
        if inertia < best_inertia:
            best_inertia, best_lab = inertia, lab.copy()
    return best_lab

# -----------------------------------------------------------------------
# 3. Briques DGAH (réutilisées de dgah.py, adaptées au LFR)
# -----------------------------------------------------------------------
def sigmoid(x): return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))

def forman_curvature(W):
    deg = W.sum(1)
    tri = (W @ W) * W
    Wi, Wj = np.where(np.triu(W, 1) > 0)
    if len(Wi) == 0:
        return Wi, Wj, np.array([])
    kappa = 4 - deg[Wi] - deg[Wj] + 3 * tri[Wi, Wj]
    return Wi, Wj, kappa

def rewire(W, n_changes, rng):
    Wi, Wj, kappa = forman_curvature(W)
    if len(kappa) == 0: return W
    order = np.argsort(kappa)
    for idx in order[:n_changes]:
        u, v = Wi[idx], Wj[idx]
        nu = np.where(W[u] > 0)[0]; nv = np.where(W[v] > 0)[0]
        cand = [(a, b) for a in nu for b in nv if a != b and W[a, b] == 0]
        if cand:
            a, b = cand[rng.integers(len(cand))]
            W[a, b] = W[b, a] = 1
    for idx in order[::-1][:n_changes]:
        u, v = Wi[idx], Wj[idx]
        if W.sum(1)[u] > 2 and W.sum(1)[v] > 2:
            W[u, v] = W[v, u] = 0
    return W

def make_pred_fns(A, N):
    iu = np.triu_indices(N, 1)
    M_pairs = N * (N - 1) / 2
    w_pos = (A[iu] == 0).sum() / max(1, (A[iu] == 1).sum())

    def pred_error(Z):
        S = Z @ Z.T
        Phat = sigmoid(S)
        p = np.clip(Phat[iu], 1e-6, 1 - 1e-6); a = A[iu]
        bce = -(w_pos * a * np.log(p) + (1 - a) * np.log(1 - p)).mean()
        G = Phat * (1 + (w_pos - 1) * A) - w_pos * A
        np.fill_diagonal(G, 0.0); G /= M_pairs
        gradZ = (G + G.T) @ Z
        return bce, gradZ

    def mdl_prune(Z, floor=2):
        d = Z.shape[1]
        if d <= floor: return Z
        base, _ = pred_error(Z)
        base_bits = 2 * M_pairs * base
        worst, gain = None, 0.0
        for j in range(d):
            Zc = Z.copy(); Zc[:, j] = 0.0
            bce, _ = pred_error(Zc)
            delta = 2 * M_pairs * bce - base_bits
            param_bits = 0.3 * N * 0.5 * np.log(M_pairs)
            if delta < param_bits and (param_bits - delta) > gain:
                gain, worst = param_bits - delta, j
        if worst is not None:
            Z = np.delete(Z, worst, axis=1)
        return Z

    return pred_error, mdl_prune, M_pairs

# -----------------------------------------------------------------------
# 4. Trois conditions expérimentales
# -----------------------------------------------------------------------
def run_condition(condition, A, labels, N, K, seed, steps=400):
    rng = np.random.default_rng(seed)
    pred_error, mdl_prune, M_pairs = make_pred_fns(A, N)

    d0 = 12
    Z = 0.05 * rng.standard_normal((N, d0))
    W = np.zeros((N, N))
    S = Z @ Z.T
    for i in range(N):
        for j in np.argsort(-S[i])[1:6]:
            W[i, j] = W[j, i] = 1

    lr, eta_g, mom = 1.5, 0.008, 0.9
    vel = np.zeros_like(Z)

    nmi_hist, bce_hist, d_hist = [], [], []
    step_80 = steps  # pas pour atteindre 80% NMI max

    for t in range(steps):
        anneal = max(0.15, 1 - t / steps)
        bce, gradZ = pred_error(Z)
        vel = mom * vel - lr * gradZ
        Z = Z + vel - 1e-3 * Z

        if condition in ("dgah", "no_curv", "curv_only"):
            deg = W.sum(1)
            L = np.diag(deg) - W
            Z = Z - eta_g * anneal * (L @ Z) / (deg.mean() + 1e-9)

        if t % 10 == 0:
            if condition in ("dgah", "curv_only"):
                W = rewire(W, 3, rng)
            if condition in ("dgah", "no_curv"):
                Z = mdl_prune(Z)
                if vel.shape[1] != Z.shape[1]:
                    vel = np.zeros_like(Z)

        lab = kmeans_labels(Z, K, rng, n_restarts=3)
        score = nmi(labels, lab)
        bce_now, _ = pred_error(Z)

        nmi_hist.append(score)
        bce_hist.append(bce_now)
        d_hist.append(Z.shape[1])

    nmi_max = max(nmi_hist)
    thr = 0.8 * nmi_max
    for t, s in enumerate(nmi_hist):
        if s >= thr:
            step_80 = t
            break

    return dict(nmi=nmi_hist, bce=bce_hist, d=d_hist,
                nmi_final=nmi_hist[-1], bce_final=bce_hist[-1],
                d_final=d_hist[-1], step_80=step_80)

# -----------------------------------------------------------------------
# 5. Exécution multi-seed
# -----------------------------------------------------------------------
print("=== DGAH Ablation — LFR Benchmark ===\n")
SEEDS = [0, 1, 2]
CONDITIONS = ["dgah", "no_curv", "curv_only", "gcn"]
LABELS_FR = {
    "dgah":      "DGAH complet  (courbure + MDL)",
    "no_curv":   "Sans courbure (MDL seul)",
    "curv_only": "Courbure seule (pas de MDL)",
    "gcn":       "Baseline GCN  (diffusion fixe)"
}

results = {c: [] for c in CONDITIONS}
all_nmi_curves = {c: [] for c in CONDITIONS}

for seed in SEEDS:
    print(f"  Seed {seed} : construction LFR...", end=" ", flush=True)
    t0 = time.time()
    A, labels, N = build_lfr(seed)
    K = len(np.unique(labels))
    print(f"N={N}, K={K}, E={int(A.sum()//2)} ({time.time()-t0:.1f}s)")
    for cond in CONDITIONS:
        r = run_condition(cond, A, labels, N, K, seed=seed*100)
        results[cond].append(r)
        all_nmi_curves[cond].append(r["nmi"])

# -----------------------------------------------------------------------
# 6. Résultats
# -----------------------------------------------------------------------
print("\n" + "="*62)
print(f"{'Condition':<30} {'NMI final':>9} {'BCE final':>9} {'d final':>7} {'Pas→80%':>8}")
print("-"*62)
for cond in CONDITIONS:
    nmi_f  = np.mean([r["nmi_final"] for r in results[cond]])
    nmi_s  = np.std( [r["nmi_final"] for r in results[cond]])
    bce_f  = np.mean([r["bce_final"] for r in results[cond]])
    d_f    = np.mean([r["d_final"]   for r in results[cond]])
    s80    = np.mean([r["step_80"]   for r in results[cond]])
    print(f"{LABELS_FR[cond]:<30} {nmi_f:.3f}±{nmi_s:.3f}  {bce_f:>8.4f}  {d_f:>6.1f}  {s80:>7.0f}")
print("="*62)

# delta NMI DGAH vs sans courbure
nmi_dgah   = [r["nmi_final"] for r in results["dgah"]]
nmi_nocurv = [r["nmi_final"] for r in results["no_curv"]]
nmi_gcn       = [r["nmi_final"] for r in results["gcn"]]
nmi_curvonly  = [r["nmi_final"] for r in results["curv_only"]]
delta_curv    = np.mean(nmi_dgah)    - np.mean(nmi_nocurv)
delta_base    = np.mean(nmi_dgah)    - np.mean(nmi_gcn)
delta_co_gcn  = np.mean(nmi_curvonly)- np.mean(nmi_gcn)
delta_co_nc   = np.mean(nmi_curvonly)- np.mean(nmi_nocurv)
print(f"\nEffet du rewiring Forman (courbure seule vs GCN)   : Δ NMI = {delta_co_gcn:+.3f}")
print(f"Effet du rewiring Forman (courbure seule vs no_curv): Δ NMI = {delta_co_nc:+.3f}")
print(f"Effet MDL seul (DGAH vs courbure seule)            : Δ NMI = {np.mean(nmi_dgah)-np.mean(nmi_curvonly):+.3f}")
print(f"Effet MDL seul (no_curv vs GCN)                    : Δ NMI = {np.mean(nmi_nocurv)-np.mean(nmi_gcn):+.3f}")
print(f"\nConclusion : MDL prune {np.mean([r['d'][0] for r in results['dgah']]):.0f} → "
      f"{np.mean([r['d_final'] for r in results['dgah']]):.1f} dims sur LFR "
      f"(sur-élagage potentiel : LFR a des communautés qui se chevauchent)")
print()

# -----------------------------------------------------------------------
# 7. Figure
# -----------------------------------------------------------------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

STEPS = 400
colors = {"dgah": "#4caf84", "no_curv": "#f4a261", "curv_only": "#e040fb", "gcn": "#7aa2ff"}
fig, axes = plt.subplots(1, 3, figsize=(13, 4))

# courbe NMI moyenne
ax = axes[0]
for cond in CONDITIONS:
    curves = np.array(all_nmi_curves[cond])
    if curves.shape[1] == 0: continue
    mu = curves.mean(0)
    sd = curves.std(0)
    xs = np.arange(len(mu))
    ax.plot(xs, mu, color=colors[cond], label=LABELS_FR[cond], lw=1.8)
    ax.fill_between(xs, mu - sd, mu + sd, alpha=0.18, color=colors[cond])
ax.set_title("NMI (communautés récupérées)", fontsize=11)
ax.set_xlabel("pas"); ax.set_ylabel("NMI")
ax.legend(fontsize=7.5); ax.set_ylim(0, 1)

# barres NMI final
ax = axes[1]
nmi_means = [np.mean([r["nmi_final"] for r in results[c]]) for c in CONDITIONS]
nmi_stds  = [np.std( [r["nmi_final"] for r in results[c]]) for c in CONDITIONS]
bars = ax.bar(range(len(CONDITIONS)), nmi_means, yerr=nmi_stds, capsize=6,
              color=[colors[c] for c in CONDITIONS], alpha=0.85, width=0.5)
ax.set_xticks(range(len(CONDITIONS)))
ax.set_xticklabels(["DGAH", "Sans curv.", "Curv. only", "GCN"], fontsize=8)
ax.set_title("NMI final (moyenne ± std, 3 seeds)", fontsize=11)
ax.set_ylabel("NMI"); ax.set_ylim(0, 1)
for bar, v in zip(bars, nmi_means):
    ax.text(bar.get_x() + bar.get_width()/2, v + 0.02, f"{v:.3f}", ha="center", fontsize=9)

# dimensions actives (MDL)
ax = axes[2]
for cond in ["dgah", "no_curv", "curv_only"]:
    d_curves = [r["d"] for r in results[cond]]
    mu = np.mean(d_curves, axis=0)
    ax.plot(mu, color=colors[cond], label=LABELS_FR[cond], lw=1.8)
ax.axhline(np.mean([r["d_final"] for r in results["gcn"]]),
           color=colors["gcn"], ls="--", lw=1.2, label="GCN (fixe)")
ax.set_title("Dimensions actives (MDL)", fontsize=11)
ax.set_xlabel("pas"); ax.set_ylabel("d")
ax.legend(fontsize=7.5)

plt.suptitle("Ablation DGAH — LFR Benchmark (N≈200, μ=0.2)", fontsize=12, y=1.02)
plt.tight_layout()
plt.savefig("/home/user/curly-enigma/dgah_ablation_results.png", dpi=120, bbox_inches="tight")
print("Figure → dgah_ablation_results.png")
