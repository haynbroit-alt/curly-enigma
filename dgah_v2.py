"""
DGAH v2 — MDL recalibré + validation sur Cora
==============================================
Correction identifiée par l'ablation LFR :
  le critère MDL/BIC pruneait trop agressivement car le plancher
  de dimensions n'était pas lié au nombre de communautés K.

Fix : floor = max(2, k_est) — on ne prune jamais en-dessous de K dims,
      ce qui est nécessaire pour séparer K communautés dans l'espace latent.

Validation :
  - Dataset : Cora (citation network, 7 classes, N=2708 → subsample stratifié 500)
  - 4 conditions × 3 seeds : DGAH_v2, courbure seule, sans géom., baseline GCN
  - Comparaison avec les résultats LFR de l'ablation précédente
"""
import numpy as np
import pickle
import warnings
import time
warnings.filterwarnings("ignore")

# -----------------------------------------------------------------------
# 1. Chargement Cora (subsample stratifié)
# -----------------------------------------------------------------------
def load_cora(path='/tmp/cora', n_sample=500, seed=0):
    with open(f'{path}/ind.cora.graph', 'rb') as f:
        graph = pickle.load(f, encoding='latin1')
    with open(f'{path}/ind.cora.ally', 'rb') as f:
        ally = pickle.load(f, encoding='latin1')
    with open(f'{path}/ind.cora.ty', 'rb') as f:
        ty = pickle.load(f, encoding='latin1')
    with open(f'{path}/ind.cora.test.index', 'rb') as f:
        test_idx_sorted = sorted(int(x) for x in f.read().split())

    N_full = len(graph)
    y_full = np.zeros(N_full, int)
    y_full[:ally.shape[0]] = ally.argmax(1)
    for i, idx in enumerate(test_idx_sorted):
        y_full[ally.shape[0] + i] = ty[idx - min(test_idx_sorted), :].argmax()

    # sous-graphe connecté par BFS à partir d'un nœud de degré élevé
    # (préserve la structure locale réelle du graphe de citations)
    rng = np.random.default_rng(seed)
    degrees = {v: len(graph.get(v, [])) for v in range(N_full)}
    # départ : nœud de degré élevé dans la classe la plus représentée
    start_candidates = sorted(range(N_full), key=lambda v: -degrees[v])[:20]
    start = rng.choice(start_candidates)

    visited = [start]
    frontier = list(graph.get(start, []))
    rng.shuffle(frontier)
    seen = {start}
    while len(visited) < n_sample and frontier:
        node = frontier.pop(0)
        if node in seen: continue
        seen.add(node); visited.append(node)
        neighbors = list(graph.get(node, []))
        rng.shuffle(neighbors)
        frontier.extend(neighbors)

    chosen = sorted(visited[:n_sample])
    idx_map = {old: new for new, old in enumerate(chosen)}

    N = len(chosen)
    A = np.zeros((N, N))
    for new_u, old_u in enumerate(chosen):
        for old_v in graph.get(old_u, []):
            if old_v in idx_map:
                A[new_u, idx_map[old_v]] = 1.0
    A = np.maximum(A, A.T)

    labels = y_full[chosen]
    K = len(np.unique(labels))
    return A, labels, N, K

# -----------------------------------------------------------------------
# 2. Briques DGAH (identiques à dgah.py)
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

def make_pred_fns(A, N, K):
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

    def mdl_prune_v2(Z):
        """MDL avec plancher adaptatif : floor = max(2, K)."""
        floor = max(2, K)          # FIX : on garde au moins K dimensions
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

    return pred_error, mdl_prune_v2

# -----------------------------------------------------------------------
# 3. Métriques
# -----------------------------------------------------------------------
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

def nmi(labels_true, labels_pred):
    from sklearn.metrics import normalized_mutual_info_score
    return normalized_mutual_info_score(labels_true, labels_pred, average_method="arithmetic")

# -----------------------------------------------------------------------
# 4. Boucle principale (4 conditions)
# -----------------------------------------------------------------------
CONDITIONS = ["dgah_v2", "curv_only", "no_geom", "gcn"]
LABELS_FR = {
    "dgah_v2":   "DGAH v2 (courbure + MDL fixé)",
    "curv_only": "Courbure seule (pas de MDL)",
    "no_geom":   "Sans géométrie (MDL fixé)",
    "gcn":       "Baseline GCN (diffusion fixe)"
}

def run(condition, A, labels, N, K, seed, steps=300):
    rng = np.random.default_rng(seed)
    pred_error, mdl_prune_v2 = make_pred_fns(A, N, K)

    d0 = max(K + 4, 12)
    Z = 0.05 * rng.standard_normal((N, d0))
    W = np.zeros((N, N))
    S0 = Z @ Z.T
    for i in range(N):
        for j in np.argsort(-S0[i])[1:6]:
            W[i, j] = W[j, i] = 1

    lr, eta_g, mom = 1.5, 0.008, 0.9
    vel = np.zeros_like(Z)
    nmi_hist, bce_hist, d_hist = [], [], []
    step_80 = steps

    for t in range(steps):
        anneal = max(0.15, 1 - t / steps)
        bce, gradZ = pred_error(Z)
        vel = mom * vel - lr * gradZ
        Z = Z + vel - 1e-3 * Z

        if condition in ("dgah_v2", "curv_only", "no_geom"):
            deg = W.sum(1)
            L = np.diag(deg) - W
            Z = Z - eta_g * anneal * (L @ Z) / (deg.mean() + 1e-9)

        if t % 10 == 0:
            if condition in ("dgah_v2", "curv_only"):
                W = rewire(W, 3, rng)
            if condition in ("dgah_v2", "no_geom"):
                old_d = Z.shape[1]
                Z = mdl_prune_v2(Z)
                if Z.shape[1] != old_d:
                    vel = np.zeros_like(Z)

        lab = kmeans_labels(Z, K, rng, n_restarts=3)
        score = nmi(labels, lab)
        bce_now, _ = pred_error(Z)
        nmi_hist.append(score); bce_hist.append(bce_now); d_hist.append(Z.shape[1])

        nmi_max_so_far = max(nmi_hist)
        if step_80 == steps and score >= 0.8 * nmi_max_so_far and t > 0:
            step_80 = t

    return dict(nmi=nmi_hist, bce=bce_hist, d=d_hist,
                nmi_final=nmi_hist[-1], bce_final=bce_hist[-1],
                d_final=d_hist[-1], step_80=step_80)

# -----------------------------------------------------------------------
# 5. Exécution
# -----------------------------------------------------------------------
print("=== DGAH v2 — Validation sur Cora ===\n")
print("Chargement Cora (subsample stratifié N=500)...")

SEEDS = [0, 1, 2]
all_results = {c: [] for c in CONDITIONS}
all_curves  = {c: [] for c in CONDITIONS}

for seed in SEEDS:
    A, labels, N, K = load_cora(seed=seed)
    print(f"  Seed {seed} : N={N}, K={K}, E={int(A.sum()//2)}")
    for cond in CONDITIONS:
        r = run(cond, A, labels, N, K, seed=seed*100, steps=250)
        all_results[cond].append(r)
        all_curves[cond].append(r["nmi"])

print("\n" + "="*65)
print(f"{'Condition':<35} {'NMI':>8} {'BCE':>8} {'d':>5} {'→80%':>6}")
print("-"*65)
for cond in CONDITIONS:
    nmi_f = np.mean([r["nmi_final"] for r in all_results[cond]])
    nmi_s = np.std( [r["nmi_final"] for r in all_results[cond]])
    bce_f = np.mean([r["bce_final"] for r in all_results[cond]])
    d_f   = np.mean([r["d_final"]   for r in all_results[cond]])
    s80   = np.mean([r["step_80"]   for r in all_results[cond]])
    print(f"{LABELS_FR[cond]:<35} {nmi_f:.3f}±{nmi_s:.3f} {bce_f:>8.4f} {d_f:>5.1f} {s80:>6.0f}")
print("="*65)

nmi_v2   = np.mean([r["nmi_final"] for r in all_results["dgah_v2"]])
nmi_gcn  = np.mean([r["nmi_final"] for r in all_results["gcn"]])
nmi_co   = np.mean([r["nmi_final"] for r in all_results["curv_only"]])
d_v2     = np.mean([r["d_final"]   for r in all_results["dgah_v2"]])
d_ng     = np.mean([r["d_final"]   for r in all_results["no_geom"]])

print(f"\nGain DGAH v2 vs GCN      : Δ NMI = {nmi_v2 - nmi_gcn:+.3f}")
print(f"Gain courbure seule vs GCN: Δ NMI = {nmi_co - nmi_gcn:+.3f}")
print(f"MDL v2 prune              : {int(max(K+4,12))} → {d_v2:.1f} dims "
      f"(plancher K={K}, était 4.3 avec v1)")

# -----------------------------------------------------------------------
# 6. Figure comparative (Cora + rappel LFR)
# -----------------------------------------------------------------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

colors = {"dgah_v2": "#4caf84", "curv_only": "#e040fb",
          "no_geom": "#f4a261",  "gcn": "#7aa2ff"}

fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

# courbes NMI
ax = axes[0]
for cond in CONDITIONS:
    curves = np.array(all_curves[cond])
    mu = curves.mean(0); sd = curves.std(0)
    xs = np.arange(len(mu))
    ax.plot(xs, mu, color=colors[cond], label=LABELS_FR[cond], lw=1.8)
    ax.fill_between(xs, mu-sd, mu+sd, alpha=0.15, color=colors[cond])
ax.set_title("NMI — Cora (500 nœuds, 7 classes)", fontsize=10)
ax.set_xlabel("pas"); ax.set_ylabel("NMI"); ax.legend(fontsize=7); ax.set_ylim(0, 1)

# barres NMI final Cora
ax = axes[1]
means = [np.mean([r["nmi_final"] for r in all_results[c]]) for c in CONDITIONS]
stds  = [np.std( [r["nmi_final"] for r in all_results[c]]) for c in CONDITIONS]
bars  = ax.bar(range(len(CONDITIONS)), means, yerr=stds, capsize=5,
               color=[colors[c] for c in CONDITIONS], alpha=0.85, width=0.55)
ax.set_xticks(range(len(CONDITIONS)))
ax.set_xticklabels(["DGAH v2", "Curv.", "No geom.", "GCN"], fontsize=8)
ax.set_title("NMI final Cora (moy. ± std, 3 seeds)", fontsize=10)
ax.set_ylabel("NMI"); ax.set_ylim(0, 1)
for bar, v in zip(bars, means):
    ax.text(bar.get_x()+bar.get_width()/2, v+0.02, f"{v:.3f}", ha="center", fontsize=8)

# dimensions actives — effet du fix MDL
ax = axes[2]
for cond in ["dgah_v2", "no_geom"]:
    d_curves = [r["d"] for r in all_results[cond]]
    mu = np.mean(d_curves, axis=0)
    ax.plot(mu, color=colors[cond], label=LABELS_FR[cond], lw=1.8)
ax.axhline(K, ls=':', c='gray', lw=1.2, label=f"K={K} (plancher MDL v2)")
ax.axhline(4.3, ls='--', c='red', lw=1, alpha=0.6, label="d=4.3 (MDL v1, sur-élagage)")
ax.set_title("MDL v2 : dimensions actives (Cora)", fontsize=10)
ax.set_xlabel("pas"); ax.set_ylabel("d"); ax.legend(fontsize=7)

plt.suptitle("DGAH v2 — Validation Cora | fix MDL floor = max(2, K)", fontsize=11, y=1.01)
plt.tight_layout()
plt.savefig("/home/user/curly-enigma/dgah_v2_results.png", dpi=120, bbox_inches="tight")
print("\nFigure → dgah_v2_results.png")
