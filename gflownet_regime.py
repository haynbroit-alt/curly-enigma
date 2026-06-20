"""
Preuve de la loi de régime : GFlowNet > uniforme si et seulement si
le paysage de récompense est apprenable (structure locale dans l'espace des tokens).

Deux tâches, même espace, même budget :
  A — récompense chaotique (arithmétique, cible=42)  → GFlowNet ≈ uniforme  (attendu)
  B — récompense lisse     (Hamming vers cible)       → GFlowNet > uniforme  (attendu)

La différence est LE mécanisme : en régime lisse, la politique apprend à généraliser
des zones vues vers les zones non vues. En régime chaotique, elle ne peut pas.
"""
import math, torch, torch.nn as nn, torch.optim as optim
import numpy as np
from itertools import product as iproduct

torch.manual_seed(0); np.random.seed(0)

# ── Espace commun : séquences de 6 tokens dans {0..4} (5^6 = 15625 états) ──
V_TOK  = 5     # taille du vocabulaire de contenu (tokens 0..4)
SEQ_L  = 6     # longueur des séquences
N_EVAL = 4000  # budget par seed

ALL_SEQS = list(iproduct(range(V_TOK), repeat=SEQ_L))  # 15625 séquences
N_TOTAL  = len(ALL_SEQS)

# ── Tâche A : récompense chaotique (polynôme mod p, non local) ──────────────
# Simule l'irrégularité de l'arithmétique sans dépendre d'eval()
_P = 97  # nombre premier
TARGET_A = 42

def reward_A(seq):
    """Polynôme mod p — deux séquences voisines peuvent avoir des récompenses très différentes."""
    val = 0
    for i, t in enumerate(seq):
        val = (val * 11 + (t + 1) * (i + 3)) % _P
    return float(math.exp(-abs(val - TARGET_A) / 3.0))

# ── Tâche B : récompense lisse (distance de Hamming vers cible) ──────────────
TARGET_B = (2, 4, 1, 3, 0, 2)  # cible arbitraire dans l'espace

def reward_B(seq):
    """exp(-Hamming(seq, cible)) — 1-Lipschitz par flip → structure apprenable."""
    d = sum(a != b for a, b in zip(seq, TARGET_B))
    return float(math.exp(-d))

# Vérité terrain
R_A = np.array([reward_A(s) for s in ALL_SEQS])
R_B = np.array([reward_B(s) for s in ALL_SEQS])
Z_A, Z_B = R_A.sum(), R_B.sum()
P_A, P_B = R_A / Z_A, R_B / Z_B

# Modes = top 1% de chaque distribution
THR_A = np.quantile(R_A, 0.99); MODES_A = {s for s, r in zip(ALL_SEQS, R_A) if r >= THR_A}
THR_B = np.quantile(R_B, 0.99); MODES_B = {s for s, r in zip(ALL_SEQS, R_B) if r >= THR_B}
print(f"Espace : {N_TOTAL} séquences")
print(f"Tâche A (chaotique) : seuil={THR_A:.3f}, {len(MODES_A)} modes ({len(MODES_A)/N_TOTAL*100:.1f}%)")
print(f"Tâche B (lisse)     : seuil={THR_B:.3f}, {len(MODES_B)} modes ({len(MODES_B)/N_TOTAL*100:.1f}%)")

# ── Architecture : MLP simple (même pour les deux tâches) ────────────────────
DIM_IN = SEQ_L + SEQ_L * V_TOK   # position one-hot + tokens posés

class Policy(nn.Module):
    def __init__(self, h=64):
        super().__init__()
        self.net   = nn.Sequential(nn.Linear(DIM_IN, h), nn.ReLU(),
                                   nn.Linear(h, h),      nn.ReLU(),
                                   nn.Linear(h, V_TOK))
        self.log_Z = nn.Parameter(torch.zeros(()))

    def forward(self, x):  # [B, DIM_IN] → [B, V_TOK]
        return self.net(x)

def encode(partial):
    v = np.zeros(DIM_IN, dtype=np.float32)
    t = len(partial)
    v[t] = 1.0
    for i, tok in enumerate(partial):
        v[SEQ_L + i * V_TOK + tok] = 1.0
    return v

def rollout(policy, B, eps, reward_fn):
    partials  = [[] for _ in range(B)]
    sum_lpf   = torch.zeros(B)
    for t in range(SEQ_L):
        states = torch.tensor(np.stack([encode(p) for p in partials]))
        logits = policy(states)                      # [B, V_TOK]
        lp     = torch.log_softmax(logits, dim=-1)
        if np.random.rand() < eps:
            actions = torch.randint(0, V_TOK, (B,))
        else:
            actions = torch.multinomial(lp.exp(), 1).squeeze(1)
        sum_lpf = sum_lpf + lp.gather(1, actions[:, None]).squeeze(1)
        for b in range(B):
            partials[b].append(int(actions[b]))
    seqs    = [tuple(p) for p in partials]
    rewards = torch.tensor([reward_fn(s) for s in seqs],
                           dtype=torch.float32).clamp(min=1e-6)
    return seqs, sum_lpf, rewards

# ── Méthodes ──────────────────────────────────────────────────────────────────
B_TRAIN = 64

def run_gflownet(seed, reward_fn, n_eval=N_EVAL):
    torch.manual_seed(seed); np.random.seed(seed)
    pol = Policy(); opt = optim.Adam(pol.parameters(), lr=5e-3)
    found, used = set(), 0
    cov_curve = []
    while used < n_eval:
        eps = max(0.05, 0.4 * (1 - used / (n_eval * 0.5)))
        seqs, lpf, R = rollout(pol, B_TRAIN, eps, reward_fn)
        loss = ((pol.log_Z + lpf - torch.log(R)) ** 2).mean()
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(pol.parameters(), 1.0)
        opt.step()
        for s in seqs:
            found.add(s); used += 1
            if used % (n_eval // 20) == 0:
                cov_curve.append(used)
            if used >= n_eval: break
    return found, cov_curve

def run_uniform(seed, n_eval=N_EVAL):
    rng = np.random.default_rng(seed)
    found, cov_curve = set(), []
    for used in range(1, n_eval + 1):
        s = tuple(int(rng.integers(V_TOK)) for _ in range(SEQ_L))
        found.add(s)
        if used % (n_eval // 20) == 0:
            cov_curve.append(used)
    return found, cov_curve

def coverage(found, modes):
    return sum(1 for s in modes if s in found) / max(len(modes), 1)

def tv_distance(found, P_true, all_seqs):
    """Distance TV entre distribution empirique et P_cible."""
    counts = {s: 0 for s in all_seqs}
    for s in found:
        counts[s] += 1
    total = len(found)
    emp   = np.array([counts[s] / total for s in all_seqs])
    return float(np.abs(emp - P_true).sum())

# ── Campagne multi-seeds ───────────────────────────────────────────────────────
N_SEEDS = 5
print(f"\n=== Campagne {N_SEEDS} seeds × {N_EVAL} évaluations ===")

results = {}
for task_name, reward_fn, P_true, modes in [
        ("A_chaotique", reward_A, P_A, MODES_A),
        ("B_lisse",     reward_B, P_B, MODES_B)]:
    cov_G, cov_U, tv_G, tv_U = [], [], [], []
    print(f"\nTâche {task_name}")
    for seed in range(N_SEEDS):
        found_G, _ = run_gflownet(seed, reward_fn)
        found_U, _ = run_uniform(seed)
        cov_G.append(coverage(found_G, modes))
        cov_U.append(coverage(found_U, modes))
        tv_G.append(tv_distance(found_G, P_true, ALL_SEQS))
        tv_U.append(tv_distance(found_U, P_true, ALL_SEQS))
        print(f"  seed {seed}  GFN cov={cov_G[-1]:.2f} tv={tv_G[-1]:.3f}"
              f"  UNI cov={cov_U[-1]:.2f} tv={tv_U[-1]:.3f}")
    results[task_name] = dict(cov_G=cov_G, cov_U=cov_U, tv_G=tv_G, tv_U=tv_U)

# ── Résumé ────────────────────────────────────────────────────────────────────
print("\n\n=== RÉSUMÉ ===")
print(f"{'Tâche':<18} {'GFN coverage':>13} {'UNI coverage':>13} {'ΔTVA GFN-UNI':>14}")
for task, d in results.items():
    cG = np.array(d["cov_G"]); cU = np.array(d["cov_U"])
    tG = np.array(d["tv_G"]);  tU = np.array(d["tv_U"])
    delta_tv = (tU - tG).mean()  # positif = GFN meilleur (TV plus basse)
    print(f"{task:<18}  {cG.mean():.2f}±{cG.std():.2f}     "
          f"{cU.mean():.2f}±{cU.std():.2f}     Δtv={delta_tv:+.3f}")

# Test t GFN vs UNI sur coverage, tâche B
cG_B = np.array(results["B_lisse"]["cov_G"])
cU_B = np.array(results["B_lisse"]["cov_U"])
diff = cG_B.mean() - cU_B.mean()
se   = np.sqrt(cG_B.var(ddof=1)/len(cG_B) + cU_B.var(ddof=1)/len(cU_B)) + 1e-9
print(f"\nTâche B — GFN vs UNI : Δcoverage = {diff:+.3f}  (t ≈ {diff/se:.1f})")
print("Interprétation :")
if diff > 0.05:
    print("  → GFN > UNI en régime lisse ✓  (loi de régime confirmée)")
elif diff > -0.05:
    print("  → GFN ≈ UNI même en régime lisse (structure encore insuffisante)")
else:
    print("  → UNI > GFN (résultat inattendu — investiguer)")

# ── Courbes de convergence ─────────────────────────────────────────────────────
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Re-run pour les courbes (on stocke les trajectoires de coverage)
N_CKPTS = 20

def run_gfn_curve(seed, reward_fn, modes, n_eval=N_EVAL):
    torch.manual_seed(seed); np.random.seed(seed)
    pol = Policy(); opt = optim.Adam(pol.parameters(), lr=5e-3)
    found, used, curve = set(), 0, []
    step = n_eval // N_CKPTS
    while used < n_eval:
        eps = max(0.05, 0.4 * (1 - used / (n_eval * 0.5)))
        seqs, lpf, R = rollout(pol, B_TRAIN, eps, reward_fn)
        loss = ((pol.log_Z + lpf - torch.log(R)) ** 2).mean()
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(pol.parameters(), 1.0)
        opt.step()
        for s in seqs:
            found.add(s); used += 1
            if used % step == 0:
                curve.append(coverage(found, modes))
            if used >= n_eval: break
    return np.array(curve)

def run_uni_curve(seed, modes, n_eval=N_EVAL):
    rng = np.random.default_rng(seed)
    found, curve = set(), []
    step = n_eval // N_CKPTS
    for used in range(1, n_eval + 1):
        s = tuple(int(rng.integers(V_TOK)) for _ in range(SEQ_L))
        found.add(s)
        if used % step == 0:
            curve.append(coverage(found, modes))
    return np.array(curve)

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
xs = np.linspace(N_EVAL // N_CKPTS, N_EVAL, N_CKPTS)

for ax, (task, reward_fn, modes, label) in zip(axes, [
        ("A — Récompense chaotique", reward_A, MODES_A, "non apprenable"),
        ("B — Récompense lisse (Hamming)", reward_B, MODES_B, "apprenable")]):
    curves_G = np.stack([run_gfn_curve(s, reward_fn, modes) for s in range(N_SEEDS)])
    curves_U = np.stack([run_uni_curve(s, modes) for s in range(N_SEEDS)])
    for curves, color, lbl in [(curves_G, "#4caf84", "GFlowNet"),
                                (curves_U, "#7aa2ff", "Uniforme")]:
        mu, sd = curves.mean(0), curves.std(0)
        ax.plot(xs, mu, lw=2, color=color, label=lbl)
        ax.fill_between(xs, mu - sd, mu + sd, alpha=0.2, color=color)
    ax.set_xlabel("Budget d'évaluations")
    ax.set_ylabel("Coverage des modes (top 1%)")
    ax.set_title(f"{task}\n({label})", fontsize=10)
    ax.legend(fontsize=9)

plt.suptitle(
    "Loi de régime : GFlowNet > Uniforme seulement si la récompense est apprenable\n"
    f"Espace : {V_TOK}^{SEQ_L}={N_TOTAL} séquences, {N_SEEDS} seeds × {N_EVAL} évaluations",
    fontsize=10, y=1.02
)
plt.tight_layout()
out = "/home/user/curly-enigma/gflownet_regime_results.png"
plt.savefig(out, dpi=130, bbox_inches="tight")
print(f"\nFigure → {out}")
