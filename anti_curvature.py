"""
anti_curvature.py — Niveau 2 : SAM vs Adam sur paysages GP-RBF.

HYPOTHÈSE (Anti-Curvature) :
  Les paysages de récompense rugueux (l petit) induisent un paysage de loss
  TB avec de nombreuses vallées étroites et des points selle. Un optimiseur
  "conscient de la courbure" (SAM) devrait mieux s'en sortir que Adam seul.

  ΔTV(l) = TV_Adam(l) − TV_SAM(l)

  Prédiction :
    l faible (rugueux) → ΔTV > 0  (SAM aide)
    l grand  (lisse)   → ΔTV ≈ 0  (SAM n'apporte rien)

SAM (Sharpness-Aware Minimization, Foret et al. 2021) :
  1. Trouver la perturbation adverse : ε* = ρ · ∇L / ‖∇L‖
  2. Monter vers le pire voisin : θ' = θ + ε*
  3. Calculer ∇L(θ') et mettre à jour θ avec Adam

Espace : {0..5}^4 = 1296 états (énumérable → TV exacte)
Budget  : 4000 échantillons ≈ 3.1×N
"""
import numpy as np, torch, torch.nn as nn, itertools, math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

D, K = 4, 6
STATES = list(itertools.product(range(K), repeat=D))
COORDS = np.array(STATES, dtype=float)
N      = len(STATES)
BUDGET = 4000; BATCH = 50

L_SCALES    = [0.3, 0.6, 1.2, 2.5, 5.0]
LAND_SEEDS  = [0, 1, 2, 3]
SAM_RHO     = 0.05   # rayon de perturbation SAM

print(f"Espace : {N} états  |  budget = {BUDGET}  |  SAM ρ = {SAM_RHO}")

# ── GP-RBF ────────────────────────────────────────────────────────────────────
def gp_reward(l, seed):
    rng = np.random.default_rng(seed)
    sq  = ((COORDS[:, None, :] - COORDS[None, :, :]) ** 2).sum(-1)
    Cov = np.exp(-sq / (2 * l**2)) + 1e-4 * np.eye(N)
    g   = np.linalg.cholesky(Cov) @ rng.standard_normal(N)
    R   = np.exp(1.5 * (g - g.mean()) / (g.std() + 1e-9))
    return R / R.sum()

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

# ── Rollout ───────────────────────────────────────────────────────────────────
def sample_batch(net, B, eps, pR_lu):
    partials = [[] for _ in range(B)]; lpf = torch.zeros(B)
    for t in range(D):
        lp = logprobs(net, partials)
        a  = (torch.randint(0, K, (B,)) if np.random.rand() < eps
              else torch.multinomial(lp.exp(), 1).squeeze(1))
        lpf = lpf + lp.gather(1, a[:, None]).squeeze(1)
        for b in range(B): partials[b].append(int(a[b]))
    R = torch.tensor([pR_lu[tuple(s)] for s in partials], dtype=torch.float32)
    return partials, lpf, R

# ── SAM optimizer (wraps Adam) ────────────────────────────────────────────────
class SAMOptimizer:
    """
    SAM avec base optimizer = Adam.
    Usage :
      loss = compute_loss(); loss.backward()
      sam.first_step()                  # perturbe θ → θ+ε*
      loss2 = compute_loss(); loss2.backward()
      sam.second_step()                 # revient à θ, applique Adam(∇L(θ+ε*))
    """
    def __init__(self, params, lr=5e-3, rho=0.05):
        self.rho    = rho
        self.base   = torch.optim.Adam(params, lr=lr)
        self.params = list(self.base.param_groups[0]["params"])

    def _grad_norm(self):
        norms = [p.grad.norm() for p in self.params if p.grad is not None]
        return torch.stack(norms).norm()

    def first_step(self):
        """Applique la perturbation ε* = ρ · g / ‖g‖ (en mémoire)."""
        scale = self.rho / (self._grad_norm() + 1e-12)
        for p in self.params:
            if p.grad is not None:
                p._e_w = p.grad.detach() * scale
                p.data.add_(p._e_w)
        self.base.zero_grad()

    def second_step(self):
        """Retire la perturbation, puis fait un pas Adam sur ∇L(θ+ε*)."""
        for p in self.params:
            if hasattr(p, "_e_w"):
                p.data.sub_(p._e_w)
        self.base.step(); self.base.zero_grad()

    def zero_grad(self): self.base.zero_grad()

# ── Entraînement ──────────────────────────────────────────────────────────────
def train_eval(pR, train_seed, use_sam):
    torch.manual_seed(train_seed); np.random.seed(train_seed)
    net    = Net()
    pR_lu  = {s: float(pR[i]) for i, s in enumerate(STATES)}

    if use_sam:
        opt = SAMOptimizer(list(net.parameters()), lr=5e-3, rho=SAM_RHO)
    else:
        opt = torch.optim.Adam(net.parameters(), lr=5e-3)

    used = 0
    while used < BUDGET:
        eps = max(0.05, 0.4 * (1 - used / (BUDGET * 0.6)))

        if use_sam:
            # SAM : deux passes de forward/backward
            _, lpf, R = sample_batch(net, BATCH, eps, pR_lu)
            loss = ((net.log_Z + lpf - torch.log(R + 1e-12)) ** 2).mean()
            opt.zero_grad(); loss.backward()
            opt.first_step()                             # perturbe θ → θ+ε*

            _, lpf2, R2 = sample_batch(net, BATCH, eps, pR_lu)
            loss2 = ((net.log_Z + lpf2 - torch.log(R2 + 1e-12)) ** 2).mean()
            loss2.backward()
            opt.second_step()                            # Adam sur ∇L(θ+ε*)
            used += 2 * BATCH                            # deux batches consommés
        else:
            _, lpf, R = sample_batch(net, BATCH, eps, pR_lu)
            loss = ((net.log_Z + lpf - torch.log(R + 1e-12)) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            used += BATCH

    pth = exact_dist(net)
    return 0.5 * np.abs(pth - pR).sum()

# ── Campagne ──────────────────────────────────────────────────────────────────
print(f"\n{'l':>5} | {'TV Adam':>15} | {'TV SAM':>15} | {'ΔTV':>9}")
print("-" * 52)

results = {}
for l in L_SCALES:
    tv_adam, tv_sam = [], []
    for seed in LAND_SEEDS:
        pR = gp_reward(l, seed)
        tv_adam.append(train_eval(pR, train_seed=seed, use_sam=False))
        tv_sam.append(train_eval(pR,  train_seed=seed, use_sam=True))
    ta = np.array(tv_adam); ts = np.array(tv_sam)
    delta = ta.mean() - ts.mean()
    results[l] = dict(adam=ta, sam=ts, delta=delta)
    sign = "+" if delta >= 0 else ""
    print(f"{l:5.2f} | {ta.mean():.4f}±{ta.std():.4f}  | "
          f"{ts.mean():.4f}±{ts.std():.4f}  | {sign}{delta:.4f}")

# ── Résumé ─────────────────────────────────────────────────────────────────────
print("\n=== VERDICT ANTI-CURVATURE ===")
deltas = [results[l]["delta"] for l in L_SCALES]
rough_delta  = np.mean([results[l]["delta"] for l in L_SCALES[:2]])   # l=0.3, 0.6
smooth_delta = np.mean([results[l]["delta"] for l in L_SCALES[-2:]])  # l=2.5, 5.0
print(f"  ΔTV moyen (l≤0.6, rugueux)  : {rough_delta:+.4f}")
print(f"  ΔTV moyen (l≥2.5, lisse)    : {smooth_delta:+.4f}")
if rough_delta > 0.01 and rough_delta > smooth_delta:
    print("  → Hypothèse CONFIRMÉE : SAM aide davantage sur paysages rugueux.")
elif rough_delta > 0.005:
    print("  → Signal faible : tendance dans le bon sens, mais bruit élevé.")
else:
    print("  → Hypothèse non confirmée : SAM n'apporte pas de gain différentiel.")

# ── Figure ────────────────────────────────────────────────────────────────────
ls       = np.array(L_SCALES)
tv_adam  = np.array([results[l]["adam"].mean() for l in L_SCALES])
tv_sam   = np.array([results[l]["sam"].mean()  for l in L_SCALES])
tv_adam_std = np.array([results[l]["adam"].std() for l in L_SCALES])
tv_sam_std  = np.array([results[l]["sam"].std()  for l in L_SCALES])
deltas   = tv_adam - tv_sam

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Gauche : courbes TV
ax = axes[0]
ax.errorbar(ls, tv_adam, yerr=tv_adam_std, fmt="s--",
            color="#7aa2ff", lw=2, ms=7, capsize=4, label="Adam (standard)")
ax.errorbar(ls, tv_sam,  yerr=tv_sam_std,  fmt="o-",
            color="#4caf84", lw=2, ms=7, capsize=4, label=f"SAM (ρ={SAM_RHO})")
ax.axhline(0, c="gray", ls=":", lw=1)
ax.set_xscale("log")
ax.set_xlabel("Longueur de corrélation l\n(grand l = paysage lisse)")
ax.set_ylabel("TV(p_θ, p_R)  [0=parfait]")
ax.set_title("Adam vs SAM : TV exacte\nen fonction de la rugosité", fontsize=10)
ax.legend(fontsize=10)

# Droite : ΔTV = TV_Adam - TV_SAM
ax = axes[1]
colors_bar = ["#e74c3c" if d < 0 else "#2ecc71" for d in deltas]
bars = ax.bar(range(len(L_SCALES)), deltas, color=colors_bar, alpha=0.8)
for i, (bar, d) in enumerate(zip(bars, deltas)):
    sign = "+" if d >= 0 else ""
    ax.text(i, d + (0.002 if d >= 0 else -0.004),
            f"{sign}{d:.4f}", ha="center", va="bottom" if d >= 0 else "top",
            fontsize=9)
ax.axhline(0, c="black", lw=1)
ax.set_xticks(range(len(L_SCALES)))
ax.set_xticklabels([f"l={l}" for l in L_SCALES])
ax.set_ylabel("ΔTV = TV_Adam − TV_SAM\n(vert = SAM aide, rouge = SAM nuit)")
ax.set_title("Avantage de SAM par rugosité\n(hypothèse : ΔTV↑ quand l↓)", fontsize=10)

plt.suptitle(
    f"Anti-Curvature (SAM, ρ={SAM_RHO}) vs Adam — GFlowNet TB\n"
    f"Espace {K}^{D}={N} états, budget={BUDGET}, {len(LAND_SEEDS)} seeds GP",
    fontsize=10, y=1.02
)
plt.tight_layout()
out = "/home/user/curly-enigma/anti_curvature.png"
plt.savefig(out, dpi=130, bbox_inches="tight")
print(f"\nFigure → {out}")
