"""
phase.py — Le GFlowNet comme ECHANTILLONNEUR, metriques exactes.

Question honnete : la qualite d'echantillonnage d'un GFlowNet (sa distance a
la cible p_R = R/Z) depend-elle de la RUGOSITE du paysage de recompense ?

Paysage = tirage d'un processus gaussien a noyau RBF sur une grille D-dim.
  petite longueur de correlation l  -> paysage en dents de scie (rugueux)
  grande l                          -> paysage lisse
On balaie l et on mesure, en ENUMERANT tout l'espace (pas d'argmax, pas
d'estimation Monte-Carlo) :
  TV(p_theta, p_R), KL(p_theta || p_R), H(p_theta) vs H(p_R)
Reference : TV(uniforme, p_R) — pour savoir si le GFlowNet bat vraiment le hasard.

Budget d'entrainement LIMITE (~3x le nb d'etats) pour que la STRUCTURE du
paysage compte (sinon le reseau memorise tout et la rugosite n'a aucun effet).
"""
import numpy as np, torch, torch.nn as nn, itertools

D, K = 4, 6                                   # grille 6^4 = 1296 etats (enumerables)
STATES = list(itertools.product(range(K), repeat=D))
COORDS = np.array(STATES, dtype=float)
N = len(STATES)
BUDGET = 4000
BATCH = 50
L_SCALES = [0.3, 0.6, 1.2, 2.5, 5.0]
LAND_SEEDS = [0, 1, 2, 3]

def gp_reward(length_scale, seed):
    """Tire un paysage GP-RBF et renvoie p_R = R/Z (vecteur de taille N)."""
    rng = np.random.default_rng(seed)
    sq = ((COORDS[:, None, :] - COORDS[None, :, :]) ** 2).sum(-1)
    Cov = np.exp(-sq / (2 * length_scale ** 2)) + 1e-4 * np.eye(N)
    L = np.linalg.cholesky(Cov)
    g = L @ rng.standard_normal(N)            # echantillon du GP
    R = np.exp(1.5 * (g - g.mean()) / (g.std() + 1e-9))
    return R / R.sum()

class Net(nn.Module):
    def __init__(self, h=128):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(D + D * K, h), nn.ReLU(),
                                  nn.Linear(h, h), nn.ReLU(), nn.Linear(h, K))
        self.log_Z = nn.Parameter(torch.zeros(()))
    def forward(self, x): return self.body(x)

def enc(partial):
    v = np.zeros(D + D * K, dtype=np.float32)
    v[len(partial)] = 1.0
    for i, c in enumerate(partial):
        v[D + i * K + c] = 1.0
    return v

def logprobs(net, partials):
    t = len(partials[0])
    return torch.log_softmax(net(torch.tensor(np.stack([enc(p) for p in partials]))), -1)

def sample(net, B, eps):
    partials = [[] for _ in range(B)]; lpf = torch.zeros(B)
    for t in range(D):
        lp = logprobs(net, partials)
        a = (torch.randint(0, K, (B,)) if np.random.rand() < eps
             else torch.multinomial(lp.exp(), 1).squeeze(1))
        lpf = lpf + lp.gather(1, a[:, None]).squeeze(1)
        for b in range(B): partials[b].append(int(a[b]))
    return partials, lpf

def exact_policy_dist(net):
    """p_theta(x) pour TOUS les etats, par enumeration exacte (pas d'echantillonnage)."""
    logp = torch.zeros(N)
    with torch.no_grad():
        for t in range(D):
            prefixes = [list(s[:t]) for s in STATES]
            lp = logprobs(net, prefixes)
            nxt = torch.tensor([s[t] for s in STATES])
            logp = logp + lp.gather(1, nxt[:, None]).squeeze(1)
    return logp.exp().numpy()

def train_and_eval(pR, train_seed):
    torch.manual_seed(train_seed); np.random.seed(train_seed)
    net = Net(); opt = torch.optim.Adam(net.parameters(), lr=5e-3)
    pR_lookup = {s: pR[i] for i, s in enumerate(STATES)}
    used = 0
    while used < BUDGET:
        eps = max(0.05, 0.4 * (1 - used / (BUDGET * 0.6)))
        seqs, lpf = sample(net, BATCH, eps)
        R = torch.tensor([pR_lookup[tuple(s)] for s in seqs], dtype=torch.float32)
        loss = ((net.log_Z + lpf - torch.log(R + 1e-12)) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        used += BATCH
    pth = exact_policy_dist(net); pth = pth / pth.sum()
    tv = 0.5 * np.abs(pth - pR).sum()
    kl = float((pth * (np.log(pth + 1e-12) - np.log(pR + 1e-12))).sum())
    Hth = -(pth * np.log(pth + 1e-12)).sum()
    return tv, kl, Hth

HR_by_l = {}
print(f"espace = {N} etats | budget = {BUDGET} echantillons (~{BUDGET/N:.1f}x)\n")
print(f"{'l (corr)':>9} | {'TV GFlowNet':>22} | {'TV uniforme':>11} | {'H(pR)':>6}")
print("-" * 62)
results = {}
for l in L_SCALES:
    tvs, kls, Hs, tvu, HRs = [], [], [], [], []
    for seed in LAND_SEEDS:
        pR = gp_reward(l, seed)
        tv, kl, Hth = train_and_eval(pR, train_seed=seed)
        tvs.append(tv); kls.append(kl); Hs.append(Hth)
        tvu.append(0.5 * np.abs(np.full(N, 1 / N) - pR).sum())
        HRs.append(-(pR * np.log(pR + 1e-12)).sum())
    tvs = np.array(tvs)
    results[l] = (tvs.mean(), tvs.std(), np.mean(tvu))
    print(f"{l:9.2f} | {tvs.mean():.3f} ± {tvs.std():.3f}        | "
          f"{np.mean(tvu):.3f}       | {np.mean(HRs):.2f}")

print("\nLecture : TV=0 -> echantillonnage parfait. "
      "Si TV(GFN) << TV(uniforme), le mecanisme apporte qq chose.")
print("Si TV(GFN) chute quand l augmente -> la rugosite controle la difficulte.")

# Figure
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ls   = np.array(L_SCALES)
tvs  = np.array([results[l][0] for l in L_SCALES])
tvs_std = np.array([results[l][1] for l in L_SCALES])
tvu  = np.array([results[l][2] for l in L_SCALES])

fig, ax = plt.subplots(figsize=(8, 5))
ax.errorbar(ls, tvs, yerr=tvs_std, fmt="o-", color="#4caf84", lw=2,
            ms=7, capsize=4, label="GFlowNet TV(p_θ, p_R)")
ax.plot(ls, tvu, "s--", color="#7aa2ff", lw=2, ms=7,
        label="Uniforme TV(uniform, p_R)  [référence]")
ax.axhline(0, c="gray", ls=":", lw=1)
ax.set_xlabel("Longueur de corrélation GP-RBF  l  (grand l = paysage lisse)", fontsize=11)
ax.set_ylabel("TV(p_apprise, p_cible)    [0=parfait, 1=max]", fontsize=11)
ax.set_title(
    "Qualité d'échantillonnage exacte vs rugosité du paysage\n"
    f"Espace {K}^{D}={N} états, budget={BUDGET}≈{BUDGET/N:.1f}×N, {len(LAND_SEEDS)} seeds GP",
    fontsize=11
)
ax.legend(fontsize=10)
ax.set_xscale("log")
plt.tight_layout()
out = "/home/user/curly-enigma/phase_results.png"
plt.savefig(out, dpi=130, bbox_inches="tight")
print(f"\nFigure → {out}")

import json; json.dump({str(k): list(v) for k, v in results.items()}, open("/home/user/curly-enigma/phase_results.json", "w"))
print("phase_results.json écrit")
