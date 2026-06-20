"""
phase2d.py — Diagramme de phase 2D : structure × alignement du prior.

Question : quand un prior structuré aide-t-il un GFlowNet ?
  - axe X : longueur de corrélation l (rugosité du paysage)
  - axe Y : alignement A (corrélation entre prior et récompense)

Avantage = TV(GFN sans prior) - TV(GFN avec prior)
  > 0 : le prior aide
  < 0 : le prior nuit

3 agents comparés :
  U  — Uniforme  (référence absolue)
  G0 — GFN sans prior (α=0 dans la loss)
  Gp — GFN avec prior (α=2.0 : prior amplifié dans le sampling initial)

Espace : {0..4}^4 = 625 états (énumérable)
Budget  : 1800 échantillons ≈ 2.9×N
"""
import numpy as np, torch, torch.nn as nn, itertools

D, K = 4, 5                                   # grille 5^4 = 625 états
STATES  = list(itertools.product(range(K), repeat=D))
COORDS  = np.array(STATES, dtype=float)
N       = len(STATES)                         # 625
IDX     = {s: i for i, s in enumerate(STATES)}

BUDGET   = 1800
BATCH    = 40
L_SCALES = [0.4, 0.8, 1.6, 3.2]
A_LEVELS = [0.0, 0.4, 0.8, 1.0]
SEEDS    = [0, 1]

# ── GP-RBF ────────────────────────────────────────────────────────────────────
def gp_sample(length_scale, seed):
    """Tirage d'un champ GP-RBF → vecteur sur les N états."""
    rng = np.random.default_rng(seed)
    sq  = ((COORDS[:, None, :] - COORDS[None, :, :]) ** 2).sum(-1)
    Cov = np.exp(-sq / (2 * length_scale ** 2)) + 1e-4 * np.eye(N)
    L   = np.linalg.cholesky(Cov)
    return L @ rng.standard_normal(N)

def make_reward(g):
    R = np.exp(1.5 * (g - g.mean()) / (g.std() + 1e-9))
    return R / R.sum()

def make_aligned_prior(g_reward, g_noise, A):
    """
    Mélange entre g_reward (aligné) et g_noise (orthogonal).
    A=1 → prior parfait ; A=0 → prior aléatoire non corrélé.
    """
    # orthogonaliser g_noise par rapport à g_reward
    g_r   = g_reward - g_reward.mean()
    g_n   = g_noise  - g_noise.mean()
    g_n  -= (g_n @ g_r / (g_r @ g_r + 1e-9)) * g_r     # résidu orthogonal
    g_n  /= (np.linalg.norm(g_n) + 1e-9)
    g_r  /= (np.linalg.norm(g_r) + 1e-9)

    g_prior = A * g_r + (1 - A) * g_n
    return g_prior / (np.linalg.norm(g_prior) + 1e-9)

# ── Architecture GFN ──────────────────────────────────────────────────────────
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
    def forward(self, x):
        return self.body(x)

def logprobs(net, partials):
    return torch.log_softmax(
        net(torch.tensor(np.stack([enc(p) for p in partials]))), -1)

# ── prior_values : score du prior pour chaque état complet ───────────────────
def prior_values(g_prior_norm):
    """
    Transforme g_prior_norm (vecteur N) en prior de politique P_prior(x) :
    on calcule log P_prior(x) = log R_prior(x) / Z_prior où
    R_prior = exp(g_prior_norm * scale) (même normalisation que make_reward).
    Retourne un vecteur numpy de log-probabilités (sum = log 1).
    """
    scale = 1.5
    lv = scale * g_prior_norm
    lv -= np.max(lv)          # stabilité numérique
    lv -= np.log(np.exp(lv).sum())
    return lv                  # log P_prior(x) pour chaque état

# ── Sampling avec/sans prior ──────────────────────────────────────────────────
def sample(net, B, eps, log_prior_state=None, alpha_prior=0.0):
    """
    alpha_prior > 0 : lors de l'exploration epsilon, on tire selon un mélange
    politique × prior (soft guidance) pour orienter l'exploration.
    """
    partials = [[] for _ in range(B)]
    lpf      = torch.zeros(B)
    for t in range(D):
        lp = logprobs(net, partials)

        if np.random.rand() < eps:
            if log_prior_state is not None and alpha_prior > 0:
                # Calculer des poids du prior marginalisé au pas t
                prior_weights = np.zeros((B, K), dtype=np.float32)
                for b in range(B):
                    pref = partials[b]
                    for k in range(K):
                        # somme sur toutes les extensions de pref+[k]
                        total = 0.0
                        for s in STATES:
                            if s[:len(pref)+1] == tuple(pref + [k]):
                                total += np.exp(log_prior_state[IDX[s]])
                        prior_weights[b, k] = total + 1e-9
                    prior_weights[b] /= prior_weights[b].sum()
                # mélange : (1-α)*uniforme + α*prior
                uni = np.full((B, K), 1.0 / K, dtype=np.float32)
                mix = (1 - alpha_prior) * uni + alpha_prior * prior_weights
                mix = np.clip(mix, 1e-9, None)
                mix /= mix.sum(axis=1, keepdims=True)
                mix_t = torch.tensor(mix)
                a = torch.multinomial(mix_t, 1).squeeze(1)
            else:
                a = torch.randint(0, K, (B,))
        else:
            a = torch.multinomial(lp.exp(), 1).squeeze(1)

        lpf = lpf + lp.gather(1, a[:, None]).squeeze(1)
        for b in range(B):
            partials[b].append(int(a[b]))

    return partials, lpf

# ── Distribution exacte ───────────────────────────────────────────────────────
def exact_policy_dist(net):
    logp = torch.zeros(N)
    with torch.no_grad():
        for t in range(D):
            prefixes = [list(s[:t]) for s in STATES]
            lp  = logprobs(net, prefixes)
            nxt = torch.tensor([s[t] for s in STATES])
            logp = logp + lp.gather(1, nxt[:, None]).squeeze(1)
    p = logp.exp().numpy()
    return p / p.sum()

# ── Entraînement ─────────────────────────────────────────────────────────────
def train_eval(pR, log_prior_state, alpha_prior, train_seed):
    torch.manual_seed(train_seed); np.random.seed(train_seed)
    net = Net(); opt = torch.optim.Adam(net.parameters(), lr=5e-3)
    pR_lookup = {s: pR[i] for i, s in enumerate(STATES)}
    used = 0
    while used < BUDGET:
        eps = max(0.05, 0.4 * (1 - used / (BUDGET * 0.6)))
        seqs, lpf = sample(net, BATCH, eps,
                           log_prior_state=log_prior_state,
                           alpha_prior=alpha_prior)
        R    = torch.tensor([pR_lookup[tuple(s)] for s in seqs], dtype=torch.float32)
        loss = ((net.log_Z + lpf - torch.log(R + 1e-12)) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        used += BATCH
    pth = exact_policy_dist(net)
    tv  = 0.5 * np.abs(pth - pR).sum()
    return float(tv)

# ── Campagne ──────────────────────────────────────────────────────────────────
print(f"Espace : {N} états  |  budget = {BUDGET} ({BUDGET/N:.1f}×N)")
print(f"L_SCALES = {L_SCALES}")
print(f"A_LEVELS = {A_LEVELS}")
print(f"Seeds    = {SEEDS}\n")

# advantage[i_l, i_a] = TV(GFN) - TV(GFN+prior)  (>0 : prior aide)
advantage = np.zeros((len(L_SCALES), len(A_LEVELS)))
tv_gfn    = np.zeros((len(L_SCALES), len(A_LEVELS)))
tv_prior  = np.zeros((len(L_SCALES), len(A_LEVELS)))

header = f"{'l':>5} | {'A':>5} | {'TV_GFN':>8} | {'TV_prior':>8} | {'avantage':>9}"
print(header); print("-" * len(header))

for i_l, l in enumerate(L_SCALES):
    for i_a, A in enumerate(A_LEVELS):
        tvs_g, tvs_p = [], []
        for seed in SEEDS:
            g_reward = gp_sample(l, seed)
            g_noise  = gp_sample(l, seed + 100)   # champ indépendant
            pR       = make_reward(g_reward)

            g_prior  = make_aligned_prior(g_reward, g_noise, A)
            log_ps   = prior_values(g_prior)

            tv_g = train_eval(pR, log_prior_state=None, alpha_prior=0.0,
                              train_seed=seed)
            tv_p = train_eval(pR, log_prior_state=log_ps, alpha_prior=0.7,
                              train_seed=seed)
            tvs_g.append(tv_g); tvs_p.append(tv_p)

        tg = np.mean(tvs_g); tp = np.mean(tvs_p)
        adv = tg - tp
        tv_gfn[i_l, i_a]   = tg
        tv_prior[i_l, i_a] = tp
        advantage[i_l, i_a] = adv
        sign = "+" if adv >= 0 else ""
        print(f"{l:5.2f} | {A:5.2f} | {tg:8.3f}   | {tp:8.3f}   | {sign}{adv:.3f}")

# ── Sauvegarde numérique ──────────────────────────────────────────────────────
np.savez("/home/user/curly-enigma/phase2d.npz",
         advantage=advantage, tv_gfn=tv_gfn, tv_prior=tv_prior,
         L_SCALES=L_SCALES, A_LEVELS=A_LEVELS)
print("\nphase2d.npz écrit")

# ── Figure ────────────────────────────────────────────────────────────────────
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

L_arr = np.array(L_SCALES)
A_arr = np.array(A_LEVELS)

vmax = np.abs(advantage).max()
norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

# ── Heatmap avantage ─────────────────────────────────────────────────────────
ax = axes[0]
im = ax.imshow(advantage.T, aspect="auto", origin="lower",
               cmap="RdYlGn", norm=norm,
               extent=[0, len(L_SCALES), 0, len(A_LEVELS)])
plt.colorbar(im, ax=ax, label="TV(GFN) − TV(GFN+prior)\n(vert = prior aide)")
ax.set_xticks(np.arange(len(L_SCALES)) + 0.5)
ax.set_xticklabels([f"{l:.1f}" for l in L_SCALES])
ax.set_yticks(np.arange(len(A_LEVELS)) + 0.5)
ax.set_yticklabels([f"{a:.1f}" for a in A_LEVELS])
ax.set_xlabel("Longueur de corrélation l\n(grand l = paysage lisse)")
ax.set_ylabel("Alignement A\n(A=1 = prior parfait)")
ax.set_title("Avantage du prior\n(+ = prior aide, − = prior nuit)", fontsize=10)
for i_l in range(len(L_SCALES)):
    for i_a in range(len(A_LEVELS)):
        adv = advantage[i_l, i_a]
        sign = "+" if adv >= 0 else ""
        ax.text(i_l + 0.5, i_a + 0.5, f"{sign}{adv:.2f}",
                ha="center", va="center", fontsize=8,
                color="black" if abs(adv) < 0.5 * vmax else "white")

# ── TV GFN vs GFN+prior par A (une ligne par A) ──────────────────────────────
ax = axes[1]
colors_A = ["#ff8c42", "#7aa2ff", "#4caf84", "#9b59b6"]
for i_a, A in enumerate(A_LEVELS):
    ax.plot(L_arr, tv_gfn[:, i_a], "s--", color=colors_A[i_a],
            alpha=0.5, lw=1.5, ms=6, label=f"GFN (A={A:.1f})")
    ax.plot(L_arr, tv_prior[:, i_a], "o-", color=colors_A[i_a],
            lw=2, ms=7, label=f"GFN+prior (A={A:.1f})")
ax.set_xscale("log")
ax.set_xlabel("Longueur de corrélation l")
ax.set_ylabel("TV(p_θ, p_R)")
ax.set_title("TV exacte en fonction de l\npour 4 niveaux d'alignement", fontsize=10)
ax.legend(fontsize=7, ncol=2)

# ── Coupe à l=0.4 (rugosité maximale) : rôle de A ────────────────────────────
ax = axes[2]
i_l_rough = 0   # l=0.4 (le plus rugueux)
i_l_smooth = -1  # l=3.2 (le plus lisse)
ax.plot(A_arr, advantage[i_l_rough, :],  "o-",  color="#e74c3c", lw=2,
        ms=8, label=f"l={L_SCALES[i_l_rough]} (rugueux)")
ax.plot(A_arr, advantage[i_l_smooth, :], "s--", color="#3498db", lw=2,
        ms=8, label=f"l={L_SCALES[i_l_smooth]} (lisse)")
ax.axhline(0, c="gray", ls=":", lw=1)
ax.fill_between(A_arr, advantage[i_l_rough, :], 0,
                where=advantage[i_l_rough, :] > 0,
                alpha=0.15, color="#e74c3c")
ax.fill_between(A_arr, advantage[i_l_rough, :], 0,
                where=advantage[i_l_rough, :] < 0,
                alpha=0.15, color="red")
ax.set_xlabel("Alignement A")
ax.set_ylabel("TV(GFN) − TV(GFN+prior)")
ax.set_title("Impact de l'alignement\n(rugosité fixée)", fontsize=10)
ax.legend(fontsize=9)

plt.suptitle(
    "Diagramme de phase : Quand un prior aide-t-il un GFlowNet ?\n"
    f"Espace {K}^{D}={N} états, budget={BUDGET}≈{BUDGET/N:.1f}×N, "
    f"{len(SEEDS)} seeds, α_prior=0.7",
    fontsize=10, y=1.02
)
plt.tight_layout()
out = "/home/user/curly-enigma/phase2d.png"
plt.savefig(out, dpi=130, bbox_inches="tight")
print(f"Figure → {out}")
