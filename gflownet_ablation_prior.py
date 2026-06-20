"""
Ablation prior : Uniforme (U) vs GRU (G) vs Oracle faible (O)

Hypothèse à tester :
    un prior structuré améliore l'efficacité d'exploration (T_success, R_bar)
    SANS dégrader la diversité des solutions (couverture, H).

3 conditions :
  U — Uniforme   : logits constants = 0, seul log_Z appris
  G — GRU        : encodeur GRU appris (condition principale)
  O — Oracle     : GRU avec biais initial fort vers '*' et '10'

4 métriques :
  T_success  : première étape où max(R_batch) >= R_TH (0 = optimal)
  R_bar      : récompense moyenne par fenêtre glissante de 50 steps
  Coverage   : % de modes (R>=R_TH) découverts à l'évaluation finale
  H          : entropie de la distribution terminale apprise
"""
import math, torch, torch.nn as nn, torch.optim as optim
import numpy as np
from itertools import product as iproduct

torch.manual_seed(0); np.random.seed(0)

# ── Espace ──────────────────────────────────────────────────────────────────
VOCAB  = ["<PAD>", "2", "5", "10", "+", "*", "EOS"]
V2I    = {t: i for i, t in enumerate(VOCAB)}
I2V    = {i: t for t, i in V2I.items()}
V      = len(VOCAB)
MAX_T  = 6          # 5 tokens max + EOS forcé
TARGET = 42.0
R_TH   = 0.5        # seuil "mode" : R >= 0.5 ↔ |val-42| <= 3.47

def execute(tokens):
    clean = [t for t in tokens if t not in ("<PAD>", "EOS")]
    if not clean: return 0.0
    try:
        val = float(eval("".join(clean), {"__builtins__": {}}))
        if not math.isfinite(val) or abs(val) > 1e6: return 0.0
        return float(math.exp(-abs(val - TARGET) / 5.0))
    except Exception:
        return 0.0

# Énumération exhaustive des modes atteignables (tokens non-spéciaux, longueur 1-5)
_content_toks = ["2", "5", "10", "+", "*"]
_all_seqs = []
for L in range(1, 6):
    for seq in iproduct(_content_toks, repeat=L):
        _all_seqs.append(list(seq))
MODE_KEYS = {
    "".join(s): execute(s)
    for s in _all_seqs
    if execute(s) >= R_TH
}
N_MODES = len(MODE_KEYS)
print(f"Modes accessibles avec R≥{R_TH} : {N_MODES}")
top3 = sorted(MODE_KEYS.items(), key=lambda kv: -kv[1])[:3]
for k, r in top3:
    print(f"  '{k}'  R={r:.4f}")

# ── Architectures ────────────────────────────────────────────────────────────
class GRUPolicy(nn.Module):
    def __init__(self, h=64, bias_tokens=None):
        super().__init__()
        self.emb   = nn.Embedding(V, h)
        self.gru   = nn.GRU(h, h, batch_first=True)
        self.head  = nn.Linear(h, V)
        self.log_Z = nn.Parameter(torch.zeros(()))
        if bias_tokens:
            with torch.no_grad():
                for tok, val in bias_tokens.items():
                    self.head.bias[V2I[tok]] += val

    def logits(self, seqs):
        out, _ = self.gru(self.emb(seqs))
        return self.head(out[:, -1])

class UniformPolicy(nn.Module):
    """Politique constante (logits=0, uniforme sur tous tokens).
    Seul log_Z est appris — la politique ne change pas."""
    def __init__(self):
        super().__init__()
        self.log_Z = nn.Parameter(torch.zeros(()))

    def logits(self, seqs):
        return torch.zeros(seqs.shape[0], V)

# ── Rollout (identique à gflownet_program.py, fixes inclus) ─────────────────
_eos_only  = torch.full((1, V), -1e9); _eos_only[0,  V2I["EOS"]]   = 0.0
_pad_block = torch.zeros(1, V);        _pad_block[0, V2I["<PAD>"]] = -1e9

def rollout(model, B, eps):
    seqs     = torch.full((B, 1), V2I["<PAD>"], dtype=torch.long)
    log_pf   = torch.zeros(B)
    finished = torch.zeros(B, dtype=torch.bool)
    programs = [[] for _ in range(B)]
    for t in range(MAX_T):
        lg  = model.logits(seqs) + _pad_block
        fin = finished.clone()[:, None].expand_as(lg)
        lg  = torch.where(fin, _eos_only.expand(B, -1), lg)
        if t == MAX_T - 1:
            lg = _eos_only.expand(B, -1)
        lp = torch.log_softmax(lg, dim=-1)
        if np.random.rand() < eps:
            actions = torch.randint(1, V, (B,))
            actions[finished] = V2I["EOS"]
        else:
            actions = torch.multinomial(lp.exp(), 1).squeeze(1)
        log_pf = torch.where(fin[:, 0], log_pf,
                             log_pf + lp.gather(1, actions[:, None]).squeeze(1))
        for b in range(B):
            if not finished[b]:
                tok = I2V[int(actions[b])]
                if tok != "EOS":
                    programs[b].append(tok)
                finished[b] = (tok == "EOS")
        seqs = torch.cat([seqs, actions[:, None]], dim=1)
    rewards = torch.tensor([execute(programs[b]) for b in range(B)],
                           dtype=torch.float32).clamp(min=1e-6)
    return log_pf, rewards, programs

# ── Entraînement + mesures ───────────────────────────────────────────────────
N_STEPS  = 2000
B        = 64
N_SEEDS  = 3
N_EVAL   = 6000   # échantillons pour évaluation finale
WINDOW   = 50     # fenêtre glissante pour R_bar

def run_condition(make_model_opt, label, n_steps=N_STEPS, n_seeds=N_SEEDS):
    all_R_hist, all_cov, all_H, all_T = [], [], [], []
    for seed in range(n_seeds):
        torch.manual_seed(seed); np.random.seed(seed)
        model, opt = make_model_opt()
        R_hist = []
        T_succ = n_steps  # jamais atteint = n_steps
        for step in range(n_steps):
            eps = max(0.05, 0.5 * (1 - step / 1000))
            lpf, R, _ = rollout(model, B, eps)
            loss = ((model.log_Z + lpf - torch.log(R)) ** 2).mean()
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            R_hist.append(float(R.mean()))
            if T_succ == n_steps and float(R.max()) >= R_TH:
                T_succ = step
        # Évaluation finale (greedy)
        prog_count = {}
        with torch.no_grad():
            for _ in range(N_EVAL // 100):
                _, _, progs = rollout(model, 100, eps=0.0)
                for p in progs:
                    k = "".join(p)
                    prog_count[k] = prog_count.get(k, 0) + 1
        total_e = sum(prog_count.values())
        cov = sum(1 for k in MODE_KEYS if prog_count.get(k, 0) > 0) / max(N_MODES, 1)
        probs = np.array(list(prog_count.values()), dtype=np.float64) / max(total_e, 1)
        H = float(-np.sum(probs * np.log(probs + 1e-15)))
        all_R_hist.append(R_hist)
        all_cov.append(cov)
        all_H.append(H)
        all_T.append(T_succ)
        print(f"  [{label}] seed {seed}  T_succ={T_succ:4d}  cov={cov:.2f}  H={H:.2f}")
    return (np.array(all_R_hist),
            np.array(all_cov),
            np.array(all_H),
            np.array(all_T))

def make_U():
    m = UniformPolicy()
    return m, optim.Adam(m.parameters(), lr=3e-3)

def make_G():
    m = GRUPolicy()
    return m, optim.Adam(m.parameters(), lr=3e-3)

def make_O():
    m = GRUPolicy(bias_tokens={"*": 1.5, "10": 1.5})
    return m, optim.Adam(m.parameters(), lr=3e-3)

CONDITIONS = [
    ("U — Uniforme", make_U, "#7aa2ff"),
    ("G — GRU",      make_G, "#4caf84"),
    ("O — Oracle",   make_O, "#ff8c42"),
]

results = {}
for label, factory, _ in CONDITIONS:
    print(f"\n=== Condition : {label} ===")
    results[label] = run_condition(factory, label)

# ── Résumé ───────────────────────────────────────────────────────────────────
print("\n\n=== RÉSUMÉ ABLATION ===")
print(f"{'Condition':<18} {'T_success':>10} {'Coverage':>10} {'H':>8}")
for label, _, _ in CONDITIONS:
    R_hist, cov, H, T = results[label]
    print(f"{label:<18} {T.mean():>9.0f}±{T.std():4.0f}"
          f"  {cov.mean():>8.2f}±{cov.std():.2f}"
          f"  {H.mean():>6.2f}±{H.std():.2f}")

# ── Figure ────────────────────────────────────────────────────────────────────
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

# 1. Courbe R_bar (moyenne glissante)
ax = axes[0]
for label, _, color in CONDITIONS:
    R_hist, *_ = results[label]
    mu = R_hist.mean(axis=0)
    sd = R_hist.std(axis=0)
    # lissage fenêtre glissante
    kernel = np.ones(WINDOW) / WINDOW
    mu_s = np.convolve(mu, kernel, mode="valid")
    x    = np.arange(WINDOW - 1, N_STEPS)
    ax.plot(x, mu_s, label=label, color=color, lw=1.8)
    sd_s = np.array([R_hist[:, i].std() for i in range(WINDOW-1, N_STEPS)])
    ax.fill_between(x, mu_s - sd_s, mu_s + sd_s, alpha=0.15, color=color)
ax.axhline(R_TH, c="gray", ls="--", lw=1, label=f"R_TH={R_TH}")
ax.set_xlabel("Étape"); ax.set_ylabel("R̄ (fenêtre 50)")
ax.set_title("Récompense moyenne\n(élevée = meilleure exploitation)", fontsize=10)
ax.legend(fontsize=8)

# 2. T_success (boxplot)
ax = axes[1]
T_data  = [results[l][3] for l, *_ in CONDITIONS]
labels_ = [l.split(" — ")[0] for l, *_ in CONDITIONS]
colors_ = [c for _, _, c in CONDITIONS]
bps = ax.boxplot(T_data, tick_labels=labels_, patch_artist=True, widths=0.5)
for patch, color in zip(bps["boxes"], colors_):
    patch.set_facecolor(color); patch.set_alpha(0.7)
ax.set_ylabel("Étape du premier R ≥ 0.5")
ax.set_title("T_success\n(bas = convergence rapide)", fontsize=10)
ax.set_ylim(0, N_STEPS + 100)

# 3. Coverage vs H (scatter, un point par seed)
ax = axes[2]
for label, _, color in CONDITIONS:
    _, cov, H, _ = results[label]
    ax.scatter(cov, H, s=80, color=color, label=label, zorder=5, alpha=0.8)
    ax.scatter(cov.mean(), H.mean(), s=200, color=color, marker="*", zorder=6)
ax.set_xlabel("Couverture des modes (Coverage)")
ax.set_ylabel("Entropie H(X)")
ax.set_title("Coverage vs Entropie\n(★ = moyenne, idéal : haut-droite)", fontsize=10)
ax.legend(fontsize=8)

plt.suptitle(
    "Ablation prior : Uniforme vs GRU vs Oracle\n"
    f"Vocab={V} tokens, MAX_T={MAX_T}, {N_MODES} modes (R≥{R_TH}), "
    f"{N_SEEDS} seeds × {N_STEPS} étapes",
    fontsize=10, y=1.02
)
plt.tight_layout()
out = "/home/user/curly-enigma/gflownet_ablation_prior_results.png"
plt.savefig(out, dpi=130, bbox_inches="tight")
print(f"\nFigure → {out}")
