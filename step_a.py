"""
step_a.py — GFlowNet + LM prior : deux mécanismes d'intégration.

ÉTAPE A : démonstration pratique des deux rôles du prior LM.

Protocole :
  LM prior = LSTM entraîné sur target=24 (hors boucle de récompense 42).
  4 conditions sur la tâche cible = 42 :

  1. REINFORCE  : RL classique, mode collapse attendu.
  2. GFN_std    : TB standard, P_F ∝ R  (Théorème B avec q=uniforme).
  3. GFN+LM_expl: GFN + prior LM en EXPLORATION uniquement (eps-greedy biaisé).
     → Prédit par Corollaire 1 : converge vers R/Z, pas q·R/Z.
     → Effet : découverte rapide des modes, mais politique finale potentiellement dégradée.
  4. GFN+LM_base: GFN + prior LM comme MESURE DE BASE (Théorème B).
     loss = (log Z + log P_F - log q_LM(x) - log R(x))²
     → Converge vers q_LM · R / Z_q (Proposition 1 prouvée).
     → Distribution finale = compromis syntaxe × sémantique.

Métriques :
  - T_success  : première étape où un mode (R≥0.5) est découvert.
  - R_policy   : récompense moyenne des 500 trajectoires GREEDY finales (eps=0).
  - Coverage   : % modes couverts en greedy final.
"""
import math, torch, torch.nn as nn, torch.optim as optim
import numpy as np
from itertools import product as iproduct
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

torch.manual_seed(0); np.random.seed(0)

# ── Vocabulaire ────────────────────────────────────────────────────────────────
VOCAB = ["<PAD>", "2", "5", "10", "+", "*", "EOS"]
V2I   = {t: i for i, t in enumerate(VOCAB)}
I2V   = {i: t for t, i in V2I.items()}
V     = len(VOCAB)
MAX_T = 6    # 5 tokens + EOS forcé

def execute(tokens, target=42.0):
    clean = [t for t in tokens if t not in ("<PAD>", "EOS")]
    if not clean: return 0.0
    try:
        val = float(eval("".join(clean), {"__builtins__": {}}))
        if not math.isfinite(val) or abs(val) > 1e6: return 0.0
        return float(math.exp(-abs(val - target) / 5.0))
    except Exception:
        return 0.0

# ── Espace complet ─────────────────────────────────────────────────────────────
def all_programs():
    nums, ops = ["2", "5", "10"], ["+", "*"]
    progs, Rs = [], []
    for n in nums:
        progs.append([n]); Rs.append(execute([n]))
    for a, op, b in iproduct(nums, ops, nums):
        progs.append([a, op, b]); Rs.append(execute([a, op, b]))
    for a, op1, b, op2, c in iproduct(nums, ops, nums, ops, nums):
        progs.append([a, op1, b, op2, c]); Rs.append(execute([a, op1, b, op2, c]))
    return progs, np.array(Rs, dtype=np.float64)

PROGS, R_ALL = all_programs()
MODE_KEYS  = {tuple(p) for p in PROGS if execute(p) >= 0.5}   # |val-42| ≤ 3.5
PROG_IDX   = {"".join(p): i for i, p in enumerate(PROGS)}
P_TRUE     = R_ALL / R_ALL.sum()
print(f"Espace : {len(PROGS)} programmes  |  {len(MODE_KEYS)} modes (R≥0.5)")
for p in sorted(MODE_KEYS, key=lambda p: -execute(p)):
    print(f"  {''.join(p):<14} val={eval(''.join(p),{'__builtins__':{}}):.0f}"
          f"  R={execute(p):.4f}")

# ── PHASE 1 : LM entraîné sur target=24 ───────────────────────────────────────
print("\n=== PHASE 1 : LM prior (target=24, hors boucle) ===")

class LMPrior(nn.Module):
    def __init__(self, h=64):
        super().__init__()
        self.emb  = nn.Embedding(V, h)
        self.gru  = nn.GRU(h, h, batch_first=True)
        self.head = nn.Linear(h, V)

    def forward(self, seqs):
        out, _ = self.gru(self.emb(seqs))
        return self.head(out)     # [B, T, V]

    def logits_step(self, seqs):
        out, _ = self.gru(self.emb(seqs))
        return self.head(out[:, -1])  # [B, V]

def make_lm_data(target=24.0, n=6000):
    rng = np.random.default_rng(42)
    progs_w = [(p, execute(p, target)) for p in PROGS if execute(p, target) > 1e-3]
    seqs, ws = zip(*progs_w); ws = np.array(ws); ws /= ws.sum()
    chosen = rng.choice(len(seqs), size=n, replace=True, p=ws)
    return [list(seqs[i]) for i in chosen]

lm_data = make_lm_data(target=24.0)
lm = LMPrior()
lm_opt = optim.Adam(lm.parameters(), lr=3e-3)

def batch_lm(data, B=64):
    PAD = V2I["<PAD>"]
    idx = np.random.choice(len(data), B, replace=True)
    seqs_in, seqs_tgt = [], []
    for i in idx:
        toks = [PAD] + [V2I[t] for t in data[i]] + [V2I["EOS"]]
        seqs_in.append(toks[:-1]); seqs_tgt.append(toks[1:])
    maxL = max(len(s) for s in seqs_in)
    inp  = torch.full((B, maxL), PAD, dtype=torch.long)
    tgt  = torch.full((B, maxL), PAD, dtype=torch.long)
    for b in range(B):
        inp[b, :len(seqs_in[b])] = torch.tensor(seqs_in[b])
        tgt[b, :len(seqs_tgt[b])] = torch.tensor(seqs_tgt[b])
    return inp, tgt

for step in range(1500):
    inp, tgt = batch_lm(lm_data)
    logits   = lm(inp)             # [B, T, V]
    mask     = (tgt != V2I["<PAD>"]).float()
    lp       = torch.log_softmax(logits, dim=-1)
    nll      = -(lp.gather(2, tgt.unsqueeze(2)).squeeze(2) * mask).sum() / (mask.sum() + 1e-9)
    lm_opt.zero_grad(); nll.backward()
    torch.nn.utils.clip_grad_norm_(lm.parameters(), 1.0)
    lm_opt.step()
    if step % 500 == 0:
        print(f"  step {step:4d}  NLL = {nll.item():.4f}")

# Précompute log_q_LM pour TOUS les programmes (129 états)
def log_q_lm_batch(progs):
    """Calcule log q_LM(x) pour une liste de programmes (listes de str)."""
    PAD = V2I["<PAD>"]
    maxL = max(len(p) for p in progs) + 2   # PAD + tokens + EOS
    inp = torch.full((len(progs), maxL), PAD, dtype=torch.long)
    tgt = torch.full((len(progs), maxL), PAD, dtype=torch.long)
    for b, p in enumerate(progs):
        toks = [PAD] + [V2I[t] for t in p] + [V2I["EOS"]]
        inp[b, :len(toks)-1] = torch.tensor(toks[:-1])
        tgt[b, :len(toks)-1] = torch.tensor(toks[1:])
    with torch.no_grad():
        logits = lm(inp)   # [B, T, V]
        lp     = torch.log_softmax(logits, dim=-1)
    log_q = torch.zeros(len(progs))
    for b, p in enumerate(progs):
        T = len(p) + 1   # tokens + EOS
        log_q[b] = lp[b, :T, :].gather(1, tgt[b, :T].unsqueeze(1)).sum()
    return log_q.numpy()

print("\nPré-calcul log q_LM pour tous les programmes...")
LQ_ALL = log_q_lm_batch(PROGS)
# log-normalise pour en faire une proba
LQ_ALL = LQ_ALL - float(np.log(np.exp(LQ_ALL).sum()))
LQ_LU  = {"".join(p): LQ_ALL[i] for i, p in enumerate(PROGS)}

print("  Top 5 programmes par log q_LM :")
for i in np.argsort(LQ_ALL)[::-1][:5]:
    p = PROGS[i]
    print(f"    {''.join(p):<12} log_q={LQ_ALL[i]:.3f}  "
          f"R(42)={execute(p):.3f}  R(24)={execute(p,24):.3f}")

# ── Architecture GFN ───────────────────────────────────────────────────────────
_eos_only  = torch.full((1, V), -1e9); _eos_only[0, V2I["EOS"]] = 0.0
_pad_block = torch.zeros(1, V);       _pad_block[0, V2I["<PAD>"]] = -1e9

class PolicyGFN(nn.Module):
    def __init__(self, h=64):
        super().__init__()
        self.emb   = nn.Embedding(V, h)
        self.gru   = nn.GRU(h, h, batch_first=True)
        self.head  = nn.Linear(h, V)
        self.log_Z = nn.Parameter(torch.zeros(()))
    def logits_step(self, seqs):
        out, _ = self.gru(self.emb(seqs)); return self.head(out[:, -1])

LOG_N = float(np.log(len(PROGS)))   # décalage de normalisation pour base-measure

def rollout(pol, B, eps, use_lm_expl=False, alpha=0.7):
    seqs     = torch.full((B, 1), V2I["<PAD>"], dtype=torch.long)
    log_pf   = torch.zeros(B)
    finished = torch.zeros(B, dtype=torch.bool)
    programs = [[] for _ in range(B)]

    for t in range(MAX_T):
        lg  = pol.logits_step(seqs) + _pad_block
        fin = finished.clone()[:, None].expand_as(lg)
        lg  = torch.where(fin, _eos_only.expand(B, -1), lg)
        if t == MAX_T - 1: lg = _eos_only.expand(B, -1)
        lp  = torch.log_softmax(lg, dim=-1)

        if np.random.rand() < eps:
            if use_lm_expl:
                with torch.no_grad():
                    lm_lg = lm.logits_step(seqs) + _pad_block
                fin_b  = finished[:, None].expand_as(lm_lg)
                lm_lg  = torch.where(fin_b, _eos_only.expand(B, -1), lm_lg)
                if t == MAX_T - 1: lm_lg = _eos_only.expand(B, -1)
                lm_lp  = torch.log_softmax(lm_lg, dim=-1)
                mix    = (1 - alpha) / V + alpha * lm_lp.exp()
                mix    = mix.clamp(1e-9); mix /= mix.sum(-1, keepdim=True)
                actions = torch.multinomial(mix, 1).squeeze(1)
            else:
                actions = torch.randint(1, V, (B,))
                actions[finished] = V2I["EOS"]
        else:
            actions = torch.multinomial(lp.exp(), 1).squeeze(1)

        log_pf = torch.where(fin[:, 0], log_pf,
                             log_pf + lp.gather(1, actions[:, None]).squeeze(1))
        for b in range(B):
            if not finished[b]:
                tok = I2V[int(actions[b])]
                if tok != "EOS": programs[b].append(tok)
                finished[b] = (tok == "EOS")
        seqs = torch.cat([seqs, actions[:, None]], dim=1)

    R   = torch.tensor([execute(programs[b]) for b in range(B)],
                       dtype=torch.float32).clamp(1e-6)
    lqb = torch.tensor([LQ_LU.get("".join(programs[b]), -30.0)
                        for b in range(B)], dtype=torch.float32)
    return log_pf, R, lqb, programs

# ── Entraînement ───────────────────────────────────────────────────────────────
N_STEPS = 2000; B_TRAIN = 64; N_SEEDS = 3; CKPT = 100

def run(seed, mode):
    """
    mode : 'reinforce' | 'gfn_std' | 'gfn_lm_expl' | 'gfn_lm_base'
    """
    torch.manual_seed(seed); np.random.seed(seed)
    pol = PolicyGFN()
    opt = optim.Adam(pol.parameters(), lr=3e-3)
    found, T_succ, R_hist, cov_hist = set(), N_STEPS, [], []
    for step in range(N_STEPS):
        eps = max(0.05, 0.5 * (1 - step / 1000))
        log_pf, R, lq, progs = rollout(pol, B_TRAIN, eps,
                                       use_lm_expl=(mode == "gfn_lm_expl"))
        if mode == "reinforce":
            base = R.mean().detach()
            loss = (-(log_pf * (R - base))).mean()
        elif mode == "gfn_std":
            loss = ((pol.log_Z + log_pf - torch.log(R)) ** 2).mean()
        elif mode == "gfn_lm_expl":
            # MÊME loss que gfn_std : seule l'exploration change
            loss = ((pol.log_Z + log_pf - torch.log(R)) ** 2).mean()
        else:  # gfn_lm_base : Théorème B (Proposition 1)
            # target = log q_LM(x) + log R(x) + LOG_N (décalage normalisation)
            loss = ((pol.log_Z + log_pf - (lq + torch.log(R) + LOG_N)) ** 2).mean()

        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(pol.parameters(), 1.0)
        opt.step()
        R_hist.append(float(R.mean()))
        if T_succ == N_STEPS and any(tuple(p) in MODE_KEYS for p in progs):
            T_succ = step
        for p in progs: found.add(tuple(p))
        if step % CKPT == 0:
            cov_hist.append(sum(1 for m in MODE_KEYS if m in found) / max(len(MODE_KEYS), 1))

    # Évaluation greedy finale
    with torch.no_grad():
        _, R_eval, _, progs_eval = rollout(pol, 500, eps=0.0)
        R_policy = float(R_eval.mean())
        cov_final = sum(1 for p in progs_eval if tuple(p) in MODE_KEYS) / 500

    return dict(T=T_succ, R_hist=np.array(R_hist),
                cov_hist=np.array(cov_hist), R_policy=R_policy,
                cov_final=cov_final, found=found)

CONDITIONS = [
    ("REINFORCE",    "reinforce",    "#e74c3c"),
    ("GFN_std",      "gfn_std",      "#7aa2ff"),
    ("GFN+LM_expl",  "gfn_lm_expl",  "#f39c12"),
    ("GFN+LM_base",  "gfn_lm_base",  "#4caf84"),
]

results = {}
for label, mode, _ in CONDITIONS:
    print(f"\n--- {label} ---")
    runs = []
    for seed in range(N_SEEDS):
        r = run(seed, mode)
        runs.append(r)
        print(f"  seed {seed}  T_succ={r['T']:4d}  "
              f"R_policy={r['R_policy']:.4f}  cov_final={r['cov_final']:.2f}")
    results[label] = runs

# ── Résumé ─────────────────────────────────────────────────────────────────────
print("\n=== RÉSUMÉ (moyennes sur 3 seeds) ===")
print(f"{'Condition':<16} {'T_success':>12} {'R_policy':>10} {'Coverage':>10}")
print("-" * 52)
for label, _, _ in CONDITIONS:
    rs = results[label]
    T  = np.array([r["T"] for r in rs])
    Rp = np.array([r["R_policy"] for r in rs])
    Cf = np.array([r["cov_final"] for r in rs])
    print(f"{label:<16}  {T.mean():>7.0f}±{T.std():4.0f}"
          f"  {Rp.mean():>8.4f}±{Rp.std():.4f}"
          f"  {Cf.mean():>8.3f}±{Cf.std():.3f}")

print("\nInterprétation :")
print("  Corollaire 1 : GFN+LM_expl converge vers R/Z (même que GFN_std)")
print("  Proposition 1 : GFN+LM_base converge vers q_LM·R/Z_q (Théorème B)")

# ── Figure ──────────────────────────────────────────────────────────────────────
COLORS = {l: c for l, _, c in CONDITIONS}
WINDOW = 50

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# 1. Coverage au fil du temps
ax = axes[0]
for label, _, _ in CONDITIONS:
    rs  = results[label]
    cys = np.array([r["cov_hist"] for r in rs])
    mu  = cys.mean(0); sd = cys.std(0)
    xs  = np.arange(len(mu)) * CKPT * B_TRAIN
    ax.plot(xs, mu, lw=2, color=COLORS[label], label=label)
    ax.fill_between(xs, mu - sd, mu + sd, alpha=0.15, color=COLORS[label])
ax.set_xlabel("Évaluations utilisées")
ax.set_ylabel("Coverage des modes (R≥0.5)")
ax.set_title("Courbes de coverage\n(découverte cumulative)", fontsize=10)
ax.legend(fontsize=8.5); ax.set_ylim(0, 1.05)

# 2. Récompense moyenne de la POLITIQUE (greedy final, barre)
ax = axes[1]
x  = np.arange(len(CONDITIONS)); w = 0.6
for i, (label, _, _) in enumerate(CONDITIONS):
    rs = results[label]
    Rp = np.array([r["R_policy"] for r in rs])
    ax.bar(i, Rp.mean(), w, yerr=Rp.std(), capsize=5,
           color=COLORS[label], alpha=0.85)
    ax.text(i, Rp.mean() + Rp.std() + 0.003,
            f"{Rp.mean():.4f}", ha="center", fontsize=8)
ax.set_xticks(x)
ax.set_xticklabels([l for l, *_ in CONDITIONS], fontsize=8, rotation=10)
ax.set_ylabel("R̄ politique greedy (eps=0)")
ax.set_title("Récompense finale de la politique\n(haute = bonne exploitation)", fontsize=10)

# 3. T_success
ax = axes[2]
T_data  = [[r["T"] for r in results[l]] for l, *_ in CONDITIONS]
labels_ = [l for l, *_ in CONDITIONS]
colors_ = [c for _, _, c in CONDITIONS]
bps = ax.boxplot(T_data, tick_labels=labels_, patch_artist=True, widths=0.5)
for patch, color in zip(bps["boxes"], colors_):
    patch.set_facecolor(color); patch.set_alpha(0.75)
ax.set_ylabel("Étape de première découverte (mode R≥0.5)")
ax.set_title("T_success\n(bas = découverte rapide)", fontsize=10)
ax.set_ylim(0, N_STEPS + 100)
ax.tick_params(axis="x", labelsize=8)

plt.suptitle(
    "Étape A : Prior LM (target=24) appliqué sur target=42\n"
    "Exploration seule (Corollaire 1) vs Mesure de base (Proposition 1)\n"
    f"{N_SEEDS} seeds × {N_STEPS} étapes, batch={B_TRAIN}",
    fontsize=10, y=1.03
)
plt.tight_layout()
out = "/home/user/curly-enigma/step_a.png"
plt.savefig(out, dpi=130, bbox_inches="tight")
print(f"\nFigure → {out}")
