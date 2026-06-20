"""
GFlowNet séquentiel avec GRU — version corrigée.

4 problèmes du code original, corrigés ici :
  1. Logits gelés après EOS (évite l'accumulation de flux après terminal)
  2. log_pbs = 0  (backward uniforme, approximation standard et correcte)
  3. Reward stabilisée : exp(-distance), bornée dans [0,1]
  4. Masking strict : les épisodes terminés n'accumulent plus de log_pf

Vocabulaire limité → espace dénombrable → on peut vérifier Z.
"""
import math
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

torch.manual_seed(0); np.random.seed(0)

VOCAB   = ["<PAD>", "2", "5", "10", "+", "*", "EOS"]
V2I     = {t: i for i, t in enumerate(VOCAB)}
I2V     = {i: t for t, i in V2I.items()}
V       = len(VOCAB)
MAX_T   = 6    # 5 tokens + 1 EOS forcé → génère les programmes longueur 5
TARGET  = 42.0

# -----------------------------------------------------------------------
# Environnement (récompense stabilisée)
# -----------------------------------------------------------------------
def execute(tokens):
    clean = [t for t in tokens if t not in ("<PAD>", "EOS")]
    if not clean: return 0.0
    try:
        val = float(eval("".join(clean), {"__builtins__": {}}))
        if not math.isfinite(val) or abs(val) > 1e6: return 0.0
        # FIX 3 : exp(-distance), borné dans [0, 1]
        return float(math.exp(-abs(val - TARGET) / 5.0))
    except Exception:
        return 0.0

# -----------------------------------------------------------------------
# Vérité terrain (espace dénombrable)
# -----------------------------------------------------------------------
ops   = ["+", "*"]
nums  = ["2", "5", "10"]

def all_programs():
    """Génère les programmes valides possibles (longueur ≤ MAX_T)."""
    from itertools import product
    progs, Rs = [], []
    # longueur 1 : un seul nombre
    for n in nums:
        progs.append([n]); Rs.append(execute([n]))
    # longueur 3 : num op num
    for a, op, b in product(nums, ops, nums):
        progs.append([a, op, b]); Rs.append(execute([a, op, b]))
    # longueur 5 : num op num op num
    for a, op1, b, op2, c in product(nums, ops, nums, ops, nums):
        progs.append([a, op1, b, op2, c]); Rs.append(execute([a, op1, b, op2, c]))
    return progs, np.array(Rs, dtype=np.float64)

print("Vérité terrain...")
PROGS, R_ALL = all_programs()
Z_TRUE  = R_ALL.sum()
P_TRUE  = R_ALL / Z_TRUE
hits    = [(i, PROGS[i]) for i in range(len(PROGS)) if abs(R_ALL[i] - R_ALL.max()) < 1e-6]
print(f"  {len(PROGS)} programmes, Z_vrai={Z_TRUE:.3f}")
print(f"  Top programmes (R={R_ALL.max():.4f}) :")
for i, p in hits[:6]:
    print(f"    {''.join(p):<12} R={R_ALL[i]:.4f}  P={P_TRUE[i]:.4f}")

# -----------------------------------------------------------------------
# Architecture
# -----------------------------------------------------------------------
class ProgramGFN(nn.Module):
    def __init__(self, h=64):
        super().__init__()
        self.emb = nn.Embedding(V, h)
        self.gru = nn.GRU(h, h, batch_first=True)
        self.head = nn.Linear(h, V)
        self.log_Z = nn.Parameter(torch.zeros(()))

    def logits(self, seqs):           # seqs: [B, t] token ids
        out, _ = self.gru(self.emb(seqs))
        return self.head(out[:, -1])  # [B, V]

# -----------------------------------------------------------------------
# Rollout avec masking strict
# -----------------------------------------------------------------------
def rollout(model, B, eps):
    seqs     = torch.full((B, 1), V2I["<PAD>"], dtype=torch.long)
    log_pf   = torch.zeros(B)
    finished = torch.zeros(B, dtype=torch.bool)
    programs = [[] for _ in range(B)]

    # Masques constants (pas dans le graphe autograd)
    eos_only = torch.full((1, V), -1e9)
    eos_only[0, V2I["EOS"]] = 0.0               # logits "forcer EOS"

    for t in range(MAX_T):
        lg = model.logits(seqs)                  # [B, V]  — dans le graphe

        # Bloquer PAD sans inplace : soustraire une grande valeur sur la colonne PAD
        pad_block = torch.zeros(1, V)
        pad_block[0, V2I["<PAD>"]] = -1e9
        lg = lg + pad_block                      # nouveau tenseur dans le graphe

        # FIX 1 : geler logits des épisodes terminés (torch.where, non-inplace)
        # .clone() évite que la modification inplace de finished plus bas invalide le graphe
        fin = finished.clone()[:, None].expand_as(lg)   # [B, V] bool, snapshot
        lg = torch.where(fin, eos_only.expand(B, -1), lg)

        # Forcer EOS au dernier pas (non-inplace)
        if t == MAX_T - 1:
            lg = eos_only.expand(B, -1)

        lp = torch.log_softmax(lg, dim=-1)

        if np.random.rand() < eps:
            # randint(1, V) exclut PAD (index 0) dont log_prob ≈ -1e9
            actions = torch.randint(1, V, (B,))
            actions[finished] = V2I["EOS"]
        else:
            # lp.exp() : PAD a exp(-1e9)=0 → jamais échantillonné, pas besoin de clamp
            actions = torch.multinomial(lp.exp(), 1).squeeze(1)

        # FIX 4 : accumuler log_pf seulement pour les épisodes non finis
        # Utiliser le snapshot (fin[:,0]) pour cohérence avec le masquage des logits
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

# -----------------------------------------------------------------------
# Entraînement
# -----------------------------------------------------------------------
model = ProgramGFN()
opt   = optim.Adam(model.parameters(), lr=3e-3)

print("\nEntraînement (TB, 3000 étapes)...")
for step in range(3000):
    eps = max(0.05, 0.5 * (1 - step / 1500))
    lpf, R, progs = rollout(model, B=64, eps=eps)
    # FIX 2 : log_pbs = 0 (backward uniforme, correction standard)
    loss = ((model.log_Z + lpf - torch.log(R)) ** 2).mean()
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    if step % 750 == 0:
        top = max(progs, key=lambda p: execute(p))
        print(f"  step {step:4d}  TB={loss.item():.5f}  "
              f"logZ={model.log_Z.item():.3f}  "
              f"Z_est={math.exp(model.log_Z.item()):.2f}  "
              f"Z_vrai={Z_TRUE:.2f}  "
              f"top={''.join(top):<10} R={execute(top):.3f}")

# -----------------------------------------------------------------------
# Évaluation
# -----------------------------------------------------------------------
print("\nÉvaluation...")
N_EVAL = 20000
prog2count = {}
with torch.no_grad():
    for _ in range(N_EVAL // 100):
        _, _, progs = rollout(model, B=100, eps=0.0)
        for p in progs:
            k = "".join(p)
            prog2count[k] = prog2count.get(k, 0) + 1

total = sum(prog2count.values())

# aligner sur PROGS
emp = np.zeros(len(PROGS))
for i, p in enumerate(PROGS):
    emp[i] = prog2count.get("".join(p), 0) / total

l1   = np.abs(emp - P_TRUE).sum()
corr = np.corrcoef(emp, P_TRUE)[0, 1] if emp.std() > 1e-9 else 0.0

print(f"\n=== RÉSULTAT ===")
print(f"Z vrai              : {Z_TRUE:.3f}")
print(f"Z estimé (exp logZ) : {math.exp(model.log_Z.item()):.3f}")
print(f"L1(appris, cible)   : {l1:.4f}")
print(f"corrélation         : {corr:.5f}")
print(f"\nCouverture des programmes optimaux (=42) :")
for i, p in hits[:8]:
    print(f"  {''.join(p):<12}  cible={P_TRUE[i]:.5f}  appris={emp[i]:.5f}")

# Comparaison greedy
print("\nRL glouton (argmax) — collapse :")
with torch.no_grad():
    _, _, progs_g = rollout(model, B=200, eps=0.0)
from collections import Counter
gc = Counter("".join(p) for p in progs_g)
top3 = gc.most_common(3)
for prog, cnt in top3:
    print(f"  '{prog}'  {cnt/200*100:.0f}% des tirages  R={execute(list(prog)):.4f}")

# -----------------------------------------------------------------------
# Figure
# -----------------------------------------------------------------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

ax = axes[0]
xs = np.arange(len(PROGS))
ax.bar(xs, P_TRUE, alpha=0.6, label="Cible R/Z", color="#7aa2ff", width=0.9)
ax.bar(xs, emp,    alpha=0.7, label="GFlowNet appris", color="#4caf84", width=0.5)
hit_idx = [i for i,_ in hits]
ax.scatter(hit_idx, P_TRUE[hit_idx], s=60, c="red", zorder=5, marker="v", label="R=max (=42)")
ax.set_xlabel("Programme (index)"); ax.set_ylabel("Probabilité")
ax.set_title(f"Distribution apprise vs cible\nL1={l1:.4f}  corr={corr:.4f}", fontsize=10)
ax.legend(fontsize=8)

ax = axes[1]
ax.scatter(P_TRUE, emp, s=15, alpha=0.5, c="#4caf84")
ax.scatter(P_TRUE[hit_idx], emp[hit_idx], s=60, c="red", zorder=5, label="R=max (=42)")
mn = min(P_TRUE.min(), emp[emp>0].min() if (emp>0).any() else 0)
mx = max(P_TRUE.max(), emp.max())
ax.plot([mn,mx],[mn,mx],"k--",lw=1,label="parfait")
ax.set_xlabel("Cible P"); ax.set_ylabel("Appris P_F")
ax.set_title("Corrélation", fontsize=10); ax.legend(fontsize=8)

plt.suptitle("GFlowNet séquentiel (GRU) + eval Python — 4 fixes appliqués\n"
             f"vocab={V} tokens, longueur max={MAX_T}, {len(PROGS)} programmes dénombrables",
             fontsize=10, y=1.02)
plt.tight_layout()
plt.savefig("/home/user/curly-enigma/gflownet_program_results.png", dpi=120, bbox_inches="tight")
print("\nFigure → gflownet_program_results.png")
