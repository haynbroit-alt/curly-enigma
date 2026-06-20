"""
GFlowNet + compilateur Python — boucle complète vérifiable.

Tâche : générer des expressions arithmétiques de longueur 3
        (vocabulaire : chiffres 0-9 + opérateurs + *)
        dont le résultat est proche de 42.

Pourquoi intéressant :
  - L'espace (12^3 = 1728 séquences) est ENTIÈREMENT dénombrable.
  - On peut calculer la distribution cible exacte P(x) = R(x)/Z.
  - Le récompense est multimodale : "6*7", "7*6", "42" tous → R≈1.
  - Un RL glouton collapse sur "6*7" ; le GFlowNet doit couvrir tous les pics.
  - Le compilateur Python (eval) est la vraie récompense — pas de simulation.

C'est le pas intermédiaire honnête avant de brancher un LLM local :
  on vérifie que la boucle GFlowNet + compilateur fonctionne sur un espace
  assez petit pour avoir une ground truth exacte.
"""
import numpy as np
import torch
import torch.nn as nn
import subprocess

torch.manual_seed(0); np.random.seed(0)

# -----------------------------------------------------------------------
# 1. Espace des programmes
# -----------------------------------------------------------------------
VOCAB = list("0123456789+*")   # 12 tokens
N_TOK = 3                       # longueur fixe : 12^3 = 1728 séquences
V = len(VOCAB)
tok2id = {c: i for i, c in enumerate(VOCAB)}
id2tok = {i: c for c, i in tok2id.items()}

TARGET = 42.0                   # valeur cible

def safe_eval(tokens):
    """Évalue l'expression Python, renvoie (valeur, succès)."""
    expr = "".join(tokens)
    try:
        val = eval(expr, {"__builtins__": {}})  # sandbox minimal
        if not isinstance(val, (int, float)): return None, False
        if abs(val) > 1e6: return None, False
        return float(val), True
    except Exception:
        return None, False

def reward(tokens):
    """R(x) = exp(-|eval(x) - 42| / 3) si valide, 0.01 sinon."""
    val, ok = safe_eval(tokens)
    if not ok: return 0.01
    return float(np.exp(-abs(val - TARGET) / 3.0)) + 0.01

# -----------------------------------------------------------------------
# 2. Vérité terrain : énumérer les 1728 séquences
# -----------------------------------------------------------------------
print("Énumération des 1728 séquences...", flush=True)
ALL_SEQ = []
ALL_R   = []
for i in range(V):
    for j in range(V):
        for k in range(V):
            toks = [id2tok[i], id2tok[j], id2tok[k]]
            ALL_SEQ.append(toks)
            ALL_R.append(reward(toks))

ALL_R   = np.array(ALL_R, dtype=np.float64)
Z_TRUE  = ALL_R.sum()
P_TRUE  = ALL_R / Z_TRUE

# top modes
top5 = np.argsort(-ALL_R)[:5]
print("Top 5 expressions par récompense :")
for idx in top5:
    t = ALL_SEQ[idx]
    val, ok = safe_eval(t)
    print(f"  '{''.join(t)}'  val={val}  R={ALL_R[idx]:.4f}  P={P_TRUE[idx]:.4f}")

# expressions qui atteignent exactement 42
hits = [(i, ALL_SEQ[i]) for i in range(len(ALL_SEQ))
        if abs((safe_eval(ALL_SEQ[i])[0] or -999) - 42) < 1e-6]
print(f"\n{len(hits)} expressions qui donnent exactement 42 :")
for i, t in hits[:10]:
    print(f"  '{''.join(t)}'  R={ALL_R[i]:.4f}")

# -----------------------------------------------------------------------
# 3. Politique GFlowNet : MLP qui prédit le prochain token
# -----------------------------------------------------------------------
class Policy(nn.Module):
    def __init__(self, h=128):
        super().__init__()
        # état : one-hot des tokens déjà posés (N_TOK × V), + position
        in_dim = N_TOK * V + N_TOK
        self.net = nn.Sequential(
            nn.Linear(in_dim, h), nn.ReLU(),
            nn.Linear(h, h),      nn.ReLU(),
            nn.Linear(h, V),
        )
        self.log_Z = nn.Parameter(torch.zeros(()))

    def forward(self, states):   # [B, in_dim]
        return torch.log_softmax(self.net(states), dim=-1)

def encode(chosen, pos):
    """chosen : liste de token-ids déjà posés (longueur pos)."""
    v = np.zeros(N_TOK * V + N_TOK, dtype=np.float32)
    for i, tid in enumerate(chosen):
        v[i * V + tid] = 1.0
    v[N_TOK * V + pos] = 1.0
    return v

def rollout(policy, batch, eps):
    chosen = [[] for _ in range(batch)]
    sum_lpf = torch.zeros(batch)
    for pos in range(N_TOK):
        states = torch.tensor(
            np.stack([encode(chosen[b], pos) for b in range(batch)]))
        logp = policy(states)                        # [B, V]
        if np.random.rand() < eps:
            actions = torch.randint(0, V, (batch,))
        else:
            actions = torch.multinomial(logp.exp(), 1).squeeze(1)
        sum_lpf += logp.gather(1, actions[:, None]).squeeze(1)
        for b in range(batch):
            chosen[b].append(int(actions[b]))
    seqs = [[id2tok[tid] for tid in chosen[b]] for b in range(batch)]
    R = torch.tensor([reward(s) for s in seqs], dtype=torch.float32)
    return sum_lpf, R, seqs

# -----------------------------------------------------------------------
# 4. Entraînement — perte Trajectory Balance
# -----------------------------------------------------------------------
print("\nEntraînement GFlowNet (perte TB, 6000 étapes)...")
policy = Policy()
opt = torch.optim.Adam(policy.parameters(), lr=3e-3)

for step in range(6000):
    eps = max(0.05, 0.4 * (1 - step / 3000))
    slpf, R, _ = rollout(policy, 128, eps)
    residual = policy.log_Z + slpf - torch.log(R)
    loss = (residual ** 2).mean()
    opt.zero_grad(); loss.backward(); opt.step()
    if step % 1500 == 0:
        print(f"  step {step:5d}  TB={loss.item():.5f}  "
              f"logZ={policy.log_Z.item():.3f}  "
              f"Z_est={np.exp(policy.log_Z.item()):.2f}  "
              f"Z_vrai={Z_TRUE:.2f}")

# -----------------------------------------------------------------------
# 5. Évaluation : comparer P_appris à P_TRUE sur toutes les 1728 séquences
# -----------------------------------------------------------------------
print("\nÉvaluation sur les 1728 séquences...")
N_EVAL = 60000
counts = np.zeros(len(ALL_SEQ))
seq2idx = {"".join(s): i for i, s in enumerate(ALL_SEQ)}

with torch.no_grad():
    BATCH = 300
    for _ in range(N_EVAL // BATCH):
        _, _, seqs = rollout(policy, BATCH, eps=0.0)
        for s in seqs:
            key = "".join(s)
            if key in seq2idx:
                counts[seq2idx[key]] += 1

emp_p = counts / counts.sum()
l1   = np.abs(emp_p - P_TRUE).sum()
corr = np.corrcoef(emp_p, P_TRUE)[0, 1]

print(f"\n=== RÉSULTAT ===")
print(f"Z vrai              : {Z_TRUE:.3f}")
print(f"Z estimé (exp logZ) : {np.exp(policy.log_Z.item()):.3f}")
print(f"L1(appris, cible)   : {l1:.4f}   (0=parfait, 2=max)")
print(f"corrélation         : {corr:.5f}")

print(f"\nTop 8 expressions par récompense — cible vs appris :")
for idx in top5:
    t = ALL_SEQ[idx]
    val, _ = safe_eval(t)
    print(f"  '{''.join(t)}'  val={val}  cible={P_TRUE[idx]:.5f}  appris={emp_p[idx]:.5f}")

print(f"\nExpressions = 42 — couverture :")
total_hit_true = sum(P_TRUE[i] for i,_ in hits)
total_hit_emp  = sum(emp_p[i]  for i,_ in hits)
print(f"  Probabilité totale sur {{expr=42}} — cible:{total_hit_true:.4f}  appris:{total_hit_emp:.4f}")
for i, t in hits[:8]:
    print(f"  '{''.join(t)}'  cible={P_TRUE[i]:.5f}  appris={emp_p[i]:.5f}")

# -----------------------------------------------------------------------
# 6. Comparaison RL glouton (greedy argmax) — pour montrer le collapse
# -----------------------------------------------------------------------
print("\nComparaison RL glouton (argmax, pas de couverture) :")
greedy_counts = np.zeros(len(ALL_SEQ))
with torch.no_grad():
    for _ in range(N_EVAL // BATCH):
        chosen = [[] for _ in range(BATCH)]
        for pos in range(N_TOK):
            states = torch.tensor(
                np.stack([encode(chosen[b], pos) for b in range(BATCH)]))
            actions = policy(states).argmax(dim=1)
            for b in range(BATCH):
                chosen[b].append(int(actions[b]))
        for b in range(BATCH):
            key = "".join(id2tok[tid] for tid in chosen[b])
            if key in seq2idx:
                greedy_counts[seq2idx[key]] += 1
greedy_p = greedy_counts / max(greedy_counts.sum(), 1)
top_greedy = np.argmax(greedy_p)
print(f"  Toutes les évaluations greedy → '{''.join(ALL_SEQ[top_greedy])}'  "
      f"({greedy_p[top_greedy]*100:.0f}% du temps)")
print(f"  Modes couverts par greedy : {(greedy_p > 0.001).sum()} / {len(hits)} expressions=42")

# -----------------------------------------------------------------------
# 7. Figure
# -----------------------------------------------------------------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

# distributions
ax = axes[0]
xs = np.arange(len(ALL_SEQ))
ax.semilogy(xs, P_TRUE + 1e-7, lw=0.5, alpha=0.7, label="Cible P=R(x)/Z", c="#7aa2ff")
ax.semilogy(xs, emp_p  + 1e-7, lw=0.5, alpha=0.7, label="GFlowNet appris", c="#4caf84")
hit_idx = [i for i,_ in hits]
ax.scatter(hit_idx, P_TRUE[hit_idx], s=30, c="red", zorder=5, label="expr=42", marker="v")
ax.set_xlabel("Séquence (index)"); ax.set_ylabel("Prob. (log)")
ax.set_title(f"Distribution complète (1728 séquences)\nL1={l1:.4f}  corr={corr:.4f}", fontsize=10)
ax.legend(fontsize=8)

# scatter cible vs appris
ax = axes[1]
ax.scatter(P_TRUE, emp_p, s=8, alpha=0.4, c="#4caf84")
ax.scatter(P_TRUE[hit_idx], emp_p[hit_idx], s=40, c="red", zorder=5, label="expr=42")
mn = min(P_TRUE.min(), emp_p.min()); mx = max(P_TRUE.max(), emp_p.max())
ax.plot([mn,mx],[mn,mx],"k--",lw=1, label="x=y (parfait)")
ax.set_xlabel("Cible P(x)=R(x)/Z"); ax.set_ylabel("Appris P_F(x)")
ax.set_title("Corrélation cible ↔ appris", fontsize=10); ax.legend(fontsize=8)

# couverture des modes =42
ax = axes[2]
hit_true = P_TRUE[hit_idx]; hit_emp = emp_p[hit_idx]
labels_hit = ["".join(ALL_SEQ[i]) for i in hit_idx]
xh = np.arange(len(hit_idx))
ax.bar(xh - 0.2, hit_true, 0.35, label="Cible", color="#7aa2ff", alpha=0.85)
ax.bar(xh + 0.2, hit_emp,  0.35, label="GFlowNet", color="#4caf84", alpha=0.85)
ax.bar(xh,        greedy_p[hit_idx], 0.1, label="RL glouton", color="red", alpha=0.9)
ax.set_xticks(xh); ax.set_xticklabels(labels_hit, rotation=90, fontsize=8)
ax.set_ylabel("Probabilité"); ax.set_title("Couverture des modes = 42\n(RL glouton collapse sur 1)", fontsize=10)
ax.legend(fontsize=8)

plt.suptitle(f"GFlowNet + eval Python — expressions de {N_TOK} tokens vers 42\n"
             f"Espace: {V}^{N_TOK}={V**N_TOK} séquences, {len(hits)} solutions exactes",
             fontsize=10, y=1.02)
plt.tight_layout()
plt.savefig("/home/user/curly-enigma/gflownet_code_results.png", dpi=130, bbox_inches="tight")
print("\nFigure → gflownet_code_results.png")
