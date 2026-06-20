"""
GFlowNet minimal — la brique fondamentale, vérifiable.

Objectif : prouver empiriquement la SEULE propriété qui justifie un GFlowNet
plutôt qu'un RL classique :

    apres entrainement,  P_F(x) ~ R(x) / Z   pour TOUS les modes, pas seulement le max.

Tache jouet : sequences de N_BITS bits, construites bit par bit (DAG a chemin unique).
Recompense multimodale (3 pics separes) -> un RL glouton collapse sur 1 pic,
un vrai GFlowNet doit couvrir les 3 proportionnellement.

Pas de LLM, pas de LoRA, pas de sandbox : juste le coeur, pour qu'on puisse
comparer la distribution apprise a la verite terrain calculee exactement.
"""
import numpy as np
import torch
import torch.nn as nn

torch.manual_seed(0); np.random.seed(0)

N_BITS = 6
ALL = [np.array(list(map(int, format(i, f"0{N_BITS}b")))) for i in range(2 ** N_BITS)]

# --- Recompense multimodale (3 cibles separees) -----------------------------
TARGETS = [np.array([1,1,1,1,1,1]), np.array([0,0,0,0,0,0]), np.array([1,0,1,0,1,0])]
def reward(x):
    r = 0.05
    for t in TARGETS:
        d = np.sum(x != t)          # distance de Hamming a la cible
        r += 3.0 * np.exp(-1.2 * d) # pic exponentiel
    return float(r)

R = np.array([reward(x) for x in ALL])
Z_TRUE = R.sum()
TARGET_P = R / Z_TRUE               # verite terrain : distribution cible exacte

# --- Politique forward : petit MLP -> proba du prochain bit -----------------
# Etat encode sur 2*N_BITS : [bits poses, masque "deja decide"]
class Policy(nn.Module):
    def __init__(self, h=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * N_BITS, h), nn.ReLU(),
            nn.Linear(h, h), nn.ReLU(),
            nn.Linear(h, 2),                      # logits pour bit=0 / bit=1
        )
        self.log_Z = nn.Parameter(torch.zeros(()))  # estimateur de log Z

    def forward(self, state):                       # state: [B, 2*N_BITS]
        return torch.log_softmax(self.net(state), dim=-1)

def encode(bits, t):
    """bits: liste des bits poses (longueur t). Renvoie le vecteur d'etat."""
    v = np.zeros(2 * N_BITS, dtype=np.float32)
    for i in range(t):
        v[i] = bits[i]
        v[N_BITS + i] = 1.0
    return v

def rollout(policy, batch, eps):
    """Echantillonne `batch` trajectoires. eps = part d'exploration uniforme.
       Renvoie : somme des log P_F le long de chaque trajectoire (sous la politique),
       et la recompense terminale. TB est valable hors-politique : on credite
       toujours le log-prob DE LA POLITIQUE pour l'action reellement prise."""
    sum_logpf = torch.zeros(batch)
    xs = [[] for _ in range(batch)]
    for t in range(N_BITS):
        states = torch.tensor(np.stack([encode(xs[b], t) for b in range(batch)]))
        logp = policy(states)                       # [B, 2]
        probs = logp.exp()
        if np.random.rand() < eps:                  # exploration : action uniforme
            actions = torch.randint(0, 2, (batch,))
        else:
            actions = torch.multinomial(probs, 1).squeeze(1)
        sum_logpf = sum_logpf + logp.gather(1, actions[:, None]).squeeze(1)
        for b in range(batch):
            xs[b].append(int(actions[b]))
    rewards = torch.tensor([reward(np.array(x)) for x in xs], dtype=torch.float32)
    return sum_logpf, rewards

# --- Entrainement : perte Trajectory Balance --------------------------------
# Chemin unique => P_B = 1 => log P_B = 0. Residu = log_Z + sum log P_F - log R.
policy = Policy()
opt = torch.optim.Adam(policy.parameters(), lr=5e-3)

for step in range(4000):
    eps = max(0.05, 0.5 * (1 - step / 2000))        # exploration decroissante
    sum_logpf, rewards = rollout(policy, batch=64, eps=eps)
    residual = policy.log_Z + sum_logpf - torch.log(rewards)
    loss = (residual ** 2).mean()
    opt.zero_grad(); loss.backward(); opt.step()
    if step % 1000 == 0:
        print(f"step {step:4d} | TB loss {loss.item():.4f} | logZ {policy.log_Z.item():.3f}")

# --- Evaluation : la distribution apprise colle-t-elle a R/Z ? --------------
N_SAMPLES = 40000
counts = np.zeros(2 ** N_BITS)
with torch.no_grad():
    for _ in range(N_SAMPLES // 200):
        bits = [[] for _ in range(200)]
        for t in range(N_BITS):
            states = torch.tensor(np.stack([encode(bits[b], t) for b in range(200)]))
            a = torch.multinomial(policy(states).exp(), 1).squeeze(1)
            for b in range(200):
                bits[b].append(int(a[b]))
        for b in range(200):
            idx = int("".join(map(str, bits[b])), 2)
            counts[idx] += 1
emp_p = counts / counts.sum()

l1 = np.abs(emp_p - TARGET_P).sum()
corr = np.corrcoef(emp_p, TARGET_P)[0, 1]
Z_est = float(np.exp(policy.log_Z.item()))

print("\n=== RESULTAT ===")
print(f"Z vrai            : {Z_TRUE:.3f}")
print(f"Z estime (exp logZ): {Z_est:.3f}")
print(f"L1(distrib apprise, cible) : {l1:.3f}   (0 = parfait, 2 = max)")
print(f"correlation                : {corr:.4f}")
print("\nTop 5 etats par recompense — cible vs appris :")
order = np.argsort(-R)[:5]
for i in order:
    print(f"  {format(i,f'0{N_BITS}b')}  R={R[i]:5.2f}  cible={TARGET_P[i]:.4f}  appris={emp_p[i]:.4f}")

# couverture des 3 modes
mode_idx = [int("".join(map(str, t)), 2) for t in TARGETS]
print("\nCouverture des 3 modes (un RL glouton n'en garderait qu'un) :")
for mi, t in zip(mode_idx, TARGETS):
    print(f"  mode {format(mi,f'0{N_BITS}b')}  cible={TARGET_P[mi]:.4f}  appris={emp_p[mi]:.4f}")

# --- Figure -----------------------------------------------------------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

ax = axes[0]
xs = np.arange(2 ** N_BITS)
ax.bar(xs, TARGET_P, alpha=0.6, label="Cible R(x)/Z", color="#7aa2ff", width=0.9)
ax.bar(xs, emp_p, alpha=0.6, label="GFlowNet appris", color="#4caf84", width=0.5)
for mi in mode_idx:
    ax.axvline(mi, ls="--", c="red", lw=1, alpha=0.5)
ax.set_xlabel("Sequence (index binaire)"); ax.set_ylabel("Probabilite")
ax.set_title(f"Distribution apprise vs cible\nL1={l1:.3f}  corr={corr:.4f}", fontsize=10)
ax.legend(fontsize=9)

ax = axes[1]
ax.scatter(TARGET_P, emp_p, s=18, alpha=0.7, c="#4caf84")
mn, mx = min(TARGET_P.min(), emp_p.min()), max(TARGET_P.max(), emp_p.max())
ax.plot([mn, mx], [mn, mx], "k--", lw=1, label="x=y (parfait)")
for mi, t in zip(mode_idx, TARGETS):
    ax.annotate(format(mi, f"0{N_BITS}b"), (TARGET_P[mi], emp_p[mi]),
                fontsize=7, ha="left", va="bottom")
ax.set_xlabel("Cible P(x) = R(x)/Z"); ax.set_ylabel("Appris P_F(x)")
ax.set_title("Corrélation cible ↔ appris", fontsize=10)
ax.legend(fontsize=9)

plt.suptitle("GFlowNet minimal — Trajectory Balance\n"
             f"N_BITS={N_BITS}, 3 modes, 4000 étapes", fontsize=11, y=1.02)
plt.tight_layout()
plt.savefig("/home/user/curly-enigma/gflownet_results.png", dpi=120, bbox_inches="tight")
print("\nFigure → gflownet_results.png")
