"""
verify.py — Test empirique exact des deux 'theoremes' du papier colle.

Affirmation 1 (neutralisation) : prior DANS LA POLITIQUE, cible = R
   -> a l'optimum, P_theta -> R/Z  (le prior s'evapore)
Affirmation 2 (base measure)   : prior DANS LA CIBLE, cible = q*R
   -> P_theta -> q*R/Z

On prend un prior q DELIBEREMENT DESALIGNE de R (paysage independant) pour que,
si le prior survivait, la difference soit visible. Espace enumerable -> TV exacte.
Cle du test : on fait varier le BUDGET. La neutralisation est ASYMPTOTIQUE,
donc TV(B, R/Z) doit DESCENDRE vers 0 quand le budget augmente, tandis que
TV(B, qR/Z) reste grand (B n'echantillonne jamais q*R).
"""
import numpy as np, torch, torch.nn as nn, itertools
from scipy.special import logsumexp

D, K = 4, 4
STATES = list(itertools.product(range(K), repeat=D))
IDX = {s: i for i, s in enumerate(STATES)}
COORDS = np.array(STATES, float); N = len(STATES)
BATCH = 50
BUDGETS = [256, 1024, 4096, 16384]
SEEDS = [0, 1]
ALPHA = 2.0
SQ = ((COORDS[:, None, :] - COORDS[None, :, :]) ** 2).sum(-1)

def gp(ls, rng):
    Cov = np.exp(-SQ / (2 * ls ** 2)) + 1e-4 * np.eye(N)
    g = np.linalg.cholesky(Cov) @ rng.standard_normal(N)
    return (g - g.mean()) / (g.std() + 1e-9)

def worlds(seed):
    rng = np.random.default_rng(seed)
    gR, gQ = gp(1.0, rng), gp(1.0, rng)              # R et q independants (desalignes)
    R = np.exp(1.5 * gR);  pR = R / R.sum()
    QR = np.exp(1.5 * (gR + gQ)); pQR = QR / QR.sum()
    return gR, gQ, pR, pQR

class Net(nn.Module):
    def __init__(s, h=128):
        super().__init__()
        s.b = nn.Sequential(nn.Linear(D + D * K, h), nn.ReLU(),
                            nn.Linear(h, h), nn.ReLU(), nn.Linear(h, K))
        s.logZ = nn.Parameter(torch.zeros(()))
    def forward(s, x): return s.b(x)

def enc(p):
    v = np.zeros(D + D * K, np.float32); v[len(p)] = 1.0
    for i, c in enumerate(p): v[D + i * K + c] = 1.0
    return v

def prior_V(gQ):
    V = {s: float(gQ[IDX[s]]) for s in STATES}
    for t in range(D - 1, -1, -1):
        for p in itertools.product(range(K), repeat=t):
            V[p] = float(logsumexp([V[p + (a,)] for a in range(K)]))
    return V

def plogits(partials, V):
    return torch.tensor(np.array([[V[tuple(p) + (a,)] for a in range(K)] for p in partials], np.float32))

def logpF(net, partials, V, alpha):
    lg = net(torch.tensor(np.stack([enc(p) for p in partials])))
    if alpha > 0: lg = lg + alpha * plogits(partials, V)
    return torch.log_softmax(lg, -1)

def run(target_logp, V, alpha, budget, seed):
    """target_logp : log de la cible non normalisee, par etat (dict-like via index)."""
    torch.manual_seed(seed); np.random.seed(seed)
    net = Net(); opt = torch.optim.Adam(net.parameters(), lr=5e-3); used = 0
    tl = {s: target_logp[IDX[s]] for s in STATES}
    while used < budget:
        eps = max(0.05, 0.4 * (1 - used / (budget * 0.6 + 1)))
        partials = [[] for _ in range(BATCH)]; lpf = torch.zeros(BATCH)
        for t in range(D):
            lp = logpF(net, partials, V, alpha)
            a = (torch.randint(0, K, (BATCH,)) if np.random.rand() < eps
                 else torch.multinomial(lp.exp(), 1).squeeze(1))
            lpf = lpf + lp.gather(1, a[:, None]).squeeze(1)
            for b in range(BATCH): partials[b].append(int(a[b]))
        tgt = torch.tensor([tl[tuple(p)] for p in partials], dtype=torch.float32)
        loss = ((net.logZ + lpf - tgt) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step(); used += BATCH
    logp = torch.zeros(N)
    with torch.no_grad():
        for t in range(D):
            pre = [list(s[:t]) for s in STATES]
            lp = logpF(net, pre, V, alpha)
            nx = torch.tensor([s[t] for s in STATES])
            logp = logp + lp.gather(1, nx[:, None]).squeeze(1)
    p = logp.exp().numpy(); return p / p.sum()

def TV(a, b): return 0.5 * np.abs(a - b).sum()

print(f"espace {N} etats | prior q DESALIGNE de R | alpha={ALPHA}\n")
print(f"{'budget':>7} | {'A: TV->R/Z':>11} | {'B(beh): TV->R/Z':>15} | {'B: TV->qR/Z':>12} | {'C(base): TV->qR/Z':>17}")
print("-" * 78)
for budget in BUDGETS:
    A_R, B_R, B_QR, C_QR = [], [], [], []
    for seed in SEEDS:
        gR, gQ, pR, pQR = worlds(seed); V = prior_V(gQ)
        logR = np.log([pR[IDX[s]] for s in STATES])          # cible R
        logQR = np.log([pQR[IDX[s]] for s in STATES])        # cible q*R
        pA = run(logR, None, 0.0, budget, seed)              # A : standard
        pB = run(logR, V,   ALPHA, budget, seed)             # B : prior dans la politique
        pC = run(logQR, None, 0.0, budget, seed)             # C : prior dans la cible
        A_R.append(TV(pA, pR)); B_R.append(TV(pB, pR))
        B_QR.append(TV(pB, pQR)); C_QR.append(TV(pC, pQR))
    print(f"{budget:7d} | {np.mean(A_R):11.3f} | {np.mean(B_R):15.3f} | "
          f"{np.mean(B_QR):12.3f} | {np.mean(C_QR):17.3f}")

print("\nLecture attendue si le papier a raison :")
print(" - B: TV->R/Z descend vers 0 quand le budget grandit (le prior s'evapore)")
print(" - B: TV->qR/Z reste GRAND (B n'echantillonne jamais q*R)")
print(" - C: TV->qR/Z petit (le prior dans la cible, lui, change le point fixe)")
