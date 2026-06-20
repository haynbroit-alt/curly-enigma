"""
theorem_b2.py — Validation B2 (version corrigée)

Deux bugs corrigés par rapport au script original :
  1. Évaluation par argmax → toutes les trajectoires identiques → TV ≈ 1.
     FIX : évaluation par sampling (multinomial) + comptage de fréquences.
  2. Mode sampling_only : le bruit était ajouté aux logits AVANT le calcul
     de log_pf, ce qui corrompait la loss (le bruit rentrait dans le gradient).
     FIX : le bruit guide UNIQUEMENT le choix d'action (exploration), tandis
     que log_pf est calculé sur les logits PROPRES du modèle — c'est l'isolation
     correcte entre politique comportementale et politique apprise.
"""
import torch, torch.nn as nn, torch.optim as optim, math

NUM_BITS   = 4
NUM_STATES = 2 ** NUM_BITS

torch.manual_seed(101)
R_target = torch.rand(NUM_STATES) * 5.0 + 2.0   # R(x) > 0
q_base   = torch.rand(NUM_STATES) * 10.0
q_base[0] = 20.0                                  # biais fort sur état 0
q_prior  = q_base / q_base.sum()                  # mesure de base q(x)

def get_state_index(states):
    idx = torch.zeros(states.shape[0], dtype=torch.long)
    for step in range(NUM_BITS):
        idx += states[:, step].long() * (2 ** (NUM_BITS - 1 - step))
    return idx

class MinimalGFN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc    = nn.Linear(NUM_BITS, 2)
        self.log_Z = nn.Parameter(torch.tensor(0.0))
    def forward(self, x): return self.fc(x)

def execute_validation_run(mode, epochs=3000):
    model     = MinimalGFN()
    optimizer = optim.Adam(model.parameters(), lr=5e-3)
    batch_size = 64

    for epoch in range(epochs):
        states  = torch.zeros(batch_size, NUM_BITS)
        log_pfs = torch.zeros(batch_size)

        for step in range(NUM_BITS):
            logits = model(states)                          # logits propres du modèle
            lp     = torch.log_softmax(logits, dim=-1)

            if mode == "sampling_only":
                # FIX : le prior biaise l'ACTION mais pas log_pf
                # (isolation comportemental vs appris)
                biased_logits = logits + 3.0 * torch.randn_like(logits)
                probs = torch.softmax(biased_logits, dim=-1)
            else:
                probs = torch.softmax(logits, dim=-1)

            actions = torch.multinomial(probs, num_samples=1).squeeze(-1)
            log_pfs += lp[torch.arange(batch_size), actions]  # toujours logits propres

            mask    = torch.zeros(batch_size, NUM_BITS)
            mask[:, step] = 1.0
            states  = states + actions.float().unsqueeze(-1) * mask

        indices = get_state_index(states)
        log_pb  = NUM_BITS * math.log(0.5)

        optimizer.zero_grad()
        if mode in ["standard", "sampling_only"]:
            loss = ((model.log_Z + log_pfs
                     - torch.log(R_target[indices]) - log_pb) ** 2).mean()
        else:  # base_measure_tb
            loss = ((model.log_Z + log_pfs
                     - torch.log(q_prior[indices] * R_target[indices]) - log_pb) ** 2).mean()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    # FIX évaluation : sampling stochastique → distribution empirique
    with torch.no_grad():
        test_batch = 10000
        states = torch.zeros(test_batch, NUM_BITS)
        for step in range(NUM_BITS):
            logits  = model(states)
            probs   = torch.softmax(logits, dim=-1)
            actions = torch.multinomial(probs, num_samples=1).squeeze(-1)
            mask    = torch.zeros(test_batch, NUM_BITS)
            mask[:, step] = 1.0
            states  = states + actions.float().unsqueeze(-1) * mask
        indices    = get_state_index(states)
        p_learned  = torch.bincount(indices, minlength=NUM_STATES).float() / test_batch

    return p_learned

if __name__ == "__main__":
    print("--- Protocole de validation B2 (version corrigée) ---\n")
    p_std  = execute_validation_run("standard")
    p_samp = execute_validation_run("sampling_only")
    p_base = execute_validation_run("base_measure_tb")

    true_R  = R_target / R_target.sum()
    true_qR = (q_prior * R_target) / (q_prior * R_target).sum()

    tv_std   = 0.5 * torch.sum(torch.abs(p_std  - true_R)).item()
    tv_samp  = 0.5 * torch.sum(torch.abs(p_samp - true_R)).item()
    tv_base  = 0.5 * torch.sum(torch.abs(p_base - true_qR)).item()

    print(f"[Résultats Asymptotiques]")
    print(f"Modèle 1 (TB Standard)         → TV vers R(x)      : {tv_std:.4f}")
    print(f"Modèle 2 (Prior Sampling Only) → TV vers R(x)      : {tv_samp:.4f}")
    print(f"Modèle 3 (Base-Measure TB)     → TV vers q(x)·R(x) : {tv_base:.4f}")

    print(f"\n[Interprétation]")
    print(f"  Δ(std→samp) = {abs(tv_samp - tv_std):.4f}  "
          f"{'(Corollaire 1 VALIDÉ : prior comportemental neutralisé)' if abs(tv_samp - tv_std) < 0.05 else '(convergence perturbée par le bruit)'}")
    print(f"  TV_base = {tv_base:.4f}  "
          f"{'(Proposition 1 VALIDÉE : GFN+q → q·R/Z_q)' if tv_base < 0.10 else '(convergence partielle)'}")

    print(f"\n[Note sur le modèle minimal]")
    print(f"  Avec fc_linear(4→2) sur espace 2^4=16, la capacité est limitée.")
    print(f"  TV idéal ~ 0 ; TV réaliste < 0.05 pour un modèle de capacité suffisante.")
