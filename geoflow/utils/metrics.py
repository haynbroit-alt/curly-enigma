"""Evaluation metrics for GeoFlow-GFN."""
import numpy as np
import torch

from ..envs.base import DiscreteEnv
from ..core.gflownet import GFlowNetPolicy


def exact_policy_dist(policy: GFlowNetPolicy, env: DiscreteEnv) -> np.ndarray:
    """Enumerate all states and compute P_theta(x) exactly.

    Returns a (N,) numpy array normalised to sum to 1.
    Only valid for enumerable environments.
    """
    states = env.all_states()
    N = len(states)
    D = env.depth

    log_p = torch.zeros(N)
    with torch.no_grad():
        for t in range(D):
            prefixes = [list(s[:t]) for s in states]
            enc = torch.tensor(np.stack([env.encode(p) for p in prefixes]))
            lp = policy.log_probs(enc)          # (N, K)
            nxt = torch.tensor([s[t] for s in states])
            log_p += lp.gather(1, nxt[:, None]).squeeze(1)

    p = log_p.exp().numpy()
    return p / (p.sum() + 1e-30)


def tv_distance(p: np.ndarray, q: np.ndarray) -> float:
    """Total variation distance: 0.5 * sum |p(x) - q(x)|."""
    return float(0.5 * np.abs(p - q).sum())


def mode_coverage(p_theta: np.ndarray, reward_vec: np.ndarray,
                  threshold: float = 0.5) -> dict:
    """Fraction of high-reward modes covered by the policy.

    A mode is covered if p_theta(x) > 1/(10*N).

    Parameters
    ----------
    p_theta     : (N,) policy distribution
    reward_vec  : (N,) raw reward values
    threshold   : modes are states with reward >= threshold * max(reward)

    Returns
    -------
    dict with keys: n_modes, n_covered, coverage (fraction), p_on_modes
    """
    N = len(p_theta)
    R_max = reward_vec.max()
    mode_mask = reward_vec >= threshold * R_max
    n_modes = int(mode_mask.sum())
    n_covered = int((p_theta[mode_mask] > 1.0 / (10 * N)).sum())
    coverage = n_covered / max(1, n_modes)
    p_on_modes = float(p_theta[mode_mask].sum())
    return {
        "n_modes": n_modes,
        "n_covered": n_covered,
        "coverage": coverage,
        "p_on_modes": p_on_modes,
    }


def evaluate(policy: GFlowNetPolicy, env: DiscreteEnv,
             threshold: float = 0.5) -> dict:
    """Full evaluation for enumerable environments.

    Returns
    -------
    dict with keys: tv, coverage, n_modes, n_covered, p_on_modes
    """
    if not env.is_enumerable():
        raise ValueError("evaluate() requires an enumerable environment.")

    R_vec = env.reward_vector()
    p_R = R_vec / R_vec.sum()
    p_theta = exact_policy_dist(policy, env)

    tv = tv_distance(p_theta, p_R)
    cov = mode_coverage(p_theta, R_vec, threshold=threshold)

    return {"tv": tv, **cov}
