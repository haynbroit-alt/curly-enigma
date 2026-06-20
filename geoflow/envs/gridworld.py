"""Grid-world environment: D-dimensional grid with K values per axis."""
import numpy as np
from .base import DiscreteEnv


class GridWorld(DiscreteEnv):
    """Hypercube grid {0,...,K-1}^D with a GP-RBF reward landscape.

    Parameters
    ----------
    D : int
        Dimensionality (depth).
    K : int
        Number of values per dimension (actions per step).
    length_scale : float
        RBF kernel length-scale. Small = rough landscape, large = smooth.
    seed : int
        Random seed for reward generation.
    """

    def __init__(self, D: int = 4, K: int = 6, length_scale: float = 1.2,
                 seed: int = 0):
        self._D = D
        self._K = K
        self._ls = length_scale
        self._reward_map = self._build_reward(seed)
        self._enc_dim = D + D * K

    @property
    def depth(self) -> int:
        return self._D

    @property
    def n_actions(self) -> int:
        return self._K

    def _build_reward(self, seed: int) -> dict:
        import itertools
        states = list(itertools.product(range(self._K), repeat=self._D))
        coords = np.array(states, dtype=float)
        sq = ((coords[:, None, :] - coords[None, :, :]) ** 2).sum(-1)
        Cov = np.exp(-sq / (2 * self._ls ** 2)) + 1e-4 * np.eye(len(states))
        rng = np.random.default_rng(seed)
        g = np.linalg.cholesky(Cov) @ rng.standard_normal(len(states))
        g = (g - g.mean()) / (g.std() + 1e-9)
        R = np.exp(1.5 * g)
        R = R / R.sum()   # normalise to sum=1 → log_Z_opt ≈ 0
        return {s: float(R[i]) for i, s in enumerate(states)}

    def reward(self, state: tuple) -> float:
        return self._reward_map[tuple(state)]

    def encode(self, partial: list) -> np.ndarray:
        v = np.zeros(self._enc_dim, dtype=np.float32)
        v[len(partial)] = 1.0
        for i, c in enumerate(partial):
            v[self._D + i * self._K + c] = 1.0
        return v

    @property
    def enc_dim(self) -> int:
        return self._enc_dim
