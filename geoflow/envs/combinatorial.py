"""Combinatorial environment: evaluate arithmetic expressions."""
import numpy as np
from .base import DiscreteEnv


class ArithmeticEnv(DiscreteEnv):
    """Generate arithmetic expressions and evaluate their proximity to a target.

    Tokens: ["2", "5", "10", "+", "*", "EOS"]
    Valid expression: sequence of non-EOS tokens followed by EOS.
    Reward: exp(-|eval(expr) - target| / scale)
    """

    VOCAB = ["2", "5", "10", "+", "*", "EOS"]
    EOS_IDX = 5

    def __init__(self, target: float = 42.0, scale: float = 5.0,
                 max_tokens: int = 6):
        self._target = target
        self._scale = scale
        self._max_tokens = max_tokens
        self._D = max_tokens
        self._K = len(self.VOCAB)

    @property
    def depth(self) -> int:
        return self._D

    @property
    def n_actions(self) -> int:
        return self._K

    def reward(self, state: tuple) -> float:
        tokens = [self.VOCAB[a] for a in state if a != self.EOS_IDX]
        expr = " ".join(tokens)
        try:
            val = float(eval(expr))  # noqa: S307 — controlled token set
            return float(np.exp(-abs(val - self._target) / self._scale))
        except Exception:
            return 1e-6

    def encode(self, partial: list) -> np.ndarray:
        v = np.zeros(self._D + self._D * self._K, dtype=np.float32)
        v[len(partial)] = 1.0
        for i, tok in enumerate(partial):
            v[self._D + i * self._K + tok] = 1.0
        return v

    @property
    def enc_dim(self) -> int:
        return self._D + self._D * self._K

    def is_enumerable(self) -> bool:
        return False

    def decode(self, state: tuple) -> str:
        tokens = [self.VOCAB[a] for a in state if a != self.EOS_IDX]
        return " ".join(tokens)
