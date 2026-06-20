"""Base environment interface for GeoFlow."""
from abc import ABC, abstractmethod
import numpy as np


class DiscreteEnv(ABC):
    """Abstract base for discrete sequential environments.

    States are built step by step: at each of D steps, choose from K actions.
    Terminal states are elements of {0,...,K-1}^D.
    """

    @property
    @abstractmethod
    def depth(self) -> int:
        """Number of sequential steps D."""

    @property
    @abstractmethod
    def n_actions(self) -> int:
        """Number of actions K per step."""

    @property
    def n_states(self) -> int:
        """Total number of terminal states K^D."""
        return self.n_actions ** self.depth

    @abstractmethod
    def reward(self, state: tuple) -> float:
        """Scalar reward for a terminal state. Must be > 0."""

    @abstractmethod
    def encode(self, partial: list) -> np.ndarray:
        """Encode a partial state (list of t actions) as a float32 vector."""

    def is_enumerable(self) -> bool:
        """Return True if n_states is small enough to enumerate exactly."""
        return self.n_states <= 100_000

    def all_states(self) -> list:
        """Return all terminal states as tuples (only for enumerable envs)."""
        import itertools
        return list(itertools.product(range(self.n_actions), repeat=self.depth))

    def reward_vector(self) -> np.ndarray:
        """Return R(x) for all states x (only for enumerable envs)."""
        states = self.all_states()
        return np.array([self.reward(s) for s in states], dtype=np.float64)
