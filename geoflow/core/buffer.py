"""Trajectory replay buffer with priority weighting."""
from __future__ import annotations
from dataclasses import dataclass, field
from collections import deque
import numpy as np
import torch


@dataclass
class Trajectory:
    actions: list          # [a_0, ..., a_{D-1}]
    state: tuple           # terminal state tuple
    state_idx: int         # flat index in {0,...,N-1}
    log_reward: float      # log R(state)
    log_pf: float          # log P_F(tau) at time of collection


class ReplayBuffer:
    """Fixed-capacity FIFO buffer with reward-priority sampling.

    High-reward trajectories are sampled more often when
    priority_alpha > 0 (0 = uniform replay, 1 = fully reward-proportional).
    """

    def __init__(self, capacity: int = 2000, priority_alpha: float = 0.6):
        self.capacity = capacity
        self.alpha = priority_alpha
        self._buf: deque[Trajectory] = deque(maxlen=capacity)

    def add(self, traj: Trajectory) -> None:
        self._buf.append(traj)

    def add_batch(self, trajs: list[Trajectory]) -> None:
        for t in trajs:
            self._buf.append(t)

    def sample(self, n: int) -> list[Trajectory]:
        if len(self._buf) == 0:
            return []
        buf = list(self._buf)
        if self.alpha == 0:
            idx = np.random.choice(len(buf), size=min(n, len(buf)), replace=False)
        else:
            log_rs = np.array([t.log_reward for t in buf])
            log_rs -= log_rs.max()
            weights = np.exp(self.alpha * log_rs)
            weights /= weights.sum()
            idx = np.random.choice(len(buf), size=min(n, len(buf)),
                                   replace=True, p=weights)
        return [buf[i] for i in idx]

    def best(self) -> Trajectory | None:
        if not self._buf:
            return None
        return max(self._buf, key=lambda t: t.log_reward)

    def __len__(self) -> int:
        return len(self._buf)

    def state_indices(self, trajs: list[Trajectory]) -> torch.Tensor:
        return torch.tensor([t.state_idx for t in trajs], dtype=torch.long)

    def log_rewards(self, trajs: list[Trajectory]) -> torch.Tensor:
        return torch.tensor([t.log_reward for t in trajs], dtype=torch.float32)

    def log_pfs(self, trajs: list[Trajectory]) -> torch.Tensor:
        return torch.tensor([t.log_pf for t in trajs], dtype=torch.float32)
