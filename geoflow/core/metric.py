"""Learnable metric over terminal states."""
import torch
import torch.nn as nn


class LearnableMetric(nn.Module):
    """Pairwise distance matrix over N terminal states.

    g[i, j] represents the "distance" between state i and state j.
    The transition kernel derived from this metric is:
        W[i, j] = softmax(-g[i, :])[j]

    For small spaces (N ≲ 10_000), g is a direct N×N parameter.
    """

    def __init__(self, n_states: int):
        super().__init__()
        self.n_states = n_states
        self.g = nn.Parameter(torch.zeros(n_states, n_states))

    def kernel(self) -> torch.Tensor:
        """Row-stochastic kernel W[i, j] = softmax(-g[i, :])[j]."""
        return torch.softmax(-self.g, dim=-1)

    def log_kernel(self) -> torch.Tensor:
        """Log-probabilities log W[i, j]."""
        return torch.log_softmax(-self.g, dim=-1)

    def symmetry_loss(self) -> torch.Tensor:
        """Penalise asymmetry: encourage g ≈ g^T."""
        return (self.g - self.g.T).pow(2).mean()

    def smoothness_loss(self) -> torch.Tensor:
        """Penalise large off-diagonal variance (spiky distances)."""
        return self.g.var()

    def update_from_rewards(self, state_indices: torch.Tensor,
                            log_rewards: torch.Tensor,
                            lr_metric: float = 0.1) -> torch.Tensor:
        """Soft-update g so that high-reward states are metrically close.

        For states i, j in the batch:
          target_dist(i, j) = |log R(i) - log R(j)|
          metric_loss = mean_{i,j} (g[i,j] - target_dist(i,j))^2

        Returns the metric loss (scalar tensor, differentiable through g).
        """
        idx = state_indices  # (B,)
        lr = log_rewards     # (B,)
        target = (lr[:, None] - lr[None, :]).abs()   # (B, B)
        g_sub = self.g[idx[:, None], idx[None, :]]   # (B, B)
        return (g_sub - target.detach()).pow(2).mean()
