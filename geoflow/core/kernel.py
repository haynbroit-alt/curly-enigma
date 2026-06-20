"""Transition kernel derived from the learnable metric."""
import torch
import torch.nn as nn

from .metric import LearnableMetric


class ExplorationKernel(nn.Module):
    """Wraps LearnableMetric to provide exploration biases for the policy.

    The kernel W encodes a preference over which states to visit next.
    During exploration, the policy is soft-biased toward states that
    the kernel predicts to be "rewarding neighbours" of visited states.
    """

    def __init__(self, metric: LearnableMetric, temperature: float = 1.0):
        super().__init__()
        self.metric = metric
        self.temperature = temperature

    def kl_coupling_loss(self, batch_indices: torch.Tensor) -> torch.Tensor:
        """KL(P_batch || W_marginal) coupling loss.

        Given a batch of visited states {x_i} (by their flat index),
        compute the empirical distribution P_batch and the kernel-induced
        marginal W_marginal[j] = mean_i W[x_i, j].

        Minimising this KL encourages the policy to visit states that the
        kernel points toward — i.e., states near already-found high-reward
        regions.

        Parameters
        ----------
        batch_indices : (B,) integer tensor of terminal-state flat indices.

        Returns
        -------
        kl : scalar tensor (differentiable through metric.g)
        """
        B = batch_indices.shape[0]
        N = self.metric.n_states

        # Empirical distribution over visited states
        p_batch = torch.zeros(N, device=batch_indices.device)
        p_batch.scatter_add_(0, batch_indices,
                             torch.ones(B, device=batch_indices.device))
        p_batch = p_batch / p_batch.sum()

        # Kernel-induced marginal: W_marginal[j] = mean_i W[x_i, j]
        log_W = self.metric.log_kernel()                  # (N, N)
        log_W_rows = log_W[batch_indices, :]              # (B, N)
        log_W_marginal = torch.logsumexp(
            log_W_rows - torch.log(torch.tensor(float(B))), dim=0
        )                                                 # (N,)
        W_marginal = log_W_marginal.exp()
        W_marginal = W_marginal / W_marginal.sum()

        # KL(P_batch || W_marginal), only over states with P_batch > 0
        mask = p_batch > 0
        kl = (p_batch[mask] * (p_batch[mask].log() - W_marginal[mask].log())).sum()
        return kl

    def exploration_logbias(self, partial_state_indices: torch.Tensor,
                            best_state_idx: int) -> torch.Tensor:
        """Log-bias toward the neighbourhood of the best known state.

        Used to soft-guide epsilon-exploration: during random steps,
        prefer states that the kernel predicts to be close to best_state_idx.

        Parameters
        ----------
        partial_state_indices : not used here (kernel is over full states).
            Placeholder for future hierarchical metric.
        best_state_idx : flat index of highest-reward state found so far.

        Returns
        -------
        log_bias : (N,) tensor — higher = prefer visiting this state
        """
        log_W = self.metric.log_kernel()       # (N, N)
        return log_W[best_state_idx, :] / self.temperature
