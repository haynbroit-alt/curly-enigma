"""GFlowNet policy network — sequential discrete actions."""
import numpy as np
import torch
import torch.nn as nn


class GFlowNetPolicy(nn.Module):
    """MLP forward policy for sequential discrete environments.

    Maps an encoded partial state to logits over K actions.
    Includes log_Z as a learnable scalar parameter.
    """

    def __init__(self, enc_dim: int, n_actions: int, hidden: int = 128):
        super().__init__()
        self.n_actions = n_actions
        self.net = nn.Sequential(
            nn.Linear(enc_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )
        self.log_Z = nn.Parameter(torch.zeros(()))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw logits for each action."""
        return self.net(x)

    def log_probs(self, enc_batch: torch.Tensor,
                  bias: torch.Tensor | None = None) -> torch.Tensor:
        """Log-softmax probabilities, optionally biased by the metric kernel.

        Parameters
        ----------
        enc_batch : (B, enc_dim) tensor
        bias : (B, K) tensor of log-scale additive bias (from metric kernel)

        Returns
        -------
        (B, K) log-probabilities
        """
        logits = self.forward(enc_batch)
        if bias is not None:
            logits = logits + bias
        return torch.log_softmax(logits, dim=-1)

    def sample_step(self, enc_batch: torch.Tensor, epsilon: float = 0.0,
                    bias: torch.Tensor | None = None) -> tuple:
        """Sample one action per item in the batch.

        Returns
        -------
        actions : (B,) int tensor
        log_pf  : (B,) log-probability of chosen action
        """
        lp = self.log_probs(enc_batch, bias=bias)
        B = enc_batch.shape[0]
        if epsilon > 0 and np.random.rand() < epsilon:
            actions = torch.randint(0, self.n_actions, (B,))
        else:
            actions = torch.multinomial(lp.exp(), num_samples=1).squeeze(1)
        log_pf = lp.gather(1, actions[:, None]).squeeze(1)
        return actions, log_pf
