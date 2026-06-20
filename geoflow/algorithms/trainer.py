"""Core training loop for GeoFlow-GFN."""
from __future__ import annotations
import numpy as np
import torch

from ..core.gflownet import GFlowNetPolicy
from ..core.metric import LearnableMetric
from ..core.buffer import ReplayBuffer, Trajectory
from ..envs.base import DiscreteEnv
from .losses import total_loss


class Trainer:
    """Online training loop: sample → store → update.

    Parameters
    ----------
    env          : DiscreteEnv instance
    policy       : GFlowNetPolicy
    metric       : LearnableMetric or None (ablation without metric)
    buffer       : ReplayBuffer
    lr           : learning rate for policy + log_Z
    lr_metric    : learning rate for metric g
    lambda_kl    : weight for KL coupling + metric update losses
    beta         : weight for smoothness regularisation
    batch_size   : trajectories per gradient step
    eps_start    : initial epsilon for exploration
    eps_end      : final epsilon
    eps_decay    : fraction of total steps over which eps decays
    """

    def __init__(self, env: DiscreteEnv, policy: GFlowNetPolicy,
                 metric: LearnableMetric | None = None,
                 buffer: ReplayBuffer | None = None,
                 lr: float = 5e-3, lr_metric: float = 1e-3,
                 lambda_kl: float = 0.1, beta: float = 0.01,
                 batch_size: int = 50,
                 eps_start: float = 0.4, eps_end: float = 0.05,
                 eps_decay: float = 0.6):
        self.env = env
        self.policy = policy
        self.metric = metric
        self.buffer = buffer or ReplayBuffer()
        self.lambda_kl = lambda_kl
        self.beta = beta
        self.batch_size = batch_size
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay

        params = list(policy.parameters())
        self.opt_policy = torch.optim.Adam(params, lr=lr)

        if metric is not None:
            self.opt_metric = torch.optim.Adam(metric.parameters(), lr=lr_metric)
        else:
            self.opt_metric = None

        self._step = 0
        self._total_steps = 1  # updated when training starts
        self._state_index: dict = {}
        if env.is_enumerable():
            states = env.all_states()
            for i, s in enumerate(states):
                self._state_index[s] = i
            # Structural initialisation: g[i,j] = L2(x_i, x_j) normalised.
            # This primes the metric so that nearby states (in feature space)
            # are metrically close even before any rewards are observed.
            if metric is not None:
                coords = np.array(states, dtype=np.float32)
                sq = ((coords[:, None, :] - coords[None, :, :]) ** 2).sum(-1)
                with torch.no_grad():
                    metric.g.data = torch.tensor(
                        sq / (sq.max() + 1e-9), dtype=torch.float32
                    )

    def _epsilon(self) -> float:
        progress = min(1.0, self._step / max(1, int(self._total_steps * self.eps_decay)))
        return max(self.eps_end, self.eps_start * (1 - progress))

    def _metric_targets(self, B: int) -> list | None:
        """Draw B target states from W[best_state, :] (metric-guided exploration).

        Returns a list of D-tuples, or None if metric unavailable.
        """
        if self.metric is None:
            return None
        best = self.buffer.best()
        if best is None or best.state_idx < 0:
            return None
        with torch.no_grad():
            w = self.metric.kernel()[best.state_idx]   # (N,) probs
        all_states = self.env.all_states()
        chosen_idx = torch.multinomial(w, num_samples=B, replacement=True)
        return [all_states[int(i)] for i in chosen_idx]

    def collect_batch(self, B: int) -> list[Trajectory]:
        """Roll out B trajectories under the current policy.

        During epsilon-exploration, if a metric is available, half the batch
        pursues metric-guided targets (states near the best found so far);
        the other half explores uniformly. log_pf is always from the clean policy.
        """
        env = self.env
        eps = self._epsilon()
        D, K = env.depth, env.n_actions

        # Metric-guided targets — active whenever buffer has data and metric exists.
        # (do not condition on eps so guidance is active even at minimum epsilon)
        targets = self._metric_targets(B) if (self.metric is not None and len(self.buffer) > 0) else None

        partials = [[] for _ in range(B)]
        log_pf_acc = torch.zeros(B)

        with torch.no_grad():
            for t in range(D):
                enc = torch.tensor(
                    np.stack([env.encode(p) for p in partials])
                )
                # Always compute log_pf from the clean policy
                lp = self.policy.log_probs(enc)

                if np.random.rand() < eps:
                    # Exploration: metric-guided if available, else uniform
                    if targets is not None:
                        actions = torch.tensor([tgt[t] for tgt in targets])
                    else:
                        actions = torch.randint(0, K, (B,))
                else:
                    actions = torch.multinomial(lp.exp(), num_samples=1).squeeze(1)

                log_pf_acc += lp.gather(1, actions[:, None]).squeeze(1)
                for b in range(B):
                    partials[b].append(int(actions[b]))

        trajs = []
        for b in range(B):
            state = tuple(partials[b])
            r = env.reward(state)
            log_r = float(np.log(max(r, 1e-30)))
            idx = self._state_index.get(state, -1)
            trajs.append(Trajectory(
                actions=partials[b],
                state=state,
                state_idx=idx,
                log_reward=log_r,
                log_pf=float(log_pf_acc[b]),
            ))
        return trajs

    def _compute_log_pf(self, trajs: list[Trajectory]) -> torch.Tensor:
        """Recompute log P_F(tau) under current policy (for gradient)."""
        env = self.env
        D, K = env.depth, env.n_actions
        B = len(trajs)
        log_pf = torch.zeros(B)

        for t in range(D):
            partials = [list(tr.actions[:t]) for tr in trajs]
            enc = torch.tensor(np.stack([env.encode(p) for p in partials]))
            lp = self.policy.log_probs(enc)
            actions_t = torch.tensor([tr.actions[t] for tr in trajs])
            log_pf += lp.gather(1, actions_t[:, None]).squeeze(1)

        return log_pf

    def _ema_metric_update(self, batch_indices: torch.Tensor,
                           log_reward: torch.Tensor, momentum: float = 0.7) -> None:
        """Direct EMA update of g from observed rewards.

        g[i, j] ← momentum * g[i, j] + (1-momentum) * |logR_i - logR_j|

        This bypasses slow gradient descent and lets the metric track the
        reward landscape after just a few batches. Gradient-based updates
        (for smoothness/KL) run on top of this.
        """
        with torch.no_grad():
            idx = batch_indices
            lr = log_reward
            target = (lr[:, None] - lr[None, :]).abs()   # (B, B)
            old = self.metric.g.data[idx[:, None], idx[None, :]]
            self.metric.g.data[idx[:, None], idx[None, :]] = (
                momentum * old + (1.0 - momentum) * target
            )

    def step(self) -> dict:
        """One training step: collect + update.

        Returns
        -------
        Dict with loss breakdown and current epsilon.
        """
        B = self.batch_size
        trajs = self.collect_batch(B)
        self.buffer.add_batch(trajs)

        log_pf = self._compute_log_pf(trajs)
        log_reward = torch.tensor([t.log_reward for t in trajs])

        batch_indices = None
        if self.metric is not None and all(t.state_idx >= 0 for t in trajs):
            batch_indices = torch.tensor([t.state_idx for t in trajs])
            # Fast direct metric update (before gradient step)
            self._ema_metric_update(batch_indices, log_reward, momentum=0.7)

        losses = total_loss(
            log_Z=self.policy.log_Z,
            log_pf=log_pf,
            log_reward=log_reward,
            metric=self.metric,
            batch_indices=batch_indices,
            lambda_kl=self.lambda_kl,
            beta=self.beta,
        )

        self.opt_policy.zero_grad()
        if self.opt_metric is not None:
            self.opt_metric.zero_grad()

        losses["total"].backward()

        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
        self.opt_policy.step()
        if self.opt_metric is not None:
            self.opt_metric.step()

        self._step += 1
        detach = lambda v: float(v.detach()) if hasattr(v, "detach") else float(v)
        return {k: detach(v) for k, v in losses.items()} | {"eps": self._epsilon()}

    def train(self, n_steps: int, log_every: int = 100,
              callback=None) -> list[dict]:
        """Train for n_steps gradient steps.

        Parameters
        ----------
        n_steps   : total number of gradient steps
        log_every : log metrics every this many steps
        callback  : callable(step, metrics) called at each log point

        Returns
        -------
        List of metric dicts at each log point.
        """
        self._total_steps = n_steps
        history = []

        for step in range(n_steps):
            metrics = self.step()
            if step % log_every == 0 or step == n_steps - 1:
                metrics["step"] = step
                history.append(metrics)
                if callback is not None:
                    callback(step, metrics)

        return history
