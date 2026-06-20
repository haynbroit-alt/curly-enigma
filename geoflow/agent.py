"""GeoFlowAgent — public API for GeoFlow-GFN."""
from __future__ import annotations
import numpy as np
import torch

from .envs.base import DiscreteEnv
from .envs.gridworld import GridWorld
from .core.gflownet import GFlowNetPolicy
from .core.metric import LearnableMetric
from .core.buffer import ReplayBuffer, Trajectory
from .algorithms.trainer import Trainer
from .algorithms.losses import total_loss
from .utils.metrics import evaluate, exact_policy_dist
from .callbacks.base import Callback


class GeoFlowAgent:
    """Adaptive-exploration GFlowNet with learnable metric kernel.

    The agent jointly trains:
      1. A GFlowNet policy (TB loss) to match the reward distribution.
      2. A metric g over terminal states so that high-reward states are
         metrically close and the exploration kernel W = softmax(-g)
         guides future trajectories toward promising regions.

    The coupling loss KL(π || W) + metric_update_loss is added to the
    standard TB loss, scaled by lambda_kl and beta.

    Parameters
    ----------
    env          : DiscreteEnv instance, or None to use a default GridWorld.
    hidden       : hidden size of the GFlowNet MLP.
    lr           : learning rate for policy + log_Z.
    lr_metric    : learning rate for metric g.
    lambda_kl    : weight for KL coupling and metric update losses.
    beta         : weight for smoothness regularisation of g.
    buffer_size  : replay buffer capacity (currently not used for replay,
                   only for bookkeeping / best-state tracking).
    batch_size   : trajectories per gradient step.
    use_metric   : if False, trains a plain GFlowNet (ablation baseline).
    seed         : random seed.

    Example
    -------
    >>> from geoflow import GeoFlowAgent
    >>> agent = GeoFlowAgent(n_states=256, lambda_kl=0.1, beta=0.01)
    >>> for step in range(1000):
    ...     traj  = agent.sample()
    ...     reward = env.evaluate(traj)
    ...     loss   = agent.update(traj, reward)
    """

    def __init__(self,
                 env: DiscreteEnv | None = None,
                 n_states: int | None = None,     # convenience: ignored if env given
                 hidden: int = 128,
                 lr: float = 5e-3,
                 lr_metric: float = 1e-3,
                 lambda_kl: float = 0.1,
                 beta: float = 0.01,
                 buffer_size: int = 2000,
                 batch_size: int = 50,
                 use_metric: bool = True,
                 seed: int = 0):

        torch.manual_seed(seed)
        np.random.seed(seed)

        # Default env: 4-D grid with 4 actions per step (N=256)
        if env is None:
            D = 4
            K = int(round(n_states ** (1 / D))) if n_states else 4
            env = GridWorld(D=D, K=K, seed=seed)

        self.env = env
        self.use_metric = use_metric

        enc_dim = getattr(env, "enc_dim", env.depth + env.depth * env.n_actions)
        self.policy = GFlowNetPolicy(enc_dim, env.n_actions, hidden=hidden)

        self.metric = LearnableMetric(env.n_states) if use_metric and env.is_enumerable() else None

        self.buffer = ReplayBuffer(capacity=buffer_size)
        self.trainer = Trainer(
            env=env, policy=self.policy, metric=self.metric,
            buffer=self.buffer,
            lr=lr, lr_metric=lr_metric,
            lambda_kl=lambda_kl, beta=beta,
            batch_size=batch_size,
        )

        self._state_index: dict = {}
        if env.is_enumerable():
            for i, s in enumerate(env.all_states()):
                self._state_index[s] = i

    # ── High-level API ────────────────────────────────────────────────────────

    def sample(self, n: int = 1, epsilon: float | None = None) -> list[tuple]:
        """Sample n terminal states under the current policy.

        Parameters
        ----------
        n       : number of trajectories
        epsilon : exploration rate; None = use trainer's schedule

        Returns
        -------
        List of terminal state tuples.
        """
        trajs = self.trainer.collect_batch(n)
        return [t.state for t in trajs]

    def update(self, states: list[tuple], rewards: list[float]) -> dict:
        """One gradient step given externally provided (state, reward) pairs.

        This is an alternative to agent.fit() for custom environments where
        the reward function is external (e.g. a protein oracle).

        Parameters
        ----------
        states  : list of terminal state tuples
        rewards : corresponding reward values (must be > 0)

        Returns
        -------
        Dict with loss breakdown (total, gfn, kl, smooth, metric_update).
        """
        trajs = []
        for state, r in zip(states, rewards):
            idx = self._state_index.get(tuple(state), -1)
            # Recompute log_pf for this state
            lp = self._log_pf_for_state(state)
            trajs.append(Trajectory(
                actions=list(state),
                state=tuple(state),
                state_idx=idx,
                log_reward=float(np.log(max(r, 1e-30))),
                log_pf=lp,
            ))
        self.buffer.add_batch(trajs)

        log_pf = self.trainer._compute_log_pf(trajs)
        log_reward = torch.tensor([t.log_reward for t in trajs])
        batch_indices = (
            torch.tensor([t.state_idx for t in trajs])
            if self.metric and all(t.state_idx >= 0 for t in trajs)
            else None
        )

        losses = total_loss(
            log_Z=self.policy.log_Z,
            log_pf=log_pf,
            log_reward=log_reward,
            metric=self.metric,
            batch_indices=batch_indices,
            lambda_kl=self.trainer.lambda_kl,
            beta=self.trainer.beta,
        )

        self.trainer.opt_policy.zero_grad()
        if self.trainer.opt_metric:
            self.trainer.opt_metric.zero_grad()
        losses["total"].backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
        self.trainer.opt_policy.step()
        if self.trainer.opt_metric:
            self.trainer.opt_metric.step()

        return {k: float(v) for k, v in losses.items()}

    def fit(self, n_steps: int, log_every: int = 200,
            verbose: bool = True,
            callbacks: "list[Callback] | None" = None) -> list[dict]:
        """Train for n_steps gradient steps using the internal env reward.

        Parameters
        ----------
        n_steps   : number of gradient steps
        log_every : log metrics every this many steps
        verbose   : print step summary to stdout
        callbacks : list of Callback instances executed at each log point

        Returns
        -------
        Training history: list of metric dicts (one per log point).
        """
        cbs = callbacks or []
        for cb in cbs:
            cb.on_train_start(self, n_steps)

        prefix = "GeoFlow" if self.use_metric else "GFN-std"

        def _cb(step, metrics):
            metrics["_n_steps"] = n_steps
            if verbose:
                parts = f"loss={metrics['total']:.4f} | gfn={metrics['gfn']:.4f}"
                if self.use_metric:
                    parts += f" | kl={metrics['kl']:.4f} | smooth={metrics['smooth']:.4f}"
                parts += f" | eps={metrics['eps']:.3f}"
                print(f"[{prefix}] step={step:6d} | {parts}", flush=True)
            for cb in cbs:
                cb.on_step_end(self, step, metrics)
            metrics.pop("_n_steps", None)

        history = self.trainer.train(
            n_steps, log_every=log_every, callback=_cb,
            stop_fn=lambda: any(cb.should_stop() for cb in cbs),
        )
        for cb in cbs:
            cb.on_train_end(self, history)
        return history

    def evaluate(self, threshold: float = 0.5) -> dict:
        """Compute exact TV and mode coverage (enumerable envs only).

        Returns
        -------
        Dict with keys: tv, n_modes, n_covered, coverage, p_on_modes.
        """
        return evaluate(self.policy, self.env, threshold=threshold)

    def policy_dist(self) -> np.ndarray:
        """Return the exact policy distribution P_theta(x) over all states."""
        return exact_policy_dist(self.policy, self.env)

    # ── Internals ─────────────────────────────────────────────────────────────

    def _log_pf_for_state(self, state: tuple) -> float:
        """Compute log P_F(tau) for the deterministic trajectory to `state`."""
        env = self.env
        partial = []
        lp_total = 0.0
        with torch.no_grad():
            for a in state:
                enc = torch.tensor(env.encode(partial)).unsqueeze(0)
                lp = self.policy.log_probs(enc)[0]
                lp_total += float(lp[a])
                partial.append(a)
        return lp_total
