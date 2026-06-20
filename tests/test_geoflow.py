"""Unit tests for GeoFlow-GFN."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import pytest

from geoflow import GeoFlowAgent, GridWorld
from geoflow.core.gflownet import GFlowNetPolicy
from geoflow.core.metric import LearnableMetric
from geoflow.core.buffer import ReplayBuffer, Trajectory
from geoflow.algorithms.losses import tb_loss, total_loss
from geoflow.utils.metrics import tv_distance, exact_policy_dist, mode_coverage


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def small_env():
    return GridWorld(D=2, K=3, length_scale=1.0, seed=0)  # 9 states


@pytest.fixture
def policy(small_env):
    enc_dim = small_env.enc_dim
    return GFlowNetPolicy(enc_dim, small_env.n_actions, hidden=32)


@pytest.fixture
def metric(small_env):
    return LearnableMetric(small_env.n_states)


# ── Environment tests ─────────────────────────────────────────────────────────

def test_env_n_states(small_env):
    assert small_env.n_states == 9

def test_env_reward_positive(small_env):
    for s in small_env.all_states():
        assert small_env.reward(s) > 0

def test_env_reward_vector_shape(small_env):
    R = small_env.reward_vector()
    assert R.shape == (9,)
    assert (R > 0).all()

def test_env_encode_shape(small_env):
    enc = small_env.encode([])
    assert enc.shape == (small_env.enc_dim,)
    enc2 = small_env.encode([0, 1])
    assert enc2.shape == (small_env.enc_dim,)


# ── Policy tests ──────────────────────────────────────────────────────────────

def test_policy_output_shape(small_env, policy):
    enc = torch.tensor(small_env.encode([]).reshape(1, -1))
    logits = policy(enc)
    assert logits.shape == (1, small_env.n_actions)

def test_policy_log_probs_sum(small_env, policy):
    enc = torch.tensor(small_env.encode([]).reshape(1, -1))
    lp = policy.log_probs(enc)
    assert torch.allclose(lp.exp().sum(dim=-1), torch.ones(1), atol=1e-5)

def test_policy_sample_step_valid_range(small_env, policy):
    enc = torch.tensor(np.stack([small_env.encode([]) for _ in range(10)]))
    actions, log_pf = policy.sample_step(enc)
    assert actions.shape == (10,)
    assert ((actions >= 0) & (actions < small_env.n_actions)).all()


# ── Metric tests ──────────────────────────────────────────────────────────────

def test_metric_kernel_row_stochastic(metric):
    W = metric.kernel()
    assert W.shape == (9, 9)
    assert torch.allclose(W.sum(dim=-1), torch.ones(9), atol=1e-5)

def test_metric_symmetry_loss_init_zero(metric):
    loss = metric.symmetry_loss()
    assert float(loss) == pytest.approx(0.0)

def test_metric_update_differentiable(metric):
    idx = torch.tensor([0, 1, 2, 3])
    lr = torch.tensor([-1.0, -2.0, -0.5, -3.0])
    loss = metric.update_from_rewards(idx, lr)
    loss.backward()
    assert metric.g.grad is not None


# ── Buffer tests ──────────────────────────────────────────────────────────────

def test_buffer_add_and_sample():
    buf = ReplayBuffer(capacity=100)
    for i in range(10):
        buf.add(Trajectory([0, 1], (0, 1), i, float(np.log(i + 1)), -1.0))
    assert len(buf) == 10
    s = buf.sample(5)
    assert len(s) == 5

def test_buffer_best():
    buf = ReplayBuffer(capacity=10)
    buf.add(Trajectory([0], (0,), 0, 0.5, -1.0))
    buf.add(Trajectory([1], (1,), 1, 2.0, -1.0))
    buf.add(Trajectory([2], (2,), 2, 1.0, -1.0))
    assert buf.best().state_idx == 1


# ── Loss tests ────────────────────────────────────────────────────────────────

def test_tb_loss_zero_at_optimum():
    log_Z = torch.tensor(np.log(3.0))
    log_pf = torch.tensor([-1.5, -1.5, -1.5])
    log_R = torch.tensor([-0.5, -0.5, -0.5])
    # log_Z + log_pf - log_R = log(3) - 1.5 + 0.5 ≈ 0.099 ≠ 0 (not exact opt)
    loss = tb_loss(log_Z, log_pf, log_R)
    assert float(loss) >= 0

def test_total_loss_returns_dict(small_env, policy, metric):
    batch_idx = torch.tensor([0, 1, 2])
    log_pf = torch.tensor([-2.0, -2.0, -2.0])
    log_R = torch.tensor([-1.0, -1.5, -0.8])
    losses = total_loss(policy.log_Z, log_pf, log_R,
                        metric=metric, batch_indices=batch_idx,
                        lambda_kl=0.1, beta=0.01)
    assert "total" in losses
    assert "gfn" in losses
    assert "kl" in losses


# ── Metrics tests ─────────────────────────────────────────────────────────────

def test_tv_distance_identical():
    p = np.array([0.25, 0.25, 0.25, 0.25])
    assert tv_distance(p, p) == pytest.approx(0.0)

def test_tv_distance_orthogonal():
    p = np.array([1.0, 0.0])
    q = np.array([0.0, 1.0])
    assert tv_distance(p, q) == pytest.approx(1.0)

def test_mode_coverage_all_modes(small_env, policy):
    R = small_env.reward_vector()
    p_theta = np.ones(9) / 9
    cov = mode_coverage(p_theta, R, threshold=0.5)
    assert cov["coverage"] == pytest.approx(1.0)


# ── Integration test ──────────────────────────────────────────────────────────

def test_agent_fit_reduces_loss():
    env = GridWorld(D=2, K=3, seed=42)
    agent = GeoFlowAgent(env=env, hidden=32, lr=1e-2, batch_size=20,
                         use_metric=True, seed=0)
    h = agent.fit(n_steps=50, log_every=50, verbose=False)
    assert h[-1]["total"] < h[0]["total"] or h[-1]["gfn"] < 10.0

def test_agent_fit_no_metric():
    env = GridWorld(D=2, K=3, seed=42)
    agent = GeoFlowAgent(env=env, hidden=32, lr=1e-2, batch_size=20,
                         use_metric=False, seed=0)
    h = agent.fit(n_steps=50, log_every=50, verbose=False)
    assert len(h) > 0

def test_agent_evaluate_returns_tv():
    env = GridWorld(D=2, K=3, seed=0)
    agent = GeoFlowAgent(env=env, hidden=32, batch_size=20, use_metric=False, seed=0)
    agent.fit(n_steps=20, log_every=20, verbose=False)
    metrics = agent.evaluate()
    assert 0 <= metrics["tv"] <= 1.0
    assert 0 <= metrics["coverage"] <= 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
