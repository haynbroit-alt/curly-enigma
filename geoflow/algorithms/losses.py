"""Loss functions for GeoFlow-GFN training."""
import torch
import torch.nn.functional as F


def tb_loss(log_Z: torch.Tensor, log_pf: torch.Tensor,
            log_reward: torch.Tensor) -> torch.Tensor:
    """Trajectory Balance loss (Malkin et al., 2022).

    L_TB = E[(log_Z + log_P_F(tau) - log R(x))^2]

    Parameters
    ----------
    log_Z     : scalar learnable parameter
    log_pf    : (B,) log-probability of each trajectory
    log_reward: (B,) log R(x) for each terminal state

    Returns
    -------
    Scalar loss.
    """
    residual = log_Z + log_pf - log_reward
    return (residual ** 2).mean()


def tb_base_measure_loss(log_Z: torch.Tensor, log_pf: torch.Tensor,
                         log_q: torch.Tensor, log_reward: torch.Tensor,
                         log_N: float) -> torch.Tensor:
    """TB loss with base measure q (Theorem 1).

    L_TB+q = E[(log_Z + log_P_F(tau) - log q(x) - log R(x) - log N)^2]

    The log_N offset re-centres log_Z* to ≈ 0 when both q and R are
    normalised (see proof.tex, Remark on offset Lambda).
    """
    residual = log_Z + log_pf - log_q - log_reward - log_N
    return (residual ** 2).mean()


def kl_coupling_loss(p_batch: torch.Tensor,
                     w_marginal: torch.Tensor) -> torch.Tensor:
    """KL(P_batch || W_marginal) — encourages policy to visit kernel-preferred states.

    Parameters
    ----------
    p_batch    : (N,) empirical distribution over terminal states in the batch
    w_marginal : (N,) kernel-induced marginal W_marginal[j] = mean_i W[x_i, j]

    Returns
    -------
    Scalar KL divergence (differentiable through w_marginal).
    """
    mask = p_batch > 1e-9
    kl = (p_batch[mask] * (p_batch[mask].log() - (w_marginal[mask] + 1e-12).log())).sum()
    return kl


def smoothness_loss(g: torch.Tensor) -> torch.Tensor:
    """Regularise the metric: symmetry + bounded variance.

    L_smooth = ||g - g^T||^2 / N^2  +  Var(g)
    """
    sym = (g - g.T).pow(2).mean()
    var = g.var()
    return sym + var


def total_loss(log_Z, log_pf, log_reward,
               metric=None, batch_indices=None,
               lambda_kl: float = 0.1, beta: float = 0.01,
               use_base_measure: bool = False,
               log_q: torch.Tensor | None = None,
               log_N: float = 0.0) -> dict:
    """Compute all loss components and their weighted sum.

    Returns a dict with keys: total, gfn, kl, smooth, metric_update.
    """
    if use_base_measure and log_q is not None:
        l_gfn = tb_base_measure_loss(log_Z, log_pf, log_q, log_reward, log_N)
    else:
        l_gfn = tb_loss(log_Z, log_pf, log_reward)

    losses = {"gfn": l_gfn, "kl": torch.zeros(()), "smooth": torch.zeros(()),
              "metric_update": torch.zeros(())}

    if metric is not None and batch_indices is not None:
        # Metric update loss: make high-reward states metrically close
        l_metric = metric.update_from_rewards(batch_indices, log_reward)
        losses["metric_update"] = l_metric

        # KL coupling: policy marginal should match kernel marginal
        N = metric.n_states
        p_batch = torch.zeros(N)
        p_batch.scatter_add_(0, batch_indices,
                             torch.ones(batch_indices.shape[0]))
        p_batch = p_batch / p_batch.sum()

        log_W = metric.log_kernel()
        log_W_rows = log_W[batch_indices, :]
        log_W_marginal = torch.logsumexp(
            log_W_rows - torch.log(torch.tensor(float(batch_indices.shape[0]))),
            dim=0,
        )
        w_marginal = (log_W_marginal - torch.logsumexp(log_W_marginal, dim=0)).exp()

        l_kl = kl_coupling_loss(p_batch, w_marginal)
        l_smooth = smoothness_loss(metric.g)

        losses["kl"] = l_kl
        losses["smooth"] = l_smooth
        losses["metric_update"] = l_metric

    total = (l_gfn
             + lambda_kl * losses["kl"]
             + beta * losses["smooth"]
             + lambda_kl * losses["metric_update"])
    losses["total"] = total
    return losses
