"""Built-in callbacks for GeoFlow-GFN."""
from __future__ import annotations
import csv
import json
import time
from pathlib import Path
from typing import TYPE_CHECKING

from .base import Callback

if TYPE_CHECKING:
    from ..agent import GeoFlowAgent


class CoverageTracker(Callback):
    """Evaluates mode coverage and TV at regular intervals.

    Adds 'tv', 'coverage', 'n_covered', 'n_modes', 'p_on_modes'
    to the metrics dict at each eval point.

    Parameters
    ----------
    eval_every  : evaluate every this many steps (default: same as log_every)
    threshold   : reward fraction that defines a 'mode' (default 0.5)
    verbose     : print coverage line to stdout
    """

    def __init__(self, eval_every: int | None = None,
                 threshold: float = 0.5, verbose: bool = True):
        self.eval_every = eval_every
        self.threshold = threshold
        self.verbose = verbose
        self._log_every = 1

    def on_train_start(self, agent: "GeoFlowAgent", n_steps: int) -> None:
        if self.eval_every is None:
            self.eval_every = max(1, n_steps // 20)

    def on_step_end(self, agent: "GeoFlowAgent", step: int, metrics: dict) -> None:
        if step % self.eval_every != 0 and step != metrics.get("_n_steps", step + 1) - 1:
            return
        if not agent.env.is_enumerable():
            return
        m = agent.evaluate(threshold=self.threshold)
        metrics.update(m)
        if self.verbose:
            print(
                f"  [CoverageTracker] step={step:6d} | "
                f"tv={m['tv']:.4f} | coverage={m['coverage']:.1%} "
                f"({m['n_covered']}/{m['n_modes']} modes)",
                flush=True,
            )


class EarlyStopping(Callback):
    """Stop training when TV drops below a threshold.

    Parameters
    ----------
    tv_threshold : stop when TV(p_theta, p_R) < this value
    patience     : number of consecutive evals below threshold before stopping
    eval_every   : evaluate every this many steps
    """

    def __init__(self, tv_threshold: float = 0.05,
                 patience: int = 3, eval_every: int | None = None):
        self.tv_threshold = tv_threshold
        self.patience = patience
        self.eval_every = eval_every
        self._count = 0
        self._stop = False

    def on_train_start(self, agent: "GeoFlowAgent", n_steps: int) -> None:
        if self.eval_every is None:
            self.eval_every = max(1, n_steps // 20)

    def on_step_end(self, agent: "GeoFlowAgent", step: int, metrics: dict) -> None:
        if step % self.eval_every != 0:
            return
        if not agent.env.is_enumerable():
            return
        tv = metrics.get("tv") or agent.evaluate()["tv"]
        if tv < self.tv_threshold:
            self._count += 1
            if self._count >= self.patience:
                self._stop = True
                print(
                    f"  [EarlyStopping] TV={tv:.4f} < {self.tv_threshold} "
                    f"for {self.patience} evals — stopping at step {step}.",
                    flush=True,
                )
        else:
            self._count = 0

    def should_stop(self) -> bool:
        return self._stop


class CSVLogger(Callback):
    """Log all step metrics to a CSV file.

    Parameters
    ----------
    path     : output CSV file path
    append   : if True, append to an existing file; otherwise overwrite
    """

    def __init__(self, path: str | Path, append: bool = False):
        self.path = Path(path)
        self._append = append
        self._writer: csv.DictWriter | None = None
        self._file = None
        self._fieldnames: list[str] | None = None

    def on_train_start(self, agent: "GeoFlowAgent", n_steps: int) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        mode = "a" if self._append else "w"
        self._file = self.path.open(mode, newline="")

    def on_step_end(self, agent: "GeoFlowAgent", step: int, metrics: dict) -> None:
        row = {"step": step, "timestamp": round(time.time(), 3), **metrics}
        row.pop("_n_steps", None)
        if self._writer is None:
            self._fieldnames = list(row.keys())
            self._writer = csv.DictWriter(
                self._file, fieldnames=self._fieldnames, extrasaction="ignore"
            )
            self._writer.writeheader()
        self._writer.writerow(row)
        self._file.flush()

    def on_train_end(self, agent: "GeoFlowAgent", history: list[dict]) -> None:
        if self._file:
            self._file.close()


class JSONLogger(Callback):
    """Log all training history to a JSON file at the end of training.

    Parameters
    ----------
    path : output JSON file path
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def on_train_end(self, agent: "GeoFlowAgent", history: list[dict]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w") as f:
            json.dump(history, f, indent=2)


class MetricAlignmentTracker(Callback):
    """Track the alignment between the learned metric g and the reward landscape.

    Measures Spearman correlation between |g[i,j]| and |logR_i - logR_j|
    every eval_every steps. Requires an enumerable env with a metric.

    Parameters
    ----------
    eval_every : evaluate every this many steps
    verbose    : print correlation to stdout
    """

    def __init__(self, eval_every: int | None = None, verbose: bool = True):
        self.eval_every = eval_every
        self.verbose = verbose

    def on_train_start(self, agent: "GeoFlowAgent", n_steps: int) -> None:
        if self.eval_every is None:
            self.eval_every = max(1, n_steps // 10)

    def on_step_end(self, agent: "GeoFlowAgent", step: int, metrics: dict) -> None:
        if step % self.eval_every != 0:
            return
        if agent.metric is None or not agent.env.is_enumerable():
            return
        try:
            from scipy.stats import spearmanr
            import numpy as np

            g = agent.metric.g.detach().numpy()
            r_vec = agent.env.reward_vector()
            log_r = np.log(np.maximum(r_vec, 1e-30))
            target = np.abs(log_r[:, None] - log_r[None, :])

            mask = ~np.eye(len(log_r), dtype=bool)
            corr, _ = spearmanr(g[mask], target[mask])
            metrics["metric_alignment"] = float(corr)
            if self.verbose:
                print(
                    f"  [MetricAlignment] step={step:6d} | "
                    f"Spearman(g, |ΔlogR|) = {corr:.3f}",
                    flush=True,
                )
        except ImportError:
            pass
