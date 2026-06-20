"""Experiment runner: Config → Agent → Results.

The runner is the only layer that knows both the config schema and the
GeoFlowAgent API. The CLI delegates entirely to it.

Flow:
    CLI / script
        ↓
    ExperimentRunner.run(config)
        ↓
    _build_env(config)  +  _build_agent(config)  +  _build_callbacks(config)
        ↓
    agent.fit(callbacks=...)
        ↓
    RunResult / BenchmarkResult
"""
from __future__ import annotations
import copy
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from ..agent import GeoFlowAgent
from ..callbacks.base import Callback
from .results import BenchmarkResult, RunResult

# ── Registry: callback name → class ──────────────────────────────────────────

def _callback_registry() -> dict:
    from ..callbacks.builtin import (
        CoverageTracker, EarlyStopping, CSVLogger, JSONLogger,
        MetricAlignmentTracker,
    )
    return {
        "CoverageTracker": CoverageTracker,
        "EarlyStopping": EarlyStopping,
        "CSVLogger": CSVLogger,
        "JSONLogger": JSONLogger,
        "MetricAlignmentTracker": MetricAlignmentTracker,
    }


# ── Config loading ────────────────────────────────────────────────────────────

def load_config(path: str | Path) -> dict:
    """Load a YAML config file and return as a plain dict."""
    with open(path) as f:
        return yaml.safe_load(f)


def merge_config(base: dict, overrides: dict) -> dict:
    """Deep-merge overrides into base (overrides win)."""
    result = copy.deepcopy(base)
    for k, v in overrides.items():
        if isinstance(v, dict) and isinstance(result.get(k), dict):
            result[k] = merge_config(result[k], v)
        else:
            result[k] = v
    return result


# ── Builder functions ─────────────────────────────────────────────────────────

def _build_env(env_cfg: dict) -> Any:
    env_type = env_cfg.get("type", "gridworld").lower()
    if env_type == "gridworld":
        from ..envs.gridworld import GridWorld
        return GridWorld(
            D=env_cfg.get("D", 4),
            K=env_cfg.get("K", 4),
            length_scale=env_cfg.get("length_scale", 1.2),
            seed=env_cfg.get("seed", 0),
        )
    if env_type == "combinatorial":
        from ..envs.combinatorial import ArithmeticEnv
        return ArithmeticEnv(
            target=env_cfg.get("target", 42),
            scale=env_cfg.get("scale", 5.0),
            max_tokens=env_cfg.get("max_tokens", 6),
        )
    raise ValueError(f"Unknown env type: {env_type!r}. Choose 'gridworld' or 'combinatorial'.")


def _build_agent(env: Any, agent_cfg: dict, seed: int) -> GeoFlowAgent:
    return GeoFlowAgent(
        env=env,
        hidden=agent_cfg.get("hidden", 128),
        lr=agent_cfg.get("lr", 5e-3),
        lr_metric=agent_cfg.get("lr_metric", 5e-3),
        lambda_kl=agent_cfg.get("lambda_kl", 0.1),
        beta=agent_cfg.get("beta", 0.01),
        batch_size=agent_cfg.get("batch_size", 50),
        use_metric=agent_cfg.get("use_metric", True),
        seed=seed,
    )


def _build_callbacks(cb_cfg: dict, out_dir: Path) -> list[Callback]:
    registry = _callback_registry()
    cbs = []
    for name, kwargs in cb_cfg.items():
        cls = registry.get(name)
        if cls is None:
            raise ValueError(f"Unknown callback: {name!r}. Available: {list(registry)}")
        kw = kwargs or {}
        # Resolve relative paths inside output dir
        if "path" in kw and not Path(kw["path"]).is_absolute():
            kw = {**kw, "path": str(out_dir / kw["path"])}
        cbs.append(cls(**kw))
    return cbs


# ── Runner ────────────────────────────────────────────────────────────────────

class ExperimentRunner:
    """Run a single-seed experiment from a config dict.

    Usage
    -----
    cfg = load_config("configs/gridworld_default.yaml")
    result = ExperimentRunner().run(cfg, seed=0)
    """

    def run(self, config: dict, seed: int | None = None,
            verbose: bool | None = None) -> RunResult:
        env_cfg = config.get("env", {})
        agent_cfg = config.get("agent", {})
        train_cfg = config.get("training", {})
        cb_cfg = config.get("callbacks", {})
        out_dir = Path(config.get("output", {}).get("dir", "outputs/"))

        effective_seed = seed if seed is not None else train_cfg.get("seeds", [0])[0]
        effective_verbose = verbose if verbose is not None else train_cfg.get("verbose", False)

        torch.manual_seed(effective_seed)
        np.random.seed(effective_seed)

        env = _build_env(env_cfg)
        agent = _build_agent(env, agent_cfg, seed=effective_seed)
        callbacks = _build_callbacks(cb_cfg, out_dir)

        history = agent.fit(
            n_steps=train_cfg.get("n_steps", 500),
            log_every=train_cfg.get("log_every", 50),
            verbose=effective_verbose,
            callbacks=callbacks,
        )

        if env.is_enumerable():
            m = agent.evaluate(threshold=0.5)
        else:
            m = {"tv": float("nan"), "coverage": float("nan"),
                 "n_modes": 0, "n_covered": 0, "p_on_modes": float("nan")}

        return RunResult(
            seed=effective_seed,
            tv=m["tv"],
            coverage=m["coverage"],
            n_modes=m.get("n_modes", 0),
            n_covered=m.get("n_covered", 0),
            p_on_modes=m.get("p_on_modes", float("nan")),
            history=history,
            config=config,
        )


class MultiSeedRunner:
    """Run the same experiment over multiple seeds and aggregate results.

    Usage
    -----
    cfg = load_config("configs/gridworld_default.yaml")
    result = MultiSeedRunner().run(cfg)
    result.print_summary()
    result.save_json("outputs/benchmark.json")
    result.save_csv("outputs/benchmark.csv")
    """

    def run(self, config: dict, verbose: bool = True) -> BenchmarkResult:
        train_cfg = config.get("training", {})
        seeds = train_cfg.get("seeds", [0, 1, 2, 3, 4])
        runner = ExperimentRunner()
        benchmark = BenchmarkResult()

        for i, seed in enumerate(seeds):
            if verbose:
                print(f"  Seed {seed} ({i+1}/{len(seeds)}) …", end=" ", flush=True)
            result = runner.run(config, seed=seed, verbose=False)
            benchmark.runs.append(result)
            if verbose:
                print(f"TV={result.tv:.4f}  coverage={result.coverage:.1%}")

        return benchmark
