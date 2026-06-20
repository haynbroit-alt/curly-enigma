"""Execute a parsed intent config against the GeoFlow experiment system."""
from __future__ import annotations
from ..experiment import ExperimentRunner, MultiSeedRunner


def _runner_cfg(cfg: dict) -> dict:
    return {
        "env": {
            "type": cfg["env"],
            "D": cfg["D"],
            "K": cfg["K"],
            "length_scale": cfg["length_scale"],
        },
        "agent": {
            "use_metric": cfg["use_metric"],
        },
        "training": {
            "n_steps": cfg["n_steps"],
            "seeds": list(range(cfg["seeds"])),
            "verbose": False,
        },
    }


def execute(cfg: dict) -> dict:
    rcfg = _runner_cfg(cfg)

    if cfg["task"] == "train":
        result = ExperimentRunner().run(rcfg, seed=0, verbose=False)
        return {"task": "train", "result": result}

    bench = MultiSeedRunner().run(rcfg, verbose=False)

    if cfg["compare"]:
        std_rcfg = _runner_cfg({**cfg, "use_metric": False})
        bench_std = MultiSeedRunner().run(std_rcfg, verbose=False)
        return {"task": "benchmark", "bench": bench, "bench_std": bench_std}

    return {"task": "benchmark", "bench": bench}
