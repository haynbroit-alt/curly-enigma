"""Results aggregation and export for GeoFlow experiments."""
from __future__ import annotations
import csv
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class RunResult:
    """Metrics from a single training run."""
    seed: int
    tv: float
    coverage: float
    n_modes: int
    n_covered: int
    p_on_modes: float
    history: list[dict] = field(default_factory=list)
    config: dict = field(default_factory=dict)


@dataclass
class BenchmarkResult:
    """Aggregated results across multiple seeds."""
    runs: list[RunResult] = field(default_factory=list)

    # ── Computed properties ───────────────────────────────────────────────────

    @property
    def tv_values(self) -> list[float]:
        return [r.tv for r in self.runs]

    @property
    def coverage_values(self) -> list[float]:
        return [r.coverage for r in self.runs]

    def summary(self) -> dict:
        tvs = np.array(self.tv_values)
        covs = np.array(self.coverage_values)
        return {
            "n_seeds": len(self.runs),
            "tv": {"mean": float(tvs.mean()), "std": float(tvs.std()),
                   "min": float(tvs.min()), "max": float(tvs.max())},
            "coverage": {"mean": float(covs.mean()), "std": float(covs.std()),
                         "min": float(covs.min()), "max": float(covs.max())},
        }

    # ── Export ────────────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "summary": self.summary(),
            "runs": [
                {"seed": r.seed, "tv": r.tv, "coverage": r.coverage,
                 "n_modes": r.n_modes, "n_covered": r.n_covered,
                 "p_on_modes": r.p_on_modes}
                for r in self.runs
            ],
        }

    def save_json(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w") as f:
            json.dump(self.to_dict(), f, indent=2)
        return p

    def save_csv(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fields = ["seed", "tv", "coverage", "n_modes", "n_covered", "p_on_modes"]
        with p.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for r in self.runs:
                w.writerow({k: getattr(r, k) for k in fields})
        return p

    def print_summary(self, label: str = "") -> None:
        s = self.summary()
        header = "Benchmark results" + (f" — {label}" if label else "")
        print(f"\n{header}")
        print(f"  Seeds   : {s['n_seeds']}")
        print(f"  TV      : {s['tv']['mean']:.4f} ± {s['tv']['std']:.4f}"
              f"  [min={s['tv']['min']:.4f}, max={s['tv']['max']:.4f}]")
        print(f"  Coverage: {s['coverage']['mean']:.3f} ± {s['coverage']['std']:.3f}"
              f"  [min={s['coverage']['min']:.3f}, max={s['coverage']['max']:.3f}]")
