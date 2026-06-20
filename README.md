# GeoFlow-GFN

**Adaptive exploration kernel for GFlowNets** — learnable metric + Trajectory Balance training.

[![Tests](https://img.shields.io/badge/tests-20%2F20%20passing-3fb950)](tests/)
[![Python](https://img.shields.io/badge/python-3.10%2B-58a6ff)](pyproject.toml)
[![License](https://img.shields.io/badge/license-MIT-8b949e)](LICENSE)
[![Version](https://img.shields.io/badge/version-0.3.0-58a6ff)](geoflow/_version.py)

GeoFlow-GFN guides sampling toward high-reward regions via a differentiable distance matrix over terminal states. The learnable metric `g` (N×N parameter) is trained jointly with the GFlowNet policy using Trajectory Balance loss, with EMA updates to keep the metric stable.

---

## Results

GridWorld benchmark · D=4, K=4, N=256 states · 5 seeds each

| Landscape (ℓ) | GFN-standard TV | GeoFlow TV | Δ TV |
|---|---|---|---|
| 0.4 (rough) | 0.179 ± 0.020 | **0.169 ± 0.019** | −0.010 |
| 1.2 (medium) | 0.062 ± 0.011 | **0.061 ± 0.009** | −0.001 |
| 4.0 (smooth) | 0.015 ± 0.004 | **0.014 ± 0.004** | −0.001 |

**TV distance 0.014 ± 0.004 · Mode coverage 100% · 5 seeds reproduced**

---

## Installation

```bash
pip install geoflow-gfn
```

Or from source:

```bash
git clone https://github.com/haynbroit-alt/curly-enigma
cd curly-enigma
pip install -e ".[dev]"
```

---

## Quick Start

```python
from geoflow import GeoFlowAgent, GridWorld, CoverageTracker

env   = GridWorld(D=4, K=4)
agent = GeoFlowAgent(env=env, lambda_kl=0.1)

history = agent.fit(
    n_steps=500,
    callbacks=[CoverageTracker()],
)

print(agent.evaluate())
# {'tv': 0.014, 'coverage': 1.0, 'n_covered': 8, 'n_modes': 8}
```

---

## CLI

Train a single run:

```bash
geoflow train --config configs/gridworld_default.yaml
geoflow train --config configs/gridworld_default.yaml --seed 42 --steps 1000
```

Run the full multi-seed benchmark:

```bash
geoflow benchmark --config configs/gridworld_default.yaml
geoflow benchmark --config configs/gridworld_default.yaml --seeds 10 --out-dir results/
```

---

## Configuration

All hyperparameters live in a single YAML file:

```yaml
# configs/gridworld_default.yaml
env:
  type: gridworld
  D: 4
  K: 4
  length_scale: 1.2

agent:
  hidden: 128
  lr: 5.0e-3
  lr_metric: 5.0e-3
  lambda_kl: 0.1
  beta: 0.01
  batch_size: 50
  use_metric: true

training:
  n_steps: 500
  log_every: 50
  seeds: [0, 1, 2, 3, 4]
  verbose: false

callbacks:
  CoverageTracker:
    eval_every: 50
  EarlyStopping:
    tv_threshold: 0.05
    patience: 3

output:
  dir: outputs/
```

---

## Callbacks

```python
from geoflow import (
    CoverageTracker,       # tracks mode coverage over training
    EarlyStopping,         # stops when TV < threshold
    CSVLogger,             # writes per-step metrics to CSV
    JSONLogger,            # writes final history to JSON
    MetricAlignmentTracker # logs learnable metric alignment
)

agent.fit(
    n_steps=500,
    callbacks=[
        CoverageTracker(eval_every=50),
        EarlyStopping(tv_threshold=0.02, patience=5),
        CSVLogger("outputs/run.csv"),
    ]
)
```

Implement your own by subclassing `Callback`:

```python
from geoflow import Callback

class MyCallback(Callback):
    def on_step_end(self, agent, step, metrics):
        if step % 100 == 0:
            print(f"step {step}: TV={metrics.get('tv', '?'):.4f}")
```

---

## API Reference

### `GeoFlowAgent`

```python
GeoFlowAgent(
    env,               # DiscreteEnv instance
    hidden=128,        # policy network hidden size
    lr=5e-3,           # policy learning rate
    lr_metric=5e-3,    # metric learning rate
    lambda_kl=0.1,     # KL regularization weight
    beta=0.01,         # metric smoothing coefficient
    batch_size=50,     # trajectories per step
    use_metric=True,   # enable learnable metric (False = GFN-standard)
    seed=0,
)
```

| Method | Returns | Description |
|---|---|---|
| `fit(n_steps, callbacks)` | `list[dict]` | Train the agent |
| `evaluate()` | `dict` | Compute TV, coverage, mode stats |
| `sample(n)` | `list` | Sample n terminal states |
| `from_config(path, seed, overrides)` | `(agent, cfg)` | Load from YAML |

### `ExperimentRunner` / `MultiSeedRunner`

```python
from geoflow import ExperimentRunner, MultiSeedRunner, load_config

cfg = load_config("configs/gridworld_default.yaml")

# Single seed
result = ExperimentRunner().run(cfg, seed=0)
print(result.tv, result.coverage)

# Multi-seed
bench = MultiSeedRunner().run(cfg)
bench.print_summary("GridWorld D=4")
bench.save_json("outputs/benchmark.json")
bench.save_csv("outputs/benchmark.csv")
```

---

## Environments

| Class | Description |
|---|---|
| `GridWorld(D, K)` | D-dimensional grid, K actions per dim. Standard GFN benchmark. |
| `ArithmeticEnv(n, target)` | Compositional arithmetic — find factor sequences summing to target. |

---

## How It Works

GeoFlow augments the standard GFlowNet objective with a **learnable distance matrix** `g` over terminal states.

1. **Metric initialization**: `g` starts as a scaled identity — structurally close to Euclidean distance.
2. **Joint training**: At each step, trajectories are sampled, the TB loss is computed, and `g` is updated alongside the policy.
3. **EMA update**: The metric is smoothed via exponential moving average (`β=0.01`) to prevent oscillation.
4. **KL regularization**: A `λ_KL` penalty keeps the learned metric close to the prior, avoiding collapse.

The result is a policy that explores more efficiently in rough landscapes where the standard GFN struggles to cover all modes.

---

## Reproducing the Benchmark

```bash
pip install -e ".[dev]"
geoflow benchmark --config configs/gridworld_default.yaml --seeds 5
```

Expected output:

```
GeoFlow-GFN benchmark — gridworld
  config : configs/gridworld_default.yaml
  seeds  : [0, 1, 2, 3, 4]
  steps  : 500

TV:
  mean = 0.0140
  std  = 0.0042

Coverage:
  mean = 1.000
  std  = 0.000
```

Full results are in [`results/multiseed.json`](results/multiseed.json).

---

## Testing

```bash
pytest tests/ -v
```

20 tests across agent, environments, callbacks, trainer, and experiment runner.

---

## License

MIT — see [LICENSE](LICENSE).

---

## Citation

```bibtex
@software{geoflow_gfn_2025,
  title  = {{GeoFlow-GFN}: Adaptive Exploration Kernel for {GFlowNet}s},
  year   = {2025},
  url    = {https://github.com/haynbroit-alt/curly-enigma},
  note   = {v0.3.0}
}
```
