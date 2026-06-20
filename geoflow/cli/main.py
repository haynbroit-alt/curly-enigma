"""GeoFlow-GFN command-line interface.

Usage
-----
    geoflow train --config configs/gridworld_default.yaml
    geoflow train --config configs/gridworld_default.yaml --seed 42
    geoflow benchmark --config configs/gridworld_default.yaml
    geoflow benchmark --config configs/gridworld_default.yaml --seeds 10
    geoflow chat
    geoflow --version
"""
from __future__ import annotations
import sys
from pathlib import Path
from typing import Optional

try:
    import typer
except ImportError:
    print("Error: 'typer' is required for the CLI. Install it with:\n  pip install typer", file=sys.stderr)
    sys.exit(1)

from .._version import __version__
from ..experiment.runner import load_config, merge_config, ExperimentRunner, MultiSeedRunner

app = typer.Typer(
    name="geoflow",
    help="GeoFlow-GFN: Adaptive exploration kernel for GFlowNets.",
    add_completion=False,
    rich_markup_mode=None,
)


def _resolve_config(config: Path) -> dict:
    if not config.exists():
        typer.echo(f"Error: config file not found: {config}", err=True)
        raise typer.Exit(1)
    return load_config(config)


# ── geoflow train ─────────────────────────────────────────────────────────────

@app.command()
def train(
    config: Path = typer.Option(..., "--config", "-c", help="Path to YAML config file."),
    seed: Optional[int] = typer.Option(None, "--seed", "-s", help="Training seed (overrides config)."),
    n_steps: Optional[int] = typer.Option(None, "--steps", "-n", help="Number of gradient steps."),
    verbose: bool = typer.Option(True, "--verbose/--quiet", help="Print step metrics."),
    out: Optional[Path] = typer.Option(None, "--out", "-o", help="Output JSON path."),
) -> None:
    """Train GeoFlow-GFN for a single seed and report final metrics."""
    cfg = _resolve_config(config)
    if n_steps is not None:
        cfg = merge_config(cfg, {"training": {"n_steps": n_steps}})

    env_label = cfg.get("env", {}).get("type", "env")
    typer.echo(f"\nGeoFlow-GFN train — {env_label}")
    typer.echo(f"  config : {config}")
    typer.echo(f"  seed   : {seed if seed is not None else cfg.get('training', {}).get('seeds', [0])[0]}")
    typer.echo(f"  steps  : {cfg.get('training', {}).get('n_steps', 500)}")
    typer.echo()

    result = ExperimentRunner().run(cfg, seed=seed, verbose=verbose)

    typer.echo("\nFinal evaluation:")
    typer.echo(f"  TV(p_θ, p_R)  = {result.tv:.4f}")
    typer.echo(f"  Coverage      = {result.coverage:.1%}  ({result.n_covered}/{result.n_modes} modes)")
    typer.echo(f"  P(modes)      = {result.p_on_modes:.4f}")

    if out is not None:
        from ..experiment.results import BenchmarkResult
        br = BenchmarkResult(runs=[result])
        p = br.save_json(out)
        typer.echo(f"\nSaved → {p}")


# ── geoflow benchmark ─────────────────────────────────────────────────────────

@app.command()
def benchmark(
    config: Path = typer.Option(..., "--config", "-c", help="Path to YAML config file."),
    seeds: Optional[int] = typer.Option(None, "--seeds", "-s", help="Number of seeds (overrides config list)."),
    n_steps: Optional[int] = typer.Option(None, "--steps", "-n", help="Number of gradient steps."),
    out_dir: Optional[Path] = typer.Option(None, "--out-dir", "-o", help="Output directory."),
) -> None:
    """Run multi-seed benchmark and report aggregated metrics."""
    cfg = _resolve_config(config)
    overrides: dict = {}
    if seeds is not None:
        overrides.setdefault("training", {})["seeds"] = list(range(seeds))
    if n_steps is not None:
        overrides.setdefault("training", {})["n_steps"] = n_steps
    if out_dir is not None:
        overrides["output"] = {"dir": str(out_dir)}
    if overrides:
        cfg = merge_config(cfg, overrides)

    seed_list = cfg.get("training", {}).get("seeds", [0, 1, 2, 3, 4])
    env_label = cfg.get("env", {}).get("type", "env")

    typer.echo(f"\nGeoFlow-GFN benchmark — {env_label}")
    typer.echo(f"  config : {config}")
    typer.echo(f"  seeds  : {seed_list}")
    typer.echo(f"  steps  : {cfg.get('training', {}).get('n_steps', 500)}")
    typer.echo()

    result = MultiSeedRunner().run(cfg, verbose=True)

    s = result.summary()
    typer.echo(f"\nEnvironment : {env_label}")
    typer.echo(f"Seeds       : {s['n_seeds']}")
    typer.echo("\nTV:")
    typer.echo(f"  mean = {s['tv']['mean']:.4f}")
    typer.echo(f"  std  = {s['tv']['std']:.4f}")
    typer.echo("\nCoverage:")
    typer.echo(f"  mean = {s['coverage']['mean']:.3f}")
    typer.echo(f"  std  = {s['coverage']['std']:.3f}")

    effective_out = Path(cfg.get("output", {}).get("dir", "outputs/"))
    json_path = result.save_json(effective_out / "benchmark.json")
    csv_path = result.save_csv(effective_out / "benchmark.csv")

    typer.echo("\nResults saved:")
    typer.echo(f"  {json_path}")
    typer.echo(f"  {csv_path}")


# ── geoflow chat ──────────────────────────────────────────────────────────────

_CHAT_HELP = """\
Examples:
  run a benchmark with 5 seeds
  train on a rough landscape
  compare GeoFlow vs standard GFN with 3 seeds
  run 1000 steps on D=6 K=4 with 5 seeds
  how well does it explore on a smooth landscape?
"""


@app.command()
def chat() -> None:
    """Interactive natural language interface to GeoFlow experiments."""
    from ..chat import GeoFlowChatAgent

    agent = GeoFlowChatAgent()
    typer.echo("\nGeoFlow Chat  (type 'help' for examples, 'quit' to exit)\n")

    while True:
        try:
            text = typer.prompt("GeoFlow")
        except (KeyboardInterrupt, EOFError):
            typer.echo("\nBye.")
            break

        text = text.strip()
        if not text:
            continue
        if text.lower() in ("quit", "exit", "q", "bye"):
            typer.echo("Bye.")
            break
        if text.lower() in ("help", "?", "aide"):
            typer.echo(_CHAT_HELP)
            continue

        typer.echo()
        typer.echo(agent.chat(text))
        typer.echo()


# ── geoflow version ───────────────────────────────────────────────────────────

@app.callback(invoke_without_command=True)
def _main(
    ctx: typer.Context,
    version: bool = typer.Option(False, "--version", "-V", is_eager=True, help="Show version and exit."),
) -> None:
    if version:
        typer.echo(f"geoflow-gfn {__version__}")
        raise typer.Exit()
    if ctx.invoked_subcommand is None:
        typer.echo(ctx.get_help())
