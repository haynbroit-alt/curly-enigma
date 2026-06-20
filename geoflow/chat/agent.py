"""GeoFlowChatAgent: natural language interface to GeoFlow experiments."""
from __future__ import annotations
from .parser import parse
from .executor import execute


class GeoFlowChatAgent:
    """Translate natural language into GeoFlow experiment runs and return formatted results."""

    def __init__(self) -> None:
        self._history: list[dict] = []

    def chat(self, text: str) -> str:
        cfg = parse(text)
        result = execute(cfg)
        response = _format(cfg, result)
        self._history.append({"input": text, "cfg": cfg, "response": response})
        return response

    @property
    def history(self) -> list[dict]:
        return list(self._history)


# ── Formatting helpers ────────────────────────────────────────────────────────

def _interpret(tv: float) -> str:
    if tv < 0.02:
        return "Convergence: excellent — distribution closely matches reward."
    if tv < 0.05:
        return "Convergence: good — minor deviation from target."
    if tv < 0.15:
        return "Convergence: moderate — some modes underweighted."
    return "Convergence: poor — exploration struggling on this landscape."


def _format(cfg: dict, result: dict) -> str:
    task = result.get("task", "")

    if task == "train":
        r = result["result"]
        return "\n".join([
            f"Training complete  ({cfg['n_steps']} steps, seed 0)",
            f"  TV distance : {r.tv:.4f}",
            f"  Coverage    : {r.coverage:.1%}  ({r.n_covered}/{r.n_modes} modes)",
            f"  P(modes)    : {r.p_on_modes:.4f}",
            "",
            _interpret(r.tv),
        ])

    if task == "benchmark":
        bench = result["bench"]
        s = bench.summary()
        lines = [
            f"Benchmark  GridWorld D={cfg['D']} K={cfg['K']}  "
            f"({'rough' if cfg['length_scale'] < 0.8 else 'smooth' if cfg['length_scale'] > 2.0 else 'medium'} landscape)"
            f"  {cfg['seeds']} seeds",
            f"  TV       : {s['tv']['mean']:.4f} +/- {s['tv']['std']:.4f}",
            f"  Coverage : {s['coverage']['mean']:.1%} +/- {s['coverage']['std']:.1%}",
        ]
        if "bench_std" in result:
            s_std = result["bench_std"].summary()
            delta = s["tv"]["mean"] - s_std["tv"]["mean"]
            lines += [
                "",
                "  vs GFN-standard:",
                f"    GFN-std  TV : {s_std['tv']['mean']:.4f} +/- {s_std['tv']['std']:.4f}",
                f"    GeoFlow  TV : {s['tv']['mean']:.4f} +/- {s['tv']['std']:.4f}",
                f"    delta    TV : {delta:+.4f}",
            ]
        lines += ["", _interpret(s["tv"]["mean"])]
        return "\n".join(lines)

    return "Unknown task. Try: 'run a benchmark', 'train', or 'compare GeoFlow vs standard'."
