"""Rule-based natural language → intent parser."""
from __future__ import annotations
import re

_DEFAULTS: dict = {
    "task": "benchmark",
    "env": "gridworld",
    "D": 4,
    "K": 4,
    "length_scale": 1.2,
    "seeds": 5,
    "n_steps": 500,
    "use_metric": True,
    "compare": False,
}

_TRAIN_WORDS = {"train", "entraîne", "entraîner", "fit", "learn", "apprend"}
_ROUGH_WORDS = {"rough", "difficile", "complex", "hard"}
_SMOOTH_WORDS = {"smooth", "simple", "easy", "facile"}


def parse(text: str) -> dict:
    """Convert a natural language string into a structured experiment config."""
    cfg = dict(_DEFAULTS)
    t = text.lower()

    # Task: train vs benchmark
    if any(w in t for w in _TRAIN_WORDS) and "benchmark" not in t:
        cfg["task"] = "train"
    else:
        cfg["task"] = "benchmark"

    # Comparison mode
    if any(w in t for w in ("compare", "vs", "versus", "baseline", "standard")):
        cfg["compare"] = True

    # Explicit GFN-standard (disable learnable metric)
    if "standard" in t and "gfn" in t:
        cfg["use_metric"] = False

    # Seeds: "5 seeds", "3 seeds"
    m = re.search(r"(\d+)\s*seed", t)
    if m:
        cfg["seeds"] = int(m.group(1))

    # Steps: "1000 steps", "500 steps"
    m = re.search(r"(\d+)\s*step", t)
    if m:
        cfg["n_steps"] = int(m.group(1))

    # Grid dimensions (case-sensitive: D=6, K=4)
    m = re.search(r"\bD\s*=\s*(\d+)", text)
    if m:
        cfg["D"] = int(m.group(1))
    m = re.search(r"\bK\s*=\s*(\d+)", text)
    if m:
        cfg["K"] = int(m.group(1))

    # Landscape difficulty
    if any(w in t for w in _ROUGH_WORDS):
        cfg["length_scale"] = 0.4
    elif any(w in t for w in _SMOOTH_WORDS):
        cfg["length_scale"] = 4.0

    return cfg
