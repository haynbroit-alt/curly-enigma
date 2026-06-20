"""Lightweight structured logging — no external dependencies."""
from __future__ import annotations
import json
import time
from pathlib import Path


class Logger:
    """Append-only JSONL logger with optional console output.

    Each call to log() writes one JSON object per line to the log file
    and optionally prints a summary to stdout.

    Parameters
    ----------
    path     : where to write the JSONL log (None = no file)
    verbose  : if True, print a one-line summary to stdout
    prefix   : string prefix for console output (e.g. experiment name)
    """

    def __init__(self, path: str | Path | None = None,
                 verbose: bool = True, prefix: str = ""):
        self.path = Path(path) if path else None
        self.verbose = verbose
        self.prefix = prefix
        self._start = time.time()
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, step: int, metrics: dict) -> None:
        record = {"step": step, "t": round(time.time() - self._start, 2),
                  **metrics}
        if self.path:
            with self.path.open("a") as f:
                f.write(json.dumps(record) + "\n")
        if self.verbose:
            parts = " | ".join(
                f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                for k, v in metrics.items()
            )
            tag = f"[{self.prefix}] " if self.prefix else ""
            print(f"{tag}step={step:6d} | {parts}", flush=True)

    def load(self) -> list[dict]:
        """Load all logged records from the JSONL file."""
        if not self.path or not self.path.exists():
            return []
        with self.path.open() as f:
            return [json.loads(line) for line in f if line.strip()]
