"""Base callback protocol for GeoFlow-GFN training."""
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..agent import GeoFlowAgent


class Callback:
    """Base class for training callbacks.

    Override any subset of the four hooks; the default implementations are no-ops.

    Hook call order during agent.fit():
        on_train_start → [on_step_end × n_steps] → on_train_end
    """

    def on_train_start(self, agent: "GeoFlowAgent", n_steps: int) -> None:
        pass

    def on_step_end(self, agent: "GeoFlowAgent", step: int, metrics: dict) -> None:
        pass

    def on_train_end(self, agent: "GeoFlowAgent", history: list[dict]) -> None:
        pass

    def should_stop(self) -> bool:
        """Return True to trigger early stopping after the current step."""
        return False
