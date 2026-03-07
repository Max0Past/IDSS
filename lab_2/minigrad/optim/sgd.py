"""Optimizers for gradient-based parameter updates."""

from __future__ import annotations

import numpy as np

from minigrad.tensor import Tensor

from minigrad.optim.optimizer import Optimizer


class SGD(Optimizer):
    """Stochastic Gradient Descent: w = w - lr * dw

    Args:
        parameters: Trainable parameters.
        lr: Learning rate (default: 0.01).
    """

    def __init__(self, parameters: list[Tensor], lr: float = 0.01) -> None:
        super().__init__(parameters, lr)

    def step(self) -> None:
        """Update each parameter by subtracting ``lr * grad``."""
        for p in self.parameters:
            p.data -= self.lr * p.grad
