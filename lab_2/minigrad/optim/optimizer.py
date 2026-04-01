from __future__ import annotations

import numpy as np

from minigrad.tensor import Tensor

class Optimizer:
    """Base class for all optimizers.

    Args:
        parameters: List of trainable :class:`Tensor` parameters.
        lr: Learning rate.
    """

    def __init__(self, parameters: list[Tensor], lr: float = 0.01) -> None:
        self.parameters = parameters
        self.lr = lr

    def step(self) -> None:
        """Perform a single optimization step. Must be overridden."""
        raise NotImplementedError

    def zero_grad(self) -> None:
        """Reset all parameter gradients to zero."""
        for p in self.parameters:
            p.grad = np.zeros_like(p.data)