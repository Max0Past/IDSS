from __future__ import annotations

import numpy as np

from minigrad.tensor import Tensor
from minigrad.optim.optimizer import Optimizer

class GD(Optimizer):
    """
    Gradient Descent optimizer: w = w - lr * dw
    Args:
        parameters: List of trainable :class:`Tensor` parameters.
        lr: Learning rate.
    
    """
    def __init__(self, parameters: list[Tensor], lr: float = 0.01) -> None:
        super().__init__(parameters, lr)
    
    def step(self) -> None:
        for p in self.parameters:
            p.data -= self.lr * p.grad