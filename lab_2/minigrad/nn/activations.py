"""Activation layers — Module wrappers around functional activations."""

from __future__ import annotations

import numpy as np

from minigrad.tensor import Tensor
from minigrad.nn.module import Module
from minigrad.nn import functional as F


class ReLU(Module):
    """Applies element-wise ReLU: ``max(0, x)``."""

    def forward(self, x: Tensor) -> Tensor:
        return F.relu(x)

    def __repr__(self) -> str:
        return "ReLU()"

class LeakyReLU(Module):
    """Leaky ReLU layer with a fixed negative slope.

    ``alpha`` is the slope for inputs < 0.
    """

    def __init__(self, alpha: float = 0.01):
        self.alpha = alpha

    def forward(self, x: Tensor) -> Tensor:
        return F.leaky_relu(x, self.alpha)

    def __repr__(self) -> str:
        return f"LeakyReLU(alpha={self.alpha})"


class PReLU(Module):
    """Parametric ReLU with a learnable slope parameter.

    The parameter ``a`` is initialized to ``init`` and is a scalar Tensor.
    """

    def __init__(self, init: float = 0.25):
        self.a = Tensor(np.array(init, dtype=np.float32), requires_grad=True)

    def forward(self, x: Tensor) -> Tensor:
        return F.prelu(x, self.a)

    def __repr__(self) -> str:
        return f"PReLU(init={self.a.data})"


class Sigmoid(Module):
    """Applies element-wise sigmoid: ``1 / (1 + exp(-x))``."""

    def forward(self, x: Tensor) -> Tensor:
        return F.sigmoid(x)

    def __repr__(self) -> str:
        return "Sigmoid()"


class ELU(Module):
    """Exponential linear unit: ``x`` if ``x>0`` else ``alpha*(exp(x)-1)``."""

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha

    def forward(self, x: Tensor) -> Tensor:
        return F.elu(x, self.alpha)

    def __repr__(self) -> str:
        return f"ELU(alpha={self.alpha})"


class Tanh(Module):
    """Applies element-wise tanh."""

    def forward(self, x: Tensor) -> Tensor:
        return F.tanh(x)

    def __repr__(self) -> str:
        return "Tanh()"


class Softmax(Module):
    """Applies the Softmax function to an n-dimensional input Tensor.
    
    Rescales elements to the range [0, 1] such that the elements along
    the specified axis sum to 1.
    """

    def __init__(self, axis: int = -1):
        self.axis = axis

    def forward(self, x: Tensor) -> Tensor:
        return F.softmax(x, axis=self.axis)

    def __repr__(self) -> str:
        return f"Softmax(axis={self.axis})"
