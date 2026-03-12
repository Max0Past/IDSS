"""Layers — Linear and Perceptron."""

from __future__ import annotations

import numpy as np

from minigrad.tensor import Tensor
from minigrad.nn.module import Module


class Linear(Module):
    """Fully-connected layer: ``y = x @ W.T + b``.

    Uses Xavier (Glorot) uniform initialization for weights.

    Args:
        in_features: Size of each input sample.
        out_features: Size of each output sample.
    """

    def __init__(self, in_features: int, out_features: int) -> None:
        # Initializing with a normal distribution with zero mean and small variance
        self.weight = Tensor(
            np.random.normal(0, 0.01, (out_features, in_features)),
            requires_grad=True,
        )
        self.bias = Tensor(
            np.zeros(out_features),
            requires_grad=True,
        )

    def forward(self, x: Tensor) -> Tensor:
        """Compute ``x @ W.T + b``."""
        return x @ self.weight.T + self.bias

    def __repr__(self) -> str:
        out_f, in_f = self.weight.shape
        return f"Linear(in_features={in_f}, out_features={out_f})"


class Perceptron(Module):
    """Single-neuron perceptron with Rosenblatt learning rule.

    Uses a hard step activation (non-differentiable), so it is **not**
    trained via backpropagation — use :meth:`train_step` with the
    Rosenblatt update rule instead.

    Args:
        in_features: Number of input features.
    """

    def __init__(self, in_features: int) -> None:
        self.weight = Tensor(
            np.random.randn(in_features) * 0.01,
            requires_grad=False,
        )
        self.bias = Tensor(np.zeros(1), requires_grad=False)

    def forward(self, x: Tensor) -> Tensor:
        """Compute ``step(x @ w + b)``."""
        linear = x @ self.weight + self.bias
        return Tensor((linear.data >= 0).astype(np.float64))

    def train_step(self, x: Tensor, y_true: Tensor, lr: float = 0.1) -> None:
        """One Rosenblatt update: ``w += lr * (y_true - y_pred) * x``.

        Args:
            x: Input sample (1-D).
            y_true: Target label (0 or 1).
            lr: Learning rate.
        """
        y_pred = self.forward(x)
        error = y_true.data - y_pred.data
        self.weight.data += lr * error * x.data
        self.bias.data += lr * error

    def __repr__(self) -> str:
        return f"Perceptron(in_features={self.weight.shape[0]})"
