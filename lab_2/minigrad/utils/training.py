"""Training utilities."""

from __future__ import annotations

from typing import Callable

import numpy as np

from minigrad.tensor import Tensor
from minigrad.nn.module import Module
from minigrad.optim.sgd import Optimizer


def train_epoch(
    model: Module,
    X: np.ndarray,
    y: np.ndarray,
    loss_fn: Callable[[Tensor, Tensor], Tensor],
    optimizer: Optimizer,
    batch_size: int = 32,
) -> float:
    """Run one training epoch over the dataset.

    Splits ``(X, y)`` into mini-batches of size ``batch_size``, performs
    forward pass, computes loss, backpropagates, and updates parameters.

    Args:
        model: The neural network module.
        X: Input data array of shape ``(n_samples, ...)``.
        y: Target array of shape ``(n_samples, ...)``.
        loss_fn: A callable ``(y_pred, y_true) -> scalar Tensor``.
        optimizer: Optimizer instance bound to the model parameters.
        batch_size: Number of samples per mini-batch.

    Returns:
        Average loss over all batches.
    """
    n_samples = X.shape[0]
    indices = np.random.permutation(n_samples)
    total_loss = 0.0
    n_batches = 0

    for start in range(0, n_samples, batch_size):
        batch_idx = indices[start : start + batch_size]
        x_batch = Tensor(X[batch_idx], requires_grad=False)
        y_batch = Tensor(y[batch_idx], requires_grad=False)

        # Forward
        y_pred = model(x_batch)

        # Loss
        loss = loss_fn(y_pred, y_batch)

        # Backward
        optimizer.zero_grad()
        loss.backward()

        # Update
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / n_batches
