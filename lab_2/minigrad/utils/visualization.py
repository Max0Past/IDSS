"""Visualization utilities for training diagnostics."""

from __future__ import annotations

from typing import Callable

import numpy as np
import matplotlib.pyplot as plt

from minigrad.tensor import Tensor
from minigrad.nn.module import Module


def plot_loss(losses: list[float], title: str = "Training Loss") -> None:
    """Plot loss curve over epochs.

    Args:
        losses: List of loss values, one per epoch.
        title: Plot title.
    """
    plt.figure(figsize=(8, 5))
    plt.plot(losses, linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_decision_boundary(
    model: Module,
    X: np.ndarray,
    y: np.ndarray,
    resolution: int = 200,
    title: str = "Decision Boundary",
) -> None:
    """Plot the decision boundary of a 2-D classifier.

    Args:
        model: Trained model that accepts ``Tensor`` input of shape ``(n, 2)``.
        X: Input data of shape ``(n_samples, 2)``.
        y: Binary labels of shape ``(n_samples,)`` or ``(n_samples, 1)``.
        resolution: Grid resolution.
        title: Plot title.
    """
    y_flat = y.ravel()
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, resolution),
        np.linspace(y_min, y_max, resolution),
    )
    grid = np.c_[xx.ravel(), yy.ravel()]

    preds = model(Tensor(grid)).data.reshape(xx.shape)

    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, preds, levels=50, cmap="RdYlBu", alpha=0.8)
    plt.colorbar(label="Prediction")
    plt.scatter(X[:, 0], X[:, 1], c=y_flat, cmap="RdYlBu", edgecolors="k", s=40)
    plt.xlabel("$x_1$")
    plt.ylabel("$x_2$")
    plt.title(title)
    plt.tight_layout()
    plt.show()
