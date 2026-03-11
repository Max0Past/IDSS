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


def plot_history(history: dict[str, list[float]]) -> None:
    """Plot training and validation loss and accuracy over epochs.

    Args:
        history: Dictionary containing 'train_loss', 'val_loss', 'train_acc', 'val_acc'.
    """
    epochs = range(1, len(history['train_loss']) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot Loss
    ax1.plot(epochs, history['train_loss'], label='Train Loss', marker='o')
    ax1.plot(epochs, history['val_loss'], label='Val Loss', marker='o')
    ax1.set_title('Loss over Epochs')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot Accuracy
    ax2.plot(epochs, history['train_acc'], label='Train Acc', marker='o')
    ax2.plot(epochs, history['val_acc'], label='Val Acc', marker='o')
    ax2.set_title('Accuracy over Epochs')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def plot_misclassified(
    model: Module,
    X_test_norm: np.ndarray,
    X_test_raw: np.ndarray,
    y_test_raw: np.ndarray,
    class_names: list[str],
    num_images: int = 5
) -> None:
    """Display a few incorrectly classified images.

    Args:
        model: Trained model.
        X_test_norm: Normalized input data (n_samples, 784).
        X_test_raw: Raw input data for visualization (0-255 pixels).
        y_test_raw: Raw target labels (0-9).
        class_names: List of string names for classes.
        num_images: How many images to show.
    """
    # Forward pass without gradients
    test_preds = model(Tensor(X_test_norm, requires_grad=False)).data
    pred_labels = np.argmax(test_preds, axis=1)

    # Find indices of mistakes
    incorrect_indices = np.where(pred_labels != y_test_raw)[0]
    
    print(f"\nTotal incorrect predictions: {len(incorrect_indices)} out of {len(y_test_raw)}")
    
    if len(incorrect_indices) == 0:
        print("No incorrect predictions found! Perfect model.")
        return

    # Limit to num_images
    num_images = min(num_images, len(incorrect_indices))
    
    plt.figure(figsize=(3 * num_images, 3))
    for i, idx in enumerate(incorrect_indices[:num_images]):
        plt.subplot(1, num_images, i + 1)
        
        # Reshape 784 vector to 28x28 image
        image = X_test_raw[idx].reshape(28, 28)
        plt.imshow(image, cmap='gray')
        
        true_label = class_names[y_test_raw[idx]]
        pred_label = class_names[pred_labels[idx]]
        
        plt.title(f"True: {true_label}\nPred: {pred_label}", color='red', fontsize=10)
        plt.axis('off')

    plt.tight_layout()
    plt.show()