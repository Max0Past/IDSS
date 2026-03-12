"""Loss functions operating on Tensors."""

from __future__ import annotations

from minigrad.tensor import Tensor


def binary_cross_entropy(y_pred: Tensor, y_true: Tensor, eps: float = 1e-8) -> Tensor:
    """Binary cross-entropy loss with epsilon protection against ``log(0)``.

    .. math::
        \\text{BCE} = -\\frac{1}{N}\\sum \\bigl[y \\log(\\hat{y}+\\varepsilon)
        + (1-y)\\log(1-\\hat{y}+\\varepsilon)\\bigr]

    Args:
        y_pred: Predicted probabilities in ``[0, 1]``.
        y_true: Binary targets (0 or 1).
        eps: Small constant for numerical stability.
    """
    y_pred_clamped = Tensor(y_pred.data.clip(eps, 1 - eps),
                            requires_grad=y_pred.requires_grad,
                            _children=y_pred._prev,
                            _op=y_pred._op)
    # Re-attach backward from y_pred so gradient flows through
    y_pred_clamped._prev = {y_pred}
    y_pred_clamped._op = "clamp"

    def _clamp_backward() -> None:
        mask = ((y_pred.data >= eps) & (y_pred.data <= 1 - eps)).astype(y_pred.data.dtype)
        y_pred.grad += y_pred_clamped.grad * mask

    y_pred_clamped._backward = _clamp_backward

    loss = -(y_true * y_pred_clamped.log() + (1 - y_true) * (1 - y_pred_clamped).log())
    return loss.mean()


def mse_loss(y_pred: Tensor, y_true: Tensor) -> Tensor:
    """Mean squared error loss.

    .. math::
        \\text{MSE} = \\frac{1}{N}\\sum (\\hat{y} - y)^2

    Args:
        y_pred: Predicted values.
        y_true: Target values.
    """
    diff = y_pred - y_true
    return (diff * diff).mean()


def categorical_cross_entropy(y_pred: Tensor, y_true: Tensor, eps: float = 1e-8) -> Tensor:
    """Categorical cross-entropy loss.

    .. math:
            CCE = -1/N * sum_i sum_c y_ic * log(y_hat_ic + eps)

    Args:
        y_pred: Predicted probabilities (e.g. from softmax).
        y_true: One-hot encoded target values.
        eps: Small constant for numerical stability.
    """
    y_pred_clamped = Tensor(y_pred.data.clip(eps, 1 - eps),
                            requires_grad=y_pred.requires_grad,
                            _children=y_pred._prev,
                            _op=y_pred._op)
    # Re-attach backward from y_pred so gradient flows through
    y_pred_clamped._prev = {y_pred}
    y_pred_clamped._op = "clamp"

    def _clamp_backward() -> None:
        mask = ((y_pred.data >= eps) & (y_pred.data <= 1 - eps)).astype(y_pred.data.dtype)
        y_pred.grad += y_pred_clamped.grad * mask

    y_pred_clamped._backward = _clamp_backward

    loss = -(y_true * y_pred_clamped.log()).sum(axis=-1)
    return loss.mean()
