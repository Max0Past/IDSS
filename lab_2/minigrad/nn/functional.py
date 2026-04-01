"""Functional API — pure functions operating on Tensors."""

from __future__ import annotations

import numpy as np

from minigrad.tensor import Tensor


def relu(x: Tensor) -> Tensor:
    """Element-wise ReLU: ``max(0, x)``."""
    mask = (x.data > 0).astype(x.data.dtype)
    out = Tensor(
        x.data * mask,
        requires_grad=x.requires_grad,
        _children=(x,),
        _op="relu",
    )

    def _backward() -> None:
        x.grad += mask * out.grad

    out._backward = _backward
    return out

def leaky_relu(x: Tensor, alpha: float = 0.01) -> Tensor:
    """Element-wise Leaky ReLU: ``x`` if ``x>0`` else ``alpha*x``.

    ``alpha`` is a scalar slope for negative inputs.
    """
    mask = (x.data > 0).astype(x.data.dtype)
    out_data = x.data * mask + alpha * x.data * (1 - mask)
    out = Tensor(
        out_data,
        requires_grad=x.requires_grad,
        _children=(x,),
        _op="leaky_relu",
    )

    def _backward() -> None:
        x.grad += out.grad * (mask + alpha * (1 - mask))

    out._backward = _backward
    return out

def sigmoid(x: Tensor) -> Tensor:
    """Element-wise sigmoid: ``1 / (1 + exp(-x))``."""
    s = 1.0 / (1.0 + np.exp(-x.data))
    out = Tensor(
        s,
        requires_grad=x.requires_grad,
        _children=(x,),
        _op="sigmoid",
    )

    def _backward() -> None:
        x.grad += out.grad * s * (1.0 - s)

    out._backward = _backward
    return out


def tanh(x: Tensor) -> Tensor:
    """Element-wise hyperbolic tangent."""
    t = np.tanh(x.data)
    out = Tensor(
        t,
        requires_grad=x.requires_grad,
        _children=(x,),
        _op="tanh",
    )

    def _backward() -> None:
        x.grad += out.grad * (1.0 - t ** 2)

    out._backward = _backward
    return out


def prelu(x: Tensor, a: Tensor) -> Tensor:
    """Parametric ReLU: ``x`` if ``x>0`` else ``a*x`` where ``a`` is a learnable
    parameter tensor (broadcastable to ``x``).
    """
    mask = (x.data > 0).astype(x.data.dtype)
    out_data = x.data * mask + a.data * x.data * (1 - mask)
    out = Tensor(
        out_data,
        requires_grad=x.requires_grad or a.requires_grad,
        _children=(x, a),
        _op="prelu",
    )

    def _backward() -> None:
        x.grad += out.grad * (mask + a.data * (1 - mask))
        if a.requires_grad:
            # gradient wrt a is x for negative entries
            a.grad += np.sum(out.grad * x.data * (1 - mask))

    out._backward = _backward
    return out


def elu(x: Tensor, alpha: float = 1.0) -> Tensor:
    """Exponential linear unit: ``x`` if ``x>0`` else ``alpha*(exp(x)-1)``.

    ``alpha`` is a scalar parameter controlling the amplitude of the negative
    region.
    """
    data = np.where(x.data > 0, x.data, alpha * (np.exp(x.data) - 1))
    out = Tensor(
        data,
        requires_grad=x.requires_grad,
        _children=(x,),
        _op="elu",
    )

    def _backward() -> None:
        grad = np.where(x.data > 0, 1.0, alpha * np.exp(x.data))
        x.grad += out.grad * grad

    out._backward = _backward
    return out


def softmax(x: Tensor, axis: int = -1) -> Tensor:
    """Applies the Softmax function to an n-dimensional input Tensor.

    Rescales elements to the range [0, 1] such that the elements along
    the specified axis sum to 1.
    """
    x_max = np.max(x.data, axis=axis, keepdims=True)
    exp_x = np.exp(x.data - x_max)
    s = exp_x / np.sum(exp_x, axis=axis, keepdims=True)
    out = Tensor(
        s,
        requires_grad=x.requires_grad,
        _children=(x,),
        _op="softmax",
    )

    def _backward() -> None:
        sum_grad_s = np.sum(out.grad * s, axis=axis, keepdims=True)
        x.grad += s * (out.grad - sum_grad_s)

    out._backward = _backward
    return out
