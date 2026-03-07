"""Tensor — core computational object with automatic differentiation support.

Provides a NumPy-backed tensor that builds a computational graph for
reverse-mode autodiff, inspired by Karpathy's micrograd but extended
to full matrix operations.
"""

from __future__ import annotations

from typing import Callable, Union

import numpy as np

Numeric = Union[int, float, "Tensor"]


class Tensor:
    """Multi-dimensional array with automatic gradient computation.

    Wraps a NumPy ndarray and tracks operations to build a DAG used
    by ``backward()`` for gradient computation.

    Args:
        data: Array-like input data.
        requires_grad: Whether to track gradients for this tensor.
        _children: Parent tensors in the computational graph.
        _op: Human-readable label of the operation that created this tensor.
    """

    def __init__(
        self,
        data: np.ndarray,
        requires_grad: bool = False,
        _children: tuple[Tensor, ...] = (),
        _op: str = "",
    ) -> None:
        self.data: np.ndarray = np.array(data, dtype=np.float64)
        self.grad: np.ndarray = np.zeros_like(self.data)
        self.requires_grad: bool = requires_grad
        self._backward: Callable[[], None] = lambda: None
        self._prev: set[Tensor] = set(_children)
        self._op: str = _op

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of the underlying data array."""
        return self.data.shape

    @property
    def ndim(self) -> int:
        """Number of dimensions."""
        return self.data.ndim

    @property
    def size(self) -> int:
        """Total number of elements."""
        return self.data.size

    @property
    def T(self) -> Tensor:
        """Transpose of the tensor."""
        out = Tensor(self.data.T, requires_grad=self.requires_grad, _children=(self,), _op="T")

        def _backward() -> None:
            self.grad += out.grad.T

        out._backward = _backward
        return out

    # ------------------------------------------------------------------
    # Arithmetic operations
    # ------------------------------------------------------------------

    def __add__(self, other: Numeric) -> Tensor:
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(
            self.data + other.data,
            requires_grad=self.requires_grad or other.requires_grad,
            _children=(self, other),
            _op="+",
        )

        def _backward() -> None:
            self.grad += _unbroadcast(out.grad, self.shape)
            other.grad += _unbroadcast(out.grad, other.shape)

        out._backward = _backward
        return out

    def __radd__(self, other: Numeric) -> Tensor:
        return self.__add__(other)

    def __mul__(self, other: Numeric) -> Tensor:
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(
            self.data * other.data,
            requires_grad=self.requires_grad or other.requires_grad,
            _children=(self, other),
            _op="*",
        )

        def _backward() -> None:
            self.grad += _unbroadcast(other.data * out.grad, self.shape)
            other.grad += _unbroadcast(self.data * out.grad, other.shape)

        out._backward = _backward
        return out

    def __rmul__(self, other: Numeric) -> Tensor:
        return self.__mul__(other)

    def __matmul__(self, other: Tensor) -> Tensor:
        """Matrix multiplication: self @ other."""
        out = Tensor(
            self.data @ other.data,
            requires_grad=self.requires_grad or other.requires_grad,
            _children=(self, other),
            _op="@",
        )

        def _backward() -> None:
            if self.data.ndim == 1 and other.data.ndim == 1:
                # dot product: (n,) @ (n,) -> scalar
                self.grad += out.grad * other.data
                other.grad += out.grad * self.data
            elif self.data.ndim == 1:
                # (n,) @ (n, m) -> (m,)
                self.grad += out.grad @ other.data.T
                other.grad += self.data[:, None] @ out.grad[None, :]
            elif other.data.ndim == 1:
                # (n, m) @ (m,) -> (n,)
                self.grad += out.grad[:, None] @ other.data[None, :]
                other.grad += self.data.T @ out.grad
            else:
                # (n, k) @ (k, m) -> (n, m)
                self.grad += out.grad @ other.data.T
                other.grad += self.data.T @ out.grad

        out._backward = _backward
        return out

    def __neg__(self) -> Tensor:
        out = Tensor(
            -self.data,
            requires_grad=self.requires_grad,
            _children=(self,),
            _op="neg",
        )

        def _backward() -> None:
            self.grad += -out.grad

        out._backward = _backward
        return out

    def __sub__(self, other: Numeric) -> Tensor:
        return self + (-other if isinstance(other, Tensor) else Tensor(-np.asarray(other, dtype=np.float64)))

    def __rsub__(self, other: Numeric) -> Tensor:
        return (-self) + other

    def __truediv__(self, other: Numeric) -> Tensor:
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(
            self.data / other.data,
            requires_grad=self.requires_grad or other.requires_grad,
            _children=(self, other),
            _op="/",
        )

        def _backward() -> None:
            self.grad += _unbroadcast(out.grad / other.data, self.shape)
            other.grad += _unbroadcast(-self.data / (other.data ** 2) * out.grad, other.shape)

        out._backward = _backward
        return out

    def __rtruediv__(self, other: Numeric) -> Tensor:
        other = other if isinstance(other, Tensor) else Tensor(other)
        return other / self

    def __pow__(self, exponent: float) -> Tensor:
        """Element-wise power: self ** exponent (scalar exponent only)."""
        assert isinstance(exponent, (int, float)), "only scalar exponents supported"
        out = Tensor(
            self.data ** exponent,
            requires_grad=self.requires_grad,
            _children=(self,),
            _op=f"**{exponent}",
        )

        def _backward() -> None:
            self.grad += exponent * (self.data ** (exponent - 1)) * out.grad

        out._backward = _backward
        return out

    # ------------------------------------------------------------------
    # Reduction / transformation operations
    # ------------------------------------------------------------------

    def sum(self, axis: int | tuple[int, ...] | None = None, keepdims: bool = False) -> Tensor:
        """Sum elements, optionally along given axis."""
        out = Tensor(
            np.sum(self.data, axis=axis, keepdims=keepdims),
            requires_grad=self.requires_grad,
            _children=(self,),
            _op="sum",
        )

        def _backward() -> None:
            grad = out.grad
            if axis is not None and not keepdims:
                # Restore reduced dimensions for broadcasting
                if isinstance(axis, int):
                    grad = np.expand_dims(grad, axis)
                else:
                    for ax in sorted(axis):
                        grad = np.expand_dims(grad, ax)
            self.grad += np.broadcast_to(grad, self.shape)

        out._backward = _backward
        return out

    def mean(self, axis: int | tuple[int, ...] | None = None, keepdims: bool = False) -> Tensor:
        """Mean of elements, optionally along given axis."""
        n = self.data.size if axis is None else np.prod([self.shape[a] for a in (axis if isinstance(axis, tuple) else (axis,))])
        return self.sum(axis=axis, keepdims=keepdims) / float(n)

    def log(self) -> Tensor:
        """Element-wise natural logarithm."""
        out = Tensor(
            np.log(self.data),
            requires_grad=self.requires_grad,
            _children=(self,),
            _op="log",
        )

        def _backward() -> None:
            self.grad += out.grad / self.data

        out._backward = _backward
        return out

    def exp(self) -> Tensor:
        """Element-wise exponential."""
        out = Tensor(
            np.exp(self.data),
            requires_grad=self.requires_grad,
            _children=(self,),
            _op="exp",
        )

        def _backward() -> None:
            self.grad += out.grad * out.data

        out._backward = _backward
        return out

    def reshape(self, *shape: int) -> Tensor:
        """Reshape tensor to the given shape."""
        out = Tensor(
            self.data.reshape(*shape),
            requires_grad=self.requires_grad,
            _children=(self,),
            _op="reshape",
        )

        def _backward() -> None:
            self.grad += out.grad.reshape(self.shape)

        out._backward = _backward
        return out

    # ------------------------------------------------------------------
    # Comparison (no grad needed)
    # ------------------------------------------------------------------

    def __gt__(self, other: Numeric) -> Tensor:
        other_val = other.data if isinstance(other, Tensor) else np.asarray(other)
        return Tensor(self.data > other_val)

    def __lt__(self, other: Numeric) -> Tensor:
        other_val = other.data if isinstance(other, Tensor) else np.asarray(other)
        return Tensor(self.data < other_val)

    def __ge__(self, other: Numeric) -> Tensor:
        other_val = other.data if isinstance(other, Tensor) else np.asarray(other)
        return Tensor(self.data >= other_val)

    def __le__(self, other: Numeric) -> Tensor:
        other_val = other.data if isinstance(other, Tensor) else np.asarray(other)
        return Tensor(self.data <= other_val)

    # ------------------------------------------------------------------
    # Backward pass
    # ------------------------------------------------------------------

    def backward(self) -> None:
        """Run reverse-mode autodiff from this (scalar) tensor.

        Builds a topological ordering via DFS and propagates gradients
        in reverse order.
        """
        topo: list[Tensor] = []
        visited: set[int] = set()

        def _build_topo(v: Tensor) -> None:
            if id(v) not in visited:
                visited.add(id(v))
                for child in v._prev:
                    _build_topo(child)
                topo.append(v)

        _build_topo(self)

        # Seed gradient
        self.grad = np.ones_like(self.data)
        for node in reversed(topo):
            node._backward()

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def item(self) -> float:
        """Return scalar value for 0-d / single-element tensors."""
        return float(self.data.item())

    def detach(self) -> Tensor:
        """Return a new tensor sharing data but detached from the graph."""
        return Tensor(self.data, requires_grad=False)

    def numpy(self) -> np.ndarray:
        """Return the underlying NumPy array."""
        return self.data

    def __repr__(self) -> str:
        grad_str = ", requires_grad=True" if self.requires_grad else ""
        return f"Tensor({self.data}{grad_str})"

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx):
        return Tensor(self.data[idx])


# ======================================================================
# Helper: un-broadcast a gradient to match the original shape
# ======================================================================

def _unbroadcast(grad: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    """Sum out axes that were broadcast-expanded so grad matches *shape*."""
    # Add leading dims if shapes differ in ndim
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    # Sum along axes where shape has size 1 (broadcast axis)
    for i, s in enumerate(shape):
        if s == 1:
            grad = grad.sum(axis=i, keepdims=True)
    return grad
