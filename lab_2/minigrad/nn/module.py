"""Module — base class for all neural network layers and models."""

from __future__ import annotations

import numpy as np

from minigrad.tensor import Tensor


class Module:
    """Base class for all neural-network modules.

    Subclasses must override :meth:`forward`.  Parameters are discovered
    automatically by inspecting ``__dict__`` for :class:`Tensor` and
    :class:`Module` instances.
    """

    def forward(self, x: Tensor) -> Tensor:
        """Compute the forward pass.  Must be overridden by subclasses."""
        raise NotImplementedError

    def __call__(self, x: Tensor) -> Tensor:
        """Alias for :meth:`forward`."""
        return self.forward(x)

    def parameters(self) -> list[Tensor]:
        """Return all trainable parameters recursively."""
        params: list[Tensor] = []
        for value in self.__dict__.values():
            if isinstance(value, Tensor) and value.requires_grad:
                params.append(value)
            elif isinstance(value, Module):
                params.extend(value.parameters())
            elif isinstance(value, (list, tuple)):
                for item in value:
                    if isinstance(item, Tensor) and item.requires_grad:
                        params.append(item)
                    elif isinstance(item, Module):
                        params.extend(item.parameters())
        return params

    def zero_grad(self) -> None:
        """Reset gradients of all parameters to zero."""
        for p in self.parameters():
            p.grad = np.zeros_like(p.data)
