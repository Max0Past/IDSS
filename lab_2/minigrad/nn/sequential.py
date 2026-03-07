"""Sequential — ordered container that chains layers."""

from __future__ import annotations

from minigrad.tensor import Tensor
from minigrad.nn.module import Module


class Sequential(Module):
    """A sequential container that passes input through layers in order.

    Args:
        *layers: Variable number of :class:`Module` instances.

    Example::

        model = Sequential(
            Linear(2, 4),
            ReLU(),
            Linear(4, 1),
            Sigmoid(),
        )
        out = model(x)
    """

    def __init__(self, *layers: Module) -> None:
        self.layers: list[Module] = list(layers)

    def forward(self, x: Tensor) -> Tensor:
        """Pass *x* through each layer sequentially."""
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self) -> list[Tensor]:
        """Collect parameters from all child layers."""
        params: list[Tensor] = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params

    def __repr__(self) -> str:
        layer_strs = ",\n  ".join(repr(l) for l in self.layers)
        return f"Sequential(\n  {layer_strs}\n)"
