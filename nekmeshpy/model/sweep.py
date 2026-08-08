"""The index space every ``loft`` shares: a slice, copied and joined across layers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .._typing import IntArray

__all__ = ["Sweep"]


@dataclass(frozen=True)
class Sweep:
    """The product numbering of a sweep -- ``n_levels`` copies of one slice, joined
    across ``n_layers`` layers.

    Every entity of the result is one of exactly two kinds, and each has a closed form:

    * **carried** -- an entity *of the slice*, sitting at one level.
    * **swept** -- an entity of the slice one dimension *down*, dragged across one
      layer.

    A rung states which of its local entities are which; the ids follow.  ``loop``
    lives in :attr:`nxt` and nowhere else, so a periodic sweep is the same arithmetic
    with one modulo rather than a seam branch.
    """

    n_levels: int
    n_layers: int
    nn: int
    loop: bool = False

    @property
    def nxt(self) -> IntArray:
        """``(n_layers,)`` -- the level each layer sweeps *to*, wrapping on a loop."""
        if not self.n_layers:
            return np.zeros(0, dtype=np.int64)
        return np.arange(1, self.n_layers + 1, dtype=np.int64) % self.n_levels

    def point(self, level: IntArray | int, v: IntArray | int) -> IntArray:
        """Global id of slice point ``v`` at ``level``."""
        return np.asarray(level, dtype=np.int64) * self.nn + v

    def carried(self, level: IntArray | int, k: IntArray | int,
                n_carried: int) -> IntArray:
        """Global id of slice entity ``k`` sitting at ``level``."""
        return np.asarray(level, dtype=np.int64) * n_carried + k

    def swept(self, layer: IntArray | int, k: IntArray | int,
              n_carried: int, n_swept: int) -> IntArray:
        """Global id of sub-entity ``k`` dragged across ``layer``."""
        return (self.n_levels * n_carried
                + np.asarray(layer, dtype=np.int64) * n_swept + k)

    def total(self, n_carried: int, n_swept: int) -> int:
        """How many entities the two families come to."""
        return self.n_levels * n_carried + self.n_layers * n_swept

    def elements(self, n_per_slice: int) -> tuple[IntArray, IntArray, IntArray]:
        """``(layer, slice element, swept-to level)`` of every element, in the
        layer-major order ``e = layer * n_per_slice + k`` every rung numbers by."""
        i: IntArray = np.repeat(np.arange(self.n_layers, dtype=np.int64), n_per_slice)
        k: IntArray = np.tile(np.arange(n_per_slice, dtype=np.int64), self.n_layers)
        j = self.nxt[i] if self.n_layers else i
        return i, k, j
