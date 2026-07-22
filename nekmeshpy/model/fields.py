"""Mesh-sizing fields (gmsh-style) and 1-D node distributions.

A :class:`Field` maps points in space to a target element size.  Fields compose
(``MinField``) and drive graded edge distributions via
:func:`distribution_from_field`, so structured algorithms
(:class:`~nekmeshpy.algorithms.blocks.TransfiniteBlock`) can honour a size field instead of
a fixed division count.

For convenience there is also :func:`geometric_spacing` (a fixed geometric
grading) used when no field is supplied.
"""

from __future__ import annotations

import numpy as np


class Field:
    """Base sizing field: ``field(points) -> sizes``.  ``points`` is ``(k,3)``;
    returns a length-``k`` array of target edge sizes."""

    def __call__(self, points: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def sample(self, point: np.ndarray) -> float:
        """Target size at a single ``(3,)`` point."""
        return float(self(np.asarray(point, float)[None, :])[0])


class ConstantField(Field):
    def __init__(self, size: float) -> None:
        self.size = float(size)

    def __call__(self, points: np.ndarray) -> np.ndarray:
        return np.full(np.asarray(points).shape[0], self.size)


class AxisLinearField(Field):
    """Size varies linearly along one axis from ``size0`` at ``c0`` to ``size1``
    at ``c1`` (clamped outside)."""

    def __init__(self, axis: int, c0: float, size0: float, c1: float, size1: float) -> None:
        self.axis, self.c0, self.s0, self.c1, self.s1 = axis, c0, size0, c1, size1

    def __call__(self, points: np.ndarray) -> np.ndarray:
        x = np.asarray(points, float)[:, self.axis]
        t = np.clip((x - self.c0) / (self.c1 - self.c0), 0.0, 1.0)
        return self.s0 + t * (self.s1 - self.s0)


class DistanceField(Field):
    """Size grows from ``size_near`` at the given points to ``size_far`` beyond
    ``dist_far`` (linear ramp on distance to the nearest source point)."""

    def __init__(self, points: np.ndarray, size_near: float, dist_far: float,
                 size_far: float) -> None:
        self.src = np.asarray(points, float).reshape(-1, 3)
        self.size_near, self.dist_far, self.size_far = size_near, dist_far, size_far

    def __call__(self, points: np.ndarray) -> np.ndarray:
        P = np.asarray(points, float).reshape(-1, 3)
        d = np.sqrt(((P[:, None, :] - self.src[None, :, :]) ** 2).sum(-1)).min(1)
        t = np.clip(d / self.dist_far, 0.0, 1.0)
        return self.size_near + t * (self.size_far - self.size_near)


class MinField(Field):
    """Pointwise minimum of several fields (the finest constraint wins)."""

    def __init__(self, *fields: Field) -> None:
        self.fields = fields

    def __call__(self, points: np.ndarray) -> np.ndarray:
        return np.min([f(points) for f in self.fields], axis=0)


# -- 1-D distributions --------------------------------------------------
def geometric_spacing(n: int, ratio: float = 1.0) -> np.ndarray:
    """``n+1`` normalized node positions in ``[0,1]`` with a geometric size
    ratio between consecutive cells (``ratio==1`` -> uniform)."""
    if n < 1:
        raise ValueError("n must be >= 1")
    if abs(ratio - 1.0) < 1e-12:
        return np.linspace(0.0, 1.0, n + 1)
    w = ratio ** np.arange(n)          # cell widths
    pos = np.concatenate([[0.0], np.cumsum(w)])
    return pos / pos[-1]


def distribution_from_field(field: Field, p0: np.ndarray, p1: np.ndarray,
                            max_cells: int = 200) -> np.ndarray:
    """Choose graded node positions along the segment ``p0 -> p1`` so each cell
    length approximates the field's target size there.  Returns normalized
    positions in ``[0,1]`` (endpoints included).

    A greedy walk: step by the locally-sampled size until the far end is
    reached, then rescale to land exactly on ``p1``.
    """
    p0 = np.asarray(p0, float)
    p1 = np.asarray(p1, float)
    length = float(np.linalg.norm(p1 - p0))
    if length == 0:
        return np.array([0.0, 1.0])
    pos = [0.0]
    s = 0.0
    for _ in range(max_cells):
        pt = p0 + (s / length) * (p1 - p0)
        step = max(field.sample(pt), length / max_cells)
        s += step
        if s >= length:
            break
        pos.append(s / length)
    pos.append(1.0)
    return np.array(pos)
