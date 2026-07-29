"""Mesh-sizing fields (gmsh-style) and 1-D point distributions.

A ``Field`` maps points in space to a target element size.  Fields compose
(``MinField``) and drive graded edge distributions via
``distribution_from_field``.  ``geometric_spacing`` gives a fixed grading when no
field is supplied.
"""

from __future__ import annotations

import numpy as np

from .._typing import FloatArray, Point, PointArray


class Field:
    """Base sizing field: ``field(points) -> sizes`` mapping ``(k,3)`` points to
    ``k`` target edge sizes."""

    def __call__(self, points: PointArray) -> FloatArray:
        raise NotImplementedError

    def sample(self, point: Point) -> float:
        """Target size at a single ``(3,)`` point."""
        return float(self(np.asarray(point, float)[None, :])[0])


class ConstantField(Field):
    def __init__(self, size: float) -> None:
        self.size = float(size)

    def __call__(self, points: PointArray) -> FloatArray:
        return np.full(np.asarray(points).shape[0], self.size)


class AxisLinearField(Field):
    """Size varies linearly along one axis from ``size0`` at ``c0`` to ``size1``
    at ``c1`` (clamped outside)."""

    def __init__(self, axis: int, c0: float, size0: float, c1: float, size1: float) -> None:
        self.axis, self.c0, self.s0, self.c1, self.s1 = axis, c0, size0, c1, size1

    def __call__(self, points: PointArray) -> FloatArray:
        x = np.asarray(points, float)[:, self.axis]
        t = np.clip((x - self.c0) / (self.c1 - self.c0), 0.0, 1.0)
        return self.s0 + t * (self.s1 - self.s0)


class DistanceField(Field):
    """Size grows from ``size_near`` at the given points to ``size_far`` beyond
    ``dist_far`` (linear ramp on distance to the nearest source point)."""

    def __init__(self, points: PointArray, size_near: float, dist_far: float,
                 size_far: float) -> None:
        self.src = np.asarray(points, float).reshape(-1, 3)
        self.size_near, self.dist_far, self.size_far = size_near, dist_far, size_far

    def __call__(self, points: PointArray) -> FloatArray:
        P = np.asarray(points, float).reshape(-1, 3)
        d = np.sqrt(((P[:, None, :] - self.src[None, :, :]) ** 2).sum(-1)).min(1)
        t = np.clip(d / self.dist_far, 0.0, 1.0)
        return self.size_near + t * (self.size_far - self.size_near)


class MinField(Field):
    """Pointwise minimum of several fields (the finest constraint wins)."""

    def __init__(self, *fields: Field) -> None:
        self.fields = fields

    def __call__(self, points: PointArray) -> FloatArray:
        return np.min([f(points) for f in self.fields], axis=0)


# -- 1-D distributions --------------------------------------------------
def geometric_spacing(n: int, ratio: float = 1.0) -> FloatArray:
    """``n+1`` normalized positions in ``[0,1]`` with a geometric size ratio
    between consecutive cells (``ratio==1`` -> uniform)."""
    if n < 1:
        raise ValueError("n must be >= 1")
    if abs(ratio - 1.0) < 1e-12:
        return np.linspace(0.0, 1.0, n + 1)
    w = ratio ** np.arange(n)          # cell widths
    pos = np.concatenate([[0.0], np.cumsum(w)])
    return pos / pos[-1]


def uniform_spacing(n: int) -> FloatArray:
    """``n+1`` uniformly spaced positions in ``[0, 1]``; shorthand for
    ``geometric_spacing(n, 1.0)``."""
    return geometric_spacing(n, 1.0)


def symmetric_spacing(n: int, ratio: float = 1.0) -> FloatArray:
    """``n+1`` positions in ``[0,1]`` clustered toward both ends: geometric
    spacing per half, mirrored (``n`` must be even; ``ratio==1`` -> uniform)."""
    if n % 2:
        raise ValueError("symmetric_spacing needs an even n, got %d" % n)
    m = n // 2
    g = geometric_spacing(m, ratio)                  # [0,1], cells grow away from 0
    first = 0.5 * g                                  # [0, 0.5], clustered near 0
    second = 1.0 - 0.5 * g[::-1]                      # [0.5, 1], clustered near 1
    return np.concatenate([first[:-1], second])      # 2*m+1 = n+1 fractions


def validate_layers(positions: FloatArray, who: str) -> FloatArray:
    """Validate a normalized layer-position array and return it flattened:
    strictly increasing values in ``[0, 1]`` with an explicit first position and a
    last position of ``1``.  ``who`` labels the caller in errors."""
    p = np.asarray(positions, dtype=float).ravel()
    if p.size < 2:
        raise ValueError("%s: needs at least 2 layer positions" % who)
    if np.any(p < 0.0) or np.any(p > 1.0):
        raise ValueError("%s: layer positions must lie in [0, 1]" % who)
    if np.any(np.diff(p) <= 0.0):
        raise ValueError("%s: layer positions must be strictly increasing" % who)
    if not np.isclose(float(p[-1]), 1.0):
        raise ValueError("%s: last layer position must be 1.0" % who)
    return p


def distribution_from_field(field: Field, p0: Point, p1: Point,
                            max_cells: int = 200) -> FloatArray:
    """Graded positions along ``p0 -> p1`` where each cell length approximates the
    field's target size.  Greedy walk stepping by the local size, then rescaled to
    land on ``p1``.  Returns normalized positions in ``[0,1]``."""
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
