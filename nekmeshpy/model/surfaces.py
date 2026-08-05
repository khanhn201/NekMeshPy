"""Curves carried as their **parametrization on a surface**, rather than as points.

A :class:`SurfaceCurve` is a map from a curve parameter to the ``(u, v)`` coordinates
of a surface, plus the parameter values of its own nodes.  Nothing here knows what the
surface *is*: a caller supplies the ``(K,2) -> (K,3)`` map separately, when it finally
meshes the curve with :func:`linemesh.shape.on_surface
<nekmeshpy.linemesh.shape.on_surface>` or fills a patch with
:func:`quadmesh.shape.tri_patch <nekmeshpy.quadmesh.shape.tri_patch>`.

**Why parameters and not points.**  Interpolating between two curves *on* a surface has
to happen in parameter space.  Lerping their points instead cuts a chord, and a chord
between two points of a cylinder dips inside it -- so every intermediate station of a
transition would sit a little proud of the wall, by an amount no refinement removes.
In parameter space there is no chord: each node lands on the true surface exactly, at
any number of stations and at any order.

The combinators below are therefore all closed over ``g``: they build a new
parametrization out of old ones and never sample anything.  Sampling happens once, at
the end, when the surface map is applied to the whole node lattice.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Callable, NamedTuple

import numpy as np

from .._typing import FloatArray, PointArray

#: A surface map: ``(K,2)`` surface parameters to ``(K,3)`` points.
SurfaceMap = Callable[[FloatArray], PointArray]


class SurfaceCurve(NamedTuple):
    """A curve living in a surface's parameter domain."""

    #: ``(K,)`` curve parameters -> ``(K, 2)`` surface parameters.  Vectorized.
    g: Callable[[FloatArray], FloatArray]
    #: The curve-parameter values of this curve's own nodes, one per node.  May run
    #: **descending** -- that is how :func:`reverse` traverses a curve the other way,
    #: and what ``loft_fn`` accepts a descending sequence for.
    fr: FloatArray


def curve(g: Callable[[FloatArray], FloatArray],
          fr: FloatArray | Sequence[float]) -> SurfaceCurve:
    """A :class:`SurfaceCurve` from a parametrization and its node parameters."""
    f: FloatArray = np.asarray(fr, dtype=float).ravel()
    if f.size < 2:
        raise ValueError("surfaces.curve: need at least 2 node parameters, got %d"
                         % f.size)
    return SurfaceCurve(g, f)


def ruled(pa: FloatArray | Sequence[float], pb: FloatArray | Sequence[float],
          n: int) -> SurfaceCurve:
    """The straight segment from ``pa`` to ``pb`` **in parameter space**, with ``n``
    elements evenly spaced in the parameter, on the domain ``[0, 1]``.

    Straight in parameters, not in space: on a cylinder this is a helix, which is what
    makes it stay on the wall."""
    a: FloatArray = np.asarray(pa, dtype=float).reshape(2)
    b: FloatArray = np.asarray(pb, dtype=float).reshape(2)

    def g(x: FloatArray) -> FloatArray:
        xi = np.asarray(x, dtype=float)[:, None]
        return (1.0 - xi) * a + xi * b

    return SurfaceCurve(g, np.linspace(0.0, 1.0, int(n) + 1))


def blend(a: SurfaceCurve, b: SurfaceCurve, lam: float) -> SurfaceCurve:
    """The two curves interpolated in parameter space at ``lam`` -- so the result is
    still exactly on the surface, which a point-space lerp would not be.

    Both must share a parameter domain; the result carries ``a``'s nodes."""
    def g(x: FloatArray) -> FloatArray:
        return (1.0 - lam) * a.g(x) + lam * b.g(x)

    return SurfaceCurve(g, a.fr)


def reverse(c: SurfaceCurve) -> SurfaceCurve:
    """The same curve traversed the other way, by reversing its node parameters."""
    return SurfaceCurve(c.g, c.fr[::-1])


def shift(c: SurfaceCurve, delta: FloatArray | Sequence[float]) -> SurfaceCurve:
    """The same curve with a constant offset added in parameter space.

    For a periodic coordinate this is a **rebranching**, not a move: shifting an angle
    by a whole turn names the identical points on a different branch, which is what
    lets curves meeting at a corner be blended in that angle rather than jumping a
    period between them."""
    d: FloatArray = np.asarray(delta, dtype=float).reshape(2)

    def g(x: FloatArray) -> FloatArray:
        return c.g(x) + d

    return SurfaceCurve(g, c.fr)


def reparam(c: SurfaceCurve, pa: FloatArray | Sequence[float],
            pb: FloatArray | Sequence[float]) -> SurfaceCurve:
    """The straight ``pa -> pb`` parameter segment, but expressed on ``c``'s **own**
    parameter domain and carrying ``c``'s own nodes -- so the two can be
    :func:`blend`-ed station by station.

    The companion to :func:`ruled`: same curve, different domain.  A transition that
    morphs a shaped edge into a plain one needs both ends written in one parameter."""
    a: FloatArray = np.asarray(pa, dtype=float).reshape(2)
    b: FloatArray = np.asarray(pb, dtype=float).reshape(2)
    t0, t1 = float(c.fr[0]), float(c.fr[-1])
    if t0 == t1:
        raise ValueError("surfaces.reparam: the curve's node parameters span nothing "
                         "(first == last == %g), so there is no domain to map onto" % t0)

    def g(x: FloatArray) -> FloatArray:
        xi = (np.asarray(x, dtype=float) - t0) / (t1 - t0)
        return a + xi[:, None] * (b - a)

    return SurfaceCurve(g, c.fr)


def node(c: SurfaceCurve, i: int) -> FloatArray:
    """The ``(2,)`` surface parameters of node ``i``.

    Read off the curve's *own* sampling rather than at the midpoint of its parameter
    range: on a graded curve those are different points, and a patch split at the
    wrong one no longer meets its neighbour."""
    return np.asarray(c.g(c.fr[i:i + 1]), dtype=float)[0]


def segment(c: SurfaceCurve, i0: int, i1: int) -> Callable[[FloatArray], FloatArray]:
    """The piece of ``c`` between nodes ``i0`` and ``i1``, as a parametrization on
    ``[0, 1]`` -- ready for a Coons patch, which wants plain callables.

    Remapped **through the curve's own node values**, so the piece reproduces the
    parent's grading rather than re-spacing it evenly; ``i1 < i0`` walks backwards."""
    step = 1 if i1 > i0 else -1
    idx = np.arange(i0, i1 + step, step)
    fr, m = c.fr[idx], idx.size - 1
    if m < 1:
        raise ValueError("surfaces.segment: i0 and i1 name the same node (%d)" % i0)

    def remap(s: FloatArray) -> FloatArray:
        u = np.clip(np.asarray(s, dtype=float), 0.0, 1.0) * m
        i = np.clip(np.floor(u).astype(int), 0, m - 1)
        return (1.0 - (u - i)) * fr[i] + (u - i) * fr[i + 1]

    def g(s: FloatArray) -> FloatArray:
        return c.g(remap(s))

    return g


def spoke(start: FloatArray | Sequence[float], end: FloatArray | Sequence[float],
          ) -> Callable[[FloatArray], FloatArray]:
    """The straight parameter-space segment ``start -> end`` on ``[0, 1]``, as a plain
    callable -- :func:`ruled` without a node table, for a Coons boundary."""
    a: FloatArray = np.asarray(start, dtype=float).reshape(2)
    b: FloatArray = np.asarray(end, dtype=float).reshape(2)

    def g(s: FloatArray) -> FloatArray:
        return a + np.asarray(s, dtype=float)[:, None] * (b - a)

    return g


__all__ = [
    "SurfaceCurve",
    "SurfaceMap",
    "blend",
    "curve",
    "node",
    "reparam",
    "reverse",
    "ruled",
    "segment",
    "shift",
    "spoke",
]
