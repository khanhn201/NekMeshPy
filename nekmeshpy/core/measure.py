"""Metric measures over an element's node block: extent, size, centroid.

The rung-agnostic kernel behind ``linemesh.length`` / ``quadmesh.area`` /
``hexmesh.volume`` and the ``bounds`` / ``centroid`` each rung spells the same way.
Everything here takes a **node block** -- ``(E, (order+1)**dim, 3)`` in the
lexicographic lattice order the ``element_blocks`` queries return -- so the same
quadrature serves the corner (linear) and the curved order-N reading of a mesh, and
nothing in this module knows a container.
"""

from __future__ import annotations

from typing import Iterator, NamedTuple

import numpy as np

from .._typing import FloatArray, IntArray, Point, PointArray, Vec3
from .fields import gll_nodes, gll_weights, lagrange_derivative_matrix, lagrange_matrix
from .interp import corner_indices

#: Rough ceiling on the floats one quadrature chunk materializes.  The evaluated
#: tables are ``(chunk, q**dim, 3)`` and there are four of them at ``dim=3``, so a
#: high order over a large mesh is walked in pieces rather than allocated whole.
_CHUNK_FLOATS = 4_000_000


class Bounds(NamedTuple):
    """The axis-aligned bounding box of a point set -- ``min`` / ``max`` corners, with
    the derived readings (:attr:`size`, :attr:`center`, :attr:`diagonal`) that callers
    actually ask for."""

    #: Per-axis minimum ``(3,)``.
    min: Point
    #: Per-axis maximum ``(3,)``.
    max: Point

    @property
    def size(self) -> Vec3:
        """Per-axis extent ``max - min`` ``(3,)``."""
        out: Vec3 = self.max - self.min
        return out

    @property
    def center(self) -> Point:
        """The box centre -- the midpoint of the extremes, *not* a centroid."""
        out: Point = 0.5 * (self.min + self.max)
        return out

    @property
    def diagonal(self) -> float:
        """Length of the box diagonal -- the scalar "how big is this" reading."""
        return float(np.linalg.norm(self.max - self.min))


def bounds_of(points: PointArray) -> Bounds:
    """The bounding box of ``points`` (any leading shape, trailing axis the 3
    components).  Raises on an empty set: a box around nothing has no corners."""
    P: PointArray = np.asarray(points, dtype=float).reshape(-1, 3)
    if P.shape[0] == 0:
        raise ValueError("bounds: the mesh has no points, so it has no bounding box")
    return Bounds(P.min(axis=0), P.max(axis=0))


def corner_blocks(points: PointArray, conn: IntArray, dim: int) -> PointArray:
    """``(E, 2**dim, 3)`` node blocks of the **straight-sided** elements ``conn`` names
    -- the order-1 block of a mesh read at its corners only, whatever order it stores.
    """
    c: IntArray = np.asarray(conn, dtype=np.int64).reshape(-1, 2 ** dim)
    out: PointArray = np.empty((c.shape[0], 2 ** dim, 3), dtype=float)
    out[:, corner_indices(1, dim), :] = np.asarray(points, dtype=float)[c]
    return out


def _order_of(blocks: PointArray, dim: int) -> int:
    """The order a ``(E, (order+1)**dim, 3)`` block stores, read off its middle axis --
    the same "the nodes say what order this is" rule the containers hold to."""
    m = int(blocks.shape[1])
    row = int(round(m ** (1.0 / dim)))
    if row ** dim != m or row < 2:
        raise ValueError("measure: a dim-%d node block must be (E,(order+1)**%d,3); "
                         "got %d nodes per element" % (dim, dim, m))
    return row - 1


_RULE_CACHE: dict[int, tuple[FloatArray, FloatArray, FloatArray]] = {}


def _rule(order: int) -> tuple[FloatArray, FloatArray, FloatArray]:
    """The 1-D quadrature for an order-``order`` element: ``(w, B, D)`` -- GLL weights
    on ``[0,1]`` and the order-``order`` Lagrange basis and its derivative sampled at
    those points.

    The rule is taken at ``2*order`` so it integrates a **volume** exactly: a
    ``dim``-D Jacobian determinant reaches degree ``3*order - 1`` per axis, and a GLL
    rule of ``q+1`` points is exact to ``2*q - 1``.  A *curved* length or area is not
    a polynomial at all (the integrand carries a square root), so there the same rule
    is an approximation that converges with the order -- see the ``high_order`` note on
    each rung's measure."""
    cached = _RULE_CACHE.get(order)
    if cached is not None:
        return cached
    q = max(2 * order, 2)
    x = gll_nodes(q)
    nodes = gll_nodes(order)
    rule = (gll_weights(q), lagrange_matrix(nodes, x), lagrange_derivative_matrix(
        nodes, x))
    _RULE_CACHE[order] = rule
    return rule


def _chunks(n: int, per_element: int) -> Iterator[slice]:
    """Element slices sized so one chunk's evaluated tables stay near
    :data:`_CHUNK_FLOATS`."""
    step = max(1, _CHUNK_FLOATS // max(1, per_element))
    for start in range(0, n, step):
        yield slice(start, min(start + step, n))


def integrate(blocks: PointArray, dim: int, *, moments: bool = False
              ) -> tuple[FloatArray, PointArray]:
    """``(measure (E,), moment (E,3))`` -- each element's ``integral d(Omega)`` and, when
    ``moments`` is asked for, its ``integral x d(Omega)`` (zeros otherwise).

    At ``dim=3`` the measure is the **signed** volume integral of the isoparametric
    map, so an inverted hex contributes negatively and a wholly mis-oriented mesh
    reads negative -- the same sign convention as the scaled Jacobian.  A length
    (``dim=1``) or a surface area (``dim=2``) embedded in 3-D has no such sign and
    comes back non-negative."""
    if dim not in (1, 2, 3):
        raise ValueError("measure: dim must be 1, 2 or 3, got %d" % dim)
    P: PointArray = np.asarray(blocks, dtype=float)
    if P.ndim != 3 or P.shape[2] != 3:
        raise ValueError("measure: blocks must be (E,(order+1)**dim,3), got %s"
                         % (P.shape,))
    order = _order_of(P, dim)
    w, B, D = _rule(order)
    n, q = order + 1, w.shape[0]
    E = P.shape[0]
    meas: FloatArray = np.zeros(E, dtype=float)
    mom: PointArray = np.zeros((E, 3), dtype=float)
    for sl in _chunks(E, q ** dim * 3):
        blk = P[sl]
        if dim == 1:
            V = np.einsum("ai,eic->eac", D, blk, optimize=True)
            g: FloatArray = np.linalg.norm(V, axis=-1)
            meas[sl] = np.einsum("a,ea->e", w, g, optimize=True)
            if moments:
                X = np.einsum("ai,eic->eac", B, blk, optimize=True)
                mom[sl] = np.einsum("a,ea,eac->ec", w, g, X, optimize=True)
        elif dim == 2:
            b2 = blk.reshape(-1, n, n, 3)                      # [e, j, i, c]
            Xu = np.einsum("bj,ai,ejic->ebac", B, D, b2, optimize=True)
            Xv = np.einsum("bj,ai,ejic->ebac", D, B, b2, optimize=True)
            g = np.linalg.norm(np.cross(Xu, Xv), axis=-1)
            meas[sl] = np.einsum("b,a,eba->e", w, w, g, optimize=True)
            if moments:
                X = np.einsum("bj,ai,ejic->ebac", B, B, b2, optimize=True)
                mom[sl] = np.einsum("b,a,eba,ebac->ec", w, w, g, X, optimize=True)
        else:
            b3 = blk.reshape(-1, n, n, n, 3)                   # [e, k, j, i, c]
            Xu = np.einsum("ck,bj,ai,ekjid->ecbad", B, B, D, b3, optimize=True)
            Xv = np.einsum("ck,bj,ai,ekjid->ecbad", B, D, B, b3, optimize=True)
            Xw = np.einsum("ck,bj,ai,ekjid->ecbad", D, B, B, b3, optimize=True)
            g = np.einsum("...d,...d->...", np.cross(Xu, Xv), Xw, optimize=True)
            meas[sl] = np.einsum("c,b,a,ecba->e", w, w, w, g, optimize=True)
            if moments:
                X = np.einsum("ck,bj,ai,ekjid->ecbad", B, B, B, b3, optimize=True)
                mom[sl] = np.einsum("c,b,a,ecba,ecbad->ed", w, w, w, g, X,
                                    optimize=True)
    return meas, mom


def centroid_of(blocks: PointArray, dim: int, who: str) -> Point:
    """The measure-weighted centroid ``integral x d(Omega) / integral d(Omega)`` of the
    elements ``blocks`` describes -- the mass-property centroid, not the mean of the
    nodes."""
    meas, mom = integrate(blocks, dim, moments=True)
    total = float(meas.sum())
    if total == 0.0:
        raise ValueError(
            "%s: the elements measure zero in total, so they have no centroid "
            "(an empty mesh, or one whose elements cancel because some are inverted)"
            % who)
    out: Point = mom.sum(axis=0) / total
    return out


__all__ = ["Bounds", "bounds_of", "centroid_of", "corner_blocks", "integrate"]
