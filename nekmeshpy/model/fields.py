"""Mesh-sizing fields (gmsh-style) and 1-D point distributions."""

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


def validate_layers(positions: int | FloatArray, who: str) -> FloatArray:
    """Normalize a layer *specification* to the flattened array of normalized layer
    positions every sweep / fill factory consumes. ``who`` labels the caller in errors.
    """
    if isinstance(positions, (int, np.integer)) and not isinstance(positions, bool):
        n = int(positions)
        if n < 1:
            raise ValueError(
                "%s: an int layer count means n uniform layers, so it needs n >= 1; "
                "got %d -- pass an explicit position array (e.g. "
                "geometric_spacing(n, ratio)) for a graded sweep" % (who, n))
        return uniform_spacing(n)
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


def reject_loop_caps(who: str, *caps: object) -> None:
    """Guard for the periodic (``loop=True``) branch of the ``loft`` sweep at every rung
    of the ladder: a closed sweep has no near / far cap, so any non-empty end-cap tag
    argument is necessarily a caller mistake and is rejected loudly rather than silently
    dropped."""
    for cap in caps:
        if cap is None:
            continue
        if isinstance(cap, str):
            named = bool(cap)
        else:
            arr = np.asarray(cap, dtype=np.str_).reshape(-1)
            named = bool(arr.size) and bool(np.any(arr != ""))
        if named:
            raise ValueError(
                "%s: loop=True is a periodic sweep with no near/far cap, so "
                "first_tag/last_tag cannot be placed (got %r) -- drop the cap tags, "
                "or use loop=False for an open stack." % (who, cap))


# -- high-order reference nodes (Gauss-Lobatto-Legendre) ----------------
_GLL_CACHE: dict[int, FloatArray] = {}


def gll_nodes(order: int) -> FloatArray:
    """The ``order+1`` Gauss-Lobatto-Legendre nodes mapped to ``[0, 1]``, ascending."""
    if order < 1:
        raise ValueError("gll_nodes needs order >= 1")
    cached = _GLL_CACHE.get(order)
    if cached is not None:
        return cached
    if order == 1:
        nodes = np.array([0.0, 1.0])
    else:
        # interior GLL nodes on [-1,1] are the roots of P'_order; build P'_order
        # from the Legendre series and take its roots, then add the endpoints.
        dP = np.polynomial.legendre.Legendre.basis(order).deriv()
        interior = np.sort(dP.roots().real)
        x = np.concatenate([[-1.0], interior, [1.0]])
        nodes = 0.5 * (x + 1.0)                       # map [-1,1] -> [0,1]
        nodes[0], nodes[-1] = 0.0, 1.0                # pin endpoints exactly
    _GLL_CACHE[order] = nodes
    return nodes


def gll_weights(order: int) -> FloatArray:
    """The ``order+1`` GLL quadrature weights on ``[0, 1]`` (sum to 1), matching
    :func:`gll_nodes`.  Used by the (deferred) order-N quality metric."""
    if order < 1:
        raise ValueError("gll_weights needs order >= 1")
    x = 2.0 * gll_nodes(order) - 1.0                  # back to [-1,1]
    Pn = np.polynomial.legendre.Legendre.basis(order)(x)
    w = 2.0 / (order * (order + 1) * Pn ** 2)         # standard GLL weight
    return w / w.sum()                                # normalized to [0,1] span


def lagrange_matrix(nodes: FloatArray, eval_pts: FloatArray) -> FloatArray:
    """The ``(len(eval_pts), len(nodes))`` matrix of Lagrange basis values: row
    ``i`` holds ``L_k(eval_pts[i])`` for each node ``k``.  ``M @ f`` interpolates
    nodal values ``f`` to ``eval_pts``; each row sums to 1 (partition of unity)."""
    xn = np.asarray(nodes, dtype=float).ravel()
    xe = np.asarray(eval_pts, dtype=float).ravel()
    n = xn.size
    M = np.ones((xe.size, n))
    for k in range(n):
        for j in range(n):
            if j != k:
                M[:, k] *= (xe - xn[j]) / (xn[k] - xn[j])
    return M


def lagrange_derivative_matrix(nodes: FloatArray, eval_pts: FloatArray) -> FloatArray:
    """The ``(len(eval_pts), len(nodes))`` matrix of Lagrange basis derivatives: entry
    ``(i, k)`` is ``L'_k(eval_pts[i])``."""
    xn = np.asarray(nodes, dtype=float).ravel()
    xe = np.asarray(eval_pts, dtype=float).ravel()
    n = xn.size
    D = np.zeros((xe.size, n))
    for k in range(n):
        # L'_k = sum_{m != k} 1/(xn[k]-xn[m]) * prod_{j != k,m} (x-xn[j])/(xn[k]-xn[j])
        for m in range(n):
            if m == k:
                continue
            term = np.full(xe.size, 1.0 / (xn[k] - xn[m]))
            for j in range(n):
                if j != k and j != m:
                    term *= (xe - xn[j]) / (xn[k] - xn[j])
            D[:, k] += term
    return D


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
