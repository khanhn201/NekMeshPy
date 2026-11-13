"""Shape model of prenek's **Autosphere** (``Nek5000/tools/prenek/curve.f``): the filled
cylinder quadrant (2-D, ``filled_cyl_quadrant``) and the filled sphere octant (3-D,
``filled_sphere_oct``), as the **parent elements** prenek builds before ``msplite``
subdivides them.

Each parent is a map from the unit reference square / cube onto the domain, reproduced
from prenek's own construction rather than approximated:

* 2-D quadrant, radius ``R`` in the first quadrant of the ``(x, y)`` plane -- one core
  element with corners ``0, ra e_x, rb (e_x+e_y)/sqrt2, ra e_y`` whose two outer edges are
  quadratic (midside node ``W2 * P_ra(m) + (1 - W2) * m``: the chord midpoint ``m``
  projected onto the circle of radius ``ra``, over-weighted by ``W2 = 1.1``), and two shell
  elements from those curved edges out to exact arcs of radius ``R``.  prenek's
  ``xyzquad`` (Gordon-Hall from the edges) and ``arcsrf_e`` (exact uniform-angle arc
  blended linearly to the opposite edge) together are the Coons patch of the four edges.
* 3-D octant -- one core hex with corners ``s2vec``-projected to ``ra`` / ``rb`` / ``rc``
  by the number of non-zero coordinates, the 9 edges not touching the origin quadratic
  with midside ``W3 * m`` (``W3 = 1 + 0.2 a / R``), mapped by the edge-based transfinite
  interpolation (= ``gh_face_extend`` from edges); and three shell hexes on the core's
  outer faces, whose outer face is the sphere patch of ``sphsrf_e`` (great-circle edges at
  uniform angle, interior nodes at the intersection of the great circles through opposite
  edge points) added to the base map with a linear blend in the radial direction.  The
  three shells are one shell rotated by 120 / 240 degrees about ``(1, 1, 1)``.

with prenek's constants ``a = 0.24 R`` (shell thickness on the axes), ``b = 0.9 a``,
``c = 0.85 a``, ``ra = R - a``, ``rb = R - 2b + a``, ``rc = R - 2c + a``.

``msplite`` then splits the core uniformly into ``n_core`` per direction and each shell
into ``n_core`` tangentially by a geometric progression ``ratio`` radially
(:func:`geometric_fractions`), sampling the parent map -- which is exactly what the
``loft_fn`` factories built on these maps do, at any order.  (prenek stores the children
as quadratic elements and converts their midside nodes to circles; sampling the parent
map at the mesh order is the exact version of that.)

Every map takes broadcastable arrays of reference coordinates in ``[0, 1]`` and returns
``(..., 3)`` points in the canonical frame (centre at the origin, quadrant / octant in the
positive coordinates); the factories place them.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from .._typing import FloatArray, PointArray
from .fields import validate_layers

A_FRAC = 0.24            # a / R  (prenek: a = 0.12 for R = 0.5)
B_FRAC = 0.90            # b / a
C_FRAC = 0.85            # c / a
W2 = 1.10                # 2-D core: weight on the projected midside node
W3_GAIN = 0.2            # 3-D core: midside = (1 + W3_GAIN * a / R) * chord midpoint
DEFAULT_RATIO = 0.7

Curve = Callable[[FloatArray], PointArray]
Map2 = Callable[[FloatArray, FloatArray], PointArray]
Map3 = Callable[[FloatArray, FloatArray, FloatArray], PointArray]


def geometric_fractions(n: int, ratio: float) -> FloatArray:
    """prenek's ``get_ratio``: ``n + 1`` fractions in ``[0, 1]`` whose steps form a
    geometric progression, each ``ratio`` times the one before (``ratio < 1`` shrinks
    them toward 1, the wall)."""
    if n < 1:
        raise ValueError("geometric_fractions needs n >= 1, got %d" % n)
    if ratio <= 0:
        raise ValueError("geometric_fractions needs ratio > 0, got %g" % ratio)
    steps = ratio ** np.arange(n, dtype=float)
    return np.concatenate([[0.0], np.cumsum(steps) / steps.sum()])


def shell_fractions(shell: int | FloatArray, ratio: float = DEFAULT_RATIO) -> FloatArray:
    """Radial fractions of a shell, 0 on the core and 1 on the wall: an ``int`` count
    graded by :func:`geometric_fractions` with ``ratio``, or explicit fractions (prenek's
    negative-count "custom ratio")."""
    if isinstance(shell, (int, np.integer)):
        return geometric_fractions(int(shell), ratio)
    return validate_layers(shell)


def radii(radius: float, a_frac: float = A_FRAC) -> tuple[float, float, float]:
    """``(ra, rb, rc)``: the core corner radii on an axis, a face diagonal, the body
    diagonal."""
    if not 0.0 < a_frac < 0.5:
        raise ValueError("a_frac must be in (0, 0.5), got %g" % a_frac)
    a = a_frac * radius
    return radius - a, radius - 2 * B_FRAC * a + a, radius - 2 * C_FRAC * a + a


# --------------------------------------------------------------------------- helpers
def _xyz(*c: FloatArray) -> PointArray:
    """Stack 2 or 3 coordinate arrays (broadcast) into ``(..., 3)`` points."""
    arrs = list(np.broadcast_arrays(*[np.asarray(v, dtype=float) for v in c]))
    if len(arrs) == 2:
        arrs.append(np.zeros_like(arrs[0]))
    return np.stack(arrs, axis=-1)


def _quad_curve(p0: PointArray, pm: PointArray, p1: PointArray) -> Curve:
    """The quadratic through ``p0, pm, p1`` at parameters ``0, 1/2, 1``."""
    def f(t: FloatArray) -> PointArray:
        t = np.asarray(t, dtype=float)[..., None]
        return (2 * (t - 0.5) * (t - 1) * p0 - 4 * t * (t - 1) * pm + 2 * t * (t - 0.5) * p1)
    return f


def _line(p0: PointArray, p1: PointArray) -> Curve:
    def f(t: FloatArray) -> PointArray:
        t = np.asarray(t, dtype=float)[..., None]
        return (1 - t) * p0 + t * p1
    return f


def _arc(radius: float, th0: float, th1: float) -> Curve:
    """Circle of ``radius`` about the origin in the ``(x, y)`` plane, uniform in angle."""
    def f(t: FloatArray) -> PointArray:
        th = th0 + np.asarray(t, dtype=float) * (th1 - th0)
        return _xyz(radius * np.cos(th), radius * np.sin(th))
    return f


def _coons(bottom: Curve, top: Curve, left: Curve, right: Curve) -> Map2:
    """Coons patch of four edge curves: ``bottom(u)`` (v = 0), ``top(u)`` (v = 1),
    ``left(v)`` (u = 0), ``right(v)`` (u = 1)."""
    c00, c10, c01, c11 = bottom(0.0), bottom(1.0), top(0.0), top(1.0)

    def f(u: FloatArray, v: FloatArray) -> PointArray:
        u, v = np.broadcast_arrays(np.asarray(u, float), np.asarray(v, float))
        U, V = u[..., None], v[..., None]
        return ((1 - V) * bottom(u) + V * top(u) + (1 - U) * left(v) + U * right(v)
                - ((1 - U) * (1 - V) * c00 + U * (1 - V) * c10
                   + (1 - U) * V * c01 + U * V * c11))
    return f


def _edge_tfi(corner: Callable[[int, int, int], PointArray],
              edge: Callable[[int, int, int], Curve]) -> Map3:
    """Edge-based transfinite interpolation of a hex from its 12 edge curves (prenek's
    Gordon-Hall ``gh_face_extend`` from edges).  ``edge(axis, i, j)`` is the curve along
    ``axis`` at the other two (ordered) coordinates fixed to ``i, j`` in ``{0, 1}``."""
    E = {(ax, i, j): edge(ax, i, j) for ax in range(3) for i in (0, 1) for j in (0, 1)}
    C = {(i, j, k): corner(i, j, k) for i in (0, 1) for j in (0, 1) for k in (0, 1)}

    def f(u: FloatArray, v: FloatArray, w: FloatArray) -> PointArray:
        u, v, w = np.broadcast_arrays(*(np.asarray(a, float) for a in (u, v, w)))
        r = (u, v, w)
        phi = [((1 - x)[..., None], x[..., None]) for x in r]
        out = np.zeros(u.shape + (3,))
        for ax in range(3):
            o1, o2 = [b for b in range(3) if b != ax]
            for i in (0, 1):
                for j in (0, 1):
                    out += phi[o1][i] * phi[o2][j] * E[(ax, i, j)](r[ax])
        for (i, j, k), c in C.items():
            out -= 2 * phi[0][i] * phi[1][j] * phi[2][k] * c
        return out
    return f


def _unit(p: PointArray) -> PointArray:
    return p / np.linalg.norm(p, axis=-1, keepdims=True)


# --------------------------------------------------------------------------- 2-D quadrant
def quadrant_maps(radius: float, a_frac: float = A_FRAC) -> tuple[Map2, Map2, Map2]:
    """``(core, shell_x, shell_y)`` parent maps of the filled quadrant.  ``core(u, v)``:
    ``u`` along x, ``v`` along y.  Both shells: ``u`` radial (0 on the core, 1 on the
    circle), ``v`` counter-clockwise; ``shell_x`` spans 0..45 degrees, ``shell_y``
    45..90."""
    ra, rb, _ = radii(radius, a_frac)
    c00 = _xyz(0.0, 0.0)
    c10 = _xyz(ra, 0.0)
    c01 = _xyz(0.0, ra)
    c11 = _xyz(rb, rb) / np.sqrt(2.0)

    def midside(p: PointArray, q: PointArray) -> PointArray:
        m = 0.5 * (p + q)
        return W2 * ra * _unit(m) + (1 - W2) * m

    right = _quad_curve(c10, midside(c10, c11), c11)      # core edge x = ra side (u = 1)
    top = _quad_curve(c01, midside(c01, c11), c11)        # core edge y = ra side (v = 1)
    core = _coons(_line(c00, c10), top, _line(c00, c01), right)

    q = np.pi / 4
    shell_x = _coons(_line(c10, _xyz(radius, 0.0)),               # v = 0: theta = 0
                     _line(c11, radius * _unit(c11)),             # v = 1: theta = 45
                     right,                                       # u = 0: on the core
                     _arc(radius, 0.0, q))                        # u = 1: circle
    shell_y = _coons(_line(c11, radius * _unit(c11)),             # v = 0: theta = 45
                     _line(c01, _xyz(0.0, radius)),               # v = 1: theta = 90
                     lambda t: top(1 - np.asarray(t, float)),     # u = 0: on the core
                     _arc(radius, q, 2 * q))                      # u = 1: circle
    return core, shell_x, shell_y


# --------------------------------------------------------------------------- 3-D octant
def _sphere_patch(radius: float, corners: dict[tuple[int, int], PointArray]) -> Map2:
    """prenek's ``sphsrf_e`` face: great-circle edges at uniform angle between the four
    corner directions, interior points where the great circle through the two ``u``-edge
    points at ``v`` meets the one through the two ``v``-edge points at ``u``."""
    n = {k: _unit(c) for k, c in corners.items()}

    def slerp(p: PointArray, q: PointArray, t: FloatArray) -> PointArray:
        t = np.asarray(t, float)[..., None]
        th = np.arccos(np.clip(np.dot(p, q), -1.0, 1.0))
        b = _unit(np.cross(np.cross(p, q), p))
        return np.cos(t * th) * p + np.sin(t * th) * b

    def f(u: FloatArray, v: FloatArray) -> PointArray:
        u, v = np.broadcast_arrays(np.asarray(u, float), np.asarray(v, float))
        e_left = slerp(n[0, 0], n[0, 1], v)      # u = 0
        e_right = slerp(n[1, 0], n[1, 1], v)     # u = 1
        e_bot = slerp(n[0, 0], n[1, 0], u)       # v = 0
        e_top = slerp(n[0, 1], n[1, 1], u)       # v = 1
        p = np.cross(np.cross(e_left, e_right), np.cross(e_bot, e_top))
        # exact on the edges, where the two great circles coincide
        p = np.where(np.isclose(u, 0.0)[..., None], e_left, p)
        p = np.where(np.isclose(u, 1.0)[..., None], e_right, p)
        p = np.where(np.isclose(v, 0.0)[..., None], e_bot, p)
        p = np.where(np.isclose(v, 1.0)[..., None], e_top, p)
        mean = n[0, 0] + n[1, 0] + n[0, 1] + n[1, 1]
        p = p * np.sign(np.sum(p * mean, axis=-1, keepdims=True))
        return radius * _unit(p)
    return f


def octant_maps(radius: float, a_frac: float = A_FRAC) -> tuple[Map3, Map3, Map3, Map3]:
    """``(core, shell_z, shell_x, shell_y)`` parent maps of the filled octant.
    ``core(u, v, w)``: ``u, v, w`` along ``x, y, z``.  ``shell_z(u, v, t)`` sits on the
    core's ``z = top`` face (``u, v`` along that face's ``x, y``, ``t`` radial: 0 on the
    core, 1 on the sphere); ``shell_x`` / ``shell_y`` are it rotated by 120 / 240 degrees
    about ``(1, 1, 1)`` (the cyclic permutations ``(x, y, z) -> (y, z, x) -> (z, x, y)``),
    so they sit on the core's ``x`` / ``y`` faces with the same parametrization."""
    ra, rb, rc = radii(radius, a_frac)
    rk = (0.0, ra, rb, rc)
    w3 = 1.0 + W3_GAIN * a_frac

    def corner(i: int, j: int, k: int) -> PointArray:
        d = np.array([i, j, k], dtype=float)
        nz = int(d.sum())
        return d * (rk[nz] / np.sqrt(nz)) if nz else np.zeros(3)

    def core_edge(ax: int, i: int, j: int) -> Curve:
        ij = [i, j]
        lo = [0, 0, 0]
        hi = [0, 0, 0]
        o = [b for b in range(3) if b != ax]
        for b, val in zip(o, ij):
            lo[b] = hi[b] = val
        hi[ax] = 1
        p, q = corner(*lo), corner(*hi)
        if not any(lo):                        # touches the origin: straight
            return _line(p, q)
        return _quad_curve(p, w3 * 0.5 * (p + q), q)

    core = _edge_tfi(corner, core_edge)

    # shell on the core's top face: bottom = core face w = 1, top = sphere patch
    def bot_corner(i: int, j: int) -> PointArray:
        return corner(i, j, 1)

    def top_corner(i: int, j: int) -> PointArray:
        return radius * _unit(np.array([i, j, 1.0]))

    def shell_corner(i: int, j: int, k: int) -> PointArray:
        return top_corner(i, j) if k else bot_corner(i, j)

    def shell_edge(ax: int, i: int, j: int) -> Curve:
        if ax == 2:                              # radial edges: straight
            return _line(bot_corner(i, j), top_corner(i, j))
        if j == 0:                               # bottom edges: the core's top edges
            return core_edge(ax, i, 1)
        # top edges: straight chords (prenek blanks them; the sphere comes from sphsrf_e)
        p = shell_corner(*([0, i, 1] if ax == 0 else [i, 0, 1]))
        q = shell_corner(*([1, i, 1] if ax == 0 else [i, 1, 1]))
        return _line(p, q)

    base = _edge_tfi(shell_corner, shell_edge)
    sph = _sphere_patch(radius, {(i, j): top_corner(i, j) for i in (0, 1) for j in (0, 1)})

    def shell_z(u: FloatArray, v: FloatArray, t: FloatArray) -> PointArray:
        u, v, t = np.broadcast_arrays(*(np.asarray(a, float) for a in (u, v, t)))
        return base(u, v, t) + t[..., None] * (sph(u, v) - base(u, v, np.ones_like(t)))

    def cyc(m: Map3, k: int) -> Map3:
        def f(u: FloatArray, v: FloatArray, t: FloatArray) -> PointArray:
            p = m(u, v, t)
            return np.roll(p, k, axis=-1)        # (x, y, z) -> (z, x, y) for k = 1
        return f

    return core, shell_z, cyc(shell_z, 1), cyc(shell_z, 2)
