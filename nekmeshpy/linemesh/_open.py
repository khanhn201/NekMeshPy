"""Open :class:`~nekmeshpy.LineMesh` factories: curves with free ends that do not
close on themselves (``line`` / ``arc``), plus the ``arclength_fractions`` /
``sweep_fractions`` sampling helpers.

The general parametrized curve is **not** here: it authors its own connectivity and
takes the same ``loop`` flag as the sweep primitive, so it lives beside it as
:func:`~nekmeshpy.linemesh._assemble.loft_fn`.

These are plain free functions returning a ``LineMesh``; ``linemesh/__init__.py``
binds each entry of ``FACTORIES`` onto the class, so callers use ``LineMesh.line(...)``
while ``linemesh.py`` stays a pure container.  Internal toolkit code calls the free
functions directly.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np

from .._typing import FloatArray, Point, PointArray, StrArray, Vec3
from ._assemble import _eval_curve, loft
from ._plane import _arc_interior, _arc_points, _in_plane_axes

if TYPE_CHECKING:
    from collections.abc import Callable

    from .linemesh import LineMesh


def line(start: Point, end: Point, fractions: float | FloatArray, *,
         element_tag: str = "", order: int = 1) -> LineMesh:
    """A straight open line from ``start`` to ``end`` sampled at normalized
    arc-length ``fractions`` in ``[0, 1]`` (``0`` = start, ``1`` = end): the
    graded-edge sibling of ``circle``/``rectangle``. The points are placed
    exactly at ``start + f*(end - start)`` -- no resampling. ``element_tag``
    names every resulting line element (e.g. to tag a structured edge as one
    wall).

    ``order`` (default 1 = linear) sets the polynomial order: at ``order > 1``
    each line element carries ``order+1`` GLL nodes placed on the straight
    segment -- the two endpoints are the corners in ``points``, and the
    ``order-1`` nodes strictly between them are the straight GLL blend
    :meth:`LineMesh.loft <nekmeshpy.linemesh.LineMesh.loft>` places by default
    (read by high-order ``vtu`` export)."""
    frac = np.atleast_1d(np.asarray(fractions, dtype=float))
    s: Point = np.asarray(start, dtype=float).ravel()
    e: Point = np.asarray(end, dtype=float).ravel()
    pts = s + frac[:, None] * (e - s)
    tags = [element_tag] * (pts.shape[0] - 1) if element_tag else None
    # the segment is straight, so ``loft``'s default straight GLL interior is exact
    return loft(pts, loop=False, element_tags=tags, order=order)


def arc(radius: float, n: int, *,
        center: Point = (0.0, 0.0, 0.0),
        normal: Vec3 = (0.0, 0.0, 1.0),
        start_theta: float = 0.0,
        end_theta: float = np.pi,
        element_tags: StrArray | Sequence[str] | None = None,
        order: int = 1) -> LineMesh:
    """An **open** circular arc: ``n`` line elements over ``n+1`` points evenly
    spaced in angle from ``start_theta`` to ``end_theta`` on the circle of
    ``radius`` about ``center``, in the plane with the given ``normal`` (default
    ``+z``).  Point ``k`` sits at ``start_theta + k*(end_theta - start_theta)/n``
    measured from the in-plane ``e1`` axis, so ``end_theta < start_theta`` runs the
    arc clockwise.  ``element_tags`` (length ``n``) tags its line elements at
    construction, e.g. to name the whole arc ``wall`` for a section factory.

    This is the open sibling of
    :meth:`LineMesh.circle <nekmeshpy.linemesh.LineMesh.circle>` -- the analytic
    curve to hand to :meth:`QuadMesh.structured <nekmeshpy.quadmesh.QuadMesh.structured>` (or to
    weld into a composite edge with
    :meth:`LineMesh.merge <nekmeshpy.linemesh.LineMesh.merge>`) instead of sampling
    points and calling :meth:`LineMesh.loft <nekmeshpy.linemesh.LineMesh.loft>`,
    which can only subdivide straight between the samples.

    ``order`` (default 1 = linear) sets the polynomial order: at ``order > 1`` each
    element carries ``order+1`` GLL nodes placed on the **true arc** (not the
    chord) -- the two endpoints are the corners in ``points`` and the ``order-1``
    nodes strictly between them are the element's private ``interior``, so a
    high-order ``vtu`` export renders the exact arc.

    ``circle`` does **not** delegate here: over a full turn the angular step is
    exactly ``2*pi/n``, while this factory must form ``(end_theta - start_theta)/n``
    -- and ``(s + 2*pi) - s`` is not ``2*pi`` in floating point for a general ``s``,
    so delegating would move ``circle``'s nodes by a ulp.  The two share the node
    placement instead (``_plane._arc_points`` / ``_arc_interior``)."""
    ni = int(n)
    if ni < 1:
        raise ValueError("arc needs n >= 1 elements, got %d" % ni)
    s, e = float(start_theta), float(end_theta)
    if s == e:
        raise ValueError("arc needs start_theta != end_theta (got %g twice)" % s)
    c: Point = np.asarray(center, dtype=float).ravel()
    e1, e2 = _in_plane_axes(np.asarray(normal, dtype=float))
    th: FloatArray = np.linspace(s, e, ni + 1)
    pts: PointArray = _arc_points(radius, c, e1, e2, th)
    if order == 1:
        return loft(pts, element_tags=element_tags)
    # element l spans th[l] .. th[l] + dth; its private interior rides the exact arc,
    # overriding ``loft``'s default straight chord blend.
    interior: PointArray = _arc_interior(
        radius, c, e1, e2, th[:-1], (e - s) / ni, order)     # (n, order-1, 3)
    return loft(pts, interior=interior, element_tags=element_tags,
                         order=order)


def _arclength_params(f: Callable[[FloatArray], PointArray], t0: float, t1: float,
                      samples: int) -> tuple[FloatArray, FloatArray]:
    """``(t_dense, s_dense)`` -- ``samples`` parameters spanning ``[t0, t1]`` and their
    normalized cumulative **chord** length in ``[0, 1]``, the table used to invert
    arc length into the curve parameter."""
    td: FloatArray = np.linspace(t0, t1, samples)
    Pd: PointArray = _eval_curve(f, td)
    s: FloatArray = np.concatenate(
        [[0.0], np.cumsum(np.linalg.norm(np.diff(Pd, axis=0), axis=1))])
    if s[-1] <= 0.0:
        raise ValueError(
            "curve has zero length over t_range -- cannot space by arc length")
    return td, s / s[-1]


def arclength_fractions(f: Callable[[FloatArray], PointArray], n: int, *,
                        t_range: tuple[float, float] = (0.0, 1.0),
                        samples: int = 1001) -> FloatArray:
    """The ``(n+1,)`` **parameter values** spanning ``t_range`` -- from ``t_range[0]``
    to ``t_range[1]`` -- at which ``f`` must be evaluated for the resulting ``n+1``
    points to be evenly spaced by **arc length**: hand the result straight to
    :meth:`LineMesh.loft_fn <nekmeshpy.linemesh.LineMesh.loft_fn>` as its
    ``fractions``, with no further scaling
    (``loft_fn(f, arclength_fractions(f, n, t_range=...), order=N)``).

    ``t_range`` is the parameter interval to invert over -- unlike ``loft_fn``, this
    helper genuinely needs a domain, because the chord table is built by sampling it
    densely.  A descending range needs no special handling: the returned values simply
    run from ``t_range[0]`` down to ``t_range[1]``, which meshes the curve backwards.

    The inversion goes through a cumulative **chord**-length table of ``samples`` dense
    evaluations of ``f``, so only *where along* the curve the nodes end up inherits that
    table's discretization error.  Every node of the resulting mesh still lies on the
    curve to machine precision, because ``loft_fn`` places it by evaluating ``f`` and
    never by interpolating this table -- raise ``samples`` for a more even spacing, not
    for a more accurate curve."""
    ni = int(n)
    if ni < 1:
        raise ValueError("arclength_fractions needs n >= 1 elements, got %d" % ni)
    if samples < 2:
        raise ValueError("arclength_fractions needs samples >= 2, got %d" % samples)
    t0, t1 = float(t_range[0]), float(t_range[1])
    if t0 == t1:
        raise ValueError(
            "arclength_fractions needs t_range endpoints to differ (got %g twice)" % t0)
    td, s = _arclength_params(f, t0, t1, samples)
    t: FloatArray = np.interp(np.linspace(0.0, 1.0, ni + 1), s, td)
    return t


def sweep_fractions(breaks: FloatArray | Sequence[float], total_length: float,
                    target: float) -> FloatArray:
    """Normalized sweep stations in ``[0, 1]`` that put a node **exactly on every
    junction** of a piecewise path, subdividing each piece at roughly ``target``.

    ``breaks`` are the cumulative arc-length positions (in the same units as
    ``total_length``, strictly ascending, strictly inside ``(0, total_length)``) of the
    path's interior junctions -- for a turtle-walked centerline, ``cumsum(seg_len)``
    with its first and last entries dropped.  ``target`` is the desired element length
    along the sweep.

    Each interval between consecutive breaks (and the two end intervals against ``0``
    and ``total_length``) is split into ``max(1, round(interval / target))`` equal
    steps *on its own*, so the breaks reappear in the output bit-for-bit rather than
    being approached by a global ``linspace``.  That is the whole point: a path's
    curvature is piecewise constant and **jumps** at a junction, so an element that
    straddled one would be fitted across two different geometries -- visible as a kink
    in the wall of a swept bend.  The result is strictly ascending, opens with ``0.0``,
    contains every ``break / total_length``, and closes with ``1.0``.

    Hand it straight to the ``fractions`` of
    :meth:`HexMesh.sweep <nekmeshpy.hexmesh.HexMesh.sweep>` /
    :meth:`QuadMesh.sweep <nekmeshpy.quadmesh.QuadMesh.sweep>` (or of
    :meth:`LineMesh.loft_fn <nekmeshpy.linemesh.LineMesh.loft_fn>`) whose path is
    parametrized by normalized arc length.  Like
    :func:`arclength_fractions` it is a ``HELPERS`` entry, not a factory: it answers a
    question about a sweep's input contract and returns a plain array, since the sweep
    itself meshes exactly at the stations given."""
    L = float(total_length)
    if not L > 0.0:
        raise ValueError("sweep_fractions needs total_length > 0, got %g" % L)
    tgt = float(target)
    if not tgt > 0.0:
        raise ValueError("sweep_fractions needs target > 0, got %g" % tgt)
    br: FloatArray = np.asarray(breaks, dtype=float).ravel()
    if br.size and (br[0] <= 0.0 or br[-1] >= L):
        raise ValueError(
            "sweep_fractions breaks must lie strictly inside (0, %g) -- they are the "
            "path's *interior* junctions, and 0 / total_length are always stations "
            "anyway (got %g .. %g)" % (L, br[0], br[-1]))
    if br.size > 1 and not np.all(np.diff(br) > 0.0):
        raise ValueError(
            "sweep_fractions breaks must be strictly ascending cumulative arc lengths")
    s: FloatArray = br / L
    pieces: list[FloatArray] = []
    for a, b in zip(np.concatenate([[0.0], s]), np.concatenate([s, [1.0]])):
        # round-to-nearest on the piece's own length, floored at one element: a piece
        # shorter than ``target`` still gets an element rather than vanishing.
        n = max(1, int(round((b - a) * L / tgt)))
        pieces.append(np.linspace(a, b, n + 1)[:-1])   # drop the shared end station
    out: FloatArray = np.concatenate(pieces + [np.array([1.0])])
    return out
