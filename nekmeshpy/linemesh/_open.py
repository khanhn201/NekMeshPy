"""Open :class:`~nekmeshpy.LineMesh` factories: curves with free ends that do not
close on themselves (``line`` / ``arc`` / ``curve``).

These are plain free functions returning a ``LineMesh``; ``linemesh/__init__.py``
binds each entry of ``FACTORIES`` onto the class, so callers use ``LineMesh.line(...)``
while ``linemesh.py`` stays a pure container.  Internal toolkit code calls the free
functions directly.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np

from .._typing import FloatArray, IntArray, Point, PointArray, StrArray, Vec3
from ..model.fields import gll_nodes
from ._assemble import loft
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


def _refined_lattice(fractions: FloatArray, order: int) -> FloatArray:
    """The ``n*order + 1`` parameter positions of **every** node of the order-``order``
    chain graded by ``fractions`` (``n = len(fractions) - 1`` elements): element ``i``'s
    node ``a`` sits at ``fr[i] + g[a]*(fr[i+1] - fr[i])`` for the GLL nodes ``g`` on
    ``[0, 1]``, and the chain ends at ``fr[-1]``.

    This is the 1-D twin of :func:`~nekmeshpy.quadmesh._open._refined_params`; the
    grading rides in ``fractions`` rather than being assumed uniform, so each element's
    private interior lands inside that element's own span.  At ``order == 1``
    (``g = [0, 1]``) it is exactly ``fractions``, so the order-1 placement falls out by
    construction rather than by a branch."""
    g: FloatArray = gll_nodes(order)
    fr = fractions
    u: FloatArray = (fr[:-1, None]
                     + g[None, :order] * np.diff(fr)[:, None]).ravel()
    return np.concatenate([u, fr[-1:]])


def _eval_curve(f: Callable[[FloatArray], PointArray], t: FloatArray) -> PointArray:
    """``f(t)`` as a validated ``(len(t), 3)`` array."""
    P: PointArray = np.asarray(f(t), dtype=float)
    if P.shape != (t.shape[0], 3):
        raise ValueError(
            "curve callable must return (len(t), 3) points; got shape %r for %d "
            "parameters" % (P.shape, t.shape[0]))
    return P


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
    :func:`curve` as its ``fractions``, with no further scaling
    (``curve(f, arclength_fractions(f, n, t_range=...), order=N)``).

    ``t_range`` is the parameter interval to invert over -- unlike :func:`curve`, this
    helper genuinely needs a domain, because the chord table is built by sampling it
    densely.  A descending range needs no special handling: the returned values simply
    run from ``t_range[0]`` down to ``t_range[1]``, which meshes the curve backwards.

    The inversion goes through a cumulative **chord**-length table of ``samples`` dense
    evaluations of ``f``, so only *where along* the curve the nodes end up inherits that
    table's discretization error.  Every node of the resulting mesh still lies on the
    curve to machine precision, because :func:`curve` places it by evaluating ``f`` and
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


def curve(f: Callable[[FloatArray], PointArray], fractions: float | FloatArray, *,
          order: int = 1,
          element_tags: StrArray | Sequence[str] | None = None) -> LineMesh:
    """An **open** curve meshed on its own analytic parametrization, with **every**
    node -- corners *and* the private high-order ``interior`` -- evaluated by calling
    ``f``, so nothing is ever placed on a chord.

    ``f`` maps a ``(K,)`` parameter array to ``(K, 3)`` points and is called **once**
    with the whole node lattice (vectorize it; ``np.column_stack`` of the three
    component expressions is the usual shape).

    ``fractions`` are the **parameter values themselves**, passed to ``f`` with no
    normalization and no remapping: node ``k`` is ``f(fractions[k])``, and there are
    ``len(fractions) - 1`` line elements.  The caller states the domain by choosing the
    values -- for an ``f`` written on ``[0, 1]`` they are exactly the normalized
    fractions the sibling
    :meth:`LineMesh.line <nekmeshpy.linemesh.LineMesh.line>` takes, and an ``f`` written
    on any other interval is sampled in its own units
    (``np.linspace(0.0, np.pi, n + 1)`` for a uniform chain over ``[0, pi]``).  A
    descending sequence runs the curve backwards; nothing here requires ascending order.

    The values grade the nodes in **parameter** space; for nodes spaced evenly by **arc
    length** pass :meth:`LineMesh.arclength_fractions
    <nekmeshpy.linemesh.LineMesh.arclength_fractions>`, whose chord-length table
    perturbs only *where along* the curve the nodes sit -- every node still lies on the
    curve to machine precision, because it is placed by evaluating ``f`` and never by
    interpolating the table.  At ``order > 1`` the grading is honored **per element**:
    element ``i``'s private ``interior`` rides the GLL nodes of its own
    ``fractions[i] .. fractions[i+1]`` span.

    This is the general sibling of
    :meth:`LineMesh.arc <nekmeshpy.linemesh.LineMesh.arc>`, which is the special case
    ``f = circle`` (kept separate because it can place its nodes without an inversion
    and to the last ulp).  Reach for ``curve`` whenever a curve has a closed form that
    is not a circular arc -- an ellipse, a helix, a cylinder-cylinder intersection --
    instead of sampling it into an array and calling
    :meth:`LineMesh.loft <nekmeshpy.linemesh.LineMesh.loft>`, which can only subdivide
    straight between the samples and therefore loses the curve at ``order > 1``.  For a
    curve with **no** closed form (a scanned polyline) there is nothing to evaluate;
    resample it with ``trimesh.ops.resample_polyline`` and accept the chord.

    ``element_tags`` (length ``len(fractions) - 1``) tags the line elements at
    construction; ``order`` (default 1 = linear) sets the polynomial order.  The result
    is always an open chain -- for a closed parametric loop, mesh it here and weld the
    ends with :meth:`LineMesh.merge <nekmeshpy.linemesh.LineMesh.merge>`."""
    fr: FloatArray = np.atleast_1d(np.asarray(fractions, dtype=float))
    ni = fr.shape[0] - 1
    if ni < 1:
        raise ValueError(
            "curve needs at least 2 fractions (one element), got %d" % fr.shape[0])

    # every node of the chain, corners and interiors alike, as one parameter array --
    # so the interiors ride the true curve instead of ``loft``'s straight chord blend.
    t: FloatArray = _refined_lattice(fr, order)
    P: PointArray = _eval_curve(f, t)

    if order == 1:
        return loft(P, element_tags=element_tags)
    slot: IntArray = (np.arange(ni)[:, None] * order
                      + np.arange(1, order)[None, :])        # (n, order-1)
    return loft(P[::order], interior=P[slot], element_tags=element_tags, order=order)


#: Open-curve factories bound onto ``LineMesh`` by ``linemesh/__init__.py``.
FACTORIES: dict[str, Callable[..., LineMesh]] = {
    "line": line,
    "arc": arc,
    "curve": curve,
}

#: Open-curve helpers bound onto ``LineMesh`` as ``staticmethod``s.  These answer a
#: question *about* a factory's input contract and return plain arrays rather than a
#: mesh, which is what keeps them out of ``FACTORIES``.
HELPERS: dict[str, Callable[..., FloatArray]] = {
    "arclength_fractions": arclength_fractions,
}
