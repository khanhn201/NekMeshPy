"""Open :class:`~nekmeshpy.LineMesh` factories: curves with free ends that do not
close on themselves (``line`` / ``arc``).

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
    from .linemesh import LineMesh
    frac = np.atleast_1d(np.asarray(fractions, dtype=float))
    s: Point = np.asarray(start, dtype=float).ravel()
    e: Point = np.asarray(end, dtype=float).ravel()
    pts = s + frac[:, None] * (e - s)
    tags = [element_tag] * (pts.shape[0] - 1) if element_tag else None
    # the segment is straight, so ``loft``'s default straight GLL interior is exact
    return LineMesh.loft(pts, loop=False, element_tags=tags, order=order)


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
    points and calling :meth:`LineMesh.open <nekmeshpy.linemesh.LineMesh.open>`,
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
    from .linemesh import LineMesh
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
        return LineMesh.open(pts, element_tags)
    # element l spans th[l] .. th[l] + dth; its private interior rides the exact arc,
    # overriding ``loft``'s default straight chord blend.
    interior: PointArray = _arc_interior(
        radius, c, e1, e2, th[:-1], (e - s) / ni, order)     # (n, order-1, 3)
    return LineMesh.open(pts, element_tags, order=order, interior=interior)


#: Open-curve factories bound onto ``LineMesh`` by ``linemesh/__init__.py``.
FACTORIES: dict[str, Callable[..., LineMesh]] = {
    "line": line,
    "arc": arc,
}
