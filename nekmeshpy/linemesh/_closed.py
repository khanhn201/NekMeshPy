"""Closed-loop :class:`~nekmeshpy.LineMesh` factories: parametric shapes whose
boundary closes on itself (``circle`` / ``rectangle``).

These are plain free functions returning a ``LineMesh``; ``linemesh/__init__.py``
binds each entry of ``FACTORIES`` onto the class, so callers use
``LineMesh.circle(...)`` while ``linemesh.py`` stays a pure container (adding a shape
here needs no edit there).  Internal toolkit code calls the free functions directly.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

import numpy as np

from .._typing import Point, PointArray, StrArray, Vec3
from ._assemble import loft
from ._plane import _arc_interior, _arc_points, _in_plane_axes

if TYPE_CHECKING:

    from .linemesh import LineMesh


def circle(radius: float, n: int, *,
           center: Point = (0.0, 0.0, 0.0),
           normal: Vec3 = (0.0, 0.0, 1.0),
           start_theta: float = 0.0,
           element_tags: StrArray | Sequence[str] | None = None,
           order: int = 1) -> LineMesh:
    """A closed loop of ``n`` points evenly spaced on a circle of ``radius``
    about ``center`` in the plane with the given ``normal`` (default ``+z``).
    Point ``k`` sits at angle ``2*pi*k/n + start_theta`` from the in-plane
    ``e1`` axis, so ``start_theta`` rotates the whole loop -- e.g. to align its
    index 0 with a :meth:`rectangle` far-field box's lower-left corner before an
    index-paired :meth:`QuadMesh.annulus <nekmeshpy.quadmesh.QuadMesh.annulus>`.
    ``element_tags`` tags the loop's line elements at construction.

    ``order`` (default 1 = linear) sets the polynomial order: at ``order > 1``
    each arc element carries ``order+1`` GLL nodes placed on the **true circle**
    (not the chord) -- the two endpoints are the corners in ``points`` and the
    ``order-1`` nodes strictly between them are built here, still on the exact
    circle, as the element's private ``interior``, so a high-order ``vtu`` export
    renders the exact arc.

    The open sibling is :meth:`LineMesh.arc <nekmeshpy.linemesh.LineMesh.arc>`; the
    two share the node
    placement (``_plane._arc_points`` / ``_arc_interior``) but not the angle sampling:
    a full turn's step is exactly ``2*pi/n`` here, whereas ``arc`` must form
    ``(end_theta - start_theta)/n``, so ``circle`` keeps its own ``linspace`` rather
    than delegating (see ``arc``'s docstring)."""
    th = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False) + float(start_theta)
    c: Point = np.asarray(center, dtype=float).ravel()
    e1, e2 = _in_plane_axes(np.asarray(normal, dtype=float))
    pts: PointArray = _arc_points(radius, c, e1, e2, th)
    if order == 1:
        return loft(pts, loop=True, element_tags=element_tags)
    # element l spans angle th[l] .. th[l] + dth; place the interior GLL nodes on
    # the arc (the two endpoint nodes are already the loop's corner points), so the
    # explicit ``interior`` overrides ``loft``'s default straight chord blend.
    interior: PointArray = _arc_interior(
        radius, c, e1, e2, th, 2.0 * np.pi / n, order)             # (n, order-1, 3)
    return loft(pts, loop=True, interior=interior,
                         element_tags=element_tags, order=order)


def rectangle(width: float, height: float, n: int, *,
              center: Point = (0.0, 0.0, 0.0),
              normal: Vec3 = (0.0, 0.0, 1.0),
              side_tags: Mapping[str, str] | None = None,
              order: int = 1) -> LineMesh:
    """A closed rectangle loop of the given ``width`` x ``height`` about
    ``center`` in the plane with the given ``normal`` (default ``+z``),
    discretized into ``n`` line elements (``n`` must be a positive multiple of
    4): ``n // 4`` evenly spaced per side, running CCW from the lower-left corner
    (bottom / right / top / left), corners always landing on a point.

    With ``n`` points it feeds
    :meth:`QuadMesh.annulus <nekmeshpy.quadmesh.QuadMesh.annulus>` directly as
    the outer far-field loop against a ``circle(radius, n)`` body -- rotate the
    circle with ``start_theta`` so its index 0 meets the lower-left corner
    (``atan2(-height, -width)``) and the two loops pair index-for-index (the
    radial spokes are not straight, but the mesh conforms).

    ``side_tags`` (keyed ``bottom`` / ``right`` / ``top`` / ``left``) names each
    side's line elements; an absent key leaves that side untagged and
    ``side_tags=None`` leaves the whole loop untagged.  The keys -- rather than a
    positional 4-sequence -- are what make this spelling identical to its one-rung-up
    twin :meth:`QuadMesh.rectangle <nekmeshpy.quadmesh.QuadMesh.rectangle>`, which
    takes the same keyword with the same four names; an unrecognized key is a loud
    ``ValueError`` because a silent typo would otherwise just lose a wall.

    ``order`` (default 1 = linear) sets the polynomial order: at ``order > 1``
    each element carries ``order+1`` GLL nodes on its straight side -- the two
    endpoints are the corners in ``points`` and the ``order-1`` nodes strictly
    between them are the straight GLL blend
    :meth:`LineMesh.loft <nekmeshpy.linemesh.LineMesh.loft>` places by default
    (read by high-order ``vtu`` export)."""
    ni = int(n)
    if ni < 4 or ni % 4 != 0:
        raise ValueError("rectangle n must be a positive multiple of 4, "
                         "got %d" % ni)
    m = ni // 4
    c: Point = np.asarray(center, dtype=float).ravel()
    e1, e2 = _in_plane_axes(np.asarray(normal, dtype=float))
    hw, hh = width / 2.0, height / 2.0
    bl = c - hw * e1 - hh * e2
    br = c + hw * e1 - hh * e2
    tr = c + hw * e1 + hh * e2
    tl = c - hw * e1 + hh * e2
    f = np.linspace(0.0, 1.0, m, endpoint=False)[:, None]  # corner..next, open

    def _side(p: Point, q: Point) -> PointArray:
        return p + f * (q - p)
    pts = np.concatenate([_side(bl, br), _side(br, tr),
                          _side(tr, tl), _side(tl, bl)])
    tags: list[str] | None = None
    if side_tags is not None:
        sides = ("bottom", "right", "top", "left")   # the loop's CCW traversal order
        for key in side_tags:
            if key not in sides:
                raise ValueError(
                    "rectangle side_tags key must be one of "
                    "bottom/right/top/left, got %r" % key)
        tags = [t for side in sides for t in [side_tags.get(side, "")] * m]
    # every side is straight, so ``loft``'s default straight GLL interior is exact
    return loft(pts, loop=True, element_tags=tags, order=order)
