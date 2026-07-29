"""Closed-loop :class:`~nekmeshpy.LineMesh` factories: parametric shapes whose
boundary closes on itself (``circle`` / ``rectangle``).

These are plain free functions returning a ``LineMesh``; ``linemesh/__init__.py``
binds each entry of ``FACTORIES`` onto the class, so callers use
``LineMesh.circle(...)`` while ``linemesh.py`` stays a pure container (adding a shape
here needs no edit there).  Internal toolkit code calls the free functions directly.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np

from .._typing import Point, PointArray, StrArray, Vec3
from ._plane import _in_plane_axes

if TYPE_CHECKING:
    from collections.abc import Callable

    from .linemesh import LineMesh


def circle(radius: float, n: int, *,
           center: Point = (0.0, 0.0, 0.0),
           normal: Vec3 = (0.0, 0.0, 1.0),
           start_theta: float = 0.0,
           element_tags: StrArray | Sequence[str] | None = None) -> LineMesh:
    """A closed loop of ``n`` points evenly spaced on a circle of ``radius``
    about ``center`` in the plane with the given ``normal`` (default ``+z``).
    Point ``k`` sits at angle ``2*pi*k/n + start_theta`` from the in-plane
    ``e1`` axis, so ``start_theta`` rotates the whole loop -- e.g. to align its
    index 0 with a :meth:`rectangle` far-field box's lower-left corner before an
    index-paired :meth:`QuadMesh.annulus <nekmeshpy.quadmesh.QuadMesh.annulus>`.
    ``element_tags`` tags the loop's line elements at construction."""
    from .linemesh import LineMesh
    th = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False) + float(start_theta)
    c: Point = np.asarray(center, dtype=float).ravel()
    e1, e2 = _in_plane_axes(np.asarray(normal, dtype=float))
    local = radius * np.cos(th)[:, None] * e1 + radius * np.sin(th)[:, None] * e2
    return LineMesh.loop(c + local, element_tags)


def rectangle(width: float, height: float, n: int, *,
              center: Point = (0.0, 0.0, 0.0),
              normal: Vec3 = (0.0, 0.0, 1.0),
              side_tags: Sequence[str] | None = None) -> LineMesh:
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

    ``side_tags`` (length-4 ``[bottom, right, top, left]``) names each side's
    line elements; ``side_tags=None`` leaves the loop untagged."""
    from .linemesh import LineMesh
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
        st = list(side_tags)
        if len(st) != 4:
            raise ValueError("rectangle side_tags must be 4 names "
                             "[bottom, right, top, left]")
        tags = [st[0]] * m + [st[1]] * m + [st[2]] * m + [st[3]] * m
    return LineMesh.loop(pts, tags)


#: Closed-loop shape factories bound onto ``LineMesh`` by ``linemesh/__init__.py``.
FACTORIES: dict[str, Callable[..., LineMesh]] = {
    "circle": circle,
    "rectangle": rectangle,
}
