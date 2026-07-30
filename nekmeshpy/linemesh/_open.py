"""Open :class:`~nekmeshpy.LineMesh` factories: curves with free ends that do not
close on themselves (``line``).

These are plain free functions returning a ``LineMesh``; ``linemesh/__init__.py``
binds each entry of ``FACTORIES`` onto the class, so callers use ``LineMesh.line(...)``
while ``linemesh.py`` stays a pure container.  Internal toolkit code calls the free
functions directly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .._typing import FloatArray, Point
from ..model.interp import straight_edges

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
    segment (the ``curved`` block, read by high-order ``vtu`` export)."""
    from .linemesh import LineMesh
    frac = np.atleast_1d(np.asarray(fractions, dtype=float))
    s: Point = np.asarray(start, dtype=float).ravel()
    e: Point = np.asarray(end, dtype=float).ravel()
    pts = s + frac[:, None] * (e - s)
    tags = [element_tag] * (pts.shape[0] - 1) if element_tag else None
    lm = LineMesh.open(pts, element_tags=tags)
    if order == 1:
        return lm
    curved = straight_edges(lm.points[lm.lines[:, 0]],
                            lm.points[lm.lines[:, 1]], order)
    return LineMesh(lm.points, lm.lines, lm.element_tags, lm.boundaries,
                    lm.boundary_tags, closed=False, order=order, curved=curved)


#: Open-curve factories bound onto ``LineMesh`` by ``linemesh/__init__.py``.
FACTORIES: dict[str, Callable[..., LineMesh]] = {
    "line": line,
}
