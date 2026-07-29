"""Shared internals for the :class:`~nekmeshpy.QuadMesh` factory functions.

``_apply_smoothing`` and ``_check_boundary`` are used by both the core container
(``quadmesh.py``) and the split-out factory files (``_open.py``); they live here so
those files can share them without an import cycle.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .._typing import PointArray
from ..linemesh import LineMesh

if TYPE_CHECKING:
    from .quadmesh import QuadMesh


def _apply_smoothing(qm: QuadMesh, smoothing_method: str | None) -> QuadMesh:
    """Reposition ``qm``'s interior points in place (``None`` = no smoothing)."""
    if smoothing_method is not None:
        from . import smoothing
        smoothing.set_section_smoothing(qm, smoothing_method)
    return qm


def _check_boundary(obj: LineMesh, name: str,
                    closed: bool, min_pts: int) -> PointArray:
    """Validate a ``LineMesh`` factory argument (open/closed topology, minimum
    point count, finite coordinates), returning its ``(N,3)`` points."""
    if not isinstance(obj, LineMesh):
        raise TypeError("%s must be a LineMesh, got %s"
                        % (name, type(obj).__name__))
    if obj.is_closed != closed:
        raise TypeError("%s must be a %s LineMesh"
                        % (name, "closed" if closed else "open"))
    pts = obj.points
    if pts.shape[0] < min_pts:
        raise ValueError("%s needs at least %d points, got %d"
                         % (name, min_pts, pts.shape[0]))
    if not np.all(np.isfinite(pts)):
        raise ValueError("%s has non-finite coordinates" % name)
    return pts
