"""0-D mesh container: the ``(N,3)`` point array the whole ladder is built on."""

from __future__ import annotations

import numpy as np

from .._typing import PointArray
from ..core.tags import ElementTags


class PointMesh:
    """The ladder's bottom rung: an ``(N,3)`` point array and the tags on those points.

    A point has no interior, no orientation and no connectivity -- its "incidence" is
    the identity, so element ``i`` *is* point ``i``. That leaves only the two fields
    every rung has: the coordinates, and an :class:`ElementTags
    <nekmeshpy.core.tags.ElementTags>` naming whichever of them are named.

    It exists so the rule that holds between every other pair of rungs -- **a mesh's
    side tags are its rung-below's element tags** -- holds at the bottom too, rather
    than the line rung carrying a table of its own shape. Almost nothing constructs one
    directly: :class:`LineMesh <nekmeshpy.linemesh.linemesh.LineMesh>` promotes a bare ``(N,3)``
    array into an untagged one, which is what a factory building fresh geometry wants.
    Reach for it when you need the tags to survive -- a transform rebuilding a mesh
    around moved coordinates, or a point group named before there is a line to hang it
    on. There are deliberately no factories and no shape model here: a point cloud is
    not something this toolkit meshes, only something the rungs above index into.
    """

    def __init__(
        self,
        points: PointArray,
        element_tags: ElementTags | None = None,
    ) -> None:
        """Construct from ``points`` ``(N,3)`` (must be 3-D) and an optional
        :class:`ElementTags <nekmeshpy.core.tags.ElementTags>` over point ids."""
        self.points: PointArray = np.asarray(points, dtype=float)
        if self.points.ndim != 2 or self.points.shape[1] != 3:
            raise ValueError(
                "PointMesh: points must be (N,3) 3-D coordinates; got %s -- add a z "
                "column (all geometry lives in 3-D)" % (self.points.shape,))
        self.element_tags = ElementTags.empty() if element_tags is None else element_tags
        self.element_tags.check_within(self.points.shape[0])

    def __repr__(self) -> str:
        from ..linemesh.linemesh import _repr_tags
        try:
            return ("<PointMesh %d points, element_tags=%s>"
                    % (self.points.shape[0], _repr_tags(self.element_group_tags)))
        except Exception:                     # a repr must never break a debug session
            return "<PointMesh (unprintable)>"

    # -- sizes -----------------------------------------------------------
    @property
    def n_points(self) -> int:
        """Number of points."""
        return self.points.shape[0]

    @property
    def n_elements(self) -> int:
        """Number of elements -- the same number, since a point *is* an element."""
        return self.points.shape[0]

    @property
    def element_group_tags(self) -> list[str]:
        """Sorted unique non-empty per-point tags present on the mesh."""
        return self.element_tags.group_tags
