"""1-D mesh container: line elements ``(L,2)`` over a shared ``(N,3)`` point array."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .._typing import (
    IntArray,
    PointArray,
)
from ..model.tags import ElementTags, PointTags


def _repr_tags(tags: Sequence[str], limit: int = 4) -> str:
    """Render a tag vocabulary as ``{inlet,outlet}`` for a container's ``__repr__``,
    eliding past ``limit`` entries with ``...`` so one loud tag scheme cannot push the
    counts off the line. Empty tags are dropped, so an untagged mesh reads ``{}``."""
    kept = [t for t in tags if t]
    shown = kept[:limit] + (["..."] if len(kept) > limit else [])
    return "{%s}" % ",".join(shown)


class LineMesh:
    """A 1-D mesh: an ``(N,3)`` point array with ``(L,2)`` line connectivity, a sparse
    per-line ``element_tags``, and a ``point_tags`` table of tagged end points (``side``
    1-2)."""

    # local line "edges": row s-1 is side s -> the single local vertex it names.
    EDGE_POINTS = np.array([[0], [1]], dtype=np.int64)

    def __init__(
        self,
        points: PointArray,
        lines: IntArray,
        interior: PointArray | None = None,
        point_tags: PointTags | None = None,
        element_tags: ElementTags | None = None,
    ) -> None:
        """Construct from arrays: ``points`` ``(N,3)`` (must be 3-D), the **required**
        ``lines`` ``(L,2)`` connectivity, the per-line ``interior`` nodes, an optional
        :class:`PointTags <nekmeshpy.model.tags.PointTags>` naming end points of lines,
        and an optional :class:`ElementTags <nekmeshpy.model.tags.ElementTags>` naming
        whichever lines are tagged."""

        self.points: PointArray = np.asarray(points, dtype=float)
        if self.points.ndim != 2 or self.points.shape[1] != 3:
            raise ValueError(
                "LineMesh: points must be (N,3) 3-D coordinates; got %s -- add a z "
                "column (all geometry lives in 3-D)" % (self.points.shape,))

        F = self.points.shape[0]                 # the rung below: shared points
        self.lines: IntArray = np.asarray(lines, dtype=np.int64).reshape(-1, 2)
        if self.lines.size and (self.lines.min() < 0 or self.lines.max() >= F):
            raise ValueError(
                "LineMesh: lines must index the %d points; got ids in [%d, %d]"
                % (F, int(self.lines.min()), int(self.lines.max())))
        E = self.lines.shape[0]

        if interior is None:
            interior = np.zeros((E, 0, 3), dtype=float)
        self.interior: PointArray = np.asarray(interior, dtype=float)
        if (self.interior.ndim != 3 or self.interior.shape[0] != E
                or self.interior.shape[2] != 3):
            raise ValueError(
                "LineMesh: interior must be (L, order-1, 3) with L = %d lines, "
                "got %s" % (E, self.interior.shape))

        self.element_tags = ElementTags.empty() if element_tags is None else element_tags
        self.point_tags = PointTags.empty() if point_tags is None else point_tags
        self.element_tags.check_within(E)
        self.point_tags.check_within(E)

    def __repr__(self) -> str:
        """One-line REPL summary: element / point counts, ``order``, and the tag
        vocabulary -- the questions a caller actually has at the prompt, where this
        toolkit is mostly driven from."""
        try:
            return ("<LineMesh %d points, %d lines, order %d, element_tags=%s, "
                    "point_tags=%s>"
                    % (self.points.shape[0], self.lines.shape[0], self.order,
                       _repr_tags(self.element_group_tags),
                       _repr_tags(self.point_group_tags)))
        except Exception:                     # a repr must never break a debug session
            return "<LineMesh (unprintable)>"

    @property
    def order(self) -> int:
        """Global polynomial order (1 = linear)."""
        return int(self.interior.shape[1]) + 1

    # -- sizes -----------------------------------------------------------
    @property
    def n_points(self) -> int:
        """Number of points."""
        return self.points.shape[0]

    @property
    def n_lines(self) -> int:
        """Number of line elements."""
        return self.lines.shape[0]

    @property
    def n_point_tags(self) -> int:
        """Number of tagged end points."""
        return len(self.point_tags)

    @property
    def point_group_tags(self) -> list[str]:
        """Sorted unique tags of the tagged end points present on the mesh."""
        return self.point_tags.group_tags

    @property
    def element_group_tags(self) -> list[str]:
        """Sorted unique non-empty per-line element tags present on the mesh."""
        return self.element_tags.group_tags
