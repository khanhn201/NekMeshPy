"""1-D mesh container: line elements ``(L,2)`` over a shared ``(N,3)`` point array."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .._typing import (
    IntArray,
    PointArray,
)
from ..core.tags import ElementTags
from ..pointmesh import PointMesh


def _repr_tags(tags: Sequence[str], limit: int = 4) -> str:
    """Render a tag vocabulary as ``{inlet,outlet}`` for a container's ``__repr__``,
    eliding past ``limit`` entries with ``...`` so one loud tag scheme cannot push the
    counts off the line. Empty tags are dropped, so an untagged mesh reads ``{}``."""
    kept = [t for t in tags if t]
    shown = kept[:limit] + (["..."] if len(kept) > limit else [])
    return "{%s}" % ",".join(shown)


class LineMesh:
    """A 1-D mesh: a :class:`PointMesh <nekmeshpy.pointmesh.pointmesh.PointMesh>` of the shared
    points with ``(L,2)`` line connectivity and a sparse per-line ``element_tags``.

    Its **point tags are the point mesh's own element tags** -- a point is one object
    the whole mesh shares, so naming it is naming that object, not naming it once per
    incident line. :attr:`point_tags` reads them through."""

    # local line "edges": row s-1 is side s -> the single local vertex it names.
    EDGE_POINTS = np.array([[0], [1]], dtype=np.int64)

    def __init__(
        self,
        point_mesh: PointMesh | PointArray,
        lines: IntArray,
        interior: PointArray | None = None,
        element_tags: ElementTags | None = None,
    ) -> None:
        """Construct from the rung below: ``point_mesh`` (a ``PointMesh``, or a bare
        ``(N,3)`` array promoted to an untagged one), the **required** ``lines``
        ``(L,2)`` connectivity, the per-line ``interior`` nodes, and an optional
        :class:`ElementTags <nekmeshpy.core.tags.ElementTags>` naming whichever lines
        are tagged.

        Passing an array is the authoring form and the one every factory building fresh
        geometry wants -- there are no point tags to carry yet. Pass a ``PointMesh``
        when there are: an operation rebuilding this mesh around moved coordinates has
        to hand the tags on rather than let the promotion drop them."""
        self.point_mesh: PointMesh = (
            point_mesh if isinstance(point_mesh, PointMesh)
            else PointMesh(point_mesh))

        F = self.point_mesh.n_points               # the rung below: shared points
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
        self.element_tags.check_within(E)

    @property
    def corners(self) -> IntArray:
        """``(L,2)`` incidence into the point mesh -- the same array ``lines`` holds,
        under the name the rungs above use for this slot (``corners`` on ``QuadMesh`` /
        ``HexMesh``).

        A point *is* its own corner, so at this rung the incidence into the rung below
        and the corner connectivity are one table; the two names exist so code that
        walks every rung can ask for either without a special case."""
        return self.lines

    @property
    def points(self) -> PointArray:
        """The ``(N,3)`` coordinates -- a **derived view** of the point mesh's own
        array, so ``mesh.points[:] = X`` repositions every rung built on it."""
        return self.point_mesh.points

    @property
    def point_tags(self) -> ElementTags:
        """The tags on the shared points, over **point ids**.

        This is ``point_mesh``'s own ``element_tags`` read through, not a table of its
        own: rung *N*'s side tags are rung *N-1*'s element tags all the way up the
        ladder, and the bottom is no exception."""
        return self.point_mesh.element_tags

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
