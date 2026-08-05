"""1-D mesh container: line elements ``(L,2)`` over a shared ``(N,3)`` point array.

The line sibling of QuadMesh/HexMesh; it can branch rather than being a single
ordered path. It carries sparse per-line ``element_tags`` and a sparse tagged
:class:`~nekmeshpy.model.tags.PointTags` table of tagged end points, both of which
sweep up on extrude.
Open vs closed is a property of the ``lines`` connectivity itself -- a loop is a
cycle of line elements with no degree-1 end point -- and is stored nowhere;
factories build the common cases (``loft`` / ``line`` / ``arc`` / ``circle`` /
``rectangle``) and every curve is meshed exactly at the points given -- there is no
resampling here.

``lines`` is a **required** constructor argument: the container never invents
connectivity, so there is nothing in ``LineMesh`` that could imply a wrap.  The
bottom rung of the uniform sweep primitive, :func:`linemesh.assemble.loft <nekmeshpy.linemesh.assemble.loft>`, is what authors it
-- one dimension below ``QuadMesh.loft``/``HexMesh.loft``, each "profile" is a single
point and the rungs joining consecutive profiles *are* the line elements, with
``loop=True`` adding the closing rung from the last point back to the first.  It is
the **only** connectivity-authoring entry point: a chain is ``loft(points)``, a ring
is ``loft(points, loop=True)``, and anything else comes in through the constructor
with its ``lines`` spelled out.

This file stays a **pure container**: storage, validation, and the derived views.
Every operation on a finished mesh lives beside it as a free function bound onto the
class in ``linemesh/__init__.py``, split by arity and by rung delta -- ``_assemble.py``
(the n-ary ``loft`` / ``merge``, which build a new index space), ``_morph.py`` (the
rung-preserving ``blend``), ``_query.py`` (read-only queries), ``_open.py`` /
``_closed.py`` (shape factories).  Adding an operation touches only the sibling module,
never this one.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .._typing import (
    IntArray,
    PointArray,
    StrArray,
)
from ..model.tags import ElementTags, PointTags


def _as_points(points: PointArray) -> PointArray:
    """Normalize an array-like to a validated ``(N,3)`` float point array, raising
    the one actionable "points live in 3-D" error for anything else.  Shared by
    ``LineMesh.__init__`` and :func:`linemesh.assemble.loft <nekmeshpy.linemesh.assemble.loft>` so both report it identically."""
    a: PointArray = np.asarray(points, dtype=float)
    if a.ndim == 1:
        a = a.reshape(1, -1)
    if a.ndim != 2 or a.shape[1] != 3:
        raise ValueError(
            f"points must be (N,3) 3-D coordinates; got "
            f"{a.shape} -- add a z column (all geometry lives in 3-D)")
    return a


def _repr_tags(tags: Sequence[str], limit: int = 4) -> str:
    """Render a tag vocabulary as ``{inlet,outlet}`` for a container's ``__repr__``,
    eliding past ``limit`` entries with ``...`` so one loud tag scheme cannot push the
    counts off the line.  Empty tags are dropped, so an untagged mesh reads ``{}``.

    Lives here, at the bottom rung, because all three ladder containers' reprs share it
    and must render identically -- ``QuadMesh`` / ``HexMesh`` import it directly (they
    already import ``LineMesh`` from this package).  ``TriMesh`` carries no tags at all
    and needs none."""
    kept = [t for t in tags if t]
    shown = kept[:limit] + (["..."] if len(kept) > limit else [])
    return "{%s}" % ",".join(shown)


class LineMesh:
    """A 1-D mesh: an ``(N,3)`` point array with ``(L,2)`` line connectivity, a
    sparse per-line ``element_tags``, and a ``point_tags`` table of tagged end points
    (``side`` 1-2). Build with ``loft`` / ``line`` / ``arc`` / ``circle``
    / ``rectangle``."""

    # local line "edges": row s-1 is side s -> the single local vertex it names.
    EDGE_POINTS = np.array([[0], [1]], dtype=np.int64)

    def __init__(
        self,
        points: PointArray,
        lines: IntArray,
        interior: PointArray | None = None,
        point_tags: PointTags | None = None,
        element_tags: ElementTags | None = None,
        *,
        order: int = 1,
    ) -> None:
        """Construct from arrays: ``points`` ``(N,3)`` (must be 3-D), the **required**
        ``lines`` ``(L,2)`` connectivity, the per-line ``interior`` nodes, an optional
        :class:`PointTags <nekmeshpy.model.tags.PointTags>` naming end points of
        lines, and an optional :class:`ElementTags
        <nekmeshpy.model.tags.ElementTags>` naming whichever lines are tagged.

        The argument order is the ladder's: ``(rung below, incidence, interior,
        side tags, element_tags, *, order)``, matching
        :class:`QuadMesh <nekmeshpy.quadmesh.QuadMesh>` (``lines, quad, flip,
        interior, ...``) and :class:`HexMesh <nekmeshpy.hexmesh.HexMesh>`
        (``quads, hex, face_orient, interior, ...``) position for position -- a line element has no orientation bit, so it simply
        has no ``flip`` / ``face_orient`` slot.

        The container never synthesizes connectivity -- there is no "consecutive
        chain" default and therefore nothing here that could imply a wrap.  Callers
        either own their ``lines`` outright (``merge``'s rewelded lines, ``blend``'s
        copy of ``a.lines``, the quad/hex edge meshes built from
        ``conform.unique_edges``) or author them with :func:`loft <nekmeshpy.linemesh.assemble.loft>`, which is the only
        connectivity-generating entry point (``loop=False`` chain / ``loop=True``
        ring).

        ``order`` is the global polynomial order (default 1 = linear).  At
        ``order > 1`` pass ``interior``: the per-line *private* interior nodes
        ``(L, order-1, 3)`` in ascending GLL order, i.e. the nodes strictly between
        each line's two endpoints.  A line element has no shared edges or faces, so
        its endpoints (owned by ``points[lines]``) are its only conformal nodes and
        there is nothing to reconcile between lines -- ``interior`` is the whole
        high-order state of a ``LineMesh``.

        ``re2`` export stays linear; only ``vtu`` reads the high-order nodes."""
        self.points: PointArray = _as_points(points)
        self.lines: IntArray = np.asarray(lines, dtype=np.int64).reshape(-1, 2)

        #: which lines carry a region tag (sparse -- an untagged mesh stores nothing)
        self.element_tags: ElementTags = (
            ElementTags.empty() if element_tags is None else element_tags)
        #: tagged end points, ``side`` 1-2, coupled with their names
        self.point_tags: PointTags = (
            PointTags.empty() if point_tags is None else point_tags)
        self.element_tags.check_within(self.lines.shape[0], "lines")
        self.point_tags.check_within(self.lines.shape[0], "lines")

        self._order = int(order)
        #: ``(L, order-1, 3)`` per-line private high-order interior nodes (ascending GLL
        #: order).  A line has no shared edges/faces, so its endpoints (owned by
        #: ``points[lines]``) are its only conformal nodes and every interior node is
        #: private -- there is nothing to reconcile between lines.  Empty at order 1.
        self.interior: PointArray = self._check_interior(interior)

    def _check_interior(self, interior: PointArray | None) -> PointArray:
        """Validate the **native** per-line private interior ``(L, order-1, 3)`` and
        return it as float.  Nothing is shared between line elements beyond the
        endpoints in ``points``, so there is no corner check to run here -- the corners
        are single-sourced by construction.  ``interior=None`` is allowed only at
        order 1 (empty interior)."""
        order = self._order
        if order < 1:
            raise ValueError("LineMesh: order must be >= 1, got %d" % order)
        n_lines = self.lines.shape[0]
        k = order - 1
        if interior is None:
            if order > 1:
                raise ValueError(
                    "LineMesh: order %d > 1 requires the per-line interior nodes "
                    "(pass interior=(L, order-1, 3), or build the curve with a "
                    "factory such as LineMesh.circle(..., order=%d))"
                    % (order, order))
            return np.zeros((n_lines, 0, 3), dtype=float)
        ia: PointArray = np.asarray(interior, dtype=float)
        if ia.shape != (n_lines, k, 3):
            raise ValueError(
                "LineMesh: interior must be (L, order-1, 3) = (%d,%d,3), got %s"
                % (n_lines, k, ia.shape))
        return ia

    @property
    def order(self) -> int:
        """Global polynomial order (1 = linear)."""
        return self._order

    # -- sizes / topology -----------------------------------------------
    def __repr__(self) -> str:
        """One-line REPL summary: element / point counts, ``order``, and the tag
        vocabulary -- the questions a caller actually has at the prompt, where this
        toolkit is mostly driven from.  The same field set is rendered by
        :class:`QuadMesh <nekmeshpy.quadmesh.QuadMesh>` and
        :class:`HexMesh <nekmeshpy.hexmesh.HexMesh>` so the three read as a family.

        Deliberately **cheap** -- it reads stored array shapes and the two tag
        properties only, deriving no topology -- and **total**: any failure degrades to
        a bare marker instead of raising, because a repr that throws on a half-built or
        degenerate mesh makes the debugging session it was meant to serve strictly
        worse."""
        try:
            return ("<LineMesh %d points, %d lines, order %d, element_tags=%s, "
                    "point_tags=%s>"
                    % (self.points.shape[0], self.lines.shape[0], self._order,
                       _repr_tags(self.element_group_tags),
                       _repr_tags(self.point_group_tags)))
        except Exception:                     # a repr must never break a debug session
            return "<LineMesh (unprintable)>"

    def __len__(self) -> int:
        """Number of points."""
        return self.points.shape[0]

    def __array__(self, dtype: np.dtype[Any] | None = None) -> NDArray[Any]:
        """Expose the ``(N,3)`` point array to ``numpy.asarray``."""
        return self.points if dtype is None else self.points.astype(dtype)

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
    def element_group_tags(self) -> list[str]:
        """Sorted unique non-empty per-line element tags present on the mesh."""
        return self.element_tags.group_tags

    @property
    def point_group_tags(self) -> list[str]:
        """Sorted unique tags of the tagged end points present on the mesh."""
        return self.point_tags.group_tags

    # -- helpers for the operation modules -----------------------------
    @staticmethod
    def _cap_tags(cap: str | Sequence[str] | StrArray, N: int = 1) -> list[str]:
        """Normalize a cap tag to one tag per cap **node** (length ``N``): a scalar
        ``str`` tags the whole cap, an array-like is one tag per node.

        The bottom rung of the cap-tag normalizer shared with
        :meth:`QuadMesh._cap_tags <nekmeshpy.quadmesh.QuadMesh._cap_tags>` (one tag
        per section line) and
        :meth:`HexMesh._cap_tags <nekmeshpy.hexmesh.HexMesh._cap_tags>` (one per
        section quad).  One rung down a "profile" is a single point, so a chain's
        near / far cap is that one end **node** and ``N`` is 1 -- the array form is
        therefore a one-element list.  It exists so the three rungs accept the same
        argument shapes: a caller (or a generic wrapper) can pass ``["inlet"]`` at
        any rung and get the same meaning."""
        if isinstance(cap, str):
            return [cap] * N
        arr = np.asarray(cap, dtype=np.str_).reshape(-1)
        if arr.shape[0] != N:
            raise ValueError("cap tags length (%d) must match cap nodes (%d)"
                             % (arr.shape[0], N))
        return [str(x) for x in arr.tolist()]

    def _seg_tags(self) -> list[str] | None:
        """The element tags densified to a ``list[str]`` for the ordered ops, or
        ``None`` if every element is untagged (so an untagged mesh stays untagged)."""
        if not self.element_tags:
            return None
        return [str(x) for x in self.element_tags.dense(self.lines.shape[0]).tolist()]

