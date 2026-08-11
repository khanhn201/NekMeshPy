"""Variable-arity ``LineMesh`` operations -- the only ones that build a numbering."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

from .._typing import (
    BoolArray,
    FloatArray,
    IntArray,
    PointArray,
)
from ..core import conform, stations
from ..core.conform import entity_tol
from ..core.fields import gll_nodes
from ..core.tags import ElementTags, PointTags, TagBuilder, element_mask
from .linemesh import LineMesh
from .query import boundary_points


def _one_tag(tag: str | None, who: str) -> str:
    """A rung-1 tag argument as a plain string: one point carries one name."""
    if tag is None:
        return ""
    if not isinstance(tag, str):
        raise TypeError(
            "LineMesh.loft: %s must be a single tag string or None -- a slice here is "
            "one point, so there is nothing to tag per element; got %s"
            % (who, type(tag).__name__))
    return tag


def loft(
    points: PointArray,
    *,
    loop: bool = False,
    interior: PointArray | None = None,
    element_tags: str | None = None,
    first_tag: str | None = None,
    last_tag: str | None = None,
    order: int = 1,
) -> LineMesh:
    """Loft a stack of point "profiles" into a 1-D mesh -- the bottom rung of the
    uniform sweep primitive shared with :func:`QuadMesh.loft
    <nekmeshpy.quadmesh.assemble.loft>` and :func:`HexMesh.loft
    <nekmeshpy.hexmesh.assemble.loft>`.

    A slice here is a single **point**, so the three tag arguments the upper rungs
    take as "one tag or an ``ElementTags`` over the slice" reduce to one tag each:
    ``element_tags`` names every lofted line, and ``first_tag`` / ``last_tag`` name
    the chain's two end points."""
    pts = np.asarray(points, dtype=float)
    n = pts.shape[0]
    idx = np.arange(n, dtype=np.int64)
    if n < 2:
        lines: IntArray = np.zeros((0, 2), dtype=np.int64)
    elif loop:
        lines = np.column_stack([idx, np.roll(idx, -1)])
    else:
        lines = np.column_stack([idx[:-1], idx[1:]])

    first = _one_tag(first_tag, "first_tag")
    last = _one_tag(last_tag, "last_tag")
    bnd = PointTags.empty()
    if first or last:
        bb = TagBuilder(PointTags)
        L = lines.shape[0]
        if first and L:
            bb.add(0, 1, first)
        if last and L:
            bb.add(L - 1, 2, last)
        bnd = bb.build_ordered()

    if order > 1 and interior is None:
        # straight GLL blend between each line's two endpoints -- the same
        # expression the straight-sided factories (``line`` / ``rectangle``) use.
        a: PointArray = pts[lines[:, 0]]
        b: PointArray = pts[lines[:, 1]]
        g = gll_nodes(order)[1:order]              # interior GLL nodes only
        interior = a[:, None, :] + g[None, :, None] * (b - a)[:, None, :]
    tag = _one_tag(element_tags, "element_tags")
    etags = ElementTags.uniform(lines.shape[0], tag) if tag else ElementTags.empty()
    return LineMesh(pts, lines, interior, bnd, etags)


def loft_spline(
    points: PointArray,
    *,
    loop: bool = False,
    element_tags: str | None = None,
    first_tag: str | None = None,
    last_tag: str | None = None,
    order: int = 1,
) -> LineMesh:
    """:func:`loft <nekmeshpy.linemesh.assemble.loft>` with the high-order nodes read off
    a **cubic spline through the whole stack** instead of the chord of their own element.

    Same arguments, same numbering, same tags -- and the corners are the points given,
    untouched, because the spline interpolates them.  Only the private ``interior`` nodes
    differ: ``loft`` places them on the straight line between their element's two
    endpoints, which is the "high order in storage, linear in geometry" trap, while here
    they sit on a curve fitted through the neighbouring points as well.  A chain of points
    around a bend comes out bent rather than faceted.

    Nothing is resampled: this is not a smoother, and it does not move what it was
    given."""
    pts: PointArray = np.asarray(points, dtype=float)
    nz = pts.shape[0] if loop else pts.shape[0] - 1
    if order < 2 or nz < 1:
        return loft(pts, loop=loop, element_tags=element_tags, first_tag=first_tag,
                    last_tag=last_tag, order=order)
    fr: FloatArray = np.arange(nz + 1, dtype=float)
    t: FloatArray = stations.refined_lattice(fr, order)
    P: PointArray = stations.spline_levels(pts, t, loop=loop)
    slot: IntArray = (np.arange(nz)[:, None] * order
                      + np.arange(1, order)[None, :])        # (nz, order-1)
    return loft(pts, loop=loop, interior=P[slot], element_tags=element_tags,
                first_tag=first_tag, last_tag=last_tag, order=order)


def _eval_curve(f: Callable[[FloatArray], PointArray], t: FloatArray) -> PointArray:
    """``f(t)`` as a validated ``(len(t), 3)`` array."""
    P: PointArray = np.asarray(f(t), dtype=float)
    if P.shape != (t.shape[0], 3):
        raise ValueError(
            "curve callable must return (len(t), 3) points; got shape %r for %d "
            "parameters" % (P.shape, t.shape[0]))
    return P


def loft_fn(f: Callable[[FloatArray], PointArray], fractions: float | FloatArray, *,
            loop: bool = False, order: int = 1,
            element_tags: str | None = None,
            first_tag: str | None = None,
            last_tag: str | None = None) -> LineMesh:
    """Loft a curve given as its own analytic parametrization -- :func:`loft
    <nekmeshpy.linemesh.assemble.loft>` with the profiles **evaluated** rather than
    handed in, so **every** node (corners *and* the private high-order ``interior``)
    comes from calling ``f`` and nothing is ever placed on a chord."""
    fr: FloatArray = np.atleast_1d(np.asarray(fractions, dtype=float))
    ni = fr.shape[0] - 1
    if ni < 1:
        raise ValueError(
            "loft_fn needs at least 2 fractions (one element), got %d" % fr.shape[0])
    if loop and ni < 2:
        raise ValueError(
            "loft_fn(loop=True) needs at least 3 fractions (two elements), got %d -- "
            "the last one is the wrap back to the first point, so it is not a point of "
            "its own" % fr.shape[0])

    # every node of the chain, corners and interiors alike, as one parameter array --
    # so the interiors ride the true curve instead of ``loft``'s straight chord blend.
    t: FloatArray = stations.refined_lattice(fr, order)
    P: PointArray = _eval_curve(f, t)
    corners: PointArray = P[::order]
    if loop:
        # the wrap node is the seam element's far end; ``loft(loop=True)`` supplies it
        # from point 0, so it must not also be numbered as a point of its own.
        gap = float(np.linalg.norm(corners[-1] - corners[0]))
        tol = entity_tol(corners)
        if gap > tol:
            raise ValueError(
                "loft_fn(loop=True) needs the last fraction to map back to the "
                "first point, but f(%g) and f(%g) are %g apart (tolerance %g).  Pass "
                "the trailing wrap value -- np.linspace(0, 2*np.pi, n+1) for a "
                "2*pi-periodic f -- so the seam element's own nodes can be evaluated."
                % (fr[-1], fr[0], gap, tol))
        corners = corners[:-1]

    if order == 1:
        return loft(corners, loop=loop, element_tags=element_tags,
                    first_tag=first_tag, last_tag=last_tag)
    slot: IntArray = (np.arange(ni)[:, None] * order
                      + np.arange(1, order)[None, :])        # (n, order-1)
    return loft(corners, loop=loop, interior=P[slot], element_tags=element_tags,
                first_tag=first_tag, last_tag=last_tag, order=order)


def merge(meshes: Sequence[LineMesh], *,
          tol: float | None = None) -> LineMesh:
    """Merge line meshes into one, welding coincident **topological end points** (the
    degree-1 chain ends -- the 1-D analogue of the boundary vertices
    ``QuadMesh.merge``/``HexMesh.merge`` weld)."""
    meshes = list(meshes)
    pos = [np.asarray(m.points, dtype=float).reshape(-1, 3) for m in meshes]
    counts = [p.shape[0] for p in pos]
    # the weldable points here are the chain **ends** -- the 1-D boundary
    points, point_id = conform.weld_points(pos, [boundary_points(m) for m in meshes], tol)

    line_list: list[IntArray] = []
    bnd_list: list[PointTags] = []
    etag_list: list[ElementTags] = []
    noff = loff = 0
    for m, c in zip(meshes, counts):
        line_list.append(point_id[m.lines + noff])   # local -> welded id
        # ids shift by this block's offset; sides stay local to their element
        etag_list.append(m.element_tags.offset(loff))
        bnd_list.append(m.point_tags.offset(loff))
        noff += c
        loff += m.n_lines
    lines = (np.concatenate(line_list, axis=0) if line_list
             else np.zeros((0, 2), np.int64))
    etags = ElementTags.concat(etag_list)
    bnd = PointTags.concat(bnd_list)

    # order-N: welding only touches endpoints (corners, which are re-numbered into
    # the merged points), and every high-order node of a line is *private*, so the
    # merged interior is just the blocks concatenated in the same order the lines
    # were -- nothing to reconcile, nothing to re-pin.
    order = meshes[0].order if meshes else 1
    if any(m.order != order for m in meshes):
        raise ValueError("LineMesh.merge: all meshes must share the same order")
    interior: PointArray | None = None
    if meshes:
        interior = np.concatenate([m.interior for m in meshes], axis=0)

    return LineMesh(points, lines, interior, bnd, etags)

def _subset(mesh: LineMesh, keep: BoolArray) -> tuple[LineMesh, IntArray]:
    """``(the kept lines as a LineMesh, new_line_of)`` -- the rung-1 half of every
    ``select``, and what the rung above calls to carry its shared edges across.

    Points referenced by no kept line are dropped and the rest compacted, so the result
    is a mesh in its own right rather than a view with holes in its numbering.  Order is
    preserved: the private interior nodes ride along untouched."""
    kept, new_line_of = conform.renumber_map(keep)
    lines: IntArray = mesh.lines[kept]
    used: IntArray = np.unique(lines) if lines.size else np.zeros(0, np.int64)
    new_point_of: IntArray = np.full(mesh.n_points, -1, dtype=np.int64)
    new_point_of[used] = np.arange(used.shape[0], dtype=np.int64)
    return (LineMesh(mesh.points[used], new_point_of[lines], mesh.interior[kept],
                     mesh.point_tags.renumber(new_line_of),
                     mesh.element_tags.gather(kept)),
            new_line_of)


def select(mesh: LineMesh, which: str | BoolArray | IntArray | Sequence[int]
           ) -> LineMesh:
    """The named lines as a curve of their own, renumbered from zero.

    ``which`` is a tag string (every line carrying it), a ``(L,)`` boolean mask, or an
    array of line ids.  Kept lines hold their relative order and their tags; points no
    kept line touches are dropped.  The inverse of :func:`merge`, and one of the three
    operations that manufacture a numbering."""
    return _subset(mesh, element_mask(which, mesh.element_tags, mesh.n_lines,
                                      "linemesh.select"))[0]


def remove(mesh: LineMesh, which: str | BoolArray | IntArray | Sequence[int]
           ) -> LineMesh:
    """The complement of :func:`select`: everything ``which`` does **not** name."""
    return _subset(mesh, ~element_mask(which, mesh.element_tags, mesh.n_lines,
                                       "linemesh.remove"))[0]


def components(mesh: LineMesh) -> list[LineMesh]:
    """The mesh split into its connected pieces -- one ``LineMesh`` per group of lines
    reachable through shared points, in the order their first line appears.

    A single chain or loop comes back as a one-element list; what makes this worth
    calling is the mesh that turns out to be two."""
    n, labels = conform.element_components(mesh.lines, mesh.n_points)
    return [_subset(mesh, labels == c)[0] for c in range(n)]


__all__ = [
    "components",
    "loft",
    "loft_fn",
    "loft_spline",
    "merge",
    "remove",
    "select",
]
