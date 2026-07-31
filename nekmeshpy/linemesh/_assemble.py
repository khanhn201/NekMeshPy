"""Variable-arity ``LineMesh`` operations -- the only ones that build a numbering.

``loft`` (``n`` points -> a line mesh, rung delta +1) and ``merge`` (``n`` line meshes
-> one, rung delta 0) are the two n-ary operations at this rung, and they are the only
code here that manufactures a global point/element index space from scratch: ``loft``
numbers the chain it authors, ``merge`` builds the ``remap`` / ``survivors`` /
``point_id`` tables of the weld.  Every fixed-arity operation either reuses an existing
numbering (``blend``) or delegates to one of these two.

Free functions bound onto :class:`~nekmeshpy.LineMesh` by ``linemesh/__init__.py``;
internal toolkit code imports them from here directly rather than through the bound
``LineMesh.<name>`` sugar.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .._typing import (
    BoolArray,
    IntArray,
    PointArray,
    StrArray,
)
from ..model.fields import gll_nodes, reject_loop_caps
from ._query import boundary_points
from .linemesh import LineMesh, _as_points


def loft(
    points: NDArray[Any],
    *,
    loop: bool = False,
    interior: PointArray | None = None,
    boundaries: IntArray | None = None,
    boundary_tags: StrArray | Sequence[str] | None = None,
    element_tags: StrArray | Sequence[str] | None = None,
    first_tag: str | Sequence[str] | StrArray = "",
    last_tag: str | Sequence[str] | StrArray = "",
    order: int = 1,
) -> LineMesh:
    """Loft a stack of point "profiles" into a 1-D mesh -- the bottom rung of the
    uniform sweep primitive shared with
    :meth:`QuadMesh.loft <nekmeshpy.quadmesh.QuadMesh.loft>` and
    :meth:`HexMesh.loft <nekmeshpy.hexmesh.HexMesh.loft>`.

    One dimension below a quad loft each profile is a **single point**, so the
    rungs joining consecutive profiles *are* the line elements: ``points``
    ``(N,3)`` lofts into the consecutive chain ``[[0,1], ..., [N-2,N-1]]``, and
    ``loop=True`` appends one more rung -- from the last point back to the first
    (``[N-1,0]``) -- closing the curve.  The seam rung is appended exactly once
    and no profile is duplicated, so a lofted loop carries ``N`` points and ``N``
    lines with no degree-1 end.  This is the only method that authors ``lines``:
    a chain is ``loft(points)``, a ring ``loft(points, loop=True)``.

    ``element_tags`` is the dense per-line tag array (line ``m`` = point ``m`` ->
    ``m+1``, and for ``loop=True`` line ``N-1`` = point ``N-1`` -> ``0``);
    ``boundaries`` / ``boundary_tags`` are passed through verbatim.
    ``first_tag`` / ``last_tag`` name the near / far **end points** of the chain
    (the 1-D end caps: line ``0`` side ``1`` and line ``L-1`` side ``2``).  A cap
    here is a single node, so each takes a scalar ``str`` or -- for shape parity
    with :meth:`QuadMesh.loft <nekmeshpy.quadmesh.QuadMesh.loft>` /
    :meth:`HexMesh.loft <nekmeshpy.hexmesh.HexMesh.loft>`, whose caps carry one tag
    per section line / quad -- a one-element array-like.  A closed sweep has no
    near/far cap, so passing either with ``loop=True`` raises ``ValueError``
    rather than silently dropping it.

    At ``order > 1`` an explicit ``interior`` ``(L, order-1, 3)`` is used as-is
    (that is how ``circle`` stamps true-arc nodes); when it is omitted each line's
    private interior is built here as the **straight GLL blend** between its two
    endpoints, which is exactly what a straight-sided curve wants."""
    pts = _as_points(points)
    n = pts.shape[0]
    if loop:
        reject_loop_caps("LineMesh.loft", first_tag, last_tag)
    idx = np.arange(n, dtype=np.int64)
    if n < 2:
        lines: IntArray = np.zeros((0, 2), dtype=np.int64)
    elif loop:
        lines = np.column_stack([idx, np.roll(idx, -1)])
    else:
        lines = np.column_stack([idx[:-1], idx[1:]])

    bnd = boundaries
    names = boundary_tags
    # a chain's cap is a single end node, so ``_cap_tags`` normalizes to one tag --
    # the rung-1 form of the same scalar-or-per-element argument ``QuadMesh.loft`` /
    # ``HexMesh.loft`` take, so all three rungs accept the same shapes.
    first, last = LineMesh._cap_tags(first_tag)[0], LineMesh._cap_tags(last_tag)[0]
    if first or last:
        rows = [[int(r[0]), int(r[1])]
                for r in np.asarray(bnd if bnd is not None else
                                    np.zeros((0, 2), np.int64),
                                    dtype=np.int64).reshape(-1, 2)]
        tags = [str(t) for t in np.asarray(
            names if names is not None else np.empty(0, dtype=np.str_),
            dtype=np.str_).reshape(-1).tolist()]
        L = lines.shape[0]
        if first and L:
            rows.append([0, 1])
            tags.append(first)
        if last and L:
            rows.append([L - 1, 2])
            tags.append(last)
        bnd, names = LineMesh._order_bnd(rows, tags)

    if order > 1 and interior is None:
        # straight GLL blend between each line's two endpoints -- the same
        # expression the straight-sided factories (``line`` / ``rectangle``) use.
        a: PointArray = pts[lines[:, 0]]
        b: PointArray = pts[lines[:, 1]]
        g = gll_nodes(order)[1:order]              # interior GLL nodes only
        interior = a[:, None, :] + g[None, :, None] * (b - a)[:, None, :]
    return LineMesh(pts, lines, interior, bnd, names, element_tags, order=order)

def merge(meshes: list[LineMesh], *,
          tol: float | None = None) -> LineMesh:
    """Merge line meshes into one, welding coincident **topological end
    points** (the degree-1 chain ends -- the 1-D analogue of the boundary
    vertices ``QuadMesh.merge``/``HexMesh.merge`` weld).  ``tol`` is the
    absolute coincidence distance (default ``1e-7`` x the extent).  Dense
    ``element_tags`` and tagged ``boundaries`` concatenate with each block's
    line ids offset; interior points are never welded.  Closedness is not
    tracked anywhere -- it simply falls out of the welded connectivity: if no
    degree-1 end survives the result *is* a loop, so two shared-endpoint
    ``A1 -> A2`` arcs (reverse one so the traversal doesn't cross) weld at
    ``A1`` and ``A2`` into a single cycle, the clean way to close a ring from
    two half-arcs."""
    meshes = list(meshes)
    pos = [np.asarray(m.points, dtype=float).reshape(-1, 3) for m in meshes]
    counts = [p.shape[0] for p in pos]
    P = np.concatenate(pos, axis=0) if pos else np.zeros((0, 3))
    total = P.shape[0]

    remap = np.arange(total, dtype=np.int64)
    is_bnd: BoolArray = np.zeros(total, dtype=bool)
    noff = 0
    for m, c in zip(meshes, counts):
        is_bnd[noff + boundary_points(m)] = True
        noff += c
    bidx = np.flatnonzero(is_bnd)
    if bidx.size:
        scl = float(np.max(P.max(axis=0) - P.min(axis=0))) if total else 0.0
        t = tol if tol is not None else (1e-7 * scl if scl > 0 else 1.0)
        keys = np.round(P[bidx, :] / t).astype(np.int64)
        _, first_local, inverse = np.unique(
            keys, axis=0, return_index=True, return_inverse=True)
        remap[bidx] = bidx[first_local][inverse.ravel()]

    survivors = np.unique(remap)
    new_id: IntArray = np.empty(total, dtype=np.int64)
    new_id[survivors] = np.arange(survivors.size)
    point_id = new_id[remap]
    points = P[survivors, :]

    line_list: list[IntArray] = []
    bnd_list: list[IntArray] = []
    name_list: list[StrArray] = []
    etag_list: list[StrArray] = []
    noff = loff = 0
    for m, c in zip(meshes, counts):
        line_list.append(point_id[m.lines + noff])   # local -> welded id
        etag_list.append(m.element_tags)
        if m.boundaries.shape[0]:
            b: IntArray = m.boundaries.copy()
            b[:, 0] += loff                          # shift line ids; sides local
            bnd_list.append(b)
            name_list.append(m.boundary_tags)
        noff += c
        loff += m.n_lines
    lines = (np.concatenate(line_list, axis=0) if line_list
             else np.zeros((0, 2), np.int64))
    etags = (np.concatenate(etag_list) if etag_list
             else np.empty(0, dtype=np.str_))
    bnd = (np.concatenate(bnd_list, axis=0) if bnd_list
           else np.zeros((0, 2), np.int64))
    names = (np.concatenate(name_list) if name_list
             else np.empty(0, dtype=np.str_))

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

    return LineMesh(points, lines, interior, bnd, names, etags, order=order)


#: Variable-arity combinators bound onto ``LineMesh`` as ``staticmethod``.
FACTORIES: dict[str, Any] = {
    "loft": loft,
    "merge": merge,
}
