"""Variable-arity ``QuadMesh`` operations -- the only ones that build a numbering.

``loft`` (``n`` line profiles -> a section, rung delta +1) and ``merge`` (``n`` sections
-> one, rung delta 0) are the two n-ary operations at this rung, and the only code here
that manufactures a global point/element index space from scratch: ``loft`` numbers the
layer-by-layer B-rep it assembles (``prof_off`` / ``rung_off``, global id
``i*nn + v``), ``merge`` builds the ``remap`` / ``survivors`` / ``point_id`` tables of
the weld.  Every fixed-arity operation either reuses an existing numbering (``blend``)
or delegates here (``extrude``, ``from_grid``, the region factories).

They also differ in what they owe the B-rep: ``loft`` *generates* topology, so no
shared entity is ever duplicated and nothing needs reconciling; ``merge`` *rewrites* it
against a new corner numbering, so it must re-scatter the shared edge nodes owner-wins
and verify every other incident copy.

Free functions bound onto :class:`~nekmeshpy.QuadMesh` by ``quadmesh/__init__.py``;
internal toolkit code imports them from here directly rather than through the bound
``QuadMesh.<name>`` sugar.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from .._typing import (
    BoolArray,
    FloatArray,
    IntArray,
    PointArray,
    StrArray,
)
from ..linemesh import LineMesh
from ..model import conform
from ..model.fields import gll_nodes, reject_loop_caps
from ._query import _boundary_mask
from .quadmesh import (
    NO_BOUNDARY,
    QuadMesh,
    _coons_at,
    _quad_interior_slots,
)


def loft(
    slices: Sequence[LineMesh],
    *,
    loop: bool = False,
    first_tag: str | Sequence[str] | StrArray = "",
    last_tag: str | Sequence[str] | StrArray = "",
) -> QuadMesh:
    """Loft a stack of conformal ``LineMesh`` profiles into a quad section
    (the general primitive behind :meth:`extrude`, and the middle rung of the
    uniform sweep shared with
    :meth:`LineMesh.loft <nekmeshpy.linemesh.LineMesh.loft>` and
    :meth:`HexMesh.loft <nekmeshpy.hexmesh.HexMesh.loft>`).

    ``slices`` is ``nz+1`` line profiles sharing the same ``lines``,
    ``element_tags``, and ``boundaries``; consecutive profiles form ``nz`` quad
    layers.  For line ``(a, b)`` at layer ``i`` the column quad is
    ``[a_i, b_i, b_{i+1}, a_{i+1}]``.  The line's ``element_tags`` ride onto every
    quad in its column and tagged boundary points onto the swept wall edges;
    ``first_tag`` / ``last_tag`` name the near / far cap edges (scalar or per-line
    array).

    ``loop=True`` makes the sweep **periodic**: the last profile is joined back to
    the *first*, so ``M`` profiles give ``M`` layers instead of ``M-1``.  It falls
    out of the layer-by-layer assembly as one extra iteration -- exactly one more
    rung block, appended once at the end, with the first profile's lines *not*
    duplicated -- so the seam is a genuine shared entity and the closed tube has
    no free boundary edge in the sweep direction.  A closed sweep has no near/far
    cap, so ``first_tag`` / ``last_tag`` with ``loop=True`` raise ``ValueError``
    rather than being silently dropped, and no cap boundary row is emitted.
    (Side-wall boundaries derived from the profiles' own boundary points are
    unaffected.)

    The section's B-rep is assembled **layer by layer**, never re-derived from
    corner connectivity: each profile's own ``LineMesh`` is merged into the growing
    global edge mesh (its ``lines`` and their private ``interior`` nodes carry
    through untouched), then the *rung* lines joining that level's points to the
    previous level's are appended (their ``interior`` is the straight GLL blend
    between the two corner points).  A quad is then just its four line indices plus
    four ``flip`` bits -- no ``unique_edges`` dedupe and no owner-wins
    ``scatter_edge_nodes`` reconciliation, because no shared edge is ever
    duplicated in the first place."""
    slices = list(slices)
    if loop:
        reject_loop_caps("QuadMesh.loft", first_tag, last_tag)
    lines = np.asarray(slices[0].lines, dtype=np.int64).reshape(-1, 2)
    L = lines.shape[0]
    n_prof = len(slices)
    # periodic: profile M-1 sweeps back onto profile 0, so there are M layers.
    nz = n_prof if loop else n_prof - 1
    S = np.stack([np.asarray(s.points, dtype=float).reshape(-1, 3)
                  for s in slices], axis=0)              # (n_prof, nn, 3)
    nn = S.shape[1]
    points = S.reshape(n_prof * nn, 3)                   # global id = i*nn + v

    order = slices[0].order
    if any(s.order != order for s in slices):
        raise ValueError("loft: all slices must share the same order")
    ho = order > 1

    a = lines[:, 0]
    b = lines[:, 1]
    # only points actually carried by a profile line get a rung (an isolated point
    # borders no quad, so a rung there would be a dangling edge).
    used: IntArray = (np.unique(lines.ravel()) if L
                      else np.zeros(0, dtype=np.int64))
    nu = used.shape[0]
    rung_slot: IntArray = np.full(nn, -1, dtype=np.int64)
    rung_slot[used] = np.arange(nu, dtype=np.int64)

    # -- grow the global edge LineMesh layer by layer -------------------
    # level i's profile lines occupy [prof_off[i], prof_off[i]+L); the rungs of
    # transition i->next(i) occupy [rung_off[i], rung_off[i]+nu).  ``nxt`` is the
    # profile each layer sweeps *to*: i+1 normally, wrapping to 0 for the closing
    # layer of a periodic sweep -- the single extra iteration below.
    nxt: IntArray = (np.arange(1, nz + 1, dtype=np.int64) % n_prof if nz
                     else np.zeros(0, dtype=np.int64))
    gi: FloatArray = gll_nodes(order)[1:order] if ho else np.zeros(0)
    prof_off: IntArray = np.empty(n_prof, dtype=np.int64)
    rung_off: IntArray = np.empty(max(nz, 0), dtype=np.int64)
    line_rows: list[IntArray] = []
    inter_rows: list[PointArray] = []
    cur = 0

    def _append_rung(layer: int) -> None:
        """Append the rung block of layer ``layer`` (profile ``layer`` ->
        ``nxt[layer]``) to the growing edge mesh -- appended exactly once per
        layer, so the seam rung of a periodic sweep is never duplicated."""
        nonlocal cur
        j = int(nxt[layer])
        rung_off[layer] = cur
        line_rows.append(np.column_stack([layer * nn + used, j * nn + used]))
        if ho:
            lo, hi = S[layer, used, :], S[j, used, :]
            inter_rows.append(lo[:, None, :]
                              + gi[None, :, None] * (hi - lo)[:, None, :])
        cur += nu

    for i in range(n_prof):
        prof_off[i] = cur                                # merge level i's LineMesh
        line_rows.append(lines + i * nn)
        if ho:
            inter_rows.append(np.asarray(slices[i].interior, dtype=float))
        cur += L
        if i:                                            # rungs (i-1) -> i
            _append_rung(i - 1)
    if loop and nz:
        # the one extra iteration: the closing rung (M-1) -> 0.  No profile is
        # re-appended, so the seam shares profile 0's lines and nodes outright.
        _append_rung(nz - 1)
    all_lines: IntArray = (np.concatenate(line_rows, axis=0) if line_rows
                           else np.zeros((0, 2), dtype=np.int64))
    edge_nodes: PointArray | None = (
        np.concatenate(inter_rows, axis=0) if ho else None)

    # -- quads purely by line index -------------------------------------
    # quad row for (layer i, line l) = i*L + l; corners [a_i, b_i, b_{i+1}, a_{i+1}].
    # local edge 1 = level-i profile line l (forward), 2 = rung at b (forward),
    # 3 = level-(i+1) profile line l (reversed), 4 = rung at a (reversed).
    i_idx: IntArray = np.repeat(np.arange(nz, dtype=np.int64), L)
    l_idx = np.tile(np.arange(L, dtype=np.int64), nz)
    j_idx: IntArray = nxt[i_idx] if nz else i_idx        # the swept-to profile
    av = a[l_idx]
    bv = b[l_idx]
    quad: IntArray = np.stack([
        prof_off[i_idx] + l_idx,
        rung_off[i_idx] + rung_slot[bv],
        prof_off[j_idx] + l_idx,
        rung_off[i_idx] + rung_slot[av]], axis=1)
    flip: BoolArray = np.tile(
        np.array([False, False, True, True]), (nz * L, 1))
    etags: StrArray = np.asarray(slices[0].element_tags, dtype=np.str_)[l_idx]

    # order-N: each column quad is a transfinite (Coons) patch -- curved along the
    # profile line (from the slices' own points + private interior nodes), straight
    # along the sweep between consecutive slices.  Its boundary already lives on the
    # merged/rung lines, so the patch is evaluated only at the private interior
    # slots.
    interior: PointArray | None = None
    if ho:
        g = gll_nodes(order)
        row = order + 1
        # each profile's own high-order curve, assembled natively from the shared
        # corner points and that profile's private interior nodes
        Scur: PointArray = np.empty((n_prof, L, row, 3), dtype=float)
        Scur[:, :, 0, :] = S[:, a, :]
        Scur[:, :, order, :] = S[:, b, :]
        Scur[:, :, 1:order, :] = np.stack(
            [np.asarray(s.interior, dtype=float) for s in slices], axis=0)
        bottom = Scur[i_idx, l_idx]                     # (Q,row,3) a->b at i
        top = Scur[j_idx, l_idx]                        # (Q,row,3) a->b at next(i)
        a_lo, a_hi = S[i_idx, av], S[j_idx, av]         # (Q,3) sweep at a
        b_lo, b_hi = S[i_idx, bv], S[j_idx, bv]         # (Q,3) sweep at b
        gg = g[None, :, None]
        left = a_lo[:, None, :] + gg * (a_hi - a_lo)[:, None, :]   # (Q,row,3)
        right = b_lo[:, None, :] + gg * (b_hi - b_lo)[:, None, :]
        islots = _quad_interior_slots(order)
        interior = _coons_at(bottom, top, left, right, g,
                             islots % row, islots // row)

    # tagged boundary point -> swept wall edge: vertex 0 -> side 4, vertex 1 -> 2
    sec_b = np.asarray(slices[0].boundaries, dtype=np.int64).reshape(-1, 2)
    sec_t = slices[0].boundary_tags
    bnd: list[list[int]] = []
    names: list[str] = []
    for r in range(sec_b.shape[0]):
        tag = str(sec_t[r])
        if tag == NO_BOUNDARY:
            continue
        l0 = int(sec_b[r, 0])
        qside = 4 if int(sec_b[r, 1]) == 1 else 2
        for ii in range(nz):
            bnd.append([ii * L + l0, qside])
            names.append(tag)
    # caps: scalar tags the whole cap, an array tags per section line.  A periodic
    # sweep has no near/far cap edge at all, so it emits no cap row (and rejected
    # the tags above).
    if not loop:
        first_caps = QuadMesh._cap_tags(first_tag, L)
        last_caps = QuadMesh._cap_tags(last_tag, L)
        for l0 in range(L):
            if first_caps[l0]:
                bnd.append([l0, 1])
                names.append(first_caps[l0])
        if nz:
            for l0 in range(L):
                if last_caps[l0]:
                    bnd.append([(nz - 1) * L + l0, 3])
                    names.append(last_caps[l0])
    b_ord, n_ord = QuadMesh._order_bnd(bnd, names)
    lm = LineMesh(points, all_lines, order=order, interior=edge_nodes)
    return QuadMesh(lm, quad, flip, interior, b_ord, n_ord,
               element_tags=etags, order=order)

def merge(meshes: list[QuadMesh], *, tol: float | None = None) -> QuadMesh:
    """Merge quad sections into one, welding coincident boundary points.
    ``tol`` is the absolute coincidence distance (default ``1e-7`` x the extent).
    Tagged ``boundaries`` and dense ``element_tags`` concatenate with each
    block's quad ids offset; an interior seam is not auto-dropped."""
    meshes = list(meshes)
    pos = [np.asarray(m.points, dtype=float).reshape(-1, 3) for m in meshes]
    counts = [p.shape[0] for p in pos]
    P = np.concatenate(pos, axis=0) if pos else np.zeros((0, 3))
    total = P.shape[0]

    remap = np.arange(total, dtype=np.int64)
    is_bnd: BoolArray = np.zeros(total, dtype=bool)
    noff = 0
    for m, c in zip(meshes, counts):
        edges, mask = _boundary_mask(m.quads)
        is_bnd[noff + np.unique(edges[mask])] = True
        noff += c
    bidx = np.flatnonzero(is_bnd)
    if bidx.size:
        scl = float(np.max(P.max(axis=0) - P.min(axis=0)))
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

    quad_list, bnd_list, name_list, etag_list = [], [], [], []
    noff = qoff = 0
    for m, c in zip(meshes, counts):
        quad_list.append(point_id[m.quads + noff])   # local -> welded id
        etag_list.append(m.element_tags)
        if m.boundaries.shape[0]:
            b: IntArray = m.boundaries.copy()
            b[:, 0] += qoff                          # shift quad ids; sides local
            bnd_list.append(b)
            name_list.append(m.boundary_tags)
        noff += c
        qoff += m.n_quads
    quads = np.concatenate(quad_list, axis=0) if quad_list else np.zeros((0, 4), np.int64)
    etags = (np.concatenate(etag_list) if etag_list
             else np.empty(0, dtype=np.str_))
    bnd = np.concatenate(bnd_list, axis=0) if bnd_list else np.zeros((0, 2), np.int64)
    names = np.concatenate(name_list) if name_list else np.empty(0, dtype=np.str_)
    b_ord, n_ord = QuadMesh._order_bnd(bnd, names)

    # order-N: the private per-quad interiors just concatenate, but the shared edge
    # tables must be rebuilt against the *merged* topology -- gather each block's
    # nodes into its own element traversal order, concatenate in merged element
    # order, then re-scatter.  That scatter is the conformal-weld guard: two blocks
    # that disagree on a welded shared edge raise instead of silently welding.
    order = meshes[0].order if meshes else 1
    if any(m.order != order for m in meshes):
        raise ValueError("merge: all sections must share the same order")
    edges, elem_edges, flip = conform.unique_edges(quads, 2)
    edge_nodes: PointArray | None = None
    interior: PointArray | None = None
    if order > 1:
        local: PointArray = np.concatenate(
            [conform.gather_edge_nodes(m.lines.interior, m.quad, m.flip)
             for m in meshes], axis=0)                     # (Q,4,order-1,3)
        edge_nodes = conform.scatter_edge_nodes(
            local, elem_edges, flip, edges.shape[0],
            conform.entity_tol(points), "QuadMesh.merge")
        interior = np.concatenate([m.interior for m in meshes], axis=0)
    lm = LineMesh(points, edges, order=order, interior=edge_nodes)
    return QuadMesh(lm, elem_edges, flip, interior, b_ord, n_ord,
               element_tags=etags, order=order)


#: Variable-arity combinators bound onto ``QuadMesh`` as ``staticmethod``.
FACTORIES: dict[str, Any] = {
    "loft": loft,
    "merge": merge,
}
