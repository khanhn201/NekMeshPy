"""Variable-arity ``HexMesh`` operations -- the only ones that build a numbering.

``loft`` (``n`` quad sections -> a block, rung delta +1) and ``merge`` (``n`` blocks ->
one, rung delta 0) are the two n-ary operations at this rung, and the only code here
that manufactures a global point/element index space from scratch: ``loft`` numbers the
swept corner table (global id ``i*nn + v``), ``merge`` builds the ``remap`` /
``survivors`` / ``point_id`` tables of the weld.  Every fixed-arity operation either
reuses an existing numbering (``blend``) or delegates here (``extrude``, ``annulus``,
``from_grid``).

Both *rewrite* topology against a new corner numbering rather than merely generating
it, which is why both must re-scatter the shared edge **and** face nodes owner-wins and
verify every other incident copy -- unlike ``QuadMesh.loft``, which assembles its B-rep
layer by layer and never duplicates a shared entity in the first place.

Free functions bound onto :class:`~nekmeshpy.HexMesh` by ``hexmesh/__init__.py``;
internal toolkit code imports them from here directly rather than through the bound
``HexMesh.<name>`` sugar.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from .._typing import (
    BoolArray,
    IntArray,
    PointArray,
    StrArray,
)
from ..linemesh import LineMesh
from ..model import conform
from ..model.fields import gll_nodes, reject_loop_caps
from ..quadmesh import NO_BOUNDARY, QuadMesh
from ._query import _boundary_points
from .hexmesh import HexMesh, _slice_block, _sweep_at


def loft(
    slices: Sequence[QuadMesh],
    *,
    loop: bool = False,
    first_tag: str | Sequence[str] | StrArray = "",
    last_tag: str | Sequence[str] | StrArray = "",
) -> HexMesh:
    """Loft a stack of conformal quad profiles into a hex block (the general
    primitive behind ``extrude``, and the top rung of the uniform sweep shared
    with :meth:`LineMesh.loft <nekmeshpy.linemesh.LineMesh.loft>` and
    :meth:`QuadMesh.loft <nekmeshpy.quadmesh.QuadMesh.loft>`).

    ``slices`` is ``nz+1`` profiles sharing the same quad connectivity,
    ``boundary_tags``, and ``element_tags``; consecutive profiles form ``nz`` hex
    layers. ``first_tag`` names the first bottom cap (face 5), ``last_tag`` the
    last top cap (face 6) -- each a scalar or a per-quad array. Side faces are
    named from the section's ``boundary_tags`` (unnamed or ``NO_BOUNDARY`` edges
    stay untagged), and every hex inherits its quad's ``element_tags``. Points
    are shared by construction.

    ``loop=True`` makes the sweep **periodic**: the last profile is joined back to
    the *first*, so ``M`` profiles give ``M`` layers instead of ``M-1`` -- one
    extra layer whose top corners are profile 0's own points.  No profile is
    duplicated, so the seam faces are genuine shared entities (``unique_edges`` /
    ``canonical_faces`` resolve them from the shared corner ids) and the closed
    solid is watertight in the sweep direction, e.g. a solid torus lofted from
    disc sections.  A closed sweep has no bottom/top cap, so ``first_tag`` /
    ``last_tag`` with ``loop=True`` raise ``ValueError`` rather than being
    silently dropped, and no cap boundary row is emitted; side faces from the
    section's ``boundary_tags`` are unaffected."""
    slices = list(slices)
    if loop:
        reject_loop_caps("HexMesh.loft", first_tag, last_tag)
    quads = np.asarray(slices[0].quads, dtype=np.int64).reshape(-1, 4)
    # section (quad, side) -> name; each swept side face inherits its section edge
    sec_bnd = np.asarray(slices[0].boundaries, dtype=np.int64).reshape(-1, 2)
    sec_tags = slices[0].boundary_tags
    side_name: dict[tuple[int, int], str] = {
        (int(sec_bnd[r, 0]), int(sec_bnd[r, 1])): str(sec_tags[r])
        for r in range(sec_bnd.shape[0])}
    tag_sides = bool(side_name)
    qtag = np.asarray(slices[0].element_tags, dtype=np.str_).reshape(-1)
    M = quads.shape[0]
    n_prof = len(slices)
    # periodic: profile M-1 sweeps back onto profile 0, so there are M layers.
    nz = n_prof if loop else n_prof - 1
    # ``nxt[i]`` is the profile layer ``i`` sweeps *to*: i+1 normally, wrapping to
    # 0 for the closing layer of a periodic sweep.
    nxt: IntArray = (np.arange(1, nz + 1, dtype=np.int64) % n_prof if nz
                     else np.zeros(0, dtype=np.int64))
    S = np.stack([np.asarray(s.points, dtype=float).reshape(-1, 3)
                  for s in slices], axis=0)             # (n_prof, nn, 3)
    nn = S.shape[1]
    points = S.reshape(n_prof * nn, 3)                   # global id = i*nn + v

    # Decide handedness once from the first layer and flip the quad template if
    # left-handed; reject a mixed-winding section rather than invert elements.
    signs = np.array([HexMesh._signed_vol(np.vstack([S[0, quads[q], :], S[1, quads[q], :]]))
                      for q in range(M)]) if nz else np.zeros(0)
    if nz and not (np.all(signs > 0) or np.all(signs < 0)):
        raise ValueError(
            "extrude: section is not consistently wound (mixed hex "
            "orientation) -- the section mesher must emit uniform winding")
    flip = bool(nz and signs[0] < 0)
    qw = quads[:, [0, 3, 2, 1]] if flip else quads

    # caps stay faces 5/6 by q (the flip only reorders a quad's 4 corners); a
    # periodic sweep has no cap at all, so it emits none (and rejected the tags).
    first_caps = ([""] * M if loop else HexMesh._cap_tags(first_tag, M))
    last_caps = ([""] * M if loop else HexMesh._cap_tags(last_tag, M))

    hexes = np.empty((nz * M, 8), dtype=np.int64)
    # every layer repeats the section's per-quad tags (hex ``e = i*M + q``).  Tiling
    # keeps the section's own string width: ``np.empty(..., dtype=np.str_)`` is
    # ``<U1`` and would clip each tag to its first character on assignment.
    etags: StrArray = (np.tile(qtag, nz) if qtag.size
                       else np.full(nz * M, "", dtype=np.str_))
    bnd: list[list[int]] = []
    names: list[str] = []
    e = 0
    for i in range(nz):
        j = int(nxt[i])                     # the profile this layer sweeps to
        for q in range(M):
            v = qw[q, :]
            hexes[e] = np.concatenate([i * nn + v, j * nn + v])
            if tag_sides:
                # section side s -> hex face s, or 5-s when the quad was flipped
                for s in (1, 2, 3, 4):
                    nm = side_name.get((q, s))
                    if nm is None or nm == NO_BOUNDARY:
                        continue
                    bnd.append([e, (5 - s) if flip else s])
                    names.append(nm)
            if i == 0 and first_caps[q]:
                bnd.append([e, 5])
                names.append(first_caps[q])
            if i == nz - 1 and last_caps[q]:
                bnd.append([e, 6])
                names.append(last_caps[q])
            e += 1

    # order-N: each hex column is a straight GLL sweep between the two bounding
    # profiles' in-plane blocks, evaluated only at the entity slots -- the two
    # z-face interiors come from the slices' own quad interiors, the four side-face
    # interiors and the cell interior are swept from the slices' edge / quad nodes,
    # the in-slice edges are the slices' shared edge nodes and the vertical edges
    # straight corner blends.  ``scatter_*`` then canonicalizes.
    order = slices[0].order
    if any(s.order != order for s in slices):
        raise ValueError("loft: all slices must share the same order")
    edges, elem_edges, eflip = conform.unique_edges(hexes, 3)
    canonical_conn, elem_faces, face_orient = conform.canonical_faces(hexes)
    edge_nodes: PointArray | None = None
    face_nodes: PointArray | None = None
    interior: PointArray | None = None
    if order > 1:
        g = gll_nodes(order)
        row = order + 1
        m2 = row * row
        SC = np.stack([_slice_block(s, order) for s in slices], axis=0)
        # (nz, M, m2, 3) in-plane blocks of the bottom/top slice of each layer,
        # flattened to hex order e = i*M + q.  ``nxt`` picks the top slice, so the
        # periodic closing layer sweeps back onto profile 0's own block.
        bottom = SC[np.arange(nz, dtype=np.int64)].reshape(nz * M, m2, 3)
        top = SC[nxt].reshape(nz * M, m2, 3)
        if flip:
            kk = np.arange(m2)
            trans = (kk // row) + row * (kk % row)    # transpose the in-plane grid
            bottom = bottom[:, trans, :]
            top = top[:, trans, :]
        E = nz * M
        k2 = (order - 1) ** 2
        eslots = conform._edge_slots(3, order)[:, 1:-1]         # (12, order-1)
        local_e = _sweep_at(bottom, top, g, eslots.ravel(), m2).reshape(
            E, 12, order - 1, 3)
        fslots = conform._face_interior_slots(order)            # (6, k2)
        local_f = _sweep_at(bottom, top, g, fslots.ravel(), m2).reshape(
            E, 6, k2, 3)
        interior = _sweep_at(bottom, top, g,
                             conform._interior_slots(3, order), m2)
        tol = conform.entity_tol(points)
        edge_nodes = conform.scatter_edge_nodes(
            local_e, elem_edges, eflip, edges.shape[0], tol, "HexMesh.loft")
        face_nodes = conform.scatter_face_nodes(
            local_f, elem_faces, face_orient, canonical_conn.shape[0], tol,
            "HexMesh.loft")
    # the hex edge table unique_edges(hexes, 3) and the shared-face table
    # unique_edges(canonical_conn, 2) are the same array (both canonicalize
    # min-corner-id first over the same global corner ids), so ``edge_nodes``
    # scattered with the hex incidence indexes the shared-face QuadMesh directly.
    q_edges, q_elem_edges, q_flip = conform.unique_edges(canonical_conn, 2)
    edge_lm = LineMesh(points, q_edges, order=order, interior=edge_nodes)
    faces = QuadMesh(edge_lm, q_elem_edges, q_flip, face_nodes, order=order)
    return HexMesh(faces, elem_faces, face_orient, interior,
               *HexMesh._order_bnd(bnd, names),
               element_tags=etags, order=order)

def merge(
    meshes: Sequence[HexMesh],
    *,
    tol: float | None = None,
) -> HexMesh:
    """Stitch several hex blocks into one, coordinate-welding coincident seam
    points in a single pass.  ``tol`` is the absolute coincidence distance
    (default ``1e-7`` x the merged bounding-box extent).

    Only points on each block's domain boundary (faces carried by a single hex)
    are weld candidates; interior points are always kept distinct."""
    meshes = list(meshes)
    pos = [m.points for m in meshes]
    counts = [p.shape[0] for p in pos]
    P = np.concatenate(pos, axis=0) if pos else np.zeros((0, 3))
    total = P.shape[0]

    # remap: concat point index -> representative concat index (self by default)
    remap = np.arange(total, dtype=np.int64)
    is_bnd: BoolArray = np.zeros(total, dtype=bool)
    noff = 0
    for m, c in zip(meshes, counts):
        is_bnd[noff + _boundary_points(m.hexes)] = True
        noff += c
    bidx = np.flatnonzero(is_bnd)
    if bidx.size:
        scl = float(np.max(P.max(axis=0) - P.min(axis=0)))
        t = tol if tol is not None else (1e-7 * scl if scl > 0 else 1.0)
        keys = np.round(P[bidx, :] / t).astype(np.int64)
        _, first_local, inverse = np.unique(
            keys, axis=0, return_index=True, return_inverse=True)
        remap[bidx] = bidx[first_local][inverse.ravel()]

    survivors = np.unique(remap)                    # concat indices kept
    new_id: IntArray = np.empty(total, dtype=np.int64)
    new_id[survivors] = np.arange(survivors.size)
    point_id = new_id[remap]                         # concat index -> final id
    points = P[survivors, :]

    hex_list, bnd_list, name_list, etag_list = [], [], [], []
    noff = eoff = 0
    for m, c in zip(meshes, counts):
        hex_list.append(point_id[m.hexes + noff])    # local -> concat -> welded id
        etag_list.append(np.asarray(m.element_tags, dtype=np.str_).reshape(-1))
        if m.boundaries.shape[0]:
            b: IntArray = m.boundaries.copy()
            b[:, 0] += eoff
            bnd_list.append(b)
            name_list.append(m.boundary_tags)
        noff += c
        eoff += m.hexes.shape[0]
    hexes = (np.concatenate(hex_list, axis=0) if hex_list
             else np.zeros((0, 8), np.int64))
    etags = (np.concatenate(etag_list) if etag_list
             else np.empty(0, dtype=np.str_))
    bnd = np.concatenate(bnd_list, axis=0) if bnd_list else np.zeros((0, 2), np.int64)
    names = (np.concatenate(name_list) if name_list
             else np.empty(0, dtype=np.str_))
    # order-N: the private per-hex interiors just concatenate, but the shared edge /
    # face tables must be rebuilt against the *merged* topology -- gather each
    # block's nodes into its own element-local order, concatenate in merged element
    # order, then re-scatter.  Those scatters are the conformal-weld guard: two
    # blocks that disagree on a welded shared edge / face raise instead of silently
    # welding.
    order = meshes[0].order if meshes else 1
    if any(mm.order != order for mm in meshes):
        raise ValueError("merge: all blocks must share the same order")
    edges, elem_edges, eflip = conform.unique_edges(hexes, 3)
    canonical_conn, elem_faces, face_orient = conform.canonical_faces(hexes)
    edge_nodes: PointArray | None = None
    face_nodes: PointArray | None = None
    interior: PointArray | None = None
    if order > 1:
        local_e: PointArray = np.concatenate(
            [conform.gather_edge_nodes(mm.quads.lines.interior, mm._elem_edges,
                                       mm._edge_flip)
             for mm in meshes], axis=0)                    # (E,12,order-1,3)
        local_f: PointArray = np.concatenate(
            [conform.gather_face_nodes(mm.quads.interior, mm.hex, mm.face_orient)
             for mm in meshes], axis=0)                    # (E,6,(order-1)**2,3)
        tol = conform.entity_tol(points)
        edge_nodes = conform.scatter_edge_nodes(
            local_e, elem_edges, eflip, edges.shape[0], tol, "HexMesh.merge")
        face_nodes = conform.scatter_face_nodes(
            local_f, elem_faces, face_orient, canonical_conn.shape[0], tol,
            "HexMesh.merge")
        interior = np.concatenate([mm.interior for mm in meshes], axis=0)
    q_edges, q_elem_edges, q_flip = conform.unique_edges(canonical_conn, 2)
    edge_lm = LineMesh(points, q_edges, order=order, interior=edge_nodes)
    faces = QuadMesh(edge_lm, q_elem_edges, q_flip, face_nodes, order=order)
    return HexMesh(faces, elem_faces, face_orient, interior,
               *HexMesh._order_bnd(bnd, names),
               element_tags=etags, order=order)


#: Variable-arity combinators bound onto ``HexMesh`` as ``staticmethod``.
FACTORIES: dict[str, Any] = {
    "loft": loft,
    "merge": merge,
}
