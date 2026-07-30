"""Entity-based conformal high-order storage.

The corner layer is already conformal: a shared corner is one node in ``points (P,3)``
referenced by every incident element's connectivity.  This module gives the high-order
(edge / face / interior) nodes the same property.  Instead of a per-element
``(E,(N+1)^d,3)`` block that stores a shared edge's ``N-1`` interior nodes once per
incident element, the non-corner nodes are decomposed by **topology** into

* **edges** -- unique undirected edges (canonical: min-corner-id first) with their
  ``N-1`` shared interior nodes, plus a per-element incidence + a *flip* bit when an
  element traverses an edge anti-canonically,
* **faces** (hex only) -- unique faces with their ``(N-1)**2`` shared interior nodes,
  plus a per-element incidence + a D4 orientation code (Phase 2),
* **interior** -- the per-element private nodes (quad ``(N-1)**2``, hex ``(N-1)**3``,
  line ``N-1``) that are never shared.

Sharing is decided by corner ids (**structural / exact** conformality), not by a
coordinate search.  :func:`split` scatters a full block into these tables, taking the
owning (lowest-id) element's nodes for each shared entity and **verifying** the other
incident copies agree within a scale-relative tolerance (a non-conforming input becomes
a loud error, not a silent weld).  :func:`assemble` is its inverse, gathering the full
``(E,(N+1)^d,3)`` block back so the container ``.curved`` contract is unchanged.
:func:`to_conformal` exposes the conformal view directly: one global ``nodes (M,3)`` plus
dense ``conn (E,(N+1)^d)`` -- the high-order analog of ``points`` + ``quads``.

At ``order == 1`` every table is empty and :func:`assemble` returns just ``points[conn]``,
so the order-1 path is byte-identical to the plain corner mesh.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .._typing import BoolArray, FloatArray, IntArray, PointArray
from .interp import (
    _CORNER_IJK,
    corner_indices,
    hex_edge_indices,
    nodes_per_element,
    quad_edge_indices,
)

#: Local edges as corner-index pairs, per dim.  dim 1 has none (a line element *is* a
#: 1-cell -- two lines sharing an endpoint share only that corner, never edge-interior
#: nodes).  dim 2 matches ``QuadMesh.EDGE_POINTS``; dim 3 matches the hex 12-edge table.
_LOCAL_EDGES: dict[int, IntArray] = {
    2: np.array([[0, 1], [1, 2], [2, 3], [3, 0]], dtype=np.int64),
    3: np.array([[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4],
                 [0, 4], [1, 5], [2, 6], [3, 7]], dtype=np.int64),
}

#: Hex local faces as corner-index quads, matching ``HexMesh.FACE_POINTS``.
_LOCAL_FACES: IntArray = np.array(
    [[0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7],
     [0, 1, 2, 3], [4, 5, 6, 7]], dtype=np.int64)


@dataclass(frozen=True)
class EntityTables:
    """Topological decomposition of a container's non-corner high-order nodes.

    Small immutable value stored on each container in place of the old per-element
    ``_curved_interior``.  All node coordinates are absolute (not deltas); corners are
    **not** stored here -- they stay owned by ``points[conn]`` -- so
    :func:`assemble` re-derives them on every read and an in-place ``points`` edit is
    reflected automatically.  Empty (zero-width) tables at ``order == 1``."""

    order: int
    dim: int
    #: ``(Ne,2)`` canonical (min-first) corner-id pairs; ``(0,2)`` for dim 1.
    edges: IntArray
    #: ``(Ne, order-1, 3)`` shared edge-interior nodes, canonical (min->max) order.
    edge_nodes: FloatArray
    #: ``(E, n_local_edges)`` edge id per element-local edge; ``(E,0)`` for dim 1.
    elem_edges: IntArray
    #: ``(E, n_local_edges)`` bool, True where the element traverses the edge max->min.
    edge_flip: BoolArray
    #: ``(Nf,4)`` canonical face corner ids (hex only; ``(0,4)`` otherwise) -- Phase 2.
    faces: IntArray
    #: ``(Nf,(order-1)**2,3)`` shared face-interior nodes (hex only) -- Phase 2.
    face_nodes: FloatArray
    #: ``(E,6)`` face id per hex face (``(E,0)`` otherwise) -- Phase 2.
    elem_faces: IntArray
    #: ``(E,6)`` D4 orientation code per hex face (``(E,0)`` otherwise) -- Phase 2.
    face_orient: IntArray
    #: ``(E, k, 3)`` per-element private interior nodes (never shared).
    interior: FloatArray


# -- reference-lattice helpers ------------------------------------------
def _lattice(dim: int, order: int) -> IntArray:
    """``(M, dim)`` integer lattice coords ``(i,j,k)`` of the block nodes, in the same
    lexicographic (``i`` fastest) order the curved block uses."""
    row = order + 1
    idx = np.arange(row ** dim)
    return np.stack([(idx // (row ** a)) % row for a in range(dim)], axis=1)


def _edge_slots(dim: int, order: int) -> IntArray:
    """``(n_local_edges, order+1)`` block slots along each local edge, start->end."""
    if dim == 2:
        return np.stack([quad_edge_indices(s, order) for s in (1, 2, 3, 4)])
    if dim == 3:
        return np.stack([hex_edge_indices(e, order) for e in range(12)])
    raise ValueError("edges are only defined for dim 2 or 3, got %d" % dim)


def _interior_slots(dim: int, order: int) -> IntArray:
    """Block slots of the private interior nodes -- those strictly interior on **every**
    axis (line ``N-1``; quad ``(N-1)**2``; hex ``(N-1)**3``)."""
    c = _lattice(dim, order)
    mask: BoolArray = np.all((c > 0) & (c < order), axis=1)
    return np.flatnonzero(mask).astype(np.int64)


# -- hex face geometry (dim 3) ------------------------------------------
def _face_axes() -> list[tuple[int, int, int, int, IntArray]]:
    """Per hex face: ``(pinned_axis, pin, u_ax, v_ax, corner_uv (4,2))`` -- the axis held
    constant on the face and its ``0/1`` value, the two in-face axes (ascending), and the
    ``{0,1}`` ``(u,v)`` position of each of the face's four :data:`_LOCAL_FACES` corners.
    Defines the element-local ``(u,v)`` frame used for the D4 orientation."""
    meta: list[tuple[int, int, int, int, IntArray]] = []
    for f in range(6):
        t = np.array([_CORNER_IJK[3][c] for c in _LOCAL_FACES[f]])   # (4,3) in {0,1}
        pinned = next(a for a in range(3) if len(set(t[:, a].tolist())) == 1)
        pin = int(t[0, pinned])
        u_ax, v_ax = (a for a in range(3) if a != pinned)            # ascending
        meta.append((pinned, pin, u_ax, v_ax, t[:, [u_ax, v_ax]].astype(np.int64)))
    return meta


def _face_interior_slots(order: int) -> IntArray:
    """``(6, (order-1)**2)`` block slots of each hex face's strictly-interior nodes, in
    the element ``(u,v)`` frame with ``u`` fastest (matches the D4 permutation frame)."""
    n = order
    strides = (1, n + 1, (n + 1) ** 2)
    out: IntArray = np.empty((6, (n - 1) ** 2), dtype=np.int64)
    for f, (pinned, pin, u_ax, v_ax, _) in enumerate(_face_axes()):
        m = 0
        for vv in range(1, n):
            for uu in range(1, n):                       # u fastest
                coord = [0, 0, 0]
                coord[pinned] = pin * n
                coord[u_ax] = uu
                coord[v_ax] = vv
                out[f, m] = sum(coord[a] * strides[a] for a in range(3))
                m += 1
    return out


def _d4_apply(a: int, b: int, code: int, s: int) -> tuple[int, int]:
    """Apply D4 symmetry ``code`` (0-7 = reflect bit + 2 rotation bits) to grid coords
    ``(a,b)`` on a ``[0,s]`` square.  Scale-independent in ``code``: the same ``code``
    is fitted at corner scale (``s=1``) and replayed at interior scale (``s=order-2``)."""
    if code & 1:
        a, b = b, a
    for _ in range(code >> 1):
        a, b = b, s - a
    return a, b


def _perm_tables(order: int) -> tuple[IntArray, IntArray]:
    """``(P (8,k*k), INV (8,k*k))`` with ``k = order-1``: for each D4 ``code``, the
    permutation ``P[code]`` mapping an element-frame interior-flat index (``u`` fastest)
    to its canonical-frame index, and its inverse ``INV[code]``."""
    k = order - 1
    p: IntArray = np.empty((8, k * k), dtype=np.int64)
    for code in range(8):
        for m in range(k * k):
            a2, b2 = _d4_apply(m % k, m // k, code, k - 1)
            p[code, m] = a2 + k * b2
    return p, np.argsort(p, axis=1)


def _face_code(ids: IntArray, uv: IntArray) -> int:
    """D4 ``code`` taking an element-local face frame to the canonical frame defined by
    the four corner ``ids``: origin = min id -> ``(0,0)``, its smaller-id neighbour ->
    ``(1,0)``, other neighbour -> ``(0,1)``, opposite -> ``(1,1)``.  Two hexes sharing a
    face compute the same canonical frame (same ids), so their stored nodes coincide."""
    origin = int(np.argmin(ids))
    o = uv[origin]
    nb = [i for i in range(4)
          if i != origin and int(np.sum(uv[i] != o)) == 1]           # edge-neighbours
    opp = next(i for i in range(4) if i != origin and i not in nb)
    nb.sort(key=lambda i: int(ids[i]))
    canon = {origin: (0, 0), nb[0]: (1, 0), nb[1]: (0, 1), opp: (1, 1)}
    for code in range(8):
        if all(_d4_apply(int(uv[i, 0]), int(uv[i, 1]), code, 1) == canon[i]
               for i in range(4)):
            return code
    raise AssertionError("no D4 code matches face orientation")   # pragma: no cover


def unique_faces(hexes: IntArray) -> tuple[IntArray, IntArray, IntArray]:
    """Deduplicate hex faces into unique faces plus a per-hex incidence and D4 code.

    Returns ``(faces (Nf,4), elem_faces (E,6), face_orient (E,6))`` where ``faces`` are
    canonical (sorted corner ids), ``elem_faces[e,f]`` is the unique-face id of hex
    ``e``'s local face ``f``, and ``face_orient[e,f]`` is the D4 code (:func:`_face_code`)
    taking that hex's local face frame to the shared canonical frame."""
    e = hexes.shape[0]
    corner_uv = [m[4] for m in _face_axes()]                # per-face (4,2) in {0,1}
    ids = hexes[:, _LOCAL_FACES]                            # (E,6,4)
    key = np.sort(ids, axis=2).reshape(e * 6, 4)
    uniq, inv = np.unique(key, axis=0, return_inverse=True)
    elem_faces = inv.reshape(e, 6).astype(np.int64)
    orient = np.empty((e, 6), dtype=np.int64)
    for ei in range(e):
        for f in range(6):
            orient[ei, f] = _face_code(ids[ei, f], corner_uv[f])
    return uniq.astype(np.int64), elem_faces, orient


#: CCW corner ``(u,v)`` order of a canonical face frame; slot ``i`` is corner ``i``.
_CANON_UV: list[tuple[int, int]] = [(0, 0), (1, 0), (1, 1), (0, 1)]
_QIDX: dict[tuple[int, int], int] = {uv: i for i, uv in enumerate(_CANON_UV)}


def _canon_qidx() -> IntArray:
    """``(6, 8, 4)`` table: for hex local face ``f``, D4 ``code``, and the face's local
    corner ``p``, the CCW slot (0-3) that corner occupies in the canonical face frame
    (``_d4_apply`` of its element-local ``(u,v)``).  Precomputed so
    :func:`canonical_faces` / :func:`hex_corners_from_faces` are pure array gathers."""
    corner_uv = [m[4] for m in _face_axes()]                # per-face (4,2) in {0,1}
    out: IntArray = np.empty((6, 8, 4), dtype=np.int64)
    for f in range(6):
        for code in range(8):
            for p in range(4):
                u, v = int(corner_uv[f][p, 0]), int(corner_uv[f][p, 1])
                out[f, code, p] = _QIDX[_d4_apply(u, v, code, 1)]
    return out


_CANON_QIDX: IntArray = _canon_qidx()


def canonical_faces(hexes: IntArray) -> tuple[IntArray, IntArray, IntArray]:
    """CCW-connectivity sibling of :func:`unique_faces`.

    Returns ``(canonical_conn (Nf,4), elem_faces (E,6), face_orient (E,6))`` where
    ``canonical_conn`` is the **CCW corner connectivity** of each unique face in its
    canonical D4 frame (corner order ``(0,0)->(1,0)->(1,1)->(0,1)``), suitable directly
    as the ``quads`` of a shared-face :class:`~nekmeshpy.QuadMesh`; ``elem_faces`` and
    ``face_orient`` are exactly as :func:`unique_faces` returns them.  The owner (lowest
    incidence) hex of each unique face supplies its corner ids."""
    _, elem_faces, face_orient = unique_faces(hexes)
    flat = elem_faces.ravel()
    fids, first = np.unique(flat, return_index=True)         # owner = lowest flat index
    nf = fids.shape[0]
    owner_e = first // 6
    owner_f = first % 6
    codes = face_orient.ravel()[first]
    canonical_conn: IntArray = np.empty((nf, 4), dtype=np.int64)
    for f in range(6):
        sel = owner_f == f
        if not np.any(sel):
            continue
        ids = fids[sel]
        corner_ids = hexes[owner_e[sel][:, None], _LOCAL_FACES[f]]   # (nsel,4)
        q = _CANON_QIDX[f][codes[sel]]                               # (nsel,4)
        for p in range(4):
            canonical_conn[ids, q[:, p]] = corner_ids[:, p]
    return canonical_conn, elem_faces, face_orient


def hex_corners_from_faces(face_conn: IntArray, elem_faces: IntArray,
                           face_orient: IntArray) -> IntArray:
    """Recover ``hexes (E,8)`` (Nek corner order) from the shared-face connectivity.

    Inverse of :func:`canonical_faces`: each hex's 8 corners are read off its two ``z``
    faces (local faces 5/6 = corners ``[0,1,2,3]`` / ``[4,5,6,7]``) through the D4
    ``face_orient`` code, so the shared faces are the single source of truth for the
    corners (as ``points`` are for a linear mesh)."""
    e = elem_faces.shape[0]
    hexes: IntArray = np.empty((e, 8), dtype=np.int64)
    for f in (4, 5):
        q = _CANON_QIDX[f][face_orient[:, f]]                        # (E,4)
        hexes[:, _LOCAL_FACES[f]] = face_conn[elem_faces[:, f][:, None], q]
    return hexes


# -- topology -----------------------------------------------------------
def unique_edges(conn: IntArray, dim: int) -> tuple[IntArray, IntArray, BoolArray]:
    """Deduplicate the local edges of every element into unique undirected edges.

    Returns ``(edges (Ne,2), elem_edges (E,ne), edge_flip (E,ne))`` where ``edges`` are
    canonical (min-corner-id first), ``elem_edges[e,le]`` is the unique-edge id of
    element ``e``'s local edge ``le``, and ``edge_flip[e,le]`` is True when that element
    traverses the edge from the larger to the smaller corner id (i.e. anti-canonically).
    dim 1 has no shared edges -> empty tables."""
    e = conn.shape[0]
    if dim < 2:
        return (np.zeros((0, 2), np.int64), np.zeros((e, 0), np.int64),
                np.zeros((e, 0), bool))
    le = _LOCAL_EDGES[dim]                                  # (ne,2)
    ne = le.shape[0]
    a = conn[:, le[:, 0]]                                   # (E,ne) directed endpoints
    b = conn[:, le[:, 1]]
    lo = np.minimum(a, b)
    hi = np.maximum(a, b)
    pairs = np.stack([lo.ravel(), hi.ravel()], axis=1)      # (E*ne,2) canonical
    uniq, inv = np.unique(pairs, axis=0, return_inverse=True)
    elem_edges = inv.reshape(e, ne).astype(np.int64)
    edge_flip: BoolArray = (a > b)
    return uniq.astype(np.int64), elem_edges, edge_flip


# -- split / assemble / conformal view ----------------------------------
def _empty_tables(order: int, dim: int, e: int) -> EntityTables:
    k = max(order - 1, 0)
    return EntityTables(
        order=order, dim=dim,
        edges=np.zeros((0, 2), np.int64),
        edge_nodes=np.zeros((0, k, 3), float),
        elem_edges=np.zeros((e, 0), np.int64),
        edge_flip=np.zeros((e, 0), bool),
        faces=np.zeros((0, 4), np.int64),
        face_nodes=np.zeros((0, k * k, 3), float),
        elem_faces=np.zeros((e, 0), np.int64),
        face_orient=np.zeros((e, 0), np.int64),
        interior=np.zeros((e, 0, 3), float))


def _tol(points: PointArray) -> float:
    scale = (float(np.max(points.max(axis=0) - points.min(axis=0)))
             if points.size else 0.0)
    return 1e-9 * scale if scale > 0 else 1e-12


def split(order: int, curved: FloatArray | None, points: PointArray,
          conn: IntArray, dim: int, who: str) -> EntityTables:
    """Decompose a full ``(E,(order+1)**dim,3)`` block into :class:`EntityTables`.

    The entity-store analog of ``check_order_curved``: validates the block's shape and
    that its ``2**dim`` corner sub-slice equals ``points[conn]`` (scale-relative tol),
    then scatters the non-corner nodes into edge / interior tables.  For each shared
    edge the owning (lowest incidence) element's interior nodes are taken in canonical
    orientation and every other incident copy is **verified** to agree within tol (a
    non-conforming block raises ``ValueError``).  ``curved=None`` is allowed only at
    order 1 (empty tables)."""
    if order < 1:
        raise ValueError("%s: order must be >= 1, got %d" % (who, order))
    e = conn.shape[0]
    m = nodes_per_element(order, dim)
    if curved is None:
        if order > 1:
            raise ValueError("%s: order %d > 1 requires a curved block" % (who, order))
        return _empty_tables(order, dim, e)
    cb: FloatArray = np.asarray(curved, dtype=float)
    if cb.shape != (e, m, 3):
        raise ValueError(
            "%s: curved block must be (E,(N+1)^d,3) = (%d,%d,3), got %s"
            % (who, e, m, cb.shape))
    tol = _tol(points)
    corners = cb[:, corner_indices(order, dim), :]
    if not np.allclose(corners, points[conn], rtol=0.0, atol=tol):
        raise ValueError(
            "%s: curved block corners disagree with points[conn] "
            "(the 2^d corner nodes must equal the linear corners)" % who)
    if order == 1:
        return _empty_tables(order, dim, e)

    empty = _empty_tables(order, dim, e)
    if dim >= 2:
        edges, elem_edges, edge_flip = unique_edges(conn, dim)
        edge_nodes = _split_edges(cb, edges, elem_edges, edge_flip, dim, order,
                                  tol, who)
    else:                                              # dim 1: no shared edges
        edges, elem_edges, edge_flip = (empty.edges, empty.elem_edges,
                                        empty.edge_flip)
        edge_nodes = empty.edge_nodes
    if dim == 3:
        faces, elem_faces, face_orient = unique_faces(conn)
        face_nodes = _split_faces(cb, faces, elem_faces, face_orient, order, tol, who)
    else:                                              # dim 1/2: no shared faces
        faces, elem_faces, face_orient = (empty.faces, empty.elem_faces,
                                          empty.face_orient)
        face_nodes = empty.face_nodes
    interior = cb[:, _interior_slots(dim, order), :]
    return EntityTables(
        order=order, dim=dim, edges=edges, edge_nodes=edge_nodes,
        elem_edges=elem_edges, edge_flip=edge_flip,
        faces=faces, face_nodes=face_nodes,
        elem_faces=elem_faces, face_orient=face_orient,
        interior=interior)


def _split_edges(cb: FloatArray, edges: IntArray, elem_edges: IntArray,
                 edge_flip: BoolArray, dim: int, order: int, tol: float,
                 who: str) -> FloatArray:
    """Owner-wins + verify: gather each unique edge's ``order-1`` interior nodes in
    canonical order, checking all incident copies agree within ``tol``."""
    ne_uniq = edges.shape[0]
    islots = _edge_slots(dim, order)[:, 1:-1]                   # (ne, order-1)
    inc = cb[:, islots, :]                                      # (E,ne,order-1,3)
    inc = np.where(edge_flip[:, :, None, None], inc[:, :, ::-1, :], inc)  # canonical
    flat_eid = elem_edges.ravel()
    inc_flat = inc.reshape(flat_eid.shape[0], order - 1, 3)
    _, first = np.unique(flat_eid, return_index=True)          # owner per edge id
    edge_nodes: FloatArray = inc_flat[first]                   # (Ne,order-1,3)
    if not np.allclose(inc_flat, edge_nodes[flat_eid], rtol=0.0, atol=tol):
        raise ValueError(
            "%s: non-conforming high-order edge -- incident elements disagree on a "
            "shared edge's interior nodes beyond tolerance (%.3e). The inputs are not "
            "structurally conformal." % (who, tol))
    if edge_nodes.shape[0] != ne_uniq:                         # pragma: no cover
        raise AssertionError("%s: edge owner count mismatch" % who)
    return edge_nodes


def _split_faces(cb: FloatArray, faces: IntArray, elem_faces: IntArray,
                 face_orient: IntArray, order: int, tol: float,
                 who: str) -> FloatArray:
    """Owner-wins + verify for hex faces: gather each unique face's ``(order-1)**2``
    interior nodes in canonical (D4-normalized) order, checking incident copies agree."""
    k2 = (order - 1) ** 2
    fslots = _face_interior_slots(order)                       # (6, k2)
    _, inv = _perm_tables(order)                               # INV: elem->canon gather
    elem = cb[:, fslots, :]                                    # (E,6,k2,3) element order
    invp = inv[face_orient]                                    # (E,6,k2)
    canon = np.take_along_axis(elem, np.broadcast_to(
        invp[..., None], invp.shape + (3,)), axis=2)           # (E,6,k2,3) canonical
    flat_fid = elem_faces.ravel()
    canon_flat = canon.reshape(flat_fid.shape[0], k2, 3)
    _, first = np.unique(flat_fid, return_index=True)
    face_nodes: FloatArray = canon_flat[first]                 # (Nf,k2,3)
    if not np.allclose(canon_flat, face_nodes[flat_fid], rtol=0.0, atol=tol):
        raise ValueError(
            "%s: non-conforming high-order face -- incident hexes disagree on a shared "
            "face's interior nodes beyond tolerance (%.3e). The inputs are not "
            "structurally conformal." % (who, tol))
    if face_nodes.shape[0] != faces.shape[0]:                  # pragma: no cover
        raise AssertionError("%s: face owner count mismatch" % who)
    return face_nodes


def assemble(tables: EntityTables, points: PointArray,
             conn: IntArray) -> FloatArray:
    """Gather the full ``(E,(order+1)**dim,3)`` curved block from :class:`EntityTables`.

    The inverse of :func:`split` and the reader behind every container's ``.curved``
    property: corners come from ``points[conn]`` (single source of truth, so a live view
    of in-place ``points`` edits), edge-interior nodes from ``edge_nodes`` (reversed per
    ``edge_flip``), and the private ``interior`` verbatim.  At order 1 returns just the
    corners."""
    order, dim = tables.order, tables.dim
    e = conn.shape[0]
    m = nodes_per_element(order, dim)
    block: FloatArray = np.empty((e, m, 3), dtype=float)
    block[:, corner_indices(order, dim), :] = points[conn]
    if order == 1:
        return block
    block[:, _interior_slots(dim, order), :] = tables.interior
    if dim >= 2:
        islots = _edge_slots(dim, order)[:, 1:-1]                  # (ne, order-1)
        nodes = tables.edge_nodes[tables.elem_edges]               # (E,ne,order-1,3)
        nodes = np.where(tables.edge_flip[:, :, None, None],
                         nodes[:, :, ::-1, :], nodes)              # element order
        block[:, islots, :] = nodes
    if dim == 3:
        fslots = _face_interior_slots(order)                      # (6, k2)
        perm, _ = _perm_tables(order)                             # canon->elem gather
        canon = tables.face_nodes[tables.elem_faces]              # (E,6,k2,3)
        permp = perm[tables.face_orient]                          # (E,6,k2)
        elem = np.take_along_axis(canon, np.broadcast_to(
            permp[..., None], permp.shape + (3,)), axis=2)        # element order
        block[:, fslots, :] = elem
    return block


def to_conformal(tables: EntityTables, points: PointArray,
                 conn: IntArray) -> tuple[FloatArray, IntArray]:
    """Conformal high-order view: ``(nodes (M,3), conn_ho (E,(order+1)**dim))``.

    Every node -- corner, shared edge-interior, private interior -- is numbered once in a
    single global array, and ``conn_ho`` indexes into it in lexicographic block order.
    The direct high-order analog of ``points`` + ``quads``: shared edges resolve to the
    same node ids from both incident elements.  At order 1 this is just
    ``(points, conn)`` re-expressed in block order."""
    order, dim = tables.order, tables.dim
    e = conn.shape[0]
    m = nodes_per_element(order, dim)
    p = points.shape[0]
    conn_ho: IntArray = np.empty((e, m), dtype=np.int64)
    conn_ho[:, corner_indices(order, dim)] = conn
    parts: list[FloatArray] = [np.asarray(points, dtype=float)]
    off = p
    if order > 1:
        if dim >= 2:
            k = order - 1
            ne = tables.edges.shape[0]
            parts.append(tables.edge_nodes.reshape(ne * k, 3))
            islots = _edge_slots(dim, order)[:, 1:-1]              # (nloc, order-1)
            base = off + tables.elem_edges * k                    # (E,nloc)
            ids = base[:, :, None] + np.arange(k)[None, None, :]  # (E,nloc,k) canonical
            ids = np.where(tables.edge_flip[:, :, None], ids[:, :, ::-1], ids)
            conn_ho[:, islots] = ids
            off += ne * k
        if dim == 3:
            k2 = (order - 1) ** 2
            nf = tables.faces.shape[0]
            parts.append(tables.face_nodes.reshape(nf * k2, 3))
            fslots = _face_interior_slots(order)                  # (6, k2)
            perm, _ = _perm_tables(order)                         # canonical order
            fbase = off + tables.elem_faces * k2                  # (E,6)
            fids = fbase[:, :, None] + perm[tables.face_orient]   # (E,6,k2)
            conn_ho[:, fslots] = fids
            off += nf * k2
        kint = tables.interior.shape[1]
        parts.append(tables.interior.reshape(e * kint, 3))
        iids = off + (np.arange(e)[:, None] * kint + np.arange(kint)[None, :])
        conn_ho[:, _interior_slots(dim, order)] = iids
        off += e * kint
    nodes: FloatArray = np.concatenate(parts, axis=0)
    return nodes, conn_ho
