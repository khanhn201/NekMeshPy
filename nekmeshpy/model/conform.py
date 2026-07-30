"""Entity-based conformal high-order storage.

The corner layer is already conformal: a shared corner is one node in ``points (P,3)``
referenced by every incident element's connectivity.  This module gives the high-order
(edge / face / interior) nodes the same property.  Rather than let each element carry
its own copy of a shared edge's ``N-1`` interior nodes, the non-corner nodes are
decomposed by **topology** into

* **edges** -- unique undirected edges (canonical: min-corner-id first) with their
  ``N-1`` shared interior nodes, plus a per-element incidence + a *flip* bit when an
  element traverses an edge anti-canonically,
* **faces** (hex only) -- unique faces with their ``(N-1)**2`` shared interior nodes,
  plus a per-element incidence + a D4 orientation code,
* **interior** -- the per-element private nodes (quad ``(N-1)**2``, hex ``(N-1)**3``,
  line ``N-1``) that are never shared.

The containers store that decomposition **natively** (``LineMesh.interior``;
``QuadMesh.lines`` / ``quad`` / ``flip`` / ``interior``; ``HexMesh.quads`` / ``hex`` /
``face_orient`` / ``interior``), so this module holds no storage value of its own and
imports no container -- everything crosses the boundary as plain arrays:

* :func:`unique_edges` / :func:`unique_faces` / :func:`canonical_faces` /
  :func:`hex_corners_from_faces` -- the topology: dedupe an element's local edges /
  faces into shared entities and walk back the other way.
* :func:`entity_tol` -- the one scale-relative coincidence tolerance every
  reconciliation step judges conformality with.
* :func:`scatter_edge_nodes` / :func:`scatter_face_nodes` -- push element-local entity
  nodes into the canonical shared tables (owner-wins, with every other incident copy
  **verified** within tolerance, so a non-conforming input is a loud ``ValueError``
  rather than a silent weld), and their exact inverses :func:`gather_edge_nodes` /
  :func:`gather_face_nodes`.
* :func:`conformal_line` / :func:`conformal_quad` / :func:`conformal_hex` -- the
  conformal walk: number every node once into ``nodes (M,3)`` with dense
  ``conn_ho (E,(N+1)^d)`` into it, the high-order analog of ``points`` + ``quads``.
  This is the single node numbering both export and the order-N quality metrics read;
  ``nodes[conn_ho]`` is the transient per-element block.

Sharing is decided by corner ids (**structural / exact** conformality), never by a
coordinate search.  At ``order == 1`` every entity table is empty and the conformal walk
returns just ``(points, conn)`` in block order, so the order-1 path is byte-identical to
the plain corner mesh.
"""

from __future__ import annotations

import numpy as np

from .._typing import BoolArray, IntArray, PointArray
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


# -- array engine: tolerance / scatter / gather / conformal walk ---------
def entity_tol(points: PointArray) -> float:
    """Scale-relative coincidence tolerance for entity sharing: ``1e-9`` of the point
    cloud's bounding-box diagonal extent, falling back to ``1e-12`` for a degenerate
    (single-point or empty) cloud.  The one tolerance every scatter/verify step uses, so
    conformality is judged the same way everywhere."""
    scale = (float(np.max(points.max(axis=0) - points.min(axis=0)))
             if points.size else 0.0)
    return 1e-9 * scale if scale > 0 else 1e-12


def _conformal_walk(points: PointArray, conn: IntArray, dim: int, order: int,
                    elem_edges: IntArray, edge_flip: BoolArray,
                    edge_nodes: PointArray, elem_faces: IntArray,
                    face_orient: IntArray, face_nodes: PointArray,
                    interior: PointArray) -> tuple[PointArray, IntArray]:
    """Shared body of the ``conformal_*`` walks.

    Numbers every node once -- corners (``points``) ++ edge interiors ++ face interiors
    ++ cell interiors, in that order -- and fills ``conn_ho`` in lexicographic block
    order (``i`` fastest).  Unused entity tables (dim 1 edges, dim 1/2 faces) are passed
    as zero-width arrays and contribute nothing."""
    e = conn.shape[0]
    m = nodes_per_element(order, dim)
    p = points.shape[0]
    conn_ho: IntArray = np.empty((e, m), dtype=np.int64)
    conn_ho[:, corner_indices(order, dim)] = conn
    parts: list[PointArray] = [np.asarray(points, dtype=float)]
    off = p
    if order > 1:
        if dim >= 2:
            k = order - 1
            ne = edge_nodes.shape[0]
            parts.append(edge_nodes.reshape(ne * k, 3))
            islots = _edge_slots(dim, order)[:, 1:-1]              # (nloc, order-1)
            base = off + elem_edges * k                           # (E,nloc)
            ids = base[:, :, None] + np.arange(k)[None, None, :]  # (E,nloc,k) canonical
            ids = np.where(edge_flip[:, :, None], ids[:, :, ::-1], ids)
            conn_ho[:, islots] = ids
            off += ne * k
        if dim == 3:
            k2 = (order - 1) ** 2
            nf = face_nodes.shape[0]
            parts.append(face_nodes.reshape(nf * k2, 3))
            fslots = _face_interior_slots(order)                  # (6, k2)
            perm, _ = _perm_tables(order)                         # canonical order
            fbase = off + elem_faces * k2                         # (E,6)
            fids = fbase[:, :, None] + perm[face_orient]          # (E,6,k2)
            conn_ho[:, fslots] = fids
            off += nf * k2
        kint = interior.shape[1]
        parts.append(interior.reshape(e * kint, 3))
        iids = off + (np.arange(e)[:, None] * kint + np.arange(kint)[None, :])
        conn_ho[:, _interior_slots(dim, order)] = iids
        off += e * kint
    nodes: PointArray = np.concatenate(parts, axis=0)
    return nodes, conn_ho


def _k_from_face_width(k2: int) -> int:
    """``order-1`` from a face table's ``(order-1)**2`` node width."""
    k = int(round(float(np.sqrt(k2))))
    if k * k != k2:                                            # pragma: no cover
        raise ValueError("face node width %d is not a perfect square" % k2)
    return k


def scatter_edge_nodes(local: PointArray, elem_edges: IntArray, edge_flip: BoolArray,
                       n_edges: int, tol: float, who: str) -> PointArray:
    """Scatter element-local edge-interior nodes into the shared canonical table.

    ``local`` is ``(E, n_local_edges, order-1, 3)``: every element's own copy of its
    local edges' interior nodes, in **element traversal order** (that local edge's
    start corner -> end corner).  Each copy is reversed where ``edge_flip`` says the
    element traverses the edge anti-canonically, the owning (lowest flat incidence)
    element supplies each unique edge's nodes, and every other incident copy is
    **verified** to agree within ``tol`` -- a non-conforming input raises ``ValueError``
    naming ``who`` rather than silently welding.  Returns ``(n_edges, order-1, 3)`` in
    canonical (min-corner-id first) order; the inverse of
    :func:`gather_edge_nodes`."""
    k = local.shape[2]
    canon = np.where(edge_flip[:, :, None, None], local[:, :, ::-1, :], local)
    flat_eid = elem_edges.ravel()
    canon_flat = canon.reshape(flat_eid.shape[0], k, 3)
    _, first = np.unique(flat_eid, return_index=True)          # owner per edge id
    edge_nodes: PointArray = canon_flat[first]                 # (Ne,order-1,3)
    if not np.allclose(canon_flat, edge_nodes[flat_eid], rtol=0.0, atol=tol):
        raise ValueError(
            "%s: non-conforming high-order edge -- incident elements disagree on a "
            "shared edge's interior nodes beyond tolerance (%.3e). The inputs are not "
            "structurally conformal." % (who, tol))
    if edge_nodes.shape[0] != n_edges:                         # pragma: no cover
        raise AssertionError("%s: edge owner count mismatch" % who)
    return edge_nodes


def scatter_face_nodes(local: PointArray, elem_faces: IntArray, face_orient: IntArray,
                       n_faces: int, tol: float, who: str) -> PointArray:
    """Scatter element-local hex-face interior nodes into the shared canonical table.

    ``local`` is ``(E, 6, (order-1)**2, 3)`` in each hex's **element-local ``(u,v)``
    face frame with ``u`` fastest** -- exactly the frame :func:`_face_interior_slots`
    reads.  Each copy is permuted to the shared canonical D4 frame through
    ``face_orient`` (:func:`_perm_tables`), the owning (lowest flat incidence) hex wins,
    and every other incident copy is **verified** within ``tol``, raising ``ValueError``
    naming ``who`` otherwise.  Returns ``(n_faces, (order-1)**2, 3)``; the inverse of
    :func:`gather_face_nodes`."""
    k2 = local.shape[2]
    _, inv = _perm_tables(_k_from_face_width(k2) + 1)          # INV: elem->canon gather
    invp = inv[face_orient]                                    # (E,6,k2)
    canon = np.take_along_axis(local, np.broadcast_to(
        invp[..., None], invp.shape + (3,)), axis=2)           # (E,6,k2,3) canonical
    flat_fid = elem_faces.ravel()
    canon_flat = canon.reshape(flat_fid.shape[0], k2, 3)
    _, first = np.unique(flat_fid, return_index=True)
    face_nodes: PointArray = canon_flat[first]                 # (Nf,k2,3)
    if not np.allclose(canon_flat, face_nodes[flat_fid], rtol=0.0, atol=tol):
        raise ValueError(
            "%s: non-conforming high-order face -- incident hexes disagree on a shared "
            "face's interior nodes beyond tolerance (%.3e). The inputs are not "
            "structurally conformal." % (who, tol))
    if face_nodes.shape[0] != n_faces:                         # pragma: no cover
        raise AssertionError("%s: face owner count mismatch" % who)
    return face_nodes


def gather_edge_nodes(edge_nodes: PointArray, elem_edges: IntArray,
                      edge_flip: BoolArray) -> PointArray:
    """Gather the shared edge table back into element-local order.

    Exact inverse of :func:`scatter_edge_nodes`: returns
    ``(E, n_local_edges, order-1, 3)`` with each element's copy running along its own
    traversal direction (reversed wherever ``edge_flip`` is set)."""
    nodes: PointArray = edge_nodes[elem_edges]                 # (E,nloc,order-1,3)
    return np.where(edge_flip[:, :, None, None], nodes[:, :, ::-1, :], nodes)


def gather_face_nodes(face_nodes: PointArray, elem_faces: IntArray,
                      face_orient: IntArray) -> PointArray:
    """Gather the shared hex-face table back into element-local order.

    Exact inverse of :func:`scatter_face_nodes`: returns ``(E, 6, (order-1)**2, 3)`` in
    each hex's element-local ``(u,v)`` frame (``u`` fastest), undoing the canonical D4
    permutation carried by ``face_orient``."""
    k2 = face_nodes.shape[1]
    perm, _ = _perm_tables(_k_from_face_width(k2) + 1)         # canon->elem gather
    canon: PointArray = face_nodes[elem_faces]                 # (E,6,k2,3)
    permp = perm[face_orient]                                  # (E,6,k2)
    return np.take_along_axis(canon, np.broadcast_to(
        permp[..., None], permp.shape + (3,)), axis=2)


def conformal_line(points: PointArray, lines: IntArray, interior: PointArray,
                   order: int) -> tuple[PointArray, IntArray]:
    """Conformal high-order view of a line mesh: ``(nodes (M,3), conn_ho (L,order+1))``.

    ``lines (L,2)`` is the corner connectivity and ``interior (L,order-1,3)`` each line
    element's private interior nodes (a line element *is* a 1-cell, so it shares only
    corners).  Global numbering is ``points`` ++ interiors; ``conn_ho`` indexes ``nodes``
    in lexicographic block order.  At order 1 this is ``(points, lines)``."""
    e = lines.shape[0]
    empty_i: IntArray = np.zeros((e, 0), np.int64)
    return _conformal_walk(
        points, lines, 1, order,
        empty_i, np.zeros((e, 0), bool), np.zeros((0, max(order - 1, 0), 3), float),
        empty_i, empty_i, np.zeros((0, 0, 3), float), interior)


def conformal_quad(points: PointArray, quads: IntArray, elem_edges: IntArray,
                   edge_flip: BoolArray, edge_nodes: PointArray,
                   interior: PointArray, order: int) -> tuple[PointArray, IntArray]:
    """Conformal high-order view of a quad mesh: ``(nodes (M,3), conn_ho (Q,(order+1)**2))``.

    ``quads (Q,4)`` is the **corner** connectivity; ``elem_edges``/``edge_flip`` are the
    per-quad incidence into ``edge_nodes (Ne,order-1,3)`` (canonical, min-corner-id
    first) and ``interior (Q,(order-1)**2,3)`` the private face interiors.  Global
    numbering is ``points`` ++ edge interiors ++ quad interiors, so two quads sharing an
    edge resolve to the same node ids.  At order 1 this is ``(points, quads)`` in block
    order."""
    e = quads.shape[0]
    empty_i: IntArray = np.zeros((e, 0), np.int64)
    return _conformal_walk(
        points, quads, 2, order, elem_edges, edge_flip, edge_nodes,
        empty_i, empty_i, np.zeros((0, 0, 3), float), interior)


def conformal_hex(points: PointArray, hexes: IntArray, elem_edges: IntArray,
                  edge_flip: BoolArray, edge_nodes: PointArray, elem_faces: IntArray,
                  face_orient: IntArray, face_nodes: PointArray,
                  interior: PointArray, order: int) -> tuple[PointArray, IntArray]:
    """Conformal high-order view of a hex mesh: ``(nodes (M,3), conn_ho (E,(order+1)**3))``.

    ``hexes (E,8)`` is the **corner** connectivity; ``elem_edges``/``edge_flip`` index
    ``edge_nodes (Ne,order-1,3)``, ``elem_faces``/``face_orient`` index
    ``face_nodes (Nf,(order-1)**2,3)`` (canonical D4 frame), and
    ``interior (E,(order-1)**3,3)`` holds the private cell interiors.  Global numbering
    is ``points`` ++ edge interiors ++ face interiors ++ cell interiors, so hexes sharing
    an edge or a face resolve to the same node ids.  At order 1 this is
    ``(points, hexes)`` in block order."""
    return _conformal_walk(
        points, hexes, 3, order, elem_edges, edge_flip, edge_nodes,
        elem_faces, face_orient, face_nodes, interior)
