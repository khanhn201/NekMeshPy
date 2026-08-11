"""Entity-based conformal high-order storage."""

from __future__ import annotations

from collections.abc import Sequence

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
    ``(1,0)``, other neighbour -> ``(0,1)``, opposite -> ``(1,1)``."""
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


def _orient_tables() -> tuple[IntArray, IntArray, IntArray]:
    """``(NB (6,4,2), CODE (6,4,2))`` driving the vectorized D4 lookup in
    :func:`unique_faces`, plus the face corner ``uv`` they were fitted against."""
    corner_uv = [m[4] for m in _face_axes()]
    nb_tab: IntArray = np.zeros((6, 4, 2), dtype=np.int64)
    code_tab: IntArray = np.zeros((6, 4, 2), dtype=np.int64)
    for f in range(6):
        uv = corner_uv[f]
        for o in range(4):
            nb = [i for i in range(4)
                  if i != o and int(np.sum(uv[i] != uv[o])) == 1]
            opp = next(i for i in range(4) if i != o and i not in nb)
            nb_tab[f, o] = nb
            for swap in range(2):
                ids: IntArray = np.empty(4, dtype=np.int64)
                ids[o] = 0                       # origin: the minimum, by definition
                ids[nb[0]], ids[nb[1]] = (2, 1) if swap else (1, 2)
                ids[opp] = 3                     # opp never competes for the minimum
                code_tab[f, o, swap] = _face_code(ids, uv)
    return nb_tab, code_tab, np.stack(corner_uv)


_FACE_NB, _FACE_CODE_TAB, _FACE_CORNER_UV = _orient_tables()


def unique_rows(rows: IntArray, *, return_counts: bool = False
                ) -> tuple[IntArray, IntArray, IntArray]:
    """``np.unique(rows, axis=0, return_inverse=True[, return_counts])``, but fast."""
    a = np.ascontiguousarray(rows, dtype=np.int64)
    if a.ndim != 2:
        raise ValueError("unique_rows expects (M,k) rows, got %s" % (a.shape,))
    m, k = a.shape
    if m == 0:
        return (a.reshape(0, k), np.zeros(0, np.int64), np.zeros(0, np.int64))
    n = int(a.max()) + 1 if a.size else 1
    # ``np.unique(axis=0)`` views each row as a void scalar and argsorts that, which is
    # far slower than sorting an integer.  ``k`` ids pack into one int64 as a positional
    # numeral system in base ``n``, so ascending key order *is* lexicographic row order
    # and the result is identical, not merely equivalent.  Only the product can
    # overflow, so fall back to a lexsort when it would.
    if int(a.min()) >= 0 and n ** k <= (1 << 63) - 1:
        key = a[:, 0]
        for c in range(1, k):
            key = key * n + a[:, c]
        if return_counts:                    # split for the typed np.unique overloads
            ukey, inv2, cnt = np.unique(key, return_inverse=True, return_counts=True)
        else:
            ukey, inv2 = np.unique(key, return_inverse=True)
            cnt = np.zeros(0, dtype=np.int64)
        uniq: IntArray = np.empty((ukey.shape[0], k), dtype=np.int64)
        rest = ukey
        for c in range(k - 1, -1, -1):
            uniq[:, c] = rest % n
            rest = rest // n
        return uniq, inv2.ravel().astype(np.int64), cnt.astype(np.int64)
    order = np.lexsort(tuple(a[:, c] for c in range(k - 1, -1, -1)))
    srt = a[order]
    first: BoolArray = np.empty(m, dtype=bool)
    first[0] = True
    np.any(srt[1:] != srt[:-1], axis=1, out=first[1:])
    uniq = srt[first]
    inv: IntArray = np.empty(m, dtype=np.int64)
    inv[order] = np.cumsum(first) - 1
    if not return_counts:
        return uniq, inv, np.zeros(0, np.int64)
    starts = np.flatnonzero(first)
    counts = np.diff(np.concatenate((starts, [m]))).astype(np.int64)
    return uniq, inv, counts


def weld_points(pos: Sequence[PointArray], seams: Sequence[IntArray],
          tol: float | None) -> tuple[PointArray, IntArray]:
    """The corner half of every ``merge``: concatenate the blocks' points, fuse the
    coincident *weldable* ones, and renumber.  Coordinate identity -- unlike the entity
    identity the rest of this module resolves by corner ids, which is why a weld is the
    one place a mesh is decided by geometry."""
    P: PointArray = np.concatenate(list(pos), axis=0) if pos else np.zeros((0, 3))
    total = P.shape[0]

    remap = np.arange(total, dtype=np.int64)
    is_bnd: BoolArray = np.zeros(total, dtype=bool)
    noff = 0
    for p, seam in zip(pos, seams):
        is_bnd[noff + seam] = True
        noff += p.shape[0]
    bidx = np.flatnonzero(is_bnd)
    if bidx.size:
        scl = float(np.max(P.max(axis=0) - P.min(axis=0)))
        t = tol if tol is not None else (1e-7 * scl if scl > 0 else 1.0)
        keys = np.round(P[bidx, :] / t).astype(np.int64)
        uniq, inverse, _ = unique_rows(keys)
        first_local = first_occurrence(inverse, uniq.shape[0])
        remap[bidx] = bidx[first_local][inverse]

    # a cluster's representative maps to itself and everything else maps to a lower
    # index, so the survivors *are* the fixed points -- same sorted set ``np.unique``
    # would return, without sorting the whole point cloud to find it.
    survivors: IntArray = np.flatnonzero(remap == np.arange(total, dtype=np.int64))
    new_id: IntArray = np.empty(total, dtype=np.int64)
    new_id[survivors] = np.arange(survivors.size)
    return P[survivors, :], new_id[remap]


def fuse_entities(rows: IntArray, welded: BoolArray) -> tuple[IntArray, IntArray]:
    """``(new_id (N,), survivors (K,))`` for a merge's concatenated entity table.

    Each block's own entities are already unique, so a weld can only ever join two whose
    corners are *all* welded points -- everything else keeps its own row and is merely
    renumbered.  That makes the dedup proportional to the seam rather than the volume,
    which is the whole reason ``merge`` does not simply re-derive from the corners.

    Survivors come out in concatenation order; an entity's row is whatever its surviving
    representative stores, so callers that care about direction (edges) should canonicalize
    ``rows`` first, and callers that care about frame (faces) should refit afterwards."""
    n = rows.shape[0]
    rep: IntArray = np.arange(n, dtype=np.int64)
    cand: IntArray = np.flatnonzero(np.all(welded[rows], axis=1))
    if cand.size:
        uniq, inv, _ = unique_rows(np.sort(rows[cand], axis=1))
        rep[cand] = cand[first_occurrence(inv, uniq.shape[0])][inv]
    survivors: IntArray = np.flatnonzero(rep == np.arange(n, dtype=np.int64))
    new_id: IntArray = np.empty(n, dtype=np.int64)
    new_id[survivors] = np.arange(survivors.shape[0], dtype=np.int64)
    return new_id[rep], survivors


def first_occurrence(inverse: IntArray, n_groups: int) -> IntArray:
    """``(n_groups,)`` lowest position in ``inverse`` carrying each group -- what
    ``np.unique(..., return_index=True)`` returns alongside the unique rows, for a caller
    that already has the inverse from :func:`unique_rows`.  Scatters the positions in
    descending order, so the smallest is the write that lands."""
    out: IntArray = np.empty(n_groups, dtype=np.int64)
    out[inverse[::-1]] = np.arange(inverse.shape[0] - 1, -1, -1, dtype=np.int64)
    return out


def renumber_map(keep: BoolArray) -> tuple[IntArray, IntArray]:
    """``(kept ids ascending, new_id_of)`` for a subset: ``new_id_of[old]`` is the
    survivor's new id, or ``-1`` where ``keep`` is False.

    The index bookkeeping every ``select`` / ``remove`` rests on, and the reason those
    operations *manufacture* a numbering rather than preserve one."""
    m: BoolArray = np.asarray(keep, dtype=bool).reshape(-1)
    ids: IntArray = np.flatnonzero(m).astype(np.int64)
    new_of: IntArray = np.full(m.shape[0], -1, dtype=np.int64)
    new_of[ids] = np.arange(ids.shape[0], dtype=np.int64)
    return ids, new_of


def element_components(conn: IntArray, n_points: int) -> tuple[int, IntArray]:
    """``(n_components, label per element)`` over the graph in which two elements are
    connected when they **share at least one corner point** -- the weakest join, and so
    the one that answers "is this one body or several".

    Labels are numbered in first-appearance order, so component ``0`` is element ``0``'s.
    Walked on the bipartite element/point graph rather than a materialized element
    adjacency, which keeps it linear in the incidence rather than quadratic in the
    valence."""
    import scipy.sparse as sp
    from scipy.sparse.csgraph import connected_components

    c: IntArray = np.asarray(conn, dtype=np.int64)
    e, k = c.shape
    if e == 0:
        return 0, np.zeros(0, dtype=np.int64)
    rows: IntArray = np.repeat(np.arange(e, dtype=np.int64), k)
    cols: IntArray = e + c.ravel()
    n = e + int(n_points)
    graph = sp.coo_matrix((np.ones(rows.shape[0], dtype=np.int8), (rows, cols)),
                          shape=(n, n))
    _, raw = connected_components(graph, directed=False)
    labels: IntArray = np.asarray(raw[:e], dtype=np.int64)
    # ``connected_components`` also counts every point in no element as its own
    # component, and numbers by its own walk -- so renumber off the *elements* alone.
    _, first = np.unique(labels, return_index=True)
    seen: IntArray = labels[np.sort(first)]
    remap: IntArray = np.zeros(int(labels.max()) + 1, dtype=np.int64)
    remap[seen] = np.arange(seen.shape[0], dtype=np.int64)
    return seen.shape[0], remap[labels]


def unique_faces(hexes: IntArray) -> tuple[IntArray, IntArray, IntArray]:
    """Deduplicate hex faces into unique faces plus a per-hex incidence and D4 code."""
    e = hexes.shape[0]
    ids = hexes[:, _LOCAL_FACES]                            # (E,6,4)
    key = np.sort(ids, axis=2).reshape(e * 6, 4)
    uniq, inv, _ = unique_rows(key)
    elem_faces = inv.reshape(e, 6).astype(np.int64)
    # origin = slot of the minimum id; column = which of its two edge-neighbours
    # (ascending slot order) carries the larger id.
    org = np.argmin(ids, axis=2)                            # (E,6)
    fidx = np.arange(6)[None, :]
    nb = _FACE_NB[fidx, org]                                # (E,6,2) slot ids
    pair = np.take_along_axis(ids, nb, axis=2)              # (E,6,2) their corner ids
    orient = _FACE_CODE_TAB[fidx, org, (pair[..., 0] > pair[..., 1]).astype(np.int64)]
    return uniq.astype(np.int64), elem_faces, orient


#: CCW corner ``(u,v)`` order of a canonical face frame; slot ``i`` is corner ``i``.
_CANON_UV: list[tuple[int, int]] = [(0, 0), (1, 0), (1, 1), (0, 1)]
_QIDX: dict[tuple[int, int], int] = {uv: i for i, uv in enumerate(_CANON_UV)}


def _canon_qidx() -> IntArray:
    """``(6, 8, 4)`` table: for hex local face ``f``, D4 ``code``, and the face's local
    corner ``p``, the CCW slot (0-3) that corner occupies in the canonical face frame
    (``_d4_apply`` of its element-local ``(u,v)``)."""
    corner_uv = [m[4] for m in _face_axes()]                # per-face (4,2) in {0,1}
    out: IntArray = np.empty((6, 8, 4), dtype=np.int64)
    for f in range(6):
        for code in range(8):
            for p in range(4):
                u, v = int(corner_uv[f][p, 0]), int(corner_uv[f][p, 1])
                out[f, code, p] = _QIDX[_d4_apply(u, v, code, 1)]
    return out


_CANON_QIDX: IntArray = _canon_qidx()


def _frame_code_table() -> IntArray:
    """``(6, 256)`` inverse of :data:`_CANON_QIDX`: for hex local face ``f``, the D4 code
    whose corner permutation packs to ``p0*64 + p1*16 + p2*4 + p3``.  ``-1`` marks a
    packing no D4 element realizes."""
    tab: IntArray = np.full((6, 256), -1, dtype=np.int64)
    w: IntArray = np.array([64, 16, 4, 1], dtype=np.int64)
    for f in range(6):
        for code in range(8):
            tab[f, int(_CANON_QIDX[f, code] @ w)] = code
    return tab


_FRAME_CODE_TAB: IntArray = _frame_code_table()

#: The ``(u,v)`` grid position of each corner of a **bare CCW quad row** -- ``u`` along
#: corner 0 -> 1, ``v`` along corner 0 -> 3.  That is ``_CORNER_IJK[2]``, and so the
#: lattice a :class:`QuadMesh <nekmeshpy.quadmesh.quadmesh.QuadMesh>`'s own ``interior``
#: is stored on.  A hex's *local face* frame (:data:`_FACE_CORNER_UV`) is not always
#: this one -- it takes the two in-face axes ascending, which turns faces 3 and 4 -- so
#: a face read out of a hex **into a quad** cannot reuse the hex's ``face_orient``.
_CCW_UV: IntArray = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.int64)


def _ccw_frame_tables() -> tuple[IntArray, IntArray]:
    """``(QIDX (8,4), CODE (256,))`` for a bare CCW quad row: where D4 ``code`` sends
    each of its four slots, and the inverse keyed by the packed permutation -- the
    row-only twin of :data:`_CANON_QIDX` / :data:`_FRAME_CODE_TAB`, which need a hex
    face to index."""
    qidx: IntArray = np.empty((8, 4), dtype=np.int64)
    for code in range(8):
        for p in range(4):
            u, v = int(_CCW_UV[p, 0]), int(_CCW_UV[p, 1])
            qidx[code, p] = _QIDX[_d4_apply(u, v, code, 1)]
    tab: IntArray = np.full(256, -1, dtype=np.int64)
    w: IntArray = np.array([64, 16, 4, 1], dtype=np.int64)
    for code in range(8):
        tab[int(qidx[code] @ w)] = code
    return qidx, tab


_CCW_QIDX, _CCW_FRAME_CODE = _ccw_frame_tables()


def quad_frame_code(rows: IntArray, canonical: IntArray) -> IntArray:
    """``(K,)`` D4 codes carrying each CCW quad ``rows`` entry's own ``(u,v)`` frame onto
    the matching ``canonical`` row's frame -- both orderings of the same four corner ids.

    The row-wise counterpart of :func:`face_frame_code`, which fits the same relation for
    a hex's six local faces at once.  Fitting it against a **given** canonical row rather
    than deriving one from the ids is the whole point: which row a shared face stores is
    the builder's choice, and only that row says what frame its interior nodes are in."""
    a: IntArray = np.asarray(rows, dtype=np.int64).reshape(-1, 4)
    b: IntArray = np.asarray(canonical, dtype=np.int64).reshape(-1, 4)
    if a.shape != b.shape:
        raise ValueError("quad_frame_code: %d rows against %d canonical rows"
                         % (a.shape[0], b.shape[0]))
    # ``sigma[k,p]`` is the canonical slot row ``k``'s corner ``p`` lands in; a slot fits
    # in a byte and so does the packed row, exactly as ``face_frame_code`` packs it.
    sigma = np.zeros(a.shape, dtype=np.uint8)
    for c in range(1, 4):
        sigma += np.uint8(c) * (a == b[:, c, None])
    key: IntArray = (sigma[:, 0].astype(np.int64) * 64 + sigma[:, 1] * 16
                     + sigma[:, 2] * 4 + sigma[:, 3])
    codes: IntArray = _CCW_FRAME_CODE[key]
    if codes.size and int(codes.min()) < 0:
        raise ValueError(
            "quad_frame_code: a quad row is not a D4 image of its canonical row -- the "
            "two do not describe the same quadrilateral")
    return codes


def face_frame_code(local: IntArray, canonical: IntArray) -> IntArray:
    """``(E,6)`` D4 codes carrying each hex's element-local face frame (``local``
    ``(E,6,4)`` corner ids in :data:`_LOCAL_FACES` order) onto the ``canonical``
    ``(E,6,4)`` CCW rows.  :func:`canonical_faces` reads those rows off an owner element;
    this is the same relation for a caller that **chose** its rows in advance, and so
    needs the codes fitted against them."""
    # ``perm[e,f,p]`` is the canonical slot local corner ``p`` lands in; a slot fits in a
    # byte, and so does the packed row (max 3*64+3*16+3*4+3 = 255).
    perm = np.zeros(local.shape, dtype=np.uint8)
    for c in range(1, 4):
        perm += np.uint8(c) * (local == canonical[:, :, c, None])
    key: IntArray = (perm[..., 0].astype(np.int64) * 64 + perm[..., 1] * 16
                     + perm[..., 2] * 4 + perm[..., 3])
    codes: IntArray = _FRAME_CODE_TAB[np.arange(6, dtype=np.int64)[None, :], key]
    if codes.size and int(codes.min()) < 0:
        raise ValueError(
            "face_frame_code: an element-local face frame is not a D4 image of its "
            "canonical row -- the two do not describe the same quadrilateral")
    return codes


def canonical_faces(hexes: IntArray) -> tuple[IntArray, IntArray, IntArray]:
    """CCW-connectivity sibling of :func:`unique_faces`."""
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


def _edge_on_face() -> IntArray:
    """``(12,3)`` -- for each hex local edge, one ``(local face, element-local side of
    that face, reversed)`` it can be read through.  Every hex edge borders two faces;
    which one is picked is arbitrary, because a conformal B-rep gives the same answer
    through either."""
    out: IntArray = np.zeros((12, 3), dtype=np.int64)
    for e, (ca, cb) in enumerate(_LOCAL_EDGES[3].tolist()):
        for f in range(6):
            fc = _LOCAL_FACES[f].tolist()
            sides = [(fc[p], fc[(p + 1) % 4]) for p in range(4)]
            if (ca, cb) in sides:
                out[e] = (f, sides.index((ca, cb)), 0)
                break
            if (cb, ca) in sides:
                out[e] = (f, sides.index((cb, ca)), 1)
                break
    return out


def _canon_side() -> tuple[IntArray, IntArray]:
    """``(SIDE (6,8,4), REV (6,8,4))`` -- the canonical side each element-local face side
    lands on under D4 ``code``, and whether the element walks it backwards.  Canonical
    side ``s`` runs slot ``s`` -> ``s+1``, so an element side whose two slots descend is
    that lower slot's side, traversed against it."""
    side: IntArray = np.zeros((6, 8, 4), dtype=np.int64)
    rev: IntArray = np.zeros((6, 8, 4), dtype=bool)
    for f in range(6):
        for code in range(8):
            for p in range(4):
                c0 = int(_CANON_QIDX[f, code, p])
                c1 = int(_CANON_QIDX[f, code, (p + 1) % 4])
                forward = c1 == (c0 + 1) % 4
                side[f, code, p] = c0 if forward else c1
                rev[f, code, p] = not forward
    return side, rev


_EDGE_ON_FACE: IntArray = _edge_on_face()
_CANON_SIDE, _CANON_SIDE_REV = _canon_side()


def _face_side_edge() -> IntArray:
    """``(6,4,2)`` -- for hex local face ``f`` and its element-local side ``p``, the hex
    local edge that side *is*, and whether the side runs against that edge's direction.
    The transpose of :data:`_EDGE_ON_FACE`, which picks one face per edge."""
    out: IntArray = np.zeros((6, 4, 2), dtype=np.int64)
    edges = _LOCAL_EDGES[3].tolist()
    for f in range(6):
        fc = _LOCAL_FACES[f].tolist()
        for p in range(4):
            side = [fc[p], fc[(p + 1) % 4]]
            out[f, p] = ((edges.index(side), 0) if side in edges
                         else (edges.index(side[::-1]), 1))
    return out


_FACE_SIDE_EDGE: IntArray = _face_side_edge()
#: ``(6,8,4)`` inverse of :data:`_CANON_SIDE`: which element-local side lands on each
#: canonical one.
_SIDE_FROM_CANON: IntArray = np.argsort(_CANON_SIDE, axis=2)


def face_edges_from_hexes(elem_faces: IntArray, face_orient: IntArray,
                          elem_edges: IntArray, edge_flip: BoolArray, n_faces: int
                          ) -> tuple[IntArray, BoolArray]:
    """``(face_edges (F,4), face_flip (F,4))`` -- each shared face's own edge incidence,
    read through **one owning hex** instead of deduplicated again.

    The hex edges and the shared faces' edges are the same set stored in the same table,
    so once ``unique_edges(hexes, 3)`` has run there is nothing left to discover: a
    canonical side maps back through the owner's D4 code to an element-local face side
    (:data:`_SIDE_FROM_CANON`), and that side *is* a hex local edge
    (:data:`_FACE_SIDE_EDGE`).  The flip is three XORs -- stored row vs hex edge, hex edge
    vs face side, face side vs canonical side."""
    first = first_occurrence(elem_faces.reshape(-1), n_faces)
    owner_e, owner_f = np.divmod(first, 6)
    code = face_orient[owner_e, owner_f]
    fo, co = owner_f[:, None], code[:, None]
    p = _SIDE_FROM_CANON[fo, co, np.arange(4)[None, :]]            # (F,4)
    local = _FACE_SIDE_EDGE[fo, p]                                 # (F,4,2)
    against: BoolArray = local[:, :, 1].astype(bool) ^ _CANON_SIDE_REV[fo, co, p]
    return (elem_edges[owner_e[:, None], local[:, :, 0]],
            edge_flip[owner_e[:, None], local[:, :, 0]] ^ against)


def hex_edges_from_faces(elem_faces: IntArray, face_orient: IntArray,
                         face_edges: IntArray, face_flip: BoolArray
                         ) -> tuple[IntArray, BoolArray]:
    """``(elem_edges (E,12), edge_flip (E,12))`` -- each hex's incidence on the shared
    edge table, read *through* the shared faces instead of deduplicated out of its
    corners.

    The edge-rung sibling of :func:`hex_corners_from_faces`, and the reason a ``HexMesh``
    does not care what order its shared edges are stored in: the ids come from the table
    itself (``face_edges`` / ``face_flip`` are the shared-face ``QuadMesh``'s own ``quad``
    / ``flip``), not from re-deriving a numbering that would then have to agree with it.
    """
    f, p = _EDGE_ON_FACE[None, :, 0], _EDGE_ON_FACE[None, :, 1]
    rev: BoolArray = _EDGE_ON_FACE[None, :, 2].astype(bool)
    code = face_orient[:, _EDGE_ON_FACE[:, 0]]                       # (E,12)
    s = _CANON_SIDE[f, code, p]
    fid = elem_faces[:, _EDGE_ON_FACE[:, 0]]
    return (face_edges[fid, s],
            face_flip[fid, s] ^ _CANON_SIDE_REV[f, code, p] ^ rev)


def hex_corners_from_faces(face_conn: IntArray, elem_faces: IntArray,
                           face_orient: IntArray) -> IntArray:
    """Recover ``hexes (E,8)`` (Nek corner order) from the shared-face connectivity."""
    e = elem_faces.shape[0]
    hexes: IntArray = np.empty((e, 8), dtype=np.int64)
    for f in (4, 5):
        q = _CANON_QIDX[f][face_orient[:, f]]                        # (E,4)
        hexes[:, _LOCAL_FACES[f]] = face_conn[elem_faces[:, f][:, None], q]
    return hexes


# -- topology -----------------------------------------------------------
def unique_edges(conn: IntArray, dim: int) -> tuple[IntArray, IntArray, BoolArray]:
    """Deduplicate the local edges of every element into unique undirected edges."""
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
    lo_f, hi_f = lo.ravel(), hi.ravel()
    uniq, inv, _ = unique_rows(np.stack([lo_f, hi_f], axis=1))
    elem_edges = inv.reshape(e, ne).astype(np.int64)
    edge_flip: BoolArray = (a > b)
    return uniq.astype(np.int64), elem_edges, edge_flip


def locate_rows(haystack: IntArray, needles: IntArray, *,
                who: str, what: str) -> IntArray:
    """``idx`` such that ``haystack[idx[i]]`` holds the same *set* of ids as
    ``needles[i]`` -- the id-set lookup behind reading one mesh's entities out of
    another's tables, for edges (2 columns), quads (4) or any width."""
    a: IntArray = np.sort(np.asarray(haystack, dtype=np.int64), axis=1)
    b: IntArray = np.sort(np.asarray(needles, dtype=np.int64), axis=1)
    if a.shape[1] != b.shape[1]:
        raise ValueError("%s: %s rows are %d wide on one side and %d on the other"
                         % (who, what, a.shape[1], b.shape[1]))
    h = a.shape[0]
    uniq, inv, _ = unique_rows(np.concatenate([a, b], axis=0))
    pos: IntArray = np.full(uniq.shape[0], -1, dtype=np.int64)
    pos[inv[:h]] = np.arange(h, dtype=np.int64)
    idx: IntArray = pos[inv[h:]]
    missing = int(np.count_nonzero(idx < 0))
    if missing:
        raise ValueError(
            "%s: %d of %d %s rows have no counterpart in the target -- the two do not "
            "describe the same connectivity" % (who, missing, idx.size, what))
    return idx


# -- array engine: tolerance / scatter / gather / conformal walk ---------
def entity_tol(points: PointArray) -> float:
    """Scale-relative coincidence tolerance for entity sharing: ``1e-8`` of the point
    cloud's bounding-box diagonal extent, falling back to ``1e-12`` for a degenerate
    (single-point or empty) cloud.

    Every rung's sharing check funnels through here, so this coefficient is the one
    place the whole toolkit's idea of "the same point" is set.  It was ``1e-9``, which
    is tight enough that a mesh cut from a *solved* field -- where the field came from
    a tet mesh some other machine's gmsh built slightly differently -- could miss a
    seam it structurally shares.  Coincidence here is a question about construction,
    not about measurement: entities are meant to be the same object, so the number only
    has to stay far below any real feature, and ``1e-8`` of the model extent is."""
    scale = (float(np.max(points.max(axis=0) - points.min(axis=0)))
             if points.size else 0.0)
    return 1e-8 * scale if scale > 0 else 1e-12


def _conformal_walk(points: PointArray, conn: IntArray, dim: int, order: int,
                    elem_edges: IntArray, edge_flip: BoolArray,
                    edge_nodes: PointArray, elem_faces: IntArray,
                    face_orient: IntArray, face_nodes: PointArray,
                    interior: PointArray) -> tuple[PointArray, IntArray]:
    """Shared body of the ``conformal_*`` walks."""
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
    """Scatter element-local edge-interior nodes into the shared canonical table."""
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
    """Scatter element-local hex-face interior nodes into the shared canonical table."""
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
    """Gather the shared edge table back into element-local order."""
    nodes: PointArray = edge_nodes[elem_edges]                 # (E,nloc,order-1,3)
    return np.where(edge_flip[:, :, None, None], nodes[:, :, ::-1, :], nodes)


def gather_face_nodes(face_nodes: PointArray, elem_faces: IntArray,
                      face_orient: IntArray) -> PointArray:
    """Gather the shared hex-face table back into element-local order."""
    k2 = face_nodes.shape[1]
    perm, _ = _perm_tables(_k_from_face_width(k2) + 1)         # canon->elem gather
    canon: PointArray = face_nodes[elem_faces]                 # (E,6,k2,3)
    permp = perm[face_orient]                                  # (E,6,k2)
    return np.take_along_axis(canon, np.broadcast_to(
        permp[..., None], permp.shape + (3,)), axis=2)


def face_nodes_in_frame(canon_nodes: PointArray, rows: IntArray,
                        canonical: IntArray) -> PointArray:
    """``(K,(order-1)**2,3)`` shared face-interior nodes, turned out of their stored
    ``canonical`` row's frame into the frame of the CCW quad ``rows``.

    :func:`gather_face_nodes` is this same read for a *hex*, which was handed its codes
    at construction; here they are fitted row against row, which is what lets one mesh's
    faces be read out into another mesh's quads. Skipping the turn silently permutes the
    interior nodes of every face whose fit is not the identity -- and leaves the corners
    and edges right, so the damage shows up only as geometry."""
    canon: PointArray = np.asarray(canon_nodes, dtype=float)
    if canon.shape[1] == 0:                                    # order 1: nothing inside
        return canon
    perm, _ = _perm_tables(_k_from_face_width(canon.shape[1]) + 1)
    p: IntArray = perm[quad_frame_code(rows, canonical)]        # (K,k2) canon->row
    return np.take_along_axis(canon, np.broadcast_to(
        p[..., None], p.shape + (3,)), axis=1)


def conformal_line(points: PointArray, lines: IntArray, interior: PointArray,
                   order: int) -> tuple[PointArray, IntArray]:
    """Conformal high-order view of a line mesh: ``(nodes (M,3), conn_ho (L,order+1))``.
    """
    e = lines.shape[0]
    empty_i: IntArray = np.zeros((e, 0), np.int64)
    return _conformal_walk(
        points, lines, 1, order,
        empty_i, np.zeros((e, 0), bool), np.zeros((0, max(order - 1, 0), 3), float),
        empty_i, empty_i, np.zeros((0, 0, 3), float), interior)


def conformal_quad(points: PointArray, quads: IntArray, elem_edges: IntArray,
                   edge_flip: BoolArray, edge_nodes: PointArray,
                   interior: PointArray, order: int) -> tuple[PointArray, IntArray]:
    """Conformal high-order view of a quad mesh: ``(nodes (M,3), conn_ho
    (Q,(order+1)**2))``."""
    e = quads.shape[0]
    empty_i: IntArray = np.zeros((e, 0), np.int64)
    return _conformal_walk(
        points, quads, 2, order, elem_edges, edge_flip, edge_nodes,
        empty_i, empty_i, np.zeros((0, 0, 3), float), interior)


def conformal_hex(points: PointArray, hexes: IntArray, elem_edges: IntArray,
                  edge_flip: BoolArray, edge_nodes: PointArray, elem_faces: IntArray,
                  face_orient: IntArray, face_nodes: PointArray,
                  interior: PointArray, order: int) -> tuple[PointArray, IntArray]:
    """Conformal high-order view of a hex mesh: ``(nodes (M,3), conn_ho
    (E,(order+1)**3))``."""
    return _conformal_walk(
        points, hexes, 3, order, elem_edges, edge_flip, edge_nodes,
        elem_faces, face_orient, face_nodes, interior)
