"""Mesh topology / validity checks, decoupled from the mesh containers."""

from __future__ import annotations

from typing import Any, NamedTuple, Union

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from .._typing import BoolArray, IntArray, PointArray
from . import conform

# Nek face -> the 4 corner point positions (0-based), cyclic and outward-wound.
_FACE_POINTS = conform._LOCAL_FACES


class TopologyReport(NamedTuple):
    """Facet inventory and validity verdicts for an all-hex mesh."""

    #: Number of hex elements.
    n_elements: int
    #: Distinct quad facets, counted orientation-free.
    n_faces: int
    #: Facets borne by exactly one hex -- the domain boundary.
    n_boundary_faces: int
    #: Facets shared by exactly two hexes.
    n_internal_faces: int
    #: Facets shared by three or more hexes -- always a defect.
    n_nonmanifold_faces: int
    #: Boundary-surface edges bounded by an **odd** number of boundary faces.  The
    #: boundary of any union of cells is closed, so with faces matched by their corners
    #: this cannot happen in a consistent mesh: a non-zero count means combinatorially
    #: broken input (a degenerate face, a repeated corner).  A crack against a
    #: T-junction is *not* this -- it reads as hanging points.
    n_open_edges: int
    #: Boundary points lying strictly inside another boundary edge (T-junctions).
    n_hanging_points: int
    #: Connected components of the hex-adjacency graph.
    n_components: int
    #: ``True`` when the boundary is a closed, non-empty 2-manifold.  Says nothing
    #: about conformity: a T-junction interface still reads watertight.
    watertight: bool
    #: ``True`` when there are no hanging points and no improper pairs.
    conformal: bool
    #: Element pairs that meet improperly, plus elements inconsistent on their own (a
    #: repeated corner, line or face).  Two cells of a valid mesh meet in nothing or in
    #: exactly **one** shared entity: one point, one stored line (its two endpoints and
    #: nothing else), or at the hex rung one stored face (its 4 corners and 4 lines).
    #: Anything else is a defect no tolerance can fix -- two corners with no line
    #: between them (opposite corners, or one quad's diagonal on the other's edge), three
    #: corners (a doublet: an interface that failed to become one face, which still reads
    #: watertight), four corners without one shared face, two shared faces.  Nek's
    #: parRSB connectivity (``face_check``) refuses the mesh.  Purely combinatorial; see
    #: :func:`improper_pairs` for which.
    n_improper: int = 0
    #: Boundary-surface edges bounded by an even number **>= 4** of boundary faces: two
    #: (or more) cells meeting only along that edge, the domain pinched to a line.
    #: Conformal and readable by a solver, but the boundary is not a manifold there.
    #: One rung down (quad meshes) this counts boundary points where two quads meet
    #: only at a corner.
    n_pinch_edges: int = 0


# -- shared helpers -----------------------------------------------------
def _incidence(rows: IntArray, n: int) -> sp.csr_matrix:
    """``(E, n)`` 0/1 incidence of an ``(E, k)`` id table (repeats within a row count once)."""
    r = np.asarray(rows, dtype=np.int64)
    m = sp.csr_matrix((np.ones(r.size, np.int32), r.ravel(),
                       np.arange(0, r.size + 1, max(r.shape[1], 1))), shape=(r.shape[0], n))
    m.sum_duplicates()
    m.data[:] = 1
    return m


#: ``(points, lines, faces)`` two cells may share: one point, one line, one face.
_PROPER = {1: {(1, 0, 0)}, 2: {(1, 0, 0), (2, 1, 0)},
           3: {(1, 0, 0), (2, 1, 0), (4, 4, 1)}}
#: distinct ``(points, lines, faces)`` one cell holds.
_OWN = {1: (2, 0, 0), 2: (4, 4, 0), 3: (8, 12, 6)}


def improper_pairs(cell_points: IntArray, cell_lines: IntArray | None = None,
                   cell_faces: IntArray | None = None, *,
                   chunk: int = 200_000) -> tuple[IntArray, IntArray]:
    """``(pairs (K,2), cells (J,))``: element pairs that do not meet in exactly one
    shared point, line or face, and elements that are inconsistent on their own.

    ``cell_points`` is each element's corner ids (2 / 4 / 8 wide, which sets the
    rung), ``cell_lines`` its line ids and ``cell_faces`` (hexes) its face ids -- the
    *stored* ones, which is what makes "the same line" mean one entity rather than two
    that happen to join the same corners.  A row may list an id more than once (a hex's
    lines read off its six faces do); it counts once."""
    P = np.asarray(cell_points, dtype=np.int64)
    E, k = P.shape
    dim = {2: 1, 4: 2, 8: 3}[k]
    blocks = [_incidence(P, int(P.max()) + 1 if P.size else 1)]
    for rows in (cell_lines, cell_faces):
        if rows is not None and np.asarray(rows).size:
            r = np.asarray(rows, dtype=np.int64)
            blocks.append(_incidence(r, int(r.max()) + 1))
        else:
            blocks.append(None)
    weight = (1, 16, 256)
    own = sum(w * (np.diff(b.indptr) if b is not None else 0) for w, b in zip(weight, blocks))
    expect = sum(w * c for w, c, b in zip(weight, _OWN[dim], blocks) if b is not None)
    bad_cells = np.flatnonzero(own != expect)
    ok = np.array(sorted(sum(w * c for w, c, b in zip(weight, t, blocks) if b is not None)
                         for t in _PROPER[dim]))
    out = []
    for s0 in range(0, E, chunk):
        s1 = min(E, s0 + chunk)
        terms = [(b[s0:s1] @ b.T) * w for w, b in zip(weight, blocks) if b is not None]
        M = sp.triu(sum(terms[1:], terms[0]).tocoo(), k=s0 + 1)   # j > i in global ids
        sel = ~np.isin(M.data, ok)
        if sel.any():
            out.append(np.stack([M.row[sel] + s0, M.col[sel]], 1))
    pairs = np.concatenate(out) if out else np.zeros((0, 2), np.int64)
    return pairs.astype(np.int64), bad_cells.astype(np.int64)


def _count_components(n: int, edges: IntArray) -> int:
    """Number of connected components of an ``n``-point graph with ``(E,2)`` edges."""
    if n == 0:
        return 0
    e = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    graph = sp.coo_matrix(
        (np.ones(e.shape[0], dtype=np.int8), (e[:, 0], e[:, 1])), shape=(n, n))
    return int(connected_components(graph, directed=False, return_labels=False))


def _adjacency_edges(inverse: IntArray, counts: IntArray,
                     owner: IntArray) -> IntArray:
    """Cell-adjacency edges from a face/edge incidence: facets shared by exactly
    two cells contribute one cell-cell edge."""
    order = np.argsort(inverse, kind="stable")
    inv_s = inverse[order]
    owner_s = owner[order]
    if inv_s.size == 0:
        return np.zeros((0, 2), dtype=np.int64)
    # group boundaries, then take the runs of length two without materializing a
    # Python list of ~2M single-facet subarrays (``np.split`` was 5 s of the build)
    starts = np.concatenate(([0], np.flatnonzero(np.diff(inv_s)) + 1))
    sizes = np.diff(np.concatenate((starts, [inv_s.size])))
    two = starts[sizes == 2]
    if two.size == 0:
        return np.zeros((0, 2), dtype=np.int64)
    return np.stack([owner_s[two], owner_s[two + 1]], axis=1).astype(np.int64)


def _count_hanging_points(points: PointArray, edges: IntArray,
                         candidates: IntArray, rtol: float = 1e-7) -> int:
    """Number of ``candidates`` points lying strictly inside an ``edges`` edge
    (not an endpoint) -- the signature of a non-conforming T-junction interface.
    """
    X = np.asarray(points, dtype=float)
    E = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    cand = np.asarray(candidates, dtype=np.int64).ravel()
    if E.shape[0] == 0 or cand.size == 0 or X.shape[0] == 0:
        return 0
    from scipy.spatial import cKDTree
    scale = conform.bbox_scale(X)
    tol = rtol * (scale if scale > 0 else 1.0)
    Xc = X[cand]
    tree = cKDTree(Xc)
    # One batched ball query and one vectorized projection, rather than a Python loop
    # per edge: the loop was ~12 s of a 490k-hex build, almost all of it in its own
    # body rather than in the tree.
    A, B = X[E[:, 0]], X[E[:, 1]]
    D = B - A
    L2 = np.einsum("ij,ij->i", D, D)
    idx = np.flatnonzero(L2 > tol * tol)              # skip degenerate edges
    if idx.size == 0:
        return 0
    hits = tree.query_ball_point(0.5 * (A[idx] + B[idx]),
                                 0.5 * np.sqrt(L2[idx]) + tol, workers=-1)
    counts: IntArray = np.fromiter((len(h) for h in hits), dtype=np.int64,
                                   count=len(hits))
    if not counts.any():
        return 0
    ci = np.concatenate([np.asarray(h, dtype=np.int64) for h in hits if h])
    ei: IntArray = np.repeat(idx, counts)              # the edge each hit belongs to
    k = cand[ci]
    keep = (k != E[ei, 0]) & (k != E[ei, 1])           # an endpoint is not hanging
    ci, ei, k = ci[keep], ei[keep], k[keep]
    if ci.size == 0:
        return 0
    t = np.einsum("ij,ij->i", Xc[ci] - A[ei], D[ei]) / L2[ei]
    keep = (t > 1e-9) & (t < 1.0 - 1e-9)               # strictly interior
    ci, ei, k, t = ci[keep], ei[keep], k[keep], t[keep]
    if ci.size == 0:
        return 0
    perp = Xc[ci] - (A[ei] + t[:, None] * D[ei])
    on_edge = np.sqrt(np.einsum("ij,ij->i", perp, perp)) <= tol
    return int(np.unique(k[on_edge]).size)


#: ``_FACE_POINTS[i]``'s opposite face -- (eta=0, eta=1), (xi=1, xi=0), (zeta=0, zeta=1)
#: in the corner numbering ``_CN`` (``hexmesh.quality.scaled_jacobian``) also reads
#: local axes off of. ``fcenter[i] - fcenter[_OPPOSITE[i]]`` is therefore a robust
#: outward direction for face ``i`` -- along the hex's own parametric axis, wrong only
#: if the hex is squashed to zero thickness along it, a much weaker degeneracy than
#: "which side of the centroid".
_OPPOSITE = np.array([2, 3, 0, 1, 5, 4], dtype=np.int64)


def _anchored_face_points(X: PointArray, HC: IntArray) -> PointArray:
    """``(N,6,4,3)`` face corner points, each face's own 4-point cyclic sequence
    rotated to start at whichever corner holds the *smallest global point id* --
    not an arbitrary hex-local starting corner (``_FACE_POINTS``' own ``[0]``).

    This is what lets two hexes that share a face agree, point-set for point-set,
    on how it splits into 2 triangles: the shared face is one physical quad but two
    *different* local corner orderings (each hex enumerates it starting from its own
    unrelated corner 0, one hex's winding the reverse of the other's), so anchoring
    on ``_FACE_POINTS[..., 0]`` picks two different diagonals -- and on a warped
    (non-planar) face, routine on curved geometry, those two triangulations are
    measurably different surfaces, not the same one twice. Anchoring on the global
    id instead is winding- and starting-point-independent: rotating a cyclic
    sequence to a fixed *value* rather than a fixed *position* lands on the same
    physical corner regardless of which hex is doing the enumerating."""
    fids = HC[:, _FACE_POINTS]                                 # (N,6,4) global ids
    anchor = np.argmin(fids, axis=2)                           # (N,6)
    roll = (anchor[:, :, None] + np.arange(4)[None, None, :]) % 4   # (N,6,4)
    fpts = X[HC][:, _FACE_POINTS, :]                           # (N,6,4,3)
    return np.take_along_axis(fpts, np.broadcast_to(roll[..., None], fpts.shape),
                              axis=2)


def _hex_geometry(X: PointArray, HC: IntArray) -> tuple[
        PointArray, PointArray, PointArray, PointArray]:
    """Per-hex centroid, bounding-sphere radius, and each of its 6 quad faces split
    into 2 triangles (12 total) as an exact ``(N,12,3)`` vertex + outward unit normal
    pair -- a genuinely planar constraint each, unlike one average plane per quad
    face, which a warped (non-planar) face -- routine on curved geometry -- can be
    off by a fraction of the element's own size, several orders past floating-point
    noise. Orientation comes from the hex's own parametric axes (a face's triangles
    vs. its opposite face's centre), not the stored corner winding or a
    centroid-relative heuristic, either of which a skewed element can fool."""
    corners = X[HC]                                          # (N,8,3)
    centroid = corners.mean(axis=1)                           # (N,3)
    radius = np.linalg.norm(corners - centroid[:, None, :], axis=2).max(axis=1)
    fpts = _anchored_face_points(X, HC)                        # (N,6,4,3)
    fcenter = fpts.mean(axis=2)                                # (N,6,3)
    opp_center = fcenter[:, _OPPOSITE, :]
    # triangles (a,b,c) and (a,c,d) of each face [a,b,c,d]
    tri_a = np.stack([fpts[:, :, 0, :], fpts[:, :, 0, :]], axis=2)   # (N,6,2,3)
    tri_b = np.stack([fpts[:, :, 1, :], fpts[:, :, 2, :]], axis=2)
    tri_c = np.stack([fpts[:, :, 2, :], fpts[:, :, 3, :]], axis=2)
    normal = np.cross(tri_b - tri_a, tri_c - tri_a)            # (N,6,2,3)
    out = fcenter[:, :, None, :] - opp_center[:, :, None, :]
    sign = np.sign(np.einsum("nftd,nftd->nft", normal, out))
    sign[sign == 0.0] = 1.0
    normal = normal * sign[..., None]
    length = np.linalg.norm(normal, axis=-1, keepdims=True)
    length[length == 0.0] = 1.0
    normal = (normal / length).reshape(-1, 12, 3)
    vertex = tri_a.reshape(-1, 12, 3)
    return centroid, radius, vertex, normal


#: the 4 triangles of a face [a,b,c,d] under BOTH diagonal splits (a-c and b-d)
_BOTH_SPLITS = np.array([[0, 1, 2], [0, 2, 3], [1, 2, 3], [1, 3, 0]], dtype=np.int64)


def _hex_planes_both(X: PointArray, HC: IntArray) -> tuple[PointArray, PointArray]:
    """``(N,24,3)`` vertex and outward unit normal of every face triangle under *both*
    diagonal splits.  A linear hex's warped face is a bilinear patch lying between its
    two triangulations, so a point strictly inside all 24 planes is inside the true
    element whichever way the face is split -- the overlap verdict no longer depends
    on corner numbering (which the single anchored split of ``_hex_geometry``
    does, by a few hundredths of a millimetre on a warped face)."""
    fpts = X[HC][:, _FACE_POINTS, :]                           # (N,6,4,3)
    fcenter = fpts.mean(axis=2)
    out = fcenter - fcenter[:, _OPPOSITE, :]                   # (N,6,3) outward axis
    tri = fpts[:, :, _BOTH_SPLITS, :]                          # (N,6,4,3,3)
    normal = np.cross(tri[..., 1, :] - tri[..., 0, :], tri[..., 2, :] - tri[..., 0, :])
    sign = np.sign(np.einsum("nftd,nfd->nft", normal, out))
    sign[sign == 0.0] = 1.0
    normal = normal * sign[..., None]
    length = np.linalg.norm(normal, axis=-1, keepdims=True)
    length[length == 0.0] = 1.0
    return tri[..., 0, :].reshape(-1, 24, 3), (normal / length).reshape(-1, 24, 3)


#: bilinear weights of the face samples: the centre and the 4 points at (+-1/2, +-1/2)
_FACE_W = np.array([[1, 1, 1, 1], [9, 3, 1, 3], [3, 9, 3, 1], [1, 3, 9, 3], [3, 1, 3, 9]],
                   dtype=float) / np.array([[4.0], [16.0], [16.0], [16.0], [16.0]])
_HEX_EDGES = np.array([[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4],
                       [0, 4], [1, 5], [2, 6], [3, 7]], dtype=np.int64)


def _hex_surface_samples(X: PointArray, HC: IntArray) -> PointArray:
    """``(N,50,3)`` points ON each linear hex's true (bilinear-faced) surface: the 8
    corners, the 12 edge midpoints, and per face its centre and the 4 points at local
    ``(+-1/2, +-1/2)``.  Two elements sharing a face sample exactly the same surface there,
    so a clean adjacency never reads as overlap however warped the face is."""
    C = X[HC]
    mids = C[:, _HEX_EDGES, :].mean(axis=2)                    # (N,12,3)
    fpts = C[:, _FACE_POINTS, :]                               # (N,6,4,3)
    face = np.einsum("sk,nfkd->nfsd", _FACE_W, fpts).reshape(-1, 30, 3)
    return np.concatenate([C, mids, face], axis=1)


def _hex_sample_points(X: PointArray, HC: IntArray) -> PointArray:
    """``(N,20,3)``: each hex's 8 corners plus the centroid of each of its 12
    triangles (the same anchored split ``_hex_geometry`` constrains against, so a
    sample point on a face shared with another hex is exactly what that hex's own
    constraint surface was built from too) -- what
    ``_count_overlapping_pairs`` samples from one hex to test against another.
    Corners alone would miss a face bulging into a neighbour without any corner
    crossing it; a quad face's own *average*-of-4-corners centre is not that
    fallback, though -- on a warped face that average can sit slightly off the true
    (triangulated) surface. A triangle centroid has no such gap: it is exactly on
    its own (necessarily planar) triangle by construction."""
    corners = X[HC]                                          # (N,8,3)
    fpts = _anchored_face_points(X, HC)                        # (N,6,4,3)
    tri1 = fpts[:, :, (0, 1, 2), :].mean(axis=2)               # (N,6,3)
    tri2 = fpts[:, :, (0, 2, 3), :].mean(axis=2)               # (N,6,3)
    return np.concatenate([corners, tri1, tri2], axis=1)       # (N,20,3)


#: Pairs processed per ``_points_inside`` batch. The natural vectorization is one
#: ``(K,20,12,3)`` array per direction -- at the full candidate count of a large mesh
#: (bounding-*sphere* broad phase over-collects badly for cube-like elements, whose
#: corner-to-corner diagonal touches far more neighbours than the faces that matter)
#: that is gigabytes, not a speed cost but a memory-thrashing one; chunking keeps each
#: allocation bounded without changing the answer.
_OVERLAP_BATCH = 50_000


def _points_inside(sample: PointArray, vertex: PointArray, normal: PointArray,
                   tol: float) -> IntArray:
    """``(K,)`` bool: for each pair ``k``, whether any of ``sample[k]``'s points sits
    strictly inside the hex whose 12 triangle vertices/normals are
    ``vertex[k]``/``normal[k]`` (past every triangle's own plane, by ``tol``)."""
    K = sample.shape[0]
    out = np.zeros(K, dtype=bool)
    for lo in range(0, K, _OVERLAP_BATCH):
        hi = min(lo + _OVERLAP_BATCH, K)
        diff = sample[lo:hi, :, None, :] - vertex[lo:hi, None, :, :]   # (k,P,12,3)
        dist = np.einsum("kpfd,kfd->kpf", diff, normal[lo:hi])
        out[lo:hi] = np.any(np.all(dist < -tol, axis=2), axis=1)
    return out


def _aabb_pairs(lo: PointArray, hi: PointArray) -> IntArray:
    """``(K,2)`` pairs ``i < j`` whose axis-aligned boxes ``[lo, hi]`` intersect, each once.

    A uniform hash grid: every box goes into each cell it touches and a pair is reported
    only from the cell holding the low corner of the two boxes' overlap, so nothing is
    generated twice.  The cell size is the cheapest of a few multiples of the median box
    extent (cell entries plus in-cell candidate pairs), which keeps both the long cells of
    a coarse region and the crowded cells of a refined one in check."""
    n = len(lo)
    if n < 2:
        return np.zeros((0, 2), np.int64)
    o = lo.min(0)
    ext = (hi - lo).max(1)
    med = max(float(np.median(ext)), 1e-300)
    best: tuple[float, float] | None = None
    for f in (0.25, 0.35, 0.5, 0.7, 1.0, 1.4, 2.0):
        hh = f * med
        cl = np.floor((lo - o) / hh).astype(np.int64)
        ch = np.floor((hi - o) / hh).astype(np.int64)
        ne = int((ch - cl + 1).prod(1).sum())
        if ne > 60 * n:
            continue
        cen = np.floor(((lo + hi) / 2 - o) / hh).astype(np.int64)
        _, occ = np.unique(_cell_key(cen), return_counts=True)
        est = ne + float(((occ * ne / n) ** 2).sum()) / 2
        if best is None or est < best[0]:
            best = (est, hh)
    h = best[1] if best is not None else 2.0 * med
    clo = np.floor((lo - o) / h).astype(np.int64)
    span = np.floor((hi - o) / h).astype(np.int64) - clo + 1
    cnt = span.prod(1)
    own: IntArray = np.repeat(np.arange(n, dtype=np.int64), cnt)
    r = np.arange(int(cnt.sum()), dtype=np.int64) - np.repeat(np.cumsum(cnt) - cnt, cnt)
    sx: IntArray = np.repeat(span[:, 0], cnt)
    sy: IntArray = np.repeat(span[:, 1], cnt)
    cell = np.column_stack([np.repeat(clo[:, 0], cnt) + r % sx,
                            np.repeat(clo[:, 1], cnt) + (r // sx) % sy,
                            np.repeat(clo[:, 2], cnt) + r // (sx * sy)])
    srt = np.argsort(_cell_key(cell))
    cell, own = cell[srt], own[srt]
    key = _cell_key(cell)
    starts = np.flatnonzero(np.r_[True, key[1:] != key[:-1]])
    sizes = np.diff(np.r_[starts, len(key)])
    out = []
    for m in np.unique(sizes):
        if m < 2:
            continue
        st_all = starts[sizes == m]
        iu, ju = np.triu_indices(int(m), 1)
        step = max(1, 4_000_000 // len(iu))
        for s0 in range(0, len(st_all), step):
            st = st_all[s0:s0 + step]
            grp = own[st[:, None] + np.arange(m)]
            I, J = grp[:, iu].ravel(), grp[:, ju].ravel()
            ok = (lo[I] <= hi[J]).all(1) & (lo[J] <= hi[I]).all(1)
            ref = np.floor((np.maximum(lo[I], lo[J]) - o) / h).astype(np.int64)
            ok &= (ref == np.repeat(cell[st], len(iu), axis=0)).all(1)
            out.append(np.column_stack([np.minimum(I[ok], J[ok]), np.maximum(I[ok], J[ok])]))
    return np.concatenate(out) if out else np.zeros((0, 2), np.int64)


def _cell_key(c: IntArray) -> IntArray:
    c = c + 1
    return np.asarray((c[:, 0] * 2_097_152 + c[:, 1]) * 2_097_152 + c[:, 2])


def _overlap_workers(n: int) -> int:
    from .parallel import default_workers
    return default_workers(n)


#: the 8 corners of the reference cube [-1,1]^3 in Nek corner order
_REF = np.array([[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
                 [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]], dtype=float)


def _inside_trilinear(P: PointArray, C: PointArray, iters: int = 30) -> BoolArray:
    """``(m,)``: is ``P[k]`` strictly inside the trilinear hex with corners ``C[k]`` (Nek
    order) -- Newton on the map, inside when the local coordinates land strictly within
    the reference cube.  The exact geometry of a linear element (what a ``.re2`` is)."""
    xi = np.zeros_like(P)
    ok: BoolArray = np.ones(len(P), dtype=bool)
    for _ in range(iters):
        f = 1.0 + _REF[None, :, :] * xi[:, None, :]                        # (m,8,3)
        N = f.prod(2) / 8.0                                                  # shape functions
        r = np.einsum("mi,mid->md", N, C) - P
        dN = np.stack([_REF[None, :, a] * f[:, :, (a + 1) % 3] * f[:, :, (a + 2) % 3]
                       for a in range(3)], axis=2) / 8.0                     # (m,8,3)
        Jm = np.einsum("mid,mia->mda", C, dN)                                # dx/dxi
        det = np.linalg.det(Jm)
        good = np.abs(det) > 1e-300
        ok &= good
        step = np.zeros_like(P)
        step[good] = np.linalg.solve(Jm[good], r[good][..., None])[..., 0]
        xi = np.clip(xi - step, -2.0, 2.0)
        if np.abs(step).max(initial=0.0) < 1e-12:
            break
    f = 1.0 + _REF[None, :, :] * xi[:, None, :]
    res = np.linalg.norm(np.einsum("mi,mid->md", f.prod(2) / 8.0, C) - P, axis=1)
    h = np.linalg.norm(C.max(1) - C.min(1), axis=1)
    return np.asarray(ok & (res < 1e-9 * h) & (np.abs(xi) < 1.0 - 1e-6).all(1))


def _overlap_chunk(st: dict[str, Any], lo_: int, hi_: int) -> IntArray:
    p = st["P"][lo_:hi_]
    S, N, C, LO, HI, tol, X, HC = (st[k] for k in ("S", "N", "C", "LO", "HI", "tol", "X", "HC"))
    hit: BoolArray = np.zeros(len(p), dtype=bool)
    for a, b in ((0, 1), (1, 0)):
        A, B = p[:, a], p[:, b]
        sa = S[A]                                              # (k,50,3)
        inb = ((sa > LO[B][:, None]) & (sa < HI[B][:, None])).all(2)
        kk, ss = np.nonzero(inb)                               # only samples inside B's box
        if not len(kk):
            continue
        bb = B[kk]
        pts = sa[kk, ss]
        d = (pts[:, None, :] * N[bb]).sum(2) - C[bb]          # signed distance to B's 24 planes
        inn = (d < -tol).reshape(-1, 6, 4)                     # per face: split 1 = [0,1], split 2 = [2,3]
        sure = inn.all((1, 2))                                 # inside under both splits
        maybe = ((inn[:, :, 0] & inn[:, :, 1]) | (inn[:, :, 2] & inn[:, :, 3])).all(1) & ~sure
        hit[kk[sure]] = True
        if maybe.any():                                        # between the splits: the exact trilinear test
            inside = _inside_trilinear(pts[maybe], X[HC[bb[maybe]]])
            hit[kk[maybe][inside]] = True
    return p[hit]


def overlapping_pairs(points: PointArray, hexes: IntArray, *,
                      workers: int | None = None) -> IntArray:
    """``(K,2)`` pairs (``i < j``) of hexes whose volumes geometrically overlap: a point of
    one's surface (``_hex_surface_samples``) sits strictly inside the other as a
    **trilinear** element (the exact geometry of a linear hex).  A sample inside the other's face planes
    under both diagonal splits (``_hex_planes_both``, by ``1e-6`` of the model size) is
    inside; one between the two splits of a warped face is decided by inverting the
    trilinear map (``_inside_trilinear``).  Adjacent pairs are tested
    too -- two elements can share a face and still fold into each other beyond it -- and
    a shared face's own points sit on that boundary, not inside it, so a clean adjacency
    does not trip it.  Two coincident elements (identical centroids) count as overlapping.
    Splitting every face both ways makes the verdict independent of corner numbering.

    Broad phase: an axis-aligned-box hash grid (``_aabb_pairs``); narrow phase: only
    the samples inside the other element's box are tested against its planes.  ``workers``
    splits the narrow phase across forked processes (default:
    :func:`nekmeshpy.core.parallel.default_workers`); the answer does not depend on it."""
    X = np.asarray(points, dtype=float)
    HC = np.asarray(hexes, dtype=np.int64).reshape(-1, 8)
    n = HC.shape[0]
    if n < 2:
        return np.zeros((0, 2), np.int64)
    C8 = X[HC]
    lo, hi = C8.min(1), C8.max(1)
    P = _aabb_pairs(lo, hi)
    if not len(P):
        return P
    scale = conform.bbox_scale(X)
    tol = 1e-6 * (scale if scale > 0.0 else 1.0)
    from .parallel import chunk_map
    vertex, normal = _hex_planes_both(X, HC)
    state = dict(P=P, S=_hex_surface_samples(X, HC), N=normal, LO=lo, HI=hi, tol=tol, X=X,
                 HC=HC, C=np.einsum("efd,efd->ef", vertex, normal))
    del vertex
    res = chunk_map(_overlap_chunk, len(P), 100_000, state,
                    workers if workers is not None else _overlap_workers(n))
    cen = C8.mean(1)
    dup = P[np.linalg.norm(cen[P[:, 0]] - cen[P[:, 1]], axis=1) < tol]
    out = np.concatenate(res + [dup])
    if not len(out):
        return np.zeros((0, 2), np.int64)
    return np.unique(out, axis=0)


# -- hex (volume) meshes ------------------------------------------------
def hex_report(points: PointArray, hexes: IntArray, hex_lines: IntArray | None = None,
               hex_faces: IntArray | None = None) -> TopologyReport:
    """Topology / watertightness report for an all-hex mesh.  ``hex_lines`` /
    ``hex_faces`` are each hex's stored line and face ids, which
    :attr:`TopologyReport.n_improper` reads; without them they are derived from the
    corners, which cannot tell two stored lines on the same corners apart."""
    X = np.asarray(points, dtype=float)
    HC = np.asarray(hexes, dtype=np.int64).reshape(-1, 8)
    N = HC.shape[0]
    faces = HC[:, _FACE_POINTS].reshape(N * 6, 4)          # cyclic-order faces
    keys = np.sort(faces, axis=1)                         # orientation-free key
    _, inverse, counts = conform.unique_rows(keys, return_counts=True)
    if hex_faces is None:
        hex_faces = inverse.reshape(N, 6)
    if hex_lines is None:
        hex_lines = conform.unique_edges(HC, 3)[1]
    pairs, cells = improper_pairs(HC, hex_lines, hex_faces)
    n_improper = int(len(pairs) + len(cells))

    n_boundary = int(np.sum(counts == 1))
    n_internal = int(np.sum(counts == 2))
    n_nonmanifold = int(np.sum(counts >= 3))

    # boundary surface = the once-used faces, in their true cyclic order
    bmask = counts[inverse] == 1
    bfaces = faces[bmask]
    be = np.concatenate([bfaces[:, [0, 1]], bfaces[:, [1, 2]],
                         bfaces[:, [2, 3]], bfaces[:, [3, 0]]], axis=0)
    be = np.sort(be, axis=1)
    if be.size:
        ube, _, bec = conform.unique_rows(be, return_counts=True)
    else:
        ube, bec = be.reshape(0, 2), np.zeros(0, np.int64)
    n_open_edges = int(np.sum(bec % 2 == 1))
    n_pinch_edges = int(np.sum((bec % 2 == 0) & (bec >= 4)))

    # hanging points: a boundary point interior to any boundary edge
    bpoints = np.unique(bfaces) if bfaces.size else np.zeros(0, np.int64)
    n_hanging = _count_hanging_points(X, ube, bpoints)

    owner: IntArray = np.repeat(np.arange(N, dtype=np.int64), 6)
    n_components = _count_components(N, _adjacency_edges(inverse, counts, owner))

    watertight = bool(n_nonmanifold == 0 and n_open_edges == 0 and n_pinch_edges == 0
                      and n_boundary > 0)
    return TopologyReport(
        n_elements=N,
        n_faces=int(counts.size),
        n_boundary_faces=n_boundary,
        n_internal_faces=n_internal,
        n_nonmanifold_faces=n_nonmanifold,
        n_open_edges=n_open_edges,
        n_hanging_points=n_hanging,
        n_components=n_components,
        watertight=watertight,
        conformal=bool(n_hanging == 0 and n_improper == 0),
        n_improper=n_improper,
        n_pinch_edges=n_pinch_edges,
    )


def is_watertight(points: PointArray, hexes: IntArray) -> bool:
    """``True`` if the all-hex mesh's boundary is a closed 2-manifold."""
    return hex_report(points, hexes).watertight


def count_overlapping_pairs(points: PointArray, hexes: IntArray, *,
                            workers: int | None = None) -> int:
    """Number of hex pairs whose volumes geometrically overlap (see
    :func:`overlapping_pairs`)."""
    return int(len(overlapping_pairs(points, hexes, workers=workers)))


def is_overlap_free(points: PointArray, hexes: IntArray) -> bool:
    """``True`` if no two hexes geometrically overlap (see
    :func:`count_overlapping_pairs`)."""
    return count_overlapping_pairs(points, hexes) == 0


# -- quad (section) meshes ----------------------------------------------
_QUAD_EDGE_POINTS = np.array([[0, 1], [1, 2], [2, 3], [3, 0]], dtype=np.int64)


def quad_report(points: PointArray, quads: IntArray,
                quad_lines: IntArray | None = None) -> TopologyReport:
    """Topology / watertightness report for an all-quad mesh -- :func:`hex_report` one
    rung down: the facets are edges, the boundary is a curve, and its "edges" are
    boundary **points**: :attr:`TopologyReport.n_open_edges` counts open points (an odd
    number of boundary edges -- broken input) and :attr:`TopologyReport.n_pinch_edges`
    pinch points (four or more: quads meeting only at a corner)."""
    X = np.asarray(points, dtype=float)
    QC = np.asarray(quads, dtype=np.int64).reshape(-1, 4)
    N = QC.shape[0]
    edges = QC[:, _QUAD_EDGE_POINTS].reshape(N * 4, 2)
    keys = np.sort(edges, axis=1)
    _, inverse, counts = conform.unique_rows(keys, return_counts=True)

    n_boundary = int(np.sum(counts == 1))
    n_internal = int(np.sum(counts == 2))
    n_nonmanifold = int(np.sum(counts >= 3))

    bmask = counts[inverse] == 1
    bedges = keys[bmask]
    if bedges.size:
        _, bpc = np.unique(bedges, return_counts=True)
        n_open = int(np.sum(bpc % 2 == 1))
        n_pinch = int(np.sum((bpc % 2 == 0) & (bpc >= 4)))
        bpoints = np.unique(bedges)
    else:
        n_open, n_pinch, bpoints = 0, 0, np.zeros(0, np.int64)
    n_hanging = _count_hanging_points(X, bedges, bpoints)
    pairs, cells = improper_pairs(QC, inverse.reshape(N, 4) if quad_lines is None
                                  else quad_lines)
    n_improper = int(len(pairs) + len(cells))

    owner: IntArray = np.repeat(np.arange(N, dtype=np.int64), 4)
    n_components = _count_components(N, _adjacency_edges(inverse, counts, owner))

    watertight = bool(n_nonmanifold == 0 and n_open == 0 and n_pinch == 0 and n_boundary > 0)
    return TopologyReport(
        n_elements=N,
        n_faces=int(counts.size),
        n_boundary_faces=n_boundary,
        n_internal_faces=n_internal,
        n_nonmanifold_faces=n_nonmanifold,
        n_open_edges=n_open,
        n_hanging_points=n_hanging,
        n_components=n_components,
        watertight=watertight,
        conformal=bool(n_hanging == 0 and n_improper == 0),
        n_improper=n_improper,
        n_pinch_edges=n_pinch,
    )


def _quad_geometry(X: PointArray, QC: IntArray) -> tuple[
        PointArray, PointArray, PointArray, PointArray, PointArray, PointArray]:
    """Per-quad centroid, bounding-circle radius, the ``(N,4,3)`` edge start points
    and outward in-plane unit normals, the ``(N,9,3)`` sample points (4 corners,
    4 edge midpoints, the centroid) and the ``(N,3)`` unit normal of the quad's own plane.  The plane is each quad's own mean-direction
    normal, so a section placed anywhere in space is read in its own plane; the
    outward sign comes from the centroid, not the stored winding."""
    corners = X[QC]                                              # (N,4,3)
    centroid = corners.mean(axis=1)
    radius = np.linalg.norm(corners - centroid[:, None, :], axis=2).max(axis=1)
    nxt = np.roll(corners, -1, axis=1)
    edge = nxt - corners                                         # (N,4,3)
    plane = np.cross(corners - centroid[:, None, :], nxt - centroid[:, None, :]).sum(
        axis=1)
    plane_len = np.linalg.norm(plane, axis=1, keepdims=True)
    plane = plane / np.where(plane_len == 0.0, 1.0, plane_len)
    normal = np.cross(edge, plane[:, None, :])
    mid = 0.5 * (corners + nxt)
    sign = np.sign(np.einsum("nfd,nfd->nf", normal, mid - centroid[:, None, :]))
    sign[sign == 0.0] = 1.0
    normal = normal * sign[..., None]
    length = np.linalg.norm(normal, axis=-1, keepdims=True)
    normal = normal / np.where(length == 0.0, 1.0, length)
    samples = np.concatenate([corners, mid, centroid[:, None, :]], axis=1)
    return centroid, radius, corners, normal, samples, plane


#: How far out of a quad's own plane a point may lie, in units of the quad's bounding radius, and still
#: count as overlapping it.  Zero for a planar section; a curved surface's overlap is within its sagitta.
_OVERLAP_PLANE = 0.5


def _overlapping_quad_pairs(X: PointArray, QC: IntArray) -> IntArray:
    """``(K, 2)`` pairs of quads whose areas geometrically overlap -- ``_count_overlapping_pairs``
    in 2-D: any of one's corners, edge midpoints or centroid sits strictly inside the
    other, past a half-plane test on all 4 of its edges (each quad is star-convex about
    its own centroid).  Adjacent pairs are not excluded; a shared edge's points sit
    exactly on the boundary, not strictly inside it.  A coincident centroid is a
    duplicate."""
    none: IntArray = np.zeros((0, 2), dtype=np.int64)
    N = QC.shape[0]
    if N < 2:
        return none
    centroid, radius, vertex, normal, samples, plane = _quad_geometry(X, QC)
    scale = conform.bbox_scale(X)
    tol = 1e-6 * (scale if scale > 0.0 else 1.0)

    tree = cKDTree(centroid)
    pairs = tree.query_pairs(r=2.0 * float(radius.max()), output_type="ndarray")
    if pairs.size == 0:
        return none
    d = np.linalg.norm(centroid[pairs[:, 0]] - centroid[pairs[:, 1]], axis=1)
    keep = d < (radius[pairs[:, 0]] + radius[pairs[:, 1]])
    pairs, d = pairs[keep], d[keep]
    if pairs.size == 0:
        return none

    i, j = pairs[:, 0], pairs[:, 1]
    hit = np.zeros(i.size, dtype=bool)
    for lo in range(0, i.size, _OVERLAP_BATCH):
        sl = slice(lo, lo + _OVERLAP_BATCH)
        for a, b in ((i[sl], j[sl]), (j[sl], i[sl])):
            diff = samples[a][:, :, None, :] - vertex[b][:, None, :, :]   # (k,9,4,3)
            dist = np.einsum("kpfd,kfd->kpf", diff, normal[b])
            # ... and close to the other quad's plane: on a surface that wraps back on itself (a coarse pipe wall)
            # the far side projects into the footprint without touching it
            off = np.abs(np.einsum("kpd,kd->kp", samples[a] - centroid[b][:, None, :], plane[b]))
            hit[sl] |= np.any(np.all(dist < -tol, axis=2) & (off <= _OVERLAP_PLANE * radius[b][:, None]), axis=1)
    return np.asarray(pairs[hit | (d < tol)], dtype=np.int64)


def _count_overlapping_quad_pairs(X: PointArray, QC: IntArray) -> int:
    return int(_overlapping_quad_pairs(X, QC).shape[0])


def overlapping_quads(points: PointArray, quads: IntArray) -> IntArray:
    """Sorted ids of the quads that take part in at least one overlapping pair (see
    :func:`count_overlapping_quad_pairs`): the *which* behind its count."""
    pairs = _overlapping_quad_pairs(np.asarray(points, dtype=float),
                                    np.asarray(quads, dtype=np.int64).reshape(-1, 4))
    return np.unique(pairs.ravel()).astype(np.int64)


def count_overlapping_quad_pairs(points: PointArray, quads: IntArray) -> int:
    """Number of quad pairs whose areas geometrically overlap -- independent of, and
    not part of, :func:`quad_report`, for the same cost reason as
    :func:`count_overlapping_pairs`."""
    return _count_overlapping_quad_pairs(
        np.asarray(points, dtype=float),
        np.asarray(quads, dtype=np.int64).reshape(-1, 4))


def is_overlap_free_quads(points: PointArray, quads: IntArray) -> bool:
    """``True`` if no two quads geometrically overlap (see
    :func:`count_overlapping_quad_pairs`)."""
    return count_overlapping_quad_pairs(points, quads) == 0


class TagReport(NamedTuple):
    """How a mesh's side-tag table lines up with its **topological** boundary -- the
    same report at the hex rung (faces) and the quad rung (edges).

    The two are independent by design -- ``boundary`` is what connectivity says, a
    side-tag table is a named subset that may also name interior entities -- so they can
    disagree in exactly two ways, and both are worth seeing. An
    :attr:`n_untagged_boundary` above zero means the export has open sides no boundary
    condition covers; an :attr:`n_tagged_interior` above zero means it will write
    boundary conditions onto sides that are not on the boundary at all. Either can be
    deliberate (a flux-measurement plane is a tagged interior face; so is a conjugate
    fluid/solid interface that keeps the fluid's wall condition), so these are counts
    to recognise, not assertions."""

    #: Named sides. One tag per shared entity, so this counts entities, not rows -- the
    #: two could differ only while a tag was addressed by ``(element, side)``.
    n_rows: int
    #: Sides on the topological boundary that carry no name.
    n_untagged_boundary: int
    #: Named sides that are **not** on the topological boundary.
    n_tagged_interior: int


# -- triangle (surface) meshes ------------------------------------------
def surface_report(points: PointArray, tris: IntArray) -> dict[str, Any]:
    """Topology report for a triangle surface mesh."""
    T = np.asarray(tris, dtype=np.int64).reshape(-1, 3)
    M = T.shape[0]
    edges = np.concatenate([T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]], axis=0)
    keys = np.sort(edges, axis=1)
    # id rows, so the packed-key fast path applies -- identical result, and
    # ``np.unique(axis=0)`` argsorts each row as a void scalar instead
    uniq, inverse, counts = conform.unique_rows(keys, return_counts=True)
    inverse = inverse.ravel()

    n_boundary = int(np.sum(counts == 1))
    n_interior = int(np.sum(counts == 2))
    n_nonmanifold = int(np.sum(counts >= 3))

    # boundary loops = components of the once-used-edge graph
    bedges = uniq[counts == 1]
    if bedges.size:
        bverts = np.unique(bedges)
        remap = {int(v): i for i, v in enumerate(bverts)}
        be = np.array([[remap[int(a)], remap[int(b)]] for a, b in bedges],
                      dtype=np.int64)
        n_loops = _count_components(bverts.size, be)
    else:
        n_loops = 0

    owner: IntArray = np.tile(np.arange(M, dtype=np.int64), 3)
    n_components = _count_components(M, _adjacency_edges(inverse, counts, owner))

    closed = bool(n_boundary == 0 and n_nonmanifold == 0 and M > 0)
    return {
        "kind": "surface",
        "n_faces": M,
        "n_edges": int(counts.size),
        "n_boundary_edges": n_boundary,
        "n_interior_edges": n_interior,
        "n_nonmanifold_edges": n_nonmanifold,
        "n_boundary_loops": n_loops,
        "n_components": n_components,
        "closed": closed,
    }


def is_closed(points: PointArray, tris: IntArray) -> bool:
    """``True`` if the surface is a closed 2-manifold."""
    return bool(surface_report(points, tris)["closed"])


# -- reporting ----------------------------------------------------------
def format_report(report: Union[TopologyReport, dict[str, Any]],
                  kind: str = "hex") -> str:
    """Human-readable multi-line summary of a hex, quad or surface report.

    ``kind`` (``"hex"`` or ``"quad"``) only changes the vocabulary: a quad mesh's
    facets are edges, and its boundary's edges are points."""
    if isinstance(report, TopologyReport):
        quad = kind == "quad"
        return "\n".join([
            "quad elements  : %d" % report.n_elements if quad
            else "hex elements   : %d" % report.n_elements,
            ("edges          : %d (%d boundary, %d internal, %d non-manifold)" if quad
             else "faces          : %d (%d boundary, %d internal, %d non-manifold)")
            % (report.n_faces, report.n_boundary_faces,
               report.n_internal_faces, report.n_nonmanifold_faces),
            ("pinch points   : %d (quads meeting only at a corner)" if quad
             else "pinch edges    : %d (cells meeting only along an edge)")
            % report.n_pinch_edges,
            ("open points    : %d (odd boundary count: broken input)" if quad
             else "open edges     : %d (odd boundary count: broken input)")
            % report.n_open_edges,
            "hanging points  : %d" % report.n_hanging_points,
            "improper       : %d (cells not meeting in one point, line or face)"
            % report.n_improper,
            "components     : %d" % report.n_components,
            "watertight     : %s" % report.watertight,
            "conformal      : %s" % report.conformal,
        ])
    return "\n".join([
        "triangles      : %d" % report["n_faces"],
        "edges          : %d (%d boundary, %d interior, %d non-manifold)"
        % (report["n_edges"], report["n_boundary_edges"],
           report["n_interior_edges"], report["n_nonmanifold_edges"]),
        "boundary loops : %d" % report["n_boundary_loops"],
        "components     : %d" % report["n_components"],
        "closed         : %s" % report["closed"],
    ])
