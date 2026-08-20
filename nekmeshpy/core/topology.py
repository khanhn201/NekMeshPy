"""Mesh topology / validity checks, decoupled from the mesh containers."""

from __future__ import annotations

from typing import Any, NamedTuple, Union

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from .._typing import IntArray, PointArray
from . import conform

# Nek face -> the 4 corner point positions (0-based), cyclic order.
_FACE_POINTS = np.array([[0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6],
                        [3, 0, 4, 7], [0, 1, 2, 3], [4, 5, 6, 7]], dtype=np.int64)


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
    #: Boundary-surface edges not bounded by exactly two boundary faces (leaks).
    n_open_edges: int
    #: Boundary points lying strictly inside another boundary edge (T-junctions).
    n_hanging_points: int
    #: Connected components of the hex-adjacency graph.
    n_components: int
    #: ``True`` when the boundary is a closed, non-empty 2-manifold.  Says nothing
    #: about conformity: a T-junction interface still reads watertight.
    watertight: bool
    #: ``True`` when there are no hanging points.
    conformal: bool


# -- shared helpers -----------------------------------------------------
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


#: Pairs processed per :func:`_points_inside` batch. The natural vectorization is one
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


def _count_overlapping_pairs(X: PointArray, HC: IntArray) -> int:
    """Pairs of hexes whose volumes geometrically overlap: any of one's own corners
    or triangle centroids sits strictly inside the other, past a half-space test on
    all 12 of its (triangulated) face planes (assumes each hex is star-convex about
    its own centroid, true of any non-degenerate element). Adjacent (face-sharing)
    pairs are not excluded -- two
    elements can share a face *and* still fold into each other beyond it, which is
    exactly the kind of defect this is meant to catch; their shared face's own corners
    sit exactly on that boundary (distance ~0) rather than strictly inside, so a
    normal, non-overlapping adjacency does not falsely trip this on its own."""
    N = HC.shape[0]
    if N < 2:
        return 0
    centroid, radius, vertex, normal = _hex_geometry(X, HC)
    samples = _hex_sample_points(X, HC)
    scale = conform.bbox_scale(X)
    tol = 1e-6 * (scale if scale > 0.0 else 1.0)

    tree = cKDTree(centroid)
    pairs = tree.query_pairs(r=2.0 * float(radius.max()), output_type="ndarray")
    if pairs.size == 0:
        return 0
    d = np.linalg.norm(centroid[pairs[:, 0]] - centroid[pairs[:, 1]], axis=1)
    keep = d < (radius[pairs[:, 0]] + radius[pairs[:, 1]])
    pairs, d = pairs[keep], d[keep]
    if pairs.size == 0:
        return 0

    i, j = pairs[:, 0], pairs[:, 1]
    i_in_j = _points_inside(samples[i], vertex[j], normal[j], tol)
    j_in_i = _points_inside(samples[j], vertex[i], normal[i], tol)
    # a perfect duplicate (identical geometry stacked on itself) has every one of its
    # sample points sitting exactly *on* the other's boundary rather than strictly
    # inside it, which the half-space test above cannot see -- coincident centroids
    # are the signature of that degenerate case, since two distinct non-overlapping
    # elements never share one.
    coincident = d < tol
    return int(np.count_nonzero(i_in_j | j_in_i | coincident))


# -- hex (volume) meshes ------------------------------------------------
def hex_report(points: PointArray, hexes: IntArray) -> TopologyReport:
    """Topology / watertightness report for an all-hex mesh."""
    X = np.asarray(points, dtype=float)
    HC = np.asarray(hexes, dtype=np.int64).reshape(-1, 8)
    N = HC.shape[0]
    faces = HC[:, _FACE_POINTS].reshape(N * 6, 4)          # cyclic-order faces
    keys = np.sort(faces, axis=1)                         # orientation-free key
    _, inverse, counts = conform.unique_rows(keys, return_counts=True)

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
    n_open_edges = int(np.sum(bec != 2))

    # hanging points: a boundary point interior to any boundary edge
    bpoints = np.unique(bfaces) if bfaces.size else np.zeros(0, np.int64)
    n_hanging = _count_hanging_points(X, ube, bpoints)

    owner: IntArray = np.repeat(np.arange(N, dtype=np.int64), 6)
    n_components = _count_components(N, _adjacency_edges(inverse, counts, owner))

    watertight = bool(n_nonmanifold == 0 and n_open_edges == 0 and n_boundary > 0)
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
        conformal=bool(n_hanging == 0),
    )


def is_watertight(points: PointArray, hexes: IntArray) -> bool:
    """``True`` if the all-hex mesh's boundary is a closed 2-manifold."""
    return hex_report(points, hexes).watertight


def count_overlapping_pairs(points: PointArray, hexes: IntArray) -> int:
    """Number of hex pairs whose volumes geometrically overlap (see
    ``_count_overlapping_pairs``) -- independent of, and not part of,
    :func:`hex_report`: unlike watertight/conformal (a fixed-size facet-incidence
    scan), this is a geometric broad-then-narrow-phase search whose candidate count
    can run into the hundreds of thousands on a large mesh, so it is not folded into
    the fast report every :func:`is_watertight`/:func:`hexmesh.is_conforming
    <nekmeshpy.hexmesh.query.is_conforming>` call already pays for -- call it
    explicitly where the extra work is wanted (a summary, typically, alongside
    those two)."""
    X = np.asarray(points, dtype=float)
    HC = np.asarray(hexes, dtype=np.int64).reshape(-1, 8)
    return _count_overlapping_pairs(X, HC)


def is_overlap_free(points: PointArray, hexes: IntArray) -> bool:
    """``True`` if no two hexes geometrically overlap (see
    :func:`count_overlapping_pairs`)."""
    return count_overlapping_pairs(points, hexes) == 0


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
def format_report(report: Union[TopologyReport, dict[str, Any]]) -> str:
    """Human-readable multi-line summary of a hex or surface report."""
    if isinstance(report, TopologyReport):
        return "\n".join([
            "hex elements   : %d" % report.n_elements,
            "faces          : %d (%d boundary, %d internal, %d non-manifold)"
            % (report.n_faces, report.n_boundary_faces,
               report.n_internal_faces, report.n_nonmanifold_faces),
            "open edges     : %d" % report.n_open_edges,
            "hanging points  : %d" % report.n_hanging_points,
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
