"""Mesh topology / validity checks, decoupled from the mesh containers."""

from __future__ import annotations

from typing import Any, NamedTuple, Union

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

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
    scale = float(np.max(X.max(axis=0) - X.min(axis=0)))
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


# -- triangle (surface) meshes ------------------------------------------
def surface_report(points: PointArray, tris: IntArray) -> dict[str, Any]:
    """Topology report for a triangle surface mesh."""
    T = np.asarray(tris, dtype=np.int64).reshape(-1, 3)
    M = T.shape[0]
    edges = np.concatenate([T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]], axis=0)
    keys = np.sort(edges, axis=1)
    uniq, inverse, counts = np.unique(keys, axis=0, return_inverse=True,
                                      return_counts=True)
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
