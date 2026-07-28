"""Mesh topology / validity checks, decoupled from the mesh containers.

Two families of free functions, both operating on the shared-point representation
so they work on a welded :class:`~nekmeshpy.hexmesh.HexMesh` or a
:class:`~nekmeshpy.trimesh.TriMesh` (mirroring :mod:`nekmeshpy.hexmesh.quality`):

* :func:`hex_report` / :func:`is_watertight` -- all-hex volume meshes.  A hex
  mesh is *watertight* when its boundary (every quad face carried by a single
  hex) is a closed 2-manifold: each boundary-face edge is shared by exactly two
  boundary faces, and no face is shared by three or more hexes.  A crack or a
  hole breaks that.  Separately, :func:`hex_report` flags *non-conformal*
  T-junctions (a hanging point sitting inside a coarse face's edge): those leave
  the boundary closed -- so they are watertight -- but invalid for a conforming
  hex solver, and are reported as ``conformal=False``.
* :func:`surface_report` / :func:`is_closed` -- triangle surface meshes.  A
  surface is *closed* when every edge is shared by exactly two triangles (no
  boundary edges, no non-manifold edges).

Both reports also count connected components, so a mesh that silently splits into
disconnected pieces is caught.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .._typing import IntArray, PointArray

# Nek face -> the 4 corner point positions (0-based), cyclic order (matches
# HexMesh.FACE_POINTS; kept local so this module stays independent of geometry/).
_FACE_POINTS = np.array([[0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6],
                        [3, 0, 4, 7], [0, 1, 2, 3], [4, 5, 6, 7]], dtype=np.int64)


# -- shared helpers -----------------------------------------------------
def _count_components(n: int, edges: IntArray) -> int:
    """Number of connected components of an ``n``-point graph whose undirected
    edges are the rows of ``edges`` (an ``(E,2)`` int array)."""
    parent: IntArray = np.arange(n, dtype=np.int64)

    def find(x: int) -> int:
        root = x
        while parent[root] != root:
            root = int(parent[root])
        while parent[x] != root:
            parent[x], x = root, int(parent[x])
        return root

    for a, b in np.asarray(edges, dtype=np.int64).reshape(-1, 2):
        ra, rb = find(int(a)), find(int(b))
        if ra != rb:
            parent[ra] = rb
    return int(np.unique([find(i) for i in range(n)]).size) if n else 0


def _adjacency_edges(inverse: IntArray, counts: IntArray,
                     owner: IntArray) -> IntArray:
    """Cell-adjacency edges from a face/edge incidence.  ``inverse`` maps each
    incidence row to its unique-facet id, ``counts`` is the per-facet
    multiplicity, and ``owner`` maps each incidence row to its owning cell.
    Facets shared by exactly two cells contribute one cell-cell edge."""
    order = np.argsort(inverse, kind="stable")
    inv_s = inverse[order]
    owner_s = owner[order]
    if inv_s.size == 0:
        return np.zeros((0, 2), dtype=np.int64)
    groups = np.split(owner_s, np.flatnonzero(np.diff(inv_s)) + 1)
    pairs = [g for g in groups if g.size == 2]
    return np.array(pairs, dtype=np.int64) if pairs else np.zeros((0, 2), np.int64)


def _count_hanging_points(points: PointArray, edges: IntArray,
                         candidates: IntArray, rtol: float = 1e-7) -> int:
    """Number of ``candidates`` points that lie strictly inside an ``edges`` edge
    without being one of its endpoints -- the geometric signature of a
    non-conforming (T-junction / hanging-point) interface.

    ``edges`` are point-index pairs; ``candidates`` are point indices to test.
    A conforming mesh has none (every point is an edge endpoint).  Detection is
    on edges only, which is the reliable signature of standard 1->N refinement
    (a subdivided face always plants points on the coarse edges); a point interior
    to a coarse *face* but on no edge is not counted.
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
    hanging: set[int] = set()
    for i, j in E:
        i, j = int(i), int(j)
        a, b = X[i], X[j]
        d = b - a
        L2 = float(d @ d)
        if L2 <= tol * tol:
            continue
        for c in tree.query_ball_point(0.5 * (a + b), 0.5 * float(np.sqrt(L2)) + tol):
            k = int(cand[c])
            if k == i or k == j:
                continue
            t = float((Xc[c] - a) @ d) / L2
            if t <= 1e-9 or t >= 1.0 - 1e-9:
                continue
            if float(np.linalg.norm(Xc[c] - (a + t * d))) <= tol:
                hanging.add(k)
    return len(hanging)


# -- hex (volume) meshes ------------------------------------------------
def hex_report(points: PointArray, hexes: IntArray) -> dict[str, Any]:
    """Topology / watertightness report for an all-hex mesh.

    ``points`` is ``(P,3)`` and ``hexes`` is ``(N,8)`` in Nek corner order
    (a welded :class:`~nekmeshpy.hexmesh.HexMesh` view).  Returns a dict with the facet inventory
    (``n_faces`` unique quad faces, split into ``n_boundary_faces`` /
    ``n_internal_faces`` / ``n_nonmanifold_faces``), the number of ``n_open_edges``
    on the boundary surface (edges not shared by exactly two boundary faces),
    ``n_hanging_points`` (points sitting inside a boundary edge -- a non-conforming
    T-junction), ``n_components`` (hexes joined through shared internal faces),
    and two verdicts: ``watertight`` (closed, leak-tight boundary) and
    ``conformal`` (no hanging points).  A T-junction is typically *watertight but
    not conformal* -- its faces still pair into a closed boundary, so the
    hanging-point test is what catches it.
    """
    X = np.asarray(points, dtype=float)
    HC = np.asarray(hexes, dtype=np.int64).reshape(-1, 8)
    N = HC.shape[0]
    faces = HC[:, _FACE_POINTS].reshape(N * 6, 4)          # cyclic-order faces
    keys = np.sort(faces, axis=1)                         # orientation-free key
    _, inverse, counts = np.unique(keys, axis=0, return_inverse=True,
                                   return_counts=True)
    inverse = inverse.ravel()

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
        ube, bec = np.unique(be, axis=0, return_counts=True)
    else:
        ube, bec = be.reshape(0, 2), np.zeros(0, np.int64)
    n_open_edges = int(np.sum(bec != 2))

    # hanging points: a boundary point interior to *any* boundary edge (tested on
    # all boundary edges, not just open ones -- a T-junction leaves no open edge)
    bpoints = np.unique(bfaces) if bfaces.size else np.zeros(0, np.int64)
    n_hanging = _count_hanging_points(X, ube, bpoints)

    owner: IntArray = np.repeat(np.arange(N, dtype=np.int64), 6)
    n_components = _count_components(N, _adjacency_edges(inverse, counts, owner))

    watertight = bool(n_nonmanifold == 0 and n_open_edges == 0 and n_boundary > 0)
    return {
        "kind": "hex",
        "n_elements": N,
        "n_faces": int(counts.size),
        "n_boundary_faces": n_boundary,
        "n_internal_faces": n_internal,
        "n_nonmanifold_faces": n_nonmanifold,
        "n_open_edges": n_open_edges,
        "n_hanging_points": n_hanging,
        "n_components": n_components,
        "watertight": watertight,
        "conformal": bool(n_hanging == 0),
    }


def is_watertight(points: PointArray, hexes: IntArray) -> bool:
    """``True`` if the all-hex mesh's boundary is a closed 2-manifold with no
    non-conformal faces (see :func:`hex_report`)."""
    return bool(hex_report(points, hexes)["watertight"])


# -- triangle (surface) meshes ------------------------------------------
def surface_report(points: PointArray, tris: IntArray) -> dict[str, Any]:
    """Topology report for a triangle surface mesh.

    ``points`` is ``(nv,3)`` and ``tris`` is ``(nt,3)``.  Returns a dict with the
    edge inventory (``n_edges`` unique edges, split into ``n_boundary_edges`` /
    ``n_interior_edges`` / ``n_nonmanifold_edges``), ``n_boundary_loops``,
    ``n_components`` (triangles joined through shared edges), and the ``closed``
    verdict (no boundary and no non-manifold edges).
    """
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
    """``True`` if the surface is a closed 2-manifold (see :func:`surface_report`)."""
    return bool(surface_report(points, tris)["closed"])


# -- reporting ----------------------------------------------------------
def format_report(report: dict[str, Any]) -> str:
    """Human-readable multi-line summary of a :func:`hex_report` or
    :func:`surface_report` result."""
    if report.get("kind") == "hex":
        return "\n".join([
            "hex elements   : %d" % report["n_elements"],
            "faces          : %d (%d boundary, %d internal, %d non-manifold)"
            % (report["n_faces"], report["n_boundary_faces"],
               report["n_internal_faces"], report["n_nonmanifold_faces"]),
            "open edges     : %d" % report["n_open_edges"],
            "hanging points  : %d" % report["n_hanging_points"],
            "components     : %d" % report["n_components"],
            "watertight     : %s" % report["watertight"],
            "conformal      : %s" % report["conformal"],
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
