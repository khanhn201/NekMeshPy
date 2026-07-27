"""Triangulated surface mesh container.

``TriMesh`` is a pure container of a surface triangulation: point coordinates
``points`` (nv,3) and triangle connectivity ``tris`` (nt,3), both 0-based (input
``.tri`` files are 1-based and converted on load).  It is the triangle sibling of
:class:`~nekmeshpy.geometry.quadmesh.QuadMesh` (``points`` + ``quads``): same
``points`` coordinate array, an integer cell array named for its element type, and
matching ``n_points`` / ``n_<cell>`` size properties.

The surface *algorithms* -- cotangent Laplace operators, Dirichlet solves,
boundary-loop extraction, marching-triangle isocontours, and closest-point
projection -- live in :mod:`nekmeshpy.ops.trisurf` as free functions taking the
surface as their first argument.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .._typing import BoolArray, IntArray, PointArray


class TriMesh:
    def __init__(self, points: PointArray, tris: IntArray) -> None:
        self.points = np.asarray(points, dtype=float).reshape(-1, 3)
        self.tris = np.asarray(tris, dtype=np.int64)

    # -- construction ----------------------------------------------------
    @classmethod
    def from_files(cls, vtx_file: str, tri_file: str) -> TriMesh:
        """Load a surface triangulation (point-list + 1-based triangle-index
        files); triangle indices are converted to 0-based."""
        points = np.loadtxt(vtx_file, dtype=float)
        tris = np.loadtxt(tri_file, dtype=float).astype(np.int64)
        if points.ndim == 1:
            points = points.reshape(1, -1)
        if tris.ndim == 1:
            tris = tris.reshape(1, -1)
        return cls(points, tris - 1)

    @classmethod
    def from_faces(cls, V: PointArray, faces: IntArray) -> tuple[TriMesh, IntArray]:
        """Build a sub-mesh from vertex set ``V`` restricted to ``faces`` (a
        triangle list indexing into ``V``), compacting to the used vertices.
        Returns ``(TriMesh, vids)`` where ``vids`` maps sub-index -> V-index."""
        faces = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
        vids = np.unique(faces.ravel())
        remap = np.zeros(np.asarray(V).shape[0], dtype=np.int64)
        remap[vids] = np.arange(vids.size)
        return cls(np.asarray(V, dtype=float)[vids, :], remap[faces]), vids

    # local triangle edges; row e is edge e+1
    EDGE_POINTS = np.array([[0, 1], [1, 2], [2, 0]], dtype=np.int64)

    # -- sizes -----------------------------------------------------------
    @property
    def n_points(self) -> int:
        return self.points.shape[0]

    @property
    def n_tris(self) -> int:
        return self.tris.shape[0]

    # -- boundary queries (open surface edges) --------------------------
    @staticmethod
    def _boundary_mask(tris: IntArray) -> tuple[IntArray, BoolArray]:
        """``(edges, is_boundary)``: every triangle edge ``(3M,2)``, element-major
        (row ``3t+e`` is triangle ``t``, local edge ``e``), and a mask of those
        borne by a single triangle (the open boundary)."""
        T = np.asarray(tris, dtype=np.int64).reshape(-1, 3)
        edges: IntArray = T[:, TriMesh.EDGE_POINTS].reshape(-1, 2)
        keys = np.sort(edges, axis=1)
        _, inverse, counts = np.unique(
            keys, axis=0, return_inverse=True, return_counts=True)
        return edges, counts[inverse.ravel()] == 1

    def boundary_edges(self) -> IntArray:
        """``(K,2)`` array of ``[triangle id, local edge (1-3)]`` for every edge on
        the open boundary (an edge borne by a single triangle); empty for a closed
        surface.  An edge's point ids are ``self.tris[t, self.EDGE_POINTS[e - 1]]``."""
        _, mask = self._boundary_mask(self.tris)
        rows = np.flatnonzero(mask)
        return np.column_stack([rows // 3, rows % 3 + 1]).astype(np.int64)

    def boundary_elements(self) -> IntArray:
        """Sorted unique triangle ids with at least one edge on the open boundary."""
        return np.unique(self.boundary_edges()[:, 0])

    def boundary_points(self) -> IntArray:
        """Sorted unique point ids lying on the open boundary."""
        edges, mask = self._boundary_mask(self.tris)
        be = edges[mask]
        return np.unique(be) if be.size else np.zeros(0, dtype=np.int64)

    def boundary_loops(self) -> list[IntArray]:
        """The open boundary grouped into loops -- one array of vertex ids per
        connected component of the boundary edges (BFS order); empty for a closed
        surface.  Unlike :meth:`boundary_points` (a flat set), this separates the
        distinct openings.  (Delegates to
        :func:`nekmeshpy.ops.trisurf.boundary_loops`.)"""
        from ..ops import trisurf
        return trisurf.boundary_loops(self)

    # -- topology / validity ---------------------------------------------
    def topology_report(self) -> dict[str, Any]:
        """Manifold / connectivity report (see
        :func:`nekmeshpy.model.topology.surface_report`)."""
        from ..model import topology
        return topology.surface_report(self.points, self.tris)

    def is_closed(self) -> bool:
        """``True`` if the surface is a closed, single-component 2-manifold."""
        rep = self.topology_report()
        return bool(rep["closed"] and rep["n_components"] == 1)
