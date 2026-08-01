"""Triangulated surface mesh container (``points`` + ``tris``, both 0-based)."""

from __future__ import annotations

from typing import Any

import numpy as np

from .._typing import BoolArray, IntArray, PointArray


class TriMesh:
    """A triangular surface mesh: ``points`` ``(P,3)`` and ``tris`` ``(T,3)`` connectivity."""

    def __init__(self, points: PointArray, tris: IntArray) -> None:
        self.points = np.asarray(points, dtype=float).reshape(-1, 3)
        self.tris = np.asarray(tris, dtype=np.int64)

    # -- construction ----------------------------------------------------
    @classmethod
    def from_files(cls, vtx_file: str, tri_file: str) -> TriMesh:
        """Load a surface triangulation from point-list and 1-based triangle-index files."""
        points = np.loadtxt(vtx_file, dtype=float)
        tris = np.loadtxt(tri_file, dtype=float).astype(np.int64)
        if points.ndim == 1:
            points = points.reshape(1, -1)
        if tris.ndim == 1:
            tris = tris.reshape(1, -1)
        return cls(points, tris - 1)

    @classmethod
    def from_faces(cls, V: PointArray, faces: IntArray) -> tuple[TriMesh, IntArray]:
        """Build a sub-mesh from ``V`` restricted to ``faces``; returns ``(TriMesh, vids)``
        where ``vids`` maps sub-index -> V-index."""
        faces = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
        vids = np.unique(faces.ravel())
        remap = np.zeros(np.asarray(V).shape[0], dtype=np.int64)
        remap[vids] = np.arange(vids.size)
        return cls(np.asarray(V, dtype=float)[vids, :], remap[faces]), vids

    # local triangle edges; row e is edge e+1
    EDGE_POINTS = np.array([[0, 1], [1, 2], [2, 0]], dtype=np.int64)

    # -- sizes -----------------------------------------------------------
    def __repr__(self) -> str:
        """One-line REPL summary: element and point counts.  The same shape as the
        ladder containers' reprs (:meth:`LineMesh.__repr__
        <nekmeshpy.linemesh.LineMesh.__repr__>`) minus the fields a triangulated
        surface does not have -- a ``TriMesh`` is a linear scanned/imported surface
        carrying no polynomial order and no element or boundary tags, so those fields
        are omitted rather than rendered empty and misleading.

        Reads two stored array shapes; never raises, because a repr that throws on a
        half-loaded surface makes debugging strictly worse."""
        try:
            return "<TriMesh %d points, %d tris>" % (
                self.points.shape[0], self.tris.shape[0])
        except Exception:                     # a repr must never break a debug session
            return "<TriMesh (unprintable)>"

    @property
    def n_points(self) -> int:
        """Number of points."""
        return self.points.shape[0]

    @property
    def n_tris(self) -> int:
        """Number of triangles."""
        return self.tris.shape[0]

    # -- boundary queries (open surface edges) --------------------------
    @staticmethod
    def _boundary_mask(tris: IntArray) -> tuple[IntArray, BoolArray]:
        """``(edges, is_boundary)``: every triangle edge ``(3M,2)`` and a mask of
        those borne by a single triangle (the open boundary)."""
        T = np.asarray(tris, dtype=np.int64).reshape(-1, 3)
        edges: IntArray = T[:, TriMesh.EDGE_POINTS].reshape(-1, 2)
        keys = np.sort(edges, axis=1)
        _, inverse, counts = np.unique(
            keys, axis=0, return_inverse=True, return_counts=True)
        return edges, counts[inverse.ravel()] == 1

    def boundary_edges(self) -> IntArray:
        """``(K,2)`` array of ``[triangle id, local edge (1-3)]`` for every open-boundary edge."""
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
        """The open boundary grouped into loops -- one vertex-id array per opening."""
        from . import ops
        return ops.boundary_loops(self)

    # -- topology / validity ---------------------------------------------
    def topology_report(self) -> dict[str, Any]:
        """Manifold / connectivity report."""
        from ..model import topology
        return topology.surface_report(self.points, self.tris)

    def is_closed(self) -> bool:
        """``True`` if the surface is a closed, single-component 2-manifold."""
        rep = self.topology_report()
        return bool(rep["closed"] and rep["n_components"] == 1)
