"""Triangulated surface mesh container.

``TriMesh`` is a pure container of a surface triangulation: vertex coordinates
``xyz`` (nv,3) and triangle connectivity ``tri`` (nt,3), both 0-based (input
``.tri`` files are 1-based and converted on load).  It also holds a cache slot
``_L`` for the memoized cotangent Laplacian.

The surface *algorithms* -- cotangent Laplace operators, Dirichlet solves,
boundary-loop extraction, marching-triangle isocontours, and closest-point
projection -- live in :mod:`nekmeshpy.ops.trisurf` as free functions taking the
surface as their first argument.
"""

import numpy as np


class TriMesh:
    def __init__(self, xyz, tri):
        self.xyz = np.asarray(xyz, dtype=float)
        self.tri = np.asarray(tri, dtype=np.int64)
        self._L = None                              # cached cotangent Laplacian

    # -- construction ----------------------------------------------------
    @classmethod
    def from_files(cls, vtx_file, tri_file):
        """Load a surface triangulation (node-list + 1-based triangle-index
        files); triangle indices are converted to 0-based."""
        xyz = np.loadtxt(vtx_file, dtype=float)
        tri = np.loadtxt(tri_file, dtype=float).astype(np.int64)
        if xyz.ndim == 1:
            xyz = xyz.reshape(1, -1)
        if tri.ndim == 1:
            tri = tri.reshape(1, -1)
        return cls(xyz, tri - 1)

    @classmethod
    def from_faces(cls, V, faces):
        """Build a sub-mesh from vertex set ``V`` restricted to ``faces`` (a
        triangle list indexing into ``V``), compacting to the used vertices.
        Returns ``(TriMesh, vids)`` where ``vids`` maps sub-index -> V-index."""
        faces = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
        vids = np.unique(faces.ravel())
        remap = np.zeros(np.asarray(V).shape[0], dtype=np.int64)
        remap[vids] = np.arange(vids.size)
        return cls(np.asarray(V, dtype=float)[vids, :], remap[faces]), vids

    # -- sizes -----------------------------------------------------------
    @property
    def n_vertices(self):
        return self.xyz.shape[0]

    @property
    def n_triangles(self):
        return self.tri.shape[0]
