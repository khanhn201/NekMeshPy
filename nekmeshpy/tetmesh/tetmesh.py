"""Linear tetrahedral volume-mesh container (``points`` + ``tets``, both 0-based)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .._typing import IntArray, PointArray

if TYPE_CHECKING:
    from ..trimesh import TriMesh


class TetMesh:
    """A linear tetrahedral volume mesh: ``points`` ``(P,3)`` and ``tets`` ``(T,4)``
    connectivity.

    Deliberately linear only -- unlike the point/line/quad/hex ladder, a ``TetMesh``
    carries no order, no interior nodes and no tags.  It exists to solve a field or
    walk a volume (:mod:`tetmesh.ops <nekmeshpy.tetmesh.ops>`), not to be exported: a
    mesher's actual output is the hex mesh built *around* what the solve decided, never
    the tets themselves."""

    # local tet faces (P,Q,R) as femoral_vol's own boundary-face scan enumerated them;
    # winding is not load-bearing here since every consumer sorts before comparing.
    FACE_POINTS = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]], dtype=np.int64)

    def __init__(self, points: PointArray, tets: IntArray) -> None:
        self.points: PointArray = np.asarray(points, dtype=float).reshape(-1, 3)
        self.tets: IntArray = np.asarray(tets, dtype=np.int64).reshape(-1, 4)

    def __repr__(self) -> str:
        """One-line REPL summary: point and tet counts."""
        try:
            return "<TetMesh %d points, %d tets>" % (
                self.points.shape[0], self.tets.shape[0])
        except Exception:                     # a repr must never break a debug session
            return "<TetMesh (unprintable)>"

    # -- sizes -------------------------------------------------------------
    @property
    def n_points(self) -> int:
        """Number of points."""
        return self.points.shape[0]

    @property
    def n_tets(self) -> int:
        """Number of tetrahedra."""
        return self.tets.shape[0]

    # -- boundary ------------------------------------------------------------
    def boundary_faces(self) -> IntArray:
        """``(F,3)`` unique triangle faces borne by exactly one tet -- the open outer
        boundary."""
        faces: IntArray = self.tets[:, self.FACE_POINTS].reshape(-1, 3)
        keys = np.sort(faces, axis=1)
        _, inverse, counts = np.unique(
            keys, axis=0, return_inverse=True, return_counts=True)
        return faces[counts[inverse.ravel()] == 1]

    def boundary_mesh(self) -> TriMesh:
        """The outer boundary as a :class:`TriMesh <nekmeshpy.trimesh.TriMesh>`, over
        this mesh's own points (not renumbered -- a face id here is a face id there)."""
        from ..trimesh import TriMesh
        return TriMesh(self.points, self.boundary_faces())
