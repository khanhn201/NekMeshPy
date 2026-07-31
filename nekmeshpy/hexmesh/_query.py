"""Read-only ``HexMesh`` queries -- the operations that leave the ladder.

They take the mesh (or bare connectivity) and return plain arrays, counts, summary
dicts or formatted text, never another mesh.  This is where the topology / validity
surface lives (``topology_report``, ``is_watertight``, ``is_conforming``, ``report``)
along with the shared-point view the smoothers drive (``weld``, ``classify_points``).

Free functions bound onto :class:`~nekmeshpy.HexMesh` by ``hexmesh/__init__.py``;
internal toolkit code imports them from here directly rather than through the bound
``HexMesh.<name>`` sugar.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .._typing import (
    BoolArray,
    FloatArray,
    IntArray,
    PointArray,
)
from .hexmesh import HexMesh


def _boundary_mask(hexes: IntArray) -> tuple[IntArray, BoolArray]:
    """``(faces, is_boundary)``: every hex quad face ``(6N,4)`` in Nek order,
    element-major (row ``6e+f``), and a mask of those on the domain boundary."""
    HC = np.asarray(hexes, dtype=np.int64).reshape(-1, 8)
    faces: IntArray = HC[:, HexMesh.FACE_POINTS].reshape(-1, 4)
    keys = np.sort(faces, axis=1)
    _, inverse, counts = np.unique(
        keys, axis=0, return_inverse=True, return_counts=True)
    return faces, counts[inverse.ravel()] == 1

def _boundary_points(hexes: IntArray) -> IntArray:
    faces, mask = _boundary_mask(hexes)
    bf = faces[mask]
    return np.unique(bf) if bf.size else np.zeros(0, dtype=np.int64)

def boundary_faces(mesh: HexMesh) -> IntArray:
    """``(K,2)`` of ``[element id, local face (1-6)]`` for every face on the
    topological domain boundary (a quad carried by a single hex). Distinct from
    the tagged ``boundaries``, which may also carry interior planes."""
    _, mask = _boundary_mask(mesh.hexes)
    rows = np.flatnonzero(mask)
    return np.column_stack([rows // 6, rows % 6 + 1]).astype(np.int64)

def boundary_elements(mesh: HexMesh) -> IntArray:
    """Sorted unique element ids with at least one face on the domain boundary."""
    return np.unique(boundary_faces(mesh)[:, 0])

def boundary_points(mesh: HexMesh) -> IntArray:
    """Sorted unique point ids lying on the domain boundary."""
    return _boundary_points(mesh.hexes)

def scaled_jacobian(mesh: HexMesh, *, high_order: bool = False) -> FloatArray:
    """Per-hex minimum scaled Jacobian ``(n_hexes,)``.

    Defaults to the corner metric (the pinned linear numbers).  With
    ``high_order=True`` it is sampled at the ``(order+1)**3`` GLL nodes of the
    curved block (:func:`~nekmeshpy.hexmesh.quality.scaled_jacobian_ho`); at order
    1 the two agree."""
    from . import quality
    if high_order:
        return quality.scaled_jacobian_ho(mesh, mesh.order)
    return quality.scaled_jacobian(mesh.points, mesh.hexes)

def quality_summary(mesh: HexMesh, *, high_order: bool = False) -> dict[str, Any]:
    """Aggregate scaled-Jacobian statistics (see :meth:`scaled_jacobian` for the
    ``high_order`` flag)."""
    from . import quality
    if high_order:
        return quality.summary_ho(mesh, mesh.order)
    return quality.summary(mesh.points, mesh.hexes)

def weld(mesh: HexMesh) -> tuple[PointArray, IntArray, int]:
    """Shared-point view ``(points, hexes, n_points)``; the live positions array
    can be mutated in place to reposition the mesh."""
    return mesh.points, mesh.hexes, mesh.n_points

def classify_points(mesh: HexMesh, wall: str) -> tuple[BoolArray, BoolArray]:
    """Flag welded points: ``(is_wall, is_fixed)``.  Faces named ``wall`` are
    wall; all other tagged faces are fixed.  A point on both is treated as
    fixed."""
    _, HC, nu = weld(mesh)
    is_wall: BoolArray = np.zeros(nu, dtype=bool)
    is_fixed: BoolArray = np.zeros(nu, dtype=bool)
    for b in range(mesh.boundaries.shape[0]):
        elem = int(mesh.boundaries[b, 0])
        face = int(mesh.boundaries[b, 1])
        ids = HC[elem, HexMesh.FACE_POINTS[face - 1, :]]
        if mesh.boundary_tags[b] == wall:
            is_wall[ids] = True
        else:
            is_fixed[ids] = True
    is_wall[is_fixed] = False
    return is_wall, is_fixed

def topology_report(mesh: HexMesh) -> dict[str, Any]:
    """Watertightness / connectivity report of the welded mesh."""
    from ..model import topology
    X, HC, _ = weld(mesh)
    return topology.hex_report(X, HC)

def is_watertight(mesh: HexMesh) -> bool:
    """``True`` if the mesh boundary is a closed, leak-tight 2-manifold and the
    mesh is a single connected component. Does not imply conformity."""
    rep = topology_report(mesh)
    return bool(rep["watertight"] and rep["n_components"] == 1)

def is_conforming(mesh: HexMesh) -> bool:
    """``True`` if the mesh has no hanging points (no T-junctions)."""
    return bool(topology_report(mesh)["conformal"])

def report(mesh: HexMesh) -> str:
    """Human-readable summary: element/point counts, scaled-Jacobian quality,
    per-name tagged-face counts, and the topology report."""
    from ..model import topology
    from . import quality
    lines = ["%d hex elements, %d points" % (mesh.n_hexes, mesh.n_points)]
    lines.append(quality.format_report(quality.summary(mesh.points, mesh.hexes)))
    for name in mesh.boundary_group_tags:
        n = int(np.sum(mesh.boundary_tags == name))
        lines.append("  %-14s : %d faces" % (name, n))
    lines.append(topology.format_report(topology.hex_report(mesh.points, mesh.hexes)))
    return "\n".join(lines)

def _unique_edges(HC: IntArray, he: IntArray) -> IntArray:
    Ei = HC[:, he[:, 0]].ravel()
    Ej = HC[:, he[:, 1]].ravel()
    return np.unique(np.sort(np.column_stack([Ei, Ej]), axis=1), axis=0)


#: Read-only queries bound onto ``HexMesh``.
METHODS: dict[str, Any] = {
    "_boundary_mask": _boundary_mask,
    "_boundary_points": _boundary_points,
    "boundary_faces": boundary_faces,
    "boundary_elements": boundary_elements,
    "boundary_points": boundary_points,
    "scaled_jacobian": scaled_jacobian,
    "quality_summary": quality_summary,
    "weld": weld,
    "classify_points": classify_points,
    "topology_report": topology_report,
    "is_watertight": is_watertight,
    "is_conforming": is_conforming,
    "report": report,
    "_unique_edges": _unique_edges,
}
