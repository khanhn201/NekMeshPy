"""High order and the hex smoother.

The native stages (``laplace``, ``optimize``) are **linear**: they move the corner vertices of an order-1 mesh.  On a
curved mesh a schedule runs them on its straight-sided shadow (:func:`linearize`: same corners, connectivity,
element tags and face tags, no edge, face or interior nodes) and the curved nodes are **carried** by the vertex
displacement (:func:`carry`): every node of an element moves by the trilinear interpolation of its eight corners'
displacements, so a shared edge or face -- which sees only its own corners -- stays conformal, an element whose
corners did not move is untouched bit for bit, and nothing is re-projected onto a surface (a curved boundary
should be frozen, or smoothed by a stage that follows it).  The ``tmop`` stage instead runs **on the curved
mesh** and returns the same order (:func:`with_blocks` writes its node blocks back).  ``.re2`` exports only the
corners in either case.
"""

from __future__ import annotations

import numpy as np

from ..._typing import FloatArray, IntArray, PointArray
from ...core.smoother.blocks import carry_blocks, scatter_blocks
from ...linemesh import LineMesh
from ...pointmesh import PointMesh
from ...quadmesh import QuadMesh
from ...quadmesh.smoother.highorder import linearize as _linearize_quad
from ..hexmesh import HexMesh
from ..query import element_blocks

__all__ = ["linearize", "node_ids", "with_blocks", "carry"]


def linearize(mesh: HexMesh) -> HexMesh:
    """The straight-sided order-1 mesh with the same corners, connectivity and tags -- what the
    native smoother works on.  A new mesh; ``mesh`` is untouched (a linear mesh comes back as a
    copy)."""
    return HexMesh(_linearize_quad(mesh.quad_mesh), mesh.hexes, mesh.orient, None, mesh.element_tags)


def _flat(mesh: HexMesh) -> FloatArray:
    qm, lm = mesh.quad_mesh, mesh.quad_mesh.line_mesh
    return np.concatenate([lm.points, lm.interior.reshape(-1, 3), qm.interior.reshape(-1, 3),
                           mesh.interior.reshape(-1, 3)])


def _shadow(mesh: HexMesh, flat: FloatArray) -> HexMesh:
    """``mesh`` with its stored nodes -- points, edge interiors, face interiors, hex interiors, in that order --
    replaced by the rows of ``flat``; incidence, orientation and tags are the mesh's own."""
    qm, lm = mesh.quad_mesh, mesh.quad_mesh.line_mesh
    nP, nE, nF = lm.points.shape[0], lm.interior.shape[0] * lm.interior.shape[1], qm.interior.shape[0] * qm.interior.shape[1]
    ho = mesh.order > 1
    edge = np.array(flat[nP:nP + nE]).reshape(lm.interior.shape)
    face = np.array(flat[nP + nE:nP + nE + nF]).reshape(qm.interior.shape)
    inner = np.array(flat[nP + nE + nF:]).reshape(mesh.interior.shape)
    new_lm = LineMesh(PointMesh(np.array(flat[:nP]), lm.point_mesh.element_tags), lm.lines, edge if ho else None,
                      lm.element_tags)
    new_qm = QuadMesh(new_lm, qm.quads, qm.orient, face if ho else None, qm.element_tags)
    return HexMesh(new_qm, mesh.hexes, mesh.orient, inner if ho else None, mesh.element_tags)


def node_ids(mesh: HexMesh) -> IntArray:
    """``(E, (order+1)**3)`` the index of every block node in the flat table ``points, edge interiors, face
    interiors, hex interiors`` -- read by running :func:`~nekmeshpy.hexmesh.query.element_blocks` on the ids
    themselves, so the storage's own edge flips and face orientations are used and none re-derived."""
    flat = _flat(mesh)
    ids = np.zeros_like(flat)
    ids[:, 0] = np.arange(flat.shape[0])
    return np.rint(element_blocks(_shadow(mesh, ids))[:, :, 0]).astype(np.int64)


def with_blocks(mesh: HexMesh, blocks: FloatArray, points: PointArray | None = None) -> HexMesh:
    """A **new** mesh of the same order, connectivity and tags whose node blocks are ``blocks``
    ``(E, (order+1)**3, 3)``.  Each shared node is stored once -- an element's own value, the blocks of a
    conformal mesh agreeing -- so conformality stays structural; ``points`` (when given) are the vertices exactly."""
    flat = scatter_blocks(node_ids(mesh), np.asarray(blocks, float), _flat(mesh))
    if points is not None:
        flat[:mesh.n_points] = points
    return _shadow(mesh, flat)


def carry(mesh: HexMesh, moved: HexMesh) -> HexMesh:
    """``mesh`` (curved) with its nodes carried to the vertices of ``moved`` (a mesh of the same connectivity, e.g.
    the linear shadow a native stage smoothed): each node follows the displacement of its element's corners,
    interpolated trilinearly (:func:`~nekmeshpy.core.smoother.blocks.carry_blocks`) -- no projection onto a target."""
    d = np.asarray(moved.points, float) - np.asarray(mesh.points, float)
    new = carry_blocks(element_blocks(mesh), d[mesh.corners], mesh.order, 3)
    return with_blocks(mesh, new, np.asarray(moved.points, float))
