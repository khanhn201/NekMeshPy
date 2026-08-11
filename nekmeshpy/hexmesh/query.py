"""Read-only ``HexMesh`` queries -- the operations that leave the ladder."""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

from .._typing import (
    BoolArray,
    FloatArray,
    IntArray,
    Point,
    PointArray,
    StrArray,
)
from ..core import conform, measure
from ..core.interp import corner_indices
from ..core.quality import QualitySummary
from ..core.tags import _empty_str
from ..core.topology import TopologyReport
from .hexmesh import HexMesh


def _boundary_mask(hexes: IntArray) -> tuple[IntArray, BoolArray]:
    """``(faces, is_boundary)``: every hex quad face ``(6N,4)`` in Nek order,
    element-major (row ``6e+f``), and a mask of those on the domain boundary."""
    HC = np.asarray(hexes, dtype=np.int64).reshape(-1, 8)
    faces: IntArray = HC[:, HexMesh.FACE_POINTS].reshape(-1, 4)
    keys = np.sort(faces, axis=1)
    _, inverse, counts = conform.unique_rows(keys, return_counts=True)
    return faces, counts[inverse] == 1


def boundary_face_ids(mesh: HexMesh) -> BoolArray:
    """``(n_faces,)`` mask of the shared faces carried by exactly one hex.

    The face-id form of :func:`boundary_faces`, and the cheaper one: the faces are
    already deduplicated in ``quads``, so this is a bincount over the incidence rather
    than a hash of every element's corner tuples.

    It is also how a name is checked against the topology it was meant for --
    ``face_tags.ids`` outside this mask are the tagged interior faces
    :func:`tag_report` counts."""
    return np.asarray(
        np.bincount(np.asarray(mesh.hex, dtype=np.int64).ravel(),
                    minlength=mesh.quads.n_quads) == 1, dtype=bool)

def _boundary_points(hexes: IntArray) -> IntArray:
    faces, mask = _boundary_mask(hexes)
    bf = faces[mask]
    return np.unique(bf) if bf.size else np.zeros(0, dtype=np.int64)

def boundary_faces(mesh: HexMesh) -> IntArray:
    """``(K,2)`` of ``[element id, local face (1-6)]`` for every face on the
    topological domain boundary (a quad carried by a single hex). Distinct from
    the tagged ``face_tags``, which may also carry interior planes."""
    _, mask = _boundary_mask(mesh.hexes)
    rows = np.flatnonzero(mask)
    return np.column_stack([rows // 6, rows % 6 + 1]).astype(np.int64)

def boundary_elements(mesh: HexMesh) -> IntArray:
    """Sorted unique element ids with at least one face on the domain boundary."""
    return np.unique(boundary_faces(mesh)[:, 0])

def boundary_points(mesh: HexMesh) -> IntArray:
    """Sorted unique point ids lying on the domain boundary."""
    return _boundary_points(mesh.hexes)

def face_tag_rows(mesh: HexMesh) -> tuple[IntArray, StrArray]:
    """``((K,2) [element, local face 1-6] rows, their tags)`` for every named face,
    lexsorted by ``(element, face)``.

    The inverse of how the tags are stored. A tag names a shared face; a boundary face
    is carried by one hex and yields one row, an **interior** one by two and yields
    two -- which is the honest reading of "this face is named", and what lets an
    exporter give the two sides different codes from the regions on either side.

    Built from the sparse side: the flat ``hex`` face ids are argsorted once and the
    named ids located in them, so nothing the size of ``(E,6)`` in strings is ever
    materialised (at chimera's 438k elements that array alone would be ~170 MB)."""
    named = mesh.quads.element_tags
    if not len(named):
        return np.zeros((0, 2), dtype=np.int64), _empty_str()
    flat: IntArray = np.asarray(mesh.hex, dtype=np.int64).ravel()
    order = np.argsort(flat, kind="stable")
    lo = np.searchsorted(flat[order], named.ids, side="left")
    hi = np.searchsorted(flat[order], named.ids, side="right")
    counts: IntArray = (hi - lo).astype(np.int64)
    bounds: list[tuple[int, int]] = list(
        zip(np.asarray(lo, dtype=np.int64).tolist(),
            np.asarray(hi, dtype=np.int64).tolist()))
    picks: list[IntArray] = [np.arange(a, b, dtype=np.int64) for a, b in bounds]
    slots: IntArray = order[np.concatenate(picks) if picks
                            else np.zeros(0, dtype=np.int64)]
    rows: IntArray = np.column_stack([slots // 6, slots % 6 + 1])
    tags: StrArray = np.repeat(named.tags, counts)
    p = np.lexsort((rows[:, 1], rows[:, 0]))
    return rows[p], tags[p]


def scaled_jacobian(mesh: HexMesh, *, high_order: bool = False) -> FloatArray:
    """Per-hex minimum scaled Jacobian ``(n_hexes,)``."""
    from . import quality
    if high_order:
        return quality.scaled_jacobian_ho(mesh, mesh.order)
    return quality.scaled_jacobian(mesh.points, mesh.hexes)

def quality_summary(mesh: HexMesh, *, high_order: bool = True) -> QualitySummary:
    """Aggregate scaled-Jacobian statistics (see :func:`scaled_jacobian <nekmeshpy.hexmesh.query.scaled_jacobian>` for the
    ``high_order`` flag)."""
    from . import quality
    if high_order:
        return quality.summary_ho(mesh, mesh.order)
    return quality.summary(mesh.points, mesh.hexes)


class WeldResult(NamedTuple):
    """The flat shared-point view of a ``HexMesh`` returned by :func:`weld
    <nekmeshpy.hexmesh.query.weld>`."""

    #: The mesh's **live** ``(P,3)`` coordinate array.  Assigning into it
    #: (``points[:] = X``) repositions the mesh at every rung; rebinding does not.
    points: PointArray
    #: ``(E,8)`` corner connectivity in Nek order, indexing :attr:`points`.
    hexes: IntArray
    #: Number of points, i.e. ``points.shape[0]``.
    n_points: int


def weld(mesh: HexMesh) -> WeldResult:
    """Shared-point view of the mesh (see :class:`WeldResult`); the live positions
    array can be mutated in place to reposition the mesh."""
    return WeldResult(mesh.points, mesh.hexes, mesh.n_points)

def classify_points(mesh: HexMesh, wall: str) -> tuple[BoolArray, BoolArray]:
    """Flag welded points: ``(is_wall, is_fixed)``.  Faces named ``wall`` are
    wall; all other tagged faces are fixed.  A point on both is treated as
    fixed."""
    w = weld(mesh)
    HC, nu = w.hexes, w.n_points
    is_wall: BoolArray = np.zeros(nu, dtype=bool)
    is_fixed: BoolArray = np.zeros(nu, dtype=bool)
    rows, names = face_tag_rows(mesh)
    for (elem, face), tag in zip(rows.tolist(), names.tolist()):
        ids = HC[elem, HexMesh.FACE_POINTS[face - 1, :]]
        if tag == wall:
            is_wall[ids] = True
        else:
            is_fixed[ids] = True
    is_wall[is_fixed] = False
    return is_wall, is_fixed

def topology_report(mesh: HexMesh) -> TopologyReport:
    """Watertightness / connectivity report of the welded mesh."""
    from ..core import topology
    w = weld(mesh)
    return topology.hex_report(w.points, w.hexes)

def is_watertight(mesh: HexMesh) -> bool:
    """``True`` if the mesh boundary is a closed, leak-tight 2-manifold and the
    mesh is a single connected component. Does not imply conformity."""
    rep = topology_report(mesh)
    return rep.watertight and rep.n_components == 1

def is_conforming(mesh: HexMesh) -> bool:
    """``True`` if the mesh has no hanging points (no T-junctions)."""
    return topology_report(mesh).conformal

class TagReport(NamedTuple):
    """How a mesh's ``face_tags`` table lines up with its **topological** boundary.

    The two are independent by design -- ``boundary`` is what connectivity says, a
    side-tag table is a named subset that may also name interior planes -- so they can
    disagree in exactly two ways, and both are worth seeing. An
    :attr:`n_untagged_boundary` above zero means the export has open faces no boundary
    condition covers; an :attr:`n_tagged_interior` above zero means it will write
    boundary conditions onto faces that are not on the boundary at all. Either can be
    deliberate (a flux-measurement plane is a tagged interior face; so is a conjugate
    fluid/solid interface that keeps the fluid's wall condition), so these are counts
    to recognise, not assertions."""

    #: Named faces. One tag per shared face, so this counts faces, not rows -- the
    #: two could differ only while a tag was addressed by ``(element, side)``.
    n_rows: int
    #: Faces on the topological boundary that carry no name.
    n_untagged_boundary: int
    #: Named faces that are **not** on the topological boundary.
    n_tagged_interior: int


def tag_report(mesh: HexMesh) -> TagReport:
    """Cross-check ``face_tags`` against the topological boundary (see
    :class:`TagReport <nekmeshpy.hexmesh.query.TagReport>`)."""
    on_boundary = boundary_face_ids(mesh)
    ft = mesh.face_tags
    named: BoolArray = np.zeros(on_boundary.size, dtype=bool)
    named[ft.ids] = True
    return TagReport(len(ft),
                     int(np.count_nonzero(on_boundary & ~named)),
                     int(np.count_nonzero(named & ~on_boundary)))

def report(mesh: HexMesh) -> str:
    """Human-readable summary: element/point counts, scaled-Jacobian quality,
    per-name tagged-face counts cross-checked against the boundary
    (:func:`tag_report <nekmeshpy.hexmesh.query.tag_report>`), and the topology report."""
    from ..core import topology
    from . import quality
    lines = ["%d hex elements, %d points" % (mesh.n_hexes, mesh.n_points)]
    lines.append(quality.format_report(quality_summary(mesh)))
    for name in mesh.face_group_tags:
        n = mesh.face_tags.count(name)
        lines.append("  %-14s : %d faces" % (name, n))
    tags = tag_report(mesh)
    lines.append("  %-14s : %d faces" % ("untagged bdry", tags.n_untagged_boundary))
    lines.append("  %-14s : %d rows" % ("interior tags", tags.n_tagged_interior))
    lines.append(topology.format_report(topology.hex_report(mesh.points, mesh.hexes)))
    return "\n".join(lines)

def _unique_edges(HC: IntArray, he: IntArray) -> IntArray:
    Ei = HC[:, he[:, 0]].ravel()
    Ej = HC[:, he[:, 1]].ravel()
    return np.unique(np.sort(np.column_stack([Ei, Ej]), axis=1), axis=0)

def element_blocks(mesh: HexMesh) -> PointArray:
    """``(E, (order+1)**3, 3)`` -- each hex's own node block, assembled natively from the
    B-rep: shared corners, then the shared edge-interior nodes in element traversal order,
    then the shared face interiors turned out of each face's canonical frame into this
    hex's, then the private per-hex interiors, each written at its lattice slot.  Nothing
    is resampled or deduplicated.

    The top rung of :func:`linemesh.element_blocks
    <nekmeshpy.linemesh.query.element_blocks>` / :func:`quadmesh.element_blocks
    <nekmeshpy.quadmesh.query.element_blocks>`; this one has the extra face family."""
    order = mesh.order
    row = order + 1
    out: PointArray = np.empty((mesh.n_hexes, row ** 3, 3), dtype=float)
    out[:, corner_indices(order, 3), :] = mesh.points[mesh.hexes]
    out[:, conform._edge_slots(3, order)[:, 1:-1], :] = conform.gather_edge_nodes(
        mesh.edge_nodes, mesh._elem_edges, mesh._edge_flip)
    out[:, conform._face_interior_slots(order), :] = conform.gather_face_nodes(
        mesh.face_nodes, mesh.hex, mesh.face_orient)
    out[:, conform._interior_slots(3, order), :] = mesh.interior
    return out


def _blocks(mesh: HexMesh, high_order: bool) -> PointArray:
    """The node blocks a measure integrates: the curved ones the mesh stores, or the
    straight-sided corner blocks it reduces to."""
    if high_order:
        return element_blocks(mesh)
    return measure.corner_blocks(mesh.points, mesh.hexes, 3)


def bounds(mesh: HexMesh, *, high_order: bool = False) -> measure.Bounds:
    """The axis-aligned bounding box of the block's nodes -- corners only unless
    ``high_order=True`` asks for the stored edge / face / interior nodes too.  See
    :func:`linemesh.bounds <nekmeshpy.linemesh.query.bounds>` for why neither reading
    bounds the polynomial itself."""
    return measure.bounds_of(_blocks(mesh, high_order) if high_order else mesh.points)


def element_volumes(mesh: HexMesh, *, high_order: bool = False) -> FloatArray:
    """``(E,)`` **signed** volume of each hex -- the integral of the isoparametric
    Jacobian, so an inverted hex reads negative, on the same sign convention as
    :func:`scaled_jacobian`.

    Trilinear corners by default -- which is what ``.re2`` exports, since that format is
    linear at any order -- and ``high_order=True`` for the curved element the mesh
    stores.  Both readings are exact: the quadrature is taken high enough to integrate
    the determinant of an order-N map without error."""
    return measure.integrate(_blocks(mesh, high_order), 3)[0]


def volume(mesh: HexMesh, *, high_order: bool = False) -> float:
    """Total volume -- :func:`element_volumes` summed.

    The first number to check after :func:`is_watertight`: a mesh can be watertight,
    conforming and free of inverted elements and still enclose the wrong region because
    a profile landed at the wrong station.  Being signed, it also catches a mesh whose
    elements are wound inside out -- that comes back negative rather than plausible."""
    return float(element_volumes(mesh, high_order=high_order).sum())


def centroid(mesh: HexMesh, *, high_order: bool = False) -> Point:
    """The **volume-weighted** centroid ``integral x dV / integral dV`` -- the mass
    property, not the mean of the points (which would weight a finely meshed corner
    over a coarse bulk)."""
    return measure.centroid_of(_blocks(mesh, high_order), 3, "hexmesh.centroid")


__all__ = [
    "TagReport",
    "face_tag_rows",
    "WeldResult",
    "bounds",
    "boundary_elements",
    "boundary_face_ids",
    "boundary_faces",
    "boundary_points",
    "centroid",
    "classify_points",
    "element_blocks",
    "element_volumes",
    "is_conforming",
    "is_watertight",
    "quality_summary",
    "report",
    "scaled_jacobian",
    "tag_report",
    "topology_report",
    "volume",
    "weld",
]
