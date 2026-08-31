"""Read-only ``HexMesh`` queries -- the operations that leave the ladder."""

from __future__ import annotations

import logging
from collections.abc import Sequence
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
from ..core.quality import OrderScan, QualitySummary
from ..core.tags import _empty_str
from ..core.topology import TopologyReport
from .hexmesh import HexMesh

_log = logging.getLogger(__name__)

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
        np.bincount(np.asarray(mesh.hexes, dtype=np.int64).ravel(),
                    minlength=mesh.quad_mesh.n_quads) == 1, dtype=bool)

def _boundary_points(hexes: IntArray) -> IntArray:
    faces, mask = _boundary_mask(hexes)
    bf = faces[mask]
    return np.unique(bf) if bf.size else np.zeros(0, dtype=np.int64)

def boundary_faces(mesh: HexMesh) -> IntArray:
    """``(K,2)`` of ``[element id, local face (1-6)]`` for every face on the
    topological domain boundary (a quad carried by a single hex). Distinct from
    the tagged ``face_tags``, which may also carry interior planes."""
    _, mask = _boundary_mask(mesh.corners)
    rows = np.flatnonzero(mask)
    return np.column_stack([rows // 6, rows % 6 + 1]).astype(np.int64)

def boundary_elements(mesh: HexMesh) -> IntArray:
    """Sorted unique element ids with at least one face on the domain boundary."""
    return np.unique(boundary_faces(mesh)[:, 0])

def boundary_points(mesh: HexMesh) -> IntArray:
    """Sorted unique point ids lying on the domain boundary."""
    return _boundary_points(mesh.corners)

def face_rows(mesh: HexMesh, faces: IntArray) -> tuple[IntArray, IntArray]:
    """``((K,2) [element, local face 1-6] rows, (len(faces),) occurrence counts)`` --
    the inverse of ``hexes``, for the given shared-face ids.

    A face id is a row of ``quad_mesh``; this says which hexes reference it and as
    which of their six local faces. A **boundary** face is carried by one hex and
    yields one row, an **interior** one by two and yields two, so ``counts`` is how a
    caller expands a per-face value onto the rows it produced. Rows come out grouped by
    input face, ascending within a group; :func:`face_tag_rows` lexsorts them.

    Built from the sparse side: the flat ``hexes`` face ids are argsorted once and the
    wanted ids located in them, so nothing the size of ``(E,6)`` is ever materialised
    per face (at chimera's 438k elements a string array that shape alone is ~170 MB)."""
    ids: IntArray = np.asarray(faces, dtype=np.int64).reshape(-1)
    if not ids.shape[0]:
        return np.zeros((0, 2), dtype=np.int64), np.zeros(0, dtype=np.int64)
    flat: IntArray = np.asarray(mesh.hexes, dtype=np.int64).ravel()
    order = np.argsort(flat, kind="stable")
    lo = np.searchsorted(flat[order], ids, side="left")
    hi = np.searchsorted(flat[order], ids, side="right")
    counts: IntArray = (hi - lo).astype(np.int64)
    bounds: list[tuple[int, int]] = list(
        zip(np.asarray(lo, dtype=np.int64).tolist(),
            np.asarray(hi, dtype=np.int64).tolist()))
    picks: list[IntArray] = [np.arange(a, b, dtype=np.int64) for a, b in bounds]
    slots: IntArray = order[np.concatenate(picks) if picks
                            else np.zeros(0, dtype=np.int64)]
    return np.column_stack([slots // 6, slots % 6 + 1]).astype(np.int64), counts


def face_tag_rows(mesh: HexMesh) -> tuple[IntArray, StrArray]:
    """``((K,2) [element, local face 1-6] rows, their tags)`` for every named face,
    lexsorted by ``(element, face)``.

    The inverse of how the tags are stored -- :func:`face_rows` run over
    ``face_tags.ids``. A tag names a shared face; a boundary face is carried by one hex
    and yields one row, an **interior** one by two and yields two -- which is the honest
    reading of "this face is named", and what lets an exporter give the two sides
    different codes from the regions on either side."""
    named = mesh.quad_mesh.element_tags
    if not len(named):
        return np.zeros((0, 2), dtype=np.int64), _empty_str()
    rows, counts = face_rows(mesh, named.ids)
    tags: StrArray = np.repeat(named.tags, counts)
    p = np.lexsort((rows[:, 1], rows[:, 0]))
    return rows[p], tags[p]


def scaled_jacobian(mesh: HexMesh, *, order: int | None = None) -> FloatArray:
    """Per-hex minimum scaled Jacobian ``(n_hexes,)``, read off the **curved**
    element the mesh actually stores.

    There is deliberately no corner-only reading. A corner scaled Jacobian cannot see
    where the high-order nodes went, so it reports a contented number for a mesh whose
    interior nodes are anywhere at all -- a node moved clean outside the element still
    scores the same. Anything that has to be trusted must read the curved block.

    ``order`` samples that block on a finer GLL lattice than the mesh's own -- what a
    solver running at ``lx1 = order`` does to it. The default reads the mesh's own
    order, where the value is exact at the nodes and silent between them: positive
    there is not a proof the element is not folded."""
    from . import quality
    return quality.scaled_jacobian(mesh, mesh.order if order is None else order)

def corner_scaled_jacobian(mesh: HexMesh) -> FloatArray:
    """Per-hex minimum scaled Jacobian ``(n_hexes,)`` of the **linear** hex through the
    8 corners alone -- the geometry ``.re2`` actually exports, at any stored order.

    Not a substitute for :func:`scaled_jacobian`: a corner reading cannot see where a
    curved mesh's interior nodes went, so it cannot certify a curved element the way
    the curved block can. This answers a narrower, different question -- what happens
    when that curvature is discarded, which ``.re2`` always does -- and an element can
    disagree in *either* direction: curved-clean/corner-inverted (only valid because of
    its own curvature) is the case worth watching for before export."""
    from . import quality
    return quality.corner_scaled_jacobian(mesh.points, mesh.corners)


def corner_summary(mesh: HexMesh) -> QualitySummary:
    """Aggregate statistics over :func:`corner_scaled_jacobian` -- see there."""
    from . import quality
    return quality.corner_summary(mesh.points, mesh.corners)


def linear_scaled_jacobian(mesh: HexMesh, *, order: int | None = None) -> FloatArray:
    """Per-hex minimum scaled Jacobian of the **trilinear** hex through the 8 corners
    alone -- ``.re2``'s own geometry -- resampled at ``order``.

    Unlike :func:`corner_scaled_jacobian` (the special case ``order=1``: the 8
    vertices only), this catches a fold that sits *between* the corners, which a
    real solver's own geometry generation would find building its working nodes
    from the same corners at its own polynomial order -- ``order`` should be that
    order (:data:`SCAN_ORDER <nekmeshpy.core.quality.SCAN_ORDER>` by default, the
    same solver-order default :func:`order_scan` uses)."""
    from ..core.quality import SCAN_ORDER
    from . import quality
    return quality.linear_scaled_jacobian(mesh, SCAN_ORDER if order is None else order)


def linear_order_scan(mesh: HexMesh, orders: Sequence[int] | None = None, *,
                      budget: int | None = None) -> OrderScan:
    """:func:`order_scan`'s report shape, for the **trilinear** (``.re2``) map
    instead of the curved one -- see :func:`linear_scaled_jacobian`."""
    from . import quality
    return quality.linear_order_scan(mesh, orders, budget=budget)


def quality_summary(mesh: HexMesh, *, order: int | None = None) -> QualitySummary:
    """Aggregate scaled-Jacobian statistics over the **curved** elements -- see
    :func:`scaled_jacobian <nekmeshpy.hexmesh.query.scaled_jacobian>`.

    Above order 1, this also checks the **linear** reading -- :func:`corner_summary
    <nekmeshpy.hexmesh.query.corner_summary>`, the straight-sided hex through the 8
    corners alone -- because that, not the curved geometry just summarised, is what
    ``.re2`` actually exports: it has no curved format at any stored order. An element
    can be clean here and still invert once flattened for export, if it depends on its
    own curvature to stay valid; that is logged as a warning rather than folded into
    the returned value, the same way :func:`report <nekmeshpy.hexmesh.query.report>`
    warns on an ``order_scan`` disagreement without changing what ``quality_summary``
    itself returns."""
    from . import quality
    stats = quality.summary(mesh, mesh.order if order is None else order)
    if mesh.order > 1:
        linear = corner_summary(mesh)
        if linear.n_inverted and not stats.n_inverted:
            _log.warning(
                "mesh is clean at order %d but has %d element(s) inverted once "
                "flattened to .re2's linear corners (min corner scaled Jacobian "
                "%.4f) -- .re2 has no curved format, so this is the geometry the "
                "solver actually reads", mesh.order, linear.n_inverted, linear.min)
    return stats


def classify_points(mesh: HexMesh, wall: str) -> tuple[BoolArray, BoolArray]:
    """Flag welded points: ``(is_wall, is_fixed)``.  Faces named ``wall`` are
    wall; all other tagged faces are fixed.  A point on both is treated as
    fixed."""
    HC, nu = mesh.corners, mesh.n_points
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
    """Watertightness / connectivity report of the mesh."""
    from ..core import topology
    return topology.hex_report(mesh.points, mesh.corners)

def is_watertight(mesh: HexMesh) -> bool:
    """``True`` if the mesh boundary is a closed, leak-tight 2-manifold and the
    mesh is a single connected component. Does not imply conformity."""
    rep = topology_report(mesh)
    return rep.watertight and rep.n_components == 1

def is_conforming(mesh: HexMesh) -> bool:
    """``True`` if the mesh has no hanging points (no T-junctions)."""
    return topology_report(mesh).conformal

def is_overlap_free(mesh: HexMesh) -> bool:
    """``True`` if no two elements geometrically overlap. Independent of
    :func:`is_watertight <nekmeshpy.hexmesh.query.is_watertight>` and :func:`is_conforming
    <nekmeshpy.hexmesh.query.is_conforming>`, which are purely topological (facet-sharing)
    checks -- a duplicated or mis-placed piece that never shares a face with the rest of
    the mesh would still read watertight and conformal. Not part of :func:`topology_report
    <nekmeshpy.hexmesh.query.topology_report>`: this is a geometric search whose cost a
    plain watertight/conforming check should not have to pay, so it is computed only
    when asked for."""
    from ..core import topology
    return topology.is_overlap_free(mesh.points, mesh.corners)

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
    # ``.re2`` has no curved format at any order, so the curved summary above is not
    # what the solver actually reads -- a different map, not a coarser sampling of the
    # same one, and an element can be clean in it only because of curvature that never
    # reaches the file.
    if mesh.order > 1:
        lines.append(quality.format_linear(corner_summary(mesh), mesh.order))
    # The mesh's own order is exact at its nodes and silent between them, so the
    # summary above cannot certify the element -- read it on finer lattices too, and
    # say so loudly when one of them disagrees.
    scan = quality.order_scan(mesh)
    lines.append(quality.format_scan(scan, mesh.order))
    if scan.orders and not scan.clean:
        n, m = scan.worst
        _log.warning("mesh is clean at order %d but has %d inverted element(s) at "
                     "sampling order %d (min scaled Jacobian %.4f)",
                     mesh.order, scan.n_inverted[scan.orders.index(n)], n, m)
    # And that "curved, sampled finer" scan is still not what .re2 exports: at
    # order > 1, run the same sweep on the trilinear (corners-only) map, which can
    # fold *between* the corners even when every corner and the curved map both
    # read clean -- confirmed against a real Nek5000 run's own geometry-generation
    # check, which is exactly this computation.
    if mesh.order > 1:
        lscan = quality.linear_order_scan(mesh)
        lines.append(quality.format_linear_scan(lscan))
        if lscan.orders and not lscan.clean:
            n, m = lscan.worst
            _log.warning(
                "mesh is clean at order %d (curved) and at the corners alone, but "
                "%d element(s) fold once the .re2 trilinear geometry is resampled "
                "at order %d (min scaled Jacobian %.4f) -- this is what a real "
                "solver's own geometry generation builds and checks at its "
                "working order", mesh.order,
                lscan.n_inverted[lscan.orders.index(n)], n, m)
    for name in mesh.face_group_tags:
        n = mesh.face_tags.count(name)
        lines.append("  %-14s : %d faces" % (name, n))
    tags = tag_report(mesh)
    lines.append("  %-14s : %d faces" % ("untagged bdry", tags.n_untagged_boundary))
    lines.append("  %-14s : %d rows" % ("interior tags", tags.n_tagged_interior))
    lines.append(topology.format_report(topology.hex_report(mesh.points, mesh.corners)))
    # topology.count_overlapping_pairs is a geometric broad-then-narrow-phase search,
    # not a fixed-size facet-incidence scan like the rest of this summary -- its
    # candidate count can run into the hundreds of thousands on a large mesh, and it
    # was slow enough there to be worth pulling back out of the default summary.
    # Call it explicitly (or hexmesh.is_overlap_free) where the cost is worth paying.
    # n_overlap = topology.count_overlapping_pairs(mesh.points, mesh.corners)
    # lines.append("overlap-free   : %s (%d overlapping pair%s)"
    #              % (n_overlap == 0, n_overlap, "" if n_overlap == 1 else "s"))
    return "\n".join(lines)

def _unique_edges(HC: IntArray, he: IntArray) -> IntArray:
    Ei = HC[:, he[:, 0]].ravel()
    Ej = HC[:, he[:, 1]].ravel()
    return conform.unique_rows(
        np.sort(np.column_stack([Ei, Ej]), axis=1))[0]

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
    out[:, corner_indices(order, 3), :] = mesh.points[mesh.corners]
    out[:, conform._edge_slots(3, order)[:, 1:-1], :] = conform.gather_edge_nodes(
        mesh.edge_nodes, mesh._elem_edges, mesh._edge_flip)
    out[:, conform._face_interior_slots(order), :] = conform.gather_face_nodes(
        mesh.face_nodes, mesh.hexes, mesh.orient)
    out[:, conform._interior_slots(3, order), :] = mesh.interior
    return out


def _blocks(mesh: HexMesh, high_order: bool) -> PointArray:
    """The node blocks a measure integrates: the curved ones the mesh stores, or the
    straight-sided corner blocks it reduces to."""
    if high_order:
        return element_blocks(mesh)
    return measure.corner_blocks(mesh.points, mesh.corners, 3)


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


def tagged_faces(mesh: HexMesh, tag: str) -> IntArray:
    """The **shared-face ids** carrying ``tag``, ascending.

    The handle every face group is addressed by at this rung -- ``face_tags`` is stored
    sparse and sorted, so this is its ``ids`` filtered by name.  It is public because
    :func:`hexmesh.attach <nekmeshpy.hexmesh.assemble.attach>` works from exactly this
    list, and a caller who wants to see, reorder or subset the group it will pair needs
    to be able to get at it.

    A tag that names nothing raises rather than returning an empty group: a mis-spelled
    interface name is otherwise invisible until the solver reads the mesh."""
    t = mesh.face_tags
    hit: IntArray = np.asarray(t.ids[t.mask_for(tag)], dtype=np.int64)
    if hit.size == 0:
        raise ValueError(
            "tagged_faces: no face carries the tag %r; this mesh has %s"
            % (tag, sorted(t.group_tags) or "no tagged faces"))
    return hit


__all__ = [
    "TagReport",
    "face_rows",
    "face_tag_rows",
    "bounds",
    "boundary_elements",
    "boundary_face_ids",
    "boundary_faces",
    "boundary_points",
    "centroid",
    "classify_points",
    "corner_scaled_jacobian",
    "corner_summary",
    "element_blocks",
    "element_volumes",
    "is_conforming",
    "is_overlap_free",
    "is_watertight",
    "linear_order_scan",
    "linear_scaled_jacobian",
    "quality_summary",
    "report",
    "scaled_jacobian",
    "tag_report",
    "tagged_faces",
    "topology_report",
    "volume",
]
