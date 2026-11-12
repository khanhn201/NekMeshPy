"""Read-only ``QuadMesh`` queries -- the operations that leave the ladder."""

from __future__ import annotations

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
    Vec3,
)
from ..core import conform, frames, measure
from ..core.interp import corner_indices
from ..core.quality import QualitySummary
from ..core.topology import TagReport, TopologyReport
from .quadmesh import QuadMesh


def _boundary_mask(quads: IntArray) -> tuple[IntArray, BoolArray]:
    """``(edges, is_boundary)``: every quad edge ``(4M,2)`` element-major
    (row ``4q+e``), and a mask of those borne by a single quad."""
    Q = np.asarray(quads, dtype=np.int64).reshape(-1, 4)
    edges: IntArray = Q[:, QuadMesh.EDGE_POINTS].reshape(-1, 2)
    keys = np.sort(edges, axis=1)
    _, inverse, counts = conform.unique_rows(keys, return_counts=True)
    return edges, counts[inverse] == 1

def boundary_edges(mesh: QuadMesh) -> IntArray:
    """``(K,2)`` array of ``[quad id, local edge (1-4)]`` for every edge on
    the section boundary (borne by a single quad)."""
    _, mask = _boundary_mask(mesh.corners)
    rows = np.flatnonzero(mask)
    return np.column_stack([rows // 4, rows % 4 + 1]).astype(np.int64)

def boundary_elements(mesh: QuadMesh) -> IntArray:
    """Sorted unique quad ids with at least one edge on the section boundary."""
    return np.unique(boundary_edges(mesh)[:, 0])

def boundary_points(mesh: QuadMesh) -> IntArray:
    """Sorted unique point ids lying on the section boundary."""
    edges, mask = _boundary_mask(mesh.corners)
    be = edges[mask]
    return np.unique(be) if be.size else np.zeros(0, dtype=np.int64)

def scaled_jacobian(mesh: QuadMesh, *, order: int | None = None) -> FloatArray:
    """Per-quad minimum scaled Jacobian ``(n_quads,)``, read off the **curved**
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

def quality_summary(mesh: QuadMesh, *, order: int | None = None) -> QualitySummary:
    """Aggregate scaled-Jacobian statistics over the **curved** elements -- see
    :func:`scaled_jacobian <nekmeshpy.quadmesh.query.scaled_jacobian>`."""
    from . import quality
    return quality.summary(mesh, mesh.order if order is None else order)

def plane_normal(mesh: QuadMesh, *,
                 hint: Vec3 | Sequence[float] | None = None,
                 check: bool = True) -> Vec3:
    """The unit normal of the plane a **planar** section lies in: the smallest right
    singular vector of its centred points, i.e. the exact least-squares plane."""
    normal = None
    if not check:
        P: PointArray = np.asarray(mesh.points, dtype=float)
        c = P.mean(axis=0)
        normal = np.linalg.svd(P - c)[2][2]
    R, _ = frames.plane_frame(mesh.points, normal=normal, hint=hint)
    w: Vec3 = R[:, 2]
    return w


def element_blocks(mesh: QuadMesh) -> PointArray:
    """``(Q, (order+1)**2, 3)`` -- each quad's own node block, assembled natively from the
    B-rep: shared corners, then the shared edge-interior nodes in element traversal order,
    then the private per-quad interiors, each written at its lattice slot.  Nothing is
    resampled or deduplicated.

    The quad rung of :func:`linemesh.element_blocks
    <nekmeshpy.linemesh.query.element_blocks>`, one dimension up."""
    order = mesh.order
    row = order + 1
    out: PointArray = np.empty((mesh.n_quads, row * row, 3), dtype=float)
    out[:, corner_indices(order, 2), :] = mesh.points[mesh.corners]
    out[:, conform._edge_slots(2, order)[:, 1:-1], :] = conform.gather_edge_nodes(
        mesh.line_mesh.interior, mesh.quads, mesh.orient)
    out[:, conform._interior_slots(2, order), :] = mesh.interior
    return out


def _blocks(mesh: QuadMesh, high_order: bool) -> PointArray:
    """The node blocks a measure integrates: the curved ones the mesh stores, or the
    straight-sided corner blocks it reduces to."""
    if high_order:
        return element_blocks(mesh)
    return measure.corner_blocks(mesh.points, mesh.corners, 2)


def bounds(mesh: QuadMesh, *, high_order: bool = False) -> measure.Bounds:
    """The axis-aligned bounding box of the section's nodes -- corners only unless
    ``high_order=True`` asks for the stored interior nodes too.  See
    :func:`linemesh.bounds <nekmeshpy.linemesh.query.bounds>` for why neither reading
    bounds the polynomial itself."""
    return measure.bounds_of(_blocks(mesh, high_order) if high_order else mesh.points)


def element_areas(mesh: QuadMesh, *, high_order: bool = False) -> FloatArray:
    """``(Q,)`` surface area of each quad, integrated over the element as it sits in
    3-D (so a warped quad is measured as the ruled surface it is, not as a projection).

    Bilinear corners by default; ``high_order=True`` integrates the curved element the
    mesh stores.  A curved area is not a polynomial -- the integrand carries a square
    root -- so that reading is a quadrature approximation converging with the order,
    where :func:`hexmesh.element_volumes
    <nekmeshpy.hexmesh.query.element_volumes>` is exact."""
    return measure.integrate(_blocks(mesh, high_order), 2)[0]


def area(mesh: QuadMesh, *, high_order: bool = False) -> float:
    """Total surface area -- :func:`element_areas` summed.

    The wetted area of a block's wall is this over the wall's own mesh:
    ``quadmesh.area(hexmesh.boundary_mesh(block, "wall"))``."""
    return float(element_areas(mesh, high_order=high_order).sum())


def centroid(mesh: QuadMesh, *, high_order: bool = False) -> Point:
    """The **area-weighted** centroid ``integral x dA / integral dA`` -- the mass
    property, not the mean of the points."""
    return measure.centroid_of(_blocks(mesh, high_order), 2)


def tagged_edges(mesh: QuadMesh, tag: str) -> IntArray:
    """The **shared-edge ids** carrying ``tag``, ascending -- the quad rung's
    counterpart of :func:`hexmesh.tagged_faces
    <nekmeshpy.hexmesh.query.tagged_faces>`, and what
    :func:`quadmesh.attach <nekmeshpy.quadmesh.assemble.attach>` pairs.

    A tag that names nothing raises rather than returning an empty group."""
    t = mesh.edge_tags
    return t.flatnonzero(tag, strict=True)


def boundary_edge_ids(mesh: QuadMesh) -> BoolArray:
    """``(n_edges,)`` mask of the shared edges carried by exactly one quad.

    The edge-id form of :func:`boundary_edges`, and the cheaper one: the edges are
    already deduplicated in ``line_mesh``, so this is a bincount over the incidence.
    It is how a name is checked against the topology it was meant for --
    ``edge_tags.ids`` outside this mask are the tagged interior edges
    :func:`tag_report` counts."""
    return np.asarray(
        np.bincount(np.asarray(mesh.quads, dtype=np.int64).ravel(),
                    minlength=mesh.line_mesh.n_lines) == 1, dtype=bool)


def edge_rows(mesh: QuadMesh, edges: IntArray) -> tuple[IntArray, IntArray]:
    """``((K,2) [quad, local edge 1-4] rows, (len(edges),) occurrence counts)`` -- the
    inverse of ``quads``, for the given shared-edge ids; the quad rung's
    :func:`hexmesh.face_rows <nekmeshpy.hexmesh.query.face_rows>`.

    A **boundary** edge is carried by one quad and yields one row, an **interior** one
    by two and yields two, so ``counts`` is how a caller expands a per-edge value onto
    the rows it produced."""
    ids: IntArray = np.asarray(edges, dtype=np.int64).reshape(-1)
    if not ids.shape[0]:
        return np.zeros((0, 2), dtype=np.int64), np.zeros(0, dtype=np.int64)
    flat: IntArray = np.asarray(mesh.quads, dtype=np.int64).ravel()
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
    return np.column_stack([slots // 4, slots % 4 + 1]).astype(np.int64), counts


def edge_tag_rows(mesh: QuadMesh) -> tuple[IntArray, StrArray]:
    """``((K,2) [quad, local edge 1-4] rows, their tags)`` for every named edge,
    lexsorted by ``(quad, edge)`` -- :func:`edge_rows` run over ``edge_tags.ids``; the
    quad rung's :func:`hexmesh.face_tag_rows <nekmeshpy.hexmesh.query.face_tag_rows>`."""
    named = mesh.line_mesh.element_tags
    if not len(named):
        return np.zeros((0, 2), dtype=np.int64), named.tags
    rows, counts = edge_rows(mesh, named.ids)
    tags: StrArray = np.repeat(named.tags, counts)
    p = np.lexsort((rows[:, 1], rows[:, 0]))
    return rows[p], tags[p]


def topology_report(mesh: QuadMesh) -> TopologyReport:
    """Watertightness / connectivity report of the mesh."""
    from ..core import topology
    return topology.quad_report(mesh.points, mesh.corners)


def is_watertight(mesh: QuadMesh) -> bool:
    """``True`` if the mesh boundary is a closed curve and the mesh is a single
    connected component. Does not imply conformity."""
    rep = topology_report(mesh)
    return rep.watertight and rep.n_components == 1


def is_conforming(mesh: QuadMesh) -> bool:
    """``True`` if the mesh has no hanging points (no T-junctions)."""
    return topology_report(mesh).conformal


def is_overlap_free(mesh: QuadMesh) -> bool:
    """``True`` if no two quads geometrically overlap. Independent of
    :func:`is_watertight` and :func:`is_conforming`, which are purely topological
    (facet-sharing) checks, and computed only when asked for, as at the hex rung."""
    from ..core import topology
    return topology.is_overlap_free_quads(mesh.points, mesh.corners)


def overlapping_quads(mesh: QuadMesh) -> IntArray:
    """Ids of the quads that overlap another quad -- the *which* behind :func:`is_overlap_free`.

    A branched star (one quad covering its neighbours) can have every signed Jacobian positive, so the
    inverted-quad count cannot see it; this does.  Each pair is tested in the first quad's own mean plane,
    so quads on a curved surface are read locally.  Cost: a KD-tree over centroids, then a half-plane test of
    9 sample points per candidate pair."""
    from ..core import topology
    return topology.overlapping_quads(mesh.points, mesh.corners)


def tag_report(mesh: QuadMesh) -> TagReport:
    """Cross-check ``edge_tags`` against the topological boundary (see
    :class:`TagReport <nekmeshpy.core.topology.TagReport>`)."""
    on_boundary = boundary_edge_ids(mesh)
    et = mesh.edge_tags
    named = et.is_tagged()
    return TagReport(len(et),
                     int(np.count_nonzero(on_boundary & ~named)),
                     int(np.count_nonzero(named & ~on_boundary)))


def report(mesh: QuadMesh) -> str:
    """Human-readable summary: element/point counts, scaled-Jacobian quality (curved,
    and on the solver-order lattice), per-name tagged-edge counts cross-checked against
    the boundary (:func:`tag_report`), and the topology report -- the quad rung's
    :func:`hexmesh.report <nekmeshpy.hexmesh.query.report>`.  There is no corner-only
    reading and no linear one: ``.re2`` is hex-only, so nothing here is exported flat."""
    import logging

    from ..core import topology
    from . import quality
    lines = ["%d quad elements, %d points" % (mesh.n_quads, mesh.n_points)]
    lines.append(quality.format_report(quality_summary(mesh)))
    scan = quality.order_scan(mesh)
    lines.append(quality.format_scan(scan, mesh.order))
    if scan.orders and not scan.clean:
        n, m = scan.worst
        logging.getLogger(__name__).warning(
            "mesh is clean at order %d but has %d inverted element(s) at sampling "
            "order %d (min scaled Jacobian %.4f)",
            mesh.order, scan.n_inverted[scan.orders.index(n)], n, m)
    names, counts = mesh.edge_tags.unique(return_counts=True)
    for name, n in zip(names, counts.tolist()):
        lines.append("  %-14s : %d edges" % (name, n))
    tags = tag_report(mesh)
    lines.append("  %-14s : %d edges" % ("untagged bdry", tags.n_untagged_boundary))
    lines.append("  %-14s : %d rows" % ("interior tags", tags.n_tagged_interior))
    lines.append(topology.format_report(topology.quad_report(mesh.points, mesh.corners),
                                        "quad"))
    return "\n".join(lines)


class ConnectivityReport(NamedTuple):
    """The preflight gap check of one mesh (read-only; nothing is welded or moved).

    A *gap* is a group of stored points that gencon's relative-tolerance rule would fuse
    while the mesh keeps them apart -- an open seam or a duplicated vertex; ``gap_points``
    are their ids and ``n_gap_clusters`` the number of groups.  A group only counts when its
    members are directly within ``connectivity_tol`` of the local length of each other and no
    two of them are corners of one quad.  Otherwise (gencon's rule also chains, so on a tangled
    mesh vertices that merely sit close are linked, or a vertex lies on top of its quad's
    other corner) it is counted in ``n_chained_clusters``: a possible tangle, not a seam, whose
    fix is untangling and not welding.
    """

    n_points: int
    n_quads: int
    n_components: int
    connectivity_tol: float
    #: Number of gap clusters (see the class docstring for what counts as a gap).
    n_gap_clusters: int
    gap_points: IntArray
    #: Of those, how many sit on the boundary (an open seam) vs inside (a duplicated vertex).
    n_gap_on_boundary: int
    #: Largest separation inside a gap cluster, in units of that cluster's local length.
    worst_gap_in_lengths: float
    n_chained_clusters: int
    n_nonmanifold_edges: int          # edges borne by more than two quads
    n_degenerate_quads: int           # a quad naming one point twice
    n_collapsed_edges: int            # a quad edge between two stored points at the same place
    n_pinch_vertices: int             # a vertex whose quads form more than one fan (see pinch_vertices)
    passed: bool
    notes: tuple[str, ...]

    def __str__(self) -> str:
        head = ("connectivity (gencon tol %.2f): %d points, %d quads, %d component(s); "
                "%d gap cluster(s) (%d points, %d on the boundary, worst %.3f local lengths), "
                "%d chained; %d non-manifold edge(s); %d degenerate quad(s); %d collapsed edge(s); %d pinch vertex(es) -> %s"
                % (self.connectivity_tol, self.n_points, self.n_quads, self.n_components,
                   self.n_gap_clusters, self.gap_points.size, self.n_gap_on_boundary,
                   self.worst_gap_in_lengths, self.n_chained_clusters, self.n_nonmanifold_edges,
                   self.n_degenerate_quads, self.n_collapsed_edges, self.n_pinch_vertices,
                   "PASS" if self.passed else "FAIL"))
        return head + "".join("\n  - " + n for n in self.notes)


#: A quad edge no longer than this fraction of the model size is a *collapsed* edge: two stored points at
#: the same place.  Tight on purpose -- thin boundary layers are legitimate, coincident points are not.
COLLAPSE_FRACTION = 1e-9


def pinch_vertices(mesh: QuadMesh) -> IntArray:
    """Ids of the **pinch vertices**: points whose incident quads do not form one connected fan.

    Around a vertex each quad is a wedge between its two edges at that vertex; two wedges are
    neighbours when they share an edge.  A manifold vertex has one fan (a closed ring inside, an
    open one on the boundary); a vertex where two patches, or two sides of a folded sheet, touch
    has two or more.  Neither the gap check, the edge counts nor ``is_conforming`` sees it -- every
    edge is still used once or twice -- but the boundary angle-defect formula assumes one fan, so
    ``gauss_bonnet_residual`` does fire.  Cost: one union-find over ``4Q`` wedges."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    c = np.asarray(mesh.corners, dtype=np.int64)
    n = mesh.n_points
    if c.shape[0] == 0:
        return np.zeros(0, dtype=np.int64)
    v = c.ravel()
    nxt = np.roll(c, -1, axis=1).ravel()
    prv = np.roll(c, 1, axis=1).ravel()
    # a node of the graph is a (vertex, neighbouring vertex) edge-end; a wedge joins its two ends
    keys, inv = np.unique(np.concatenate([v * n + nxt, v * n + prv]), return_inverse=True)
    w = v.size
    graph = coo_matrix((np.ones(w), (inv[:w], inv[w:])), shape=(keys.size, keys.size))
    _, label = connected_components(graph, directed=False)
    pair = np.unique(np.stack([keys // n, label], axis=1), axis=0)
    return np.flatnonzero(np.bincount(pair[:, 0], minlength=n) > 1).astype(np.int64)


def check_connectivity(mesh: QuadMesh, *, connectivity_tol: float = 0.2) -> ConnectivityReport:
    """Preflight **gap check**: run gencon's relative-tolerance vertex rule over the mesh's own
    corner coordinates and report every pair of stored points it would fuse but the mesh keeps
    apart (an open seam, a duplicated vertex), plus non-manifold edges, degenerate quads and the
    component count.  Read-only -- fixing a gap is a separate, explicit act.

    A mesh the toolkit built is conformal by construction, so on its own output this only
    reports what a coordinate-based weld would still find (a seam closer than
    ``connectivity_tol`` of the local size that ``merge`` left open).  It cannot see a leak
    between two distinct objects that are welded: that needs the component and
    non-manifold lines, and the user's eye on the picture."""
    corners = mesh.corners
    P = np.asarray(mesh.points, dtype=float)
    soup = P[corners]                                            # (Q, 4, 3)
    X = soup.reshape(-1, 3)
    stored = corners.reshape(-1)
    # the local length of a *vertex* is its shortest edge over every quad that shares it: a corner
    # copy taken from a coarse neighbour must not make the rule lenient at a thin layer's vertex
    ell_corner = conform.corner_length(soup).reshape(-1)
    ell_vertex = np.full(mesh.n_points, np.inf)
    np.minimum.at(ell_vertex, stored, ell_corner)
    ell = ell_vertex[stored]
    cl = conform.box_refine_clusters(X, ell, connectivity_tol,
                                     conform.gencon_clusters(X, ell, connectivity_tol), stored)
    # clusters that hold more than one *stored* point id are gaps
    key = np.unique(np.stack([cl, stored], axis=1), axis=0)      # distinct (cluster, point), by cluster
    per = np.bincount(key[:, 0], minlength=int(cl.max()) + 1 if cl.size else 0)
    multi = np.flatnonzero(per > 1)
    cl4 = cl.reshape(-1, 4)
    # a cluster with two corners of one quad is a fold, whatever the distance
    folded: BoolArray = np.zeros(per.size, dtype=bool)
    for j in range(4):
        for k in range(j + 1, 4):
            folded[cl4[cl4[:, j] == cl4[:, k], j]] = True
    start = np.concatenate([[0], np.cumsum(per)])
    real: list[int] = []
    chained = 0
    worst = 0.0
    for c in multi:
        ids = key[start[c]:start[c + 1], 1]
        sep = float(np.linalg.norm(P[ids][:, None, :] - P[ids][None, :, :], axis=2).max())
        local = float(ell_vertex[ids].min())
        ratio = sep / local if local > 0 else np.inf
        if ratio <= connectivity_tol * (1.0 + 1e-9) and not folded[c]:   # a genuine near-duplicate
            real.append(int(c))
            worst = max(worst, ratio)
        else:
            chained += 1
    bad: IntArray = np.asarray(real, dtype=np.int64)
    gap_ids = np.unique(key[np.isin(key[:, 0], bad), 1]) if bad.size else np.zeros(0, dtype=np.int64)
    bnd: BoolArray = np.zeros(mesh.n_points, dtype=bool)
    bnd[boundary_points(mesh)] = True
    use = np.bincount(np.asarray(mesh.quads, dtype=np.int64).ravel(),
                      minlength=mesh.line_mesh.n_lines)
    nonman = int((use > 2).sum())
    degen = int(((corners[:, :, None] == corners[:, None, :]).sum(axis=(1, 2)) > 4).sum())
    ncomp = conform.element_components(corners, mesh.n_points)[0] if mesh.n_quads else 0
    ea, eb = corners.ravel(), np.roll(corners, -1, axis=1).ravel()
    elen = np.linalg.norm(P[ea] - P[eb], axis=1)
    ncollapsed = int(np.unique(np.sort(np.stack([ea, eb], axis=1)[(ea != eb) & (elen <= COLLAPSE_FRACTION * conform.bbox_scale(P))],
                                       axis=1), axis=0).shape[0]) if elen.size else 0
    notes: list[str] = []
    if bad.size:
        notes.append("%d points would be fused by gencon (tol %.2f of local length) but are "
                     "stored apart: patch the connectivity before smoothing" % (gap_ids.size,
                                                                                 connectivity_tol))
    if chained:
        notes.append("%d group(s) of non-adjacent points sit close together (chained by gencon's "
                     "box rule, farther apart than %.2f of the local length): a possible tangle, "
                     "not a seam -- untangle, do not weld" % (chained, connectivity_tol))
    if nonman:
        notes.append("%d edges are used by more than two quads" % nonman)
    if degen:
        notes.append("%d quads name one point twice" % degen)
    if ncollapsed:
        notes.append("%d quad edge(s) join two stored points at the same place (a collapsed edge is not a "
                     "valid quad): weld the two points or remove the quad" % ncollapsed)
    npinch = int(pinch_vertices(mesh).size)
    if npinch:
        notes.append("%d vertices are shared by more than one fan of quads (two patches touching "
                     "at a point): a pinch, not a weld -- split or fuse the patches" % npinch)
    if ncomp > 1:
        notes.append("%d disconnected components (intended? an open interface between "
                     "blocks looks like this)" % ncomp)
    return ConnectivityReport(mesh.n_points, mesh.n_quads, ncomp, connectivity_tol,
                              int(bad.size), gap_ids, int(bnd[gap_ids].sum()), worst, chained, nonman,
                              degen, ncollapsed, npinch,
                              bool(bad.size == 0 and nonman == 0 and degen == 0 and ncollapsed == 0
                                   and npinch == 0),
                              tuple(notes))


def _shorter_diagonal_triangles(P: PointArray, corners: IntArray) -> IntArray:
    """``(2Q, 3)`` triangles: each quad split along its shorter diagonal (corner ids)."""
    c0, c1, c2, c3 = (corners[:, k] for k in range(4))
    use02 = (np.linalg.norm(P[c0] - P[c2], axis=1) <= np.linalg.norm(P[c1] - P[c3], axis=1))[:, None]
    a = np.where(use02, np.stack([c0, c1, c2], 1), np.stack([c1, c2, c3], 1))
    b = np.where(use02, np.stack([c0, c2, c3], 1), np.stack([c1, c3, c0], 1))
    return np.concatenate([a, b]).astype(np.int64)


def angle_defect(mesh: QuadMesh) -> FloatArray:
    """``(P,)`` discrete Gaussian curvature of every vertex by the **angle defect**, read off the
    mesh with each quad split along its shorter diagonal: ``2 pi - sum(angles)`` at an interior
    vertex, ``pi - sum(angles)`` at a boundary one (the turning of the boundary).  The integral of
    the Gaussian curvature over the vertex's dual cell, to the mesh's own resolution.

    The quads are triangulated first on purpose: a non-planar quad's four corner angles sum to less
    than ``2 pi``, so defects read off the raw corners do not add up to ``2 pi chi``; those of a
    triangle mesh always do (see :func:`gauss_bonnet_residual`)."""
    P = np.asarray(mesh.points, dtype=float)
    tri = _shorter_diagonal_triangles(P, mesh.corners)
    ang = np.zeros(P.shape[0])
    for k in range(3):
        u = P[tri[:, (k + 1) % 3]] - P[tri[:, k]]
        v = P[tri[:, (k + 2) % 3]] - P[tri[:, k]]
        cosang = np.einsum("ij,ij->i", u, v) / np.maximum(
            np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1), 1e-300)
        ang += np.bincount(tri[:, k], weights=np.arccos(np.clip(cosang, -1.0, 1.0)),
                           minlength=P.shape[0])
    bnd = np.zeros(P.shape[0], dtype=bool)
    bnd[boundary_points(mesh)] = True
    return np.asarray(np.where(bnd, np.pi, 2.0 * np.pi) - ang)


def gauss_bonnet_residual(mesh: QuadMesh) -> float:
    """``sum(angle_defect) - 2 pi chi`` with ``chi = V - E + F``: zero (to rounding) for any
    triangulated surface, with or without boundary.  It does not measure quality -- every mesh
    satisfies it -- it catches a corrupted topology or an angle bug, and it must hold before
    *and* after a smoothing round.

    What it can and cannot see.  On a closed mesh the residual is exactly ``2 pi (E - 2Q)``, so
    it is non-zero only when the edge count is inconsistent with a closed surface: it jumps by
    ``2 pi`` for every non-manifold fin and by ``pi`` for a quad that names one vertex twice.  On a mesh
    with a boundary it also fires on a pinch vertex (:func:`pinch_vertices`).  It
    is blind to an unwelded seam (the cut mesh is still a valid surface with a boundary), to
    reversed winding, to self-intersection, to a collapsed edge, to holes and to isolated
    vertices -- :func:`check_connectivity` is the check for the first of these, not this one."""
    chi = mesh.n_points - mesh.edges.shape[0] + mesh.n_quads
    return float(angle_defect(mesh).sum() - 2.0 * np.pi * chi)


__all__ = [
    "angle_defect",
    "gauss_bonnet_residual",
    "pinch_vertices",
    "ConnectivityReport",
    "check_connectivity",
    "TagReport",
    "area",
    "bounds",
    "boundary_edge_ids",
    "boundary_edges",
    "boundary_elements",
    "boundary_points",
    "centroid",
    "edge_rows",
    "edge_tag_rows",
    "element_areas",
    "element_blocks",
    "is_conforming",
    "is_overlap_free",
    "overlapping_quads",
    "is_watertight",
    "plane_normal",
    "quality_summary",
    "report",
    "scaled_jacobian",
    "tag_report",
    "tagged_edges",
    "topology_report",
]
