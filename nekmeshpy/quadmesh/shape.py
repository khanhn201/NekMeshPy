"""Shape factories for the ``QuadMesh`` rung -- the ones owning a *shape model* rather
than being generic over any input."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Callable

import numpy as np

from .._typing import (
    FloatArray,
    IntArray,
    Point,
    PointArray,
    SmoothingMethod,
    StrArray,
    Vec3,
)
from ..core import conform, surfaces
from ..core.fields import gll_nodes, validate_layers
from ..core.interp import coons_grid, coons_grid_fn
from ..core.surfaces import SurfaceCurve, SurfaceMap
from ..core.tags import ElementTags
from ..linemesh import LineMesh
from ..linemesh.assemble import loft as line_loft
from ..linemesh.assemble import loft_fn as line_loft_fn
from ..linemesh.morph import reverse as line_reverse
from ..linemesh.shape import line
from ._helpers import Overlay, _apply_smoothing, _check_boundary, _elevate, entities_from_blocks
from .assemble import loft_fn, merge
from .lift import from_grid
from .quadmesh import NO_TAG, QuadMesh
from .query import boundary_edges
from .tag import tag_edges

#: The four sides of a :func:`structured <nekmeshpy.quadmesh.shape.structured>` patch, in the CCW loop order its ``edges``
#: are consumed in.  Both the ``edges`` mapping and ``side_tags`` are keyed by these.
_SIDES = ("bottom", "right", "top", "left")


def _seg_tags(curve: LineMesh) -> list[str] | None:
    """``curve``'s element tags densified to a ``list[str]``, or ``None`` if every
    element is untagged (so an untagged curve stays untagged)."""
    if not curve.element_tags:
        return None
    return [str(x) for x in curve.element_tags.dense(curve.n_lines).tolist()]


def _ordered_sides(edges: Sequence[LineMesh] | Mapping[str, LineMesh],
                   ) -> list[LineMesh]:
    """``edges`` as the positional ``[bottom, right, top, left]`` list :func:`structured
    <nekmeshpy.quadmesh.shape.structured>` works in, accepting either spelling."""
    if isinstance(edges, Mapping):
        missing = [s for s in _SIDES if s not in edges]
        extra = [s for s in edges if s not in _SIDES]
        if missing or extra:
            raise ValueError(
                "structured edges mapping must have exactly the keys "
                "bottom/right/top/left (missing %s, unexpected %s)"
                % (missing or "none", extra or "none"))
        return [edges[s] for s in _SIDES]
    return list(edges)


def rectangle(corners: PointArray | Sequence[Point], nx: int, ny: int, *,
              x_frac: FloatArray | None = None,
              y_frac: FloatArray | None = None,
              side_tags: Mapping[str, str] | None = None,
              smoothing_method: SmoothingMethod | None = None,
              order: int = 1) -> QuadMesh:
    """Structured quad grid over the rectangle with four CCW corners ``corners = [c0,
    c1, c2, c3]``: ``nx`` cells along ``c0->c1`` (bottom/top), ``ny`` along ``c1->c2``
    (left/right)."""
    c = np.asarray(corners, dtype=float).reshape(-1, 3)
    if c.shape[0] != 4:
        raise ValueError("rectangle needs exactly 4 corners")
    xf = (np.linspace(0.0, 1.0, nx + 1) if x_frac is None
          else np.asarray(x_frac, dtype=float).ravel())
    yf = (np.linspace(0.0, 1.0, ny + 1) if y_frac is None
          else np.asarray(y_frac, dtype=float).ravel())
    st = side_tags or {}
    specs = (("bottom", c[0], c[1], xf), ("right", c[1], c[2], yf),
             ("top", c[2], c[3], xf), ("left", c[3], c[0], yf))
    edges = [line(a, b, frac, element_tag=st.get(side, ""), order=order)
             for side, a, b, frac in specs]
    return structured(edges, smoothing_method=smoothing_method)


def _chain_intervals(edge: LineMesh, name: str) -> IntArray:
    """``(L,)`` map from the edge's line index to the point interval ``i`` it spans."""
    lines: IntArray = np.asarray(edge.lines, dtype=np.int64).reshape(-1, 2)
    lo: IntArray = np.minimum(lines[:, 0], lines[:, 1])
    hi: IntArray = np.maximum(lines[:, 0], lines[:, 1])
    n = edge.points.shape[0] - 1
    if (lines.shape[0] != n or not np.array_equal(hi, lo + 1)
            or not np.array_equal(np.sort(lo), np.arange(n, dtype=np.int64))):
        raise ValueError(
            "%s must be a simple consecutive chain (line k joining points k and k+1) "
            "-- its point order is what sets the node distribution; build it with "
            "LineMesh.loft/line/arc/loft_fn (or merge chains end to end) rather than "
            "re-indexing its lines" % name)
    return lo


def _grid_quads(ni: int, nj: int) -> IntArray:
    """The ``(ni*nj, 4)`` CCW quad table of an ``(ni+1) x (nj+1)`` structured point grid
    numbered ``id(i, j) = i*(nj+1) + j``, in i-major / j-minor element order (quad ``(i,
    j)`` is row ``i*nj + j``)."""
    row = nj + 1
    i: IntArray = np.repeat(np.arange(ni, dtype=np.int64), nj)
    j: IntArray = np.tile(np.arange(nj, dtype=np.int64), ni)
    return np.stack([i * row + j, (i + 1) * row + j,
                     (i + 1) * row + j + 1, i * row + j + 1], axis=1)


def _refined_params(n: int, order: int) -> FloatArray:
    """The ``n*order+1`` transfinite parameters of an ``n``-interval edge at order N."""
    if order == 1:
        return np.linspace(0.0, 1.0, n + 1)
    g = gll_nodes(order)
    inner: FloatArray = (np.arange(n, dtype=float)[:, None] + g[None, :order]).ravel()
    return np.concatenate([inner / n, [1.0]])


def _refined_chain(edge: LineMesh, name: str) -> PointArray:
    """An edge's own nodes in point-index order: ``(n*order+1, 3)``."""
    pts: PointArray = edge.points
    order = edge.order
    if order == 1:
        return pts
    lo = _chain_intervals(edge, name)
    lines: IntArray = np.asarray(edge.lines, dtype=np.int64).reshape(-1, 2)
    inner: PointArray = edge.interior                  # (L, order-1, 3)
    out: PointArray = np.empty(((pts.shape[0] - 1) * order + 1, 3), dtype=float)
    out[::order] = pts
    for k in range(lines.shape[0]):
        i = int(lo[k])
        blk = inner[k] if lines[k, 0] == i else inner[k][::-1]
        out[i * order + 1:i * order + order] = blk
    return out


def _straight_interior(pos: PointArray, lines: IntArray,
                       order: int) -> PointArray:
    """``(L,order-1,3)`` straight GLL interiors of the chain ``lines`` over ``pos``."""
    g = gll_nodes(order)[1:order]
    a, b = pos[lines[:, 0]], pos[lines[:, 1]]
    return a[:, None, :] + g[None, :, None] * (b - a)[:, None, :]


def _blended_ring(pos: PointArray, wall: LineMesh, tau: float,
                  peri_pos: PointArray, peri_inner: PointArray) -> LineMesh:
    """The O-ring curve at radial fraction ``tau``, as a ``LineMesh`` to overlay."""
    lines: IntArray = wall.lines
    order = wall.order
    if order == 1:
        return LineMesh(pos, lines)
    inner: PointArray = (1.0 - tau) * peri_inner + tau * wall.interior
    drift: PointArray = pos - ((1.0 - tau) * peri_pos + tau * wall.points)
    g = gll_nodes(order)[1:order]
    d0, d1 = drift[lines[:, 0]], drift[lines[:, 1]]
    inner = (inner + (1.0 - g)[None, :, None] * d0[:, None, :]
             + g[None, :, None] * d1[:, None, :])
    return LineMesh(pos, lines, interior=inner)


def _ring_overlays(ring_pts: Sequence[PointArray], wall: LineMesh,
                   radial: FloatArray, peri_pos: PointArray, q0: int,
                   width: int, outer_side: int) -> list[Overlay]:
    """Both incident copies of every O-ring curve, as ``_elevate`` overlays."""
    inner_side = 3 if outer_side == 1 else 1
    peri_inner = _straight_interior(peri_pos, wall.lines, wall.order)
    ks: IntArray = np.arange(width, dtype=np.int64)
    nr = len(ring_pts)
    overlays: list[Overlay] = []
    for r, pts in enumerate(ring_pts):
        ring_lm = _blended_ring(pts, wall, float(radial[r + 1]),
                                peri_pos, peri_inner)
        overlays.append((q0 + r * width + ks, outer_side, ring_lm))
        if r + 1 < nr:
            overlays.append((q0 + (r + 1) * width + ks, inner_side, ring_lm))
    return overlays


def _curve_rows(rows: Sequence[tuple[int, int]], curve: LineMesh,
               override: str) -> tuple[IntArray, StrArray]:
    """``((K,2) (quad, side) rows, their names)`` naming one side of a region.

    Left in element-local terms because that is what a region fill knows; ``tag_edges``
    resolves each row to the shared edge it points at."""
    seg = _seg_tags(curve)
    names = [override if override else (seg[m] if seg is not None else "")
             for m in range(len(rows))]
    return (np.asarray(rows, dtype=np.int64).reshape(-1, 2),
            np.asarray(names, dtype=np.str_))


def _sub_chain(chain: LineMesh, segs: IntArray, seg2line: IntArray) -> LineMesh:
    """The sub-``LineMesh`` of ``chain`` holding point intervals ``segs``, in that
    order."""
    idx: IntArray = seg2line[np.asarray(segs, dtype=np.int64)]
    return LineMesh(chain.points, chain.lines[idx],
                    interior=chain.interior[idx])


def _slice_chain(chain: LineMesh, a: int, b: int) -> LineMesh:
    """Points ``[a, b]`` of ``chain`` as their own ``LineMesh`` -- interior nodes and
    per-segment tags carried verbatim, nothing resampled. ``chain`` may be an open
    chain (``b < len(points)``) or a closed loop being cut at its wrap (``b ==
    len(points)``, closing through point 0)."""
    pts = chain.points
    M = pts.shape[0]
    seg = _seg_tags(chain)
    if b < M:
        p, interior = pts[a:b + 1], chain.interior[a:b]
        tags = None if seg is None else seg[a:b]
    else:
        p = np.vstack([pts[a:M], pts[0:1]])
        interior = chain.interior[a:M]
        tags = None if seg is None else seg[a:M]
    lm = line_loft(p, interior=interior, order=chain.order)
    if tags is None:
        return lm
    return LineMesh(lm.point_mesh, lm.lines, lm.interior, ElementTags.from_dense(tags))


def structured(edges: Sequence[LineMesh] | Mapping[str, LineMesh], *,
               side_tags: Mapping[str, str] | None = None,
               smoothing_method: SmoothingMethod | None = None) -> QuadMesh:
    """Transfinite (Coons-patch) quad grid over the surface bounded by four open edge
    lines ``edges = [bottom, right, top, left]`` in CCW loop order. The lines must share
    corners (form a closed loop)."""
    edge_list = _ordered_sides(edges)
    if len(edge_list) != 4:
        raise ValueError("structured needs exactly 4 edge lines "
                         "[bottom, right, top, left]")
    bottom, right, top, left = edge_list
    for nm, e in (("bottom", bottom), ("right", right),
                  ("top", top), ("left", left)):
        _check_boundary(e, "structured " + nm + " edge", 2)
    order = bottom.order
    if any(e.order != order for e in edge_list):
        raise ValueError("structured: all four edges must share the same order")
    # resolution comes from the edges' own point counts (no resampling)
    if bottom.points.shape[0] != top.points.shape[0]:
        raise ValueError(
            "structured: bottom and top edges must have equal point counts "
            "(got %d, %d); resample them to the same nx+1 first"
            % (bottom.points.shape[0], top.points.shape[0]))
    if left.points.shape[0] != right.points.shape[0]:
        raise ValueError(
            "structured: left and right edges must have equal point counts "
            "(got %d, %d); resample them to the same ny+1 first"
            % (left.points.shape[0], right.points.shape[0]))
    nx = bottom.points.shape[0] - 1
    ny = left.points.shape[0] - 1
    # the four edges must close into a loop (share corners) in CCW order
    allpts = np.vstack([e.points for e in edge_list])
    scale = float(np.max(allpts.max(axis=0) - allpts.min(axis=0)))
    tol = 1e-6 * scale if scale > 0 else 1e-9
    for lbl, p, q in (("bottom->right", bottom.points[-1], right.points[0]),
                      ("right->top", right.points[-1], top.points[0]),
                      ("top->left", top.points[-1], left.points[0]),
                      ("left->bottom", left.points[-1], bottom.points[0])):
        gap = float(np.linalg.norm(p - q))
        if gap > tol:
            raise ValueError(
                "structured: edges must form a closed loop in CCW order "
                "[bottom, right, top, left] with shared corners; "
                "gap %.3g at %s" % (gap, lbl))
    # Sample the transfinite map at the order-N lattice: at order 1 that is the plain
    # corner grid, at order N the GLL-refined one.  Every node -- interior edges and
    # element interiors included -- therefore lands on the true Coons surface spanned
    # by the four edges, instead of on a straight subdivision of a corner-only grid.
    # Both edge families are oriented to run with the lattice: bottom/top c0->c1 and
    # c3->c2 along u, left/right c0->c3 and c1->c2 along v, so ``top``/``left`` (stored
    # c2->c3 / c3->c0) are reversed.
    S = coons_grid(_refined_chain(bottom, "structured bottom edge"),
                   _refined_chain(top, "structured top edge")[::-1],
                   _refined_chain(left, "structured left edge")[::-1],
                   _refined_chain(right, "structured right edge"),
                   _refined_params(nx, order), _refined_params(ny, order))
    points = S[::order, ::order].reshape(-1, 3)      # id(i,j) = i*(ny+1) + j

    # quads in i-major / j-minor order (i in [0,nx), j in [0,ny))
    quads = _grid_quads(nx, ny)
    # boundary edges as [quad id, side]; quad id = i*ny + j (i-major).  With
    # v0=(i,j) v1=(i+1,j) v2=(i+1,j+1) v3=(i,j+1): bottom (j=0) is side 1,
    # right (i=nx-1) side 2, top (j=ny-1) side 3, left (i=0) side 4.
    side_rows: dict[str, list[tuple[int, int]]] = {
        "bottom": [(i * ny + 0, 1) for i in range(nx)],
        "right": [((nx - 1) * ny + j, 2) for j in range(ny)],
        "top": [(i * ny + (ny - 1), 3) for i in range(nx)],
        "left": [(0 * ny + j, 4) for j in range(ny)],
    }
    side_edges = {"bottom": bottom, "right": right, "top": top, "left": left}
    bt = side_tags or {}
    for side in bt:
        if side not in side_rows:
            raise ValueError("structured side_tags side must be one of "
                             "bottom/right/top/left, got %r" % side)
    # each side is named by its edge's uniform element tag; a non-empty
    # side_tags[side] overrides, a present-but-empty entry suppresses it.
    rows_all: list[tuple[int, int]] = []
    names_all: list[str] = []
    for side, rows in side_rows.items():
        if side in bt:
            nm = bt[side]
        else:
            et = side_edges[side].element_group_tags
            nm = et[0] if len(et) == 1 else ""
        if not nm:                       # NO_TAG / "" / untagged -> no row
            continue
        rows_all.extend(rows)
        names_all.extend([nm] * len(rows))
    qm = tag_edges(QuadMesh.from_corners(points, quads),
                   np.asarray(rows_all, dtype=np.int64).reshape(-1, 2),
                   np.asarray(names_all, dtype=np.str_))
    if order > 1:
        # Every node already exists in ``S``; cut it into per-element blocks and let
        # the B-rep tables fall out.  No overlay is needed -- the boundary rows of the
        # lattice *are* the edges' own nodes, so the walls are exact for free, and
        # unlike a stamp-the-walls-on elevation the interior edges are curved too.
        # Quad (i,j) = quad id i*ny + j spans lattice rows i*order..(i+1)*order and
        # columns j*order..(j+1)*order; its local (i,j) axes are the global ones.
        qi: IntArray = np.repeat(np.arange(nx, dtype=np.int64), ny)
        qj: IntArray = np.tile(np.arange(ny, dtype=np.int64), nx)
        a: IntArray = np.arange(order + 1, dtype=np.int64)
        blk: PointArray = S[(qi[:, None, None] * order + a[None, :, None]),
                            (qj[:, None, None] * order + a[None, None, :])]
        # (Q,a,b,3) -> lexicographic (i fastest) slot b*(order+1) + a
        blocks: PointArray = blk.transpose(0, 2, 1, 3).reshape(
            quads.shape[0], (order + 1) ** 2, 3)
        lm, elem_edges, flip, interior = entities_from_blocks(
            blocks, quads, points, order, "QuadMesh.structured")
        # ``quads`` is unchanged, so the rebuilt edge table pairs with the old one
        # through the same (quad, side) rows the tags were authored in
        qm = tag_edges(QuadMesh(lm, elem_edges, flip, interior, qm.element_tags),
                       np.asarray(rows_all, dtype=np.int64).reshape(-1, 2),
                       np.asarray(names_all, dtype=np.str_))
    # smooth last: a repositioning smoother rejects order > 1 (high-order smoothing
    # is not implemented).
    return _apply_smoothing(qm, smoothing_method)


def ogrid(boundary: LineMesh, n_side: int, radial: int | FloatArray, *,
          center_scale: float = 0.7, quadrant_scale: float = 0.7,
          wall_tag: str = "", smoothing_method: SmoothingMethod | None = None) -> QuadMesh:
    """O-grid filling the closed ``boundary``: four :func:`quadrant_ogrid
    <nekmeshpy.quadmesh.shape.quadrant_ogrid>` quarters meeting at the loop's centroid,
    split at the loop's own quarter points. ``center_scale`` places each quarter's own
    hub corner; ``quadrant_scale`` places the seam corner shared between neighbours (via
    :func:`quadrant_seam_fractions <nekmeshpy.quadmesh.shape.quadrant_seam_fractions>`)
    -- see there for what each one does."""
    if n_side < 1:
        raise ValueError("ogrid needs n_side >= 1")
    if n_side % 2 != 0:
        raise ValueError(
            "ogrid needs an even n_side (it is built from four quadrant_ogrid "
            "quarters, each needing a wall midpoint), got n_side=%d" % n_side)
    bpts = _check_boundary(boundary, "ogrid boundary", 3)
    # wall ring = the boundary loop itself, meshed exactly: it must already carry
    # P = 4*n_side points (the caller sizes it, e.g. circle(R, 4*n_side)), split at
    # its quarter points into the four quadrant arcs.
    P = 4 * n_side
    if bpts.shape[0] != P:
        raise ValueError(
            "ogrid boundary must have exactly 4*n_side = %d points to be meshed "
            "exactly (got %d); size the loop to match, e.g. circle(R, %d)"
            % (P, bpts.shape[0], P))
    centroid = bpts.mean(axis=0)
    rad = float(np.mean(np.linalg.norm(bpts - centroid, axis=1)))
    if rad <= 0.0:
        raise ValueError("ogrid: boundary is degenerate (all points coincide)")
    n = n_side // 2
    radial = validate_layers(radial, "ogrid radial")
    fr = quadrant_seam_fractions(n, radial, quadrant_scale)
    starts = [0, n_side, 2 * n_side, 3 * n_side]
    arcs = [_slice_chain(boundary, a, b)
            for a, b in zip(starts, starts[1:] + [P])]
    seams = [line(centroid, a.points[0], fr, order=boundary.order) for a in arcs]
    seams.append(seams[0])
    return _apply_smoothing(
        merge([quadrant_ogrid(arcs[q], seams[q], seams[q + 1], radial,
                              center_scale=center_scale, wall_tag=wall_tag)
              for q in range(4)]),
        smoothing_method)


def half_ogrid(arc: LineMesh, spine: LineMesh,
               radial: int | FloatArray, *, center_scale: float = 0.7,
               quadrant_scale: float = 0.7, wall_tag: str = "",
               smoothing_method: SmoothingMethod | None = None) -> QuadMesh:
    """Structured half-circle O-grid over a half-disk split along the ``spine`` line
    (A1..A2); the wall ``arc`` (``(4*Ntheta+1, 3)``, arc[0]=A1, arc[-1]=A2) is the open
    boundary. Built as two :func:`quadrant_ogrid
    <nekmeshpy.quadmesh.shape.quadrant_ogrid>` quarters split at the arc's own apex (its
    midpoint, directly opposite the spine) and merged there: the spine's own [north
    caps, center fan, south caps] halves become the ``O -> A1`` / ``O -> A2`` seams
    verbatim (nothing resampled, same as the rest of ``spine``), and a fresh ``O ->
    apex`` seam (via :func:`quadrant_seam_fractions
    <nekmeshpy.quadmesh.shape.quadrant_seam_fractions>`) is the only new geometry.
    ``center_scale``/``quadrant_scale`` are as there."""
    apts = _check_boundary(arc, "half_ogrid arc", 5)   # (na,3) backing array
    na = apts.shape[0]
    if (na - 1) % 4 != 0:
        raise ValueError("half_ogrid: arc must have 4*Ntheta+1 points (Ntheta >= 1)")
    Nt = (na - 1) // 4
    radial = validate_layers(radial, "half_ogrid radial")
    Nr = radial.size - 1

    # spine is meshed exactly: its points must be sampled monotonically along the
    # diameter, A1 (fraction 0) -> A2 (fraction 1) -- Nr north caps, then the 2Nt+1
    # center fan, then Nr south caps (see half_ogrid docstring for the fractions).
    sp = _check_boundary(spine, "half_ogrid spine", 2)
    n_spine = 2 * Nt + 1 + 2 * Nr
    if sp.shape[0] != n_spine:
        raise ValueError(
            "half_ogrid spine must have exactly 2*Ntheta+1 + 2*Nradial = %d points "
            "(got %d); sample the spine curve monotonically A1 -> A2 as [north caps, "
            "center fan, south caps] (see half_ogrid docstring)"
            % (n_spine, sp.shape[0]))
    if spine.order != arc.order:
        raise ValueError(
            "half_ogrid spine and arc must share an order (got spine order %d, arc "
            "order %d); the spine's own nodes are the seam geometry, so a lower-order "
            "spine cannot describe a higher-order seam"
            % (spine.order, arc.order))

    o_idx = Nr + Nt                             # the spine's own midpoint, O
    seam_a1 = line_reverse(_slice_chain(spine, 0, o_idx))       # O -> A1, exact nodes
    seam_a2 = _slice_chain(spine, o_idx, sp.shape[0] - 1)       # O -> A2, exact nodes
    fr = quadrant_seam_fractions(Nt, radial, quadrant_scale)
    seam_apex = line(sp[o_idx], apts[2 * Nt], fr, order=arc.order)

    arc_a = _slice_chain(arc, 0, 2 * Nt)         # A1 -> apex
    arc_b = _slice_chain(arc, 2 * Nt, na - 1)    # apex -> A2
    qa = quadrant_ogrid(arc_a, seam_a1, seam_apex, radial,
                        center_scale=center_scale, wall_tag=wall_tag)
    qb = quadrant_ogrid(arc_b, seam_apex, seam_a2, radial,
                        center_scale=center_scale, wall_tag=wall_tag)
    return _apply_smoothing(merge([qa, qb]), smoothing_method)


#: The two nameable seams of a :func:`quadrant_ogrid <nekmeshpy.quadmesh.shape.quadrant_ogrid>`, in the order its arguments
#: are taken.  ``side_tags`` is keyed by these.
_QUADRANT_SEAMS = ("seam1", "seam2")


def quadrant_core(arc: LineMesh, seam1: LineMesh, seam2: LineMesh, *,
                  center_scale: float = 0.5) -> PointArray:
    """The core quarter of a :func:`quadrant_ogrid
    <nekmeshpy.quadmesh.shape.quadrant_ogrid>` as an ``(n+1, n+1, 3)`` grid, indexed
    ``[i][j]`` with ``i`` running ``O -> M1`` along ``seam1`` and ``j`` running ``O ->
    M2`` along ``seam2``."""
    apts = _check_boundary(arc, "quadrant_core arc", 3)
    na = apts.shape[0]
    if na < 3 or (na - 1) % 2 != 0:
        raise ValueError(
            "quadrant_core arc must have 2*n+1 points (n >= 1), got %d" % na)
    n = (na - 1) // 2
    if not 0.0 < center_scale < 1.0:
        raise ValueError("quadrant_core needs center_scale in (0, 1)")
    s1 = _check_boundary(seam1, "quadrant_core seam1", n + 1)
    s2 = _check_boundary(seam2, "quadrant_core seam2", n + 1)
    o: Point = s1[0]
    m1, m2 = s1[n], s2[n]
    k: Point = o + center_scale * (apts[n] - o)
    t: FloatArray = np.arange(n + 1, dtype=float) / n
    return coons_grid(s1[0:n + 1], m2 + t[:, None] * (k - m2),      # j = 0 / j = n
                      s2[0:n + 1], m1 + t[:, None] * (k - m1), t, t)  # i = 0 / i = n


def quadrant_ogrid(arc: LineMesh, seam1: LineMesh, seam2: LineMesh,
                   radial: int | FloatArray, *, center_scale: float = 0.5,
                   wall_tag: str = "",
                   side_tags: Mapping[str, str] | None = None,
                   smoothing_method: SmoothingMethod | None = None) -> QuadMesh:
    """Quarter-disk O-grid: the 90-degree sibling of :func:`half_ogrid
    <nekmeshpy.quadmesh.shape.half_ogrid>`."""
    apts = _check_boundary(arc, "quadrant_ogrid arc", 3)
    na = apts.shape[0]
    if na < 3 or (na - 1) % 2 != 0:
        raise ValueError(
            "quadrant_ogrid arc must have 2*n+1 points (n >= 1), got %d" % na)
    n = (na - 1) // 2
    if not 0.0 < center_scale < 1.0:
        raise ValueError("quadrant_ogrid needs center_scale in (0, 1)")
    radial = validate_layers(radial, "quadrant_ogrid radial")
    Nr = radial.size - 1
    st = dict(side_tags or {})
    extra = [s for s in st if s not in _QUADRANT_SEAMS]
    if extra:
        raise ValueError(
            "quadrant_ogrid side_tags must be keyed seam1/seam2, got unexpected %s"
            % extra)

    n_seam = n + 1 + Nr
    seams = (seam1, seam2)
    for which, sm in zip(_QUADRANT_SEAMS, seams):
        pts = _check_boundary(sm, "quadrant_ogrid " + which, 2)
        if pts.shape[0] != n_seam:
            raise ValueError(
                "quadrant_ogrid %s must have exactly n+1 + Nradial = %d points "
                "ascending from the center O (got %d); it is meshed exactly at the "
                "points given, never resampled -- evaluate your radius curve at "
                "quadrant_seam_fractions(%d, radial, center_scale)"
                % (which, n_seam, pts.shape[0], n))
        if sm.order != arc.order:
            raise ValueError(
                "quadrant_ogrid %s and arc must share an order (got %s order %d, arc "
                "order %d); the seam's own nodes are the radius geometry, so a "
                "lower-order seam cannot describe a higher-order edge"
                % (which, which, sm.order, arc.order))
    s1: PointArray = seam1.points
    s2: PointArray = seam2.points
    tol = conform.entity_tol(np.vstack([apts, s1, s2]))
    gap_o = float(np.linalg.norm(s1[0] - s2[0]))
    if gap_o > tol:
        raise ValueError(
            "quadrant_ogrid: seam1 and seam2 must start at the same center point O "
            "(gap %.3g > %.3g); build both from one center array" % (gap_o, tol))
    for which, sm_pts, corner, name in (("seam1", s1, apts[0], "arc[0] (A1)"),
                                        ("seam2", s2, apts[-1], "arc[-1] (A2)")):
        gap = float(np.linalg.norm(sm_pts[-1] - corner))
        if gap > tol:
            raise ValueError(
                "quadrant_ogrid: %s must end at %s (gap %.3g > %.3g) -- the arc runs "
                "A1 -> A2 and seam1 is the A1 radius; reverse the arc or swap the "
                "seams" % (which, name, gap, tol))

    core: PointArray = quadrant_core(arc, seam1, seam2,
                                     center_scale=center_scale).reshape(-1, 3)

    row = n + 1

    def cid(i: int, j: int) -> int:
        return i * row + j

    cquads: IntArray = _grid_quads(n, n)

    # core perimeter, M1 -> K -> M2: 2n+1 points index-paired with arc[0..2n].
    peri_ids: IntArray = np.array([cid(n, j) for j in range(row)]
                                  + [cid(i, n) for i in range(n - 1, -1, -1)],
                                  dtype=np.int64)
    peripts: PointArray = core[peri_ids, :]
    P = 2 * n + 1

    # -- ring band: straight-chord blends out to the wall, with the two ends snapped
    # onto the exact seam samples (as half_ogrid does) so neighbours weld bit-exactly.
    lid: list[IntArray] = [peri_ids]
    layers: list[PointArray] = [core]
    ring_pts: list[PointArray] = []
    nprev = core.shape[0]
    for r in range(Nr):
        tau = float(radial[r + 1])
        rp: PointArray = (1.0 - tau) * peripts + tau * apts
        rp[0, :] = s1[n + 1 + r]
        rp[-1, :] = s2[n + 1 + r]
        layers.append(rp)
        ring_pts.append(rp)
        lid.append(nprev + np.arange(P, dtype=np.int64))
        nprev += P
    points: PointArray = np.vstack(layers)

    k: IntArray = np.arange(2 * n, dtype=np.int64)
    ring_quads = [np.stack([b[k], b[k + 1], a[k + 1], a[k]], axis=1)   # CCW, b outer
                  for a, b in zip(lid[:-1], lid[1:])]
    quads: IntArray = np.vstack([cquads, *ring_quads])

    # -- boundary rows.  Ring quad [b[k], b[k+1], a[k+1], a[k]] puts side 1 on the
    # outer layer and side 3 on the inner (ogrid's winding), so the wall is side 1 of
    # the outermost band; side 4 of band quad 0 and side 2 of band quad 2n-1 are the
    # seams' ring stations, and the core's j == 0 row / i == 0 column their fans.
    q0 = n * n
    wall_q0 = q0 + (Nr - 1) * (2 * n)
    seam_sides = (
        ("seam1", seam1, [(i * n, 1) for i in range(n)]
         + [(q0 + r * (2 * n), 4) for r in range(Nr)]),
        ("seam2", seam2, [(j, 4) for j in range(n)]
         + [(q0 + r * (2 * n) + (2 * n - 1), 2) for r in range(Nr)]),
    )
    blocks = [_curve_rows([(wall_q0 + m, 1) for m in range(2 * n)], arc, wall_tag)]
    blocks += [_curve_rows(rows, sm, st.get(which, ""))
               for which, sm, rows in seam_sides]
    qm = tag_edges(QuadMesh.from_corners(points, quads),
                   np.concatenate([r for r, _ in blocks], axis=0),
                   np.concatenate([n for _, n in blocks]))

    # -- order N.  Overlay every O-ring so the wall's curvature blends inward, and both
    # seams with their *own* nodes so a bowed radius is meshed exactly rather than
    # straight-subdivided between its samples.
    order = arc.order
    overlays: list[Overlay] = []
    if order > 1:
        overlays = _ring_overlays(ring_pts, arc, radial, peripts,
                                  q0, 2 * n, 1)
        rs: IntArray = np.arange(Nr, dtype=np.int64)
        core_fan: IntArray = np.arange(n, dtype=np.int64)
        s2l1: IntArray = np.argsort(_chain_intervals(seam1, "quadrant_ogrid seam1"))
        s2l2: IntArray = np.argsort(_chain_intervals(seam2, "quadrant_ogrid seam2"))
        overlays += [
            (core_fan * n, 1, _sub_chain(seam1, core_fan, s2l1)),
            (q0 + rs * (2 * n), 4, _sub_chain(seam1, n + rs, s2l1)),
            (core_fan, 4, _sub_chain(seam2, core_fan, s2l2)),
            (q0 + rs * (2 * n) + (2 * n - 1), 2, _sub_chain(seam2, n + rs, s2l2)),
        ]
    qm = _elevate(qm, order, overlays)
    return _apply_smoothing(qm, smoothing_method)


def quadrant_seam_fractions(n_side: int, radial: int | FloatArray,
                            quadrant_scale: float = 0.7) -> FloatArray:
    """The normalized ``O -> A`` positions of the ``n_side+1 + Nradial`` seam points
    that :func:`quadrant_ogrid <nekmeshpy.quadmesh.shape.quadrant_ogrid>` requires,
    ascending: the ``n_side+1`` core fan, then the ``Nradial`` ring stations. The seam's
    shared corner ``M`` sits at ``quadrant_scale * R`` along ``O -> A`` directly -- an
    independent knob from :func:`quadrant_ogrid
    <nekmeshpy.quadmesh.shape.quadrant_ogrid>`'s own ``center_scale``, which places its
    hub corner ``K`` at ``center_scale * R`` along the arc's *bisector* instead. Unless
    ``quadrant_scale`` is chosen to land ``M`` exactly on the chord between neighbouring
    quadrants' ``K`` corners (``quadrant_scale == center_scale * cos(45deg)``), the
    merged core's boundary bows into an octagon rather than sitting flush as a
    square."""
    ns = int(n_side)
    if ns < 1:
        raise ValueError("quadrant_seam_fractions needs n_side >= 1, got %d" % ns)
    if not 0.0 < quadrant_scale < 1.0:
        raise ValueError("quadrant_seam_fractions needs quadrant_scale in (0, 1)")
    rad = validate_layers(radial, "quadrant_seam_fractions radial")
    s_m = quadrant_scale
    fr: FloatArray = np.concatenate([np.linspace(0.0, s_m, ns + 1),
                                     s_m + rad[1:] * (1.0 - s_m)])
    return fr


def spine_fractions(n_theta: int, radial: int | FloatArray,
                    quadrant_scale: float = 0.7) -> FloatArray:
    """The normalized ``A1 -> A2`` positions of the ``2*n_theta+1 + 2*Nradial`` spine
    points that :func:`half_ogrid <nekmeshpy.quadmesh.shape.half_ogrid>` and
    :func:`spined_ogrid <nekmeshpy.quadmesh.shape.spined_ogrid>` require, in ascending
    order: ``Nradial`` north caps, the ``2*n_theta+1`` center fan, then ``Nradial``
    south caps (see :func:`half_ogrid <nekmeshpy.quadmesh.shape.half_ogrid>` for what
    each region is). ``quadrant_scale`` plays the same role here as it does in
    :func:`quadrant_seam_fractions <nekmeshpy.quadmesh.shape.quadrant_seam_fractions>`:
    it is the fraction of each half (``O -> A1`` and ``O -> A2``) given to the center
    fan before the ``Nradial`` O-ring caps take over."""
    nt = int(n_theta)
    if nt < 1:
        raise ValueError("spine_fractions needs n_theta >= 1, got %d" % nt)
    if not 0.0 < quadrant_scale < 1.0:
        raise ValueError("spine_fractions needs quadrant_scale in (0, 1)")
    rad = validate_layers(radial, "spine_fractions radial")
    s_n, s_s = (1.0 - quadrant_scale) / 2, (1.0 + quadrant_scale) / 2
    fr: FloatArray = np.concatenate([((1.0 - rad[1:]) * s_n)[::-1],
                                     np.linspace(s_n, s_s, 2 * nt + 1),
                                     s_s + rad[1:] * (1.0 - s_s)])
    return fr


def spined_ogrid(boundary: LineMesh, radial: int | FloatArray, *,
                 spine: LineMesh | None = None, center_scale: float = 0.7,
                 quadrant_scale: float = 0.7, wall_tag: str = "",
                 smoothing_method: SmoothingMethod | None = None) -> QuadMesh:
    """Full-disk O-grid over a closed ``boundary`` split along a spine diameter into two
    :func:`half_ogrid <nekmeshpy.quadmesh.shape.half_ogrid>` halves (each in turn built
    from two :func:`quadrant_ogrid <nekmeshpy.quadmesh.shape.quadrant_ogrid>` quarters)
    welded along the spine -- the clean way to O-grid a disk that has a natural
    ``A1..A2`` seam (a saddle-split vessel or pipe cross-section), so the caller need not
    hand-roll the arc split and merge. ``center_scale``/``quadrant_scale`` are as in
    :func:`quadrant_ogrid <nekmeshpy.quadmesh.shape.quadrant_ogrid>`/
    :func:`quadrant_seam_fractions <nekmeshpy.quadmesh.shape.quadrant_seam_fractions>`."""
    bpts = _check_boundary(boundary, "spined_ogrid boundary", 8)
    M = bpts.shape[0]
    if M % 8 != 0:
        raise ValueError(
            "spined_ogrid boundary must have 8*Ntheta points (a closed loop split "
            "into two 4*Ntheta+1 arcs); got %d" % M)
    nh = M // 2
    Nt = nh // 4
    radial = validate_layers(radial, "spined_ogrid radial")

    # The spine is meshed exactly at the points given -- nothing is resampled here.
    # A caller-supplied spine must already carry the [north caps, center fan, south
    # caps] sampling half_ogrid consumes; only the default straight chord is a shape
    # this factory owns, so only that one may be placed here.
    fr = spine_fractions(Nt, radial, quadrant_scale)
    if spine is None:
        spine = line(bpts[0, :], bpts[nh, :], fr, order=boundary.order)
    elif spine.points.shape[0] != fr.shape[0]:
        raise ValueError(
            "spined_ogrid spine must have exactly 2*Ntheta+1 + 2*Nradial = %d points "
            "ascending A1 -> A2 (got %d); it is meshed exactly at the points given, "
            "never resampled -- evaluate your spine curve at "
            "spine_fractions(%d, radial, quadrant_scale)"
            % (fr.shape[0], spine.points.shape[0], Nt))
    # the second half traverses A2 -> A1; ``reverse`` relabels rather than re-placing,
    # so both halves see bit-identical seam coordinates (and, at order > 1, carries the
    # spine's own interior nodes with it instead of re-subdividing them straight).
    spine2 = line_reverse(spine)

    # split the loop (and its per-segment tags) into the two half arcs: arc1 runs
    # A1 -> A2 over segments [0, nh), arc2 runs A2 -> A1 over segments [nh, M).
    arc1 = _slice_chain(boundary, 0, nh)
    arc2 = _slice_chain(boundary, nh, M)
    h1 = half_ogrid(arc1, spine, radial, center_scale=center_scale,
                    quadrant_scale=quadrant_scale, wall_tag=wall_tag,
                    smoothing_method=smoothing_method)
    h2 = half_ogrid(arc2, spine2, radial, center_scale=center_scale,
                    quadrant_scale=quadrant_scale, wall_tag=wall_tag,
                    smoothing_method=smoothing_method)
    return merge([h1, h2])


# the six box faces: outward normal n with right-handed tangents (u x v = n),
# each mapped to its {x,y,z}_{min,max} side key.
_BOX_FACES = [
    ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0), "x_max"),
    ((-1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, 1.0, 0.0), "x_min"),
    ((0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (1.0, 0.0, 0.0), "y_max"),
    ((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0), "y_min"),
    ((0.0, 0.0, 1.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), "z_max"),
    ((0.0, 0.0, -1.0), (0.0, 1.0, 0.0), (1.0, 0.0, 0.0), "z_min"),
]

# the four upright side faces of a half box: outward normal n, horizontal tangent u,
# vertical tangent +z (u x v = n), so restricting the vertical coordinate to [0, 1]
# keeps the patch in z >= 0 and puts its lower edge on the ground plane.
_HALF_BOX_SIDES = [
    ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), "x_max"),
    ((-1.0, 0.0, 0.0), (0.0, -1.0, 0.0), "x_min"),
    ((0.0, 1.0, 0.0), (-1.0, 0.0, 0.0), "y_max"),
    ((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), "y_min"),
]
_VZ: Vec3 = np.array([0.0, 0.0, 1.0])


def _axis_params(half_sizes: float | Sequence[float] | FloatArray,
                 n: int | Sequence[int] | IntArray,
                 ) -> tuple[FloatArray, tuple[int, int, int]]:
    """Normalize a box-like factory's ``half_sizes`` / ``n`` to a ``(3,)`` float array
    of half extents and a ``(nx, ny, nz)`` cell-count triple (each input a scalar or
    three values)."""
    hs: FloatArray = np.asarray(half_sizes, dtype=float).ravel()
    if hs.size == 1:
        hs = np.full(3, float(hs[0]))
    elif hs.size != 3:
        raise ValueError("half_sizes must be a scalar or 3 values (sx, sy, sz)")
    na: IntArray = np.asarray(n, dtype=np.int64).ravel()
    if na.size == 1:
        return hs, (int(na[0]), int(na[0]), int(na[0]))
    if na.size == 3:
        return hs, (int(na[0]), int(na[1]), int(na[2]))
    raise ValueError("n must be a scalar or 3 counts (nx, ny, nz)")


def box(half_sizes: float | Sequence[float] | FloatArray,
        n: int | Sequence[int] | IntArray, *,
        patch_tags: Mapping[str, str] | None = None,
        order: int = 1) -> QuadMesh:
    """Closed box surface centred at the origin: six quad patches welded with
    :func:`merge <nekmeshpy.quadmesh.assemble.merge>`. ``half_sizes`` is a scalar (cube)
    or ``(sx, sy, sz)``; ``n`` is a scalar or ``(nx, ny, nz)`` cells per axis."""
    hs, n_axis = _axis_params(half_sizes, n)
    ft = patch_tags or {}
    patches: list[QuadMesh] = []
    for nrm, u, v, key in _BOX_FACES:
        nv: Vec3 = np.asarray(nrm, dtype=float)
        uv: Vec3 = np.asarray(u, dtype=float)
        vv: Vec3 = np.asarray(v, dtype=float)
        au = np.linspace(-1.0, 1.0, n_axis[int(np.argmax(np.abs(uv)))] + 1)
        av = np.linspace(-1.0, 1.0, n_axis[int(np.argmax(np.abs(vv)))] + 1)
        A: FloatArray
        B: FloatArray
        A, B = np.meshgrid(au, av, indexing="ij")
        face = hs * (nv + A[..., None] * uv + B[..., None] * vv)
        patches.append(from_grid(face, element_tag=ft.get(key, ""),
                                          order=order))
    return merge(patches)


def sphere(radius: float, n: int | Sequence[int] | IntArray, *,
           element_tag: str = NO_TAG, order: int = 1) -> QuadMesh:
    """Closed cubed-sphere surface of ``radius`` about the origin: a unit :func:`box
    <nekmeshpy.quadmesh.shape.box>` projected radially onto the sphere (same
    connectivity, so it pairs by index with a same-``n`` box for :func:`HexMesh.annulus
    <nekmeshpy.hexmesh.lift.annulus>`).

    Untagged unless ``element_tag`` is given, like every other factory: a shape does not
    name itself, because the name belongs to the model the caller is assembling."""
    from ..linemesh import LineMesh
    cube = box(1.0, n, order=order)

    def project(a: PointArray) -> PointArray:
        """Push every node of ``a`` (last axis = xyz) radially onto the sphere."""
        return radius * a / np.linalg.norm(a, axis=-1, keepdims=True)

    etags = ElementTags.uniform(cube.n_quads, element_tag)
    # the cube's B-rep is reused verbatim (same topology, same edge numbering); only
    # the node coordinates move, so there is nothing to re-derive or reconcile.
    lines = LineMesh(project(cube.points), cube.line_mesh.lines,
                     interior=project(cube.line_mesh.interior) if order > 1 else None)
    return QuadMesh(lines, cube.quad, cube.orient,
                    project(cube.interior) if order > 1 else None,
                    element_tags=etags)


def _tag_rim(qm: QuadMesh, rim_tag: str) -> QuadMesh:
    """Return ``qm`` with every free (single-quad) edge tagged ``rim_tag``."""
    if not rim_tag:
        return qm
    rows = boundary_edges(qm)
    return tag_edges(qm, rows, [rim_tag] * rows.shape[0])


def half_box(half_sizes: float | Sequence[float] | FloatArray,
             n: int | Sequence[int] | IntArray, *,
             n_vertical: int | None = None,
             patch_tags: Mapping[str, str] | None = None,
             rim_tag: str = "",
             order: int = 1) -> QuadMesh:
    """The upper half of a :func:`box <nekmeshpy.quadmesh.shape.box>`: the five patches
    of the box surface that bound ``[-sx, sx] x [-sy, sy] x [0, sz]``, welded with
    :func:`merge <nekmeshpy.quadmesh.assemble.merge>` and left **open at the ``z = 0``
    rim** (the ``z_min`` patch is dropped)."""
    hs, n_axis = _axis_params(half_sizes, n)
    nv = n_axis[2] if n_vertical is None else int(n_vertical)
    if nv < 1:
        raise ValueError("half_box needs n_vertical >= 1, got %d" % nv)
    ft = patch_tags or {}
    patches: list[QuadMesh] = []
    b_side = np.linspace(0.0, 1.0, nv + 1)                 # upper half only
    for nrm, u, key in _HALF_BOX_SIDES:
        nvv: Vec3 = np.asarray(nrm, dtype=float)
        uv: Vec3 = np.asarray(u, dtype=float)
        au = np.linspace(-1.0, 1.0, n_axis[int(np.argmax(np.abs(uv)))] + 1)
        A: FloatArray
        B: FloatArray
        A, B = np.meshgrid(au, b_side, indexing="ij")
        face = hs * (nvv + A[..., None] * uv + B[..., None] * _VZ)
        patches.append(from_grid(face, element_tag=ft.get(key, ""),
                                          order=order))
    # the flat top patch at z = sz, spanning x and y in full
    ax = np.linspace(-1.0, 1.0, n_axis[0] + 1)
    ay = np.linspace(-1.0, 1.0, n_axis[1] + 1)
    AX: FloatArray
    AY: FloatArray
    AX, AY = np.meshgrid(ax, ay, indexing="ij")
    top = hs * (_VZ + AX[..., None] * np.array([1.0, 0.0, 0.0])
                + AY[..., None] * np.array([0.0, 1.0, 0.0]))
    patches.append(from_grid(top, element_tag=ft.get("z_max", ""),
                                      order=order))
    return _tag_rim(merge(patches), rim_tag)


def hemisphere(radius: float, n: int | Sequence[int] | IntArray, *,
               n_vertical: int | None = None,
               element_tag: str = NO_TAG,
               rim_tag: str = NO_TAG,
               order: int = 1) -> QuadMesh:
    """Cubed-**hemisphere** surface of ``radius`` sitting on the ground plane ``z = 0``:
    a unit :func:`half_box <nekmeshpy.quadmesh.shape.half_box>` projected radially onto
    the sphere.  Untagged unless ``element_tag`` is given."""
    from ..linemesh import LineMesh
    cube = half_box(1.0, n, n_vertical=n_vertical, order=order)

    def project(a: PointArray) -> PointArray:
        """Push every node of ``a`` (last axis = xyz) radially onto the sphere."""
        return radius * a / np.linalg.norm(a, axis=-1, keepdims=True)

    etags = ElementTags.uniform(cube.n_quads, element_tag)
    lines = LineMesh(project(cube.points), cube.line_mesh.lines,
                     interior=project(cube.line_mesh.interior) if order > 1 else None)
    qm = QuadMesh(lines, cube.quad, cube.orient,
                  project(cube.interior) if order > 1 else None,
                  element_tags=etags)
    return _tag_rim(qm, rim_tag)

def _patch(fn: Callable[[FloatArray, FloatArray], FloatArray], surface: SurfaceMap,
           n: int, order: int, tag: str) -> QuadMesh:
    """One ``n``-by-``n`` patch of a surface, from a parameter-space Coons map --
    evaluated on the surface at every node, corners and private interiors alike."""
    fr = np.linspace(0.0, 1.0, n + 1)
    return loft_fn(
        lambda y: line_loft_fn(
            lambda x: surface(fn(x, np.full(np.shape(x), y))), fr, order=order),
        fr, order=order, element_tags=tag or None)


def tri_patch(surface: SurfaceMap, ab: SurfaceCurve, bc: SurfaceCurve,
              ca: SurfaceCurve, *, order: int = 1, tip_bias: float = 1.0 / 3.0,
              mids: Sequence[FloatArray] | None = None,
              element_tag: str = "") -> QuadMesh:
    """The curved triangle bounded by three surface curves, meshed as the **three
    quadrilateral patches** about an interior tip -- the all-quad split of a triangle,
    and the shape :func:`HexMesh.tetra <nekmeshpy.hexmesh.shape.tetra>` wants for the
    curved side of a curvilinear tetrahedron."""
    ns = {c.fr.size for c in (ab, bc, ca)}
    if len(ns) != 1:
        raise ValueError("tri_patch: the three curves must share a node count, got %s"
                         % sorted(ns))
    k = ns.pop()
    if k < 3 or k % 2 == 0:
        raise ValueError(
            "tri_patch: each curve needs an odd node count 2n+1 (at least 3) so it can "
            "be split at its own middle node; got %d" % k)
    n = (k - 1) // 2
    u_ab, u_bc, u_ca = (
        [surfaces.node(c, n) for c in (ab, bc, ca)] if mids is None else list(mids))
    tip = tri_patch_tip(u_ab, u_bc, u_ca, tip_bias=tip_bias)
    seg, spk = surfaces.segment, surfaces.spoke
    return merge([
        _patch(coons_grid_fn(seg(ab, 0, n), spk(u_ca, tip),
                             seg(ca, 2 * n, n), spk(u_ab, tip)),
               surface, n, order, element_tag),
        _patch(coons_grid_fn(seg(ab, 2 * n, n), spk(u_bc, tip),
                             seg(bc, 0, n), spk(u_ab, tip)),
               surface, n, order, element_tag),
        _patch(coons_grid_fn(seg(ca, 0, n), spk(u_bc, tip),
                             seg(bc, 2 * n, n), spk(u_ca, tip)),
               surface, n, order, element_tag)])


def tri_patch_tip(u_ab: FloatArray, u_bc: FloatArray, u_ca: FloatArray, *,
                  tip_bias: float = 1.0 / 3.0) -> FloatArray:
    """Where :func:`tri_patch` puts its tip, in surface parameters, given the three
    curves' middle nodes."""
    return tip_bias * u_ab + (1.0 - tip_bias) * 0.5 * (u_bc + u_ca)


__all__ = [
    "box",
    "half_box",
    "half_ogrid",
    "hemisphere",
    "ogrid",
    "quadrant_core",
    "quadrant_ogrid",
    "quadrant_seam_fractions",
    "rectangle",
    "sphere",
    "spine_fractions",
    "spined_ogrid",
    "structured",
    "tri_patch",
    "tri_patch_tip",
]
