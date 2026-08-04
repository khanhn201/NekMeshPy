"""Open-region :class:`~nekmeshpy.QuadMesh` section factories: fills that take a
boundary / edges and mesh the bounded region (``structured`` / ``rectangle`` /
``ogrid`` / ``half_ogrid`` / ``quadrant_ogrid`` / ``spined_ogrid``).

These are plain free functions returning a ``QuadMesh``; ``quadmesh/__init__.py``
binds each entry of ``FACTORIES`` onto the class, so callers use ``QuadMesh.ogrid(...)``
etc.  They build on the core constructors / ``loft`` / ``_order_bnd``, referenced by a
lazy in-function import to avoid the import cycle with ``quadmesh.py`` (which the package
imports to assemble the class); the shared ``_apply_smoothing`` / ``_check_boundary``
live in ``_helpers.py``.  Internal toolkit code calls these free functions directly.

Each fills a bounded region with quads.  An optional ``smoothing_method``
("conduction" / "winslow" / "bilinear"; None = raw fill) repositions the interior
points.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

import numpy as np

from .._typing import FloatArray, IntArray, Point, PointArray, SmoothingMethod
from ..linemesh import LineMesh
from ..linemesh._assemble import loft as line_loft
from ..linemesh._morph import reverse as line_reverse
from ..linemesh._open import line
from ..model import conform
from ..model.fields import gll_nodes, validate_layers
from ..model.interp import coons_grid
from ._assemble import merge
from ._helpers import Overlay, _apply_smoothing, _check_boundary, _elevate, entities_from_blocks

if TYPE_CHECKING:
    from collections.abc import Callable

    from .quadmesh import QuadMesh

#: The four sides of a :func:`structured` patch, in the CCW loop order its ``edges``
#: are consumed in.  Both the ``edges`` mapping and ``side_tags`` are keyed by these.
_SIDES = ("bottom", "right", "top", "left")


def _ordered_sides(edges: Sequence[LineMesh] | Mapping[str, LineMesh],
                   ) -> list[LineMesh]:
    """``edges`` as the positional ``[bottom, right, top, left]`` list :func:`structured`
    works in, accepting either spelling.

    A bare 4-sequence lets the *order* silently encode which edge is which, so
    swapping two gives a valid-looking twisted patch instead of an error; the mapping
    spelling names them.  Both are accepted (the sequence form is what every existing
    caller passes), and both end up here."""
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
    """Structured quad grid over the rectangle with four CCW corners
    ``corners = [c0, c1, c2, c3]``: ``nx`` cells along ``c0->c1`` (bottom/top),
    ``ny`` along ``c1->c2`` (left/right).  ``x_frac`` / ``y_frac`` are optional
    node fractions in ``[0,1]`` (length ``nx+1`` / ``ny+1``) for grading, else
    uniform.  ``side_tags`` (keyed ``bottom`` / ``right`` / ``top`` / ``left``)
    names the outer sides; an absent side stays untagged.

    ``order`` (default 1 = linear) sets the polynomial order: the four edges are
    built at ``order`` (straight sides) and :func:`structured` inherits it."""
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
    """``(L,)`` map from the edge's line index to the point interval ``i`` it spans.

    ``structured`` reads an edge's nodes straight off ``edge.points`` in index order,
    and ``half_ogrid`` reads its spine's samples the same way, so line ``k`` must join
    two consecutive points; this returns the lower index of each line's interval
    (``lines[k] == (i, i+1)`` or ``(i+1, i)``) so an overlay can find the quad that
    line ``k`` bounds.  Raises if the edge is not a simple consecutive chain (which
    both factories already assume for their point order)."""
    lines: IntArray = np.asarray(edge.lines, dtype=np.int64).reshape(-1, 2)
    lo: IntArray = np.minimum(lines[:, 0], lines[:, 1])
    hi: IntArray = np.maximum(lines[:, 0], lines[:, 1])
    n = edge.points.shape[0] - 1
    if (lines.shape[0] != n or not np.array_equal(hi, lo + 1)
            or not np.array_equal(np.sort(lo), np.arange(n, dtype=np.int64))):
        raise ValueError(
            "%s must be a simple consecutive chain (line k joining points k and k+1) "
            "-- its point order is what sets the node distribution; build it with "
            "LineMesh.loft/line/arc/loft_curve (or merge chains end to end) rather than "
            "re-indexing its lines" % name)
    return lo


def _grid_quads(ni: int, nj: int) -> IntArray:
    """The ``(ni*nj, 4)`` CCW quad table of an ``(ni+1) x (nj+1)`` structured point
    grid numbered ``id(i, j) = i*(nj+1) + j``, in i-major / j-minor element order
    (quad ``(i, j)`` is row ``i*nj + j``).

    The corner list ``[(i,j), (i+1,j), (i+1,j+1), (i,j+1)]`` is what puts the grid's
    ``j = 0`` row on local side 1, ``i = ni-1`` on side 2, ``j = nj-1`` on side 3 and
    ``i = 0`` on side 4 -- every caller's boundary rows are read off that."""
    row = nj + 1
    i: IntArray = np.repeat(np.arange(ni, dtype=np.int64), nj)
    j: IntArray = np.tile(np.arange(nj, dtype=np.int64), ni)
    return np.stack([i * row + j, (i + 1) * row + j,
                     (i + 1) * row + j + 1, i * row + j + 1], axis=1)


def _refined_params(n: int, order: int) -> FloatArray:
    """The ``n*order+1`` transfinite parameters of an ``n``-interval edge at order N.

    Interval ``i``'s node ``a`` sits at ``(i + g[a]) / n`` for the GLL nodes ``g`` on
    ``[0,1]``.  At ``order == 1`` that is ``g = [0, 1]`` and the result is exactly
    ``linspace(0, 1, n+1)`` -- the corner grid, unchanged."""
    if order == 1:
        return np.linspace(0.0, 1.0, n + 1)
    g = gll_nodes(order)
    inner: FloatArray = (np.arange(n, dtype=float)[:, None] + g[None, :order]).ravel()
    return np.concatenate([inner / n, [1.0]])


def _refined_chain(edge: LineMesh, name: str) -> PointArray:
    """An edge's own nodes in point-index order: ``(n*order+1, 3)``.

    These are the exact samples of the edge curve at :func:`_refined_params` -- read
    off the edge's B-rep rather than interpolated -- so an analytic edge
    (:meth:`LineMesh.arc <nekmeshpy.linemesh.LineMesh.arc>`) enters the transfinite
    blend on its true geometry.  Each line's private ``interior`` is reversed where the
    line runs against the edge's point order."""
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
    """The O-ring curve at radial fraction ``tau``, as a ``LineMesh`` to overlay.

    The ring is the wall curve blended toward the straight block perimeter -- the same
    lerp the ring's corners get, applied to the private ``interior`` nodes as well, so
    an intermediate ring inherits its share of the wall's curvature instead of being a
    straight chord between blended corners.  This is what
    :func:`annulus` gets from :meth:`LineMesh.blend
    <nekmeshpy.linemesh.LineMesh.blend>`; the rings here cannot use ``blend`` directly
    because ``half_ogrid`` snaps its two end points onto exact spine samples, so the
    curve is re-anchored on ``pos`` (the ring's true corner positions) with a linear
    correction that vanishes wherever no snapping happened.  At ``tau == 1`` it
    reproduces ``wall`` exactly."""
    lines: IntArray = wall.lines
    order = wall.order
    if order == 1:
        return LineMesh(pos, lines, order=1)
    inner: PointArray = (1.0 - tau) * peri_inner + tau * wall.interior
    drift: PointArray = pos - ((1.0 - tau) * peri_pos + tau * wall.points)
    g = gll_nodes(order)[1:order]
    d0, d1 = drift[lines[:, 0]], drift[lines[:, 1]]
    inner = (inner + (1.0 - g)[None, :, None] * d0[:, None, :]
             + g[None, :, None] * d1[:, None, :])
    return LineMesh(pos, lines, order=order, interior=inner)


def _ring_overlays(ring_pts: Sequence[PointArray], wall: LineMesh,
                   radial: FloatArray, peri_pos: PointArray, q0: int,
                   width: int, outer_side: int) -> list[Overlay]:
    """Both incident copies of every O-ring curve, as ``_elevate`` overlays.

    Shared by :func:`ogrid` / :func:`half_ogrid` / :func:`quadrant_ogrid`, whose ring
    bands differ only in where they start (``q0``, the first band quad), how wide a
    band is (``width`` quads) and which local side faces outward (``outer_side``, 1 for
    the ``[b, b_next, a_next, a]`` winding of ``ogrid`` and ``quadrant_ogrid``, 3 for
    ``half_ogrid``'s ``[a, a_next, b_next, b]``).

    **Every** ring is overlaid, not just the wall, so the wall's curvature is blended
    inward instead of dying one layer in -- ring ``r`` is band ``r``'s outer side *and*
    band ``r+1``'s inner side, and both incident copies must be stamped or
    :func:`~nekmeshpy.model.conform.scatter_edge_nodes` rightly reports a
    non-conforming shared edge.  Ring 0 is the band's inner boundary (the core
    perimeter), which already matches its straight guess; the outermost ring
    (``tau == 1``) reproduces ``wall`` exactly."""
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
               override: str) -> tuple[list[list[int]], list[str]]:
    """The tagged ``(boundaries, boundary_tags)`` rows naming one side of a region.

    ``rows[m]`` is the ``(quad id, quad side)`` that carries element ``m`` of
    ``curve``, so the side is named element by element from the curve's own per-line
    ``element_tags``: a boundary assembled from several differently tagged arcs splits
    into distinct named sides instead of collapsing into one.  A non-empty scalar
    ``override`` (a factory's ``wall_tag`` / ``side_tags[...]``) names the whole side
    instead, and an element left untagged either way emits no row at all."""
    seg = curve._seg_tags()
    bnd: list[list[int]] = []
    names: list[str] = []
    for m, (qid, side) in enumerate(rows):
        nm = override if override else (seg[m] if seg is not None else "")
        if nm:
            bnd.append([qid, side])
            names.append(nm)
    return bnd, names


def _sub_chain(chain: LineMesh, segs: IntArray, seg2line: IntArray) -> LineMesh:
    """The sub-``LineMesh`` of ``chain`` holding point intervals ``segs``, in that order.

    ``seg2line`` maps a point interval ``i`` (the span ``i -> i+1``) to the line index
    that carries it, so the result's line ``k`` is the true geometry -- endpoints *and*
    private ``interior`` -- of interval ``segs[k]``.  Handed to ``_elevate`` as an
    :data:`~nekmeshpy.quadmesh._helpers.Overlay` payload, which matches each line
    against its quad's side by endpoint and reverses it where the two disagree, so the
    intervals need not be listed in ascending order."""
    idx: IntArray = seg2line[np.asarray(segs, dtype=np.int64)]
    return LineMesh(chain.points, chain.lines[idx], order=chain.order,
                    interior=chain.interior[idx])


def structured(edges: Sequence[LineMesh] | Mapping[str, LineMesh], *,
               side_tags: Mapping[str, str] | None = None,
               smoothing_method: SmoothingMethod | None = None) -> QuadMesh:
    """Transfinite (Coons-patch) quad grid over the surface bounded by four
    open edge lines ``edges = [bottom, right, top, left]`` in CCW loop order.
    The lines must share corners (form a closed loop).

    ``edges`` may equally be a **mapping** keyed ``bottom`` / ``right`` / ``top`` /
    ``left`` -- the same four names ``side_tags`` uses.  Prefer it: in the sequence
    spelling the position alone says which edge is which, so transposing two produces
    a plausible-looking twisted patch rather than an error, whereas a mapping with a
    missing or misspelt key raises.

    Resolution and node distribution come directly from the edge lines' own
    points (no resampling): ``bottom``/``top`` must share a point count
    (``nx+1``) and ``left``/``right`` another (``ny+1``), giving ``nx`` x ``ny``
    cells.

    Each side is named from its own edge line's uniform ``element_tags``;
    ``side_tags`` (keyed by ``"bottom"`` / ``"right"`` / ``"top"`` /
    ``"left"``) overrides that -- a non-empty entry replaces the side's tag, a
    present-but-empty entry suppresses the side.  It is spelt ``side_tags``, not
    ``boundary_tags``, because these are *named sides*: ``boundary_tags`` everywhere
    else in the toolkit is the dense ``StrArray`` running parallel with a mesh's
    ``boundaries (Nbc,2)`` rows, a different shape entirely.

    The order comes from the edges (all four must agree).  At ``order > 1`` each
    edge's own private high-order ``interior`` is **stamped onto the matching side**
    of the boundary quads, exactly as ``ogrid`` / ``half_ogrid`` do, so an analytic
    edge (:meth:`LineMesh.arc <nekmeshpy.linemesh.LineMesh.arc>`,
    :meth:`circle <nekmeshpy.linemesh.LineMesh.circle>`) is meshed on its true curve
    rather than straight-subdivided between its points; the transfinite interior
    stays a straight order-N fill.
    """
    from .quadmesh import QuadMesh
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
    bnd: list[list[int]] = []
    names: list[str] = []
    # each side is named by its edge's uniform element tag; a non-empty
    # side_tags[side] overrides, a present-but-empty entry suppresses it.
    for side, rows in side_rows.items():
        if side in bt:
            nm = bt[side]
        else:
            et = side_edges[side].element_group_tags
            nm = et[0] if len(et) == 1 else ""
        if not nm:                       # NO_BOUNDARY / "" / untagged -> no row
            continue
        for q, s in rows:
            bnd.append([q, s])
            names.append(nm)
    qm = QuadMesh.from_corners(points, quads, *QuadMesh._order_bnd(bnd, names))
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
        qm = QuadMesh(lm, elem_edges, flip, interior,
                      qm.boundaries, qm.boundary_tags,
                      element_tags=qm.element_tags, order=order)
    # smooth last: a repositioning smoother rejects order > 1 (high-order smoothing
    # is not implemented).
    return _apply_smoothing(qm, smoothing_method)


def ogrid(boundary: LineMesh, n_side: int, radial: int | FloatArray, *,
          center_scale: float = 0.5,
          wall_tag: str = "", smoothing_method: SmoothingMethod | None = None) -> QuadMesh:
    """O-grid filling the closed ``boundary``: a central ``n_side x n_side``
    block at the loop centroid, surrounded by O-ring layers blending its
    perimeter out to the boundary.  ``center_scale`` sizes the block (fraction
    of the mean radius).

    Built in 3-D with no projection, so a curvy / non-planar boundary keeps its
    true shape; the block-and-ring build is only an initial guess, relaxed by
    ``smoothing_method="conduction"``.

    ``radial`` is either an ``int`` count of uniform rings or the O-ring layer
    positions with the initial position explicit: strictly increasing in
    ``[0, 1]`` (``radial[0]`` = block perimeter, last = ``1`` = wall;
    :func:`validate_layers <nekmeshpy.model.fields.validate_layers>`), giving
    ``radial.size - 1`` rings.

    The outer ring (wall) is named from ``boundary``'s per-line ``element_tags``;
    a non-empty scalar ``wall_tag`` overrides that for the whole wall."""
    from .quadmesh import QuadMesh
    if n_side < 1:
        raise ValueError("ogrid needs n_side >= 1")
    if not 0.0 < center_scale < 1.0:
        raise ValueError("ogrid needs center_scale in (0, 1)")
    radial = validate_layers(radial, "ogrid radial")
    n_radial = radial.size - 1
    bpts = _check_boundary(boundary, "ogrid boundary", 3)
    # wall ring = the boundary loop itself, meshed exactly: it must already carry
    # P = 4*n_side points (the caller sizes it, e.g. circle(R, 4*n_side)).  Block
    # corners are 4 of those pulled toward the centroid and bilinearly filled;
    # rings are straight-chord blends (an initial guess for smoothing).
    P = 4 * n_side
    if bpts.shape[0] != P:
        raise ValueError(
            "ogrid boundary must have exactly 4*n_side = %d points to be meshed "
            "exactly (got %d); size the loop to match, e.g. circle(R, %d)"
            % (P, bpts.shape[0], P))
    outer_pos: PointArray = bpts                                # (P,3) true wall
    centroid = outer_pos.mean(axis=0)
    rad = float(np.mean(np.linalg.norm(outer_pos - centroid, axis=1)))
    if rad <= 0.0:
        raise ValueError("ogrid: boundary is degenerate (all points coincide)")
    row = n_side + 1

    def cid(i: int, j: int) -> int:
        return i * row + j

    # central block: 4 corners at arc-length quarters, scaled toward the
    # centroid, bilinearly interpolated into a 3-D patch.
    C00 = centroid + center_scale * (outer_pos[0] - centroid)
    C10 = centroid + center_scale * (outer_pos[n_side] - centroid)
    C11 = centroid + center_scale * (outer_pos[2 * n_side] - centroid)
    C01 = centroid + center_scale * (outer_pos[3 * n_side] - centroid)
    t_lat = np.arange(row) / n_side
    U = t_lat[:, None]                                           # (row,1)  i / n_side
    V = t_lat[None, :]                                           # (1,row)  j / n_side
    block = (((1 - U) * (1 - V))[..., None] * C00
             + (U * (1 - V))[..., None] * C10
             + (U * V)[..., None] * C11
             + ((1 - U) * V)[..., None] * C01).reshape(-1, 3)    # (row*row, 3)
    cquads = _grid_quads(n_side, n_side)

    peri_ids = np.array([cid(i, 0) for i in range(row)]
                        + [cid(n_side, j) for j in range(1, row)]
                        + [cid(i, n_side) for i in range(n_side - 1, -1, -1)]
                        + [cid(0, j) for j in range(n_side - 1, 0, -1)],
                        dtype=np.int64)
    peri_pos = block[peri_ids, :]                                # (P,3)
    # O-ring layers blending block perimeter out to boundary; radial[0]==0 is
    # the perimeter itself, so skip it
    fracs = radial[1:]
    layers = [block]
    ring = [peri_ids]
    nprev = block.shape[0]
    for t in fracs:
        layers.append((1.0 - t) * peri_pos + t * outer_pos)
        ring.append(nprev + np.arange(P, dtype=np.int64))
        nprev += P
    points = np.vstack(layers)

    k: IntArray = np.arange(P, dtype=np.int64)
    kn = (k + 1) % P
    ring_quads = [np.stack([b[k], b[kn], a[kn], a[k]], axis=1)    # CCW
                  for a, b in zip(ring[:-1], ring[1:])]
    quads = np.vstack([cquads, *ring_quads])

    # wall edges = side 1 of the outermost ring's quads (rows n_side^2 +
    # (n_radial-1)*P onward), named from the boundary loop's own per-segment tags.
    wall_q0 = n_side * n_side + (n_radial - 1) * P
    bnd, names = _curve_rows([(wall_q0 + m, 1) for m in range(P)],
                            boundary, wall_tag)
    qm = QuadMesh.from_corners(points, quads, *QuadMesh._order_bnd(bnd, names))
    # order-N: the ring quad is [b[k], b[kn], a[kn], a[k]] with b the outer layer, so
    # the outward-facing local side is 1.  ``layers[1:]`` are the ring curves, ring 0
    # being the straight block perimeter that stays with its bilinear guess.
    order = boundary.order
    overlays: list[Overlay] = []
    if order > 1:
        overlays = _ring_overlays(layers[1:], boundary, radial, peri_pos,
                                  n_side * n_side, P, 1)
    qm = _elevate(qm, order, overlays)
    return _apply_smoothing(qm, smoothing_method)


def half_ogrid(arc: LineMesh, spine: LineMesh,
               radial: int | FloatArray, *, center_scale: float = 0.5,
               wall_tag: str = "",
               smoothing_method: SmoothingMethod | None = None) -> QuadMesh:
    """Structured half-circle O-grid over a half-disk split along the ``spine``
    line (A1..A2); the wall ``arc`` (``(4*Ntheta+1, 3)``, arc[0]=A1, arc[-1]=A2)
    is the open boundary.  ``radial`` is either an ``int`` count of uniform rings
    or the O-ring layer positions with the
    initial position explicit (strictly increasing in ``[0, 1]``, ``radial[0]`` =
    inner block perimeter, last = ``1`` = wall;
    :func:`validate_layers <nekmeshpy.model.fields.validate_layers>`);
    ``center_scale`` is the inner
    block extent as a fraction of the spine.

    The ``spine`` is meshed exactly: its ``2*Ntheta+1 + 2*Nradial`` points must be
    sampled **in monotonic order along the diameter from A1 (fraction 0) to A2
    (fraction 1)** -- ``Nradial`` north caps, then the ``2*Ntheta+1`` center fan,
    then ``Nradial`` south caps.  With ``sN = (1-center_scale)/2`` and
    ``sS = (1+center_scale)/2``, the center fan is ``linspace(sN, sS, 2*Ntheta+1)``
    (spanning the inner block), the north caps are ``(1-radial[r])*sN`` and the south
    caps ``sS + radial[r]*(1-sS)`` for ``r = 1..Nradial`` -- so the full point set is
    just those fractions in ascending order (north caps rise ``0 -> sN``, the fan
    ``sN -> sS``, the south caps ``sS -> 1``).  At ``order > 1`` the spine's own
    private ``interior`` nodes are the seam geometry too: its point intervals partition
    the half-disk's flat side one-for-one, so each is overlaid onto the seam edge it
    spans and a curved spine is meshed exactly rather than straight-subdivided between
    its samples.  It must therefore carry the same ``order`` as ``arc``.
    The ``arc`` wall is named from the
    arc's per-segment ``element_tags``; a non-empty scalar ``wall_tag`` overrides
    that for the whole wall."""
    from .quadmesh import QuadMesh
    apts = _check_boundary(arc, "half_ogrid arc", 5)   # (na,3) backing array
    na = apts.shape[0]
    if (na - 1) % 4 != 0:
        raise ValueError("half_ogrid: arc must have 4*Ntheta+1 points (Ntheta >= 1)")
    Nt = (na - 1) // 4
    if not 0.0 < center_scale < 1.0:
        raise ValueError("half_ogrid needs center_scale in (0, 1)")
    radial = validate_layers(radial, "half_ogrid radial")
    Nr = radial.size - 1
    cs = center_scale

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

    north = sp[0:Nr][::-1]                      # A1 -> block; reverse to inner-first
    fe = sp[Nr:Nr + 2 * Nt + 1]                 # the center fan, sN..sS
    O = fe[Nt]                                  # spine midpoint (fan is symmetric)
    south = sp[Nr + 2 * Nt + 1:]                # block -> A2, already inner-first
    Q_N = O + cs * (apts[Nt, :] - O)
    Q_S = O + cs * (apts[3 * Nt, :] - O)
    ae = Q_N + (np.arange(2 * Nt + 1)[:, None] / (2 * Nt)) * (Q_S - Q_N)
    P_N = fe[0, :]
    P_S = fe[-1, :]

    ni = 2 * Nt
    nj = Nt
    rid: IntArray = np.zeros((ni + 1, nj + 1), dtype=np.int64)
    point_list = []
    for i in range(ni + 1):
        u = i / ni
        for j in range(nj + 1):
            v = j / nj
            left = (1 - v) * P_N + v * Q_N
            right = (1 - v) * P_S + v * Q_S
            bott = fe[i, :]
            top = ae[i, :]
            C = ((1 - v) * bott + v * top + (1 - u) * left + u * right
                 - ((1 - u) * (1 - v) * P_N + u * (1 - v) * P_S
                    + (1 - u) * v * Q_N + u * v * Q_S))
            point_list.append(C)
            rid[i, j] = len(point_list) - 1

    quads = _grid_quads(ni, nj).tolist()

    peri = np.concatenate([rid[0, 0:nj + 1], rid[1:ni + 1, nj], rid[ni, nj - 1::-1]])
    points = np.array(point_list, dtype=float)
    peripts = points[peri, :]

    lid = [peri]
    ring_pts: list[PointArray] = []
    for r in range(Nr):
        tau = radial[r + 1]                 # radial[0] == 0 is the block perimeter
        pts = (1 - tau) * peripts + tau * apts
        pts[0, :] = north[r]                # spine sample at (1-tau)*sN
        pts[-1, :] = south[r]              # spine sample at sS + tau*(1-sS)
        base = points.shape[0]
        points = np.vstack([points, pts])
        lid.append(base + np.arange(pts.shape[0]))
        ring_pts.append(pts)

    for r in range(Nr):
        a = lid[r]
        b = lid[r + 1]
        for k in range(4 * Nt):
            quads.append([a[k], a[k + 1], b[k + 1], b[k]])

    # wall arc edges = side 3 of the outermost ring's quads (rows (ni*nj) +
    # (Nr-1)*(4*Nt) onward); wall edge k tracks arc segment k.
    wall_q0 = ni * nj + (Nr - 1) * (4 * Nt)
    bnd, names = _curve_rows([(wall_q0 + k, 3) for k in range(4 * Nt)],
                            arc, wall_tag)
    qm = QuadMesh.from_corners(points, np.array(quads, dtype=np.int64),
                               *QuadMesh._order_bnd(bnd, names))
    # order-N: the ring quad here is [a[k], a[k+1], b[k+1], b[k]] with b the outer
    # layer, so the outward-facing local side is 3 (as the wall always was).
    # ``_blended_ring`` re-anchors each ring on its snapped spine end points.
    order = arc.order
    overlays: list[Overlay] = []
    if order > 1:
        q0 = ni * nj
        overlays = _ring_overlays(ring_pts, arc, radial, peripts,
                                  q0, 4 * Nt, 3)
        # ...and overlay the seam with the spine's *own* nodes, so a curved spine is
        # meshed exactly at order N too rather than straight-subdivided between its
        # samples.  Each seam edge is single-incidence (the half-disk's flat side), and
        # the spine's point intervals partition it exactly: intervals [0, Nr) are the
        # north radial edges (outermost first, so reversed), [Nr, Nr+ni) the inner
        # block's j == 0 row, and [Nr+ni, ...) the south radial edges.
        seg2line: IntArray = np.argsort(_chain_intervals(spine, "half_ogrid spine"))
        rs: IntArray = np.arange(Nr, dtype=np.int64)
        overlays += [
            (q0 + rs * (4 * Nt), 4,                       # north caps, block -> A1
             _sub_chain(spine, rs[::-1], seg2line)),
            (np.arange(ni, dtype=np.int64) * nj, 1,       # the center fan, A1 -> A2
             _sub_chain(spine, np.arange(Nr, Nr + ni, dtype=np.int64), seg2line)),
            (q0 + rs * (4 * Nt) + (4 * Nt - 1), 2,        # south caps, block -> A2
             _sub_chain(spine, Nr + ni + rs, seg2line)),
        ]
    qm = _elevate(qm, order, overlays)
    return _apply_smoothing(qm, smoothing_method)


#: The two nameable seams of a :func:`quadrant_ogrid`, in the order its arguments
#: are taken.  ``side_tags`` is keyed by these.
_QUADRANT_SEAMS = ("seam1", "seam2")


def quadrant_core(arc: LineMesh, seam1: LineMesh, seam2: LineMesh, *,
                  center_scale: float = 0.5) -> PointArray:
    """The core quarter of a :func:`quadrant_ogrid` as an ``(n+1, n+1, 3)`` grid,
    indexed ``[i][j]`` with ``i`` running ``O -> M1`` along ``seam1`` and ``j``
    running ``O -> M2`` along ``seam2``.

    It is a Coons patch whose two ``O``-ward sides are the caller's own seam fans --
    so a graded or bowed radius is honoured -- and whose two far sides are the
    straight chords into the core's far corner ``K = O + center_scale * (arc_mid - O)``.

    ``quadrant_ogrid`` builds its core with this, and it is public for the one caller
    that needs the same points *without* the surrounding mesh: a block that fills the
    corner region behind three quadrant faces -- an octant of a 3-D O-grid, whose
    core cube's three inner faces are exactly their three cores (see
    ``examples/quadrant_pipe_tjunction.py``).  Reproducing the formula outside the
    toolkit would work today and drift tomorrow; sharing it cannot.

    Only the seams' first ``n+1`` points and the arc's midpoint are read, so the
    ring-station part of a seam may be anything -- but the shapes must still be the
    ones :func:`quadrant_ogrid` takes."""
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
    """Quarter-disk O-grid: the 90-degree sibling of :func:`half_ogrid`.

    The region is bounded by the wall ``arc`` (open, ``A1 -> A2``, ``2*n+1`` points)
    and the two radii ``seam1`` (``O -> A1``) and ``seam2`` (``O -> A2``).  It is
    filled with **two** blocks -- a ``n x n`` core quarter plus one ``2n x Nradial``
    ring band wrapping the core's far corner -- which is exactly the quarter of
    :func:`ogrid` you get by cutting a full disk along two perpendicular diameters
    through its core-edge midpoints.  Four such quadrants
    ``QuadMesh.merge`` back into a conforming full disk.

    ``radial`` is either an ``int`` count of uniform rings or the O-ring layer
    positions with the initial position explicit (strictly increasing in ``[0, 1]``,
    ``radial[0]`` = core perimeter, last = ``1`` = wall;
    :func:`validate_layers <nekmeshpy.model.fields.validate_layers>`).
    ``center_scale`` sizes the core: its far corner sits at
    ``O + center_scale * (arc_midpoint - O)``.

    Both seams are **meshed exactly at the points given** -- like every other curve
    in the toolkit they are never resampled.  Each must carry exactly
    ``n+1 + Nradial`` points ascending from ``O``: the ``n+1`` core fan, then the
    ``Nradial`` ring stations.  Derive that sampling with
    :func:`quadrant_seam_fractions` and evaluate your own radius curve there
    (:meth:`LineMesh.line <nekmeshpy.linemesh.LineMesh.line>` for a straight radius,
    :meth:`LineMesh.loft_curve <nekmeshpy.linemesh.LineMesh.loft_curve>` for a bowed
    one), at ``arc``'s order -- at ``order > 1`` a seam's own private nodes *are* the
    radius geometry, so the orders must match.

    Passing the seams in rather than deriving them from a ``center`` is what makes
    adjacent quadrants weld exactly: two neighbours share one ``LineMesh`` object,
    the second through
    :meth:`LineMesh.reverse <nekmeshpy.linemesh.LineMesh.reverse>`, so the seam
    coordinates are bit-identical instead of two independent placements that agree
    only to round-off (the same reason :func:`spined_ogrid` reverses its spine).

    The wall is named from ``arc``'s per-line ``element_tags``, each seam from its
    own; a non-empty scalar ``wall_tag`` overrides the whole wall and a non-empty
    ``side_tags[...]`` (keyed ``seam1`` / ``seam2``) the whole seam."""
    from .quadmesh import QuadMesh
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
    bnd, names = _curve_rows([(wall_q0 + m, 1) for m in range(2 * n)],
                            arc, wall_tag)
    for which, sm, rows in seam_sides:
        seam_bnd, seam_names = _curve_rows(rows, sm, st.get(which, ""))
        bnd += seam_bnd
        names += seam_names
    qm = QuadMesh.from_corners(points, quads, *QuadMesh._order_bnd(bnd, names))

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
                            center_scale: float = 0.5) -> FloatArray:
    """The normalized ``O -> A`` positions of the ``n_side+1 + Nradial`` seam points
    that :func:`quadrant_ogrid` requires, ascending: the ``n_side+1`` core fan, then
    the ``Nradial`` ring stations.

    The non-obvious term is where the core corner ``M`` lands.  ``center_scale``
    places the core's *far* corner ``K`` at ``center_scale * R`` along the arc
    midpoint's radius; ``M`` is the midpoint of the core square's side, i.e. half its
    diagonal further in, so it sits at ``center_scale * cos(45 deg) * R`` -- **not**
    at ``center_scale * R``.  Using the latter builds a quadrant whose merged core is
    not a square and whose elements are visibly skewed at ``K``.

    :func:`quadrant_ogrid` never resamples a seam, so this is how a caller derives
    the sampling to prove: evaluate its own radius curve at these fractions and hand
    the result in."""
    ns = int(n_side)
    if ns < 1:
        raise ValueError("quadrant_seam_fractions needs n_side >= 1, got %d" % ns)
    if not 0.0 < center_scale < 1.0:
        raise ValueError("quadrant_seam_fractions needs center_scale in (0, 1)")
    rad = validate_layers(radial, "quadrant_seam_fractions radial")
    s_m = center_scale * float(np.cos(np.pi / 4.0))
    fr: FloatArray = np.concatenate([np.linspace(0.0, s_m, ns + 1),
                                     s_m + rad[1:] * (1.0 - s_m)])
    return fr


def quadrant_disc(arcs: Sequence[LineMesh], center: Point, radial: int | FloatArray,
                  *, center_scale: float = 0.5, wall_tag: str = "") -> QuadMesh:
    """Full disc from its four wall arcs and centre point: the four-quadrant
    analogue of :func:`ogrid`.

    ``arcs[q]`` is the wall running from corner ``q`` to corner ``q + 1`` (mod 4),
    all at the same order and node count.  Builds the four ``O -> corner`` seams
    once each -- shared by both neighboring quadrants, which is what makes them weld
    bit-exactly rather than to a tolerance -- sampled at
    :func:`quadrant_seam_fractions`, the same fan :func:`quadrant_ogrid` requires of
    any seam it is handed, then merges the four quadrants those seams and ``arcs``
    bound."""
    from ._assemble import merge
    n = (np.asarray(arcs[0].points).shape[0] - 1) // 2
    fr = quadrant_seam_fractions(n, radial, center_scale=center_scale)
    seams = [line(center, a.points[0], fr, order=arcs[0].order) for a in arcs]
    seams.append(seams[0])
    return merge([quadrant_ogrid(arcs[q], seams[q], seams[q + 1], radial,
                                 center_scale=center_scale, wall_tag=wall_tag)
                 for q in range(4)])


def spine_fractions(n_theta: int, radial: int | FloatArray,
                    center_scale: float = 0.5) -> FloatArray:
    """The normalized ``A1 -> A2`` positions of the ``2*n_theta+1 + 2*Nradial`` spine
    points that :func:`half_ogrid` and :func:`spined_ogrid` require, in ascending
    order: ``Nradial`` north caps, the ``2*n_theta+1`` center fan, then ``Nradial``
    south caps (see :func:`half_ogrid` for what each region is).

    Neither factory resamples a spine for you -- both mesh it exactly at the points
    given -- so this is how a caller **derives** the sampling to prove: evaluate its
    own spine curve at these fractions (with
    :meth:`LineMesh.loft_curve <nekmeshpy.linemesh.LineMesh.loft_curve>` for an analytic spine,
    or ``trimesh.ops.resample_polyline`` for a scanned one) and hand the result in.
    Keeping the formula here rather than in every caller is what stops the two from
    drifting apart."""
    nt = int(n_theta)
    if nt < 1:
        raise ValueError("spine_fractions needs n_theta >= 1, got %d" % nt)
    if not 0.0 < center_scale < 1.0:
        raise ValueError("spine_fractions needs center_scale in (0, 1)")
    rad = validate_layers(radial, "spine_fractions radial")
    s_n, s_s = (1.0 - center_scale) / 2, (1.0 + center_scale) / 2
    fr: FloatArray = np.concatenate([((1.0 - rad[1:]) * s_n)[::-1],
                                     np.linspace(s_n, s_s, 2 * nt + 1),
                                     s_s + rad[1:] * (1.0 - s_s)])
    return fr


def spined_ogrid(boundary: LineMesh, radial: int | FloatArray, *,
                 spine: LineMesh | None = None, center_scale: float = 0.5,
                 wall_tag: str = "",
                 smoothing_method: SmoothingMethod | None = None) -> QuadMesh:
    """Full-disk O-grid over a closed ``boundary`` split along a spine diameter
    into two :func:`half_ogrid` halves welded along the spine -- the clean way to
    O-grid a disk that has a natural ``A1..A2`` seam (a saddle-split vessel or pipe
    cross-section), so the caller need not hand-roll the arc split and merge.

    ``boundary`` is a closed loop of ``M = 8*Ntheta`` points with index ``0`` at
    ``A1`` and index ``M//2`` at ``A2`` (the two spine ends); it is split into the
    two ``A1 -> A2`` / ``A2 -> A1`` half-arcs, each meshed exactly.

    ``spine`` is the open ``A1 -> A2`` diameter curve and is **meshed exactly at the
    points given** -- like every other curve in the toolkit, it is never resampled or
    reordered on the caller's behalf.  It must therefore carry exactly the
    :func:`spine_fractions` sampling that :func:`half_ogrid` consumes
    (``2*Ntheta+1 + 2*Nradial`` points, ascending ``A1 -> A2``); anything else is a
    loud ``ValueError`` rather than a silent reinterpolation.  Derive it with
    ``spine_fractions(M // 8, radial, center_scale)`` and evaluate your own curve
    there, at ``boundary``'s order (a curved spine's own high-order nodes *are* the
    seam geometry, so the two orders must match).  The second half consumes
    :meth:`LineMesh.reverse <nekmeshpy.linemesh.LineMesh.reverse>` of that same mesh,
    so both halves share the seam bit-for-bit and the merge welds an exact
    coincidence rather than a near one.

    Omit it (``spine=None``, the default) to use the straight chord between ``A1``
    and ``A2``, which this factory owns as a shape and so can place exactly itself --
    the common case for a planar disc; pass a curve only to bow the seam.

    ``radial`` and ``center_scale`` are as in :func:`half_ogrid`.  Wall names come
    from ``boundary``'s per-line ``element_tags`` (split onto the two arcs); a
    non-empty scalar ``wall_tag`` overrides that for the whole wall.
    ``smoothing_method`` repositions each half's interior before the merge."""
    bpts = _check_boundary(boundary, "spined_ogrid boundary", 8)
    M = bpts.shape[0]
    if M % 8 != 0:
        raise ValueError(
            "spined_ogrid boundary must have 8*Ntheta points (a closed loop split "
            "into two 4*Ntheta+1 arcs); got %d" % M)
    nh = M // 2
    Nt = nh // 4
    if not 0.0 < center_scale < 1.0:
        raise ValueError("spined_ogrid needs center_scale in (0, 1)")
    radial = validate_layers(radial, "spined_ogrid radial")

    # The spine is meshed exactly at the points given -- nothing is resampled here.
    # A caller-supplied spine must already carry the [north caps, center fan, south
    # caps] sampling half_ogrid consumes; only the default straight chord is a shape
    # this factory owns, so only that one may be placed here.
    fr = spine_fractions(Nt, radial, center_scale)
    if spine is None:
        spine = line(bpts[0, :], bpts[nh, :], fr, order=boundary.order)
    elif spine.points.shape[0] != fr.shape[0]:
        raise ValueError(
            "spined_ogrid spine must have exactly 2*Ntheta+1 + 2*Nradial = %d points "
            "ascending A1 -> A2 (got %d); it is meshed exactly at the points given, "
            "never resampled -- evaluate your spine curve at "
            "spine_fractions(%d, radial, center_scale)"
            % (fr.shape[0], spine.points.shape[0], Nt))
    # the second half traverses A2 -> A1; ``reverse`` relabels rather than re-placing,
    # so both halves see bit-identical seam coordinates (and, at order > 1, carries the
    # spine's own interior nodes with it instead of re-subdividing them straight).
    spine2 = line_reverse(spine)

    # split the loop (and its per-segment tags) into the two half arcs: arc1 runs
    # A1 -> A2 over segments [0, nh), arc2 runs A2 -> A1 over segments [nh, M).
    # order-N: the loop's per-line private ``interior`` nodes split the same way (a
    # line element shares nothing but its endpoints), so each half arc carries its
    # exact wall geometry natively -- no curved block anywhere.
    seg = boundary._seg_tags()
    o = boundary.order
    inner: PointArray = boundary.interior
    arc1 = line_loft(bpts[0:nh + 1, :], interior=inner[0:nh],
                         element_tags=None if seg is None else seg[0:nh], order=o)
    arc2 = line_loft(np.vstack([bpts[nh:M, :], bpts[0:1, :]]),
                         interior=inner[nh:M],
                         element_tags=None if seg is None else seg[nh:M], order=o)
    h1 = half_ogrid(arc1, spine, radial, center_scale=center_scale,
                    wall_tag=wall_tag, smoothing_method=smoothing_method)
    h2 = half_ogrid(arc2, spine2, radial, center_scale=center_scale,
                    wall_tag=wall_tag, smoothing_method=smoothing_method)
    return merge([h1, h2])


#: Section helpers bound onto ``QuadMesh`` as ``staticmethod``s.  These answer a
#: question *about* a factory's input contract and return plain arrays rather than a
#: mesh, which is what keeps them out of ``FACTORIES``.
HELPERS: dict[str, Callable[..., FloatArray]] = {
    "spine_fractions": spine_fractions,
    "quadrant_seam_fractions": quadrant_seam_fractions,
    "quadrant_core": quadrant_core,
}

#: Open-region section factories bound onto ``QuadMesh`` by ``quadmesh/__init__.py``.
FACTORIES: dict[str, Callable[..., QuadMesh]] = {
    "structured": structured,
    "rectangle": rectangle,
    "ogrid": ogrid,
    "half_ogrid": half_ogrid,
    "quadrant_ogrid": quadrant_ogrid,
    "spined_ogrid": spined_ogrid,
    "quadrant_disc": quadrant_disc,
}
