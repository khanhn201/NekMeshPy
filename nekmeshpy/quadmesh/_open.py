"""Open-region :class:`~nekmeshpy.QuadMesh` section factories: fills that take a
boundary / edges and mesh the bounded region (``structured`` / ``rectangle`` /
``ogrid`` / ``half_ogrid`` / ``annulus``).

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

from .._typing import CurvedBlock, FloatArray, IntArray, Point, PointArray, StrArray
from ..linemesh import LineMesh
from ..linemesh._open import line
from ..model.fields import validate_layers
from ._helpers import Overlay, _apply_smoothing, _check_boundary, _elevate

if TYPE_CHECKING:
    from collections.abc import Callable

    from .quadmesh import QuadMesh


def rectangle(corners: PointArray | Sequence[Point], nx: int, ny: int, *,
              x_frac: FloatArray | None = None,
              y_frac: FloatArray | None = None,
              side_tags: Mapping[str, str] | None = None,
              smoothing_method: str | None = None,
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


def structured(edges: list[LineMesh], *,
               boundary_tags: Mapping[str, str] | None = None,
               smoothing_method: str | None = None) -> QuadMesh:
    """Transfinite (Coons-patch) quad grid over the surface bounded by four
    open edge lines ``edges = [bottom, right, top, left]`` in CCW loop order.
    The lines must share corners (form a closed loop).

    Resolution and node distribution come directly from the edge lines' own
    points (no resampling): ``bottom``/``top`` must share a point count
    (``nx+1``) and ``left``/``right`` another (``ny+1``), giving ``nx`` x ``ny``
    cells.

    Each side is named from its own edge line's uniform ``element_tags``;
    ``boundary_tags`` (keyed by ``"bottom"`` / ``"right"`` / ``"top"`` /
    ``"left"``) overrides that -- a non-empty entry replaces the side's tag, a
    present-but-empty entry suppresses the side.
    """
    from .quadmesh import QuadMesh
    if len(edges) != 4:
        raise ValueError("structured needs exactly 4 edge lines "
                         "[bottom, right, top, left]")
    bottom, right, top, left = edges
    for nm, e in (("bottom", bottom), ("right", right),
                  ("top", top), ("left", left)):
        _check_boundary(e, "structured " + nm + " edge", False, 2)
    order = bottom.order
    if any(e.order != order for e in edges):
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
    allpts = np.vstack([e.points for e in edges])
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
    # orient the two edge families so both run c0->c1 (u) / c0->c3 (v)
    cb = bottom.points                                     # c0 -> c1
    ct = top.points[::-1]                                  # c3 -> c2
    cl = left.points[::-1]                                 # c0 -> c3
    cr = right.points                                      # c1 -> c2
    P00, P10, P01, P11 = cb[0], cb[-1], ct[0], ct[-1]      # shared corners

    u = np.linspace(0.0, 1.0, nx + 1)[:, None, None]       # (nx+1,1,1)
    v = np.linspace(0.0, 1.0, ny + 1)[None, :, None]       # (1,ny+1,1)
    # Coons blend: (edge terms) - (bilinear corner correction)
    S = ((1 - v) * cb[:, None, :] + v * ct[:, None, :]
         + (1 - u) * cl[None, :, :] + u * cr[None, :, :]
         - ((1 - u) * (1 - v) * P00 + u * (1 - v) * P10
            + (1 - u) * v * P01 + u * v * P11))
    points = S.reshape(-1, 3)                              # id(i,j) = i*row + j
    row = ny + 1

    # quads in i-major / j-minor order (i in [0,nx), j in [0,ny))
    qi: IntArray = np.repeat(np.arange(nx, dtype=np.int64), ny)
    qj = np.tile(np.arange(ny, dtype=np.int64), nx)
    quads = np.stack([qi * row + qj, (qi + 1) * row + qj,
                      (qi + 1) * row + qj + 1, qi * row + qj + 1], axis=1)
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
    bt = boundary_tags or {}
    for side in bt:
        if side not in side_rows:
            raise ValueError("structured boundary_tags side must be one of "
                             "bottom/right/top/left, got %r" % side)
    bnd: list[list[int]] = []
    names: list[str] = []
    # each side is named by its edge's uniform element tag; a non-empty
    # boundary_tags[side] overrides, a present-but-empty entry suppresses it.
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
    # elevate to order N first (a no-op at order 1), then smooth: a repositioning
    # smoother rejects order > 1 (high-order smoothing is not implemented).
    qm = _elevate(
        QuadMesh.from_corners(points, quads, *QuadMesh._order_bnd(bnd, names)), order)
    return _apply_smoothing(qm, smoothing_method)


def ogrid(boundary: LineMesh, n_side: int, radial: FloatArray, *,
          center_scale: float = 0.5,
          wall_tag: str = "", smoothing_method: str | None = None) -> QuadMesh:
    """O-grid filling the closed ``boundary``: a central ``n_side x n_side``
    block at the loop centroid, surrounded by O-ring layers blending its
    perimeter out to the boundary.  ``center_scale`` sizes the block (fraction
    of the mean radius).

    Built in 3-D with no projection, so a curvy / non-planar boundary keeps its
    true shape; the block-and-ring build is only an initial guess, relaxed by
    ``smoothing_method="conduction"``.

    ``radial`` are the O-ring layer positions with the initial position explicit:
    strictly increasing in ``[0, 1]`` (``radial[0]`` = block perimeter, last =
    ``1`` = wall), giving ``radial.size - 1`` rings.

    The outer ring (wall) is named from ``boundary``'s per-line ``element_tags``;
    a non-empty scalar ``wall_tag`` overrides that for the whole wall."""
    from .quadmesh import QuadMesh
    if n_side < 1:
        raise ValueError("ogrid needs n_side >= 1")
    if not 0.0 < center_scale < 1.0:
        raise ValueError("ogrid needs center_scale in (0, 1)")
    radial = validate_layers(radial, "ogrid radial")
    n_radial = radial.size - 1
    bpts = _check_boundary(boundary, "ogrid boundary", True, 3)
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
    bi: IntArray = np.repeat(np.arange(n_side, dtype=np.int64), n_side)
    bj = np.tile(np.arange(n_side, dtype=np.int64), n_side)
    cquads = np.stack([bi * row + bj, (bi + 1) * row + bj,
                       (bi + 1) * row + bj + 1, bi * row + bj + 1], axis=1)

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
    # (n_radial-1)*P onward).
    wall_q0 = n_side * n_side + (n_radial - 1) * P
    # wall named from the boundary loop's per-segment tags; a non-empty scalar
    # wall_tag overrides that for the whole wall.
    wall_seg = boundary._seg_tags()
    bnd: list[list[int]] = []
    names: list[str] = []
    for m in range(P):
        nm = wall_tag if wall_tag else (wall_seg[m] if wall_seg is not None else "")
        if nm:
            bnd.append([wall_q0 + m, 1])
            names.append(nm)
    qm = QuadMesh.from_corners(points, quads, *QuadMesh._order_bnd(bnd, names))
    # order-N: the wall ring (side 1 of the outer ring quads) follows the exact
    # boundary loop, the interior stays a straight order-N fill (overlay ignored at
    # order 1, where _elevate is a no-op).  Elevate first, then smooth: a
    # repositioning smoother rejects order > 1 (high-order smoothing not implemented).
    overlays: list[Overlay] = [
        (wall_q0 + np.arange(P, dtype=np.int64), 1, boundary.curved)]
    qm = _elevate(qm, boundary.order, overlays)
    return _apply_smoothing(qm, smoothing_method)


def half_ogrid(arc: LineMesh, spine: LineMesh,
               radial: FloatArray, *, center_scale: float = 0.5,
               wall_tag: str = "",
               smoothing_method: str | None = None) -> QuadMesh:
    """Structured half-circle O-grid over a half-disk split along the ``spine``
    line (A1..A2); the wall ``arc`` (``(4*Ntheta+1, 3)``, arc[0]=A1, arc[-1]=A2)
    is the open boundary.  ``radial`` are the O-ring layer positions with the
    initial position explicit (strictly increasing in ``[0, 1]``, ``radial[0]`` =
    inner block perimeter, last = ``1`` = wall); ``center_scale`` is the inner
    block extent as a fraction of the spine.

    The ``spine`` is meshed exactly: its ``2*Ntheta+1 + 2*Nradial`` points must be
    sampled **in monotonic order along the diameter from A1 (fraction 0) to A2
    (fraction 1)** -- ``Nradial`` north caps, then the ``2*Ntheta+1`` center fan,
    then ``Nradial`` south caps.  With ``sN = (1-center_scale)/2`` and
    ``sS = (1+center_scale)/2``, the center fan is ``linspace(sN, sS, 2*Ntheta+1)``
    (spanning the inner block), the north caps are ``(1-radial[r])*sN`` and the south
    caps ``sS + radial[r]*(1-sS)`` for ``r = 1..Nradial`` -- so the full point set is
    just those fractions in ascending order (north caps rise ``0 -> sN``, the fan
    ``sN -> sS``, the south caps ``sS -> 1``).  The ``arc`` wall is named from the
    arc's per-segment ``element_tags``; a non-empty scalar ``wall_tag`` overrides
    that for the whole wall."""
    from .quadmesh import QuadMesh
    apts = _check_boundary(arc, "half_ogrid arc", False, 5)   # (na,3) backing array
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
    sp = _check_boundary(spine, "half_ogrid spine", False, 2)
    n_spine = 2 * Nt + 1 + 2 * Nr
    if sp.shape[0] != n_spine:
        raise ValueError(
            "half_ogrid spine must have exactly 2*Ntheta+1 + 2*Nradial = %d points "
            "(got %d); sample the spine curve monotonically A1 -> A2 as [north caps, "
            "center fan, south caps] (see half_ogrid docstring)"
            % (n_spine, sp.shape[0]))

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

    quads = []
    for i in range(ni):
        for j in range(nj):
            quads.append([rid[i, j], rid[i + 1, j], rid[i + 1, j + 1], rid[i, j + 1]])

    peri = np.concatenate([rid[0, 0:nj + 1], rid[1:ni + 1, nj], rid[ni, nj - 1::-1]])
    points = np.array(point_list, dtype=float)
    peripts = points[peri, :]

    lid = [peri]
    for r in range(Nr):
        tau = radial[r + 1]                 # radial[0] == 0 is the block perimeter
        pts = (1 - tau) * peripts + tau * apts
        pts[0, :] = north[r]                # spine sample at (1-tau)*sN
        pts[-1, :] = south[r]              # spine sample at sS + tau*(1-sS)
        base = points.shape[0]
        points = np.vstack([points, pts])
        lid.append(base + np.arange(pts.shape[0]))

    for r in range(Nr):
        a = lid[r]
        b = lid[r + 1]
        for k in range(4 * Nt):
            quads.append([a[k], a[k + 1], b[k + 1], b[k]])

    # wall arc edges = side 3 of the outermost ring's quads (rows (ni*nj) +
    # (Nr-1)*(4*Nt) onward); wall edge k tracks arc segment k.
    wall_q0 = ni * nj + (Nr - 1) * (4 * Nt)
    # wall named from the arc's per-segment tags; a non-empty scalar wall_tag
    # overrides that for the whole wall.
    wall_seg = arc._seg_tags()
    bnd: list[list[int]] = []
    names: list[str] = []
    for k in range(4 * Nt):
        nm = wall_tag if wall_tag else (wall_seg[k] if wall_seg is not None else "")
        if nm:
            bnd.append([wall_q0 + k, 3])
            names.append(nm)
    qm = QuadMesh.from_corners(points, np.array(quads, dtype=np.int64),
                               *QuadMesh._order_bnd(bnd, names))
    # order-N: the wall arc (side 3 of the outer ring quads) follows the exact arc;
    # the interior stays a straight order-N fill (overlay ignored at order 1, where
    # _elevate is a no-op).  Elevate first, then smooth: a repositioning smoother
    # rejects order > 1 (high-order smoothing not implemented).
    overlays: list[Overlay] = [
        (wall_q0 + np.arange(4 * Nt, dtype=np.int64), 3, arc.curved)]
    qm = _elevate(qm, arc.order, overlays)
    return _apply_smoothing(qm, smoothing_method)


def spined_ogrid(boundary: LineMesh, radial: FloatArray, *,
                 spine: LineMesh | None = None, center_scale: float = 0.5,
                 wall_tag: str = "",
                 smoothing_method: str | None = None) -> QuadMesh:
    """Full-disk O-grid over a closed ``boundary`` split along a spine diameter
    into two :func:`half_ogrid` halves welded along the spine -- the clean way to
    O-grid a disk that has a natural ``A1..A2`` seam (a saddle-split vessel or pipe
    cross-section), so the caller need not hand-roll the arc split, spine sampling
    and merge.

    ``boundary`` is a closed loop of ``M = 8*Ntheta`` points with index ``0`` at
    ``A1`` and index ``M//2`` at ``A2`` (the two spine ends); it is split into the
    two ``A1 -> A2`` / ``A2 -> A1`` half-arcs, each meshed exactly.  ``spine`` is the
    open ``A1 -> A2`` diameter curve at any sampling (possibly curved / deviating);
    it is resampled by arc length at the fractions each :func:`half_ogrid` half needs,
    so it is meshed exactly too and the two halves share it point-for-point.  Omit it
    (``spine=None``, the default) to use the straight chord between ``A1`` and ``A2``
    -- the common case for a planar disc; pass a curve only to bow the seam.

    ``radial`` and ``center_scale`` are as in :func:`half_ogrid`.  Wall names come
    from ``boundary``'s per-line ``element_tags`` (split onto the two arcs); a
    non-empty scalar ``wall_tag`` overrides that for the whole wall.
    ``smoothing_method`` repositions each half's interior before the merge."""
    from ..trimesh.ops import resample_polyline
    from .quadmesh import QuadMesh
    bpts = _check_boundary(boundary, "spined_ogrid boundary", True, 8)
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

    # spine sampled monotonically A1 -> A2 as [north caps, center fan, south caps]
    # (see half_ogrid); resample the given spine curve by arc length at those fractions
    # so it is meshed exactly and each half indexes it identically.
    s_n, s_s = (1.0 - center_scale) / 2, (1.0 + center_scale) / 2
    fr = np.concatenate([((1.0 - radial[1:]) * s_n)[::-1],
                         np.linspace(s_n, s_s, 2 * Nt + 1),
                         s_s + radial[1:] * (1.0 - s_s)])
    # default spine: the straight A1..A2 chord (boundary's two split points)
    sp = (bpts[[0, nh], :] if spine is None
          else _check_boundary(spine, "spined_ogrid spine", False, 2))
    spn1 = resample_polyline(sp, fr)
    spn2 = resample_polyline(sp[::-1, :], fr)

    # split the loop (and its per-segment tags) into the two half arcs: arc1 runs
    # A1 -> A2 over segments [0, nh), arc2 runs A2 -> A1 over segments [nh, M).
    # order-N: the loop's per-segment curved blocks split the same way, so each
    # half arc carries its exact wall geometry.
    seg = boundary._seg_tags()
    o = boundary.order
    cv: CurvedBlock | None = boundary.curved
    arc1 = LineMesh.open(bpts[0:nh + 1, :],
                         element_tags=None if seg is None else seg[0:nh],
                         order=o, curved=None if cv is None else cv[0:nh])
    arc2 = LineMesh.open(np.vstack([bpts[nh:M, :], bpts[0:1, :]]),
                         element_tags=None if seg is None else seg[nh:M],
                         order=o, curved=None if cv is None else cv[nh:M])
    h1 = half_ogrid(arc1, LineMesh.open(spn1), radial, center_scale=center_scale,
                    wall_tag=wall_tag, smoothing_method=smoothing_method)
    h2 = half_ogrid(arc2, LineMesh.open(spn2), radial, center_scale=center_scale,
                    wall_tag=wall_tag, smoothing_method=smoothing_method)
    return QuadMesh.merge([h1, h2])


def annulus(inner: LineMesh, outer: LineMesh, radial: FloatArray, *,
            smoothing_method: str | None = None,
            inner_tag: str = "", outer_tag: str = "",
            ) -> QuadMesh:
    """Ring O-grid filling the region between an inner and an outer closed loop
    -- e.g. a circular body inside a square far-field box.

    The two loops are paired by index: they must carry the same number of points
    ``N``, and point ``i`` of ``inner`` joins radially to point ``i`` of
    ``outer`` (no resampling; build the outer loop with the same point count and
    aligned index 0, e.g. a ``LineMesh.rectangle(w, h, N)`` box against a
    ``LineMesh.circle(r, N, start_theta=...)`` body).  ``radial`` are the ring positions with
    the initial position explicit (strictly increasing in ``[0, 1]``,
    ``radial[0]`` = inner ring, last = ``1`` = outer loop), giving
    ``radial.size - 1`` ring layers.  ``smoothing_method`` relaxes the ring
    interior with the inner/outer rings held fixed.

    Boundary tags come from the loops' per-line ``element_tags`` (each ring edge
    tagged from the matching loop segment, so a named box splits the outer ring
    into distinct sides).  A non-empty scalar ``inner_tag`` / ``outer_tag``
    overrides that for the whole inner / outer ring.

    The rings are always a genuine high-order blend: ``LineMesh.blend`` interpolates
    the two loops' curved blocks (``blend_ho``) so every ring carries curved
    tangential edges, and :meth:`loft` sweeps them radially -- the sibling of
    :meth:`HexMesh.annulus <nekmeshpy.hexmesh.HexMesh.annulus>` one dimension down.
    A repositioning ``smoothing_method`` (``conduction`` / ``winslow``) relaxes the
    corner grid, which cannot ride a curved block, so it is rejected at ``order > 1``
    (high-order smoothing is not implemented -- use ``order=1`` or drop the smoother);
    at ``order 1`` it relaxes the ring interior as usual.

    Built by :meth:`loft`-ing the blended rings; the inner / outer rings are the
    loft's near / far caps.  Gives ``N x (radial.size - 1)`` quads."""
    from .quadmesh import QuadMesh
    radial = validate_layers(radial, "annulus radial")
    A: FloatArray = _check_boundary(inner, "annulus inner", True, 3)   # (N,3)
    B: FloatArray = _check_boundary(outer, "annulus outer", True, 3)   # (N,3)
    if A.shape[0] != B.shape[0]:
        raise ValueError(
            "annulus: inner and outer loops must have equal point counts "
            "(got %d, %d); build both with the same count, "
            "e.g. LineMesh.rectangle(w, h, N) against circle(r, N)"
            % (A.shape[0], B.shape[0]))
    if float(np.min(np.linalg.norm(B - A, axis=1))) <= 0.0:
        raise ValueError("annulus: inner and outer loops touch or cross")
    order = inner.order
    if outer.order != order:
        raise ValueError("annulus: inner and outer loops must share the same order")

    # tags from each loop's per-segment element_tags; a non-empty scalar
    # inner_tag / outer_tag overrides that for the whole ring.
    inner_caps: str | StrArray = (
        inner_tag if inner_tag
        else (inner.element_tags if inner.element_group_tags else ""))
    outer_caps: str | StrArray = (
        outer_tag if outer_tag
        else (outer.element_tags if outer.element_group_tags else ""))

    # Blend the loops (carrying their curved blocks) and loft directly -- ring k =
    # blend_ho(inner, outer, t_k), so a high-order annulus is curved throughout, not
    # just on the two walls; loft builds the curved Coons columns.  blend copies the
    # ring topology but drops element_tags (the wall tags ride in inner_caps /
    # outer_caps as the loft's cap tags).  A requested repositioning smoother rejects
    # order > 1 (high-order smoothing not implemented); at order 1 it relaxes the
    # linear loft as before.
    rings = LineMesh.blend(inner, outer, radial)
    qm = QuadMesh.loft(rings, first_tag=inner_caps, last_tag=outer_caps)
    return _apply_smoothing(qm, smoothing_method)


#: Open-region section factories bound onto ``QuadMesh`` by ``quadmesh/__init__.py``.
FACTORIES: dict[str, Callable[..., QuadMesh]] = {
    "structured": structured,
    "rectangle": rectangle,
    "ogrid": ogrid,
    "half_ogrid": half_ogrid,
    "spined_ogrid": spined_ogrid,
    "annulus": annulus,
}
