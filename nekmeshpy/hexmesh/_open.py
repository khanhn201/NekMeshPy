"""Open-region :class:`~nekmeshpy.HexMesh` block factories: fills that take the
*boundary* of a region and mesh the volume it encloses.

:func:`tetra` is the only one today -- the hex-rung sibling of ``quadmesh._open``'s
region fills, and separate from ``_lift`` for the same reason they are: it owns a
*shape* model (a curvilinear tetrahedron) rather than being generic over any input.
Like the quad-rung factories these are plain free functions returning a ``HexMesh``;
``hexmesh/__init__.py`` binds each ``FACTORIES`` entry onto the class, so callers write
``HexMesh.tetra(...)``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np

from .._typing import IntArray, Point, PointArray
from ..linemesh import LineMesh
from ..linemesh._assemble import loft as line_loft
from ..model import conform
from ..model.interp import coons_grid
from ..quadmesh import QuadMesh
from ..quadmesh._assemble import loft as quad_loft
from ._assemble import loft as hex_loft
from ._assemble import merge

if TYPE_CHECKING:
    from collections.abc import Callable

    from .hexmesh import HexMesh


def _lin(n: int) -> PointArray:
    return np.arange(n, dtype=float) / (n - 1)


def _coons2(cb: PointArray, ct: PointArray,
            cl: PointArray, cr: PointArray) -> PointArray:
    """Transfinite patch of four boundary node chains, at their own uniform
    parameters: ``(len(cb), len(cl), 3)``."""
    return coons_grid(cb, ct, cl, cr, _lin(cb.shape[0]), _lin(cl.shape[0]))


def _side_map(quads: IntArray) -> dict[tuple[int, int], list[int]]:
    """``{sorted corner pair: [quad, ...]}`` over every quad side."""
    out: dict[tuple[int, int], list[int]] = {}
    for q in range(quads.shape[0]):
        for s in range(4):
            a, b = int(quads[q, s]), int(quads[q, (s + 1) % 4])
            out.setdefault((min(a, b), max(a, b)), []).append(q)
    return out


def _patch_walk(quads: IntArray, sides: dict[tuple[int, int], list[int]],
                q0: int, l0: int, who: str) -> IntArray:
    """The ``(A, B, 2)`` grid of ``(quad, origin)`` states of one structured patch.

    Walks out from quad ``q0`` with the patch's origin corner at its local index
    ``l0``, taking ``a`` along ``l -> l+1`` and ``b`` along ``l -> l+3``.  The walk
    stops where a side has no second quad, which is exactly what separates the three
    patches of a triangular face: their common sides are the spokes to the face
    centre, and a walk started at a corner reaches one only from inside its own
    patch."""
    def step(state: tuple[int, int], axis: int) -> tuple[int, int] | None:
        q, l = state
        u, v = ((l + 1) % 4, (l + 2) % 4) if axis == 0 else ((l + 3) % 4, (l + 2) % 4)
        a, b = int(quads[q, u]), int(quads[q, v])
        nxt = [t for t in sides[(min(a, b), max(a, b))] if t != q]
        if not nxt:
            return None
        return nxt[0], int(np.flatnonzero(quads[nxt[0]] == a)[0])

    spine: list[tuple[int, int]] = [(q0, l0)]
    while (nb := step(spine[-1], 1)) is not None:
        spine.append(nb)
    rows: list[list[tuple[int, int]]] = []
    for st in spine:
        col = [st]
        while (na := step(col[-1], 0)) is not None:
            col.append(na)
        rows.append(col)
    widths = {len(c) for c in rows}
    if len(widths) != 1:
        raise ValueError(
            "%s: a face patch is not a structured grid (row lengths %s) -- each face "
            "must be three structured patches meeting at one interior node"
            % (who, sorted(widths)))
    return np.array(rows, dtype=np.int64).transpose(1, 0, 2)


def _lattice(patch: IntArray, blocks: PointArray, order: int) -> PointArray:
    """Stitch a patch's ``(A, B)`` grid of ``(order+1)**2`` element blocks into one
    ``(A*order+1, B*order+1, 3)`` node lattice in the patch's own frame.

    An element's block is stored ``i`` fastest with its corner ``0`` at ``(0,0)``, so
    a patch cell whose origin is local corner ``l`` enters rotated by ``l`` quarter
    turns."""
    g: IntArray = np.arange(order + 1, dtype=np.int64)
    ia: IntArray = g[:, None] + np.zeros_like(g)[None, :]
    ib: IntArray = np.zeros_like(g)[:, None] + g[None, :]
    n = order
    turn: tuple[tuple[IntArray, IntArray], ...] = (
        (ia, ib), (n - ib, ia), (n - ia, n - ib), (ib, n - ia))
    A, B = int(patch.shape[0]), int(patch.shape[1])
    out: PointArray = np.empty((A * order + 1, B * order + 1, 3), dtype=float)
    for a in range(A):
        for b in range(B):
            q, l = int(patch[a, b, 0]), int(patch[a, b, 1])
            i, j = turn[l]
            blk = blocks[q].reshape(order + 1, order + 1, 3)      # [j][i]
            out[a * order:(a + 1) * order + 1,
                b * order:(b + 1) * order + 1] = blk[j, i]
    return out


def _face_patches(qm: QuadMesh, who: str) -> list[PointArray]:
    """Recover a triangular face as its three patches, each a node lattice indexed
    **from its corner**: ``[0, 0]`` is the corner and ``[-1, -1]`` the face centre.

    The structure is read off the connectivity rather than declared: a triangle
    meshed as three structured patches has exactly three corner nodes lying on a
    single quad and exactly one interior node lying on three.  Anything else is
    rejected rather than silently half-meshed.

    Each walk starts at the **centre**, not at a corner.  From a corner the patch's
    two far sides are the spokes to the centre, which are interior edges with a quad
    on each side, so the walk would run straight through into the neighbouring patch;
    from the centre the two far sides are the face's own boundary, where it stops."""
    quads: IntArray = qm.quads
    val = np.bincount(quads.ravel(), minlength=qm.points.shape[0])
    corners: IntArray = np.flatnonzero(val == 1)
    centres: IntArray = np.flatnonzero(val == 3)
    if corners.shape[0] != 3 or centres.shape[0] != 1:
        raise ValueError(
            "%s: each face must be a triangle meshed as three structured patches "
            "meeting at one interior node -- expected 3 nodes on exactly one quad and "
            "1 node on exactly three, got %d and %d"
            % (who, corners.shape[0], centres.shape[0]))
    order = qm.order
    nodes, conn = conform.conformal_quad(qm.points, quads, qm.quad, qm.flip,
                                         qm.lines.interior, qm.interior, order)
    blocks: PointArray = nodes[conn]
    sides = _side_map(quads)
    c = int(centres[0])
    lats: list[PointArray] = []
    for q0 in np.flatnonzero((quads == c).any(axis=1)):
        l0 = int(np.flatnonzero(quads[int(q0)] == c)[0])
        # centre-origin, then flipped so [0,0] is the corner and [-1,-1] the centre
        lats.append(_lattice(_patch_walk(quads, sides, int(q0), l0, who),
                             blocks, order)[::-1, ::-1])
    got = np.array([lat[0, 0] for lat in lats])
    want = qm.points[corners]
    if not np.allclose(np.sort(got, axis=0), np.sort(want, axis=0),
                       rtol=0.0, atol=1e-9):
        raise ValueError(
            "%s: the three patches do not reach the face's three corners -- it is not "
            "a triangle meshed as three structured patches" % who)
    return lats


def _coons3(f: dict[tuple[int, int], PointArray]) -> PointArray:
    """The transfinite interior of a block from its six face lattices.

    ``f[(axis, end)]`` is the face at ``end`` (0 or 1) of ``axis``, indexed by the
    other two axes in increasing order.  Faces minus edges plus corners -- the
    standard 3-D Coons sum -- so all six faces come back exactly, which is what lets
    neighbouring blocks weld: each shared face is computed from the same boundary data
    on both sides."""
    ni, nj = f[(2, 0)].shape[0], f[(2, 0)].shape[1]
    nk = f[(0, 0)].shape[1]
    uu = (_lin(ni)[:, None, None, None], )
    vv = (_lin(nj)[None, :, None, None], )
    ww = (_lin(nk)[None, None, :, None], )
    u, v, w = (1 - uu[0], uu[0]), (1 - vv[0], vv[0]), (1 - ww[0], ww[0])
    out: PointArray = np.zeros((ni, nj, nk, 3), dtype=float)
    for a in (0, 1):                                        # the six faces
        out = (out + u[a] * f[(0, a)][None, :, :, :]
               + v[a] * f[(1, a)][:, None, :, :]
               + w[a] * f[(2, a)][:, :, None, :])
    for a in (0, 1):                                        # the twelve edges
        for b in (0, 1):
            out = (out
                   - v[a] * w[b] * f[(1, a)][:, -b, :][:, None, None, :]
                   - u[a] * w[b] * f[(0, a)][:, -b, :][None, :, None, :]
                   - u[a] * v[b] * f[(0, a)][-b, :, :][None, None, :, :])
    for a in (0, 1):                                        # the eight corners
        for b in (0, 1):
            for c in (0, 1):
                out = out + u[a] * v[b] * w[c] * f[(0, a)][-b, -c]
    return out


def _face_tag(qm: QuadMesh, who: str) -> str:
    """A face's single ``element_tags`` name, or ``""`` if it carries none.

    A tetrahedron side is one boundary patch, so its tag is one name: a partly tagged
    face is a caller mistake worth naming rather than a per-quad channel worth
    plumbing.  Tagging at the lowest level -- on the face that becomes the wall --
    is the toolkit-wide convention."""
    names = qm.element_group_tags
    if len(names) > 1:
        raise ValueError(
            "%s: a face must carry one element tag or none, got %s -- a tetrahedron "
            "side is a single boundary patch" % (who, list(names)))
    if names and not np.all(qm.element_tags == names[0]):
        raise ValueError(
            "%s: face tagged %r has untagged quads -- tag the whole face or none of it"
            % (who, names[0]))
    return str(names[0]) if names else ""


def _block(lat: PointArray, order: int, tags: tuple[str, str, str]) -> HexMesh:
    """A hex block from its full ``(ni*N+1, nj*N+1, nk*N+1, 3)`` node lattice.

    Every node is *placed* from ``lat``; nothing is subdivided, which is what
    ``from_grid`` cannot do -- it takes corners and blends straight, so at
    ``order > 1`` its edges are chords.  Assembled through the three rungs' ``loft``
    (``LineMesh.loft(interior=)`` -> ``QuadMesh.loft(sweep_nodes=)`` ->
    ``HexMesh.loft(sweep_nodes=)``), so the B-rep, the numbering and the D4 face codes
    all come from the existing sweep primitive rather than being re-derived here.

    ``tags`` names the block's three **outer** sides -- the faces at ``i = 0``,
    ``j = 0`` and ``k = 0``, which are the three tetrahedron sides at this corner.
    Each rides a channel the rung below already has: the ``i = 0`` side from a
    boundary *point* tag on every line profile, the ``j = 0`` side from each section
    sweep's near cap, and the ``k = 0`` side from the block sweep's near cap."""
    o = order
    ti, tj, tk = tags
    nl, nm, nn = ((s - 1) // o for s in lat.shape[:3])
    bnd = np.array([[0, 1]], dtype=np.int64) if ti else None

    def profile(j: int, k: int) -> LineMesh:
        col: PointArray = lat[:, j, k, :]
        inner = (None if o == 1 else
                 np.stack([col[i * o + 1:i * o + o] for i in range(nl)], axis=0))
        return line_loft(col[::o], interior=inner, order=o, boundaries=bnd,
                         boundary_tags=[ti] if ti else None)

    def section(k: int) -> QuadMesh:
        return quad_loft([profile(j * o, k) for j in range(nm + 1)],
                         sweep_nodes=None if o == 1 else
                         [[profile(j * o + m, k) for m in range(1, o)]
                          for j in range(nm)],
                         first_tag=tj)

    return hex_loft([section(k * o) for k in range(nn + 1)],
                    sweep_nodes=None if o == 1 else
                    [[section(k * o + m) for m in range(1, o)] for k in range(nn)],
                    first_tag=tk)


def _match(a: PointArray, b: PointArray, tol: float) -> bool:
    return (a.shape == b.shape
            and bool(np.allclose(a, b, rtol=0.0, atol=tol)))


def _orient(a: tuple[PointArray, str], b: tuple[PointArray, str],
            c: tuple[PointArray, str], tol: float,
            who: str) -> tuple[tuple[PointArray, str], ...]:
    """Put the three patches meeting at one corner on a common axis frame.

    ``a`` fixes the frame: its two axes are tetrahedron edges ``e0`` and ``e1``.  The
    other two are then identified by *which of their own edge curves matches* ``a``'s
    -- the one sharing ``e1`` is returned second (indexed ``e1, e2``) and the one
    sharing ``e0`` third (indexed ``e0, e2``).  Comparing curves rather than trusting
    the argument order is what lets a caller hand the four faces in any order, so each
    patch's **tag travels with it** rather than with its position."""
    e0, e1 = a[0][:, 0], a[0][0, :]
    out: dict[int, tuple[PointArray, str]] = {}
    for p, tag in (b, c):
        for cand in (p, p.transpose(1, 0, 2)):
            if _match(cand[:, 0], e1, tol):
                out[1] = (cand, tag)               # (e1, e2)
            elif _match(cand[:, 0], e0, tol):
                out[2] = (cand, tag)               # (e0, e2)
    if set(out) != {1, 2}:
        raise ValueError(
            "%s: the three patches meeting at a corner do not share their two edges "
            "pairwise -- the faces are not conformal along the tetrahedron's edges "
            "(or two of them are the same face)" % who)
    return a, out[1], out[2]


def tetra(faces: Sequence[QuadMesh], *,
          center: Point | Sequence[float] | None = None) -> HexMesh:
    """Mesh the curvilinear **tetrahedron** enclosed by four triangular ``faces``.

    Each face is a ``QuadMesh`` triangle meshed as **three structured patches meeting
    at one interior node** -- the standard quad split of a triangle, refined.  The
    four must share their six edges pairwise and meet at four corners; all of that is
    recovered from the connectivity, so nothing has to be declared and a mismatch is a
    loud ``ValueError`` rather than a silently twisted block.

    The fill is the classic tetrahedron split -- **one hex block per corner** -- and
    each block inherits whatever split the faces already carry: along each of the
    three edges at its corner it spans that edge's portion up to the node where the
    two incident patches meet.  A block's three outer sides *are* the patches of the
    three faces at that corner, taken verbatim and never re-interpolated, so the mesh
    is exact wherever the input faces are.  Its three inner sides are transfinite
    patches of two face spokes and two chords into the centre, computed identically
    from both blocks that share them, so they weld.

    ``center`` defaults to the centroid of the four face centres.  Pass one when that
    is a poor choice: a tetrahedron with three similar faces and one much larger has
    its natural centroid near the plane of the three small faces' centres, and three
    nearly coplanar edges at a corner is a flat cell.

    Order N rides through: each face's own nodes are read off its B-rep with the
    conformal walk, and each block is assembled from its full node lattice rather than
    from corners, so a curved input face gives a curved block at any order.

    This is what fills the crotch of a pipe junction, where three quadrant faces meet
    a patch of pipe wall -- an octant of a 3-D O-grid is a tetrahedron in exactly this
    sense, its ``n**3`` core block and three ``n x n x Nradial`` slabs being the four
    corner blocks (see ``examples/quadrant_pipe_tjunction.py``)."""
    who = "HexMesh.tetra"
    fs = list(faces)
    if len(fs) != 4:
        raise ValueError("%s needs exactly 4 faces, got %d" % (who, len(fs)))
    orders = {f.order for f in fs}
    if len(orders) != 1:
        raise ValueError(
            "%s: all four faces must share an order (got %s) -- a face's own nodes are "
            "the block boundary, so a lower-order one cannot describe it"
            % (who, sorted(orders)))
    order = fs[0].order
    rec = [_face_patches(f, who) for f in fs]
    ftag = [_face_tag(f, who) for f in fs]
    tol = conform.entity_tol(np.vstack([f.points for f in fs]))

    # -- the four tetrahedron corners, matched across faces by position.
    pts: list[Point] = []
    ids: list[list[int]] = []
    for lats in rec:
        row: list[int] = []
        for lat in lats:
            p: Point = lat[0, 0]
            hit = [k for k, q in enumerate(pts)
                   if float(np.linalg.norm(p - q)) <= tol]
            if not hit:
                pts.append(p)
                hit = [len(pts) - 1]
            row.append(hit[0])
        ids.append(row)
    if len(pts) != 4:
        raise ValueError(
            "%s: the four faces must meet at exactly 4 corners, found %d -- they do "
            "not bound a tetrahedron" % (who, len(pts)))

    at: list[list[tuple[int, int]]] = [[] for _ in range(4)]
    for fi, row in enumerate(ids):
        if len(set(row)) != 3:
            raise ValueError("%s: face %d meets the same corner twice" % (who, fi))
        for k, v in enumerate(row):
            at[v].append((fi, k))
    for v, lst in enumerate(at):
        if len(lst) != 3:
            raise ValueError(
                "%s: corner %d lies on %d faces, expected 3 -- the four faces must "
                "share their six edges pairwise" % (who, v, len(lst)))

    mid: Point = (np.mean([r[0][-1, -1] for r in rec], axis=0)
                  if center is None
                  else np.asarray(center, dtype=float).reshape(3))

    blocks = []
    for v in range(4):
        (fa, ka), (fb, kb), (fc, kc) = at[v]
        (pa, ta), (pb, tb), (pc, tc) = _orient(
            (rec[fa][ka], ftag[fa]), (rec[fb][kb], ftag[fb]),
            (rec[fc][kc], ftag[fc]), tol, who)
        # axis 0 / 1 / 2 run along the three tetrahedron edges at this corner;
        # pa spans (0, 1), pb spans (1, 2), pc spans (0, 2).
        if pb.shape[1] != pc.shape[1]:
            raise ValueError(
                "%s: the patches at a corner disagree on an edge's node count "
                "(%d vs %d) -- the faces are not conformal along that edge"
                % (who, pb.shape[1], pc.shape[1]))
        qa, qb, qc = pa[-1, -1], pb[-1, -1], pc[-1, -1]      # the three face centres
        t0, t1, t2 = _lin(pa.shape[0]), _lin(pa.shape[1]), _lin(pb.shape[1])

        def chord(p: Point, t: PointArray, m: Point = mid) -> PointArray:
            return p + t[:, None] * (m - p)

        f: dict[tuple[int, int], PointArray] = {
            (2, 0): pa, (0, 0): pb, (1, 0): pc,
            # axis 0's far side, [j][k]: corners m0, qa, mid, qc
            (0, 1): _coons2(pa[-1, :], chord(qc, t1), pc[-1, :], chord(qa, t2)),
            # axis 1's far side, [i][k]: corners m1, qa, mid, qb
            (1, 1): _coons2(pa[:, -1], chord(qb, t0), pb[-1, :], chord(qa, t2)),
            # axis 2's far side, [i][j]: corners m2, qc, mid, qb
            (2, 1): _coons2(pc[:, -1], chord(qb, t0), pb[:, -1], chord(qc, t1)),
        }
        # The three outer sides carry their own face's tag -- i = 0 is pb's, j = 0
        # is pc's and k = 0 is pa's, matching the f[(axis, 0)] assignment above.  The
        # tags come back from _orient with their patches, because it is free to swap
        # the two it is handed.
        blocks.append(_block(_coons3(f), order, (tb, tc, ta)))
    return merge(blocks)


#: Open-region block factories bound onto ``HexMesh`` by ``hexmesh/__init__.py``.
FACTORIES: dict[str, Callable[..., HexMesh]] = {"tetra": tetra}
