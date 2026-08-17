"""Shape factories for the ``HexMesh`` rung -- the ones owning a *shape model* rather
than being generic over any input."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .._typing import IntArray, Point, PointArray
from ..core import conform
from ..core.interp import coons_grid
from ..core.tags import ElementTags
from ..linemesh import LineMesh
from ..linemesh.assemble import loft as line_loft
from ..pointmesh import PointMesh
from ..quadmesh import QuadMesh
from ..quadmesh.assemble import loft as quad_loft
from .assemble import loft as hex_loft
from .assemble import merge
from .hexmesh import HexMesh


def _lin(n: int) -> PointArray:
    return np.arange(n, dtype=float) / (n - 1)


def _coons2(cb: PointArray, ct: PointArray,
            cl: PointArray, cr: PointArray) -> PointArray:
    """Transfinite patch of four boundary node chains, at their own uniform
    parameters: ``(len(cb), len(cl), 3)``."""
    return coons_grid(cb, ct, cl, cr, _lin(cb.shape[0]), _lin(cl.shape[0]))


def _chord(p: Point, q: Point, t: PointArray) -> PointArray:
    """The straight chord ``p -> q`` sampled at parameters ``t``: ``(len(t), 3)``."""
    return p + t[:, None] * (q - p)


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
    """The ``(A, B, 2)`` grid of ``(quad, origin)`` states of one structured patch."""
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
    ``(A*order+1, B*order+1, 3)`` node lattice in the patch's own frame."""
    g: IntArray = np.arange(order + 1, dtype=np.int64)
    # ``(order+1, order+1)`` index grids: ia counts up axis a, ib up axis b
    ia: IntArray = np.repeat(g[:, None], order + 1, axis=1)
    ib: IntArray = np.repeat(g[None, :], order + 1, axis=0)
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
    **from its corner**: ``[0, 0]`` is the corner and ``[-1, -1]`` the face centre."""
    quads: IntArray = qm.corners
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
    nodes, conn = conform.conformal_quad(qm.points, quads, qm.quads, qm.orient,
                                         qm.line_mesh.interior, qm.interior, order)
    blocks: PointArray = nodes[conn]
    sides = _side_map(quads)
    c = int(centres[0])
    lats: list[PointArray] = []
    for q0 in (int(q) for q in np.flatnonzero((quads == c).any(axis=1))):
        l0 = int(np.flatnonzero(quads[q0] == c)[0])
        # centre-origin, then flipped so [0,0] is the corner and [-1,-1] the centre
        lats.append(_lattice(_patch_walk(quads, sides, q0, l0, who),
                             blocks, order)[::-1, ::-1])
    got = np.array([lat[0, 0] for lat in lats])
    want = qm.points[corners]
    if not np.allclose(np.sort(got, axis=0), np.sort(want, axis=0),
                       rtol=0.0, atol=1e-9):
        raise ValueError(
            "%s: the three patches do not reach the face's three corners -- it is not "
            "a triangle meshed as three structured patches" % who)
    return lats


def _corner_incidence(rec: Sequence[Sequence[PointArray]], tol: float,
                      who: str) -> list[list[tuple[int, int]]]:
    """``[[(face, patch), x3], x4]`` -- the three patches meeting at each of the
    tetrahedron's four corners."""
    pts: list[Point] = []
    ids: list[list[int]] = []
    for lats in rec:
        row: list[int] = []
        for lat in lats:
            p: Point = lat[0, 0]
            hit = next((k for k, q in enumerate(pts)
                        if float(np.linalg.norm(p - q)) <= tol), None)
            if hit is None:
                pts.append(p)
                hit = len(pts) - 1
            row.append(hit)
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
    return at


def _coons3(f: dict[tuple[int, int], PointArray]) -> PointArray:
    """The transfinite interior of a block from its six face lattices."""
    ni, nj = f[(2, 0)].shape[0], f[(2, 0)].shape[1]
    nk = f[(0, 0)].shape[1]
    ti: PointArray = _lin(ni)[:, None, None, None]
    tj: PointArray = _lin(nj)[None, :, None, None]
    tk: PointArray = _lin(nk)[None, None, :, None]
    # per axis, the blend weights of its ``end = 0`` and ``end = 1`` faces
    u, v, w = (1 - ti, ti), (1 - tj, tj), (1 - tk, tk)
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
    """A face's single ``element_tags`` name, or ``""`` if it carries none."""
    names = qm.element_group_tags
    if len(names) > 1:
        raise ValueError(
            "%s: a face must carry one element tag or none, got %s -- a tetrahedron "
            "side is a single boundary patch" % (who, list(names)))
    if names and not qm.element_tags.is_uniform(qm.n_quads):
        raise ValueError(
            "%s: face tagged %r has untagged quads -- tag the whole face or none of it"
            % (who, names[0]))
    return str(names[0]) if names else ""


def _block(lat: PointArray, order: int, tags: tuple[str, str, str],
           element_tag: str) -> HexMesh:
    """A hex block from its full ``(ni*N+1, nj*N+1, nk*N+1, 3)`` node lattice."""
    o = order
    ti, tj, tk = tags
    nl, nm, nn = ((s - 1) // o for s in lat.shape[:3])
    bnd = ElementTags([0], [ti]) if ti else None

    def profile(j: int, k: int) -> LineMesh:
        col: PointArray = lat[:, j, k, :]
        inner = (None if o == 1 else
                 np.stack([col[i * o + 1:i * o + o] for i in range(nl)], axis=0))
        lm = line_loft(col[::o], interior=inner, order=o)
        return LineMesh(PointMesh(lm.points, bnd), lm.lines, lm.interior)

    def section(k: int) -> QuadMesh:
        return quad_loft([profile(j * o, k) for j in range(nm + 1)],
                         sweep_nodes=None if o == 1 else
                         [[profile(j * o + m, k) for m in range(1, o)]
                          for j in range(nm)],
                         first_tag=tj)

    return hex_loft([section(k * o) for k in range(nn + 1)],
                    sweep_nodes=None if o == 1 else
                    [[section(k * o + m) for m in range(1, o)] for k in range(nn)],
                    first_tag=tk, element_tags=element_tag or None)


def _match(a: PointArray, b: PointArray, tol: float) -> bool:
    return (a.shape == b.shape
            and bool(np.allclose(a, b, rtol=0.0, atol=tol)))


def _orient(a: tuple[PointArray, str], b: tuple[PointArray, str],
            c: tuple[PointArray, str], tol: float,
            who: str) -> tuple[tuple[PointArray, str], ...]:
    """Put the three patches meeting at one corner on a common axis frame."""
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
          center: Point | Sequence[float] | None = None,
          element_tag: str = "") -> HexMesh:
    """Mesh the curvilinear **tetrahedron** enclosed by four triangular ``faces``.

    ``element_tag`` names all four octants, so a tetra filling a corner of a larger
    region can carry that region's name like any other block does."""
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

    # the four tetrahedron corners, matched across faces by position
    at = _corner_incidence(rec, tol, who)
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
        # the three inner sides are transfinite patches of two face spokes and two
        # chords from a face centre into ``mid``.
        f: dict[tuple[int, int], PointArray] = {
            (2, 0): pa, (0, 0): pb, (1, 0): pc,
            # axis 0's far side, [j][k]: corners m0, qa, mid, qc
            (0, 1): _coons2(pa[-1, :], _chord(qc, mid, t1),
                            pc[-1, :], _chord(qa, mid, t2)),
            # axis 1's far side, [i][k]: corners m1, qa, mid, qb
            (1, 1): _coons2(pa[:, -1], _chord(qb, mid, t0),
                            pb[-1, :], _chord(qa, mid, t2)),
            # axis 2's far side, [i][j]: corners m2, qc, mid, qb
            (2, 1): _coons2(pc[:, -1], _chord(qb, mid, t0),
                            pb[:, -1], _chord(qc, mid, t1)),
        }
        # The three outer sides carry their own face's tag -- i = 0 is pb's, j = 0
        # is pc's and k = 0 is pa's, matching the f[(axis, 0)] assignment above.  The
        # tags come back from _orient with their patches, because it is free to swap
        # the two it is handed.
        blocks.append(_block(_coons3(f), order, (tb, tc, ta), element_tag))
    return merge(blocks)

__all__ = [
    "tetra",
]
