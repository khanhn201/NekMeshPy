"""Surface algorithms on a ``TriMesh`` as free functions. All indices 0-based."""

from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .._typing import BoolArray, FloatArray, IntArray, Point, PointArray
from ..linemesh import LineMesh

if TYPE_CHECKING:
    from .trimesh import TriMesh


# -- Laplace ------------------------------------------------------------
def cotan_laplacian(surface: TriMesh) -> sp.csr_matrix:
    """Linear (P1) cotangent-weight stiffness matrix (recomputed each call)."""
    xyz, tri = surface.points, surface.tris
    nv, nt = xyz.shape[0], tri.shape[0]
    cot_clamp = 1e3
    p = xyz[tri]                                     # (nt,3,3)
    # Build triples in the original loop order so duplicate summation is bit-identical.
    I = np.empty((nt, 3, 4))
    J = np.empty((nt, 3, 4))
    S = np.empty((nt, 3, 4))
    valid = np.zeros((nt, 3), dtype=bool)
    for c in range(3):
        a, b = (c + 1) % 3, (c + 2) % 3
        e1 = p[:, a, :] - p[:, c, :]
        e2 = p[:, b, :] - p[:, c, :]
        area2 = np.linalg.norm(np.cross(e1, e2), axis=1)
        ok = area2 >= 1e-14
        cotc = np.divide(np.sum(e1 * e2, axis=1), area2,
                         out=np.zeros(nt), where=ok)
        cotc = np.clip(cotc, -cot_clamp, cot_clamp)
        wgt = 0.5 * cotc
        ia, ib = tri[:, a], tri[:, b]
        I[:, c, 0], I[:, c, 1], I[:, c, 2], I[:, c, 3] = ia, ib, ia, ib
        J[:, c, 0], J[:, c, 1], J[:, c, 2], J[:, c, 3] = ib, ia, ia, ib
        S[:, c, 0], S[:, c, 1], S[:, c, 2], S[:, c, 3] = -wgt, -wgt, wgt, wgt
        valid[:, c] = ok
    mask = np.repeat(valid[:, :, None], 4, axis=2).ravel()
    If, Jf, Sf = I.ravel()[mask], J.ravel()[mask], S.ravel()[mask]
    return sp.coo_matrix((Sf, (If, Jf)), shape=(nv, nv)).tocsr()


def solve_dirichlet(surface: TriMesh, dpoints: IntArray, dvals: FloatArray) -> FloatArray:
    """Solve the Laplace system with Dirichlet values at ``dpoints``; returns an ``(nv,)`` field."""
    L = cotan_laplacian(surface)
    nv = surface.n_points
    dpoints = np.asarray(dpoints, dtype=np.int64).ravel()
    dvals = np.asarray(dvals, dtype=float).ravel()
    u = np.zeros(nv)
    u[dpoints] = dvals
    is_d: BoolArray = np.zeros(nv, dtype=bool)
    is_d[dpoints] = True
    fpoints = np.flatnonzero(~is_d)
    Lc = L.tocsr()
    rhs = -Lc[fpoints, :][:, dpoints] @ u[dpoints]
    A = Lc[fpoints, :][:, fpoints].tocsc()
    u[fpoints] = spla.spsolve(A, np.asarray(rhs).ravel())
    return u


# -- boundary loops -----------------------------------------------------
def _boundary_edges(surface: TriMesh) -> IntArray:
    tri = surface.tris
    E = np.vstack([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]])
    E = np.sort(E, axis=1)
    Eu, ic, cnt = np.unique(E, axis=0, return_inverse=True, return_counts=True)
    return Eu[cnt == 1, :]


def boundary_loops(surface: TriMesh) -> list[IntArray]:
    """Group open-boundary edges into connected loops; returns a list of vertex-index arrays."""
    nv = surface.n_points
    bnd_edges = _boundary_edges(surface)
    bnd_verts = np.unique(bnd_edges.ravel())
    adj: list[list[int]] = [[] for _ in range(nv)]
    for i, j in bnd_edges:
        adj[i].append(j)
        adj[j].append(i)
    visited: BoolArray = np.zeros(nv, dtype=bool)
    loops = []
    for v0 in bnd_verts:
        if visited[v0]:
            continue
        comp = [v0]
        visited[v0] = True
        queue = deque([v0])
        while queue:
            point = queue.popleft()
            for w in adj[point]:
                if not visited[w]:
                    visited[w] = True
                    comp.append(w)
                    queue.append(w)
        loops.append(np.array(comp, dtype=np.int64))
    return loops


def order_boundary_loop(surface: TriMesh, lv: IntArray) -> IntArray:
    """Return the vertices of one boundary loop ``lv`` in cyclic order."""
    nv = surface.n_points
    be = _boundary_edges(surface)
    inlv: BoolArray = np.zeros(nv, dtype=bool)
    inlv[lv] = True
    be = be[inlv[be[:, 0]] & inlv[be[:, 1]], :]
    adj: dict[int, list[int]] = {}
    for a, b in be:
        adj.setdefault(a, []).append(b)
        adj.setdefault(b, []).append(a)
    start = int(lv[0])
    ordv = [start]
    prev = -1
    cur = start
    while True:
        nb = [x for x in adj[cur] if x != prev]
        if not nb or nb[0] == start:
            break
        prev = cur
        cur = nb[0]
        ordv.append(cur)
    return np.array(ordv, dtype=np.int64)


# -- isocontours --------------------------------------------------------
def extract_isocontour(surface: TriMesh, u: FloatArray, level: float) -> LineMesh | None:
    """Extract {u == level} as the largest closed ``LineMesh`` loop, or ``None``."""
    xyz, tri = surface.points, surface.tris
    nt = tri.shape[0]
    segs = np.zeros((nt, 6))
    ns = 0
    for e in range(nt):
        v = tri[e, :]
        uv = u[v]
        p = xyz[v, :]
        pts = []
        for a in range(3):
            b = (a + 1) % 3
            ua = uv[a] - level
            ub = uv[b] - level
            if (ua > 0) != (ub > 0):
                t = ua / (ua - ub)
                pts.append(p[a, :] + t * (p[b, :] - p[a, :]))
        if len(pts) == 2:
            segs[ns, :] = np.concatenate([pts[0], pts[1]])
            ns += 1
    return LineMesh.from_segments(segs[:ns, :])


def extract_rings(
    surface: TriMesh, u: FloatArray, levels: FloatArray, min_loop_pts: int,
) -> tuple[list[LineMesh], FloatArray]:
    """Cross-section rings of ``u`` at each level; returns ``(rings, levels_kept)``."""
    fr = []
    frlev = []
    for lv in levels:
        r = extract_isocontour(surface, u, lv)
        if r is None or len(r) < min_loop_pts:
            continue
        fr.append(r)
        frlev.append(lv)
    return fr, np.asarray(frlev, dtype=float)


# -- polyline / ring conformalization -----------------------------------
def _lerp_along(P: PointArray, arclen: FloatArray, targets: FloatArray) -> PointArray:
    """Piecewise-linear resample of polyline ``P`` at each query arc length in
    ``targets`` (``arclen`` = ``P``'s cumulative arc length); callers pass already
    clamped ``targets``.  Returns ``(len(targets), 3)``."""
    K = P.shape[0]
    out: PointArray = np.zeros((targets.shape[0], 3))
    for k in range(targets.shape[0]):
        s = targets[k]
        idx = int(np.flatnonzero(arclen <= s)[-1])
        idx = min(idx, K - 2)
        span = arclen[idx + 1] - arclen[idx]
        t = 0.0
        if span > 0:
            t = (s - arclen[idx]) / span
        out[k, :] = P[idx, :] + t * (P[idx + 1, :] - P[idx, :])
    return out


def resample_polyline(
    points: PointArray, fractions: FloatArray, *, closed: bool = False,
) -> PointArray:
    """Arc-length resample of a polyline at normalized ``fractions`` in ``[0, 1]``.
    With ``closed=True`` the polyline wraps (fraction ``1`` returns to the start).
    Returns ``(len(fractions), 3)``.  Use this to prepare an analytic example's curve
    at the exact sample count a factory needs -- the intrinsic interpolation of a
    scanned / analytic curve that cannot be pushed to the caller."""
    P: PointArray = np.asarray(points, dtype=float).reshape(-1, 3)
    fr: FloatArray = np.atleast_1d(np.asarray(fractions, dtype=float))
    path: PointArray = np.vstack([P, P[0:1, :]]) if closed else P
    arclen: FloatArray = np.concatenate(
        [[0.0], np.cumsum(np.sqrt(np.sum(np.diff(path, axis=0) ** 2, axis=1)))])
    targets: FloatArray = np.clip(fr, 0.0, 1.0) * arclen[-1]
    return _lerp_along(path, arclen, targets)


def _resample_loop(pts: PointArray, n: int) -> PointArray:
    """Arc-length resample of a closed loop to ``n`` points (wrapping, endpoint
    excluded)."""
    return resample_polyline(pts, np.linspace(0.0, 1.0, n, endpoint=False), closed=True)


def _align_loop(A: PointArray, B: PointArray) -> PointArray:
    """Cyclically shift (and possibly flip) closed loop ``A`` to best match ``B`` in
    least squares; returns the reordered points."""
    M = A.shape[0]
    best = np.inf
    bestA: PointArray = A
    for f in (0, 1):
        Af: PointArray = A[::-1, :].copy() if f else A
        for s in range(M):
            As: PointArray = np.roll(Af, s, axis=0)
            d = float(np.sum((As - B) ** 2))
            if d < best:
                best = d
                bestA = As
    return bestA


def _split_ring(pts: PointArray, f: float, nh: int) -> PointArray:
    """Resample closed loop ``pts`` into ``2*nh`` points so index 0 is ``pts[0]`` and
    index ``nh`` sits at arc-length fraction ``f``.  Returns ``(2*nh, 3)``."""
    R: PointArray = np.asarray(pts, dtype=float).reshape(-1, 3)
    Rc: PointArray = np.vstack([R, R[0:1, :]])
    al: FloatArray = np.concatenate(
        [[0.0], np.cumsum(np.sqrt(np.sum(np.diff(Rc, axis=0) ** 2, axis=1)))])
    tot = al[-1]
    p1 = _lerp_along(Rc, al, np.clip(np.linspace(0.0, f * tot, nh + 1), 0.0, tot))
    p2 = _lerp_along(Rc, al, np.clip(np.linspace(f * tot, tot, nh + 1), 0.0, tot))
    return np.vstack([p1[0:nh, :], p2[0:nh, :]])


def conform_ring_stack(
    fine_rings: list[LineMesh] | list[PointArray],
    seam_ring: PointArray, frlev: FloatArray, n_half: int,
) -> list[PointArray]:
    """Conformalize a stack of scanned interior rings (arbitrary per-ring point
    counts) onto the seam ring's topology.  Each fine ring is resampled by arc length
    to ``Nfine = 4*M`` points (``M`` = seam-ring point count), cyclically aligned
    back-to-front to the seam, then split at arc-length fraction
    ``f = 0.5 + (f_seam - 0.5) * frlev[k]`` into an ``M``-point ring whose index 0 /
    index ``M/2`` sit on the two seam rails.  ``f_seam`` is the seam's first-half
    fraction.  Returns a list of ``(M, 3)`` arrays."""
    seam: PointArray = np.asarray(seam_ring, dtype=float).reshape(-1, 3)
    lev: FloatArray = np.asarray(frlev, dtype=float)
    M = seam.shape[0]
    nh = n_half
    Nfine = 4 * M

    def _plen(P: PointArray) -> float:
        return float(np.sum(np.sqrt(np.sum(np.diff(P, axis=0) ** 2, axis=1))))

    La = _plen(seam[0:nh + 1, :])
    Lb = _plen(np.vstack([seam[nh:M, :], seam[0:1, :]]))
    f_seam = La / (La + Lb)

    fine_pts: list[PointArray] = [
        np.asarray(r.points if isinstance(r, LineMesh) else r, dtype=float).reshape(-1, 3)
        for r in fine_rings]
    fr: list[PointArray] = [_resample_loop(p, Nfine) for p in fine_pts]
    ref: PointArray = _resample_loop(seam, Nfine)
    for k in range(len(fr) - 1, -1, -1):
        fr[k] = _align_loop(fr[k], ref)
        ref = fr[k]

    rings: list[PointArray] = []
    for k in range(len(fr)):
        f = 0.5 + (f_seam - 0.5) * lev[k]
        rings.append(_split_ring(fr[k], f, nh))
    return rings


# -- projection ---------------------------------------------------------
def project_points(surface: TriMesh, P: PointArray) -> PointArray:
    """Snap points onto the nearest vertex's triangle fan (points assumed near the surface)."""
    P = np.atleast_2d(np.asarray(P, dtype=float))
    Vx, T = surface.points, surface.tris
    nv = Vx.shape[0]
    VT: list[list[int]] = [[] for _ in range(nv)]
    for e in range(T.shape[0]):
        VT[T[e, 0]].append(e)
        VT[T[e, 1]].append(e)
        VT[T[e, 2]].append(e)
    Q = P.copy()
    for i in range(P.shape[0]):
        p = P[i, :]
        vs = int(np.argmin(np.sum((Vx - p) ** 2, axis=1)))
        best = np.inf
        bq = Vx[vs, :]
        for e in VT[vs]:
            q = _closest_on_tri(p, Vx[T[e, 0], :], Vx[T[e, 1], :], Vx[T[e, 2], :])
            d = np.sum((q - p) ** 2)
            if d < best:
                best = d
                bq = q
        Q[i, :] = bq
    return Q


def project_to_surface(
    surface: TriMesh, P: PointArray, faces: IntArray | None = None,
) -> PointArray:
    """Closest-point projection over triangles; ``faces`` defaults to all triangles."""
    P = np.atleast_2d(np.asarray(P, dtype=float))
    Vx = surface.points
    T = surface.tris if faces is None else np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    m = P.shape[0]
    Q = P.copy()
    best = np.full(m, np.inf)
    for e in range(T.shape[0]):
        A, B, C = Vx[T[e, 0], :], Vx[T[e, 1], :], Vx[T[e, 2], :]
        q, d2 = _closest_on_tri_vec(P, A, B, C)
        upd = d2 < best
        if np.any(upd):
            best[upd] = d2[upd]
            Q[upd, :] = q[upd, :]
    return Q


# -- closest-point helpers ----------------------------------------------
def _closest_on_tri(p: Point, a: Point, b: Point, c: Point) -> Point:
    ab = b - a
    ac = c - a
    ap = p - a
    d1 = np.dot(ab, ap)
    d2 = np.dot(ac, ap)
    if d1 <= 0 and d2 <= 0:
        return a
    bp = p - b
    d3 = np.dot(ab, bp)
    d4 = np.dot(ac, bp)
    if d3 >= 0 and d4 <= d3:
        return b
    vc = d1 * d4 - d3 * d2
    if vc <= 0 and d1 >= 0 and d3 <= 0:
        return a + (d1 / (d1 - d3)) * ab
    cp = p - c
    d5 = np.dot(ab, cp)
    d6 = np.dot(ac, cp)
    if d6 >= 0 and d5 <= d6:
        return c
    vb = d5 * d2 - d1 * d6
    if vb <= 0 and d2 >= 0 and d6 <= 0:
        return a + (d2 / (d2 - d6)) * ac
    va = d3 * d6 - d5 * d4
    if va <= 0 and (d4 - d3) >= 0 and (d5 - d6) >= 0:
        w = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        return b + w * (c - b)
    den = 1.0 / (va + vb + vc)
    v = vb * den
    w = vc * den
    return a + ab * v + ac * w


def _closest_on_seg_vec(
    P: PointArray, U: Point, V: Point,
) -> tuple[PointArray, FloatArray]:
    uv = V - U
    L2 = np.dot(uv, uv)
    if L2 == 0:
        q = np.tile(U, (P.shape[0], 1))
    else:
        t = np.clip(((P - U) @ uv) / L2, 0.0, 1.0)
        q = U + t[:, None] * uv
    return q, np.sum((q - P) ** 2, axis=1)


def _closest_on_tri_vec(
    P: PointArray, A: Point, B: Point, C: Point,
) -> tuple[PointArray, FloatArray]:
    m = P.shape[0]
    INF = np.full(m, np.inf)
    qab, dab = _closest_on_seg_vec(P, A, B)
    qbc, dbc = _closest_on_seg_vec(P, B, C)
    qca, dca = _closest_on_seg_vec(P, C, A)
    ab = B - A
    ac = C - A
    n = np.cross(ab, ac)
    nn = np.dot(n, n)
    if nn > 0:
        ap = P - A
        t = (ap @ n) / nn
        qf = P - t[:, None] * n
        v0, v1, v2 = ab, ac, qf - A
        d00 = np.dot(v0, v0)
        d01 = np.dot(v0, v1)
        d11 = np.dot(v1, v1)
        d20 = v2 @ v0
        d21 = v2 @ v1
        den = d00 * d11 - d01 * d01
        v = (d11 * d20 - d01 * d21) / den
        w = (d00 * d21 - d01 * d20) / den
        u = 1 - v - w
        inside = (u >= 0) & (v >= 0) & (w >= 0)
        df = np.where(inside, np.sum((qf - P) ** 2, axis=1), INF)
    else:
        qf, df = P.copy(), INF
    q = qab.copy()
    d2 = dab.copy()
    for qc, dc in ((qbc, dbc), (qca, dca), (qf, df)):
        upd = dc < d2
        q[upd, :] = qc[upd, :]
        d2 = np.minimum(d2, dc)
    return q, d2
