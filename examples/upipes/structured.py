"""Interval assignment (ILP) and structured (TFI) meshing of 4-sided faces."""
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
import mesher
from mesher import graph, map_gov, to_map, lift, resample_poly, R, Nodes
from corners import loop_sides
import items as IT

RHO = R + 4.0


item_key = IT.key
item_length = IT.length


def side_polyline(side, A, X):
    """map polyline of a side and the arclength positions of its internal item breaks"""
    M = map_gov(A, X)
    P, breaks = [], []
    for it in side:
        Q = IT.pts(it)
        P.extend(Q if not P else Q[1:])
        breaks.append(len(P) - 1)
    P = np.array(P)
    s, ph = to_map(M, P)
    ph = np.unwrap(ph)
    uv = np.stack([s, ph * RHO], 1)
    L = np.r_[0, np.cumsum(np.linalg.norm(np.diff(uv, axis=0), axis=1))]
    return uv, L, [L[b] for b in breaks[:-1]]


def assign(faces_sides, H, fixed=None, w_align=2.0):
    """interval assignment: opposite sides of structured faces equal; internal side
    vertices aligned (soft) with their projection on the opposite side.
    faces_sides: {face: (sides or None, loop)} -> {item_key: fine count (even)}"""
    keys, L = {}, {}
    for f, (sides, loop) in faces_sides.items():
        for it in loop:
            k = item_key(it, f)
            if k not in keys:
                keys[k] = len(keys); L[k] = item_length(it)
    n = len(keys)
    hard, soft = [], []                 # rows over m (coarse counts)
    for f, (sides, loop) in faces_sides.items():
        if sides is None:
            continue
        for a, b in ((0, 2), (1, 3)):
            r = np.zeros(n)
            for it in sides[a]: r[keys[item_key(it, f)]] += 1
            for it in sides[b]: r[keys[item_key(it, f)]] -= 1
            hard.append(r)
            # alignment of internal breaks of side b against side a (and vice versa)
            for s1, s2 in ((a, b), (b, a)):
                if len(sides[s2]) < 2:
                    continue
                uv1, L1, _ = side_polyline(sides[s1], *f)
                uv2, L2, br = side_polyline(sides[s2], *f)
                for j, lb in enumerate(br):
                    # point on side s2 at break j; project onto s1 (opposite direction)
                    p = uv2[np.searchsorted(L2, lb)]
                    k = np.argmin(np.linalg.norm(uv1 - p, axis=1))
                    tau = 1.0 - L1[k] / L1[-1]          # s1 runs the other way round
                    r = np.zeros(n)
                    for it in sides[s2][:j + 1]: r[keys[item_key(it, f)]] += 1
                    for it in sides[s1]: r[keys[item_key(it, f)]] -= tau
                    soft.append(r)
    import os
    HC = float(os.environ.get("EVG_HCORE", H))
    HF = {tuple(int(g) for g in kv.split(":")[0].split("-")): float(kv.split(":")[1])
          for kv in os.environ.get("EVG_HFACE", "").split(",") if kv}
    Hk = {}
    for f, (sides, loop) in faces_sides.items():
        if f[:2] in HF:
            for it in loop:
                k = item_key(it, f)
                Hk[k] = min(Hk.get(k, H), HF[f[:2]])
    target = np.array([L[k] / (2 * (HC if k[0] in ("oz",) else Hk.get(k, H))) for k in keys])
    ns = len(soft)
    nv = 2 * n + ns
    A_rows, lo, hi = [], [], []
    for r in hard:
        A_rows.append(np.r_[r, np.zeros(n + ns)]); lo.append(0); hi.append(0)
    for i in range(n):
        r = np.zeros(nv); r[i] = 1; r[n + i] = -1; A_rows.append(r); lo.append(-np.inf); hi.append(target[i])
        r = np.zeros(nv); r[i] = -1; r[n + i] = -1; A_rows.append(r); lo.append(-np.inf); hi.append(-target[i])
    for j, r0 in enumerate(soft):
        r = np.r_[r0, np.zeros(n + ns)]; r[2 * n + j] = -1; A_rows.append(r); lo.append(-np.inf); hi.append(0)
        r = np.r_[-r0, np.zeros(n + ns)]; r[2 * n + j] = -1; A_rows.append(r); lo.append(-np.inf); hi.append(0)
    if fixed:
        for k, v in fixed.items():
            if k in keys:
                r = np.zeros(nv); r[keys[k]] = 1; A_rows.append(r); lo.append(v / 2); hi.append(v / 2)
    c = np.r_[np.zeros(n), 1.0 / np.maximum(target, 0.5), np.full(ns, w_align)]
    integrality = np.r_[np.ones(n), np.zeros(n + ns)]
    res = milp(c, constraints=LinearConstraint(np.array(A_rows), lo, hi), integrality=integrality,
               bounds=Bounds(np.r_[np.ones(n), np.zeros(n + ns)], np.full(nv, 1e4)),
               options=dict(time_limit=120))
    assert res.x is not None, res.message
    m = np.round(res.x[:n]).astype(int)
    out = {k: 2 * int(m[i]) for k, i in keys.items()}
    dev = {k: out[k] / max(L[k] / H, 1e-9) for k in keys}
    return out, dev


item_nodes = IT.node_ids


def tfi_face(nodes, A, X, sides, counts):
    """structured quads of a 4-sided face"""
    M = map_gov(A, X)
    other = X if M == A else A
    f = (A, X)
    S = []
    for side in sides:
        ids = []
        for it in side:
            nn = item_nodes(nodes, it, counts[item_key(it, f)], M, other, f)
            ids.extend(nn if not ids else nn[1:])
        S.append(np.array(ids))
    n1, n2 = len(S[0]) - 1, len(S[1]) - 1
    assert len(S[2]) - 1 == n1 and len(S[3]) - 1 == n2, ("unbalanced", A, X, [len(s) - 1 for s in S])
    bottom, right, top, left = S[0], S[1], S[2][::-1], S[3][::-1]
    XYZ = nodes.arr()
    allid = np.concatenate([bottom, right, top, left])
    s, ph = to_map(M, XYZ[allid])
    ph = np.unwrap(ph)
    uv_all = np.stack([s, ph * RHO], 1)
    o = np.cumsum([0, len(bottom), len(right), len(top)])
    B, Rr, T, Lf = (uv_all[o[0]:o[1]], uv_all[o[1]:o[2]], uv_all[o[2]:o[3]], uv_all[o[3]:])
    def frac(P):
        d = np.r_[0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
        return d / d[-1]
    xb, xt = frac(B), frac(T); yl, yr = frac(Lf), frac(Rr)
    grid = np.empty((n1 + 1, n2 + 1), int)
    grid[:, 0] = bottom; grid[:, -1] = top; grid[0, :] = left; grid[-1, :] = right
    I, J = np.meshgrid(np.arange(1, n1), np.arange(1, n2), indexing="ij")
    if I.size:
        # blended parameters (Coons with arclength mixing)
        eta0 = (1 - xb[I]) * yl[J] + xb[I] * yr[J]
        xi0 = (1 - yl[J]) * xb[I] + yl[J] * xt[I]
        xi, eta = xi0, eta0
        P = ((1 - eta)[..., None] * B[I] + eta[..., None] * T[I]
             + (1 - xi)[..., None] * Lf[J] + xi[..., None] * Rr[J]
             - ((1 - xi) * (1 - eta))[..., None] * B[0] - (xi * (1 - eta))[..., None] * B[-1]
             - ((1 - xi) * eta)[..., None] * T[0] - (xi * eta)[..., None] * T[-1])
        Pl, lab = lift(M, P[..., 0].ravel(), P[..., 1].ravel() / RHO, other)
        for (i, j), p in zip(zip(I.ravel(), J.ravel()), Pl):
            grid[i, j] = nodes.get(("t", A, X, M, i, j), p)
    Q = np.stack([grid[:-1, :-1], grid[1:, :-1], grid[1:, 1:], grid[:-1, 1:]], -1).reshape(-1, 4)
    return Q


def smooth2d(V, Q, fixed, iters=40):
    """constrained Laplacian on a planar quad mesh: a move is kept only if every incident
    quad stays strictly convex (same orientation as the mesh)"""
    from collections import defaultdict
    V = V.copy()
    nb = defaultdict(set); inc = defaultdict(list)
    for k, q in enumerate(Q):
        for a in range(4):
            nb[q[a]].add(q[(a + 1) % 4]); nb[q[(a + 1) % 4]].add(q[a]); inc[q[a]].append(k)
    def turns(qq, P):
        Pq = P[Q[qq]]
        e1 = np.roll(Pq, -1, 1) - Pq; e2 = np.roll(Pq, 1, 1) - Pq
        return e1[..., 0] * e2[..., 1] - e1[..., 1] * e2[..., 0]
    allt = turns(np.arange(len(Q)), V)
    sg = np.sign(np.median(allt))
    free = [i for i in range(len(V)) if not fixed[i] and i in nb]
    nbl = {i: np.array(sorted(nb[i])) for i in free}
    incl = {i: np.array(inc[i]) for i in free}
    for _ in range(iters):
        moved = 0
        for i in free:
            t = V[nbl[i]].mean(0)
            old = V[i].copy()
            for w in (0.5, 0.2):
                V[i] = old + w * (t - old)
                if (sg * turns(incl[i], V) > 0).all():
                    moved += 1; break
                V[i] = old
        if not moved:
            break
    return V


def unstructured_face(nodes, A, X, loop, counts, H, fkey=None):
    """gmsh quad-dominant mesh of the loop polygon in the map, split to all-quad"""
    from mesher import mesh2d, split_all_quads
    M = map_gov(A, X)
    other = X if M == A else A
    f = (A, X) if fkey is None else fkey
    fine = []
    for it in loop:
        ids = item_nodes(nodes, it, counts[item_key(it, f)], M, other, f)
        fine.extend(ids[:-1])
    fine = np.array(fine)
    assert len(fine) % 2 == 0
    XYZ = nodes.arr()[fine]
    s, ph = to_map(M, XYZ)
    ph = np.unwrap(ph)
    uv = np.stack([s, ph * RHO], 1)
    coarse = uv[0::2]
    V, tris, quads = mesh_coarse(coarse, 2 * H)
    nb = len(coarse)
    V2, Q, keys = split_all_quads(V, tris, quads, nb)
    bnd = np.zeros(len(V2), bool); bnd[:nb] = True
    for i in range(nb, len(V2)):
        if keys[i] is not None and keys[i][0] == "b":
            bnd[i] = True
    V2 = smooth2d(V2, Q, bnd)
    gid = np.empty(len(V2), int)
    gid[:nb] = fine[0::2]
    interior = []
    for i in range(nb, len(V2)):
        k = keys[i]
        if k is not None and k[0] == "b":
            gid[i] = fine[2 * k[1] + 1]
        else:
            interior.append(i)
    interior = np.array(interior, int)
    if len(interior):
        Pi, lab = lift(M, V2[interior, 0], V2[interior, 1] / RHO, other)
        for i, p in zip(interior, Pi):
            gid[i] = nodes.get(("u", f, M, i), p)
    return gid[Q]


def self_intersections(P):
    """number of crossing pairs among the closed polygon's segments"""
    A = P; B = np.roll(P, -1, 0); n = len(P)
    d1 = B - A
    cnt = 0
    for i in range(n):
        j = np.arange(i + 2, n)
        if i == 0:
            j = j[j != n - 1]
        if not len(j):
            continue
        r, s_ = d1[i], d1[j]
        den = r[0] * s_[:, 1] - r[1] * s_[:, 0]
        q = A[j] - A[i]
        ok = np.abs(den) > 1e-14
        t = np.where(ok, (q[:, 0] * s_[:, 1] - q[:, 1] * s_[:, 0]) / np.where(ok, den, 1), -1)
        u = np.where(ok, (q[:, 0] * r[1] - q[:, 1] * r[0]) / np.where(ok, den, 1), -1)
        cnt += int(((t > 1e-9) & (t < 1 - 1e-9) & (u > 1e-9) & (u < 1 - 1e-9)).sum())
    return cnt


def mesh_coarse(coarse, hc):
    """size field: boundary spacing near the boundary, hc inside"""
    import gmsh
    nx = self_intersections(np.asarray(coarse))
    assert nx == 0, ("self-intersecting face polygon in the map", nx)
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("f")
    n = len(coarse)
    seg = np.linalg.norm(np.roll(coarse, -1, 0) - coarse, axis=1)
    loc = np.minimum(np.minimum(seg, np.roll(seg, 1)), hc)
    pt = [gmsh.model.geo.addPoint(u, v, 0, l) for (u, v), l in zip(coarse, loc)]
    ln = [gmsh.model.geo.addLine(pt[i], pt[(i + 1) % n]) for i in range(n)]
    cl = gmsh.model.geo.addCurveLoop(ln)
    sf = gmsh.model.geo.addPlaneSurface([cl])
    gmsh.model.geo.synchronize()
    for l in ln:
        gmsh.model.mesh.setTransfiniteCurve(l, 2)
    gmsh.option.setNumber("Mesh.Algorithm", 6)
    gmsh.option.setNumber("Mesh.RecombineAll", 1)
    gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 1)
    gmsh.option.setNumber("Mesh.Smoothing", 10)
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 1)
    gmsh.option.setNumber("Mesh.MeshSizeMax", hc)
    gmsh.model.mesh.generate(2)
    tags, xyz, _ = gmsh.model.mesh.getNodes()
    xyz = xyz.reshape(-1, 3)[:, :2]
    tag2i = {}
    V = [np.asarray(c) for c in coarse]
    for k, p in enumerate(pt):
        nt, _, _ = gmsh.model.mesh.getNodes(0, p)
        tag2i[int(nt[0])] = k
    for t, x in zip(tags, xyz):
        if int(t) not in tag2i:
            tag2i[int(t)] = len(V); V.append(x)
    et, _, en = gmsh.model.mesh.getElements(2, sf)
    tris, quads = [], []
    for typ, nd in zip(et, en):
        if typ == 2:
            tris = [[tag2i[int(a)] for a in t] for t in nd.reshape(-1, 3)]
        elif typ == 3:
            quads = [[tag2i[int(a)] for a in q] for q in nd.reshape(-1, 4)]
    gmsh.finalize()
    return np.array(V), np.array(tris, int).reshape(-1, 3), np.array(quads, int).reshape(-1, 4)
