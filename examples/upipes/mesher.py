"""Sweep meshing of the EVG sub-volumes.

Each Voronoi face F = A|X gets a quad mesh (meshed in the (s, phi) map of one of its
tube governors, lifted onto the face by the exact ray tie).  Every quad of F then
sweeps a column of hexes to A and to X (paper sec. 5.1: a Vs sub-volume is a sweep
between its two governors).  Faces share their Voronoi-edge nodes, so the sweeps are
conformal: two sub-volumes meet on the decomposition face spanned by a Voronoi edge
and its projection on the common governor.
"""
import numpy as np, pickle
import gmsh
from geom import R, PLATE, wall_dist
from tubes import TUBES, BOX
from rays import march

import os
H = float(os.environ.get("EVG_H", 3.0))          # target medial quad size (after the split)
NL = 2           # layers from the medial surface to each governor


# ---------------------------------------------------------------- node registry
class Nodes:
    def __init__(self):
        self.key = {}
        self.X = []

    def get(self, k, x=None):
        i = self.key.get(k)
        if i is None:
            i = len(self.X)
            self.key[k] = i
            self.X.append(np.asarray(x, float))
        return i

    def arr(self):
        return np.array(self.X)


# ---------------------------------------------------------------- map helpers
def to_map(pid, X):
    """3D points -> (s, phi) of their foot on tube pid"""
    T = TUBES[pid]
    _, s = T.dist(X)
    Tg, N, B = T.frame(s)
    v = X - T.spine(s)
    phi = np.arctan2(np.einsum("ij,ij->i", v, B), np.einsum("ij,ij->i", v, N))
    return s, phi


def lift(pid, s, phi, other, iters=48):
    """map points of tube pid -> point on the bisector with `other` along the normal ray.
    Only `other` is tested, so a map point just outside the face continues the face
    smoothly instead of jumping onto a neighbouring governor's face."""
    T = TUBES[pid]
    a, n = T.surf(np.asarray(s, float), np.asarray(phi, float))
    a, n = np.atleast_2d(a), np.atleast_2d(n)
    def dother(P):
        if other >= 0:
            return TUBES[other].dist(P)[0]
        return wall_dist(P, BOX)[:, -1 - other]
    lo = np.zeros(len(a)); hi = np.full(len(a), 80.0)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        f = dother(a + mid[:, None] * n) - mid
        lo = np.where(f > 0, mid, lo); hi = np.where(f > 0, hi, mid)
    t = 0.5 * (lo + hi)
    return a + t[:, None] * n, np.full(len(a), other)


# ---------------------------------------------------------------- polylines
def resample_poly(P, n):
    L = np.r_[0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
    u = np.linspace(0, L[-1], n + 1)
    return np.stack([np.interp(u, L, P[:, k]) for k in range(3)], 1)


def ncoarse(L):
    return max(1, int(round(L / (2 * H))))


# ---------------------------------------------------------------- 2D meshing
def mesh2d(loop_uv, hc):
    """quad-dominant mesh of a polygon (fixed boundary nodes); returns (V, tris, quads)
    boundary vertices come first, in loop order."""
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("f")
    n = len(loop_uv)
    pt = [gmsh.model.geo.addPoint(u, v, 0, hc) for u, v in loop_uv]
    ln = [gmsh.model.geo.addLine(pt[i], pt[(i + 1) % n]) for i in range(n)]
    cl = gmsh.model.geo.addCurveLoop(ln)
    sf = gmsh.model.geo.addPlaneSurface([cl])
    gmsh.model.geo.synchronize()
    for l in ln:
        gmsh.model.mesh.setTransfiniteCurve(l, 2)
    gmsh.option.setNumber("Mesh.Algorithm", 8)
    gmsh.option.setNumber("Mesh.RecombineAll", 1)
    gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 1)
    gmsh.option.setNumber("Mesh.Smoothing", 5)
    gmsh.model.mesh.generate(2)
    tags, xyz, _ = gmsh.model.mesh.getNodes()
    xyz = xyz.reshape(-1, 3)[:, :2]
    tag2i = {}
    # boundary nodes: map gmsh point nodes to loop index
    V = list(loop_uv)
    for k, p in enumerate(pt):
        nt, _, _ = gmsh.model.mesh.getNodes(0, p)
        tag2i[int(nt[0])] = k
    for t, x in zip(tags, xyz):
        if int(t) not in tag2i:
            tag2i[int(t)] = len(V)
            V.append(x)
    et, _, en = gmsh.model.mesh.getElements(2, sf)
    tris, quads = [], []
    for typ, nodes in zip(et, en):
        if typ == 2:
            tris = [[tag2i[int(a)] for a in tri] for tri in nodes.reshape(-1, 3)]
        elif typ == 3:
            quads = [[tag2i[int(a)] for a in q] for q in nodes.reshape(-1, 4)]
    gmsh.finalize()
    return np.array(V), np.array(tris, int).reshape(-1, 3), np.array(quads, int).reshape(-1, 4)


def split_all_quads(V, tris, quads, nb):
    """tri -> 3 quads, quad -> 4 quads.  Returns new vertex list (2D) with markers:
    edge midpoints of boundary segments get key ('b', i) for segment i (i -> i+1 mod nb)."""
    mid = {}
    newV = list(V)
    keys = [None] * len(V)

    def emid(a, b):
        k = (min(a, b), max(a, b))
        if k not in mid:
            mid[k] = len(newV)
            newV.append(0.5 * (V[a] + V[b]))
            bk = None
            if a < nb and b < nb and ((b - a) % nb == 1 or (a - b) % nb == 1):
                i = a if (b - a) % nb == 1 else b
                bk = ("b", i)
            keys.append(bk)
        return mid[k]

    out = []
    for t in tris:
        c = len(newV); newV.append(V[t].mean(0)); keys.append(None)
        m = [emid(t[i], t[(i + 1) % 3]) for i in range(3)]
        for i in range(3):
            out.append([t[i], m[i], c, m[i - 1]])
    for q in quads:
        c = len(newV); newV.append(V[q].mean(0)); keys.append(None)
        m = [emid(q[i], q[(i + 1) % 4]) for i in range(4)]
        for i in range(4):
            out.append([q[i], m[i], c, m[i - 1]])
    return np.array(newV), np.array(out, int), keys


# ---------------------------------------------------------------- graph-based faces
GRAPH = None


def graph():
    global GRAPH
    if GRAPH is None:
        GRAPH = pickle.load(open("maps/graph.pkl", "rb"))
    return GRAPH


def edge_nodes(nodes, eid, reverse=False):
    """fine node ids along graph edge eid (shared by every face using it)"""
    edges, verts = graph()
    e = edges[eid]
    nc = ncoarse(e["L"])
    P = resample_poly(e["pts"], 2 * nc)
    ids = []
    for k in range(2 * nc + 1):
        if k == 0:
            ids.append(nodes.get(("v", e["v"][0]), verts[e["v"][0]]["X"]))
        elif k == 2 * nc:
            ids.append(nodes.get(("v", e["v"][1]), verts[e["v"][1]]["X"]))
        else:
            ids.append(nodes.get(("e", eid, k), P[k]))
    return ids[::-1] if reverse else ids


def face_loop(A, X):
    """ordered boundary of face (A, X): list of ('e', eid, reversed) / ('p', va, vb)"""
    from faces import loops
    edges, verts = graph()
    E, chains = loops(edges, verts, A, X)
    M = map_gov(A, X)
    segs = []
    for ch in chains:
        # orient chain: vertex sequence
        vs = [None]
        e0 = edges[ch[0]]
        if len(ch) > 1:
            a, b = e0["v"]
            shared = set(edges[ch[1]]["v"])
            start = a if b in shared else b
        else:
            start = e0["v"][0]
        cur = start
        seq = []
        for i in ch:
            a, b = edges[i]["v"]
            rev = a != cur
            seq.append(("e", i, rev))
            cur = a if rev else b
        segs.append((start, cur, seq))
    # plate vertices -> side in the map of M
    loop = []
    remaining = list(segs)
    first = remaining.pop(0)
    loop.extend(first[2]); cur = first[1]; start = first[0]
    guard = 0
    while cur != start and guard < 100:
        guard += 1
        nxt = [k for k, s in enumerate(remaining) if s[0] == cur or s[1] == cur]
        if nxt:
            s = remaining.pop(nxt[0])
            if s[0] == cur:
                loop.extend(s[2]); cur = s[1]
            else:
                loop.extend([(t, i, not r) for t, i, r in reversed(s[2])]); cur = s[0]
            continue
        # plate jump: partner plate vertex on the same plate side
        assert verts[cur]["plate"], ("open loop", A, X, cur)
        cand = [v for s in remaining for v in (s[0], s[1]) if verts[v]["plate"]] + [start]
        sc = side(M, verts[cur]["X"])
        cand = [v for v in cand if verts[v]["plate"] and side(M, verts[v]["X"]) == sc]
        X0 = verts[cur]["X"]
        v = min(cand, key=lambda v: np.linalg.norm(verts[v]["X"] - X0) if v != cur else 1e9)
        loop.append(("p", cur, v)); cur = v
    assert cur == start, ("loop not closed", A, X)
    return loop


def side(pid, X):
    s, _ = to_map(pid, np.atleast_2d(X))
    return int(s[0] > TUBES[pid].L / 2)


MAPGOV = {}


def map_gov(A, X):
    """tube whose map carries the face mesh: the tube governor with the larger footprint"""
    if (A, X) in MAPGOV:
        return MAPGOV[(A, X)]
    tubes = [g for g in (A, X) if 0 <= g < 1000]
    if len(tubes) == 1:
        return tubes[0]
    area = []
    from comps import clean
    for g in tubes:
        o = X if g == A else A
        M = np.load(f"maps/tube_{g}.npz")
        area.append((clean(M["lab"]) == o).sum() * (1 if g < 1000 else 0))
    return tubes[int(np.argmax(area))]


def plate_nodes(nodes, M, other, va, vb, face_key):
    """nodes of the face's trace on the plate from plate vertex va to vb (lifted map row)"""
    edges, verts = graph()
    Xa, Xb = verts[va]["X"], verts[vb]["X"]
    sa, fa = to_map(M, Xa[None]); sb, fb = to_map(M, Xb[None])
    fb = fa + (fb - fa + np.pi) % (2 * np.pi) - np.pi
    L = np.linalg.norm(Xb - Xa) * 1.1
    nc = ncoarse(L)
    s_row = 0.0 if sa[0] < TUBES[M].L / 2 else TUBES[M].L
    ph = np.linspace(fa[0], fb[0], 2 * nc + 1)
    P, _ = lift(M, np.full(len(ph), s_row), ph, other)
    ids = [nodes.get(("v", va), Xa)]
    for k in range(1, 2 * nc):
        ids.append(nodes.get(("p", face_key, va, vb, k), P[k]))
    ids.append(nodes.get(("v", vb), Xb))
    return ids


def mesh_face(nodes, A, X):
    """medial quads (global node ids) of face (A, X)"""
    M = map_gov(A, X)
    other = X if M == A else A
    loop = face_loop(A, X)
    fine = []
    for item in loop:
        if item[0] == "e":
            ids = edge_nodes(nodes, item[1], item[2])
        else:
            ids = plate_nodes(nodes, M, other, item[1], item[2], (A, X))
        fine.extend(ids[:-1])
    fine = np.array(fine)
    assert len(fine) % 2 == 0
    XYZ = nodes.arr()[fine]
    s, ph = to_map(M, XYZ)
    ph = np.unwrap(ph)
    assert abs(ph[-1] - ph[0]) < np.pi, ("face wraps around the tube", A, X)
    # scale map to the medial metric
    T = TUBES[M]
    g_phi = np.median(np.linalg.norm(np.diff(XYZ, axis=0), axis=1) /
                      np.maximum(np.hypot(np.diff(s), np.diff(ph) * (R + 4)), 1e-9))
    uv = np.stack([s, ph * (R + 4.0)], 1)
    coarse = uv[0::2]
    V, tris, quads = mesh2d([tuple(p) for p in coarse], 2 * H)
    nb = len(coarse)
    V2, Q, keys = split_all_quads(V, tris, quads, nb)
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
        Pi, lab = lift(M, V2[interior, 0], V2[interior, 1] / (R + 4.0), other)
        for i, p in zip(interior, Pi):
            gid[i] = nodes.get(("f", A, X, i), p)
    return gid[Q]
