"""Hub core: the 6-way degenerate edge at the bundle centre (six legs at 34.46+-0.02 mm)
and pipe 83's focal point C* on top of it, replaced by one deformed box.

Frame: a = a83 (towards 83's first leg), n = n83 (83's plane normal), zeta = y - C*_y.
|a| <= EPS; zeta from the plate to ZTOP; n between the two lens surfaces pushed out by
0.577 (EPS - |a|)(1 - zeta/ZTOP) above zeta 0 (planes n = +-0.577 EPS below), so that
  +-a sides  face 83 (legs below the junction, bend above)
  +n         faces 81 (a > 0) / 102 (a < 0)      -n faces 102 (a > 0) / 82 (a < 0)
  top        faces 83 (the arch above), bottom on the plate.
All hub Voronoi faces end on the box edges / the n-face split lines.
"""
import numpy as np
from geom import R, PLATE
from tubes import TUBES

EPS = 8.0
ZTOP = 8.0
import os
NN = int(os.environ.get("EVG_HUBNN", 4))   # cells across n
D = lambda g, P: TUBES[g].dist(np.atleast_2d(P))[0]


def grad(g, P):
    P = np.atleast_2d(P)
    d, s = TUBES[g].dist(P)
    F = TUBES[g].spine(s)
    v = P - F
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def trace_triple(g3, p0, p1, h=0.5):
    """polyline of the Voronoi edge equidistant to the three pipes g3 from p0 to p1"""
    def proj(p):
        for _ in range(30):
            d = np.array([D(g, p)[0] for g in g3])
            G = np.array([grad(g, p)[0] for g in g3])
            f = np.array([d[0] - d[1], d[0] - d[2]])
            J = np.array([G[0] - G[1], G[0] - G[2]])
            dp = -J.T @ np.linalg.solve(J @ J.T, f)
            p = p + dp
            if np.linalg.norm(dp) < 1e-10:
                break
        return p
    P = [proj(np.asarray(p0, float))]
    end = proj(np.asarray(p1, float))
    for _ in range(4000):
        G = np.array([grad(g, P[-1])[0] for g in g3])
        t = np.cross(G[0] - G[1], G[0] - G[2]); t /= np.linalg.norm(t)
        if t @ (end - P[-1]) < 0:
            t = -t
        if np.linalg.norm(end - P[-1]) < 1.5 * h:
            P.append(end); break
        P.append(proj(P[-1] + h * t))
    return np.array(P)


class HubCore:
    def __init__(self):
        arc = [g for g in TUBES[83].segs if g.kind == "arc"][0]
        self.C = arc.c.copy()
        a = arc.e1 - (arc.e1 @ [0, 1, 0]) * np.array([0, 1, 0.])
        self.a = a / np.linalg.norm(a)
        n = np.cross(self.a, [0, 1, 0]); n /= np.linalg.norm(n)
        # orient: +n towards 81's leg at +h, +a towards 83's leg at -h (first leg)
        p81 = TUBES[81].spine(np.array([5.0]))[0] - self.C
        p83 = TUBES[83].spine(np.array([5.0]))[0] - self.C
        if p83 @ self.a < 0: self.a = -self.a
        n = np.cross(self.a, [0, 1, 0]); n /= np.linalg.norm(n)
        if p81 @ n < 0: n = -n
        self.n = n
        self.zp = PLATE - self.C[1]
        self.side = {+1: (81, 102), -1: (102, 82)}      # (a>0, a<0) governors on +-n

    def X(self, a, z, n):
        a, z, n = np.broadcast_arrays(np.asarray(a, float), np.asarray(z, float), np.asarray(n, float))
        return self.C + a[..., None] * self.a + z[..., None] * np.array([0, 1, 0.]) + n[..., None] * self.n

    def lens(self, a, z, sgn):
        """|n| of the 83 / neighbour boundary along sgn*n from (a, z, 0)"""
        a = np.atleast_1d(np.asarray(a, float)); z = np.broadcast_to(np.asarray(z, float), a.shape)
        lo = np.zeros_like(a); hi = np.full_like(a, 4 * EPS)
        nb = self.side[sgn]
        for _ in range(50):
            m = 0.5 * (lo + hi)
            P = self.X(a, z, sgn * m).reshape(-1, 3)
            f = D(83, P) - np.minimum(D(nb[0], P), D(nb[1], P))
            lo = np.where(f < 0, m, lo); hi = np.where(f < 0, hi, m)
        return 0.5 * (lo + hi)

    def nsurf(self, a, z, sgn):
        h = np.where(z <= 0, 1.0, 1.0 - np.clip(z / ZTOP, 0, 1))
        return sgn * (self.lens(a, z, sgn) + 0.577 * (EPS - np.abs(a)) * h)

    def split_a(self, z, sgn):
        """a of the line on the sgn*n surface where the two neighbours tie (60 / 240 deg face)"""
        g1, g2 = self.side[sgn]
        lo, hi = -EPS, EPS
        for _ in range(50):
            m = 0.5 * (lo + hi)
            P = self.X(m, z, self.nsurf(np.array([m]), z, sgn)[0])
            f = D(g1, P)[0] - D(g2, P)[0]          # g1 (a>0 side) closer -> f<0
            if f < 0: hi = m
            else: lo = m
        return 0.5 * (lo + hi)

    def grid(self, ma, nz1, nz2):
        """node coordinates (Na+1, NN+1, Nz+1, 3)"""
        Na = 2 * ma
        Z = np.r_[np.linspace(self.zp, 0, nz1 + 1), np.linspace(0, ZTOP, nz2 + 1)[1:]]
        G = np.empty((Na + 1, NN + 1, len(Z), 3))
        for k, z in enumerate(Z):
            rows = {}
            for sgn in (+1, -1):
                am = self.split_a(z, sgn)
                aa = np.r_[np.linspace(-EPS, am, ma + 1), np.linspace(am, EPS, ma + 1)[1:]]
                rows[sgn] = self.X(aa, z, self.nsurf(aa, z, sgn))
            for j in range(NN + 1):
                t = j / NN
                G[:, j, k] = (1 - t) * rows[-1] + t * rows[+1]
        return G, Z

    def pieces(self):
        """per face: chains of box pieces from the plate (names), for the loop builder"""
        return {
            (81, 102): [("line", +1)],
            (81, 83): [("corner", +1, +1), ("top", +1, +1)],
            (83, 102): [("corner", -1, +1), ("top", -1, +1), ("corner", +1, -1), ("top", +1, -1)],
            (82, 83): [("corner", -1, -1), ("top", -1, -1)],
            (82, 102): [("line", -1)],
        }

    def piece_ids(self, ids, name, ma, nzt):
        """node ids (from the plate / from the corner) of a named piece; ids grid (Na+1, NN+1, Nz+1)"""
        Na = 2 * ma
        if name[0] == "line":
            j = NN if name[1] > 0 else 0
            return [ids[ma, j, k] for k in range(nzt + 1)]
        if name[0] == "corner":
            i = Na if name[1] > 0 else 0
            j = NN if name[2] > 0 else 0
            return [ids[i, j, k] for k in range(nzt + 1)]
        if name[0] == "top":
            i0 = Na if name[1] > 0 else 0
            j = NN if name[2] > 0 else 0
            rng = range(i0, ma - 1, -1) if name[1] > 0 else range(0, ma + 1)
            return [ids[i, j, nzt] for i in rng]

    def build(self, nodes, ma, nz1, nz2):
        G, Z = self.grid(ma, nz1, nz2)
        Na, nzt = 2 * ma, len(Z) - 1
        ids = np.empty(G.shape[:3], int)
        for idx in np.ndindex(*G.shape[:3]):
            ids[idx] = nodes.get(("hub",) + idx, G[idx])
        hexes = []
        for i in range(Na):
            for j in range(NN):
                for k in range(nzt):
                    c = ids[i:i + 2, j:j + 2, k:k + 2]
                    hexes.append([c[0, 0, 0], c[1, 0, 0], c[1, 1, 0], c[0, 1, 0], c[0, 0, 1], c[1, 0, 1], c[1, 1, 1], c[0, 1, 1]])
        lat = []
        for k in range(nzt):
            for j in range(NN):
                lat.append(([ids[0, j, k], ids[0, j + 1, k], ids[0, j + 1, k + 1], ids[0, j, k + 1]], 83))
                lat.append(([ids[Na, j, k], ids[Na, j + 1, k], ids[Na, j + 1, k + 1], ids[Na, j, k + 1]], 83))
            for i in range(Na):
                gp = 81 if i >= ma else 102
                gm = 102 if i >= ma else 82
                lat.append(([ids[i, NN, k], ids[i + 1, NN, k], ids[i + 1, NN, k + 1], ids[i, NN, k + 1]], gp))
                lat.append(([ids[i, 0, k], ids[i + 1, 0, k], ids[i + 1, 0, k + 1], ids[i, 0, k + 1]], gm))
        for i in range(Na):
            for j in range(NN):
                lat.append(([ids[i, j, nzt], ids[i + 1, j, nzt], ids[i + 1, j + 1, nzt], ids[i, j + 1, nzt]], 83))
        pid = {}
        for f, pcs in self.pieces().items():
            for p in pcs:
                pid[p] = [int(x) for x in self.piece_ids(ids, p, ma, nzt)]
        tops = {+1: G[ma, NN, nzt], -1: G[ma, 0, nzt]}
        return dict(hex=hexes, lateral=lat, endq=[], pieces=pid, tops=tops, ids=ids, nzt=nzt)
