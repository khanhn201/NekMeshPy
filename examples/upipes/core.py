"""Cores: the extension the paper leaves open (Voronoi elements with a continuum of
projection points on one entity).  Inside an innermost U the bend's rays converge on
its axis (focal rim) and the legs' rays on the self face; a sweep from the governors
would collapse there.  The core replaces that neighbourhood by a structured block:
cross-section = strip |eta| < EPS (around the self face, plate -> junction plane) +
D-template of radius EPS (around the rim), swept along the axis.  Its ends lie on the
neighbouring Voronoi faces (the 'tombstones'); its lateral surface sweeps columns to
the U's own legs/bend like any Voronoi face.
"""
import numpy as np
from geom import R, PLATE, wall_dist
from tubes import TUBES, BOX

EPS = 2.5
import os
A_, B_, C_ = (int(v) for v in os.environ.get("EVG_CORE", "1,4,4").split(","))   # D-template: diameter a+c+a, arc b+c+b
NW = 2 * A_ + C_               # strip width segments
NARC = 2 * B_ + C_


def gdist(g, P):
    if g >= 0:
        return TUBES[g].dist(P)[0]
    return wall_dist(P, BOX)[:, -1 - g]


# ---------------------------------------------------------------- 2D template
def _tfi(b, r, t, l):
    """b: bottom (n1+1,2) left->right, r: right (n2+1) bottom->top, t: top left->right, l: left bottom->top"""
    n1, n2 = len(b) - 1, len(r) - 1
    xi = np.linspace(0, 1, n1 + 1)[:, None, None]; et = np.linspace(0, 1, n2 + 1)[None, :, None]
    P = ((1 - et) * b[:, None] + et * t[:, None] + (1 - xi) * l[None] + xi * r[None]
         - (1 - xi) * (1 - et) * b[0] - xi * (1 - et) * b[-1] - (1 - xi) * et * t[0] - xi * et * t[-1])
    return P


def template(nz, zeta_p, EPS=EPS):
    """2D section: nodes (eta, zeta) with names, quads, outline (plate +eps -> up -> arc -> down -> plate -eps)"""
    names, xy = {}, []

    def node(name, p):
        if name not in names:
            names[name] = len(xy); xy.append(np.asarray(p, float))
        return names[name]

    quads = []
    eta = np.linspace(EPS, -EPS, NW + 1)               # i=0 at +EPS
    zeta = np.linspace(zeta_p, 0.0, nz + 1)
    S = np.array([[node(("s", i, j), (eta[i], zeta[j])) for j in range(nz + 1)] for i in range(NW + 1)])
    for i in range(NW):
        for j in range(nz):
            quads.append([S[i, j], S[i + 1, j], S[i + 1, j + 1], S[i, j + 1]])
    # D on top: diameter = top row of the strip
    th = lambda a: EPS * np.array([np.cos(np.radians(a)), np.sin(np.radians(a))])
    aB, aC = 55.0, 125.0
    Ap, P1, P4, Dp = (eta[0], 0), (eta[A_], 0), (eta[A_ + C_], 0), (eta[-1], 0)
    P2 = np.array([0.42 * EPS, 0.55 * EPS]); P3 = np.array([-0.42 * EPS, 0.55 * EPS])
    arc = lambda a0, a1, n: np.array([th(a) for a in np.linspace(a0, a1, n + 1)])
    lin = lambda p, q, n: np.array([np.asarray(p) + (np.asarray(q) - np.asarray(p)) * t for t in np.linspace(0, 1, n + 1)])
    blocks = [
        # name, bottom, right, top, left (TFI orientation), node keys for boundary
        ("R", lin(P1, Ap, A_), arc(0, aB, B_), lin(P2, th(aB), A_), lin(P1, P2, B_)),
        ("Q", lin(P4, P1, C_), lin(P1, P2, B_), lin(P3, P2, C_), lin(P4, P3, B_)),
        ("T", lin(P3, P2, C_), lin(P2, th(aB), A_), arc(aC, aB, C_), lin(P3, th(aC), A_)),
        ("L", lin(Dp, P4, A_), lin(P4, P3, B_), lin(th(aC), P3, A_), arc(180, aC, B_)),
    ]
    G = {}
    for nm, b, r, t, l in blocks:
        P = _tfi(b, r, t, l)
        n1, n2 = P.shape[0] - 1, P.shape[1] - 1
        ids = np.empty((n1 + 1, n2 + 1), int)
        for i in range(n1 + 1):
            for j in range(n2 + 1):
                ids[i, j] = node(("d", tuple(np.round(P[i, j], 9))), P[i, j])
        G[nm] = ids
        for i in range(n1):
            for j in range(n2):
                quads.append([ids[i, j], ids[i + 1, j], ids[i + 1, j + 1], ids[i, j + 1]])
    xy = np.array(xy)
    # unify coincident nodes (strip top row = D diameter, shared block sides)
    remap = np.arange(len(xy))
    for i in range(len(xy)):
        d = np.where(np.linalg.norm(xy[:i] - xy[i], axis=1) < 1e-7)[0]
        if len(d):
            remap[i] = remap[d[0]]
    quads = remap[np.array(quads)]
    # outline
    up = [S[0, j] for j in range(nz + 1)]
    arcpts = [th(a) for a in np.r_[np.linspace(0, aB, B_ + 1), np.linspace(aB, aC, C_ + 1)[1:], np.linspace(aC, 180, B_ + 1)[1:]]]
    arcids = [remap[np.argmin(np.linalg.norm(xy - p, axis=1))] for p in arcpts]
    down = [S[-1, j] for j in range(nz, -1, -1)]
    outline = up + arcids[1:-1] + down
    used = np.unique(quads)
    return xy, quads, dict(up=up, arc=arcids, down=down, outline=outline), S


# ---------------------------------------------------------------- core in 3D
class Core:
    """hoops: pipes along one axis (in order); ends: (governor before, governor after)"""

    def __init__(self, cid, hoops, ends, xi_start=None, eps=EPS):
        self.cid, self.hoops, self.ends = cid, hoops, ends
        self.xi_start = xi_start
        self.eps = eps
        arcs = [[g for g in TUBES[p].segs if g.kind == "arc"][0] for p in hoops]
        self.C = arcs[0].c.copy()
        if len(hoops) > 1:
            B = arcs[-1].c - arcs[0].c
        else:
            B = arcs[0].n.copy()
            # orient B from end[0] side to end[1] side
        B = B - B[1] * np.array([0, 1.0, 0])        # axes are horizontal (plate-parallel)
        self.B = B / np.linalg.norm(B)
        self.e = np.cross(self.B, [0, 1, 0]); self.e /= np.linalg.norm(self.e)
        self.yJ = self.C[1]
        self.zeta_p = PLATE - self.yJ
        self.xi_h = [(a.c - self.C) @ self.B for a in arcs]       # hoop planes along the axis

    def pt(self, xi, eta, zeta):
        return self.C + np.outer(xi, self.B) + np.outer(eta, self.e) + np.outer(zeta, [0, 1, 0])

    def tie(self, H, X, eta, zeta, xi0, xi1, it=50):
        """xi in [xi0, xi1] where d_H = d_X (H closer at xi0)"""
        eta, zeta = np.asarray(eta, float), np.asarray(zeta, float)
        lo = np.full(len(eta), xi0, float); hi = np.full(len(eta), xi1, float)
        f1 = gdist(H, self.pt(hi, eta, zeta)) - gdist(X, self.pt(hi, eta, zeta))
        assert (f1 > 0).all(), ("no tie", self.cid, H, X)
        for _ in range(it):
            m = 0.5 * (lo + hi)
            f = gdist(H, self.pt(m, eta, zeta)) - gdist(X, self.pt(m, eta, zeta))
            lo = np.where(f < 0, m, lo); hi = np.where(f < 0, hi, m)
        return 0.5 * (lo + hi)

    def anchors_xi(self, eta, zeta):
        """xi of the anchor sections for template points: [end0, (mid...), end1]"""
        H = self.hoops
        out = []
        x0 = self.xi_h[0] if self.xi_start is None else self.xi_start
        out.append(self.tie(H[0], self.ends[0], eta, zeta, x0, x0 - 40.0))
        for k in range(len(H) - 1):
            out.append(self.tie(H[k], H[k + 1], eta, zeta, self.xi_h[k], self.xi_h[k + 1]))
        x1 = self.xi_h[-1] if self.xi_start is None else self.xi_start
        out.append(self.tie(H[-1], self.ends[1], eta, zeta, x1, x1 + 40.0))
        return out

    def outline_geom(self, anchor, n=60):
        """3D polylines (up, arc, down) of the tombstone at anchor index"""
        zeta = np.linspace(self.zeta_p, 0, n + 1)
        a = np.radians(np.linspace(0, 180, n + 1))
        parts = {}
        for nm, (et, ze) in dict(up=(np.full(n + 1, self.eps), zeta),
                                  arc=(self.eps * np.cos(a), self.eps * np.sin(a)),
                                  down=(np.full(n + 1, -self.eps), zeta[::-1])).items():
            xi = self.anchors_xi(et, ze)[anchor]
            parts[nm] = self.pt(xi, et, ze)
        return parts

    def build(self, nodes, nz, h=3.0):
        """nodes: Nodes registry.  Returns dict(hex=..., lateral=[(quad, gov)], endq=[(quad, gov)],
        outline={anchor: ids})"""
        xy, quads, OL, S = template(nz, self.zeta_p, self.eps)
        eta, zeta = xy[:, 0], xy[:, 1]
        anc = self.anchors_xi(eta, zeta)                      # list of (nnode,)
        # sections: anchors + interior sections in every span
        secs, gov_span = [], []
        for k in range(len(anc) - 1):
            L = np.median(anc[k + 1] - anc[k])
            ns = max(2, 2 * int(round(abs(L) / (2 * h))))
            for j in range(ns):
                secs.append(anc[k] + (anc[k + 1] - anc[k]) * j / ns)
                gov_span.append(self.hoops[k])
        secs.append(anc[-1])
        anchor_sec = [0]
        for k in range(len(anc) - 1):
            L = np.median(anc[k + 1] - anc[k]); anchor_sec.append(anchor_sec[-1] + max(2, 2 * int(round(abs(L) / (2 * h)))))
        ids = np.empty((len(secs), len(xy)), int)
        for s, xi in enumerate(secs):
            P = self.pt(xi, eta, zeta)
            a = anchor_sec.index(s) if s in anchor_sec else None
            for i in range(len(xy)):
                key = ("core", self.cid, "a", a, i) if a is not None else ("core", self.cid, s, i)
                ids[s, i] = nodes.get(key, P[i])
        hexes = []
        for s in range(len(secs) - 1):
            for q in quads:
                hexes.append(list(ids[s, q]) + list(ids[s + 1, q]))
        lateral = []
        ol = OL["outline"]
        for s in range(len(secs) - 1):
            for j in range(len(ol) - 1):
                lateral.append(([ids[s, ol[j]], ids[s, ol[j + 1]], ids[s + 1, ol[j + 1]], ids[s + 1, ol[j]]], gov_span[s]))
        endq = []
        for s, g in ((0, self.ends[0]), (len(secs) - 1, self.ends[1])):
            for q in quads:
                endq.append((list(ids[s, q]), g))
        outline = {a: [int(ids[anchor_sec[a], i]) for i in ol] for a in range(len(anchor_sec))}
        return dict(hex=hexes, lateral=lateral, endq=endq, outline=outline, nsec=len(secs))
