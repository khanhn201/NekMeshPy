"""Tiny 2D quad-section builder with exact curved edges -> nekmeshpy QuadMesh."""
import numpy as np
from nekmeshpy.core.fields import gll_nodes
from nekmeshpy.quadmesh.quadmesh import QuadMesh
from nekmeshpy.quadmesh._helpers import entities_from_blocks
from nekmeshpy.tags.tags import Tags


class Sec:
    def __init__(self):
        self.X = []
        self.key = {}
        self.quads = []
        self.lab = []
        self.cur = {}

    def P(self, p):
        p = (float(p[0]), float(p[1]))
        k = (round(p[0], 6) + 0.0, round(p[1], 6) + 0.0)
        i = self.key.get(k)
        if i is None:
            i = len(self.X)
            self.key[k] = i
            self.X.append(p)
        return i

    def curve(self, a, b, spec):
        k = (min(a, b), max(a, b))
        old = self.cur.get(k)
        if old is not None:
            return
        self.cur[k] = (a, b, spec)

    def arc(self, a, b, c, r):
        self.curve(a, b, ("arc", float(c[0]), float(c[1]), float(r)))

    def comp(self, nodes, specs):
        """composite edge nodes[0]->nodes[-1] through waypoint coords, specs per piece."""
        self.curve(nodes[0], nodes[-1], ("comp", [np.array(self.X[n]) if isinstance(n, int) else np.asarray(n) for n in nodes], specs))

    def quad(self, ids, lab):
        ids = list(ids)
        assert len(set(ids)) == 4, ids
        x = np.array([self.X[i] for i in ids])
        a = 0.5 * np.sum(x[:, 0] * np.roll(x[:, 1], -1) - np.roll(x[:, 0], -1) * x[:, 1])
        if a < 0:
            ids = [ids[0], ids[3], ids[2], ids[1]]
        self.quads.append(ids)
        self.lab.append(lab)
        return len(self.quads) - 1

    # ---- curve evaluation -------------------------------------------------
    @staticmethod
    def _eval_spec(p0, p1, spec, t):
        if spec[0] == "line":
            return p0[None] + (p1 - p0)[None] * t[:, None]
        if spec[0] == "fn":                     # exact curve: spec[1](t) -> (len(t), 2), t from p0 to p1
            return np.asarray(spec[1](np.asarray(t, float)), float)
        if spec[0] == "arc":
            c = np.array(spec[1:3])
            r = spec[3]
            a0 = np.arctan2(*(p0 - c)[::-1])
            a1 = np.arctan2(*(p1 - c)[::-1])
            d = (a1 - a0 + np.pi) % (2 * np.pi) - np.pi
            a = a0 + d * t
            return c[None] + r * np.stack([np.cos(a), np.sin(a)], 1)
        raise ValueError(spec)

    @staticmethod
    def _spec_len(p0, p1, spec):
        if spec[0] == "line":
            return np.linalg.norm(p1 - p0)
        c = np.array(spec[1:3])
        a0 = np.arctan2(*(p0 - c)[::-1])
        a1 = np.arctan2(*(p1 - c)[::-1])
        return abs((a1 - a0 + np.pi) % (2 * np.pi) - np.pi) * spec[3]

    def eval_edge(self, a, b, t):
        k = (min(a, b), max(a, b))
        rec = self.cur.get(k)
        if rec is None:
            rec = (a, b, ("line",))
        ra, rb, spec = rec
        tt = t if (ra, rb) == (a, b) else 1.0 - t
        p0, p1 = np.array(self.X[ra]), np.array(self.X[rb])
        if spec[0] != "comp":
            out = self._eval_spec(p0, p1, spec, tt)
        else:
            W, S = spec[1], spec[2]
            L = np.array([self._spec_len(W[i], W[i + 1], S[i]) for i in range(len(S))])
            cum = np.concatenate([[0], np.cumsum(L)]) / L.sum()
            out = np.zeros((len(tt), 2))
            for i in range(len(S)):
                m = (tt >= cum[i] - 1e-15) & (tt <= cum[i + 1] + 1e-15)
                s = (tt[m] - cum[i]) / (cum[i + 1] - cum[i])
                out[m] = self._eval_spec(W[i], W[i + 1], S[i], np.clip(s, 0, 1))
        return out

    # ---- export ------------------------------------------------------------
    def blocks(self, order):
        g = gll_nodes(order)
        n = order + 1
        X = np.array(self.X)
        Q = np.array(self.quads)
        B = np.zeros((len(Q), n * n, 2))
        u, v = np.meshgrid(g, g, indexing="xy")      # u varies along i (fastest)
        u = u.ravel(); v = v.ravel()
        for q, (c0, c1, c2, c3) in enumerate(Q):
            bot = self.eval_edge(c0, c1, g)
            rig = self.eval_edge(c1, c2, g)
            top = self.eval_edge(c3, c2, g)
            lef = self.eval_edge(c0, c3, g)
            I = np.tile(np.arange(n), n); J = np.repeat(np.arange(n), n)
            P = ((1 - v)[:, None] * bot[I] + v[:, None] * top[I]
                 + (1 - u)[:, None] * lef[J] + u[:, None] * rig[J]
                 - ((1 - u) * (1 - v))[:, None] * X[c0] - (u * (1 - v))[:, None] * X[c1]
                 - (u * v)[:, None] * X[c2] - ((1 - u) * v)[:, None] * X[c3])
            B[q] = P
        return B

    def to_quadmesh(self, order, z=0.0):
        B2 = self.blocks(order)
        B = np.concatenate([B2, np.full(B2.shape[:2] + (1,), z)], 2)
        X = np.array(self.X)
        pts = np.concatenate([X, np.full((len(X), 1), z)], 1)
        Q = np.array(self.quads, dtype=np.int64)
        lm, elem_edges, flip, interior = entities_from_blocks(B, Q, pts, order)
        labs = np.array(self.lab, dtype=str)
        tags = Tags(np.arange(len(Q)), labs, len(Q))
        return QuadMesh(lm, elem_edges, flip, interior, tags)
