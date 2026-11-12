"""pipes.txt -> governors (B-rep faces of the domain) and exact distance functions.

Governors follow the paper: every face of the domain boundary is one governor.
A pipe's surface is a canal surface around its spine, so its faces are the spine's
segments (leg / bend / top), and the distance to a face is the distance to its spine
segment minus R. Box walls are planes. The plate (y = PLATE) is a cut, not a governor:
every leg meets it at a right angle, so all plate edges are type [2] and the sub-volumes
simply extend down to it.
"""
import os
import re
import numpy as np

R = 6.4135
PLATE = 1938.811
SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pipes.txt")
BELOW = 40.0          # legs are extended below the plate so nothing ends near it
MARGIN = 7.0          # box clearance from the outermost tube surface (~ tube gap)

_num = r"\(([-\d.]+), ([-\d.]+), ([-\d.]+)\)"


def parse(fn=SRC):
    pipes = {}
    cur = None
    for ln in open(fn):
        m = re.match(r"pipe (\d+)", ln)
        if m:
            cur = int(m.group(1))
            pipes[cur] = []
            continue
        s = ln.strip()
        if cur is None or not s or s.startswith("#"):
            continue
        f = [np.array(t, float) for t in re.findall(_num, s)]
        kind = s.split()[0]
        if kind == "start":
            pipes[cur].append(("start", f[0]))
        elif kind == "line":
            pipes[cur].append(("line", f[0]))
        elif kind == "ARC":
            pipes[cur].append(("arc", f[0], f[1]))
    return pipes


class Seg:
    """One spine segment = one governor face."""

    def __init__(self, gid, pipe, k, kind, p0, p1, c=None, tan0=None):
        self.gid, self.pipe, self.k, self.kind = gid, pipe, k, kind
        self.p0, self.p1 = np.asarray(p0, float), np.asarray(p1, float)
        if kind == "line":
            d = self.p1 - self.p0
            self.L = np.linalg.norm(d)
            self.t = d / self.L
        else:
            self.c = np.asarray(c, float)
            a = self.p0 - self.c
            self.r = np.linalg.norm(a)
            self.e1 = a / self.r
            t = tan0 - (tan0 @ self.e1) * self.e1
            self.e2 = t / np.linalg.norm(t)
            self.n = np.cross(self.e1, self.e2)
            b = self.p1 - self.c
            ang = np.arctan2(b @ self.e2, b @ self.e1)
            self.ang = ang if ang > 1e-9 else ang + 2 * np.pi
            self.L = self.r * self.ang

    def point(self, s):
        """spine point at arclength s (array)"""
        s = np.asarray(s, float)
        if self.kind == "line":
            return self.p0 + s[..., None] * self.t
        a = s / self.r
        return self.c + self.r * (np.cos(a)[..., None] * self.e1 + np.sin(a)[..., None] * self.e2)

    def tangent(self, s):
        s = np.asarray(s, float)
        if self.kind == "line":
            return np.broadcast_to(self.t, s.shape + (3,)).copy()
        a = s / self.r
        return -np.sin(a)[..., None] * self.e1 + np.cos(a)[..., None] * self.e2

    def foot(self, X):
        """(spine distance, arclength of foot, clamped flag) for points X (...,3)"""
        X = np.asarray(X, float)
        if self.kind == "line":
            s = (X - self.p0) @ self.t
            sc = np.clip(s, 0, self.L)
            d = np.linalg.norm(X - self.p0 - sc[..., None] * self.t, axis=-1)
            return d, sc, (s < 0) | (s > self.L)
        v = X - self.c
        x1, x2, x3 = v @ self.e1, v @ self.e2, v @ self.n
        a = np.arctan2(x2, x1)
        a = np.where(a < -1e-12, a + 2 * np.pi, a)
        inside = a <= self.ang
        # outside the arc's angular range: nearest end point
        rho = np.hypot(x1, x2)
        d_in = np.sqrt((rho - self.r) ** 2 + x3 ** 2)
        d0 = np.linalg.norm(X - self.p0, axis=-1)
        d1 = np.linalg.norm(X - self.p1, axis=-1)
        d = np.where(inside, d_in, np.minimum(d0, d1))
        s = np.where(inside, a * self.r, np.where(d0 <= d1, 0.0, self.L))
        return d, s, ~inside


def build():
    """returns segs (tube governors), pipe -> list of gids, box (lo, hi)"""
    P = parse()
    segs, of_pipe = [], {}
    for pid, items in P.items():
        first = items[1][1] - items[0][1]
        cur = items[0][1] - first / np.linalg.norm(first) * BELOW   # leg extended below the plate
        of_pipe[pid] = []
        prev_dir = first / np.linalg.norm(first)
        body = items[1:]
        for j, it in enumerate(body):
            end = it[1].copy()
            if it[0] == "line":
                if j == len(body) - 1:          # last leg: extend below the plate
                    d = end - cur
                    end = end + d / np.linalg.norm(d) * BELOW
                g = Seg(len(segs), pid, len(of_pipe[pid]), "line", cur, end)
                prev_dir = g.t
            else:
                g = Seg(len(segs), pid, len(of_pipe[pid]), "arc", cur, end, c=it[2], tan0=prev_dir)
                prev_dir = g.tangent(np.array(g.L))
            segs.append(g)
            of_pipe[pid].append(g.gid)
            cur = it[1].copy()
    # box
    allp = np.concatenate([np.concatenate([g.point(np.linspace(0, g.L, 50)) for g in segs])])
    allp = allp[allp[:, 1] >= PLATE - 1e-9]
    lo = allp.min(0) - (R + MARGIN)
    hi = allp.max(0) + (R + MARGIN)
    lo[1] = PLATE
    return segs, of_pipe, (lo, hi)


WALLS = ["x-", "x+", "z-", "z+", "y+"]


def wall_dist(X, box):
    lo, hi = box
    return np.stack([X[..., 0] - lo[0], hi[0] - X[..., 0], X[..., 2] - lo[2], hi[2] - X[..., 2],
                     hi[1] - X[..., 1]], axis=-1)


def all_dist(X, segs, box):
    """(..., G) distances to every governor: tubes (spine dist - R) then walls"""
    D = [g.foot(X)[0] - R for g in segs]
    return np.concatenate([np.stack(D, -1), wall_dist(X, box)], -1)


if __name__ == "__main__":
    segs, of_pipe, box = build()
    print(len(segs), "tube faces;", "box", box[0].round(3), box[1].round(3))
    for pid, gl in of_pipe.items():
        print(pid, [(segs[g].kind, round(segs[g].L, 2)) for g in gl])
    # sanity: continuity + tangency of consecutive segments
    for pid, gl in of_pipe.items():
        for a, b in zip(gl[:-1], gl[1:]):
            A, B = segs[a], segs[b]
            gap = np.linalg.norm(A.point(np.array(A.L)) - B.point(np.array(0.0)))
            ta, tb = A.tangent(np.array(A.L)), B.tangent(np.array(0.0))
            assert gap < 2e-2 and ta @ tb > 1 - 1e-4, (pid, gap, ta @ tb)
    print("spines continuous and G1")
