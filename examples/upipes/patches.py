"""Structured (gmsh-free) meshing of the Voronoi faces that are not plain 4-sided.

A face is split, in its 2D map, into 4-sided patches:
  * a reflex corner is cut by continuing one of its two sides into the face (the shorter
    of the two rays to the far boundary);
  * a hairpin item (a core tombstone seen edge-on: a slot entering the face) is cut from
    its middle node along the slot axis to the far boundary;
  * a 3-cornered face (a wedge) gets the 3-patch template: a centre point Q near the
    sharpest corner and spokes to one split point on each side.
Every boundary piece is a whole loop item, a part of one (split point inside it), or a cut.
The ILP (``assign2``) works in final counts, with parts summing to their item and opposite
sides of every patch equal; ``mesh_face`` lays the nodes and fills each patch by TFI.
"""
import os
import numpy as np
import items as IT
import mesher
from mesher import to_map, lift, TUBES

RHO = mesher.R + 4.0
THR = 35.0


# ---------------------------------------------------------------- geometry in the map
def _uv(M, P):
    s, ph = to_map(M, P)
    return s, np.unwrap(ph)


def face_geometry(loop, M):
    """per item: 3D polyline, 3D arclength fraction, map polyline (consistently unwrapped)"""
    geo, last = [], None
    for it in loop:
        P = IT.pts(it)
        s, ph = _uv(M, P)
        if last is not None:
            ph = ph + 2 * np.pi * np.round((last - ph[0]) / (2 * np.pi))
        last = ph[-1]
        uv = np.stack([s, ph * RHO], 1)
        L3 = np.r_[0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
        geo.append(dict(P=P, u=L3 / max(L3[-1], 1e-12), uv=uv, L=L3[-1]))
    return geo


def _sub_uv(g, u0, u1, n=24):
    """map polyline of the part [u0, u1] (3D arclength fractions) of an item"""
    t = np.linspace(u0, u1, n)
    return np.stack([np.interp(t, g["u"], g["uv"][:, k]) for k in range(2)], 1)


def _end_dirs(uv):
    L = np.r_[0, np.cumsum(np.linalg.norm(np.diff(uv, axis=0), axis=1))]
    w = min(6.0, L[-1] / 3)
    k0 = int(np.clip(np.searchsorted(L, w), 1, len(uv) - 1))
    k1 = int(np.clip(np.searchsorted(L, L[-1] - w) - 1, 0, len(uv) - 2))
    d0 = uv[k0] - uv[0]; d1 = uv[-1] - uv[k1]
    return d0 / np.linalg.norm(d0), d1 / np.linalg.norm(d1)


def _turn(d_in, d_out):
    return np.degrees(np.arctan2(d_in[0] * d_out[1] - d_in[1] * d_out[0], d_in @ d_out))


def _internal_turn(uv):
    d = np.diff(uv, axis=0)
    d = d[np.linalg.norm(d, axis=1) > 1e-9]
    a = np.arctan2(d[:, 1], d[:, 0])
    return np.degrees(np.sum((np.diff(a) + np.pi) % (2 * np.pi) - np.pi))


def _ray_hit(o, d, segs):
    """first hit of ray o + t d (t > 1e-6) with segments [(A, B, tag, ua, ub)]"""
    best = None
    for A, B, tag, ua, ub in segs:
        e = B - A
        den = d[0] * e[1] - d[1] * e[0]
        if abs(den) < 1e-12:
            continue
        q = A - o
        t = (q[0] * e[1] - q[1] * e[0]) / den
        s = (q[0] * d[1] - q[1] * d[0]) / den
        if t > 1e-6 and -1e-9 <= s <= 1 + 1e-9 and (best is None or t < best[0]):
            best = (t, tag, ua + s * (ub - ua), o + t * d)
    return best


def _item_segments(geo, skip):
    out = []
    for i, g in enumerate(geo):
        if i in skip:
            continue
        uv, u = g["uv"], g["u"]
        for k in range(len(uv) - 1):
            out.append((uv[k], uv[k + 1], i, u[k], u[k + 1]))
    return out


# ---------------------------------------------------------------- decomposition
class Face:
    """the patch decomposition of one face in map M"""

    def __init__(self, f, loop, M):
        self.f, self.loop, self.M = f, loop, M
        self.geo = face_geometry(loop, M)
        allp = np.concatenate([g["uv"][:-1] for g in self.geo])
        a = 0.5 * np.sum(allp[:, 0] * np.roll(allp[:, 1], -1) - np.roll(allp[:, 0], -1) * allp[:, 1])
        self.sg = np.sign(a)
        self.splits = {i: [] for i in range(len(loop))}     # item -> sorted split fractions
        self.fixed_split = {}                               # (item, u) -> required equal halves
        self.cuts = {}                                      # cid -> dict(a=pt, b=pt, uv=(2,2) or None)
        self.centres = {}
        self.smooth = set()                                 # points that are never corners
        self.patches = []                                   # lists of 4 sides of pieces
        self.kind = None
        self.thr = THR
        self.pairs = []
        self.csplits = {}

    # points: ("b", i) start of item i;  ("s", i, u) inside item i;  ("Q", name)
    def pt_uv(self, p):
        if p[0] == "cs":
            return self.cut_uv(p[1], p[2])
        if p[0] == "b":
            return self.geo[p[1]]["uv"][0]
        if p[0] == "s":
            g = self.geo[p[1]]
            return np.array([np.interp(p[2], g["u"], g["uv"][:, k]) for k in range(2)])
        return self.centres[p[1]]

    def corners(self):
        n = len(self.loop)
        dirs = [_end_dirs(g["uv"]) for g in self.geo]
        t = np.array([_turn(dirs[i - 1][1], dirs[i][0]) for i in range(n)])
        return t

    def add_split(self, i, u):
        u = float(np.clip(u, 1e-6, 1 - 1e-6))
        self.splits[i].append(u)
        return ("s", i, u)

    # -------------------------------------------------------------- cutting
    def plan(self):
        n = len(self.loop)
        t = self.corners() * self.sg
        conv = [i for i in range(n) if t[i] > THR]
        refl = [i for i in range(n) if t[i] < -THR]
        hair = [i for i in range(n) if abs(_internal_turn(self.geo[i]["uv"])) > 120.0]
        if not refl and not hair and len(conv) not in (3, 4):
            conv62 = [i for i in range(n) if t[i] > 62.0]
            if len(conv62) == 4:
                conv = conv62
                self.thr = 62.0
        if not refl and not hair:
            if len(conv) == 3:
                self.kind = "tri"
                self._triangle(conv, t)
                return
            assert len(conv) == 4, ("corners", len(conv))
            self.kind = "quad"
        else:
            self.kind = "cut"
        dirs = [_end_dirs(g["uv"]) for g in self.geo]
        for i in refl:
            o = self.geo[i]["uv"][0]
            segs = _item_segments(self.geo, {(i - 1) % n, i})
            cand = []
            for d in (dirs[i - 1][1], -dirs[i][0]):
                h = _ray_hit(o, d, segs)
                if h is not None:
                    cand.append(h)
            assert cand, ("reflex corner without a hit", i)
            tt, j, u, X = min(cand, key=lambda c: c[0])
            b = self.add_split(j, u)
            self.cuts[len(self.cuts)] = dict(a=("b", i), b=b)
        for i in hair:
            g = self.geo[i]
            tip = self.add_split(i, 0.5)
            self.fixed_split[(i, 0.5)] = True
            self.smooth.add(tip)
            o = self.pt_uv(tip)
            d = o - 0.5 * (g["uv"][0] + g["uv"][-1]); d /= np.linalg.norm(d)
            h = _ray_hit(o, d, _item_segments(self.geo, {i}))
            assert h is not None, ("hairpin without a hit", i)
            tt, j, u, X = h
            b = self.add_split(j, u)
            self.cuts[len(self.cuts)] = dict(a=tip, b=b)
        self._split_polygons()

    def _pieces(self):
        """boundary pieces in loop order and the point at the start of each"""
        pcs, pts = [], []
        for i in range(len(self.loop)):
            us = [0.0] + sorted(self.splits[i]) + [1.0]
            for k in range(len(us) - 1):
                pcs.append(("i", i, us[k], us[k + 1]))
                pts.append(("b", i) if k == 0 else ("s", i, us[k]))
        return pcs, pts

    def cut_uv(self, cid, u):
        c = self.cuts[cid]
        A, B = self.pt_uv(c["a"]), self.pt_uv(c["b"])
        return A + u * (B - A)

    def piece_uv(self, pc):
        if pc[0] == "i":
            return _sub_uv(self.geo[pc[1]], pc[2], pc[3])
        a, b = self.cut_uv(pc[1], pc[3]), self.cut_uv(pc[1], pc[4])
        return np.array([b, a]) if pc[2] else np.array([a, b])

    def _split_polygons(self):
        pcs, pts = self._pieces()
        polys = [(pcs, pts)]
        for cid, c in self.cuts.items():
            for k, (P, S) in enumerate(polys):
                if c["a"] in S and c["b"] in S:
                    ia, ib = S.index(c["a"]), S.index(c["b"])
                    n = len(P)
                    r1 = [(ia + m) % n for m in range((ib - ia) % n)]
                    r2 = [(ib + m) % n for m in range((ia - ib) % n)]
                    p1 = ([P[m] for m in r1] + [("c", cid, True, 0.0, 1.0)], [S[m] for m in r1] + [c["b"]])
                    p2 = ([P[m] for m in r2] + [("c", cid, False, 0.0, 1.0)], [S[m] for m in r2] + [c["a"]])
                    polys[k:k + 1] = [p1, p2]
                    break
            else:
                raise AssertionError(("cut endpoints not on one polygon", cid))
        for P, S in polys:
            uvs = [self.piece_uv(pc) for pc in P]
            dirs = [_end_dirs(u) if len(u) > 2 else ((u[1] - u[0]) / np.linalg.norm(u[1] - u[0]),) * 2 for u in uvs]
            n = len(P)
            t = np.array([_turn(dirs[m - 1][1], dirs[m][0]) for m in range(n)])
            cor = [m for m in range(n) if abs(t[m]) > self.thr and S[m] not in self.smooth]
            assert len(cor) == 4, ("patch with %d corners" % len(cor), [round(x) for x in t])
            sides = []
            for j in range(4):
                a, b = cor[j], cor[(j + 1) % 4]
                rng = range(a, b) if b > a else list(range(a, n)) + list(range(0, b))
                sides.append([P[m] for m in rng])
            self.patches.append(sides)

    # -------------------------------------------------------------- wedge template
    def _triangle(self, conv, t):
        n = len(self.loop)
        tip_k = int(np.argmax([abs(t[i]) for i in conv]))
        c = conv[tip_k:] + conv[:tip_k]                    # c[0] = tip (start of the item after it)
        side = lambda a, b: list(range(a, b)) if b > a else list(range(a, n)) + list(range(0, b))
        S_out, S_base, S_in = side(c[0], c[1]), side(c[1], c[2]), side(c[2], c[0])
        L = lambda items: sum(self.geo[i]["L"] for i in items)
        Lb = L(S_base)
        def split_at(items, dist_from_start):
            acc = 0.0
            for i in items:
                if acc + self.geo[i]["L"] >= dist_from_start:
                    return self.add_split(i, (dist_from_start - acc) / self.geo[i]["L"])
                acc += self.geo[i]["L"]
            return self.add_split(items[-1], 0.999)
        tip_len_out = min(0.6 * Lb, 0.3 * L(S_out))
        tip_len_in = min(0.6 * Lb, 0.3 * L(S_in))
        m_out = split_at(S_out, tip_len_out)
        m_in = split_at(S_in, L(S_in) - tip_len_in)
        m_base = split_at(S_base, 0.5 * Lb)
        T = self.geo[c[0]]["uv"][0]
        Mo, Mi = self.pt_uv(m_out), self.pt_uv(m_in)
        Mm = 0.5 * (Mo + Mi)
        ax = (Mm - T) / np.linalg.norm(Mm - T)
        Q = Mm + 0.5 * np.linalg.norm(Mo - Mi) * ax
        self.centres["Q"] = Q
        q = ("Q", "Q")
        self.cuts = {0: dict(a=q, b=m_in), 1: dict(a=q, b=m_out), 2: dict(a=q, b=m_base)}
        pcs, pts = self._pieces()
        def run(p_from, p_to):
            i0, i1 = pts.index(p_from), pts.index(p_to)
            m = len(pcs)
            return [pcs[(i0 + k) % m] for k in range((i1 - i0) % m)]
        Tp = ("b", c[0])
        V1, V2 = ("b", c[1]), ("b", c[2])
        self.patches = [
            [run(m_in, Tp), run(Tp, m_out), [("c", 1, True, 0.0, 1.0)], [("c", 0, False, 0.0, 1.0)]],
            [run(m_out, V1), run(V1, m_base), [("c", 2, True, 0.0, 1.0)], [("c", 1, False, 0.0, 1.0)]],
            [run(m_base, V2), run(V2, m_in), [("c", 0, True, 0.0, 1.0)], [("c", 2, False, 0.0, 1.0)]],
        ]


# ---------------------------------------------------------------- piece bookkeeping
def _cut_pt(F, cid, u):
    c = F.cuts[cid]
    return c["a"] if u <= 0.0 else (c["b"] if u >= 1.0 else ("cs", cid, u))


def piece_start(F, pc):
    if pc[0] == "i":
        return ("b", pc[1]) if pc[2] == 0.0 else ("s", pc[1], pc[2])
    return _cut_pt(F, pc[1], pc[4] if pc[2] else pc[3])


def split_cut(F, cid, u, tol):
    """split cut cid at fraction u (a -> b), snapping to existing splits / ends within tol mm"""
    c = F.cuts[cid]
    Lc = float(np.linalg.norm(F.pt_uv(c["b"]) - F.pt_uv(c["a"])))
    if u * Lc < tol or (1 - u) * Lc < tol:
        return None
    lst = F.csplits.setdefault(cid, [])
    for v in lst:
        if abs(v - u) * Lc < tol:
            return ("cs", cid, v)
    u = round(float(u), 9)
    lst.append(u); lst.sort()
    for sides in F.patches:
        for sd in sides:
            for k, pc in enumerate(sd):
                if pc[0] == "c" and pc[1] == cid and pc[3] < u < pc[4]:
                    a, b = ("c", cid, pc[2], pc[3], u), ("c", cid, pc[2], u, pc[4])
                    sd[k:k + 1] = [b, a] if pc[2] else [a, b]
    return ("cs", cid, u)


def split_piece(F, i, u):
    """split item i (loop fraction u) inside every patch side that carries it"""
    u = float(u)
    if u not in F.splits[i]:
        F.splits[i].append(u)
    for sides in F.patches:
        for sd in sides:
            for k, pc in enumerate(sd):
                if pc[0] == "i" and pc[1] == i and pc[2] < u < pc[3]:
                    sd[k:k + 1] = [("i", i, pc[2], u), ("i", i, u, pc[3])]
    return ("s", i, u)


def _project(F, side, X):
    """closest point of a side's polyline to map point X: (piece index, local fraction, distance)"""
    best = None
    for k, pc in enumerate(side):
        uv = F.piece_uv(pc)
        d = np.linalg.norm(uv - X, axis=1)
        j = int(np.argmin(d))
        L = np.r_[0, np.cumsum(np.linalg.norm(np.diff(uv, axis=0), axis=1))]
        if best is None or d[j] < best[2]:
            best = (k, L[j] / max(L[-1], 1e-12), d[j])
    return best


GS = {}          # canonical item id -> sorted canonical split fractions (all faces)
GL = {}          # canonical item id -> item length


def gs_add(F, i, u_loop, tol):
    """register a split (loop fraction on item i of face F), snapping to an existing split or
    an item end within tol (mm); returns the loop fraction used, or None for an end"""
    it = F.loop[i]
    cid, rev = IT.canon(it, F.f)
    L = GL.setdefault(cid, F.geo[i]["L"])
    u = 1.0 - u_loop if rev else u_loop
    if u * L < tol or (1 - u) * L < tol:
        return None
    lst = GS.setdefault(cid, [])
    for v in lst:
        if abs(v - u) * L < tol:
            return 1.0 - v if rev else v
    lst.append(round(u, 9)); lst.sort()
    return 1.0 - round(u, 9) if rev else round(u, 9)


def sync(F):
    """refine the face's pieces with every global split on its items"""
    for i, it in enumerate(F.loop):
        cid, rev = IT.canon(it, F.f)
        for v in GS.get(cid, []):
            ul = round(1.0 - v if rev else v, 9)
            if ul not in F.splits[i]:
                split_piece(F, i, ul)


def register(F):
    """plan-time splits (cut hits, wedge points) go into the registry exactly"""
    for i, it in enumerate(F.loop):
        cid, rev = IT.canon(it, F.f)
        GL.setdefault(cid, F.geo[i]["L"])
        for ul in list(F.splits[i]):
            u = round(1.0 - ul if rev else ul, 9)
            lst = GS.setdefault(cid, [])
            if u not in lst:
                lst.append(u); lst.sort()


def align_all(faces, tol, rounds=8):
    GS.clear(); GL.clear()
    for F in faces:
        register(F)
    only = os.environ.get("EVG_ALIGNF", "")
    if only:
        sel = {tuple(int(g) for g in k.split("-")) for k in only.split(",")}
        for F in faces:
            sync(F); F.pairs = []
        for r in range(rounds):
            for F in faces:
                if tuple(F.f[:2]) in sel:
                    sync(F)
                    F.pairs = align_splits(F, tol)
            for F in faces:
                sync(F)
        return
    if os.environ.get("EVG_ALIGN", "0") != "1":
        for F in faces:
            sync(F); F.pairs = []
        return
    for r in range(rounds):
        before = sum(len(v) for v in GS.values())
        for F in faces:
            sync(F)
            F.pairs = align_splits(F, tol)
        for F in faces:
            sync(F)
        after = sum(len(v) for v in GS.values())
        print("   alignment round %d: %d splits" % (r, after), flush=True)
        if after == before:
            break


def align_splits(F, tol, eps=0.08, rounds=3):
    """for every break inside a patch side, split the opposite side where the break projects;
    returns the alignment pairs (patch, side a, point on a, side b, break point on b)"""
    pairs = []
    for _ in range(rounds):
        new = 0
        for p, sides in enumerate(F.patches):
            for a, b in ((0, 2), (2, 0), (1, 3), (3, 1)):
                for k in range(1, len(sides[b])):
                    X = F.piece_uv(sides[b][k])[0]
                    bp = piece_start(F, sides[b][k])
                    j, t, d = _project(F, sides[a], X)
                    pc = sides[a][j]
                    if pc[0] == "c" and eps <= t <= 1 - eps:
                        u = pc[4] - t * (pc[4] - pc[3]) if pc[2] else pc[3] + t * (pc[4] - pc[3])
                        sp = split_cut(F, pc[1], u, tol)
                        if sp is not None:
                            pairs.append((p, a, sp, b, bp)); new += 1
                            continue
                    if t < eps or t > 1 - eps or pc[0] != "i":
                        pt = piece_start(F, pc) if t < 0.5 else (piece_start(F, sides[a][j + 1]) if j + 1 < len(sides[a]) else None)
                        if pt is not None:
                            pairs.append((p, a, pt, b, bp))
                        continue
                    u = pc[2] + t * (pc[3] - pc[2])
                    ul = gs_add(F, pc[1], u, tol)
                    if ul is None:
                        pt = piece_start(F, pc) if t < 0.5 else (piece_start(F, sides[a][j + 1]) if j + 1 < len(sides[a]) else None)
                        if pt is not None:
                            pairs.append((p, a, pt, b, bp))
                        continue
                    if ul not in F.splits[pc[1]]:
                        new += 1
                    pairs.append((p, a, split_piece(F, pc[1], ul), b, bp))
        if not new:
            break
    return list({x: None for x in pairs})


# ---------------------------------------------------------------- global context + ILP
class Ctx:
    def __init__(self, patched):
        self.F = patched
        self.info = {}
        for f, F in patched.items():
            for i, it in enumerate(F.loop):
                cid, rev = IT.canon(it, f)
                self.info.setdefault(cid, dict(it=it, f=f))
        self.CS = {k: list(v) for k, v in GS.items() if v}

    def ivs(self, cid, a, b):
        us = [0.0] + self.CS[cid] + [1.0]
        ia = int(np.argmin(np.abs(np.array(us) - a))); ib = int(np.argmin(np.abs(np.array(us) - b)))
        return [("iv", cid, j) for j in range(ia, ib)]

    def piece_vars(self, F, pc):
        if pc[0] == "c":
            us = [0.0] + F.csplits.get(pc[1], []) + [1.0]
            if pc[3] == 0.0 and pc[4] == 1.0:
                return [("cut", F.f, F.M, pc[1])]
            return [("civ", F.f, F.M, pc[1], j) for j in range(us.index(pc[3]), us.index(pc[4]))]
        it = F.loop[pc[1]]
        cid, rev = IT.canon(it, F.f)
        a, b = (1.0 - pc[3], 1.0 - pc[2]) if rev else (pc[2], pc[3])
        if (a <= 1e-9 and b >= 1 - 1e-9) or cid not in self.CS:
            return [IT.key(it, F.f)]
        return self.ivs(cid, round(a, 9), round(b, 9))

    def side_vars(self, F, side, upto=None, after=None):
        out, on = [], upto is None and after is None
        for pc in side:
            st = piece_start(F, pc)
            if after is not None and st == after:
                on = True
            if upto is not None and st == upto:
                break
            if on or upto is not None:
                out.extend(self.piece_vars(F, pc))
        return out


W_TR = float(os.environ.get("EVG_WTR", 0.6))
W_UNDER = float(os.environ.get("EVG_WUNDER", 2.5))
BIGM = 1000


def trans_dirs(F, sides):
    """patch directions in which a 3:1 transition row may sit: 'all', or only the pair whose
    sides carry a core / hub item ('special', default), or none"""
    mode = os.environ.get("EVG_TRANS", "special")
    if mode == "none":
        return []
    if mode == "all":
        return [0, 1]
    if mode == "ringsafe":
        bad = {int(v) for v in os.environ.get("EVG_NORING", "73,80,81,82,83,90,91,92,93,94,95,102").split(",")}
        return [0, 1] if all(g in bad for g in F.f if g >= 0) else []
    if mode == "axial":
        ax = {int(v) for v in os.environ.get("EVG_AXIAL", "73,80,81,82,83,91,92,93,94,102").split(",")}
        pipes = [g for g in F.f if g >= 0]
        if not all(g in ax for g in pipes):
            return []
        free = {int(v) for v in os.environ.get("EVG_AXFREE", "-1").split(",")}
        chk = [g for g in pipes if g not in free]
        out = []
        for d, (a, b) in enumerate(((0, 2), (1, 3))):
            c1, c2 = ((1, 3), (0, 2))[d]
            if all(_arm_ok(F, sides[c1], g) and _arm_ok(F, sides[c2], g) for g in chk):
                out.append(d)
        return out
    sp = lambda sd: any(pc[0] == "i" and F.loop[pc[1]][0] in "oh" for pc in sd)
    return [d for d, (a, b) in enumerate(((0, 2), (1, 3))) if sp(sides[a]) or sp(sides[b])]


def _arm_ok(F, side, q):
    """a template's U arms run along this side: they must cross pipe q's rings, not follow them"""
    uv, _, _ = _side_poly(F, side)
    if q == F.M:
        ds = abs(uv[-1, 0] - uv[0, 0]); dp = abs(uv[-1, 1] - uv[0, 1]) / RHO * mesher.R
    else:
        P = _lift_uv(F.M, q, uv[[0, len(uv) // 2, -1]])
        s, ph = to_map(q, P)
        ph = np.unwrap(ph)
        ds = abs(s[-1] - s[0]); dp = abs(ph[-1] - ph[0]) * mesher.R
    return ds > float(os.environ.get("EVG_ARMTAN", 0.4)) * dp


def _circumferential(F, side, q):
    """does the side's chord run around pipe q rather than along it"""
    uv, _, _ = _side_poly(F, side)
    if q == F.M:
        ds = abs(uv[-1, 0] - uv[0, 0]); dp = abs(uv[-1, 1] - uv[0, 1]) / RHO * mesher.R
        return dp > 1.5 * ds
    P = _lift_uv(F.M, q, uv[[0, len(uv) // 2, -1]])
    s, ph = to_map(q, P)
    ph = np.unwrap(ph)
    ds = abs(s[-1] - s[0]); dp = abs(ph[-1] - ph[0]) * mesher.R
    return dp > 1.5 * ds


def hub_piece_nodes(piece):
    h = IT.HUB["core"]; hc = IT.HUB_COUNTS
    G, Z = h.grid(hc["ma"], hc["nz1"], hc["nz2"])
    ma, NN, nzt = hc["ma"], G.shape[1] - 1, G.shape[2] - 1
    if piece[0] == "line":
        return G[ma, NN if piece[1] > 0 else 0, :]
    if piece[0] == "corner":
        return G[2 * ma if piece[1] > 0 else 0, NN if piece[2] > 0 else 0, :]
    i0 = 2 * ma if piece[1] > 0 else 0
    rng = range(i0, ma - 1, -1) if piece[1] > 0 else range(0, ma + 1)
    return np.array([G[i, NN if piece[2] > 0 else 0, -1] for i in rng])


def assign2(faces_sides, patched, H, fixed=None, w_align=2.0, w_hard=8.0, Hk_fn=None, narc=None):
    """interval assignment in final counts; returns counts and the context"""
    from structured import side_polyline
    from scipy.optimize import milp, LinearConstraint, Bounds
    import os
    ctx = Ctx(patched)
    keys, L = {}, {}
    def addk(k, l):
        if k not in keys:
            keys[k] = len(keys); L[k] = l
    for f, (sides, loop) in faces_sides.items():
        for it in loop:
            addk(IT.key(it, f), IT.length(it))
    for f, F in patched.items():
        for i, it in enumerate(F.loop):
            addk(IT.key(it, f), IT.length(it))
    for cid, us in ctx.CS.items():
        it, f = ctx.info[cid]["it"], ctx.info[cid]["f"]
        Lt = IT.length(it); uu = [0.0] + us + [1.0]
        for j in range(len(uu) - 1):
            addk(("iv", cid, j), Lt * (uu[j + 1] - uu[j]))
    for f, F in patched.items():
        for cid in F.cuts:
            uv = F.piece_uv(("c", cid, False, 0.0, 1.0))
            Lc = float(np.linalg.norm(uv[-1] - uv[0]))
            addk(("cut", f, F.M, cid), Lc)
            us = [0.0] + F.csplits.get(cid, []) + [1.0]
            if len(us) > 2:
                for j in range(len(us) - 1):
                    addk(("civ", f, F.M, cid, j), Lc * (us[j + 1] - us[j]))
    n = len(keys)
    hard, soft = [], []                   # soft: (row, weight)
    def row(pos, neg, cneg=None):
        r = np.zeros(n)
        for k in pos: r[keys[k]] += 1
        for k in neg: r[keys[k]] -= 1
        return r
    for f, (sides, loop) in faces_sides.items():
        for a, b in ((0, 2), (1, 3)):
            hard.append(row([IT.key(it, f) for it in sides[a]], [IT.key(it, f) for it in sides[b]]))
            for s1, s2 in ((a, b), (b, a)):
                if len(sides[s2]) < 2:
                    continue
                uv1, L1, _ = side_polyline(sides[s1], *f)
                uv2, L2, br = side_polyline(sides[s2], *f)
                for j, lb in enumerate(br):
                    p = uv2[np.searchsorted(L2, lb)]
                    k = np.argmin(np.linalg.norm(uv1 - p, axis=1))
                    tau = 1.0 - L1[k] / L1[-1]
                    r = np.zeros(n)
                    for it in sides[s2][:j + 1]: r[keys[IT.key(it, f)]] += 1
                    for it in sides[s1]: r[keys[IT.key(it, f)]] -= tau
                    soft.append((r, w_align))
    # intervals sum to their item; fixed-node items snap to their nodes
    for cid, us in ctx.CS.items():
        it, f = ctx.info[cid]["it"], ctx.info[cid]["f"]
        ivk = [("iv", cid, j) for j in range(len(us) + 1)]
        hard.append(row(ivk, [IT.key(it, f)]))
        if cid[0] == "h":
            P = hub_piece_nodes(cid[1])
            Lc = np.r_[0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]; Lc /= Lc[-1]
            for j, u in enumerate(us):
                kk = int(np.argmin(np.abs(Lc - u)))
                r = row(ivk[:j + 1], []); hard.append(("eq", r, kk))
        elif cid[0] == "o" and cid[3] == "arc":
            for j, u in enumerate(us):
                r = row(ivk[:j + 1], []); hard.append(("eq", r, int(round(u * narc))))
        elif cid[0] == "o":
            for j, u in enumerate(us):
                r = row(ivk[:j + 1], []); r[keys[IT.key(it, f)]] -= u
                soft.append((r, w_hard))
    for f, F in patched.items():
        for cid in F.cuts:
            us = [0.0] + F.csplits.get(cid, []) + [1.0]
            if len(us) > 2:
                hard.append(row([("civ", f, F.M, cid, j) for j in range(len(us) - 1)], [("cut", f, F.M, cid)]))
        for p, sides in enumerate(F.patches):
            allowed = trans_dirs(F, sides)
            for d, (a, b) in enumerate(((0, 2), (1, 3))):
                r = row(ctx.side_vars(F, sides[a]), ctx.side_vars(F, sides[b]))
                if d in allowed:
                    tp, tm = ("tr", f, F.M, p, d, 1), ("tr", f, F.M, p, d, -1)
                    addk(tp, 0.0); addk(tm, 0.0)
                    r = np.r_[r, np.zeros(len(keys) - len(r))]
                    r[keys[tp]] -= 2; r[keys[tm]] += 2
                    ra = row(ctx.side_vars(F, sides[b]), []); ra = np.r_[ra, np.zeros(len(keys) - len(ra))]; ra[keys[tp]] -= 1
                    hard.append(("ge", ra, 0))
                    rb = row(ctx.side_vars(F, sides[a]), []); rb = np.r_[rb, np.zeros(len(keys) - len(rb))]; rb[keys[tm]] -= 1
                    hard.append(("ge", rb, 0))
                hard.append(r)
            if len(allowed) == 2:
                z = ("trz", f, F.M, p); addk(z, 0.0)
                r0 = np.zeros(len(keys)); r1 = np.zeros(len(keys))
                for sg in (1, -1):
                    r0[keys[("tr", f, F.M, p, 0, sg)]] = 1; r1[keys[("tr", f, F.M, p, 1, sg)]] = 1
                r0[keys[z]] = -BIGM; r1[keys[z]] = BIGM
                hard.append(("le", r0, 0)); hard.append(("le", r1, BIGM))
        for sides in F.patches:
            for a, b in ((0, 2), (1, 3)):
                for s1, s2 in ((a, b), (b, a)):
                    if len(sides[s2]) < 2:
                        continue
                    uv1, L1, _ = _side_poly(F, sides[s1])
                    uv2, L2, br = _side_poly(F, sides[s2])
                    v1 = ctx.side_vars(F, sides[s1])
                    for j, lb in enumerate(br):
                        pnt = uv2[np.searchsorted(L2, lb)]
                        k = np.argmin(np.linalg.norm(uv1 - pnt, axis=1))
                        tau = 1.0 - L1[k] / L1[-1]
                        r = np.zeros(n)
                        for pc in sides[s2][:j + 1]:
                            for v in ctx.piece_vars(F, pc): r[keys[v]] += 1
                        for v in v1: r[keys[v]] -= tau
                        soft.append((r, w_align))
        hardf = {tuple(int(g) for g in k.split("-")) for k in os.environ.get("EVG_ALIGNHARD", "").split(",") if k}
        for (p, a, pa, b, pb) in F.pairs:
            sides = F.patches[p]
            r = row(ctx.side_vars(F, sides[a], upto=pa), ctx.side_vars(F, sides[b], after=pb))
            if tuple(F.f[:2]) in hardf:
                hard.append(r)
            else:
                soft.append((r, w_hard))
    n = len(keys)
    pad = lambda r: np.r_[r, np.zeros(n - len(r))]
    hard = [(h[0], pad(h[1]), h[2]) if isinstance(h, tuple) else pad(h) for h in hard]
    soft = [(pad(r), w) for r, w in soft]
    HC = float(os.environ.get("EVG_HCORE", H))
    def tgt(k):
        if k[0] in ("tr", "trz"):
            return 0.0
        h = HC if k[0] == "oz" else (Hk_fn(k) if Hk_fn else H)
        if k[0] == "iv" and Hk_fn:
            h = Hk_fn(IT.key(ctx.info[k[1]]["it"], ctx.info[k[1]]["f"]))
        return max(L[k] / h, 0.5)
    target = np.array([tgt(k) for k in keys])
    istr = np.array([k[0] in ("tr", "trz") for k in keys])
    ns = len(soft)
    nv = 3 * n + ns                       # counts, over-refinement, under-refinement, soft slacks
    A_rows, lo, hi = [], [], []
    for r in hard:
        if isinstance(r, tuple):
            A_rows.append(np.r_[r[1], np.zeros(2 * n + ns)])
            lo.append(r[2] if r[0] in ("eq", "ge") else -np.inf); hi.append(r[2] if r[0] in ("eq", "le") else np.inf)
        else:
            A_rows.append(np.r_[r, np.zeros(2 * n + ns)]); lo.append(0); hi.append(0)
    for i in range(n):
        r = np.zeros(nv); r[i] = 1; r[n + i] = -1; A_rows.append(r); lo.append(-np.inf); hi.append(target[i])
        r = np.zeros(nv); r[i] = -1; r[2 * n + i] = -1; A_rows.append(r); lo.append(-np.inf); hi.append(-target[i])
    wsoft = []
    for j, (r0, w) in enumerate(soft):
        r = np.r_[r0, np.zeros(2 * n + ns)]; r[3 * n + j] = -1; A_rows.append(r); lo.append(-np.inf); hi.append(0)
        r = np.r_[-r0, np.zeros(2 * n + ns)]; r[3 * n + j] = -1; A_rows.append(r); lo.append(-np.inf); hi.append(0)
        wsoft.append(w)
    if fixed:
        for k, v in fixed.items():
            if k in keys:
                r = np.zeros(nv); r[keys[k]] = 1; A_rows.append(r); lo.append(v); hi.append(v)
    even = [i for k, i in keys.items() if k[0] in ("p", "q")] if os.environ.get("EVG_EVENPLATE", "1") == "1" else []
    if even:
        nv2 = nv + len(even)
        A_rows = [np.r_[r, np.zeros(len(even))] for r in A_rows]
        for j, i in enumerate(even):
            r = np.zeros(nv2); r[i] = 1; r[nv + j] = -2; A_rows.append(r); lo.append(0); hi.append(0)
    else:
        nv2 = nv
    wdev = np.where(istr, W_TR, 1.0 / np.maximum(target, 0.5))
    wdev[np.array([k[0] == "trz" for k in keys])] = 0.0
    c = np.r_[np.zeros(n), wdev, np.where(istr, 0.0, W_UNDER * wdev), np.array(wsoft), np.zeros(nv2 - nv)]
    integrality = np.r_[np.ones(n), np.zeros(2 * n + ns), np.ones(nv2 - nv)]
    lb = np.r_[np.array([0 if (k[0] in ("tr", "trz") or (k[0] in ("iv", "civ") and L[k] < 0.35 * H)) else 1
                         for k in keys]), np.zeros(2 * n + ns), np.ones(nv2 - nv)]
    ub = np.r_[np.array([1 if k[0] == "trz" else 1e5 for k in keys]), np.full(2 * n + ns, 1e5), np.full(nv2 - nv, 1e5)]
    res = milp(c, constraints=LinearConstraint(np.array(A_rows), lo, hi), integrality=integrality,
               bounds=Bounds(lb, ub), options=dict(time_limit=180))
    assert res.x is not None, res.message
    m = np.round(res.x[:n]).astype(int)
    out = {k: int(m[i]) for k, i in keys.items()}
    dev = {k: out[k] / max(L[k] / H, 1e-9) for k in keys if k[0] not in ("tr", "trz")}
    ctx.trans = {}
    for k, v in out.items():
        if k[0] == "tr" and v > 0:
            ctx.trans[(k[1], k[3])] = (k[4], k[5], v)
    # node warps of resampled items
    IT.WARP.clear()
    for cid, us in ctx.CS.items():
        if cid[0] in ("e", "t", "k", "pq"):
            cs = np.cumsum([0] + [out[("iv", cid, j)] for j in range(len(us) + 1)])
            IT.WARP[cid] = (np.array([0.0] + us + [1.0]), cs)
    return out, dev, ctx


def _side_poly(F, side):
    P, br = [], []
    for pc in side:
        u = F.piece_uv(pc)
        P.extend(u if not P else u[1:])
        br.append(len(P) - 1)
    P = np.array(P)
    L = np.r_[0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
    return P, L, [L[b] for b in br[:-1]]


def winslow(G, iters=300):
    """elliptic smoothing of the interior of a structured grid G (n1+1, n2+1, 2); keeps the
    result only if it has no more non-convex cells than the input"""
    def nonconvex(G):
        q = np.stack([G[:-1, :-1], G[1:, :-1], G[1:, 1:], G[:-1, 1:]], 2)
        e1 = np.roll(q, -1, 2) - q; e2 = np.roll(q, 1, 2) - q
        cr = e1[..., 0] * e2[..., 1] - e1[..., 1] * e2[..., 0]
        sg = np.sign(np.median(cr))
        return int(((sg * cr) <= 0).any(-1).sum())
    if G.shape[0] < 3 or G.shape[1] < 3:
        return G
    H = G.copy()
    for _ in range(iters):
        xe = 0.5 * (H[2:, 1:-1] - H[:-2, 1:-1]); xn = 0.5 * (H[1:-1, 2:] - H[1:-1, :-2])
        a = (xn ** 2).sum(-1)[..., None]; c = (xe ** 2).sum(-1)[..., None]; b = (xe * xn).sum(-1)[..., None]
        new = (a * (H[2:, 1:-1] + H[:-2, 1:-1]) + c * (H[1:-1, 2:] + H[1:-1, :-2])
               - 0.5 * b * (H[2:, 2:] - H[2:, :-2] - H[:-2, 2:] + H[:-2, :-2])) / (2 * (a + c) + 1e-30)
        H[1:-1, 1:-1] = new
    return H if nonconvex(H) <= nonconvex(G) else G


# ---------------------------------------------------------------- meshing
def mesh_face(nodes, F, counts, other, ctx):
    """node ids of every patch (TFI + Winslow in the map), returns quads (k, 4)"""
    f, M = F.f, F.M
    full = {}
    def item_ids(i):
        if i not in full:
            it = F.loop[i]
            full[i] = list(IT.node_ids(nodes, it, counts[IT.key(it, f)], M, other, f))
        return full[i]
    def idx(i, u):
        """loop-direction node index of loop fraction u on item i"""
        it = F.loop[i]
        cid, rev = IT.canon(it, f)
        n = counts[IT.key(it, f)]
        if u <= 0.0:
            return 0
        if u >= 1.0:
            return n
        uc = round(1.0 - u if rev else u, 9)
        us = ctx.CS[cid]
        j = int(np.argmin(np.abs(np.array(us) - uc)))
        c = sum(counts[("iv", cid, k)] for k in range(j + 1))
        return n - c if rev else c
    def point_id(p):
        if p[0] == "b":
            return item_ids(p[1])[0]
        if p[0] == "s":
            return item_ids(p[1])[idx(p[1], p[2])]
        if p[0] == "cs":
            return cut_nodes(p[1])[cut_idx(p[1], p[2])]
        return nodes.get(("Q", f, M, p[1]), _lift_uv(M, other, F.centres[p[1]][None])[0])
    cut_ids, cut_cs = {}, {}
    def cut_nodes(cid):
        if cid not in cut_ids:
            c = F.cuts[cid]
            ia, ib = point_id(c["a"]), point_id(c["b"])
            nf = counts[("cut", f, M, cid)]
            us = [0.0] + F.csplits.get(cid, []) + [1.0]
            cs = np.cumsum([0] + ([counts[("civ", f, M, cid, j)] for j in range(len(us) - 1)] if len(us) > 2 else [nf]))
            cut_cs[cid] = (us, cs)
            fr = np.interp(np.arange(nf + 1), cs, us)
            ua = F.centres[c["a"][1]] if c["a"][0] == "Q" else _map_of(M, nodes.X[ia])
            ub = _map_of(M, nodes.X[ib], ref=ua)
            t = fr[1:-1, None]
            P = _lift_uv(M, other, ua + t * (ub - ua)) if nf > 1 else np.zeros((0, 3))
            cut_ids[cid] = [ia] + [nodes.get(("cut", f, M, cid, k + 1), P[k]) for k in range(nf - 1)] + [ib]
        return cut_ids[cid]
    def cut_idx(cid, u):
        cut_nodes(cid)
        us, cs = cut_cs[cid]
        return int(cs[us.index(u)])
    def piece_ids(pc):
        if pc[0] == "c":
            ids = cut_nodes(pc[1])[cut_idx(pc[1], pc[3]):cut_idx(pc[1], pc[4]) + 1]
            return ids[::-1] if pc[2] else ids
        i = pc[1]
        return item_ids(i)[idx(i, pc[2]):idx(i, pc[3]) + 1]
    Q = []
    for pid, sides in enumerate(F.patches):
        S = []
        for sd in sides:
            ids = []
            for pc in sd:
                pi = piece_ids(pc)
                if len(pi) < 2:
                    continue
                assert not ids or ids[-1] == pi[0], ("piece chain broken", f, pid, pc)
                ids.extend(pi if not ids else pi[1:])
            S.append(np.array(ids))
        tr = ctx.trans.get((f, pid))
        if tr is None:
            Q.append(tfi_ids(nodes, M, other, S, ("tp", f, M, pid)))
        else:
            d, sg, t = tr
            rot = {(0, 1): 0, (0, -1): 2, (1, 1): 1, (1, -1): 3}[(d, sg)]
            S = S[rot:] + S[:rot]
            Q.append(tfi_transition(nodes, M, other, S, ("tp", f, M, pid), t))
    return np.concatenate(Q)


def _map_of(M, X, ref=None):
    s, ph = to_map(M, np.asarray(X)[None])
    uv = np.array([s[0], ph[0] * RHO])
    if ref is not None:
        uv[1] = ref[1] + (uv[1] - ref[1] + np.pi * RHO) % (2 * np.pi * RHO) - np.pi * RHO
    return uv


def _lift_uv(M, other, uv):
    P, _ = lift(M, uv[:, 0], uv[:, 1] / RHO, other)
    return P


def _template_steps(ff, fc, t):
    """choose t steps of 3 (rest 1) along the fine side so that the coarse nodes it keeps sit
    closest to the coarse side's node fractions: DP over (coarse cell, templates used)"""
    nc = len(fc) - 1
    INF = 1e18
    cost = np.full((nc + 1, t + 1), INF); prev = np.zeros((nc + 1, t + 1), int)
    cost[0, 0] = 0.0
    for i in range(nc):
        for j in range(t + 1):
            if cost[i, j] >= INF:
                continue
            for st, dj in ((1, 0), (3, 1)):
                jj = j + dj
                if jj > t:
                    continue
                pos = i + 1 + 2 * jj
                if pos >= len(ff):
                    continue
                c = cost[i, j] + (ff[pos] - fc[i + 1]) ** 2
                if c < cost[i + 1, jj]:
                    cost[i + 1, jj] = c; prev[i + 1, jj] = dj
    steps, j = [], t
    for i in range(nc, 0, -1):
        dj = prev[i, j]; steps.append(3 if dj else 1); j -= dj
    return steps[::-1]


def tfi_transition(nodes, M, other, S, kp, t):
    """patch whose side S[0] has 2t more cells than S[2]: TFI on the coarse grid, then the row
    against S[0] is replaced by t 3:1 templates (4 quads each) and plain quads"""
    nf, nc = len(S[0]) - 1, len(S[2]) - 1
    assert nf == nc + 2 * t and len(S[1]) == len(S[3]), ("transition counts", kp, nf, nc, t)
    XYZ = nodes.arr()
    def uvs(ids):
        s, ph = to_map(M, XYZ[ids]); return s, ph
    allid = np.concatenate([S[0], S[1], S[2][::-1], S[3][::-1]])
    s_, ph_ = to_map(M, XYZ[allid]); ph_ = np.unwrap(ph_)
    uv_all = np.stack([s_, ph_ * RHO], 1)
    o = np.cumsum([0, len(S[0]), len(S[1]), nc + 1])
    Bf, Rr, T, Lf = uv_all[o[0]:o[1]], uv_all[o[1]:o[2]], uv_all[o[2]:o[3]], uv_all[o[3]:]
    def frac(P):
        dd = np.r_[0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
        return dd / max(dd[-1], 1e-12)
    steps = _template_steps(frac(Bf), frac(T), t)
    pos = np.r_[0, np.cumsum(steps)]
    coarse = np.array(S[0])[pos]
    Sc = [coarse, np.array(S[1]), np.array(S[2]), np.array(S[3])]
    Qc = tfi_ids(nodes, M, other, Sc, kp)
    n2 = len(S[1]) - 1
    grid = Qc.reshape(nc, n2, 4)
    out = [Qc.reshape(nc, n2, 4)[:, 1:].reshape(-1, 4)]
    row1 = np.r_[grid[:, 0, 3], grid[-1, 0, 2]]          # nodes of grid row j = 1, i = 0..nc
    X2 = nodes.arr()
    def mp(i):
        s, ph = to_map(M, X2[[i]]); return np.array([s[0], ph[0] * RHO])
    quads = []
    for i, st in enumerate(steps):
        f0 = pos[i]
        r0, r1 = row1[i], row1[i + 1]
        if st == 1:
            quads.append([S[0][f0], S[0][f0 + 1], r1, r0]); continue
        u0, u1 = mp(r0), mp(r1); u1[1] = u0[1] + (u1[1] - u0[1] + np.pi * RHO) % (2 * np.pi * RHO) - np.pi * RHO
        q = []
        for k in (1, 2):
            fb = mp(S[0][f0 + k]); fb[1] = u0[1] + (fb[1] - u0[1] + np.pi * RHO) % (2 * np.pi * RHO) - np.pi * RHO
            top = u0 + (u1 - u0) * k / 3.0
            q.append(0.5 * (fb + top))
        P = _lift_uv(M, other, np.array(q))
        q1, q2 = nodes.get(kp + ("q", i, 1), P[0]), nodes.get(kp + ("q", i, 2), P[1])
        quads += [[S[0][f0], S[0][f0 + 1], q1, r0], [S[0][f0 + 1], S[0][f0 + 2], q2, q1],
                  [S[0][f0 + 2], S[0][f0 + 3], r1, q2], [q1, q2, r1, r0]]
    out.append(np.array(quads, int))
    return np.concatenate(out)


def tfi_ids(nodes, M, other, S, kp):
    """TFI of a patch given its 4 sides as node-id arrays in loop order"""
    n1, n2 = len(S[0]) - 1, len(S[1]) - 1
    assert len(S[2]) - 1 == n1 and len(S[3]) - 1 == n2, ("unbalanced patch", kp, [len(s) - 1 for s in S])
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
        return d / max(d[-1], 1e-12)
    xb, xt = frac(B), frac(T); yl, yr = frac(Lf), frac(Rr)
    grid = np.empty((n1 + 1, n2 + 1), int)
    grid[:, 0] = bottom; grid[:, -1] = top; grid[0, :] = left; grid[-1, :] = right
    I, J = np.meshgrid(np.arange(1, n1), np.arange(1, n2), indexing="ij")
    if I.size:
        eta = (1 - xb[I]) * yl[J] + xb[I] * yr[J]
        xi = (1 - yl[J]) * xb[I] + yl[J] * xt[I]
        P = ((1 - eta)[..., None] * B[I] + eta[..., None] * T[I]
             + (1 - xi)[..., None] * Lf[J] + xi[..., None] * Rr[J]
             - ((1 - xi) * (1 - eta))[..., None] * B[0] - (xi * (1 - eta))[..., None] * B[-1]
             - ((1 - xi) * eta)[..., None] * T[0] - (xi * eta)[..., None] * T[-1])
        G = np.empty((n1 + 1, n2 + 1, 2))
        G[:, 0] = B; G[:, -1] = T; G[0, :] = Lf; G[-1, :] = Rr
        G[1:-1, 1:-1] = P
        if SMOOTH:
            G = winslow(G)
        P = G[1:-1, 1:-1]
        Pl, _ = lift(M, P[..., 0].ravel(), P[..., 1].ravel() / RHO, other)
        for (i, j), p in zip(zip(I.ravel(), J.ravel()), Pl):
            grid[i, j] = nodes.get(kp + (i, j), p)
    return np.stack([grid[:-1, :-1], grid[1:, :-1], grid[1:, 1:], grid[:-1, 1:]], -1).reshape(-1, 4)


import os as _os
SMOOTH = _os.environ.get("EVG_WINSLOW", "1") == "1"
