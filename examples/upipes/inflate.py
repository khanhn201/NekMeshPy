"""sheet inflation in a hex mesh: pillow a shrink set, with optional cut faces (the set wraps
around and meets itself: two parallel layers) and crossing edges (the new sheet crosses itself
along a curve of edges, one extra hex per edge)."""
import numpy as np
from collections import defaultdict

FACES = [[0, 3, 2, 1], [4, 5, 6, 7], [0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]]   # outward
EDGES = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]


class UF:
    def __init__(self):
        self.p = {}

    def f(self, a):
        self.p.setdefault(a, a)
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def u(self, a, b):
        a, b = self.f(a), self.f(b)
        if a != b:
            self.p[b] = a


def fkey(nodes):
    return tuple(sorted(int(v) for v in nodes))


def face_table(H):
    """face key -> list of (hex, local face)"""
    ft = defaultdict(list)
    for h, row in enumerate(H):
        for j, f in enumerate(FACES):
            ft[fkey(row[f])].append((h, j))
    return ft


def inflate(X, H, S, cut=(), cross=(), alpha=0.3):
    """X (n,3), H (m,8), S bool (m,) shrink set; cut: face keys between two S hexes that act as
    walls; cross: edge keys (sorted node pairs) along which the new sheet crosses itself.
    -> X, H (old hexes rewritten + new hexes appended), kinds of the new hexes ('I', 'C', 'X')"""
    H = np.asarray(H).copy()
    S = np.asarray(S, bool)
    cut = sorted(set(cut)); cross = sorted(set(cross))
    cutset = set(cut)
    ft = face_table(H)
    I = [k for k, v in ft.items() if len(v) == 2 and S[v[0][0]] != S[v[1][0]]]
    for k in cut:
        v = ft[k]
        assert len(v) == 2 and S[v[0][0]] and S[v[1][0]], ("bad cut face", k)
    sigma = set(I) | cutset
    bfaces = {k for k, v in ft.items() if len(v) == 1}
    bedge = set()
    for k in bfaces:
        (h, j), = ft[k]
        f = H[h][FACES[j]]
        for a in range(4):
            bedge.add(fkey((f[a], f[(a + 1) % 4])))
    inc = defaultdict(list)
    for h, row in enumerate(H):
        for c, v in enumerate(row):
            inc[int(v)].append(h)
    snodes = sorted({v for k in sigma for v in k})
    # ---- sectors per node
    sec = {}            # (v, hex) -> sector label ; labels are (v, i)
    mid = {}            # (v, cut face key) -> sector label
    uf = UF()
    for v in snodes:
        hs = inc[v]
        loc = UF()
        for h in hs:
            loc.f(h)
        for h in hs:
            for j, f in enumerate(FACES):
                fn = H[h][f]
                if v in fn:
                    k = fkey(fn)
                    if k in sigma:
                        continue
                    for (h2, _) in ft[k]:
                        if h2 != h:
                            loc.u(h, h2)
        for h in hs:
            sec[(v, h)] = ("h", v, loc.f(h))
            uf.f(sec[(v, h)])
        cfs = [k for k in cut if v in k]
        cl = UF()
        for k in cfs:
            cl.f(k)
        for a in cfs:
            for b in cfs:
                if a < b and len(set(a) & set(b)) == 2:
                    cl.u(a, b)
        for k in cfs:
            mid[(v, k)] = ("m", v, cl.f(k))
            uf.f(mid[(v, k)])
    # ---- merges at interior boundary edges of the cut surface that do not cross
    ecount = defaultdict(list)
    for k in cut:
        (h, j) = ft[k][0]
        f = H[h][FACES[j]]
        for a in range(4):
            ecount[fkey((f[a], f[(a + 1) % 4]))].append(k)
    for e, ks in ecount.items():
        if len(ks) != 1 or e in bedge or e in cross:
            continue
        k = ks[0]
        hL, hR = ft[k][0][0], ft[k][1][0]
        around = [h for h in inc[e[0]] if e[1] in H[h]]
        others = [h for h in around if h not in (hL, hR)]
        assert others, ("cut edge with nothing opposite", e)
        for v in e:
            ls = {uf.f(sec[(v, h)]) for h in others}
            assert len(ls) == 1, ("cut edge facing several sectors", e)
            uf.u(sec[(v, others[0])], mid[(v, k)])
    # ---- node copies: the class holding a non-S hex keeps the id
    copy = {}
    Xl = list(map(np.asarray, X)); parent = []
    for v in snodes:
        classes = defaultdict(list)
        for h in inc[v]:
            classes[uf.f(sec[(v, h)])].append(h)
        for key in [mk for mk in mid if mk[0] == v]:
            classes.setdefault(uf.f(mid[key]), [])
        keep = None
        for c, hs in classes.items():
            if any(not S[h] for h in hs):
                keep = c
        if keep is None:
            keep = max(classes, key=lambda c: len(classes[c]))
        for c, hs in classes.items():
            if c == keep:
                copy[c] = v
                continue
            P = X[v] if not hs else X[v] + alpha * (np.mean([X[H[h]].mean(0) for h in hs], 0) - X[v])
            copy[c] = len(Xl); Xl.append(P); parent.append(v)
    snodes_set = set(snodes)
    Hn = H.copy()
    for h in range(len(H)):
        for c in range(8):
            v = int(H[h, c])
            if v in snodes_set:
                Hn[h, c] = copy[uf.f(sec[(v, h)])]
    new, kinds = [], []
    for k in I:
        (h1, j1), (h2, j2) = ft[k]
        hn, jn, hs_ = (h1, j1, h2) if not S[h1] else (h2, j2, h1)
        f = H[hn][FACES[jn]]
        new.append([copy[uf.f(sec[(int(v), hn)])] for v in f] + [copy[uf.f(sec[(int(v), hs_)])] for v in f])
        kinds.append("I")
    for k in cut:
        (hL, jL), (hR, _) = ft[k]
        f = H[hL][FACES[jL]]
        a = [copy[uf.f(sec[(int(v), hL)])] for v in f]
        m = [copy[uf.f(mid[(int(v), k)])] for v in f]
        b = [copy[uf.f(sec[(int(v), hR)])] for v in f]
        new.append(a + m); kinds.append("C")
        new.append(m + b); kinds.append("C")
    for e in cross:
        u, w = e
        around = [h for h in inc[u] if w in H[h]]
        fcs = {}
        for h in around:
            fcs[h] = [fkey(H[h][f]) for f in FACES if u in H[h][f] and w in H[h][f]]
        order, via = [around[0]], []
        came = None
        while True:
            h = order[-1]
            fk = [k for k in fcs[h] if k != came][0]
            h2 = [hh for hh, _ in ft[fk] if hh != h]
            assert h2, ("crossing edge on the boundary", e)
            via.append(fk)
            if h2[0] == order[0]:
                break
            order.append(h2[0]); came = fk
        labs_u, labs_w = [], []
        for h, fk in zip(order, via):
            for labs, v in ((labs_u, u), (labs_w, w)):
                l = uf.f(sec[(v, h)])
                if not labs or labs[-1] != l:
                    labs.append(l)
                if fk in cutset:
                    labs.append(uf.f(mid[(v, fk)]))
        for labs in (labs_u, labs_w):
            while len(labs) > 1 and labs[0] == labs[-1]:
                labs.pop()
        assert len(labs_u) == 4 and len(labs_w) == 4, ("crossing edge without 4 sectors", e, labs_u, labs_w)
        new.append([copy[l] for l in labs_u] + [copy[l] for l in labs_w])
        kinds.append("X")
    Xn = np.array(Xl)
    return Xn, (np.vstack([Hn, np.array(new, int)]) if new else Hn), kinds, parent
