"""Node positions of the global bottom section at every band level.

Per level: x-jlines (lines along y beside x-joints) at the core position / y-slot edge / plate edge,
y-jlines at the core position / x-slot edge / x-bar edge, channel-footprint chains on their jline,
rod transverse lines, ring radii (pin / post / plug).  Every lattice cell is smoothed as a tile
(toolkit smoother) and the displacement applied to all cells of the same local geometry; the wall
band is snapped onto R692.15 and smoothed; the basket rings follow the cone / funnel."""
import os, sys, pickle, collections, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, "..", "p3"), os.path.join(HERE, "..", "p1")]
import bsec, rules, secA
import footprint as fp
from geom import H, GL, GR, KB, IU, CENTRAL, PITCH
import bores
from nekmeshpy.quadmesh.quadmesh import QuadMesh
from nekmeshpy.quadmesh.smoother import constrain as cn
from nekmeshpy.quadmesh.smoother import smooth

R_W = rules.R_RING
Z = rules.Z
SITES = set(rules.sites())
PLUG = 14.0                                  # central plug, circular (compromise: 20.32 sq n 22.225)


def gz(r):
    """grid bottom: flat 206.4985, x-bar cone beyond R1, capped at the downcomer bottom 298.99."""
    return np.minimum(np.maximum(206.4985, 206.4985 + (r - 386.48) / 2.8637), 298.98724095080007)


# name, z, xmode, ymode, rod, ring
LEVELS = [
    ("top", Z["top"], "top", "top", False, "pin"),
    ("AB", Z["AB"], "yslot", "xslot", False, "pin"),
    ("BC", Z["BC"], "yslot", "xslot", False, "post"),
    ("CD", Z["CD"], "plate", "xslot", True, "post"),
    ("DE", Z["DE"], "plate", "xslot", True, "post"),
    ("EF", Z["EF"], "plate", "xslot", True, "post"),
    ("F3F2", Z["F3"], "plate", "xslot", True, "post"),
    ("F2F1", Z["F2"], "plate", "bar", True, "post"),
    ("F1F0", Z["F1"], "plate", "bar", True, "post"),
    ("F0H", Z["F0"], "plate", "bar", True, "post"),
    ("HI", Z["HI"], "plate", "bar", True, "post"),
    ("IJ", Z["IJ"], "plate", "bar", True, "post"),
    ("JK", Z["JK"], "plate", "bar", True, "post"),
    ("KN", None, "plate", "bar", True, "post"),        # z = gz(r) per node
]
LV = {L[0]: L for L in LEVELS}


def is_bar_joint(yj):
    return int(round((abs(yj) - 25.4) / rules.P)) % 2 == 0


def line_pos(lid, L):
    ax, kind, m = lid
    j = bsec.xj(m)
    if kind == "J":
        return j
    lo_side = kind in ("L", "B")
    mode = L[2] if ax == "x" else L[3]
    if mode == "top":
        return j - KB if lo_side else j + KB
    if mode == "plate" and ax == "x" and rules.is_plate_joint(j):
        lo, hi = rules.plate_edges(j)
        return lo if lo_side else hi
    if mode == "bar" and ax == "y" and is_bar_joint(j):
        return j - rules.PLATE_HW if lo_side else j + rules.PLATE_HW
    lo, hi = rules.y_slot(j) if ax == "x" else rules.x_slot(j)
    return lo if lo_side else hi


import os as _os
LOFT_RIM = bool(int(_os.environ.get("LOFT_RIM", "0")))
RIM_GAP = 3.0
RING_SITES_ONLY = bool(int(_os.environ.get("RING_SITES_ONLY", "1")))
RING_FREE_R = float(_os.environ.get("RING_FREE_R", "80"))      # template rings are freed only this close to the wall


def ring_held(c):
    """a ring held on its circle: a real pin / post, or any cell away from the rim (the masks there read
    centroids, so the interior must keep its old geometry exactly)."""
    if not RING_SITES_ONLY or tuple(c) in SITES:
        return True
    return float(np.hypot(*c)) * PITCH < R_W - RING_FREE_R
RIM_RMIN = float(_os.environ.get("RIM_RMIN", "8.0"))
RIM_RMIN_PIN = 5.0
RIM_GAP_PIN = float(_os.environ.get("RIM_GAP_PIN", "2.5"))


def clamp_ring(c, L, p, rmax=None):
    """p on the ring circle of cell c, slid along the circle (if needed) to stay at |p| <= rmax."""
    cc = np.array(c, float) * PITCH
    rr = ring_radius(c, L)
    d = p - cc; p = cc + rr * d / np.linalg.norm(d)
    rmax = R_W - 1.5 if rmax is None else rmax
    if np.hypot(*p) <= rmax:
        return p
    D = np.linalg.norm(cc)                                 # circle-circle intersection with |x| = rmax
    a_ = (rmax ** 2 - rr ** 2 + D ** 2) / (2 * D)
    h = np.sqrt(max(rmax ** 2 - a_ ** 2, 0.0))
    u = cc / D; n = np.array([-u[1], u[0]])
    cand = [a_ * u + h * n, a_ * u - h * n]
    return min(cand, key=lambda q: np.linalg.norm(q - p))


def ring_radius(c, L):
    if c in CENTRAL:
        return PLUG
    if L[5] == "pin":
        rr = bsec.T_PIN if c in bsec.T_CELLS else rules.R_PIN
    else:
        rr = rules.R_POST
    d = np.hypot(c[0], c[1]) * PITCH
    if d - rr > R_W - RIM_LENS:                   # post fully in the wall: keep a virtual ring
        rr = d - R_W + RIM_LENS                   # crossing it (the lens inside stays fluid)
    if LOFT_RIM and d < R_W:                      # loft rim: a pin / ring must fit inside the wall
        cap = R_W - (RIM_GAP_PIN if c in bsec.SITES else RIM_GAP) - d
        if cap >= (RIM_RMIN_PIN if c in bsec.SITES else RIM_RMIN):   # a real pin keeps a small ring; a pinless placeholder does not
            rr = min(rr, cap)
    return rr


RIM_LENS = 6.0


ROLE = {}
for (c_, s_), kk_ in bsec.FP_HALVES:
    for r_, k_ in kk_.items():
        ROLE.setdefault(k_, (r_, c_, s_))
UA = GL + np.sqrt(GR ** 2 - KB ** 2)


def fp_pos(key, L):
    nd = bsec.FP_NODES[key]
    if L[2] == "top" or ROLE[key][1] in CENTRAL:
        return np.array(nd["top"])
    r, c, s = ROLE[key]
    jl, jn = bsec.side_lines(c, s)
    f = fp.bs.frame(c, s)
    jpos = line_pos(jl, L); npos = line_pos(jn, L)
    d = abs(npos - jpos)
    nb = (c[0] + bsec.D4[s][0], c[1] + bsec.D4[s][1])
    rod = L[4] and jl[0] == "x" and not rules.is_plate_joint(bsec.xj(jl[2])) and c in SITES and nb in SITES
    ut = rules.ROD_HW if rod else IU
    g = max((H - ring_radius(c, L)) - d, 1.0)
    ua = min(UA, ut + 2.5)
    loc = dict(Jm=(-(GL + GR), 0), Jp=(GL + GR, 0), Km=(-ut, 0), K0=(0, 0), Kp=(ut, 0),
               Am=(-ua, d), Ap=(ua, d), Im=(-ut, d), I0=(0, d), Ip=(ut, d),
               Ym=(-(ut - 1.0), d + 0.3 * g), Y0=(0, d + 0.3 * g), Yp=(ut - 1.0, d + 0.3 * g),
               Bm=(-ut, d + 0.6 * g), M=(0, d + 0.6 * g), Bp=(ut, d + 0.6 * g))[r]
    return f(loc[0], H - loc[1])


def coords(key, L, inner, cache, core_xy):
    if key in cache:
        return cache[key]
    if isinstance(key[0], (float, np.floating)):
        if key in bsec.FP_NODES:
            p = fp_pos(key, L) if L[2] != "top" else np.array(core_xy.get(key, bsec.FP_NODES[key]["top"]))
        else:
            p = np.array(core_xy.get(key, key), float)
    elif key[0] == "X":
        _, a, b = key
        pa, pb = line_pos(a, L), line_pos(b, L)
        p = np.array([pa, pb]) if a[0] == "x" else np.array([pb, pa])
    elif key[0] in ("mj", "mq"):
        _, c, s, t = key
        jl, jn = bsec.side_lines(c, s)
        lid = jl if key[0] == "mj" else jn
        a = coords(bsec.cross(lid, bsec.side_lines(c, (s - 1) % 4)[1]), L, inner, cache, core_xy)
        b = coords(bsec.cross(lid, bsec.side_lines(c, (s + 1) % 4)[1]), L, inner, cache, core_xy)
        p = a + (t + 1) / 6.0 * (b - a)
    elif key[0] == "ring":
        _, c, k = key
        cc = np.array(c, float) * PITCH
        if L[5] != "pin" and c not in CENTRAL:            # post: regular O-grid off the square
            q = coords(("sqb", c, k), L, inner, cache, core_xy) - cc
        else:
            q = coords(inner[c][k], L, inner, cache, core_xy) - cc
        p = cc + ring_radius(c, L) * q / np.linalg.norm(q)
    elif key[0] in ("sqb", "sqi"):
        c = key[1]
        cc = np.array(c, float) * PITCH
        n = len(inner[c])
        na, nb = {24: (6, 6), 26: (7, 6), 28: (8, 6)}[n]
        h = 0.55 * ring_radius(c, L)
        walk = bsec.walk_of(na, nb)
        if key[0] == "sqb":
            k = key[2]
            q0 = coords(inner[c][0], L, inner, cache, core_xy) - cc
            a0 = np.arctan2(q0[1], q0[0])
            corners = [cc + h * np.sqrt(2) * np.array([np.cos(a0 + j * np.pi / 2), np.sin(a0 + j * np.pi / 2)]) for j in range(5)]
            lens = [na, nb, na, nb]
            sd = 0; tt = k
            while tt >= lens[sd]:
                tt -= lens[sd]; sd += 1
            p = corners[sd] + (tt / lens[sd]) * (corners[sd + 1] - corners[sd])
        else:
            _, _, a, b = key
            u, v = a / na, b / nb
            widx = {ab: i for i, ab in enumerate(walk)}
            E = lambda ab: coords(("sqb", c, widx[ab]), L, inner, cache, core_xy)
            p = ((1 - v) * E((a, 0)) + v * E((a, nb)) + (1 - u) * E((0, b)) + u * E((na, b))
                 - ((1 - u) * (1 - v)) * E((0, 0)) - (u * (1 - v)) * E((na, 0))
                 - (u * v) * E((na, nb)) - ((1 - u) * v) * E((0, nb)))
    else:
        raise KeyError(key)
    cache[key] = p
    return p


# ------------------------------------------------------------------ tiles ----------------
def rel_key(k, c):
    if isinstance(k[0], (float, np.floating)):
        if k not in ROLE:
            return None
        r, cc, s = ROLE[k]
        return ("fp", r, s) if cc == c else None
    if k[0] in ("ring", "sqb"):
        return (k[0], k[2]) if k[1] == c else None
    if k[0] == "sqi":
        return ("sqi", k[2], k[3]) if k[1] == c else None
    if k[0] in ("mj", "mq"):
        return (k[0], k[2], k[3]) if k[1] == c else None
    return None


def tile_geom(c, L, core_xy):
    quads, inner = bsec.build([c])
    keys, Q = {}, []
    for q, _, _ in quads:
        Q.append([keys.setdefault(k, len(keys)) for k in q])
    inv = [None] * len(keys)
    for k, i in keys.items():
        inv[i] = k
    cache = {}
    X = np.array([coords(k, L, inner, cache, core_xy) for k in inv])
    return inv, np.array(Q), X


def tile_signature(c, L, core_xy):
    inv, Q, X = tile_geom(c, L, core_xy)
    cc = np.array(c, float) * PITCH
    bnd = sorted(set(np.concatenate([Q[:, [0, 1]], Q[:, [1, 2]], Q[:, [2, 3]], Q[:, [3, 0]]]).ravel().tolist()))
    # boundary geometry relative to the centre (+ the ring radius) identifies the tile
    e = collections.Counter()
    for q in Q:
        for a in range(4):
            e[frozenset((q[a], q[(a + 1) % 4]))] += 1
    bn = sorted({v for ed, n in e.items() if n == 1 for v in ed})
    sig = tuple(sorted(tuple(np.round(X[i] - cc, 5)) for i in bn)) + (round(ring_radius(c, L), 4), len(inv))
    return hash(sig)


def smooth_tile(args):
    c, L, core_xy = args
    inv, Q, X = tile_geom(c, L, core_xy)
    # winding from the top level (a clean raw layout), not from this level's own raw areas: a quad the
    # raw layout folds here must read as folded, or no smoother will ever untangle it
    _, Qt, Xt = tile_geom(c, LV["top"], core_xy)
    sg = np.sign(0.5 * np.sum(Xt[Qt][..., 0] * np.roll(Xt[Qt][..., 1], -1, 1) - np.roll(Xt[Qt][..., 0], -1, 1) * Xt[Qt][..., 1], 1))
    Q = np.array([q if s_ > 0 else q[::-1] for q, s_ in zip(Q, sg)])
    qm = QuadMesh.from_corners(np.c_[X, np.zeros(len(X))], Q)
    e = collections.Counter()
    for q in Q:
        for a in range(4):
            e[frozenset((q[a], q[(a + 1) % 4]))] += 1
    frozen = {v for ed, n in e.items() if n == 1 for v in ed}
    cc = np.array(c, float) * PITCH
    rings, lines = [], collections.defaultdict(list)
    for i, k in enumerate(inv):
        if isinstance(k[0], (float, np.floating)):
            if k not in ROLE or L[2] == "top" or c in CENTRAL:
                frozen.add(i); continue
            r, c2, s = ROLE[k]
            if r in ("Am", "Ap", "Im", "I0", "Ip"):
                jn = bsec.side_lines(c2, s)[1]
                lines[(jn[0], round(line_pos(jn, L), 6))].append(i)
            elif r in ("Jm", "Jp", "Km", "K0", "Kp"):
                frozen.add(i)
        elif k[0] == "ring":
            if ring_held(c):                                 # a template ring near the rim (no pin / post) is not held
                rings.append(i)
        elif k[0] == "X":
            frozen.add(i)
        elif k[0] == "mq":
            jn = bsec.side_lines(k[1], k[2])[1]
            lines[(jn[0], round(line_pos(jn, L), 6))].append(i)
    cons = [cn.Frozen(np.array(sorted(frozen)))]
    if rings:
        cons.append(cn.OnCurve(np.array(rings), cn.circle(np.array([cc[0], cc[1], 0.0]), ring_radius(c, L))))
    for (ax, v), ids in lines.items():
        ids = [i for i in ids if i not in frozen]
        if ids:
            o = np.array([v, 0.0, 0.0]) if ax == "x" else np.array([0.0, v, 0.0])
            d = np.array([0.0, 1.0, 0.0]) if ax == "x" else np.array([1.0, 0.0, 0.0])
            cons.append(cn.OnCurve(np.array(ids), cn.line(o, d)))
    nrm = np.tile([0.0, 0.0, 1.0], (len(Q), 1))
    res = cn.resolve(qm, cons)
    from q2 import corner_sj
    pre = smooth.untangle_schedule() if corner_sj(X, Q).min() <= 0 else []      # a folded start: untangle first
    out, _ = smooth.run_schedule(qm, res, pre + smooth.user_schedule(outer=8), normals=nrm)
    Y = np.asarray(out.points)[:, :2]
    if corner_sj(Y, Q).min() <= 0:                   # still folded: greedy local untangle on the constraints
        from q2 import untangle
        fr = np.ones(len(Y), bool); fr[sorted(frozen)] = False
        ringset = set(rings)
        lineof = {i: (ax, v) for (ax, v), ids in lines.items() for i in ids}

        def proj(i, p):
            if i in ringset:
                d = p - cc; return cc + ring_radius(c, L) * d / np.linalg.norm(d)
            if i in lineof:
                ax, v = lineof[i]; p = p.copy()
                p[0 if ax == "x" else 1] = v
            return p
        Y = untangle(Y, Q, fr, proj, target=0.1, rounds=300, grow=2)
    if corner_sj(Y, Q).min() <= 0.05 and L[2] != "top" and c not in CENTRAL:
        # concave: free the interior coordinate-keyed footprint nodes too (tile boundary stays)
        bnd = {v for ed, n in e.items() if n == 1 for v in ed}
        fz2 = [i for i in sorted(frozen) if i in bnd or inv[i][0] == "X"]
        cons2 = [cn.Frozen(np.array(fz2))] + cons[1:]
        res2 = cn.resolve(qm, cons2)
        o2, _ = smooth.run_schedule(qm, res2, smooth.user_schedule(outer=8), normals=nrm)
        Y2 = np.asarray(o2.points)[:, :2]
        if corner_sj(Y2, Q).min() > corner_sj(Y, Q).min():
            Y = Y2
    return {rel_key(inv[i], c): Y[i] - X[i] for i in range(len(inv)) if rel_key(inv[i], c) is not None}
