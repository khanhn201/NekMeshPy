"""Bottom-zone bands A..KN lofted between level positions (pos_<level>.npz) with per-band fluid
selections on the one global section.  Each band is saved as band_<name>.pkl (checkpoint).
usage: python bands.py [band ...]"""
import os, sys, pickle, collections, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import levels as lv
from levels import bsec, rules, secA, PITCH, R_W, CENTRAL
from q2d import Sec
from nekmeshpy import quadmesh, hexmesh
from nekmeshpy.core.fields import gll_nodes

ORDER = int(os.environ.get("ORDER", "2"))
T = pickle.load(open(os.path.join(HERE, "topo.pkl"), "rb"))
QUADS, INNER, OUTER = T["quads"], T["inner"], T["outer"]
KEYS = {}
for q, _, _ in QUADS:
    for k in q:
        KEYS.setdefault(k, len(KEYS))
INV = [None] * len(KEYS)
for k, i in KEYS.items():
    INV[i] = k
Q = np.array([[KEYS[k] for k in q] for q, _, _ in QUADS])
LAB = np.array([l for _, l, _ in QUADS])
CELL = [c for _, _, c in QUADS]
CORE = secA.CORE
CK = {secA.ckey(p): i for i, p in enumerate(CORE.X)}
SITES = lv.SITES
BORES = {(1, 1), (-1, 1), (-1, -1), (1, -1)}
ONWALL = {KEYS[k] for k in OUTER}

# rim (CAD periphery outside R692.15): two radial rows R_W -> R_R1 -> R_R2 on every wall edge
R_R1, R_R2 = 701.675, 704.85
_ec = collections.defaultdict(list)
for _qi, _q in enumerate(Q):
    for _a in range(4):
        _ec[frozenset((_q[_a], _q[(_a + 1) % 4]))].append(_qi)
WALL_E = [tuple(sorted(e)) for e, qs in _ec.items() if len(qs) == 1 and all(v in ONWALL for v in e)]
WALL_Q = {qi for qi, q in enumerate(Q) if any(v in ONWALL for v in q)}     # touching the wall
_wn = sorted({v for e in WALL_E for v in e})
RIM_IDX = {v: (len(KEYS) + 2 * k, len(KEYS) + 2 * k + 1) for k, v in enumerate(_wn)}
KEY_SITES = [(sx * a, sy * b) for a, b in ((3, 13), (13, 3), (6, 12), (12, 6)) for sx in (1, -1) for sy in (1, -1)]
R_KEYPOST = 12.70

# band: (bottom level, top level, layers, rule)
BANDS = [("A", "AB", "top", 1, "A"), ("B", "BC", "AB", 2, "B"), ("C", "CD", "BC", 1, "C"),
         ("D", "DE", "CD", 1, "D"), ("E", "EF", "DE", 1, "E"), ("F3", "F3F2", "EF", 1, "F"),
         ("F2", "F2F1", "F3F2", 1, "F"), ("F1", "F1F0", "F2F1", 1, "F"), ("F0", "F0H", "F1F0", 1, "F"),
         ("H", "HI", "F0H", 1, "H"), ("I", "IJ", "HI", 1, "G"), ("J", "JK", "IJ", 1, "G"),
         ("KN", "KN", "JK", 2, "G")]


def load(name):
    return np.load(os.path.join(HERE, "pos_%s.npz" % name))["X"]


def zfield(name, X):
    L = lv.LV[name]
    return np.full(len(X), L[1]) if L[1] is not None else lv.gz(np.hypot(*X.T))


def near_joint(v):
    m = int(round((v - 25.4) / PITCH))
    return m, 25.4 + PITCH * m


def in_yslot(x):
    m, j = near_joint(x); lo, hi = rules.y_slot(j); return lo < x < hi


def in_keypost(x, y):
    return any(np.hypot(x - a * PITCH, y - b * PITCH) < R_KEYPOST for a, b in KEY_SITES)


def rim_mask(name):
    """rim rows (rim1 = R_W..701.675, rim2 = 701.675..704.85) per band, in WALL_E order."""
    out = []
    Xr = load("top")
    for a, b in WALL_E:
        u = Xr[a] / np.linalg.norm(Xr[a]) + Xr[b] / np.linalg.norm(Xr[b]); u /= np.linalg.norm(u)
        p1, p2 = 0.5 * (R_W + R_R1) * u, 0.5 * (R_R1 + R_R2) * u
        if name == "B":                       # bars end at R701.675: slots only, then the can gap
            f1, f2 = in_yslot(p1[0]), True
        elif name == "C":                     # open rim R685.8 .. 704.85 (key posts block it)
            f1, f2 = not in_keypost(*p1), True
        elif name in ("F1", "F2", "F3"):      # rim groove R692.15 .. 704.85
            f1 = f2 = True
        else:
            f1 = f2 = False
        out += [f1, f2]
    return np.array(out, bool)


def in_xslot(y):
    m, j = near_joint(y); lo, hi = rules.x_slot(j); return lo < y < hi


def in_plate(x):
    m, j = near_joint(x)
    if not rules.is_plate_joint(j):
        return False
    lo, hi = rules.plate_edges(j); return lo < x < hi


def in_bar(y):
    m, j = near_joint(y)
    return lv.is_bar_joint(j) and abs(y - j) < rules.PLATE_HW


def in_rod(x, y):
    m, j = near_joint(x)
    if rules.is_plate_joint(j):
        return False
    jy = int(round(y / PITCH))
    if abs(y - jy * PITCH) > rules.ROD_HW or abs(x - j) > PITCH / 2 - rules.R_POST:
        return False
    il = int(np.floor(j / PITCH)); ir = il + 1
    return (il, jy) in SITES and (ir, jy) in SITES


def rim_out(site, rule):
    """a rim post that lies wholly in the wall below band A: its (virtual) disc is fluid."""
    return rule != "A" and np.hypot(*site) * PITCH - rules.R_POST > lv.R_W - lv.RIM_LENS


def fluid(rule):
    X = 0.5 * (MID_A + MID_B)
    cen = X[Q].mean(1)
    out = np.zeros(len(Q), bool)
    for i, (q, lab, c) in enumerate(QUADS):
        x, y = cen[i]
        bore = isinstance(c, tuple) and c[0] == "bore"
        central = c in CENTRAL
        site = c if (isinstance(c, tuple) and len(c) == 2 and isinstance(c[0], int)) else None
        if bore:
            if rule == "A":
                f = lab not in ("bk_AB", "bk_BC")
            elif rule in ("B", "C", "D", "E"):
                f = lab in ("th_hole", "bk_A")
            else:
                f = True
        elif rule == "A":
            f = not (central and lab != "chan") and not (lab == "disc" and site in SITES)
        elif rule in ("B", "C", "D"):
            if central:
                f = lab == "chan" and all(cc in CENTRAL for cc in [c])
                f = f and abs(x) < 50.8 and abs(y) < 50.8          # internal channels only
            else:
                f = in_yslot(x) if rule == "B" else in_xslot(y)
        elif rule == "E":
            f = (not (lab == "disc")) if central else in_xslot(y)
        elif rule in ("F", "F2"):
            if central:
                f = lab != "disc" and not in_plate(x)
            else:
                f = not (lab == "disc" and site in SITES and not rim_out(site, rule)) and not in_plate(x)
                if rule == "F2" and f and in_rod(x, y):
                    f = False
        elif rule == "H":
            f = not (lab == "disc" and site in SITES and not central and not rim_out(site, rule)) and not in_plate(x) and not in_bar(y)
        else:                                       # grid only
            f = not in_plate(x) and not in_bar(y)
        if RIM and rule in ("C", "D", "E") and not central and not bore and i in WALL_Q and not in_keypost(x, y):
            f = True                                # rim annulus R685.8 .. R692.15 (wall-adjacent quads)
        out[i] = f
    return out


RIM = bool(int(os.environ.get("RIM", "1")))
STRAIGHT_F = os.path.join(HERE, "straight.pkl")
STRAIGHT = pickle.load(open(STRAIGHT_F, "rb")) if os.path.exists(STRAIGHT_F) else set()


def section(X, top, rim=False):
    S = Sec()
    S.X = [tuple(p) for p in X]
    S.quads = [list(q) for q in Q]
    S.lab = list(LAB)
    on = lambda i, cc, r: abs(np.linalg.norm(X[i] - cc) - r) < 1e-7
    for q, lab, c in QUADS:
        for a in range(4):
            ka, kb = q[a], q[(a + 1) % 4]
            if frozenset((ka, kb)) in STRAIGHT and not top:
                continue
            ia, ib = KEYS[ka], KEYS[kb]
            ca, cb = CK.get(ka), CK.get(kb)
            if ca is not None and cb is not None:
                rec = CORE.cur.get((min(ca, cb), max(ca, cb)))
                if rec is not None and rec[2][0] == "arc":
                    cc = np.array(rec[2][1:3]); ra = np.linalg.norm(X[ia] - cc)
                    if abs(ra - np.linalg.norm(X[ib] - cc)) < 1e-6 and (top or ka not in bsec.FP_NODES):
                        S.arc(ia, ib, cc, ra)
                    continue
            if isinstance(ka, tuple) and ka[0] == "ring" and isinstance(kb, tuple) and kb[0] == "ring" and ka[1] == kb[1]:
                cc = np.array(ka[1], float) * PITCH
                rr = np.linalg.norm(X[ia] - cc)
                if on(ib, cc, rr):
                    S.arc(ia, ib, cc, rr)
                    continue
            if ia in ONWALL and ib in ONWALL and on(ia, np.zeros(2), R_W) and on(ib, np.zeros(2), R_W):
                S.arc(ia, ib, np.zeros(2), R_W)
    if rim:
        S.X = S.X + [None] * (2 * len(RIM_IDX))
        for v, (i1, i2) in RIM_IDX.items():
            u = X[v] / np.linalg.norm(X[v])
            S.X[i1] = tuple(R_R1 * u); S.X[i2] = tuple(R_R2 * u)
        for a, b in WALL_E:
            a1, a2 = RIM_IDX[a]; b1, b2 = RIM_IDX[b]
            S.quad((a, b, b1, a1), "rim1"); S.quad((a1, b1, b2, a2), "rim2")
            S.arc(a1, b1, np.zeros(2), R_R1); S.arc(a2, b2, np.zeros(2), R_R2)
    Xa = np.array(S.X)
    for kk in list(S.cur):
        a, b_, spec = S.cur[kk]
        e = S.eval_edge(a, b_, np.array([0.0, 1.0]))
        if np.abs(e[0] - Xa[a]).max() > 1e-6 or np.abs(e[1] - Xa[b_]).max() > 1e-6:
            del S.cur[kk]
    return S


def slice_qm(X, zspec, top, mask):
    qm = section(X, top, rim=RIM).to_quadmesh(ORDER, 0.0)
    if zspec[0] == "flat":
        qm = quadmesh.translate(qm, np.array([0.0, 0.0, zspec[1]]))
    else:                                         # KN: z = (1-f) gz(r) + f z_JK
        f, zt = zspec[1], zspec[2]
        qm = quadmesh.transform_fn(qm, lambda P: np.c_[P[:, :2], (1 - f) * lv.gz(np.hypot(P[:, 0], P[:, 1])) + f * zt])
    sub = quadmesh.select(qm, mask)
    return quadmesh.tag_edges(sub, quadmesh.boundary_edges(sub), "wall")


def build(name):
    global MID_A, MID_B
    nm, lo, hi, nz, rule = [b for b in BANDS if b[0] == name][0]
    XA, XB = load(lo), load(hi)
    MID_A, MID_B = XA, XB
    mask = np.r_[fluid(rule), rim_mask(name)] if RIM else fluid(rule)
    zlo = lv.LV[lo][1]; zhi = lv.LV[hi][1]
    zsp = (lambda f: ("flat", (1 - f) * zlo + f * zhi)) if zlo is not None else (lambda f: ("kn", f, zhi))
    slices = []
    g = gll_nodes(ORDER)
    fr = np.linspace(0, 1, nz + 1)
    sweep = []
    for k, f in enumerate(fr):
        X = (1 - f) * XA + f * XB
        slices.append(slice_qm(X, zsp(f), top=(hi == "top" and f == 1.0), mask=mask))
        if k < nz:
            mids = []
            for t in g[1:-1]:
                ft = f + t * (fr[k + 1] - f)
                Xm = (1 - ft) * XA + ft * XB
                mids.append(slice_qm(Xm, zsp(ft), top=False, mask=mask))
            sweep.append(mids)
    blk = hexmesh.loft(slices, sweep_nodes=sweep, element_tags="fluid", first_tag="%s_bot" % name, last_tag="%s_top" % name)
    print("band %s: %d hexes, %s" % (name, len(blk.hexes), hexmesh.quality_summary(blk)), flush=True)
    pickle.dump(blk, open(os.path.join(HERE, "band_%s.pkl" % name), "wb"))
    return blk


if __name__ == "__main__":
    for nm in sys.argv[1:] or [b[0] for b in BANDS]:
        build(nm)
