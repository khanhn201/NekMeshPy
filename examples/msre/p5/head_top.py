"""Upper head: the z=2070 core cap lofted to an inner dome (offset of knuckle/dish/flat), the
quads under the outlet pipe extruded up to the top plate, a square duct pushed out along +x for
the side outlet, then one skin layer from every free boundary face of that inner volume onto the
CAD surfaces.  -> head_top.pkl / head_top.vtu"""
import os, sys, pickle, time, collections
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
P1 = os.path.join(HERE, "..", "p1")
sys.path[:0] = [HERE, P1]
from q2d import Sec                                     # noqa: F401  (unpickling)
from nekmeshpy import quadmesh, hexmesh, linemesh
from nekmeshpy.linemesh import LineMesh
from nekmeshpy.pointmesh import PointMesh
from nekmeshpy.quadmesh import QuadMesh
from nekmeshpy.core.fields import gll_nodes
from scipy.spatial import cKDTree

ORDER = int(os.environ.get("ORDER", "2"))
Z0 = 2070.0                                   # core cap
RC = 704.85
CK, RK = np.array([646.556, 2019.62]), 88.867          # knuckle
CD, RD = np.array([-67.473, 854.55]), 1455.246          # dish
RJ = 696.39                                            # knuckle / dish junction
ZF, RF = 2277.08, 239.37                               # flat annulus, its outer radius
RP = 136.54                                            # outlet pipe
ZT = 2708.1                                            # top plate
ZB, RB, XE = 2561.9, 64.1, 635.0                       # branch axis height, radius, outlet plane
D0 = 10.0                                              # skin thickness
H_RIM = 9.0                                            # inner volume's rim height above the cap
K_DOME = 3
R_SEL = 118.0
R_PULL = 120.0                                         # outline pulled to this radius
LAM = 0.3                                              # outline radius: keep 30% of its jag
SW = 42.0                                              # half side of the branch duct
DZ_COL = 50.0
DX_DUCT = 20.0
G = gll_nodes(ORDER)


# ------------------------------------------------------------------ profile ---------------
def arc_pts(c, R, a0, a1, h=0.2):
    n = max(2, int(abs(a1 - a0) * R / h))
    a = np.linspace(a0, a1, n)
    return c + R * np.c_[np.cos(a), np.sin(a)]


a_j = np.arctan2(np.sqrt(RK ** 2 - (RJ - CK[0]) ** 2), RJ - CK[0])
a_f = np.arctan2(ZF - CD[1], RF - CD[0])
a_j2 = np.arctan2(np.sqrt(RD ** 2 - (RJ - CD[0]) ** 2), RJ - CD[0])
PROF = np.concatenate([arc_pts(CK, RK, 0.0, a_j), arc_pts(CD, RD, a_j2, a_f),
                       np.c_[np.linspace(RF, RP, 300), np.full(300, ZF)],
                       np.c_[np.full(1000, RP), np.linspace(ZF, ZT, 1000)],
                       np.c_[np.linspace(RP, 0, 300), np.full(300, ZT)]])
PTREE = cKDTree(PROF)


def closest(rz):
    _, i = PTREE.query(rz)
    best = PROF[i].copy(); bd = np.linalg.norm(rz - best, axis=1)
    for j in (-1, 1):
        a = PROF[np.clip(i, 0, len(PROF) - 1)]; b = PROF[np.clip(i + j, 0, len(PROF) - 1)]
        ab = b - a; L2 = np.maximum((ab ** 2).sum(1), 1e-30)
        t = np.clip(((rz - a) * ab).sum(1) / L2, 0, 1)
        p = a + t[:, None] * ab; d = np.linalg.norm(rz - p, axis=1)
        m = d < bd; best[m] = p[m]; bd[m] = d[m]
    return best


# inner dome curve: rim point (RC, Z0+H_RIM) -> offset knuckle/dish -> flat at ZF-D0
rim = np.array([RC, Z0 + H_RIM])
a_r = np.arctan2(rim[1] - CK[1], rim[0] - CK[0])
d_r = RK - np.linalg.norm(rim - CK)
kn = arc_pts(CK, 1.0, a_r, a_j, 0.002)                 # unit-radius angles
s_kn = np.linspace(0, 1, len(kn))
w = np.clip(s_kn / 0.6, 0, 1)
dk = d_r + (D0 - d_r) * (3 * w ** 2 - 2 * w ** 3)
kn_in = CK + (RK - dk)[:, None] * (kn - CK)
di = arc_pts(CD, RD - D0, a_j2, a_f, 0.5)
INNER = np.concatenate([kn_in, di[1:]])[::-1]          # from the flat end to the rim
INNER[0, 1] = ZF - D0
R_A = INNER[0, 0]
seg = np.linalg.norm(np.diff(INNER, axis=0), axis=1)
S_IN = np.concatenate([[0], np.cumsum(seg)]) / seg.sum()


def dome_top(r):
    """section radius r -> (rho, z) on the inner dome."""
    r = np.asarray(r, float)
    s = np.clip((r - R_A) / (RC - R_A), 0, 1)
    rho = np.where(r <= R_A, r, np.interp(s, S_IN, INNER[:, 0]))
    z = np.where(r <= R_A, ZF - D0, np.interp(s, S_IN, INNER[:, 1]))
    return rho, z


SHIFT = [None]


def dome_map(t):
    def f(P):
        r = np.hypot(P[:, 0], P[:, 1]); rho, z = dome_top(r)
        s = np.where(r > 1e-9, rho / np.maximum(r, 1e-12), 1.0)
        d = SHIFT[0](P) if SHIFT[0] is not None else 0.0
        x = (1 - t) * P[:, 0] + t * (P[:, 0] * s + d[:, 0] if SHIFT[0] is not None else P[:, 0] * s)
        y = (1 - t) * P[:, 1] + t * (P[:, 1] * s + d[:, 1] if SHIFT[0] is not None else P[:, 1] * s)
        return np.c_[x, y, (1 - t) * Z0 + t * z]
    return f


# ------------------------------------------------------------------ section ---------------
S = pickle.load(open(os.path.join(P1, "section.pkl"), "rb"))
lab = np.array(S.lab)
BORE = np.array([l.startswith("th_") or l.startswith("bk_") for l in lab])
full = S.to_quadmesh(ORDER)
sec = quadmesh.select(full, ~BORE)
P2 = np.asarray(sec.points)[:, :2]
C2 = np.asarray(sec.corners)
THIM = [((50.8, 50.8), 30.137), ((-50.8, 50.8), 27.051), ((-50.8, -50.8), 27.051), ((50.8, -50.8), 27.051)]


def column_mask():
    cen = P2[C2].mean(1)
    m = np.hypot(*cen.T) < R_SEL
    edge_q = collections.defaultdict(list)
    for qi, q in enumerate(C2):
        for a in range(4):
            edge_q[frozenset((q[a], q[(a + 1) % 4]))].append(qi)
    nb = -np.ones((len(C2), 4), int)
    for qi, q in enumerate(C2):
        for a in range(4):
            for qj in edge_q[frozenset((q[a], q[(a + 1) % 4]))]:
                if qj != qi:
                    nb[qi, a] = qj
    zone = np.where(np.abs(np.hypot(*cen.T) - R_SEL) < 30)[0]
    for _ in range(50):
        ch = 0
        for i in zone:
            d = [k for k in range(4) if nb[i, k] >= 0 and m[nb[i, k]] != m[i]]
            if len(d) >= 3 or (len(d) == 2 and d[1] - d[0] == 2):
                m[i] = not m[i]; ch += 1
            elif len(d) == 2 and not m[i]:             # concave corner of the column: fill it
                m[i] = True; ch += 1
        if not ch:
            break
    return m


def outline_of(sub):
    """ordered outer boundary loop of the column section (thimble holes excluded)."""
    Ps = np.asarray(sub.points)[:, :2]; Cs = np.asarray(sub.corners)
    ec = collections.Counter()
    for q in Cs:
        for a in range(4):
            ec[frozenset((q[a], q[(a + 1) % 4]))] += 1
    nb = collections.defaultdict(list)
    for e, c in ec.items():
        if c == 1:
            u, v = tuple(e)
            if np.hypot(*Ps[u]) > 110 and np.hypot(*Ps[v]) > 110:
                nb[u].append(v); nb[v].append(u)
    s0 = next(iter(nb)); loop = [s0]; prev = None
    while True:
        nx = [w_ for w_ in nb[loop[-1]] if w_ != prev]
        if not nx or nx[0] == s0:
            break
        prev = loop[-1]; loop.append(nx[0])
    P = Ps[loop]
    if 0.5 * np.sum(P[:, 0] * np.roll(P[:, 1], -1) - np.roll(P[:, 0], -1) * P[:, 1]) < 0:
        loop = loop[::-1]; P = P[::-1]
    th = np.unwrap(np.arctan2(P[:, 1], P[:, 0]))
    ta = th.copy()
    for _ in range(40):
        ext = np.concatenate([[th[-1] - 2 * np.pi], th, [th[0] + 2 * np.pi]])
        th = 0.5 * th + 0.25 * (ext[:-2] + ext[2:])
    return loop, P, ta, th


def top_shift():
    """analytic displacement field (polar): the column outline gets a light angle spread and its
    radius pulled toward R_SEL; the change fades out over W on either side of the outline."""
    sys.path.insert(0, os.path.join(HERE, "..", "p4"))
    from q2 import corner_sj
    cm = column_mask()
    col2 = quadmesh.select(sec, cm)
    loop, LP, ta, tt = outline_of(col2)
    r0 = np.hypot(*LP.T)
    r1 = R_PULL + LAM * (r0 - R_SEL)
    th = ta.copy()
    for _ in range(6):
        ext = np.concatenate([[th[-1] - 2 * np.pi], th, [th[0] + 2 * np.pi]])
        th = 0.5 * th + 0.25 * (ext[:-2] + ext[2:])
    a0 = np.mod(ta, 2 * np.pi); o = np.argsort(a0)
    A = a0[o]; DR = (r1 - r0)[o]; DT = (th - ta)[o]; R0 = r0[o]
    A_ = np.r_[A[-1] - 2 * np.pi, A, A[0] + 2 * np.pi]
    ip = lambda a, v: np.interp(np.mod(a, 2 * np.pi), A_, np.r_[v[-1], v, v[0]])
    W = 30.0

    def f(P):
        x, y = P[:, 0], P[:, 1]
        r = np.hypot(x, y); a = np.arctan2(y, x)
        ro = ip(a, R0)
        w = np.clip(1 - np.abs(r - ro) / W, 0, 1) * np.clip((r - 104.0) / 6.0, 0, 1)
        w = w * w * (3 - 2 * w)
        rn = r + w * ip(a, DR); an = a + w * ip(a, DT)
        return np.c_[rn * np.cos(an) - x, rn * np.sin(an) - y]
    X = P2 + f(np.c_[P2, np.zeros(len(P2))])
    print("top shift: outline %d nodes, r %.1f..%.1f -> %.1f..%.1f, min corner sj %.3f (was %.3f)" % (len(loop), r0.min(), r0.max(), r1.min(), r1.max(), corner_sj(X, C2).min(), corner_sj(P2, C2).min()), flush=True)
    return f


def straighten_bad(qm, thr=0.02, rounds=6):
    """curved-invalid quads of a warped section: their edges go straight between the corners
    (GLL spacing) and every quad touching a changed edge gets Coons interior nodes."""
    from nekmeshpy.quadmesh.query import element_blocks
    from nekmeshpy.core import conform
    from nekmeshpy.linemesh import LineMesh as _LM
    from nekmeshpy.quadmesh import QuadMesh as _QM
    n = ORDER + 1
    for _ in range(rounds):
        sj = np.asarray(quadmesh.scaled_jacobian(qm)); sj = sj.min(1) if sj.ndim > 1 else sj
        bad = np.where(sj <= thr)[0]
        if not len(bad):
            break
        lm = qm.line_mesh
        lines = np.asarray(lm.lines); pts = np.asarray(lm.points)
        E = np.unique(np.asarray(qm.quads)[bad].ravel())
        lint = np.asarray(lm.interior).copy()
        t = G[1:-1][:, None]
        for e in E:
            a, b = pts[lines[e, 0]], pts[lines[e, 1]]
            lint[e] = a + t * (b - a)
        q2 = _QM(_LM(lm.point_mesh, lines, lint, lm.element_tags), qm.quads, qm.orient, qm.interior, qm.element_tags)
        aff = np.where(np.isin(np.asarray(qm.quads), E).any(1))[0]
        B = element_blocks(q2)[aff].reshape(len(aff), n, n, 3)
        u = G[None, :, None]; v = G[:, None, None]
        Cn = np.empty_like(B)
        for k in range(len(aff)):
            X = B[k]
            Cn[k] = ((1 - u) * X[:, :1] + u * X[:, -1:] + (1 - v) * X[:1, :] + v * X[-1:, :]
                     - ((1 - u) * (1 - v) * X[0, 0] + u * (1 - v) * X[0, -1] + (1 - u) * v * X[-1, 0] + u * v * X[-1, -1]))
        slots = conform._interior_slots(2, ORDER)
        inter = np.asarray(qm.interior).copy()
        inter[aff] = Cn.reshape(len(aff), n * n, 3)[:, slots, :]
        qm = _QM(q2.line_mesh, qm.quads, qm.orient, inter, qm.element_tags)
    sj = np.asarray(quadmesh.scaled_jacobian(qm)); sj = sj.min(1) if sj.ndim > 1 else sj
    if (sj <= 0).any():
        print("  straighten_bad: %d quads still curved-invalid" % (sj <= 0).sum(), flush=True)
    return qm


LZ = 40.0


def Zs(z):
    """column height -> pipe-wall height: +D0 at the dome flat and at the top plate, ramped to
    identity within LZ (so the branch window is not shifted)."""
    z = np.asarray(z, float)
    zb0, zt0 = ZF - D0, ZT - D0
    return z + D0 * np.clip(1 - (z - zb0) / LZ, 0, 1)


def key2(p):
    return (round(float(p[0]), 5), round(float(p[1]), 5))


if __name__ == "__main__":
    t0 = time.time()
    SHIFT[0] = top_shift()
    # ---- dome
    sl, sw = [], []
    fr = np.linspace(0, 1, K_DOME + 1)
    for k, f in enumerate(fr):
        sl.append(straighten_bad(quadmesh.transform_fn(sec, dome_map(f))))
        if k < K_DOME:
            sw.append([straighten_bad(quadmesh.transform_fn(sec, dome_map(f + t * (fr[k + 1] - f)))) for t in G[1:-1]])
    dome = hexmesh.loft(sl, sweep_nodes=sw, element_tags="fluid", first_tag="", last_tag="")
    print("dome: %d hexes, corner %s (%.0fs)" % (len(dome.hexes), hexmesh.corner_summary(dome), time.time() - t0), flush=True)

    # ---- column
    cm = column_mask()
    col2 = quadmesh.transform_fn(quadmesh.select(sl[-1], cm), lambda P: np.c_[P[:, :2], np.zeros(len(P))])   # = the (straightened) dome top
    loop, LP, ta, tt = outline_of(col2)
    ZB0, ZT0 = ZF - D0, ZT
    Zs_inv = lambda z: z
    yg = RP * np.sin(tt)
    pe = [j for j in range(len(loop)) if np.cos(0.5 * (tt[j] + tt[(j + 1) % len(loop)])) > 0
          and abs(RP * np.sin(0.5 * (tt[j] + tt[(j + 1) % len(loop)]))) < SW]
    n_w = max(4, len(pe))
    w0, w1 = Zs_inv(ZB - SW), Zs_inv(ZB + SW)
    rows = list(np.linspace(ZB0, w0, max(2, int(round((w0 - ZB0) / DZ_COL))) + 1))
    rows += list(np.linspace(w0, w1, n_w + 1)[1:])
    rows += list(np.linspace(w1, ZT0, max(2, int(round((ZT0 - w1) / DZ_COL))) + 1)[1:])
    zmap = lambda z: (lambda P: np.c_[P[:, :2], np.full(len(P), z)])
    column = hexmesh.loft([quadmesh.transform_fn(col2, zmap(z)) for z in rows], element_tags="fluid", first_tag="", last_tag="")
    print("column: %d hexes, %d outline nodes, patch %d edges x %d rows (%.0fs)" % (len(column.hexes), len(loop), len(pe), n_w, time.time() - t0), flush=True)

    # theta targets for every node on the outline (corners + GLL edge nodes), by xy
    TH = {}
    for j, v in enumerate(loop):
        TH[key2(LP[j])] = tt[j]
    lm = col2.line_mesh
    lines = np.asarray(lm.lines); lpts = np.asarray(lm.points)[:, :2]; lint = np.asarray(lm.interior)[..., :2] if ORDER > 1 else None
    pos = {key2(LP[j]): j for j in range(len(loop))}
    patch_lines = []
    for li, (a, b) in enumerate(lines):
        ka, kb = key2(lpts[a]), key2(lpts[b])
        if ka in pos and kb in pos:
            ja, jb = pos[ka], pos[kb]
            if (jb - ja) % len(loop) not in (1, len(loop) - 1):
                continue
            for gi, gt in enumerate(G[1:-1]):
                TH[key2(lint[li, gi])] = TH[ka] + gt * ((TH[kb] - TH[ka] + np.pi) % (2 * np.pi) - np.pi)
            jj = ja if (jb - ja) % len(loop) == 1 else jb
            if jj in pe:
                patch_lines.append(li)
    thl = lambda P: np.array([TH[key2(p)] for p in P[:, :2]])

    # ---- branch duct
    # the patch chain as its own consistently directed LineMesh (outline order)
    pe_s = sorted(pe, key=lambda j: tt[j])
    pts, li_int = [], []
    lkey = {}
    for li, (a, b) in enumerate(lines):
        lkey[(key2(lpts[a]), key2(lpts[b]))] = (li, False)
        lkey[(key2(lpts[b]), key2(lpts[a]))] = (li, True)
    for j in pe_s:
        ka, kb = key2(LP[j]), key2(LP[(j + 1) % len(loop)])
        li, rev = lkey[(ka, kb)]
        ii = np.asarray(lm.interior)[li]
        li_int.append(ii[::-1] if rev else ii)
    cpts = [np.asarray(lm.points)[np.where([key2(p) == key2(LP[pe_s[0]]) for p in lpts])[0][0]]]
    for j in pe_s:
        cpts.append(np.r_[LP[(j + 1) % len(loop)], 0.0])
    cpts[0] = np.r_[LP[pe_s[0]], 0.0]
    chain = LineMesh(PointMesh(np.array(cpts)), np.c_[np.arange(len(pe_s)), np.arange(1, len(pe_s) + 1)], np.array(li_int))
    wrows = [z for z in rows if w0 - 1e-9 <= z <= w1 + 1e-9]
    patch = quadmesh.loft([linemesh.transform_fn(chain, zmap(z)) for z in wrows])
    xs = list(np.linspace(RP + 14.0, XE, max(2, int(round((XE - RP - 14.0) / DX_DUCT))) + 1))
    st = lambda x: (lambda P: np.c_[np.full(len(P), x), RP * np.sin(thl(P)), Zs(P[:, 2])])
    duct = hexmesh.loft([patch] + [quadmesh.transform_fn(patch, st(x)) for x in xs], element_tags="fluid", first_tag="", last_tag="outlet")
    cs = hexmesh.corner_summary(duct)
    if cs.n_inverted:
        duct = hexmesh.loft([quadmesh.transform_fn(patch, st(x)) for x in xs[::-1]] + [patch],
                            element_tags="fluid", first_tag="outlet", last_tag="")
    print("duct: %d hexes, corner %s (%.0fs)" % (len(duct.hexes), hexmesh.corner_summary(duct), time.time() - t0), flush=True)
    inner = hexmesh.merge([dome, column, duct])
    print("inner: %d hexes %s (%.0fs)" % (len(inner.hexes), hexmesh.topology_report(inner), time.time() - t0), flush=True)
    pickle.dump(dict(inner=inner, loop=loop, LP=LP, ta=ta, tt=tt, TH=TH, rows=rows, w=(w0, w1),
                     ysq=(yg[[j for j in pe]].min(), yg[[j for j in pe]].max())), open(os.path.join(HERE, "inner.pkl"), "wb"))
