"""Global 2D quad section of the MSRE core (graphite + channels + thimbles + bays + gap)."""
import pickle
import numpy as np
from geom import *
from q2d import Sec

D = [np.array(d, float) for d in ((1, 0), (0, 1), (-1, 0), (0, -1))]
LAT = pickle.load(open("lattice.pkl", "rb"))
CELLS, GROOVE = dict(LAT["cells"]), dict(LAT["groove"])


def fix_half_grooves():
    """half grooves belong to the full cell (lattice.py's nearest-midpoint pick can land on the cut side)."""
    for (c, s), g in list(GROOVE.items()):
        if g != "half" or CELLS.get(c) == "full":
            continue
        n = (c[0] + int(D[s][0]), c[1] + int(D[s][1]))
        del GROOVE[(c, s)]
        GROOVE[(n, (s + 2) % 4)] = "half"


fix_half_grooves()


def frame(c, s):
    n = D[s]
    t = np.array([-n[1], n[0]])
    c = np.asarray(c, float) * PITCH
    return lambda u, v: c + u * t + v * n


def circ_hit(p0, p1, R):
    """point on segment p0->p1 at radius R (one crossing assumed)."""
    d = p1 - p0
    a, b, cc = d @ d, 2 * p0 @ d, p0 @ p0 - R * R
    disc = np.sqrt(b * b - 4 * a * cc)
    for t in ((-b + disc) / (2 * a), (-b - disc) / (2 * a)):
        if -1e-9 <= t <= 1 + 1e-9:
            return p0 + t * d
    raise ValueError("no crossing")


def on_circle_between(pa, pb, R, f):
    a0, a1 = np.arctan2(pa[1], pa[0]), np.arctan2(pb[1], pb[0])
    d = (a1 - a0 + np.pi) % (2 * np.pi) - np.pi
    a = a0 + f * d
    return R * np.array([np.cos(a), np.sin(a)])


def side_kind(c, s):
    if (c, s) in GROOVE:
        return "G"
    n = (c[0] + int(D[s][0]), c[1] + int(D[s][1]))
    if CELLS.get(n) in ("thimble", "basket"):
        return "T"
    return "F"


S = Sec()
O = np.zeros(2)


def post_corner(c, k):
    """post corner in the direction of cell corner k (between side k and k+1)."""
    return frame(c, k)(POST, POST)


def cell_corner(c, k):
    return frame(c, k)(H, H)


def ridge_node(c, k, frac=None):
    """node on the pyramid ridge from the apex towards cell corner k (RF, or RF2 inner)."""
    fr = RF if frac is None else frac
    return S.P(frame(c, k)(fr * H, fr * H))


def groove(f, lab="chan"):
    """half-groove on the side of frame f, 10 fluid quads: a joint strip above a straight
    chain (A-,I-,I0,I+,A+) at depth KB, and a D region between the chain and the groove
    curve (J-,A-,B-,M,B+,A+,J+).  Below z=411.29 the chain slides onto the bar-slot edge
    (bottom zone), which is why it is straight and ends on the end arcs."""
    ua = GL + np.sqrt(GR ** 2 - KB ** 2)
    Jm, Jp = S.P(f(-(GL + GR), H)), S.P(f(GL + GR, H))
    Km, K0, Kp = S.P(f(-IU, H)), S.P(f(0, H)), S.P(f(IU, H))
    Am, Ap = S.P(f(-ua, H - KB)), S.P(f(ua, H - KB))
    Bm, Bp = S.P(f(-GL, H - GR)), S.P(f(GL, H - GR))
    M = S.P(f(0, H - GR))
    Im, I0, Ip = S.P(f(-IU, H - KB)), S.P(f(0, H - KB)), S.P(f(IU, H - KB))
    Ym, Y0, Yp = S.P(f(-YU, H - YD)), S.P(f(0, H - YD)), S.P(f(YU, H - YD))
    cm, cp = f(-GL, H), f(GL, H)
    S.arc(Jm, Am, cm, GR); S.arc(Am, Bm, cm, GR)
    S.arc(Bp, Ap, cp, GR); S.arc(Ap, Jp, cp, GR)
    for q in ((Jm, Am, Im, Km), (Km, Im, I0, K0), (K0, I0, Ip, Kp), (Kp, Ip, Ap, Jp),
              (Am, Im, Ym, Bm), (Im, I0, Y0, Ym), (Ym, Y0, M, Bm),
              (Ip, Ap, Bp, Yp), (I0, Ip, Yp, Y0), (Y0, Yp, Bp, M)):
        S.quad(q, lab)
    return Jm, Am, Bm, M, Bp, Ap, Jp


def side_quads(c, s, inner):
    """one side of a stringer cell.  inner = ("apex", O) for a pyramid-topped cell:
    the side is the pyramid face triangle (Cm, Cp, O); both ridges carry two nodes
    (RF, RF2), so every quad lies in one face.  inner = ("post", P5) for a central
    (round-post) cell, P5 = the 5 post-square nodes along this side."""
    f = frame(c, s)
    Cm, Cp = S.P(f(-H, H)), S.P(f(H, H))
    kind = side_kind(c, s)
    if kind == "G":
        mid = groove(f, "hchan" if GROOVE[(c, s)] == "half" else "chan")
    elif kind == "T":
        e = np.sqrt(R_OPEN ** 2 - H ** 2)
        tc = f(0, 2 * H)
        us = [-e, -XU1, -XU2, 0.0, XU2, XU1, e]
        mid = [S.P(f(u, 2 * H - np.sqrt(R_OPEN ** 2 - u ** 2))) for u in us]
        for a_, b_ in zip(mid[:-1], mid[1:]):
            S.arc(a_, b_, tc, R_OPEN)
    else:
        assert inner[0] == "apex"
        F0 = S.P(f(0, H))
        Rm, Rp, O_ = ridge_node(c, (s - 1) % 4), ridge_node(c, s), inner[1]
        R2m, R2p = ridge_node(c, (s - 1) % 4, RF2), ridge_node(c, s, RF2)
        Xi, Yi = S.P(f(0, XF * H)), S.P(f(0, YF * H))
        for q in ((Cm, F0, Xi, Rm), (F0, Cp, Rp, Xi), (Rm, Xi, Yi, R2m), (Xi, Rp, R2p, Yi),
                  (R2m, Yi, R2p, O_)):
            S.quad(q, "gr")
        return
    a1, a2, a3, a4, a5, a6, a7 = mid
    if inner[0] == "apex":
        Rm, Rp, O_ = ridge_node(c, (s - 1) % 4), ridge_node(c, s), inner[1]
        R2m, R2p = ridge_node(c, (s - 1) % 4, RF2), ridge_node(c, s, RF2)
        quads = ((Cm, a1, a2, Rm), (Rm, a2, a3, R2m), (R2m, a3, a4, O_),
                 (O_, a4, a5, R2p), (R2p, a5, a6, Rp), (a6, a7, Cp, Rp))
    else:
        p1, p2, p3, p4, p5 = inner[1]
        quads = ((Cm, a1, a2, p1), (a2, a3, p2, p1), (a3, a4, p3, p2),
                 (a4, a5, p4, p3), (a5, a6, p5, p4), (a6, a7, Cp, p5))
    for q in quads:
        S.quad(q, "gr")


def joint_nodes(c, s):
    """nodes along side s of cell c (CCW), including both corners."""
    f = frame(c, s)
    n = (c[0] + int(D[s][0]), c[1] + int(D[s][1]))
    g = GROOVE.get((c, s)) or GROOVE.get((n, (s + 2) % 4))
    us = [-H, H] if g is None else [-H, -(GL + GR), -IU, 0.0, IU, GL + GR, H]
    return [f(u, H) for u in us], us


def full_cell(c):
    """stringer cell.  Pyramid-topped cells are 4 face triangles meeting at the apex
    node; the 5 central stringers carry a round post (R_POST) instead."""
    cc = np.asarray(c, float) * PITCH
    if c not in CENTRAL:
        O_ = S.P(cc)
        for s in range(4):
            side_quads(c, s, ("apex", O_))
        return
    # post square, 4 segments a side, then a ring to the post circle and a structured disc
    ts = [-1.0, -0.55, 0.0, 0.55, 1.0]
    hq = RF * H                                   # post square through the ridge nodes
    sq = []
    for s in range(4):
        f = frame(c, s)
        P5 = [S.P(f(t * hq, hq)) for t in ts]
        side_quads(c, s, ("post", P5))
        sq += P5[:-1]
    dirs = [(np.array(S.X[i]) - cc) / np.linalg.norm(np.array(S.X[i]) - cc) for i in sq]
    ring = [S.P(cc + R_POST * d) for d in dirs]
    n = len(sq)
    for k in range(n):
        j = (k + 1) % n
        S.arc(ring[k], ring[j], cc, R_POST)
        S.quad((sq[k], sq[j], ring[j], ring[k]), "gr")
    s0 = 0.45 * R_POST / np.sqrt(2)
    f0 = frame(c, 0)
    t0, n0 = f0(1, 0) - f0(0, 0), f0(0, 1) - f0(0, 0)
    g = [[S.P(cc + s0 * (ui * t0 + vj * n0)) for vj in ts] for ui in ts]
    for i in range(4):
        for j in range(4):
            S.quad((g[i][j], g[i + 1][j], g[i + 1][j + 1], g[i][j + 1]), "post")
    inner = []
    for p in (cc + R_POST * d for d in dirs):
        d = p - cc
        inner.append(S.P(cc + d * s0 / np.abs(np.array([d @ t0, d @ n0])).max()))
    for k in range(n):
        j = (k + 1) % n
        S.quad((ring[k], ring[j], inner[j], inner[k]), "post")


def corners_out(c):
    return [k for k in range(4) if np.hypot(*cell_corner(c, k)) > RG]


def strip2(poly, arcn):
    """two-row strip from an inner node chain to an arc chain (on RG) of equal length;
    the middle row puts a node at the midpoint of both end edges (bay walls)."""
    mid = [S.P(0.5 * (np.array(S.X[p]) + np.array(S.X[q]))) for p, q in zip(poly, arcn)]
    for i in range(len(poly) - 1):
        S.arc(arcn[i], arcn[i + 1], O, RG)
        S.quad((poly[i], poly[i + 1], mid[i + 1], mid[i]), "gr")
        S.quad((mid[i], mid[i + 1], arcn[i + 1], arcn[i]), "gr")


def cut_cell(c):
    """edge stringer: keep only its intact grooved pyramid triangles (the rest is
    covered by the edge loft in ring())."""
    out = corners_out(c)
    O_ = S.P(np.asarray(c, float) * PITCH)
    if len(out) == 1:
        k = out[0]
        keep = [(k + 2) % 4, (k + 3) % 4]
    else:
        k = out[0] if (out[0] + 1) % 4 == out[1] else out[1]
        keep = [(k + 3) % 4] if side_kind(c, (k + 3) % 4) == "G" else []
    for s in keep:
        side_quads(c, s, ("apex", O_))


def thimble(c, rt):
    cc = np.asarray(c, float) * PITCH
    e = np.sqrt(R_OPEN ** 2 - H ** 2)
    outer = []
    for s in range(4):
        f = frame(c, s)
        outer += [("C", f(-H, H)), ("E", f(-e, H))]
        outer += [("X", f(u, np.sqrt(R_OPEN ** 2 - u ** 2))) for u in (-XU1, -XU2, 0.0, XU2, XU1)]
        outer += [("E", f(e, H))]
    oid = [S.P(p) for _, p in outer]
    inner = [cc + rt * (p - cc) / np.linalg.norm(p - cc) for _, p in outer]
    iid = [S.P(p) for p in inner]
    n = len(outer)
    for i in range(n):
        j = (i + 1) % n
        ki, kj = outer[i][0], outer[j][0]
        if not ({ki, kj} == {"C", "E"}):
            S.arc(oid[i], oid[j], cc, R_OPEN)
        S.arc(iid[i], iid[j], cc, rt)
        S.quad((oid[i], oid[j], iid[j], iid[i]), "thim")
    # bore interior (below the thimble cap / around the basket bottom): rings at the
    # site's radii, innermost a structured square; node angles follow the ring above
    if rt == R_BASKET:
        radii, labs, sq = [7.5038, 12.7, 25.79], ["bk_A", "bk_AB", "bk_BC", "bk_CD"], 3.5
    else:
        radii, labs, sq = [12.7, 22.0], ["th_hole", "th_disc", "th_disc"], 6.0
    ang = [np.arctan2(*(p - cc)[::-1]) for _, p in outer]
    prev = iid
    for r, lab in zip(radii[::-1], labs[:0:-1]):
        cur = [S.P(cc + r * np.array([np.cos(a), np.sin(a)])) for a in ang]
        for i in range(n):
            j = (i + 1) % n
            S.arc(cur[i], cur[j], cc, r)
            S.quad((prev[i], prev[j], cur[j], cur[i]), lab)
        prev = cur
    # square: side nodes where the rays cross it, structured inside
    us = []
    for k in range(8):
        a = ang[k] - ang[4]                       # relative to the side normal of side 0
        us.append(np.tan(a))
    us = np.array(us + [-us[0]])                  # corner .. corner across one side
    us = np.sort(us)
    f0 = frame(c, 0)
    t0 = (f0(1, 0) - f0(0, 0)); n0 = (f0(0, 1) - f0(0, 0))
    grid = [[S.P(cc + sq * (ui * t0 + vj * n0)) for vj in us] for ui in us]
    m = len(us)
    for i in range(m - 1):
        for j in range(m - 1):
            S.quad((grid[i][j], grid[i + 1][j], grid[i + 1][j + 1], grid[i][j + 1]), labs[0])
    sqb = []
    for p in (cc + 1.5 * sq * np.array([np.cos(a), np.sin(a)]) for a in ang):
        d = p - cc
        q = cc + d * sq / np.abs(np.array([d @ t0, d @ n0])).max()
        sqb.append(S.P(q))
    for i in range(n):
        j = (i + 1) % n
        S.quad((sqb[i], sqb[j], prev[j], prev[i]), labs[0])


def ring():
    """edge: loft every outline node of the stringer grid radially onto RG (one quad
    strip of graphite), then one gap quad from RG to RC."""
    Q = np.array(S.quads)
    X = np.array(S.X)
    e = np.concatenate([Q[:, [0, 1]], Q[:, [1, 2]], Q[:, [2, 3]], Q[:, [3, 0]]])
    k = np.sort(e, 1)
    u, inv, cnt = np.unique(k, axis=0, return_inverse=True, return_counts=True)
    bnd = e[cnt[inv.ravel()] == 1]
    bnd = bnd[np.hypot(*X[bnd[:, 0]].T) > 500]
    nxt = {a: b for a, b in bnd}
    start = int(bnd[0, 0])
    loop = [start]
    while True:
        b = nxt[loop[-1]]
        if b == start:
            break
        loop.append(b)
    ang = np.unwrap(np.arctan2(X[loop, 1], X[loop, 0]))
    d = np.diff(np.concatenate([ang, [ang[0] + 2 * np.pi]]))
    if (d <= 0).any():
        raise RuntimeError("outline not star-shaped at %s" % X[np.array(loop)[np.where(d <= 0)[0]]])
    g = [S.P(RG * X[i] / np.hypot(*X[i])) for i in loop]
    o = [S.P(RC * X[i] / np.hypot(*X[i])) for i in loop]
    n = len(loop)
    for t in range(n):
        j = (t + 1) % n
        S.arc(g[t], g[j], O, RG)
        S.arc(o[t], o[j], O, RC)
        S.quad((loop[t], loop[j], g[j], g[t]), "gr")
        S.quad((g[t], g[j], o[j], o[t]), "gap")


def build():
    for c, t in CELLS.items():
        if t == "full":
            full_cell(c)
    for c, t in CELLS.items():
        if t == "cut":
            cut_cell(c)
    for c, rt in THIMBLES.items():
        thimble(c, rt)
    ring()
    return S


if __name__ == "__main__":
    import collections
    S = build()
    print("nodes", len(S.X), "quads", len(S.quads), collections.Counter(S.lab))
    pickle.dump(S, open("section.pkl", "wb"))
