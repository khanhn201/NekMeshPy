"""Lower head with the 48 vanes as a lofted skin -- no pillows.  -> head_L2_vloft.pkl / .vtu

2D: per vane the corridor is the section cells within DELTA of the vane footprint (plus the cells it
straddles).  The stair it leaves is smoothed by guarded, vectorised Laplacian sweeps: outline nodes toward
the middle of their outline neighbours, every free node within 40 mm toward the middle of its neighbours;
a move is shortened until no quad -- nor its halfway blend back to the band positions -- drops below SJMIN,
no edge below EMIN, and no outline segment comes within DELTA of the vane.  Slivers of the raw section are
polished, then every outline node maps onto the vane footprint in order (closest-point parameter, isotonic,
with an ALPHA blend of uniform spacing).
3D: L2a = the smoothed section from the shell S0 up to S1' (HSK above the vane top) minus the corridors;
L2b = smoothed -> band positions from S1' to S2; one skin layer from the carved surface onto the rounded
vane (side nodes from their column's mapped point, blended up the column onto the rounding; top nodes by
closest point), the lofted surface then relaxed along the vane (SKSM guarded sweeps)."""
import os, sys, pickle, time, collections
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ["ORDER"] = os.environ.get("L2_ORDER", "1")   # the lower head is built linear (.re2 is linear)
import head as HD
import head3d as H3
import vanes as V
from head import R0V, VHW, NV, R_VES
from q2 import corner_sj, untangle
from nekmeshpy import quadmesh, hexmesh, HexMesh
from nekmeshpy.core.fields import gll_nodes
from nekmeshpy.hexmesh.quality import corner_scaled_jacobian as hex_csj
from nekmeshpy.linemesh import LineMesh
from nekmeshpy.pointmesh import PointMesh
from nekmeshpy.quadmesh import QuadMesh

ORDER = int(os.environ["ORDER"])
DELTA = float(os.environ.get("DELTA", "2.5"))
HSK = float(os.environ.get("HSK", "5.0"))
NA, NB = int(os.environ.get("NA", "3")), int(os.environ.get("NB", "2"))
SM2 = int(os.environ.get("SM2", "10"))
GMAX = float(os.environ.get("GMAX", "0"))           # widest side-skin quad on a rounded vane end (mm of arc)
KCAP = 20.0
MIN_DS = float(os.environ.get("MIN_DS", "0"))       # min spacing of the side-skin dups along the vane footprint (mm)
S2T = float(os.environ.get("S2T", "0.25"))
S2T_V = {int(a): float(b) for a, b in (kv.split(":") for kv in os.environ.get("S2T_V", "").split(",") if kv)}   # per-vane override "7:0.25,19:0.3"           # 2D side-skin quality target (the 3D layering costs a little)            # Taubin passes on each outline chain
t0 = time.time()
_ug = np.linspace(R0V + 50, R_VES, 4000)
_h = HD.s1(_ug) - HD.s0(_ug)
UEND = float(_ug[np.argmax(_h < 2 * VHW)]) if (_h < 2 * VHW).any() else R_VES
UEND = min(UEND, HD.R_MID - DELTA - 1.0)         # ... and inside the annulus' outer row (corridor = closed loop)
UW = UEND - VHW                                   # vane ends (rounded) where it is still 2*VHW tall
TH = np.radians(7.5 * np.arange(NV))


def s1p(r):
    return HD.s1(r) + np.minimum(HSK, 0.4 * (HD.s2(r) - HD.s1(r)))


def vframe2(p):
    k = np.round(np.arctan2(p[:, 1], p[:, 0]) / np.radians(7.5)).astype(int) % NV
    t = np.c_[np.cos(TH[k]), np.sin(TH[k])]
    return t, np.c_[-t[:, 1], t[:, 0]]


def sd2(p):
    """plan distance to the rounded vane footprint (negative inside), and its outward unit gradient."""
    t, n = vframe2(p)
    u = np.einsum("ij,ij->i", p, t)
    m = np.clip(u, R0V + VHW, UW)[:, None] * t
    d = p - m; L = np.linalg.norm(d, axis=1); L = np.where(L < 1e-12, 1.0, L)
    return L - VHW, d / L[:, None]


def medial(p):
    t, n = vframe2(p[:, :2])
    u = np.einsum("ij,ij->i", p[:, :2], t)
    uc = np.clip(u, R0V + VHW, UW)
    return np.c_[uc[:, None] * t, np.minimum(p[:, 2], HD.s1(uc) - VHW)]


def sdist(p):
    return np.linalg.norm(p - medial(p), axis=1) - VHW


def onvane(p):
    m = medial(p); d = p - m; L = np.linalg.norm(d, axis=1); L = np.where(L < 1e-12, 1.0, L)
    return m + VHW * d / L[:, None]


# ================================================================== 2D =====================
HS = H3.HS
cx = V.Ctx(HS["quads"], HS["keys"], HS["X_kn"])
Xk = cx.X.copy()
nQ = len(cx.Q)


def nb_out(sel, i):
    return [k for k in range(4) if cx.nb[i, k] >= 0 and sel[cx.nb[i, k]] != sel[i]]


cd = sd2(cx.X[cx.Q].reshape(-1, 2))[0].reshape(-1, 4)
taken = np.zeros(nQ, bool)
chains = []
for kv in range(NV):
    t = np.array([np.cos(TH[kv]), np.sin(TH[kv])]); n = np.array([-t[1], t[0]])
    ang = np.abs(((np.arctan2(cx.cen[:, 1], cx.cen[:, 0]) - TH[kv] + np.pi) % (2 * np.pi)) - np.pi)
    mine = ang < np.radians(3.75)
    Pq = cx.X[cx.Q]; pn_ = Pq @ n; pt_ = Pq @ t
    straddle = (pn_.min(1) < VHW) & (pn_.max(1) > -VHW) & (pt_.max(1) > R0V) & (pt_.min(1) < UEND)
    core = ((cd.min(1) < 0) | straddle) & mine
    sel = ((cd.min(1) < DELTA) | straddle) & mine & ~taken
    zone = np.where(mine & (cd.min(1) < DELTA + 30))[0]
    for _ in range(300):
        ch = 0
        for i in zone:
            d = nb_out(sel, i)
            bad = len(d) >= 3 or (len(d) == 2 and (d[1] - d[0]) % 4 == 2)
            if not bad:
                continue
            if not sel[i] and not taken[i]:
                sel[i] = True; ch += 1
            elif sel[i] and not core[i] and len(d) >= 3:
                sel[i] = False; ch += 1
            elif sel[i]:
                cand = [cx.nb[i, k] for k in range(4) if cx.nb[i, k] >= 0 and not sel[cx.nb[i, k]] and not taken[cx.nb[i, k]]]
                if cand:
                    sel[min(cand, key=lambda j: cd[j].min())] = True; ch += 1
        deg = collections.Counter()
        for i in zone:
            for k in range(4):
                j = cx.nb[i, k]
                if j >= 0 and sel[i] and not sel[j]:
                    q = cx.Q[i]; deg[q[k]] += 1; deg[q[(k + 1) % 4]] += 1
        for v, c in deg.items():
            if c > 2:
                for qi in cx.inc[v]:
                    if not sel[qi] and not taken[qi]:
                        sel[qi] = True; ch += 1
        if not ch:
            break
    path = V.outline(cx, sel)
    cyc = False
    if path is None:
        nx_ = collections.defaultdict(list)
        for e, qs in cx.edge_q.items():
            if len(qs) == 2 and sel[qs[0]] != sel[qs[1]]:
                a_, b_ = tuple(e); nx_[a_].append(b_); nx_[b_].append(a_)
        if nx_ and all(len(v) == 2 for v in nx_.values()):
            st = next(iter(nx_)); lp = [st]; prev = None
            while True:
                nxt_ = [w for w in nx_[lp[-1]] if w != prev]
                prev = lp[-1]
                if nxt_[0] == st:
                    break
                lp.append(nxt_[0])
            if len(lp) == len(nx_):
                path, cyc = lp, True
    if path is None:
        nxt = collections.defaultdict(list)
        for e, qs in cx.edge_q.items():
            if len(qs) == 2 and sel[qs[0]] != sel[qs[1]]:
                u_, v2_ = tuple(e); nxt[u_].append(v2_); nxt[v2_].append(u_)
        ends = [v for v in nxt if len(nxt[v]) == 1]; big = [v for v in nxt if len(nxt[v]) > 2]
        print("vane %d: outline not one chain (%d ends at r %s, %d pinch nodes at %s)" % (kv, len(ends),
              np.round(np.hypot(*cx.X[ends].T), 1).tolist(), len(big), np.round(cx.X[big], 1).tolist()), flush=True); continue
    taken |= sel
    chains.append((kv, path, cyc))
print("vane outer end at r %.1f (vessel wall %.1f)" % (UEND, R_VES), flush=True)
print("2D corridors: %d vanes, %d cells (%.0fs)" % (len(chains), taken.sum(), time.time() - t0), flush=True)

# smooth the stair: guarded Laplacian sweeps -- every outline node moves toward the middle of its outline
# neighbours (kept >= DELTA off the vane, wall ends on the wall), every free node within 40 mm of a vane toward
# the middle of its edge neighbours; a move is shortened until no quad touching that node drops below SJMIN
X = Xk.copy()
r0 = np.hypot(*X.T)
onwall = np.abs(r0 - R_VES) < 1e-6
chain_nodes = np.array(sorted({v for _, p, _ in chains for v in p}))
print("outlines: %d closed loops, %d wall-to-wall chains" % (sum(c for _, _, c in chains), sum(not c for _, _, c in chains)), flush=True)
ec = collections.Counter(map(tuple, np.sort(np.concatenate([cx.Q[:, [0, 1]], cx.Q[:, [1, 2]], cx.Q[:, [2, 3]], cx.Q[:, [3, 0]]]), axis=1).tolist()))
bnd = np.zeros(len(X), bool); bnd[[v for e, c in ec.items() if c == 1 for v in e]] = True
nbrs = collections.defaultdict(set)
for (a_, b_) in ec:
    nbrs[a_].add(b_); nbrs[b_].add(a_)
dv, _ = sd2(Xk)
near = (dv < 40) & ~bnd
cset = set(chain_nodes.tolist())
inner = np.array([v for v in np.where(near)[0] if v not in cset])
SJMIN = float(os.environ.get("SJMIN", "0.1"))
EMIN = float(os.environ.get("EMIN", "0.5"))
NSW = int(os.environ.get("NSW", "60"))


TS = np.array([0.0, 0.2, 0.4, 0.6, 0.8])


def seg_dist(p, nb_):
    """smallest distance to the vane footprint over the node and its two outline segments (sampled)."""
    pts = np.concatenate([p[None] + TS[:, None] * (X[w] - p)[None] for w in nb_])
    return sd2(pts)[0].min()


def place(v, tgt, chain, nb_=()):
    old = X[v].copy()
    qs = cx.inc[v]
    floor = min(SJMIN, corner_sj(X, cx.Q[qs]).min())
    Ql = cx.Q[qs]; ids_, Qloc = np.unique(Ql, return_inverse=True); Qloc = Qloc.reshape(Ql.shape)
    fmid = min(SJMIN, corner_sj(0.5 * (X[ids_] + Xk[ids_]), Qloc).min())   # the L2b blend back to the band positions too
    nbl = list(nbrs[v])
    efl = min(EMIN, np.linalg.norm(X[nbl] - old, axis=1).min())   # no edge at this node shorter than EMIN (or than now)
    if chain:
        dfl = min(DELTA, seg_dist(old, nb_))              # outline (nodes AND segments) never nearer the vane than DELTA, or than now
    for t_ in (1.0, 0.5, 0.25, 0.1):
        p = old + t_ * (tgt - old)
        if chain and onwall[v]:
            p = R_VES * p / np.linalg.norm(p)
        if chain and seg_dist(p, nb_) < dfl:
            continue
        if np.linalg.norm(X[nbl] - p, axis=1).min() < efl:
            continue
        X[v] = p
        if corner_sj(X, cx.Q[qs]).min() >= floor and corner_sj(0.5 * (X[ids_] + Xk[ids_]), Qloc).min() >= fmid:
            return True
    X[v] = old
    return False


# ---- vectorised guarded sweeps: nodes coloured so no two of one colour share a quad; a colour moves at once
VEC = os.environ.get("VEC", "1") == "1"
if VEC:
    chain_prev, chain_next = {}, {}
    for kv, path, cyc in chains:
        n_ = len(path)
        for k_ in range(n_):
            if cyc or 0 < k_ < n_ - 1:
                chain_prev[path[k_]] = path[(k_ - 1) % n_]; chain_next[path[k_]] = path[(k_ + 1) % n_]
    act = np.array(sorted(set(chain_prev) | set(inner.tolist())))
    ischain = np.isin(act, list(chain_prev.keys()))
    # node -> quads (CSR) and node -> edge neighbours (padded)
    nq = collections.defaultdict(list)
    for qi, q in enumerate(cx.Q):
        for v in q:
            nq[v].append(qi)
    mq = max(len(nq[v]) for v in act); mn = max(len(nbrs[v]) for v in act)
    QI = np.full((len(act), mq), -1); NBR = np.full((len(act), mn), -1)
    for k_, v in enumerate(act):
        QI[k_, :len(nq[v])] = nq[v]; NBR[k_, :len(nbrs[v])] = sorted(nbrs[v])
    PV = np.array([chain_prev.get(v, v) for v in act]); NX = np.array([chain_next.get(v, v) for v in act])
    # greedy colouring (conflict = sharing a quad)
    colour = -np.ones(len(act), int); pos = {v: k_ for k_, v in enumerate(act)}
    for k_, v in enumerate(act):
        used = {colour[pos[w]] for q in nq[v] for w in cx.Q[q] if w in pos and colour[pos[w]] >= 0}
        c_ = 0
        while c_ in used:
            c_ += 1
        colour[k_] = c_
    groups = [np.where(colour == c_)[0] for c_ in range(colour.max() + 1)]
    print("  vectorised smoothing: %d nodes, %d colours" % (len(act), len(groups)), flush=True)

    def qsj(Xa, qi):                                    # (n, mq) corner sj of each node's quads; +inf for padding
        out = np.full(qi.shape, np.inf)
        m_ = qi >= 0
        out[m_] = corner_sj(Xa, cx.Q[qi[m_]])
        return out.min(1)

    def segd(Xa, idx, P):                               # (n,) smallest vane distance over node + both outline segments
        dd = sd2(P)[0]
        for nb_ in (PV[idx], NX[idx]):
            for t_ in TS[1:]:
                dd = np.minimum(dd, sd2(P + t_ * (Xa[nb_] - P))[0])
        return dd

    def emin(Xa, idx, P):
        nb_ = NBR[idx]; m_ = nb_ >= 0
        d_ = np.full(nb_.shape, np.inf)
        d_[m_] = np.linalg.norm(Xa[nb_[m_]] - np.repeat(P, m_.sum(1), axis=0), axis=1)
        return d_.min(1)

    XkM = Xk
    for sw in range(NSW):
        moved = 0
        for g_ in groups:
            v_ = act[g_]; ch = ischain[g_]
            old = X[v_].copy()
            Xm = 0.5 * (X + XkM)
            fl = np.minimum(SJMIN, qsj(X, QI[g_])); fm = np.minimum(SJMIN, qsj(Xm, QI[g_]))
            ef = np.minimum(EMIN, emin(X, g_, old))
            df = np.where(ch, np.minimum(DELTA, segd(X, g_, old)), -np.inf)
            nbm = NBR[g_]; mm = nbm >= 0
            cen = np.zeros_like(old); cen_n = mm.sum(1)
            np.add.at(cen, np.repeat(np.arange(len(g_)), cen_n), X[nbm[mm]])
            cen /= cen_n[:, None]
            tgt = np.where(ch[:, None], old + 0.5 * (0.5 * (X[PV[g_]] + X[NX[g_]]) - old), cen)
            done = np.zeros(len(g_), bool)
            for t_ in (1.0, 0.5, 0.25, 0.1):
                pend = ~done
                if not pend.any():
                    break
                P = old + t_ * (tgt - old)
                wl = ch & onwall[v_]
                P[wl] = R_VES * P[wl] / np.linalg.norm(P[wl], axis=1)[:, None]
                X[v_[pend]] = P[pend]
                Xm = 0.5 * (X + XkM)
                ok = pend.copy()
                ok &= qsj(X, QI[g_]) >= fl
                ok &= qsj(Xm, QI[g_]) >= fm
                ok &= emin(X, g_, P) >= ef
                ok &= ~ch | (segd(X, g_, P) >= df)
                X[v_[pend & ~ok]] = old[pend & ~ok]
                done |= ok
            moved += int(done.sum())
        if sw % 10 == 0 or sw == NSW - 1:
            s_ = corner_sj(X, cx.Q)
            print("  stair smoothing sweep %d: %d moves, section min corner sj %.3f (%.0fs)" % (sw, moved, s_.min(), time.time() - t0), flush=True)
else:
  for sw in range(NSW):
    moved = 0
    for kv, path, cyc in chains:
        n_ = len(path)
        for k_ in range(n_):
            if not cyc and (k_ == 0 or k_ == n_ - 1):
                continue
            v = path[k_]
            a_, b_ = path[(k_ - 1) % n_], path[(k_ + 1) % n_]
            lam = 0.5
            tgt = X[v] + lam * (0.5 * (X[a_] + X[b_]) - X[v])
            moved += place(v, tgt, True, (a_, b_))
    for v in inner:
        tgt = X[list(nbrs[v])].mean(0)
        moved += place(v, tgt, False)
    if sw % 10 == 0 or sw == NSW - 1:
        s_ = corner_sj(X, cx.Q)
        print("  stair smoothing sweep %d: %d moves, section min corner sj %.3f (%.0fs)" % (sw, moved, s_.min(), time.time() - t0), flush=True)
# slivers left over from the raw section (e.g. the 3:1 transition at the rim): a small greedy over the nodes
# of quads below POLISH, maximising the worst corner sj of each node's quads in X and in the halfway blend;
# outline nodes keep their segments >= DELTA off the vane
POLISH = float(os.environ.get("POLISH", "0.06"))
chain_nb = {}
for kv, path, cyc in chains:
    n_ = len(path)
    for k_ in range(n_):
        if cyc or 0 < k_ < n_ - 1:
            chain_nb[path[k_]] = (path[(k_ - 1) % n_], path[(k_ + 1) % n_])
rng_ = np.random.default_rng(0)


def worst(v):
    qs = cx.inc[v]; Ql = cx.Q[qs]
    ids_, Qloc = np.unique(Ql, return_inverse=True); Qloc = Qloc.reshape(Ql.shape)
    return min(corner_sj(X, Ql).min(), corner_sj(0.5 * (X[ids_] + Xk[ids_]), Qloc).min())


for rd in range(40):
    s_ = corner_sj(X, cx.Q)
    badq = np.where(s_ < POLISH)[0]
    if not len(badq):
        break
    nodes_ = {v for q in cx.Q[badq] for v in q}                     # the sliver's own nodes only
    gained = 0
    for v in nodes_:
        if bnd[v] or onwall[v] or (v in cset and v not in chain_nb):
            continue
        old = X[v].copy(); cur = worst(v); best, bp = cur, None
        hl = 0.5 * np.mean([np.linalg.norm(X[w] - old) for w in nbrs[v]])
        dfl = min(DELTA, seg_dist(old, chain_nb[v])) if v in chain_nb else None
        efl = min(EMIN, np.linalg.norm(X[list(nbrs[v])] - old, axis=1).min())
        for p in [X[list(nbrs[v])].mean(0)] + [old + hl * f * rng_.normal(size=2) for f in (0.05, 0.15, 0.3) for _ in range(8)]:
            if dfl is not None and seg_dist(p, chain_nb[v]) < dfl:
                continue
            if np.linalg.norm(X[list(nbrs[v])] - p, axis=1).min() < efl:
                continue
            X[v] = p
            w_ = worst(v)
            if w_ > best + 1e-6:
                best, bp = w_, p.copy()
        X[v] = bp if bp is not None else old
        gained += bp is not None
    if rd % 5 == 0:
        print("  sliver polish round %d: %d quads below %.2f, min %.3f, %d moves" % (rd, len(badq), POLISH, corner_sj(X, cx.Q).min(), gained), flush=True)
    if not gained:
        break
s_ = corner_sj(X, cx.Q)
P = X[cx.Q]; e = np.linalg.norm(np.roll(P, -1, 1) - P, axis=2).min(1)
print("section after stair smoothing: min corner sj %.3f, inverted %d, min edge %.3f (%.0fs)" % (s_.min(), (s_ <= 0).sum(), e.min(), time.time() - t0), flush=True)
for f in (0.25, 0.5, 0.75):
    print("  blend to band positions f=%.2f: min corner sj %.3f" % (f, corner_sj((1 - f) * X + f * Xk, cx.Q).min()), flush=True)
# ---- side skin in 2D: dup of every outline node on the rounded footprint, by arclength round the loop
def footprint(kv, m=4000):
    t = np.array([np.cos(TH[kv]), np.sin(TH[kv])]); n = np.array([-t[1], t[0]])
    a0, a1 = R0V + VHW, UW
    pts = []
    L = [a1 - a0, np.pi * VHW, a1 - a0, np.pi * VHW]
    tot = sum(L)
    for s_ in np.linspace(0, tot, m, endpoint=False):
        if s_ < L[0]:
            u, w = a0 + s_, -VHW
        elif s_ < L[0] + L[1]:
            ph = -np.pi / 2 + (s_ - L[0]) / VHW; u, w = a1 + VHW * np.cos(ph), VHW * np.sin(ph)
        elif s_ < L[0] + L[1] + L[2]:
            u, w = a1 - (s_ - L[0] - L[1]), VHW
        else:
            ph = np.pi / 2 + (s_ - L[0] - L[1] - L[2]) / VHW; u, w = a0 + VHW * np.cos(ph), VHW * np.sin(ph)
        pts.append(u * t + w * n)
    return np.array(pts)                               # ccw, starting at the inner end on the - side


def onfoot(p):
    t, n = vframe2(p)
    u = np.einsum("ij,ij->i", p, t)
    m = np.clip(u, R0V + VHW, UW)[:, None] * t
    d = p - m; L = np.linalg.norm(d, axis=1); L = np.where(L < 1e-12, 1.0, L)
    return m + VHW * d / L[:, None]


DUP = {}                                               # outline node -> dup xy
sheet_ok = True
for kv, path, cyc in chains:
    if not cyc:
        print("vane %d: outline reaches the wall, no 2D side skin" % kv, flush=True); sheet_ok = False; continue
    O = X[path]
    ar = (0.5 * np.sum(O[:, 0] * np.roll(O[:, 1], -1) - np.roll(O[:, 0], -1) * O[:, 1]))
    if ar < 0:
        path = path[::-1]; O = O[::-1]
    F = footprint(kv)
    tip = (R0V) * np.array([np.cos(TH[kv]), np.sin(TH[kv])])
    j0 = int(np.argmin(np.linalg.norm(O - tip, axis=1)))
    O = np.roll(O, -j0, 0); pth = np.roll(np.array(path), -j0)
    fi = int(np.argmin(np.linalg.norm(F - tip, axis=1))); F = np.roll(F, -fi, 0)
    sfp = np.r_[0, np.cumsum(np.linalg.norm(np.diff(np.r_[F, F[:1]], axis=0), axis=1))]; sfp /= sfp[-1]
    Fc = np.r_[F, F[:1]]
    m_ = len(O)
    # closest-point parameter on the footprint, unwrapped from the anchor, made monotone with a min gap
    sc = sfp[np.argmin(np.linalg.norm(O[:, None, :] - F[None, :, :], axis=2), axis=1)]
    sc = np.unwrap((sc - sc[0]) * 2 * np.pi) / (2 * np.pi)          # continuous around the loop, sc[0] = 0
    if sc[-1] < 0:                                                    # loop walks the other way round
        sc = -sc
    sc = np.clip(sc, 0.0, 1.0)
    # isotonic (pool-adjacent-violators): the closest monotone sequence
    vals, wts, cnt = [], [], []
    for v_ in sc:
        vals.append(v_); wts.append(1.0); cnt.append(1)
        while len(vals) > 1 and vals[-2] > vals[-1]:
            w_ = wts[-2] + wts[-1]; v2 = (vals[-2] * wts[-2] + vals[-1] * wts[-1]) / w_
            c2 = cnt[-2] + cnt[-1]
            vals[-2:] = [v2]; wts[-2:] = [w_]; cnt[-2:] = [c2]
    iso = np.repeat(vals, cnt)
    iso = (iso - iso[0]) / max(1.0, iso[-1] - iso[0] + 1.0 / m_) if iso[-1] >= 1.0 else iso - iso[0]
    ALPHA = float(os.environ.get("ALPHA", "0.05"))                   # blend of uniform spacing: a min gap ALPHA/m
    so = (1 - ALPHA) * iso + ALPHA * np.arange(m_) / m_
    D2 = np.c_[np.interp(so % 1.0, sfp, Fc[:, 0]), np.interp(so % 1.0, sfp, Fc[:, 1])]
    Y = np.r_[O, D2]
    Qs = np.array([[j, (j + 1) % m_, m_ + (j + 1) % m_, m_ + j] for j in range(m_)])
    a2 = 0.5 * np.sum(Y[Qs][..., 0] * np.roll(Y[Qs][..., 1], -1, 1) - np.roll(Y[Qs][..., 0], -1, 1) * Y[Qs][..., 1], 1)
    if np.median(a2) < 0:
        Qs = Qs[:, ::-1]
    s0_ = corner_sj(Y, Qs).min()
    # 1D optimisation of the dup parameters (order kept by construction), red-black sweeps
    from q2 import quad_csj
    Fx = lambda ss: np.c_[np.interp(ss % 1.0, sfp, Fc[:, 0]), np.interp(ss % 1.0, sfp, Fc[:, 1])]
    flip = Qs[0, 1] != 1                               # quads reversed for orientation
    def qsj(j_idx, s_j, s_n):
        """sj of quad j given dup params of j and j+1 (vectorised)."""
        P = np.stack([O[j_idx], O[(j_idx + 1) % m_], Fx(s_n), Fx(s_j)], 1)
        if flip:
            P = P[:, ::-1]
        return quad_csj(P)
    # simple loft: closest-point parameter on the footprint, made monotone with a min gap -- nothing else
    ss = so.copy()
    Fx = lambda ss: np.c_[np.interp(ss % 1.0, sfp, Fc[:, 0]), np.interp(ss % 1.0, sfp, Fc[:, 1])]
    Y = np.r_[O, Fx(ss)]
    s1_ = corner_sj(Y, Qs)
    if s1_.min() <= 0:
        sheet_ok = False
    _uu = Y[m_:] @ np.array([np.cos(TH[kv]), np.sin(TH[kv])])
    _cap = (_uu < R0V + VHW - 1e-6) | (_uu > UW + 1e-6)
    _dg = np.linalg.norm(np.diff(np.r_[Y[m_:], Y[m_:m_ + 1]], axis=0), axis=1)
    _cg = _dg[_cap | np.roll(_cap, -1)]
    _w = int(np.argmin(s1_)); _cw = Y[Qs[_w]].mean(0); _uw = float(_cw @ np.array([np.cos(TH[kv]), np.sin(TH[kv])]))
    _zn = "inner end" if _uw < R0V + VHW else ("outer end" if _uw > UW else "side")
    _nb = int((s1_ <= 0).sum())
    print("vane %d: dup gaps %.3f..%.2f mm, widest at an end %.2f, sheet min sj %.3f (%d bad; worst at u %.1f, %s)" % (kv, _dg.min(), _dg.max(), _cg.max() if len(_cg) else 0, s1_.min(), _nb, _uw, _zn), flush=True)
    for j, v in enumerate(pth):
        DUP[v] = Y[m_ + j]
    if os.environ.get("PLOTV") and int(os.environ["PLOTV"]) == kv:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        from matplotlib.collections import PolyCollection
        tt_ = np.array([np.cos(TH[kv]), np.sin(TH[kv])]); nn_ = np.array([-tt_[1], tt_[0]])
        rot = lambda P: np.c_[P @ tt_, P @ nn_]
        fig, axs = plt.subplots(3, 1, figsize=(18, 11))
        Y0 = np.r_[O, D2]
        for ax, (YY, nm) in zip(axs[:2], ((Y0, "initial dups"), (Y, "after untangle"))):
            sj_ = corner_sj(YY, Qs)
            pc = PolyCollection(rot(YY)[Qs], array=sj_, cmap="RdYlGn", clim=(-1, 1), edgecolors="k", linewidths=0.3)
            ax.add_collection(pc)
            qq = cx.Q[~taken & (np.hypot(*cx.cen.T) > 0)]
            ax.add_collection(PolyCollection(rot(X)[qq], facecolors="none", edgecolors="0.7", linewidths=0.2))
            ax.plot(*rot(F).T, "b-", lw=0.6)
            ax.set_xlim(R0V - 20, UW + 20); ax.set_ylim(-25, 25); ax.set_aspect("equal"); ax.set_title("vane %d: %s (colour: corner sj)" % (kv, nm))
        bad = np.where(corner_sj(Y, Qs) <= 0)[0]
        ax = axs[2]
        if len(bad):
            c_ = rot(Y)[Qs[bad[0]]].mean(0)
            ax.add_collection(PolyCollection(rot(Y)[Qs], array=corner_sj(Y, Qs), cmap="RdYlGn", clim=(-1, 1), edgecolors="k", linewidths=0.5))
            ax.add_collection(PolyCollection(rot(X)[cx.Q], facecolors="none", edgecolors="0.6", linewidths=0.3))
            ax.plot(*rot(F).T, "b-", lw=0.8)
            ax.set_xlim(c_[0] - 15, c_[0] + 15); ax.set_ylim(c_[1] - 6, c_[1] + 6); ax.set_aspect("equal"); ax.set_title("zoom at first inverted")
        plt.tight_layout(); plt.savefig(os.path.join(HERE, "img", "vskin_side2d.png"), dpi=110)
    if kv % 6 == 0 or s1_.min() <= 0.15 or kv == int(os.environ.get("PLOTV", "-1")):
        print("vane %d: 2D side skin %d quads, corner sj %.3f -> %.3f (min edge %.3f)" % (kv, m_, s0_, s1_.min(),
              np.linalg.norm(np.diff(np.r_[Y[m_:], Y[m_:m_ + 1]], axis=0), axis=1).min()), flush=True)
# corridor interiors: untangle with the outline fixed (they are fluid above the vane, in L2b)
inR = np.zeros(len(X), bool); inR[np.unique(cx.Q[taken])] = True
inF = np.zeros(len(X), bool); inF[np.unique(cx.Q[~taken])] = True
freeR = inR & ~inF & ~(bnd & ~onwall)
_sr = corner_sj(X, cx.Q[taken])
if _sr.min() < 0.1:
    X = untangle(X, cx.Q, freeR, lambda i, p: (R_VES * p / np.linalg.norm(p)) if onwall[i] else p, target=0.1, rounds=200, grow=2)
print("corridor cells: min corner sj %.3f -> %.3f" % (_sr.min(), corner_sj(X, cx.Q[taken]).min()), flush=True)
_dn, _ = sd2(X)
_close = np.where(inF & (_dn < 1.0))[0]
for i in _close[:8]:
    print("  fluid node %d at %.2f mm from the vane: key %s, chain %s, bnd %s, wall %s" % (i, _dn[i], str(list(cx.keys.keys())[i]) if False else "", i in cset, bnd[i], onwall[i]), flush=True)
print("  fluid nodes closer than 1 mm to a vane: %d" % len(_close), flush=True)
_sall = corner_sj(X, cx.Q)
for qi in np.argsort(_sall)[:5]:
    c_ = X[cx.Q[qi]].mean(0)
    print("  worst section quad %d: sj %.3f at r %.1f th %.1f deg, corridor %s" % (qi, _sall[qi], np.hypot(*c_), np.degrees(np.arctan2(c_[1], c_[0])), taken[qi]), flush=True)
print("2D side skins: %s; whole section min corner sj %.3f (%.0fs)" % ("all valid" if sheet_ok else "NOT all valid", _sall.min(), time.time() - t0), flush=True)
if not sheet_ok or _sall.min() <= 0:
    print("!! 2D stage not valid -- building 3D anyway to see where", flush=True)
np.savez(os.path.join(HERE, "vloft_sec.npz"), X=X, Xk=Xk, Q=cx.Q, taken=taken, chain=chain_nodes, bnd=bnd, onwall=onwall)
if os.environ.get("STOP2D"):
    sys.exit()

# ================================================================== 3D =====================
def secmesh(Xpos):
    qm = H3.section(Xpos, H3.HQ, H3.HLAB).to_quadmesh(ORDER, 0.0)
    return quadmesh.tag_edges(qm, quadmesh.boundary_edges(qm), "wall")


def loft(Xa, Xb, za, zb, nl):
    g = gll_nodes(ORDER)
    fr_ = np.linspace(0, 1, nl + 1)
    cache = {}

    def sl(f):
        if f not in cache:
            qm = secmesh((1 - f) * Xa + f * Xb)
            cache[f] = quadmesh.transform_fn(qm, lambda P: np.c_[P[:, :2], (1 - f) * za(np.hypot(P[:, 0], P[:, 1])) + f * zb(np.hypot(P[:, 0], P[:, 1]))])
        return cache[f]
    return hexmesh.loft([sl(f) for f in fr_], sweep_nodes=[[sl(fr_[k] + tt * (fr_[k + 1] - fr_[k])) for tt in g[1:-1]] for k in range(nl)],
                        element_tags="fluid", first_tag="wall", last_tag="wall")


A = loft(X, X, HD.s0, s1p, NA)
keepA = np.tile(~taken, NA)          # loft numbering: layer-major over the section's quads
assert len(A.hexes) == NA * nQ
_hc = np.asarray(A.points)[np.asarray(A.corners)].mean(1)[:, :2]
_qc = np.tile(X[cx.Q].mean(1), (NA, 1))
assert np.abs(_hc - _qc).max() < 1.0, "loft element order is not layer-major over the section quads (%.3g)" % np.abs(_hc - _qc).max()
A = hexmesh.select(A, keepA)
for nb_ in range(NB, NB + 4):                      # more layers until the smoothed -> band blend is untwisted
    Bk = loft(X, Xk, s1p, HD.s2, nb_)
    vb = hexmesh.validity(Bk)
    print("  L2b with %d layers: inverted %d, twisted %d" % (nb_, vb.inverted.size, vb.twisted.size), flush=True)
    if vb.ok:
        break
print("L2a %d hexes, L2b %d hexes (inverted %d, twisted %d) (%.0fs)" % (len(A.hexes), len(Bk.hexes), vb.inverted.size, vb.twisted.size, time.time() - t0), flush=True)
fluid = hexmesh.merge([A, Bk], clear_seam_tags=True)
blk_b = set()                                       # boundary of the UNCARVED volume (both blocks merged)
_full = hexmesh.merge([loft(X, X, HD.s0, s1p, NA), Bk], clear_seam_tags=True)
bq = hexmesh.boundary_mesh(_full)
for q in np.asarray(bq.points)[np.asarray(bq.corners)]:
    blk_b.add(tuple(np.round(np.sort(q, axis=0).ravel(), 6)))
Bq = hexmesh.boundary_mesh(fluid)
lm = Bq.line_mesh
PTS = np.asarray(lm.points); LINES = np.asarray(lm.lines)
LI = np.asarray(lm.interior); FI = np.asarray(Bq.interior)
FQ = np.asarray(Bq.quads); FC = np.asarray(Bq.corners)
skin = np.array([tuple(np.round(np.sort(q, axis=0).ravel(), 6)) not in blk_b for q in PTS[FC]])
sk_nodes = np.unique(FC[skin])
r_ = np.hypot(PTS[:, 0], PTS[:, 1])
on_shell = np.abs(PTS[:, 2] - HD.s0(r_)) < 1e-6
on_wall = np.abs(r_ - R_VES) < 1e-6
print("skin faces: %d; carved-surface distance to vane min %.2f" % (skin.sum(), sdist(PTS[sk_nodes]).min()), flush=True)


def constrain(P, ids):
    Q = onvane(P)
    sh, wl = on_shell[ids], on_wall[ids]
    if wl.any():
        rr = np.hypot(Q[wl, 0], Q[wl, 1]); Q[wl, :2] *= (R_VES / rr)[:, None]
    if sh.any():
        Q[sh, 2] = HD.s0(np.hypot(Q[sh, 0], Q[sh, 1]))
    return Q


T = PTS.copy()
T[sk_nodes] = constrain(PTS[sk_nodes], sk_nodes)
okey = {tuple(np.round(X[v], 6)): v for v in DUP}
COLINTERP = os.environ.get("COLINTERP", "1") == "1"
ttop = {}                                          # column (outline node) -> target of its top-level node
if COLINTERP:
    for i in sk_nodes:
        v = okey.get(tuple(np.round(PTS[i, :2], 6)))
        if v is None:
            continue
        r_o = np.hypot(*PTS[i, :2])
        if (PTS[i, 2] - HD.s0(r_o)) / (s1p(r_o) - HD.s0(r_o)) > 1 - 1e-9:
            ttop[v] = T[i].copy()
nside = 0
for i in sk_nodes:
    v = okey.get(tuple(np.round(PTS[i, :2], 6)))
    if v is None:
        continue
    d = DUP[v]
    r_o, r_t = np.hypot(*PTS[i, :2]), np.hypot(*d)
    tv = np.array([np.cos(TH[np.round(np.arctan2(d[1], d[0]) / np.radians(7.5)).astype(int) % NV]),
                   np.sin(TH[np.round(np.arctan2(d[1], d[0]) / np.radians(7.5)).astype(int) % NV])])
    ut = np.clip(d @ tv, R0V + VHW, UW)
    ztop = HD.s1(ut) - VHW
    f = (PTS[i, 2] - HD.s0(r_o)) / (s1p(r_o) - HD.s0(r_o))
    if f > 1 - 1e-9 and os.environ.get("TOPCP", "1") == "1":
        # top-level outline node: on the rounded top, in the SAME cross-section as its column (u of the dup),
        # at the angle the closest point would take -- so the column and its top stay in one plane
        ud = d @ tv
        if os.environ.get("TOPX", "0") == "1" and R0V + VHW < ud < UW:
            nv_ = np.array([-tv[1], tv[0]])
            pn_ = PTS[i, :2] @ nv_
            zc = HD.s1(ud) - VHW
            dz = PTS[i, 2] - zc
            L_ = np.hypot(pn_, max(dz, 0.0))
            if L_ > 1e-9:
                T[i] = np.r_[ud * tv + VHW * (pn_ / L_) * nv_, zc + VHW * max(dz, 0.0) / L_]
        continue
    T[i] = np.r_[d, HD.s0(r_t) + np.clip(f, 0, 1) * (ztop - HD.s0(r_t))]
    if COLINTERP and v in ttop and 0 < f < 1:
        # the column bends from the vane side onto the rounded top gradually: a straight line from its foot to its
        # top target, put back on the vane
        T[i] = onvane((np.r_[d, HD.s0(r_t)] * (1 - f) + ttop[v] * f)[None])[0]
    nside += 1
print("side skin nodes from the 2D skins: %d" % nside, flush=True)
sf = np.where(skin)[0]
_Pf = PTS[FC[sf]]; _n = np.cross(_Pf[:, 2] - _Pf[:, 0], _Pf[:, 3] - _Pf[:, 1]); _n /= np.linalg.norm(_n, axis=1)[:, None]
print("skin faces: %d side, %d top (facing up/down)" % ((np.abs(_n[:, 2]) <= 0.7).sum(), (np.abs(_n[:, 2]) > 0.7).sum()), flush=True)
_off = np.linalg.norm(onvane(T[sk_nodes]) - T[sk_nodes], axis=1)
_rim = on_shell[sk_nodes] | on_wall[sk_nodes]
print("skin target nodes off the rounded vane: max %.2e (rim nodes excluded: %d)" % (_off[~_rim].max(), _rim.sum()), flush=True)
SIDE2D = int(os.environ.get("SIDE2D", "0"))
if SIDE2D:
    tvn, nvn = vframe2(T[:, :2])
    uu = np.einsum("ij,ij->i", T[:, :2], tvn); pp = np.einsum("ij,ij->i", T[:, :2], nvn)
    kk = np.round(np.arctan2(T[:, 1], T[:, 0]) / np.radians(7.5)).astype(int) % NV
    flat = (np.abs(np.abs(pp) - VHW) < 1e-6) & (T[:, 2] <= HD.s1(np.clip(uu, R0V + VHW, UW)) - VHW + 1e-6) & (uu >= R0V + VHW - 1e-6)
    uwall = np.sqrt(R_VES ** 2 - VHW ** 2)
    fside = flat[FC[sf]].all(1)
    for kv in range(NV):
        for sg in (-1, 1):
            fs = sf[fside & (kk[FC[sf]] == kv).all(1) & (np.sign(pp[FC[sf]]) == sg).all(1)]
            if not len(fs):
                continue
            nodes = np.unique(FC[fs])
            loc = {v: i for i, v in enumerate(nodes)}
            Ql = np.array([[loc[v] for v in q] for q in FC[fs]])
            Y = np.c_[uu[nodes], T[nodes, 2]]
            a2 = 0.5 * np.sum(Y[Ql][..., 0] * np.roll(Y[Ql][..., 1], -1, 1) - np.roll(Y[Ql][..., 0], -1, 1) * Y[Ql][..., 1], 1)
            Pi = np.c_[np.einsum("ij,j->i", PTS[nodes, :2], np.array([np.cos(TH[kv]), np.sin(TH[kv])])), PTS[nodes, 2]]
            ai = 0.5 * np.sum(Pi[Ql][..., 0] * np.roll(Pi[Ql][..., 1], -1, 1) - np.roll(Pi[Ql][..., 0], -1, 1) * Pi[Ql][..., 1], 1)
            Ql = np.where((ai < 0)[:, None], Ql[:, ::-1], Ql)          # wind by the inner (fluid) faces
            other = set(np.unique(FC[sf[~fside]]).tolist())
            free = np.array([v not in other for v in nodes])
            sh_ = on_shell[nodes]; wl_ = on_wall[nodes]

            def pj(i, p, sh_=sh_, wl_=wl_):
                if wl_[i]:
                    p = np.array([uwall, p[1]])
                if sh_[i]:
                    p = np.array([p[0], HD.s0(np.hypot(p[0], VHW))])
                return p
            s0_ = corner_sj(Y, Ql).min()
            if s0_ < 0.1:
                Y = untangle(Y, Ql, free, pj, target=0.1, rounds=100, grow=2)
            t_, n_ = np.array([np.cos(TH[kv]), np.sin(TH[kv])]), np.array([-np.sin(TH[kv]), np.cos(TH[kv])])
            T[nodes] = np.c_[Y[:, 0:1] * t_ + sg * VHW * n_, Y[:, 1]]
    print("side targets untangled in (u, z) per vane side (%.0fs)" % (time.time() - t0), flush=True)
SKSM = int(os.environ.get("SKSM", "20"))
if SKSM:
    # smooth the lofted surface a bit: skin targets relax along the vane (tangential Laplacian, put back on the
    # vane); a step that makes any skin hex worse than min(its sj, SKF) is undone at that hex's nodes
    SKF = float(os.environ.get("SKF", "0.15"))
    HXS = np.c_[FC[sf], FC[sf] + len(PTS)]
    sn = np.unique(FC[sf])
    mov = sn[~(on_shell[sn] | on_wall[sn])]
    E_ = np.unique(np.sort(np.concatenate([FC[sf][:, [0, 1]], FC[sf][:, [1, 2]], FC[sf][:, [2, 3]], FC[sf][:, [3, 0]]]), axis=1), axis=0)
    deg = np.bincount(E_.ravel(), minlength=len(PTS)).astype(float)
    from vskin_twist_lib import hex_score as _hs

    def skq(TT):
        return _hs(np.r_[PTS, TT], HXS)                # min(corner sj, face-turn agreement): sees twisted faces too
    sj0 = skq(T)
    for it in range(SKSM):
        acc = np.zeros_like(T)
        np.add.at(acc, E_[:, 0], T[E_[:, 1]]); np.add.at(acc, E_[:, 1], T[E_[:, 0]])
        Tn = T.copy()
        Tn[mov] = onvane(0.5 * T[mov] + 0.5 * acc[mov] / np.maximum(deg[mov], 1)[:, None])
        ok_ = False
        for _ in range(50):
            sj1 = skq(Tn)
            worse = sj1 < np.minimum(sj0, SKF) - 1e-9
            if not worse.any():
                ok_ = True
                break
            back = np.unique(FC[sf][worse])
            Tn[back] = T[back]
        if not ok_:
            Tn = T.copy()
        sj1 = skq(Tn)
        T = Tn; sj0 = np.maximum(sj0, np.minimum(sj1, SKF)) if False else sj1
    print("skin surface smoothing: %d sweeps, skin min sj %.3f, p1 %.3f, below 0.05: %d" % (SKSM, sj0.min(), np.percentile(sj0, 1), (sj0 < 0.05).sum()), flush=True)
v_ = np.asarray(hex_csj(np.r_[PTS, T], np.c_[FC[sf], FC[sf] + len(PTS)]))
Pf = PTS[FC[sf]]
nrm = np.cross(Pf[:, 2] - Pf[:, 0], Pf[:, 3] - Pf[:, 1]); nrm /= np.linalg.norm(nrm, axis=1)[:, None]
top = np.abs(nrm[:, 2]) > 0.7
rim = on_shell[FC[sf]].any(1) | on_wall[FC[sf]].any(1)
print("linear skin: %d hexes, inverted %d (top %d, side %d, of which rim %d), min %.3f, p1 %.3f" % (
    len(sf), (v_ <= 0).sum(), (top & (v_ <= 0)).sum(), (~top & (v_ <= 0)).sum(), (rim & (v_ <= 0)).sum(), v_.min(), np.percentile(v_, 1)), flush=True)
inv = np.where(v_ <= 0)[0]
if not len(inv):
    inv = np.argsort(v_)[:1]                        # nothing inverted: describe the worst one
cen = Pf[inv].mean(1); tv, nv = vframe2(cen[:, :2])
u = np.einsum("ij,ij->i", cen[:, :2], tv); pn = np.einsum("ij,ij->i", cen[:, :2], nv)
zrel = (cen[:, 2] - HD.s0(np.hypot(cen[:, 0], cen[:, 1]))) / (s1p(np.hypot(cen[:, 0], cen[:, 1])) - HD.s0(np.hypot(cen[:, 0], cen[:, 1])))
tgt = T[FC[sf[inv]]]
zone = np.where(u < R0V + VHW + 3, "inner end", np.where(np.hypot(cen[:, 0], cen[:, 1]) > R_VES - 30, "near wall", "side"))
print("inverted skin by zone:", collections.Counter(zone), flush=True)
print("  rim (shell) %d, rim (wall) %d" % (on_shell[FC[sf[inv]]].any(1).sum(), on_wall[FC[sf[inv]]].any(1).sum()), flush=True)
print("  height in L2a (0 shell .. 1 S1'): p10 %.2f p50 %.2f p90 %.2f" % tuple(np.percentile(zrel, [10, 50, 90])), flush=True)
# collapse vs flip: target face area vs inner face area
ai = np.linalg.norm(np.cross(Pf[inv][:, 2] - Pf[inv][:, 0], Pf[inv][:, 3] - Pf[inv][:, 1]), axis=1)
at = np.linalg.norm(np.cross(tgt[:, 2] - tgt[:, 0], tgt[:, 3] - tgt[:, 1]), axis=1)
print("  target/inner face area ratio: p10 %.3f p50 %.3f p90 %.3f" % tuple(np.percentile(at / ai, [10, 50, 90])), flush=True)
dmv = np.linalg.norm(tgt - Pf[inv], axis=2).max(1)
print("  max node move: p50 %.1f p90 %.1f" % tuple(np.percentile(dmv, [50, 90])), flush=True)
pickle.dump(dict(P=Pf[inv], T=tgt, u=u, pn=pn, z=cen[:, 2]), open(os.path.join(HERE, "vloft_inv.pkl"), "wb"))
if os.environ.get("STOP"):
    sys.exit()
newLI = LI.copy()
sl_ = np.unique(FQ[skin])
mid = onvane(0.5 * (T[LINES[sl_, 0]] + T[LINES[sl_, 1]]))
wl2 = on_wall[LINES[sl_]].all(1); mid[wl2, :2] *= (R_VES / np.hypot(mid[wl2, 0], mid[wl2, 1]))[:, None]
sh2 = on_shell[LINES[sl_]].all(1); mid[sh2, 2] = HD.s0(np.hypot(mid[sh2, 0], mid[sh2, 1]))
if LI.shape[1] > 0:
    newLI[sl_, 0] = mid
newFI = FI.copy()
if FI.shape[1] > 0:
    newFI[sf, 0] = onvane(T[FC[sf]].mean(1))
Bo = QuadMesh(LineMesh(PointMesh(T, lm.point_mesh.element_tags), LINES, newLI, lm.element_tags),
              FQ, np.asarray(Bq.orient), newFI, Bq.element_tags)
sk = hexmesh.loft([quadmesh.select(Bq, skin), quadmesh.select(Bo, skin)], element_tags="fluid", first_tag="", last_tag="wall")
mesh = hexmesh.merge([fluid, sk], clear_seam_tags=True)
print(hexmesh.report(mesh), flush=True)
C = np.asarray(mesh.points)[np.asarray(mesh.corners)]
EE = np.array([[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]])
hmin = np.linalg.norm(C[:, EE[:, 0]] - C[:, EE[:, 1]], axis=2).min(1)
print("shortest edge: min %.4f, <0.1 %d, <0.3 %d, <0.5 %d, <1 %d" % (hmin.min(), (hmin < .1).sum(), (hmin < .3).sum(), (hmin < .5).sum(), (hmin < 1).sum()), flush=True)
pickle.dump(mesh, open(os.path.join(HERE, os.environ.get("OUTL2", "head_L2_vloft") + ".pkl"), "wb"))
from nekmeshpy.io.writer import to_vtu
to_vtu(mesh, os.path.join(HERE, os.environ.get("OUTL2", "head_L2_vloft") + ".vtu"))
print("done (%.0fs)" % (time.time() - t0))
