"""Straight vanes, take 2: pillowed L2 section (as vanepos.py), outline chains PROJECTED onto the
exact vane outline (monotone, min gap), displacement carried by a length-weighted harmonic field in
NSTEP increments, toolkit smoother after each step with chain nodes sliding on their lines.
-> vanepos.pkl (same fields as vanepos.py)"""
import os, sys, pickle, collections, time
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import vanes as V
from head import R0V, VHW, R_VES
from q2 import corner_sj, untangle
from nekmeshpy.quadmesh import QuadMesh
from nekmeshpy.quadmesh.smoother import constrain as cn, smooth

GAP = float(os.environ.get("GAP", "0.35"))         # min spacing / uniform spacing on a vane side
NSTEP = int(os.environ.get("NSTEP", "1"))
WPOW = float(os.environ.get("WPOW", "1.0"))        # edge weight 1/len**WPOW
SMOOTH = int(os.environ.get("SMOOTH", "0"))
t00 = time.time()
H = pickle.load(open(os.path.join(HERE, "head_sec.pkl"), "rb"))
VS = pickle.load(open(os.path.join(HERE, os.environ.get("VANES", "vanes2.pkl")), "rb"))
cx = V.Ctx(H["quads"], H["keys"], H["X_kn"])
PW = [] if os.environ.get("NOPILLOW") else VS["pillow"]
taken = VS["taken"]

quads = list(cx.quads); sel = list(taken); parent = {}
for qi in PW:
    q, lab, c = cx.quads[qi]
    inner = tuple(("pv", qi, j) for j in range(4))
    quads[qi] = (inner, lab, c); parent[qi] = qi
    for j in range(4):
        quads.append(((q[j], q[(j + 1) % 4], inner[(j + 1) % 4], inner[j]), lab, c))
        sel.append(taken[qi]); parent[len(quads) - 1] = qi
sel = np.array(sel)
keys = dict(cx.keys)
for q, _, _ in quads:
    for k in q:
        keys.setdefault(k, len(keys))
Q = np.array([[keys[k] for k in q] for q, _, _ in quads])
X = np.zeros((len(keys), 2))
for k, i in keys.items():
    if not (isinstance(k, tuple) and k[0] == "pv"):
        X[i] = cx.X[cx.keys[k]]
for qi in PW:
    P = np.array([cx.X[cx.keys[k]] for k in cx.quads[qi][0]]); cen = P.mean(0)
    for j in range(4):
        X[keys[("pv", qi, j)]] = P[j] + 0.5 * (cen - P[j])
X_kn = X.copy()
n = len(X)
print("pillowed section: %d quads; KN min corner sj %.3f" % (len(Q), corner_sj(X, Q).min()), flush=True)

# ---- chain targets: projection onto the 3 outline segments, monotone with a min gap
cxinv = {i: k for k, i in cx.keys.items()}
tw = np.sqrt(R_VES ** 2 - VHW ** 2)
fixed, seg, srange = {}, {}, {}


def place(ids, a, b):
    """nodes ids (ordered a -> b) onto segment a-b: projection parameter, monotone, min gap."""
    m = len(ids)
    d = b - a; L2 = d @ d
    s = np.clip(np.array([(X[i] - a) @ d / L2 for i in ids]), 0, 1)
    s[0], s[-1] = 0.0, 1.0
    g = GAP / (m - 1)
    fw = s.copy()
    for k in range(1, m):
        fw[k] = max(fw[k], fw[k - 1] + g)
    bw = s.copy()
    for k in range(m - 2, -1, -1):
        bw[k] = min(bw[k], bw[k + 1] - g)
    w = 0.5 * (fw + bw)
    if (np.diff(w) < 0.5 * g).any() or w[0] < -1e-9 or w[-1] > 1 + 1e-9:
        w = np.linspace(0, 1, m)
    w[0], w[-1] = 0.0, 1.0
    for k, (i, f) in enumerate(zip(ids, w)):
        fixed[i] = a + f * d
        seg[i] = (a, b)
        lo = w[k - 1] if k > 0 else 0.0
        hi = w[k + 1] if k < m - 1 else 1.0
        srange[i] = (f - 0.6 * (f - lo), f + 0.6 * (hi - f))


for kv, t, nn, s_, path in VS["vanes"]:
    ids = [keys[cxinv[v]] for v in path]
    P = X[ids]
    cm, cp = R0V * t - VHW * nn, R0V * t + VHW * nn
    wm, wp = tw * t - VHW * nn, tw * t + VHW * nn
    im = int(np.argmin(np.linalg.norm(P - cm, axis=1))); ip = int(np.argmin(np.linalg.norm(P - cp, axis=1)))
    if im > ip:
        ids = ids[::-1]; P = P[::-1]; im, ip = len(ids) - 1 - im, len(ids) - 1 - ip
    if im == ip:
        if np.linalg.norm(P[im + 1] - cp) < np.linalg.norm(P[im - 1] - cm):
            ip = im + 1
        else:
            im = im - 1
    place(ids[im::-1], cm, wm)
    place(ids[im:ip + 1], cm, cp)
    place(ids[ip:], cp, wp)
    for i in (ids[0], ids[im], ids[ip], ids[-1]):
        seg.pop(i, None)                    # corners: frozen
    # vessel-wall nodes inside the strip: even in angle between the chain ends
    th_v = np.arctan2(t[1], t[0])
    rel = lambda p: (np.arctan2(p[1], p[0]) - th_v + np.pi) % (2 * np.pi) - np.pi
    lo, hi = sorted((rel(fixed[ids[0]]), rel(fixed[ids[-1]])))
    r_ = np.hypot(*X.T)
    vn = np.zeros(n, bool); vn[np.unique(Q[sel])] = True
    wall_in = [i for i in np.where(vn & (r_ > R_VES - 1e-6))[0] if i not in fixed and abs(rel(X[i])) < 0.05]
    wall_in.sort(key=lambda i: rel(X[i]))
    for k, i in enumerate(wall_in):
        a_ = th_v + lo + (k + 1) / (len(wall_in) + 1) * (hi - lo)
        fixed[i] = R_VES * np.array([np.cos(a_), np.sin(a_)])
chain = np.array(sorted(fixed))
print("chain targets: %d nodes, max move %.2f, mean %.2f" % (len(chain), max(np.linalg.norm(fixed[i] - X[i]) for i in chain),
      np.mean([np.linalg.norm(fixed[i] - X[i]) for i in chain])), flush=True)

# ---- boundary classes
r = np.hypot(*X.T)
ec = collections.Counter()
for q in Q:
    for a in range(4):
        ec[frozenset((q[a], q[(a + 1) % 4]))] += 1
bnd = {v for e, c in ec.items() if c == 1 for v in e}
onwall = np.array([i in bnd and r[i] > R_VES - 1e-6 for i in range(n)])
E = np.array([tuple(e) for e in ec])
dirichlet = np.zeros(n, bool); dirichlet[chain] = True
dirichlet |= (r < 425)
dirichlet |= np.array([i in bnd and not onwall[i] for i in range(n)])
free = ~dirichlet
le = np.linalg.norm(X[E[:, 0]] - X[E[:, 1]], axis=1)
w = 1.0 / np.maximum(le, 1e-3) ** WPOW
W = sp.coo_matrix((np.r_[w, w], (np.r_[E[:, 0], E[:, 1]], np.r_[E[:, 1], E[:, 0]])), shape=(n, n)).tocsr()
L = sp.diags(np.asarray(W.sum(1)).ravel()) - W
lu = splu(L[free][:, free].tocsc()); Lfd = L[free][:, ~free]
Dfull = np.zeros((n, 2))
for i, p in fixed.items():
    Dfull[i] = p - X[i]
Dh = np.zeros((n, 2)); Dh[chain] = Dfull[chain]
for c in range(2):
    Dh[free, c] = lu.solve(-(Lfd @ Dh[~free, c]))
FQ = ~sel
zone_q = np.where((np.linalg.norm(Dh[Q], axis=2).max(1) > 1e-3) | sel)[0]
nodes = np.unique(Q[zone_q]); loc = -np.ones(n, int); loc[nodes] = np.arange(len(nodes)); SQ = loc[Q[zone_q]]
print("smoothing zone: %d quads" % len(zone_q), flush=True)
segs = collections.defaultdict(list)
for i, (a, b) in seg.items():
    segs[(tuple(np.round(a, 9)), tuple(np.round(b, 9)))].append(i)
in_zone = np.zeros(n, bool); in_zone[nodes] = True
frozen_ids = [loc[i] for i in nodes if (dirichlet[i] and i not in seg) and not (onwall[i] and i not in fixed)]
cons = [cn.Frozen(np.array(frozen_ids))]
for (a, b), ids in segs.items():
    a, b = np.array(a), np.array(b)
    cons.append(cn.OnCurve(np.array([loc[i] for i in ids]), cn.line(np.r_[a, 0.0], np.r_[b - a, 0.0])))
wall_free = [loc[i] for i in nodes if onwall[i] and i not in fixed]
if wall_free:
    cons.append(cn.OnCurve(np.array(wall_free), cn.circle(np.zeros(3), R_VES)))


PPAR = np.array(PW, int)
PCOR = np.array([[keys[k] for k in cx.quads[qi][0]] for qi in PW], int).reshape(-1, 4)
PINN = np.array([[keys[("pv", qi, j)] for j in range(4)] for qi in PW], int).reshape(-1, 4)
PQS = [[qi] + [i for i in range(len(cx.quads), len(quads)) if parent.get(i) == qi] for qi in PW]
_pq = {}
for qi in PW:
    _pq[qi] = [qi]
for i in range(len(cx.quads), len(quads)):
    _pq[parent[i]].append(i)
PQS = [_pq[qi] for qi in PW]


def fix_pillows(Y):
    """inner nodes of every pillow placed in its (possibly degenerate) parent: P_j + a_j (c - P_j),
    corner weights chosen to maximise the worst corner sj of the 5 quads."""
    Y = Y.copy()
    for k, qi in enumerate(PW):
        P = Y[PCOR[k]]; c = P.mean(0)
        e1 = np.roll(P, -1, 0) - P; e0 = np.roll(P, 1, 0) - P
        ang = np.arctan2(e1[:, 0] * e0[:, 1] - e1[:, 1] * e0[:, 0], (e1 * e0).sum(1))
        best = None
        for base in (0.3, 0.4, 0.5, 0.6):
            for hi in (base, 0.1, 0.15, 0.2, 0.25, 0.55, 0.65):
                a = np.where((ang > np.radians(150)) | (ang < 0), hi, base)
                Y[PINN[k]] = P + a[:, None] * (c - P)
                sv = corner_sj(Y, Q[PQS[k]]); v = sv.min() + 0.01 * np.sort(sv)[1]
                if best is None or v > best[0]:
                    best = (v, Y[PINN[k]].copy())
        Y[PINN[k]] = best[1]
    return Y


def report(Y, tag):
    s = corner_sj(Y, Q)
    print("%s (%.0fs): fluid bad %d min %.3f | vane bad %d min %.3f" % (tag, time.time() - t00, (s[FQ] <= 0).sum(), s[FQ].min(),
          (s[sel] <= 0).sum(), s[sel].min()), flush=True)
    return s


Xv = X.copy()
report(fix_pillows(X + Dh), "pure harmonic (1 step) + pillows")
for k in range(1, NSTEP + 1):
    Dk = np.zeros((n, 2)); Dk[chain] = (X[chain] + Dfull[chain] * k / NSTEP) - Xv[chain]
    for c in range(2):
        Dk[free, c] = lu.solve(-(Lfd @ Dk[~free, c]))
    Xv = Xv + Dk
    Xv[onwall] = R_VES * Xv[onwall] / np.linalg.norm(Xv[onwall], axis=1)[:, None]
    Xv = fix_pillows(Xv)
    s_h = report(Xv, "step %d harmonic" % k)
    if SMOOTH:
        qm = QuadMesh.from_corners(np.c_[Xv[nodes], np.zeros(len(nodes))], SQ)
        res = cn.resolve(qm, cons, boundary="frozen")
        sched = (smooth.untangle_schedule(iters=40) if (s_h <= 0).any() else []) + smooth.user_schedule(outer=2)
        out, _ = smooth.run_schedule(qm, res, sched, normals=np.tile([0.0, 0.0, 1.0], (len(SQ), 1)))
        Y = Xv.copy(); Y[nodes] = np.asarray(out.points)[:, :2]
        s_s = report(Y, "step %d smoothed" % k)
        if (s_s <= 0).sum() <= (s_h <= 0).sum():
            Xv = Y
pin_n = set(PINN.ravel().tolist())


def projU(i, p):
    if i in seg:
        a, b = seg[i]; d = b - a
        return a + np.clip((p - a) @ d / (d @ d), *srange[i]) * d
    if onwall[i]:
        return R_VES * p / np.linalg.norm(p)
    return p


frU = free.copy()
for i in seg:
    frU[i] = True
for i in fixed:
    if onwall[i] and i not in seg:
        frU[i] = False
Xv = untangle(Xv, Q, frU, projU, target=float(os.environ.get("UTGT", "0.05")), rounds=200, grow=2)
Xv = fix_pillows(Xv)
s_ = report(Xv, "local untangle")
for tgt, gr in ((0.08, 3), (0.12, 4), (0.15, 6)):          # retry wider / stricter until valid
    if s_.min() > 0.02:
        break
    Xv = untangle(Xv, Q, frU, projU, target=tgt, rounds=400, grow=gr)
    Xv = fix_pillows(Xv)
    s_ = report(Xv, "local untangle (target %.2f, grow %d)" % (tgt, gr))
dev = max(np.linalg.norm(Xv[i] - fixed[i]) for i in chain if i not in seg)
print("frozen chain dev %.2e" % dev)
pickle.dump(dict(quads=quads, keys=keys, Q=Q, X_kn=X_kn, X_v=Xv, vane=sel, pillow=PW, parent=parent,
                 fixed=fixed, seg=seg), open(os.path.join(HERE, os.environ.get("OUTP", "vanepos.pkl")), "wb"))
