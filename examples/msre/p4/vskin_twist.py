"""Twist-aware greedy untangle of the linear vskin lower head.  Score of a hex = min(corner scaled
Jacobian, face-turn agreement) -- the second is what hexmesh.validity calls a twisted face, which the
corner Jacobian does not see.  Free nodes near the bad hexes move to the best of a set of candidates;
nodes on the vane skin stay on the rounded vane, nodes on the head shell stay on the shell, every other
boundary node is fixed.  -> head_L2_skin_lin3.pkl / .vtu"""
import os, sys, pickle, time, collections
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("STOP2D", "1")
from nekmeshpy import HexMesh, hexmesh
from nekmeshpy.hexmesh import smoother as hs
import head as HD

t0 = time.time()
m = pickle.load(open(os.environ.get("IN", "head_L2_skin_lin2.pkl"), "rb"))
if m.order > 1:
    m = hs.linearize(m)
P = np.asarray(m.points).copy(); H = np.asarray(m.corners)
FP = HexMesh.FACE_POINTS
# rounded vane (same model as vskin2)
R0V, VHW, NV = HD.R0V, HD.VHW, HD.NV
TH = np.radians(7.5 * np.arange(NV))
_ug = np.linspace(R0V + 50, HD.R_VES, 4000); _h = HD.s1(_ug) - HD.s0(_ug)
UEND = min(float(_ug[np.argmax(_h < 2 * VHW)]), HD.R_MID - 1.5 - 1.0); UW = UEND - VHW


def onvane(p):
    k = np.round(np.arctan2(p[:, 1], p[:, 0]) / np.radians(7.5)).astype(int) % NV
    t = np.c_[np.cos(TH[k]), np.sin(TH[k])]
    u = np.einsum("ij,ij->i", p[:, :2], t); uc = np.clip(u, R0V + VHW, UW)
    mm = np.c_[uc[:, None] * t, np.minimum(p[:, 2], HD.s1(uc) - VHW)]
    d = p - mm; L = np.linalg.norm(d, axis=1); L = np.where(L < 1e-12, 1, L)
    return mm + VHW * d / L[:, None]


def hex_score(Pt, hx):
    from nekmeshpy.hexmesh.quality import corner_scaled_jacobian
    sj = np.asarray(corner_scaled_jacobian(Pt, hx))
    q = Pt[hx[:, FP]]                                         # (n,6,4,3)
    turn = np.cross(np.roll(q, -1, axis=2) - q, np.roll(q, 1, axis=2) - q)
    mean = turn.sum(axis=2, keepdims=True)
    ag = np.einsum("efkc,efkc->efk", turn, np.broadcast_to(mean, turn.shape))
    ag = ag / (np.linalg.norm(turn, axis=3) * np.linalg.norm(mean, axis=3) + 1e-300)
    return np.minimum(sj, ag.min(axis=(1, 2)))


v = hexmesh.validity(m)
s = hex_score(P, H)
print("start: inverted %d, twisted %d, min score %.3f" % (v.inverted.size, v.twisted.size, s.min()), flush=True)
# boundary node classes
fk = np.sort(H[:, FP].reshape(-1, 4), axis=1)
_, fi, fc = np.unique(fk, axis=0, return_inverse=True, return_counts=True)
bfaces = H[:, FP].reshape(-1, 4)[fc[fi] == 1]
bnd = np.zeros(len(P), bool); bnd[bfaces.ravel()] = True
r_ = np.hypot(P[:, 0], P[:, 1])
shell = bnd & (np.abs(P[:, 2] - HD.s0(r_)) < 1e-6)
vane = bnd & ~shell & (np.linalg.norm(onvane(P) - P, axis=1) < float(os.environ.get("VTOL", "0.05")))
free = ~bnd | vane | shell
print("boundary nodes: %d (on vane %d, on shell %d)" % (bnd.sum(), vane.sum(), shell.sum()), flush=True)


def proj(i, p):
    if vane[i]:
        p = onvane(p[None])[0]
    if shell[i]:
        p = p.copy(); p[2] = HD.s0(np.hypot(p[0], p[1]))
    return p


inc = collections.defaultdict(list); nbr = collections.defaultdict(set)
for e, q in enumerate(H):
    for a in range(8):
        inc[q[a]].append(e)
E12 = np.array([[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]])
for q in H:
    for a, b in E12:
        nbr[q[a]].add(q[b]); nbr[q[b]].add(q[a])
rng = np.random.default_rng(0)
TGT = float(os.environ.get("TGT", "0.05"))
for rd in range(int(os.environ.get("ROUNDS", "200"))):
    s = hex_score(P, H) if rd == 0 else s
    bad = np.where(s < TGT)[0]
    if not len(bad):
        break
    zone = set(np.unique(H[bad]).tolist())
    for _ in range(int(os.environ.get("GROW", "2"))):
        zone |= {w for x in list(zone) for w in nbr[x]}
    for x in sorted(zone, key=lambda x: s[inc[x]].min()):
        if not free[x]:
            continue
        hs_ = np.array(inc[x]); cur = hex_score(P, H[hs_]).min()
        nb_ = list(nbr[x]); c0 = P[nb_].mean(0)
        hl = 0.5 * np.mean(np.linalg.norm(P[nb_] - P[x], axis=1))
        best, bp = cur, None; old = P[x].copy()
        FS = (0.1, 0.3, 0.6, 1.0, 1.5) if os.environ.get("BIG") else (0.1, 0.3, 0.6)
        NC = 14 if os.environ.get("BIG") else 8
        for p in [c0, 0.5 * (c0 + old)] + [old + hl * f * rng.normal(size=3) for f in FS for _ in range(NC)]:
            P[x] = proj(x, p)
            val = hex_score(P, H[hs_]).min()
            if val > best + 1e-9:
                best, bp = val, P[x].copy()
        P[x] = bp if bp is not None else old
        s[hs_] = hex_score(P, H[hs_])
    if rd % 10 == 0:
        print("  round %d: %d below %.2f, min %.3f (%.0fs)" % (rd, len(bad), TGT, s.min(), time.time() - t0), flush=True)
out = hs.with_points(m, P)
v = hexmesh.validity(out)
rz = np.hypot(P[:, 0], P[:, 1])
top_ = np.abs(P[:, 2] - HD.s2(rz)) < 1e-6
for i in np.unique(np.r_[v.inverted, v.twisted]):
    c = P[H[i]].mean(0)
    print("  left: hex %d r %.1f z %.1f th %.2f score %.3f | free %d/8, on S2 %d, vane %d, shell %d, other-bnd %d" % (
        i, np.hypot(*c[:2]), c[2], np.degrees(np.arctan2(c[1], c[0])), hex_score(P, H[i:i + 1])[0], free[H[i]].sum(),
        top_[H[i]].sum(), vane[H[i]].sum(), shell[H[i]].sum(), (bnd[H[i]] & ~vane[H[i]] & ~shell[H[i]]).sum()), flush=True)
print("end: inverted %d, twisted %d, min score %.3f (%.0fs)" % (v.inverted.size, v.twisted.size, hex_score(P, H).min(), time.time() - t0), flush=True)
pickle.dump(out, open(os.environ.get("OUT", "head_L2_skin_lin3") + ".pkl", "wb"))
from nekmeshpy.io.writer import to_vtu
to_vtu(out, os.environ.get("OUT", "head_L2_skin_lin3") + ".vtu")
print("done")
