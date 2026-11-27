"""Order 2 for the finished (linear, diced) MSRE mesh, high-order nodes on the geometry.

1. every edge mid-node straight; boundary edges / faces whose corners match an order-2 part (core, bands,
   downcomer, volute+inlet, top head, pocket) take that part's exact node; lower-head vane / shell / vessel
   nodes are projected analytically.
2. the channel boundary layer follows its wall: an edge (face) of the split surface takes its wall edge's
   (face's) curvature displacement, scaled by the split weight w.
3. face centres and hex centres: Coons (transfinite) from their edges (faces) -- by default (NEK_FACES=1) for
   every face, since .re2 carries edge midpoints only and Nek builds the rest the same way; then every cell
   still folded at SAMPLE_ORDER (7) is straightened, with every face on a straightened edge, until none is.
usage: order2.py diced.pkl diced_split.npz OUT    env PARTS_DIR (the examples/msre root of an order-2 run)"""
import os, sys, time, pickle, glob
import numpy as np
from nekmeshpy import HexMesh, hexmesh
from nekmeshpy.linemesh import LineMesh
from nekmeshpy.pointmesh import PointMesh
from nekmeshpy.quadmesh import QuadMesh

t0 = time.time()
R = os.environ.get("PARTS_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(R, "p4"))
os.environ.setdefault("STOP2D", "1")
TOL = 1e-6                                                  # mm; endpoint match tolerance


def match(refA, refB, refV, A, B):
    """segments (A,B) matched to reference segments (refA,refB) in either orientation, by midpoint KD
    search then an endpoint check.  -> hit mask, value"""
    from scipy.spatial import cKDTree
    tr = cKDTree(0.5 * (refA + refB))
    d, k = tr.query(0.5 * (A + B), distance_upper_bound=TOL * 10)
    ok = np.isfinite(d)
    k = np.where(ok, k, 0)
    e1 = np.maximum(np.linalg.norm(refA[k] - A, axis=1), np.linalg.norm(refB[k] - B, axis=1))
    e2 = np.maximum(np.linalg.norm(refA[k] - B, axis=1), np.linalg.norm(refB[k] - A, axis=1))
    ok &= np.minimum(e1, e2) < TOL
    return ok, refV[k]


def match_quad(refC, refV, C):
    from scipy.spatial import cKDTree
    tr = cKDTree(refC.mean(1))
    d, k = tr.query(C.mean(1), distance_upper_bound=TOL * 10)
    ok = np.isfinite(d)
    k = np.where(ok, k, 0)
    # every corner of C within TOL of some corner of the reference quad
    dd = np.linalg.norm(C[:, :, None, :] - refC[k][:, None, :, :], axis=3).min(2).max(1)
    ok &= dd < TOL
    return ok, refV[k]


def code2(a, b, M):
    return np.minimum(a, b) * M + np.maximum(a, b)


def code4(Q):
    return np.ascontiguousarray(np.sort(Q, 1).astype(np.int64)).view(np.dtype((np.void, 32))).ravel()


def find(ref, q):
    """index of each q in ref (-1 if absent)"""
    o = np.argsort(ref); r = ref[o]
    p = np.minimum(np.searchsorted(r, q), len(r) - 1)
    return np.where(r[p] == q, o[p], -1)


# ---- the finished, diced mesh (linear) ----
m = pickle.load(open(sys.argv[1], "rb"))
sp = np.load(sys.argv[2])
X = np.asarray(m.points, float)
qm, lm = m.quad_mesh, m.quad_mesh.line_mesh
L = np.asarray(lm.lines); Qc = np.asarray(qm.corners); Ql = np.asarray(qm.quads); Hc = np.asarray(m.corners)
emid = 0.5 * (X[L[:, 0]] + X[L[:, 1]])
bq = np.asarray(hexmesh.boundary_face_ids(m))
bl = np.zeros(len(L), bool); bl[np.unique(Ql[bq])] = True
print("diced mesh: %d hexes, %d lines (%d on the boundary), %d faces (%d boundary) (%.0fs)" % (
    len(Hc), len(L), bl.sum(), len(Qc), bq.sum(), time.time() - t0), flush=True)

# ---- reference: boundary entities of the order-2 parts ----
parts = [os.path.join(R, "p1", "upper_o2_nz32.pkl"), os.path.join(R, "p5", "head_top.pkl"), os.path.join(R, "p5", "pocket.pkl"),
         os.path.join(R, "p4", "volute.pkl"), os.path.join(R, "p4", "dc3.pkl")] + sorted(glob.glob(os.path.join(R, "p4", "band_*.pkl")))
reA, reB, rev, rfC, rfv = [], [], [], [], []
for f in parts:
    pm = pickle.load(open(f, "rb"))
    pm = pm["mesh"] if isinstance(pm, dict) else pm
    if pm.order != 2:
        print("  skip %s (order %d)" % (os.path.basename(f), pm.order)); continue
    P = np.asarray(pm.points); pl = pm.quad_mesh.line_mesh
    pb = np.asarray(hexmesh.boundary_face_ids(pm))
    pq = pm.quad_mesh
    lines_b = np.unique(np.asarray(pq.quads)[pb])
    LL = np.asarray(pl.lines)[lines_b]
    reA.append(P[LL[:, 0]]); reB.append(P[LL[:, 1]]); rev.append(np.asarray(pl.interior)[lines_b, 0])
    QC = np.asarray(pq.corners)[pb]
    rfC.append(P[QC]); rfv.append(np.asarray(pq.interior)[pb, 0])
    print("  %-16s order 2: %d boundary lines, %d boundary faces" % (os.path.basename(f), len(LL), pb.sum()), flush=True)
reA, reB, rev = np.concatenate(reA), np.concatenate(reB), np.concatenate(rev)
rfC, rfv = np.concatenate(rfC), np.concatenate(rfv)
bi = np.where(bl)[0]
hit, val = match(reA, reB, rev, X[L[bi, 0]], X[L[bi, 1]])
emid[bi[hit]] = val[hit]
print("boundary lines: %d take a part's exact mid-node, %d do not (%.0fs)" % (hit.sum(), (~hit).sum(), time.time() - t0), flush=True)

# ---- lower head: analytic surfaces ----
import vskin_twist_lib as VT, head as HD


def on_vane(P, tol=1e-6):
    return np.linalg.norm(VT.onvane(P) - P, axis=1) < tol


def on_shell(P, tol=1e-6):
    return np.abs(P[:, 2] - HD.s0(np.hypot(P[:, 0], P[:, 1]))) < tol


def shell_proj(P):
    """closest point on z = s0(r) in the meridional plane (Newton on the squared distance)."""
    r0 = np.hypot(P[:, 0], P[:, 1]); z0 = P[:, 2]; r = r0.copy(); h = 1e-4
    for _ in range(20):
        f = HD.s0(r); fp = (HD.s0(r + h) - HD.s0(r - h)) / (2 * h); fpp = (HD.s0(r + h) - 2 * f + HD.s0(r - h)) / h ** 2
        g = (r - r0) + (f - z0) * fp; gp = 1 + fp ** 2 + (f - z0) * fpp
        r = r - g / np.where(np.abs(gp) > 1e-12, gp, 1e-12)
    Q = P.copy(); sc = np.where(r0 > 1e-12, r / np.maximum(r0, 1e-12), 1.0)
    Q[:, 0] *= sc; Q[:, 1] *= sc; Q[:, 2] = HD.s0(r)
    return Q


def guarded(old, new, L):
    """take new only where it moves the node less than a quarter of the entity size"""
    ok = np.linalg.norm(new - old, axis=1) < 0.25 * L
    return np.where(ok[:, None], new, old), (~ok).sum()


rest = bi[~hit]
A, B = X[L[rest, 0]], X[L[rest, 1]]
v = on_vane(A) & on_vane(B)
Le = np.linalg.norm(A - B, axis=1)
emid[rest[v]], rj_v = guarded(emid[rest[v]], VT.onvane(emid[rest[v]]), Le[v])
s = on_shell(A) & on_shell(B) & ~v
emid[rest[s]], rj_s = guarded(emid[rest[s]], shell_proj(emid[rest[s]]), Le[s])
rv = np.abs(np.hypot(A[:, 0], A[:, 1]) - HD.R_VES) < 1e-6
rv &= np.abs(np.hypot(B[:, 0], B[:, 1]) - HD.R_VES) < 1e-6
c = emid[rest[rv]].copy(); c[:, :2] *= (HD.R_VES / np.hypot(c[:, 0], c[:, 1]))[:, None]
emid[rest[rv]], rj_r = guarded(emid[rest[rv]], c, Le[rv])
print("  projections refused (move > edge/4): vane %d, shell %d, vessel %d" % (rj_v, rj_s, rj_r))
print("lower head analytic: %d on vanes, %d on the shell, %d on the vessel (%.0fs)" % (v.sum(), s.sum(), rv.sum(), time.time() - t0), flush=True)

# ---- the boundary layer follows its wall ----
wall, far, new, w = sp["wall"], sp["far"], sp["new"], sp["w"]
split_of = -np.ones(len(X), np.int64); split_of[wall] = new          # wall node -> its split node (unique per wall node? see below)
wt = np.zeros(len(X)); wt[wall] = w
M = np.int64(len(X) + 1)
dlt = emid - 0.5 * (X[L[:, 0]] + X[L[:, 1]])
cand = np.where(bl & (np.linalg.norm(dlt, axis=1) > 1e-9))[0]
sa, sb = split_of[L[cand, 0]], split_of[L[cand, 1]]
ok = (sa >= 0) & (sb >= 0)
cand, sa, sb = cand[ok], sa[ok], sb[ok]
jj = find(code2(L[:, 0], L[:, 1], M), code2(sa, sb, M))
ok = jj >= 0
cand, jj = cand[ok], jj[ok]
emid[jj] = 0.5 * (X[L[jj, 0]] + X[L[jj, 1]]) + (0.5 * (wt[L[cand, 0]] + wt[L[cand, 1]]))[:, None] * dlt[cand]
moved = len(jj)
print("boundary layer: %d split-surface edges follow their wall edge (%.0fs)" % (moved, time.time() - t0), flush=True)

# ---- face centres: exact on matched boundary faces, else Coons; the BL split faces follow their wall face ----
el = np.asarray(Ql)                                                 # (Q,4) line ids of each face
fc = 0.5 * emid[el].sum(1) - 0.25 * X[Qc].sum(1)
hitf, valf = match_quad(rfC, rfv, X[Qc[bq]])
bqi = np.where(bq)[0]
fc[bqi[hitf]] = valf[hitf]
nb = bqi[~hitf]                                                      # unmatched boundary faces: lower-head surfaces
FCc = X[Qc[nb]]
fv = on_vane(FCc.reshape(-1, 3)).reshape(-1, 4).all(1)
Lf = np.linalg.norm(FCc[:, 0] - FCc[:, 2], axis=1)
fc[nb[fv]], _ = guarded(fc[nb[fv]], VT.onvane(fc[nb[fv]]), Lf[fv])
fs = on_shell(FCc.reshape(-1, 3)).reshape(-1, 4).all(1) & ~fv
fc[nb[fs]], _ = guarded(fc[nb[fs]], shell_proj(fc[nb[fs]]), Lf[fs])
fdl = fc - (0.5 * emid[el].sum(1) - 0.25 * X[Qc].sum(1))            # exact minus Coons
ci = bqi[hitf]
ci = ci[np.linalg.norm(fdl[ci], axis=1) > 1e-9]
sq = split_of[Qc[ci]]
ci, sq = ci[(sq >= 0).all(1)], sq[(sq >= 0).all(1)]
jj = find(code4(Qc), code4(sq))
ci, jj = ci[jj >= 0], jj[jj >= 0]
fc[jj] = (0.5 * emid[el[jj]].sum(1) - 0.25 * X[Qc[jj]].sum(1)) + wt[Qc[ci]].mean(1)[:, None] * fdl[ci]
fm = len(jj)
print("faces: %d boundary faces take a part's exact centre, %d split faces follow theirs (%.0fs)" % (hitf.sum(), fm, time.time() - t0), flush=True)

if int(os.environ.get("NEK_FACES", 1)):
    # .re2 carries edge midpoints only; Nek builds every face (and the centre) from its edges by Gordon-Hall,
    # so the mesh checked here is that one, not the exact face centres
    fc = 0.5 * emid[el].sum(1) - 0.25 * X[Qc].sum(1)
    print("faces: every centre by Coons from its edges (Nek's own geometry)", flush=True)

# ---- assemble the order-2 mesh, hex centres by Coons from its own block; straighten what folds at N ----
from nekmeshpy.core import interp
lat = np.array([(i, j, k) for k in range(3) for j in range(3) for i in range(3)])
n1 = (lat == 1).sum(1)
corner_n, edge_n, face_n = np.where(n1 == 0)[0], np.where(n1 == 1)[0], np.where(n1 == 2)[0]
NS = int(os.environ.get("SAMPLE_ORDER", 7))
HQ = np.asarray(m.hexes)                                            # (E,6) face ids


def build(emid, fc):
    new_lm = LineMesh(PointMesh(X, lm.point_mesh.element_tags), L, emid[:, None, :], lm.element_tags)
    new_qm = QuadMesh(new_lm, qm.quads, qm.orient, fc[:, None, :], qm.element_tags)
    m2 = HexMesh(new_qm, m.hexes, m.orient, X[Hc].mean(1)[:, None, :], m.element_tags)
    blk = hexmesh.element_blocks(m2)
    centre = 0.5 * blk[:, face_n].sum(1) - 0.25 * blk[:, edge_n].sum(1) + 0.125 * blk[:, corner_n].sum(1)
    blk[:, 13] = centre
    return HexMesh(new_qm, m.hexes, m.orient, centre[:, None, :], m.element_tags), blk


straight_e = 0.5 * (X[L[:, 0]] + X[L[:, 1]])
for it in range(12):
    m2, blk = build(emid, fc)
    sj = np.asarray(interp.sampled_scaled_jacobian(blk, 2, NS, 3))
    bad = np.where(sj <= 0)[0]
    print("pass %d: %d folded at N=%d (min %.4f) (%.0fs)" % (it, len(bad), NS, sj.min(), time.time() - t0), flush=True)
    if not len(bad):
        break
    fids = np.unique(HQ[bad]); lids = np.unique(el[fids])
    emid[lids] = straight_e[lids]
    hitl = np.zeros(len(L), bool); hitl[lids] = True
    fids = np.where(hitl[el].any(1))[0]                             # every face on a straightened edge
    fc[fids] = 0.5 * emid[el[fids]].sum(1) - 0.25 * X[Qc[fids]].sum(1)
    print("  straightened %d edges, %d faces" % (len(lids), len(fids)), flush=True)
print("order-2 mesh built (%.0fs)" % (time.time() - t0), flush=True)
pickle.dump(m2, open(sys.argv[3] + ".pkl", "wb"))
print(hexmesh.report(m2), flush=True)
print("done (%.0fs)" % (time.time() - t0))
