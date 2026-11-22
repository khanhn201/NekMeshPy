"""Plots of curved-only folded hexes: where they are, and what each fold looks like
(curved edges vs straight corner edges, the negative-Jacobian region)."""
import os, sys, pickle, collections
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa
from matplotlib.collections import PolyCollection
from nekmeshpy import hexmesh
from nekmeshpy.hexmesh.query import element_blocks
from nekmeshpy.core.fields import gll_nodes
from nekmeshpy.core.interp import resample_block

MESH = sys.argv[1] if len(sys.argv) > 1 else "head_L2_o2.pkl"
OUT = os.environ.get("FOLDOUT", os.path.join("img", "folds"))
m = pickle.load(open(MESH, "rb"))
order = m.order
sj = np.asarray(hexmesh.scaled_jacobian(m)); sj = sj.min(1) if sj.ndim > 1 else sj
bad = np.where(sj <= 0)[0]
print("order %d: %d curved-folded of %d" % (order, len(bad), len(sj)))
P = np.asarray(m.points); C = np.asarray(m.corners)
cen = P[C].mean(1)
VP = pickle.load(open("vanepos.pkl", "rb"))
nq = len(VP["Q"]); nv = int(VP["vane"].sum())
edges = np.cumsum([0, (nq - nv) * 2, nq * 2, nq, 10**9])
blk = (lambda e: ["L2a", "L2b", "c1 (cap, lower)", "c2 (pillow cap)"][int(np.searchsorted(edges, e, side="right")) - 1]) if "L2" in MESH else (lambda e: "top z<2400" if cen[e, 2] < 2400 else "top z>2400")

# ---- overview: where the folds are
Xv, Q, sel = VP["X_v"], VP["Q"], VP["vane"]
fig, axs = plt.subplots(1, 2, figsize=(20, 9))
ax = axs[0]
ax.add_collection(PolyCollection(Xv[Q], facecolors=np.where(sel[:, None], [[0.9, 0.5, 0.4, 1]], [[1, 1, 1, 0]]), edgecolors="0.6", linewidths=0.1))
for b_, col in zip(sorted({blk(e) for e in bad}), "bmgc"):
    ids = [e for e in bad if blk(e) == b_]
    if ids:
        ax.plot(cen[ids, 0], cen[ids, 1], "o", color=col, ms=7, mfc="none", mew=1.5, label="%s: %d" % (b_, len(ids)))
ax.set_aspect("equal"); ax.set_xlim(-760, 760); ax.set_ylim(-760, 760); ax.legend(loc="upper left")
ax.set_title("folded hexes over the vane section (top view)")
ax = axs[1]
r = np.hypot(cen[:, 0], cen[:, 1])
import head as HD
rr = np.linspace(0, 738.5, 400)
if "L2" in MESH:
    for f, lab in ((HD.s0, "S0 head shell"), (HD.s1, "S1 vane top"), (HD.s2, "S2 grid bottom")):
        ax.plot(rr, f(rr), label=lab)
for b_, col in zip(sorted({blk(e) for e in bad}), "bmgc"):
    ids = [e for e in bad if blk(e) == b_]
    if ids:
        ax.plot(r[ids], cen[ids, 2], "o", color=col, mfc="none", label=b_)
ax.set_xlabel("r"); ax.set_ylabel("z"); ax.legend(); ax.set_title("folded hexes in (r, z)")
plt.savefig(os.path.join(OUT, "overview.png"), dpi=90, bbox_inches="tight"); plt.close()

# ---- per element detail
NF = 12
g = gll_nodes(NF); n = NF + 1
D = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        if i != j:
            D[i, j] = np.prod([g[i] - g[k] for k in range(n) if k != i]) / np.prod([g[j] - g[k] for k in range(n) if k != j]) / (g[i] - g[j])
    D[i, i] = sum(1 / (g[i] - g[k]) for k in range(n) if k != i)
B = element_blocks(m)


def jac(Xf):
    X = Xf.reshape(n, n, n, 3)
    a = np.einsum("ai,ijkc->ajkc", D, X); b = np.einsum("aj,ijkc->iakc", D, X); c = np.einsum("ak,ijkc->ijac", D, X)
    return np.einsum("...c,...c->...", a, np.cross(b, c))


def edges_of(X):
    """the 12 lattice edges of an (n,n,n,3) block as polylines."""
    out = []
    for ax_ in range(3):
        for s1 in (0, -1):
            for s2 in (0, -1):
                idx = [slice(None)] * 3
                others = [a for a in range(3) if a != ax_]
                idx[others[0]] = s1; idx[others[1]] = s2
                out.append(X[tuple(idx)])
    return out


picks = []
for b_ in sorted({blk(e) for e in bad}):
    ids = [e for e in bad if blk(e) == b_]
    ids.sort(key=lambda e: sj[e])
    picks += ids[:2] + ids[len(ids) // 2:len(ids) // 2 + 1]
for e in picks:
    Xf = resample_block(B[e][None], order, NF, 3)[0]
    J = jac(Xf)
    sgn = np.sign(np.median(J)); J = J * sgn
    Xg = Xf.reshape(n, n, n, 3)
    neg = J <= 0
    corners = Xg[[0, -1]][:, [0, -1]][:, :, [0, -1]].reshape(8, 3)
    c0 = Xg.reshape(-1, 3).mean(0)
    # local frame: z up, x along the hex's longest horizontal extent
    Y = Xg - c0
    fig = plt.figure(figsize=(21, 7))
    ax = fig.add_subplot(1, 3, 1, projection="3d")
    for pl in edges_of(Y):
        ax.plot(*pl.T, "k-", lw=1.5)
    Yc = corners - c0
    lin = [(0, 1), (2, 3), (4, 5), (6, 7), (0, 2), (1, 3), (4, 6), (5, 7), (0, 4), (1, 5), (2, 6), (3, 7)]
    for a_, b2 in lin:
        ax.plot(*np.c_[Yc[a_], Yc[b2]], "--", color="0.5", lw=0.8)
    ax.scatter(*Y[neg].T, c="r", s=6, label="J<=0 (%d of %d samples)" % (neg.sum(), neg.size))
    ax.set_title("elem %d [%s]\nmin scaled J %.3f at order %d; curved edges (black) vs corner hex (grey)" % (e, blk(e), sj[e], order))
    ax.legend(loc="upper left"); ax.set_box_aspect(np.ptp(Y.reshape(-1, 3), 0))
    # two side cuts through the worst point
    w = np.unravel_index(np.argmin(J), J.shape)
    for k, axn in enumerate((0, 1)):
        ax2 = fig.add_subplot(1, 3, 2 + k)
        sl = [slice(None)] * 3; sl[axn] = w[axn]
        S = Y[tuple(sl)]; JS = J[tuple(sl)]
        # project the slice onto its best-fit plane
        u_, s_, vt = np.linalg.svd(S.reshape(-1, 3) - S.reshape(-1, 3).mean(0), full_matrices=False)
        e1, e2 = vt[0], vt[1]
        if abs(e2[2]) < abs(e1[2]):
            e1, e2 = e2, e1
        if e2[2] < 0:
            e2 = -e2
        U = np.stack([S @ e1, S @ e2], -1)
        vmax = np.abs(JS).max()
        pc = ax2.pcolormesh(U[..., 0], U[..., 1], JS, cmap="RdBu", vmin=-vmax, vmax=vmax, shading="gouraud")
        for row in range(0, n, 1):
            ax2.plot(U[row, :, 0], U[row, :, 1], color="0.3", lw=0.3); ax2.plot(U[:, row, 0], U[:, row, 1], color="0.3", lw=0.3)
        ax2.contour(U[..., 0], U[..., 1], JS, levels=[0], colors="r", linewidths=2)
        for bd in (U[0], U[-1], U[:, 0], U[:, -1]):
            ax2.plot(bd[:, 0], bd[:, 1], "k-", lw=2)
        ax2.set_aspect("equal"); plt.colorbar(pc, ax=ax2, shrink=0.7, label="Jacobian det")
        ax2.set_title("slice through the worst sample (lattice axis %d), red = J=0" % axn)
    plt.savefig(os.path.join(OUT, "elem_%d.png" % e), dpi=85, bbox_inches="tight"); plt.close()
    print("elem %d %s min sj %.3f, neg samples %d, extent %s" % (e, blk(e), sj[e], neg.sum(), np.ptp(Y.reshape(-1, 3), 0).round(2)))
