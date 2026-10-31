"""Hex smoothing of a designed thin layer: unprotected against ``KeepLayer``.

The box is x, y in [0, 1] (6 x 6 cells) and z in [0, 1] in 8 graded layers: the first is
0.02 thick (the "wall" layer) and each next one is a constant factor thicker, the factors
chosen so the stack ends exactly at z = 1.  The perturbed input moves only interior vertices
of z-planes k >= 2 (by up to 0.25 of the local cell; z by the local layer thickness), so the
designed first layer is clean going in.  Both runs use the same schedule,
``hs.suggest_schedule(q0, outer=3)``, with the boundary frozen:

    (a) unprotected: hs.resolve(mesh, [])
    (b) protected:   hs.resolve(mesh, [hs.KeepLayer(wall_ids, 1)])   -- the z = 0 vertices and
        one ring of hexes (rings=1) frozen, so the top of the first layer cannot move

The shape metric seeks cubes, so unprotected it thickens the 0.02 layer toward the in-plane
spacing; protecting the layer holds it, and the size jump moves to the layer's edge
(``size_ratio_max``, from ``hs.shape_stats``, is the larger/smaller size of face neighbours).

    PYTHONPATH=. python examples/hexsmoother/04_thin_layer.py

Writes ``04_thin_layer_{input,unprotected,protected}.vtu`` to the current directory and
leaves ``mesh`` (the protected result), ``rows``, ``meshes``, ``sj`` and ``first_layer``
(per-hex flag) in the namespace.
"""

import numpy as np

from nekmeshpy import writer
from nekmeshpy.hexmesh import HexMesh
from nekmeshpy.hexmesh import smoother as hs

# -- parameters --------------------------------------------------------------
NX = NY = 6
NZ = 8
FIRST = 0.02                 # first-layer thickness
PERTURB = 0.25               # fraction of the local cell
SEED = 0
OUTER = 3

# -- graded box: growth factor g solves FIRST * (g^NZ - 1) / (g - 1) = 1 ------------
lo, hi = 1.0 + 1e-9, 4.0
for _ in range(200):
    g = 0.5 * (lo + hi)
    if FIRST * (g**NZ - 1) / (g - 1) > 1.0:
        hi = g
    else:
        lo = g
GROWTH = g
zs = np.concatenate([[0.0], np.cumsum(FIRST * GROWTH ** np.arange(NZ))])
zs[-1] = 1.0
thick = np.diff(zs)
xs = np.linspace(0.0, 1.0, NX + 1)
ys = np.linspace(0.0, 1.0, NY + 1)
X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
exact = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
idx = np.arange(exact.shape[0]).reshape(NX + 1, NY + 1, NZ + 1)
c = [idx[:-1, :-1, :-1], idx[1:, :-1, :-1], idx[1:, 1:, :-1], idx[:-1, 1:, :-1],
     idx[:-1, :-1, 1:], idx[1:, :-1, 1:], idx[1:, 1:, 1:], idx[:-1, 1:, 1:]]
hexes = np.stack(c, axis=-1).reshape(-1, 8)
print("graded box: %d hexes, %d layers, thickness %.4f -> %.4f (growth %.4f)"
      % (len(hexes), NZ, thick[0], thick[-1], GROWTH))

I, J, K = np.meshgrid(np.arange(NX + 1), np.arange(NY + 1), np.arange(NZ + 1), indexing="ij")
I, J, K = I.ravel(), J.ravel(), K.ravel()
pert = (I > 0) & (I < NX) & (J > 0) & (J < NY) & (K >= 2) & (K < NZ)      # interior, k >= 2
zloc = np.minimum(thick[np.clip(K - 1, 0, NZ - 1)], thick[np.clip(K, 0, NZ - 1)])
scale = np.stack([np.full(K.shape, 1.0 / NX), np.full(K.shape, 1.0 / NY), zloc], axis=1)
rng = np.random.default_rng(SEED)
P = exact.copy()
P[pert] += (rng.uniform(-PERTURB, PERTURB, (int(pert.sum()), 3)) * scale[pert])
before = HexMesh.from_corners(P, hexes)
wall_ids = np.flatnonzero(K == 0)
first_layer = (K[hexes[:, 0]] == 0)                                         # per-hex flag
print("perturbed input: %d vertices moved; first layer hexes %d" % (int(pert.sum()), int(first_layer.sum())))


def per_hex_sj(m):
    return hs.corner_metrics(m.points, m.corners).sj.min(axis=1)


def height(m):
    """Mean height of every hex: the mean length of its four vertical edges."""
    p, h = m.points, m.corners
    return np.mean([np.linalg.norm(p[h[:, k + 4]] - p[h[:, k]], axis=1) for k in range(4)], axis=0)


h_in = height(before)
q0 = hs.quality(before.points, before.corners)
s0 = hs.shape_stats(before.points, before.corners)
print("input: first-layer thickness %.4f (mean) | inverted %d | sj min %.4f mean %.4f | size_ratio_max %.2f"
      % (h_in[first_layer].mean(), q0.n_inverted, q0.sj_min, q0.sj_mean, s0.size_ratio_max))
schedule = hs.suggest_schedule(q0, outer=OUTER)

meshes = {"input": before}
rows = {"input": dict(q=q0, shape=s0, thick_mean_pct=0.0, thick_max_pct=0.0, keep_move=0.0)}
print()
for label, cons in (("unprotected", []), ("protected", [hs.KeepLayer(wall_ids, 1)])):
    r = hs.resolve(before, cons)
    out, logs = hs.run_schedule(before, r, schedule)
    q = logs[-1].after
    s = hs.shape_stats(out.points, out.corners)
    frozen = np.flatnonzero(r.kind == hs.FROZEN)
    chg = 100 * (height(out)[first_layer] - h_in[first_layer]) / h_in[first_layer]
    keep = (np.flatnonzero(K <= 1) if cons else wall_ids)                   # rings=1 also freezes k = 1
    meshes[label] = out
    rows[label] = dict(q=q, shape=s, thick_mean_pct=float(chg.mean()), thick_max_pct=float(chg.max()),
                       keep_move=float(np.abs(out.points[keep] - before.points[keep]).max()), keep_n=int(keep.size),
                       n_frozen=int(frozen.size), first_layer_mean=float(height(out)[first_layer].mean()))
    d = rows[label]
    print("%-12s first-layer thickness change: mean %+.1f %%, max %+.1f %% (mean thickness %.4f)"
          % (label, d["thick_mean_pct"], d["thick_max_pct"], d["first_layer_mean"]))
    print("             max move of the %d %s vertices: %.1e | frozen vertices: %d"
          % (keep.size, "z<=1 (KeepLayer)" if cons else "z=0 (boundary)", d["keep_move"], d["n_frozen"]))
    print("             inverted %d | sj min %.4f mean %.4f | size_ratio_max %.2f (p95 %.2f)"
          % (q.n_inverted, q.sj_min, q.sj_mean, s.size_ratio_max, s.size_ratio_p95))

sj = {k: per_hex_sj(m) for k, m in meshes.items()}
for label, m in meshes.items():
    writer.to_vtu(m, "04_thin_layer_%s.vtu" % label)

mesh = meshes["protected"]
