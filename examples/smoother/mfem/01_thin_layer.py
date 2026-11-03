"""Hex smoothing of a designed thin layer with MFEM's TMOP: what the target keeps.

The block is the one of ``examples/smoother/hex/04_thin_layer.py``: x, y in [0, 1] (6 x 6 cells), z in [0, 1] in 8 graded
layers, the first 0.02 thick; the interior vertices of z-planes k >= 2 are perturbed, so the designed
first layer is clean going in (288 hexes).  Its hexes carry the element tag ``"layer"`` (the first
layer) or ``"bulk"``.  The boundary is frozen in every run.  Three runs on the same input:

    native          hs.suggest_schedule(q0, outer=3), the toolkit's own optimizer
    ideal           the ``tmop`` stage with target="ideal": every hex is pulled toward a cube
    input_designed  the ``tmop`` stage with target="input", designed="layer": the input shape is
                    the target, so the layer is held and the interior is repaired

Two side runs (not in the viewer) tell what ``designed`` does: ``input_plain`` is ``target="input"``
without it, and ``ideal_designed`` is ``target="ideal"`` with the layer held.  Each stage logs a
``reason`` line saying what it held fixed; it is printed below.

    PYTHONPATH=. python examples/smoother/mfem/01_thin_layer.py

Needs PyMFEM: ``pip install nekmeshpy[smooth]``.  Writes
``01_thin_layer_{input,native,ideal,input_designed}.vtu`` to the current directory and leaves ``mesh``
(the input_designed result), ``rows``, ``side``, ``meshes``, ``sj``, ``before`` and ``first_layer``
(per-hex flag) in the namespace.
"""

import time

import numpy as np

from nekmeshpy import writer
from nekmeshpy.core.smoother import mfem as tmop_bridge
from nekmeshpy.hexmesh import HexMesh
from nekmeshpy.hexmesh import smoother as hs
from nekmeshpy.tags.tags import Tags

if not tmop_bridge.available():
    raise ImportError("01_thin_layer.py needs PyMFEM (MFEM's TMOP): pip install nekmeshpy[smooth]")

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
first_layer = (K[hexes[:, 0]] == 0)                                         # per-hex flag
before = HexMesh.from_corners(P, hexes, Tags.from_dense(np.where(first_layer, "layer", "bulk")))
print("perturbed input: %d vertices moved; first layer hexes %d" % (int(pert.sum()), int(first_layer.sum())))


def per_hex_sj(m):
    return hs.corner_metrics(m.points, m.corners).sj.min(axis=1)


def height(m):
    """Mean height of every hex: the mean length of its four vertical edges."""
    p, h = m.points, m.corners
    return np.mean([np.linalg.norm(p[h[:, k + 4]] - p[h[:, k]], axis=1) for k in range(4)], axis=0)


h_in = height(before)
r = hs.resolve(before, [], boundary="frozen")
q0 = hs.quality(before.points, before.corners)
print("input: first-layer thickness %.4f (mean) | inverted %d | sj min %.4f mean %.4f"
      % (h_in[first_layer].mean(), q0.n_inverted, q0.sj_min, q0.sj_mean))


def run(schedule):
    t = time.perf_counter()
    out, logs = hs.run_schedule(before, r, schedule)
    seconds = time.perf_counter() - t
    chg = 100 * (height(out)[first_layer] - h_in[first_layer]) / h_in[first_layer]
    d = dict(q=logs[-1].after, thick_mean_pct=float(chg.mean()), thick_max_abs_pct=float(np.abs(chg).max()),
             seconds=seconds, stages=len(logs), reason=logs[-1].reason)
    return out, d


main_runs = {
    "native": hs.suggest_schedule(q0, outer=OUTER),
    "ideal": [("tmop", {"target": "ideal"})],
    "input_designed": [("tmop", {"target": "input", "designed": "layer"})],
}
side_runs = {
    "input_plain": [("tmop", {"target": "input"})],
    "ideal_designed": [("tmop", {"target": "ideal", "designed": "layer"})],
}

meshes = {"input": before}
rows = {"input": dict(q=q0, thick_mean_pct=0.0, thick_max_abs_pct=0.0, seconds=0.0, stages=0)}
side, side_meshes = {}, {}
print()
for label, schedule in main_runs.items():
    meshes[label], rows[label] = run(schedule)
for label, schedule in side_runs.items():
    side_meshes[label], side[label] = run(schedule)
side["input_plain"]["max_point_diff"] = float(
    np.abs(side_meshes["input_plain"].points - meshes["input_designed"].points).max())

for label, d in list(rows.items()) + [("(side) " + k, v) for k, v in side.items()]:
    q = d["q"]
    print("%-22s first layer: mean %+.1f %%, max |change| %.1f %% | inverted %d | sj min %.4f mean %.4f | %.2f s"
          % (label, d["thick_mean_pct"], d["thick_max_abs_pct"], q.n_inverted, q.sj_min, q.sj_mean, d["seconds"]))
    if d.get("reason") and label != "native":
        print("    reason: %s" % d["reason"])
print("input_plain vs input_designed: max point difference %.1e" % side["input_plain"]["max_point_diff"])

sj = {k: per_hex_sj(m) for k, m in meshes.items()}
for label, m in meshes.items():
    writer.to_vtu(m, "01_thin_layer_%s.vtu" % label)

mesh = meshes["input_designed"]
