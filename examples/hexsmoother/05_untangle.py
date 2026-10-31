"""Repairing inverted hexes: the optimizer's delta continuation, against Laplace and delta = 0.

Important: ``hs.untangle_schedule()`` is a *preset of* ``optimize`` -- a list of optimize
stages with a shrinking ``delta`` (1, 0.3, 0.1, 0.03, 0.01) -- and NOT a separate untangler.
The exact metric (``delta = 0``) has a wall at zero Jacobian and cannot pull a folded hex
through it; a wide ``delta`` smooths that wall away so the fold can be undone, and each
narrower stage sharpens the metric back.

The start is the clean curved sector of 01_methods.py (6 x 8 x 4 = 192 hexes, frozen
boundary) with 3 interior vertices pushed by 1.0 cell in random directions, which folds a
handful of hexes.  Four runs on that input:

    laplace    -- 30 passes (its guard refuses any pass that adds an inverted hex)
    optimize0  -- optimize(iters=200, delta=0), the exact metric
    untangle   -- hs.untangle_schedule()           (5 optimize stages)
    suggested  -- hs.suggest_schedule(q0)           (= untangle_schedule + user_schedule, as q0 has
                                                    inverted hexes)

Per run: inverted count, scaled Jacobian min / mean, number of stages, wall time.

    PYTHONPATH=. python examples/hexsmoother/05_untangle.py

Writes ``05_untangle_{input,laplace,optimize0,untangle,suggested}.vtu`` to the current
directory and leaves ``mesh`` (the suggested-schedule result), ``rows``, ``meshes``, ``sj``
and ``inverted_in_input`` (per-hex flag) in the namespace.
"""

import time

import numpy as np

from nekmeshpy import writer
from nekmeshpy.hexmesh import HexMesh
from nekmeshpy.hexmesh import smoother as hs

# -- parameters --------------------------------------------------------------
R_IN, R_OUT, Z_LEN = 0.5, 1.0, 0.5
NR, NT, NZ = 6, 8, 4
N_PUSHED = 3                 # interior vertices pushed
PUSH = 1.0                   # push length, in mean cell edges
SEED = 5

# -- the clean sector (as in 01_methods.py) ------------------------------------
r = np.linspace(R_IN, R_OUT, NR + 1)
th = np.linspace(0.0, 0.5 * np.pi, NT + 1)
z = np.linspace(0.0, Z_LEN, NZ + 1)
R, T, Z = np.meshgrid(r, th, z, indexing="ij")
exact = np.stack([(R * np.cos(T)).ravel(), (R * np.sin(T)).ravel(), Z.ravel()], axis=1)
idx = np.arange(exact.shape[0]).reshape(NR + 1, NT + 1, NZ + 1)
c = [idx[:-1, :-1, :-1], idx[1:, :-1, :-1], idx[1:, 1:, :-1], idx[:-1, 1:, :-1],
     idx[:-1, :-1, 1:], idx[1:, :-1, 1:], idx[1:, 1:, 1:], idx[:-1, 1:, 1:]]
hexes = np.stack(c, axis=-1).reshape(-1, 8)
inner = np.zeros(idx.shape, bool)
inner[1:-1, 1:-1, 1:-1] = True
clean = HexMesh.from_corners(exact, hexes)
qc = hs.quality(clean.points, clean.corners)
print("clean sector: %d hexes, inverted %d, sj min %.4f" % (clean.n_hexes, qc.n_inverted, qc.sj_min))

# -- push a few interior vertices hard ------------------------------------------
cell = np.array([(R_OUT - R_IN) / NR, 0.5 * np.pi * 0.75 / NT, Z_LEN / NZ])
h = float(np.linalg.norm(cell) / np.sqrt(3.0))                    # mean cell edge
rng = np.random.default_rng(SEED)
pushed = rng.choice(np.flatnonzero(inner.ravel()), N_PUSHED, replace=False)
dirs = rng.normal(size=(N_PUSHED, 3))
dirs /= np.linalg.norm(dirs, axis=1)[:, None]
P = exact.copy()
P[pushed] += dirs * PUSH * h
before = HexMesh.from_corners(P, hexes)
resolved = hs.resolve(before, [])


def per_hex_sj(m):
    return hs.corner_metrics(m.points, m.corners).sj.min(axis=1)


q0 = hs.quality(before.points, before.corners)
inverted_in_input = per_hex_sj(before) <= 0.0
print("perturbed input: %d vertices pushed by %.2f cell (%.3f), %d hexes inverted, sj min %.4f mean %.4f"
      % (N_PUSHED, PUSH, PUSH * h, q0.n_inverted, q0.sj_min, q0.sj_mean))

schedules = {
    "laplace": [("laplace", {"passes": 30})],
    "optimize0": [("optimize", {"iters": 200, "delta": 0.0})],
    "untangle": hs.untangle_schedule(),
    "suggested": hs.suggest_schedule(q0),
}
meshes = {"input": before}
rows = {"input": dict(q=q0, stages=0, seconds=0.0)}
print()
for label, sched in schedules.items():
    t0 = time.perf_counter()
    out, logs = hs.run_schedule(before, resolved, sched)
    dt = time.perf_counter() - t0
    q = logs[-1].after
    meshes[label] = out
    rows[label] = dict(q=q, stages=len(sched), seconds=dt,
                       still_inverted_input_hexes=int((per_hex_sj(out) <= 0.0)[inverted_in_input].sum()))
    print("%-10s inverted %3d | sj min %.4f mean %.4f | %2d stages | %.2f s"
          % (label, q.n_inverted, q.sj_min, q.sj_mean, len(sched), dt))
print("\nuntangle_schedule stages: %s" % [k["delta"] for _, k in schedules["untangle"]])
print("optimize with delta=0 left %d of the %d folded hexes inverted; the delta continuation left %d"
      % (rows["optimize0"]["q"].n_inverted, q0.n_inverted, rows["untangle"]["q"].n_inverted))

sj = {k: per_hex_sj(m) for k, m in meshes.items()}
for label, m in meshes.items():
    writer.to_vtu(m, "05_untangle_%s.vtu" % label)

mesh = meshes["suggested"]
