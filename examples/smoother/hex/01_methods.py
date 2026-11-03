"""Hex smoothing, three methods on one perturbed input: Laplace, optimizer, suggested schedule.

The domain is a 90-degree cylindrical-shell sector (r in [0.5, 1], theta in [0, 90 deg],
z in [0, 0.5]) meshed 6 x 8 x 4, so it is curved and the best mesh is *not* all cubes and
the methods can be told apart.  The boundary is frozen; the interior vertices of the
perturbed input are displaced by up to 0.25 of the local cell.  The same input goes through

    laplace   -- 30 element-shrink Laplace passes (cheap, even, local)
    optimizer -- 200 conjugate-gradient iterations on the shape objective
    schedule  -- hs.suggest_schedule(q0): outer x [laplace(3) + optimize(20)] + 2 local lifts

Per run: inverted count, scaled Jacobian min / mean, worst 5 %, objective per hex (F/hex,
0 = every corner a cube corner, lower is better) and wall time.  Every number printed is
computed here.

    PYTHONPATH=. python examples/smoother/hex/01_methods.py

Writes ``01_methods_{input,laplace,optimizer,schedule}.vtu`` to the current directory and
leaves ``mesh`` (the schedule result), ``rows``, ``meshes`` and ``sj`` in the namespace.
"""

import time

import numpy as np

from nekmeshpy import writer
from nekmeshpy.hexmesh import HexMesh
from nekmeshpy.hexmesh import smoother as hs

# -- parameters --------------------------------------------------------------
R_IN, R_OUT, Z_LEN = 0.5, 1.0, 0.5
NR, NT, NZ = 6, 8, 4
PERTURB = 0.25               # interior vertex displacement, fraction of the local cell
SEED = 1
OUTER = 10                   # suggest_schedule default
LAPLACE = [("laplace", {"passes": 30})]
OPTIMIZER = [("optimize", {"iters": 200})]

# -- the sector: (r, theta, z) is right-handed, corners in Nek order ----------
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
inner = inner.ravel()
cell = np.array([(R_OUT - R_IN) / NR, 0.5 * np.pi * 0.75 / NT, Z_LEN / NZ])   # r, arc at mid-radius, z
rng = np.random.default_rng(SEED)
d = rng.uniform(-PERTURB, PERTURB, (int(inner.sum()), 3)) * cell
# displacement in the local (radial, tangential, z) frame
th_in = T.ravel()[inner]
P = exact.copy()
P[inner, 0] += d[:, 0] * np.cos(th_in) - d[:, 1] * np.sin(th_in)
P[inner, 1] += d[:, 0] * np.sin(th_in) + d[:, 1] * np.cos(th_in)
P[inner, 2] += d[:, 2]
before = HexMesh.from_corners(P, hexes)
resolved = hs.resolve(before, [])                    # frozen boundary


def per_hex_sj(m):
    return hs.corner_metrics(m.points, m.corners).sj.min(axis=1)


def show(label, q, seconds=None):
    t = "" if seconds is None else "  %.2f s" % seconds
    print("%-10s inverted %3d | sj min %.4f mean %.4f | worst 5%% %.4f | F/hex %.4f%s"
          % (label, q.n_inverted, q.sj_min, q.sj_mean, q.sj_q05, q.objective_mean, t))


q0 = hs.quality(before.points, before.corners)
schedule = hs.suggest_schedule(q0, outer=OUTER)
print("sector: %d hexes (%dx%dx%d), perturbed input %.2f cell, suggest_schedule outer=%d (%d stages)"
      % (before.n_hexes, NR, NT, NZ, PERTURB, OUTER, len(schedule)))
show("input", q0)

meshes = {"input": before}
rows = {"input": dict(q=q0, seconds=0.0)}
for label, sched in (("laplace", LAPLACE), ("optimizer", OPTIMIZER), ("schedule", schedule)):
    t0 = time.perf_counter()
    out, logs = hs.run_schedule(before, resolved, sched)
    dt = time.perf_counter() - t0
    meshes[label] = out
    rows[label] = dict(q=logs[-1].after, seconds=dt, stages=len(sched))
    show(label, logs[-1].after, dt)

sj = {k: per_hex_sj(m) for k, m in meshes.items()}
moved = float(np.abs(meshes["schedule"].points[~inner] - before.points[~inner]).max())
print("\nboundary vertices: max move %.1e (frozen)" % moved)
f = {k: v["q"] for k, v in rows.items()}
print("reading: laplace is %.1fx cheaper than the optimizer; F/hex laplace %.3f, optimizer %.3f, "
      "schedule %.3f; highest sj min: %s"
      % (rows["optimizer"]["seconds"] / rows["laplace"]["seconds"], f["laplace"].objective_mean,
         f["optimizer"].objective_mean, f["schedule"].objective_mean,
         max(("laplace", "optimizer", "schedule"), key=lambda k: f[k].sj_min)))

for label, m in meshes.items():
    writer.to_vtu(m, "01_methods_%s.vtu" % label)

mesh = meshes["schedule"]
