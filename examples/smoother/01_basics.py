"""Smoothing basics: Laplace only, optimizer only, and the user schedule, on one jittered patch.

The patch is a 90-degree annular sector (a curved patch, so the best mesh is *not* all squares
and the three methods can be told apart), boundary frozen, interior vertices perturbed by up
to 0.45 of a cell.  The same input goes through

    laplace   -- 30 element-shrink Laplace passes (cheap, even, local)
    optimizer -- 200 conjugate-gradient iterations on the shape objective
    schedule  -- smooth.suggest_schedule: 10 x [laplace(3) + optimize(20)] + 2 local lifts

and each result is reported with the numbers that matter: scaled Jacobian min / mean, the worst
5 % of quads, inverted count, and the objective per quad (0 = every corner is a perfect right
angle, so lower is better).  Every number printed is computed here.

    PYTHONPATH=. python examples/smoother/01_basics.py

Produces ``01_basics_{before,laplace,optimizer,schedule}.vtu`` (open in ParaView, colour by
``sj_min``) and ``01_basics_frames.json`` (stage-by-stage playback of the schedule run).
"""

import time

import numpy as np

from nekmeshpy import quadmesh
from nekmeshpy.io.writer import quad_to_vtu
from nekmeshpy.quadmesh.smoother import constrain as cn
from nekmeshpy.quadmesh.smoother import frames, smooth

# -- parameters --------------------------------------------------------------
N = 14                       # cells per side
R_IN, R_OUT = 0.4, 1.0       # sector radii
JITTER = 0.45                # interior vertex displacement, fraction of a cell
SEED = 4
LAPLACE = [("laplace", {"passes": 30, "w": 0.97})]
OPTIMIZER = [("optimize", {"iters": 200})]

# -- the patch: a polar grid, connectivity CCW --------------------------------
u = np.linspace(0.0, 1.0, N + 1)
U, V = np.meshgrid(u, u, indexing="ij")
rad = R_IN + (R_OUT - R_IN) * U
ang = 0.5 * np.pi * V
exact = np.stack([(rad * np.cos(ang)).ravel(), (rad * np.sin(ang)).ravel(), np.zeros(U.size)], axis=1)
idx = np.arange((N + 1) ** 2).reshape(N + 1, N + 1)
quads = np.stack([idx[:-1, :-1], idx[1:, :-1], idx[1:, 1:], idx[:-1, 1:]], axis=-1).reshape(-1, 4)
normals = smooth.reference_normals(exact, quads)        # read once from the clean patch

interior = ((U > 0) & (U < 1) & (V > 0) & (V < 1)).ravel()
jittered = exact.copy()
cell = (R_OUT - R_IN) / N
jittered[interior, :2] += np.random.default_rng(SEED).uniform(-JITTER, JITTER, (interior.sum(), 2)) * cell
before = quadmesh.QuadMesh.from_corners(jittered, quads)
resolved = cn.resolve(before, [])                       # no constraints: the boundary is frozen by default


def per_quad_sj(m):
    return smooth.corner_metrics(m.points, m.corners, normals).sj.min(axis=1)


def report(label, q, seconds=None):
    t = "" if seconds is None else "  %.2f s" % seconds
    print("%-10s inverted %3d | sj min %.4f mean %.4f | worst 5%% %.4f | F/quad %.4f%s"
          % (label, q.n_inverted, q.sj_min, q.sj_mean, q.sj_q05, q.objective_mean, t))


# -- run the three methods on the same input ----------------------------------
q0 = smooth.quality(before.points, before.corners, normals)
schedule = smooth.suggest_schedule(q0)                  # = user_schedule(): nothing starts inverted
print("patch: %d quads, jitter %.2f cell; start" % (before.n_quads, JITTER))
report("input", q0)

results = {}
for label, sched in (("laplace", LAPLACE), ("optimizer", OPTIMIZER), ("schedule", schedule)):
    t0 = time.perf_counter()
    out, logs = smooth.run_schedule(before, resolved, sched, normals=normals, record=(label == "schedule"))
    results[label] = (out, logs, time.perf_counter() - t0)
    report(label, logs[-1].after, results[label][2])

# -- convergence of the objective per quad ------------------------------------
print("\nobjective per quad after each stage of the schedule (laplace stages even, optimizer stages descend):")
slog = results["schedule"][1]
for k in range(0, len(slog), 2):
    print("  outer %2d: after laplace %.4f -> after optimize %.4f"
          % (k // 2 + 1, slog[k].after.objective_mean, slog[k + 1].after.objective_mean))
print("  local lifts: %s" % " -> ".join("%.4f" % lg.after.objective_mean for lg in slog[-2:]))
hist = results["optimizer"][1][0].history
print("optimizer-only history (pass, inverted, sj min, F/quad): first %s, last %s" % (hist[0], hist[-1]))

# -- what each method is good / bad at, from the numbers ----------------------
f = {k: v[1][-1].after for k, v in results.items()}
t = {k: v[2] for k, v in results.items()}
print("\nreading the numbers:")
print("  laplace   : %.1fx cheaper than the optimizer, keeps every quad valid, but leaves F/quad at %.3f "
      "(optimizer %.3f) -- it evens the spacing, it does not minimise the shape error"
      % (t["optimizer"] / t["laplace"], f["laplace"].objective_mean, f["optimizer"].objective_mean))
print("  optimizer : lowest F/quad (%.3f), worst 5%% %.3f vs laplace %.3f; costs %.2f s"
      % (f["optimizer"].objective_mean, f["optimizer"].sj_q05, f["laplace"].sj_q05, t["optimizer"]))
F = [lg.after.objective_mean for lg in slog]
settled = next(k // 2 + 1 for k in range(0, len(slog) - 2, 2) if abs(F[k + 1] - F[-1]) < 1e-3 * F[-1])
print("  schedule  : F/quad %.4f vs optimizer alone %.4f (difference %.1e) in %.2f s vs %.2f s; the objective has "
      "settled (within 0.1 %%) after outer loop %d of %d, so the later loops buy nothing here -- the schedule's "
      "value is robustness on harder starts (see 02_untangle), not speed"
      % (f["schedule"].objective_mean, f["optimizer"].objective_mean,
         abs(f["schedule"].objective_mean - f["optimizer"].objective_mean), t["schedule"], t["optimizer"],
         settled, (len(slog) - 2) // 2))
print("  highest sj min: %s (%.4f)" % (max(f, key=lambda k: f[k].sj_min), max(v.sj_min for v in f.values())))

# -- outputs ------------------------------------------------------------------
quad_to_vtu(before, "01_basics_before.vtu", cell_data={"sj_min": per_quad_sj(before)})
for label, (out, _, _) in results.items():
    quad_to_vtu(out, "01_basics_%s.vtu" % label, point_data={"displacement": out.points - before.points},
                cell_data={"sj_min": per_quad_sj(out)})
frames.write_frames("01_basics_frames.json", before, results["schedule"][1], normals,
                    title="01 basics: jittered annular patch, user schedule", decimals=5)

mesh = results["schedule"][0]
