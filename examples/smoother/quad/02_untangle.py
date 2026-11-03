"""Untangling: the plain user schedule against the delta-continuation, on random tangled meshes.

Each case is a unit square of N x N quads (boundary frozen) in which three interior vertices have
been dragged by up to 1.6 cells in each direction -- the protocol of the random-tangle test in
``tests/smoother/quad/test_long_schedule.py``.  Cases that happen not to tangle are skipped; the
first CASES tangled ones are used.  Both schedules start from the same tangled mesh:

    plain    -- smooth.user_schedule(): 10 x [laplace(3) + optimize(20)] + 2 local lifts
    untangle -- smooth.untangle_schedule() (optimizer stages with a shrinking barrier width,
                delta 1 -> 0.01) followed by the same user schedule; this is what
                ``suggest_schedule`` proposes when any quad starts inverted

A case counts as repaired when no quad is inverted at the end.  The demo prints what it measures,
including the cases where a method fails; the count depends on CASES and SEED, and the report's
23/40 vs 40/40 is not asserted here.

    PYTHONPATH=. python examples/smoother/quad/02_untangle.py

Produces ``02_untangle_{before,plain,untangle}.vtu`` and ``02_untangle_frames.json`` for the first
case that the plain schedule fails and the continuation repairs (or the first case, if none).
"""

import time

import numpy as np

from nekmeshpy import quadmesh
from nekmeshpy.io.writer import quad_to_vtu
from nekmeshpy.quadmesh.smoother import constrain as cn
from nekmeshpy.quadmesh.smoother import frames, smooth

# -- parameters --------------------------------------------------------------
N = 8                        # cells per side
CASES = 8                    # tangled meshes to run
DRAGGED = 3                  # vertices moved per case
DRAG = 1.6                   # largest drag, in cells
SEED = 2

xs = np.linspace(0.0, 1.0, N + 1)
X, Y = np.meshgrid(xs, xs, indexing="ij")
grid = np.stack([X.ravel(), Y.ravel(), np.zeros(X.size)], axis=1)
idx = np.arange((N + 1) ** 2).reshape(N + 1, N + 1)
quads = np.stack([idx[:-1, :-1], idx[1:, :-1], idx[1:, 1:], idx[:-1, 1:]], axis=-1).reshape(-1, 4)
clean = quadmesh.QuadMesh.from_corners(grid, quads)
normals = smooth.reference_normals(grid, quads)         # from the clean grid, never from a tangled one
inner = np.flatnonzero((grid[:, 0] > 0) & (grid[:, 0] < 1) & (grid[:, 1] > 0) & (grid[:, 1] < 1))

rng = np.random.default_rng(SEED)
plain_ok, untangle_ok, rows, show = 0, 0, [], None
t_plain = t_untangle = 0.0
while len(rows) < CASES:
    pts = grid.copy()
    for v in rng.choice(inner, DRAGGED, replace=False):
        pts[v, :2] += rng.uniform(-DRAG, DRAG, 2) / N
    tangled = smooth.with_points(clean, pts)
    q0 = smooth.quality(tangled.points, tangled.corners, normals)
    if q0.n_inverted == 0:
        continue                                        # this draw did not tangle anything
    resolved = cn.resolve(tangled, [])

    t0 = time.perf_counter()
    p_mesh, p_logs = smooth.run_schedule(tangled, resolved, smooth.user_schedule(), normals=normals)
    t_plain += time.perf_counter() - t0
    t0 = time.perf_counter()
    u_sched = smooth.untangle_schedule() + smooth.user_schedule()
    u_mesh, u_logs = smooth.run_schedule(tangled, resolved, u_sched, normals=normals, record=True)
    t_untangle += time.perf_counter() - t0

    qp, qu = p_logs[-1].after, u_logs[-1].after
    plain_ok += qp.n_inverted == 0
    untangle_ok += qu.n_inverted == 0
    rows.append((q0.n_inverted, qp.n_inverted, qu.n_inverted, qp.sj_min, qu.sj_min))
    contrast = qp.n_inverted > 0 and qu.n_inverted == 0     # plain fails, continuation repairs
    if show is None or (contrast and not show[0]):
        show = (contrast, tangled, p_mesh, u_mesh, u_logs)

print("%d tangled %dx%d meshes (%d vertices dragged up to %.1f cells, seed %d)" % (CASES, N, N, DRAGGED, DRAG, SEED))
print("case  inverted at start | plain: inverted, sj min | untangle: inverted, sj min")
for i, (n0, np_, nu, sp, su) in enumerate(rows, 1):
    print("%4d  %17d | %14d %7.3f | %17d %7.3f" % (i, n0, np_, sp, nu, su))
print("\ntangles repaired: plain %d/%d vs untangle %d/%d   (%.1f s vs %.1f s total)"
      % (plain_ok, CASES, untangle_ok, CASES, t_plain, t_untangle))
if untangle_ok < CASES:
    print("the continuation does not repair everything either: %d case(s) keep an inverted quad" % (CASES - untangle_ok))
else:
    print("the plain schedule leaves %d case(s) tangled; a Laplace pass has nothing to repair an inverted quad with, "
          "the barrier-smoothed optimizer stages pull the folded vertex through" % (CASES - plain_ok))

# -- outputs: the showcase case -----------------------------------------------
_, tangled, p_mesh, u_mesh, u_logs = show


def per_quad_sj(m):
    return smooth.corner_metrics(m.points, m.corners, normals).sj.min(axis=1)


for name, m in (("before", tangled), ("plain", p_mesh), ("untangle", u_mesh)):
    quad_to_vtu(m, "02_untangle_%s.vtu" % name, cell_data={"sj_min": per_quad_sj(m)})
frames.write_frames("02_untangle_frames.json", tangled, u_logs, normals,
                    title="02 untangle: three vertices dragged, delta continuation", decimals=5)

mesh = u_mesh
