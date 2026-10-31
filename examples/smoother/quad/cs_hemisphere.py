"""Case study: the hemisphere surface of the flow-past-hemisphere shell (examples/flow_past_hemisphere.py).

The example fills a hex shell between ``quadmesh.hemisphere`` (a cubed-sphere cap resting on the floor ``z = 0``)
and a half box.  The hemisphere is rebuilt here with the example's own call and parameters (``R``, ``N_FACE``,
``N_HALF``) at order 1 and smoothed as a surface:

* every vertex off the **rim** is ``OnSurface`` the exact sphere (sliding, snapped back every step);
* the rim (the open boundary on the ground, tagged ``ground`` in the example) is **frozen**: it is where the
  shell's ground annulus is swept from, so it is the design, not something to relax.

The metric's normals are the **surface's own** (``constrain.surface_normals``), not the quads' winding.  There is
**no propagation into the hex shell**: the shell is not touched (the volume route is not built).

Two runs of the same schedule: (1) as built (the numbers say how much changes) and (2) with every movable
vertex perturbed tangentially by ``NOISE`` of its local edge.  Inverted and overlapping quads, scaled Jacobian min/mean/worst 5 %, the size jump between
neighbours, the on-target distances (and that the corners and creases hold) and what moved are printed.

    PYTHONPATH=. python examples/smoother/quad/cs_hemisphere.py

Produces ``cs_hemisphere_<asis|stress>_{before,after}.vtu`` (``sj_min`` per cell, ``displacement`` per
point).  No playback JSON (curved surfaces: the docs viewer for them is separate).  ``mesh`` is the smoothed
as-built hemisphere.
"""

import logging
import time

import numpy as np

from nekmeshpy import quadmesh
from nekmeshpy.io.writer import quad_to_vtu
from nekmeshpy.quadmesh.smoother import constrain as cn
from nekmeshpy.quadmesh.smoother import smooth

# -- parameters (flow_past_hemisphere.py's) ---------------------------------------
R = 0.5                      # hemisphere radius (rests on the floor z = 0)
N_FACE = 6                   # horizontal cells per direction on each patch
N_HALF = 6                   # vertical cells
# -- this study's ---------------------------------------------------------------
NOISE = 0.2                  # stress: tangential perturbation, fraction of the local edge
SEED = 11
SCHEDULE = smooth.user_schedule(outer=5)

logging.getLogger().setLevel(logging.ERROR)
t_start = time.perf_counter()
start = quadmesh.hemisphere(R, N_FACE, n_vertical=N_HALF, element_tag="hemisphere", rim_tag="ground")
rim = quadmesh.boundary_points(start)
rim_edges = start.edge_tags.flatnonzero("ground")
assert np.unique(start.line_mesh.lines[rim_edges]).size == rim.size       # the boundary is exactly the "ground" rim
inside = np.setdiff1d(np.arange(start.n_points), rim)
res = cn.resolve(start, [cn.Frozen(rim), cn.OnSurface(inside, cn.sphere([0, 0, 0], R))])


def per_quad_sj(m, nrm):
    return smooth.corner_metrics(m.points, m.corners, nrm).sj.min(axis=1)


def reading(label, m, nrm):
    q = smooth.quality(m.points, m.corners, nrm)
    s = smooth.shape_stats(m.points, m.corners)
    ov = int(quadmesh.overlapping_quads(m).size)
    print("  %-8s inverted %d | overlapping %d | scaled J min %.4f mean %.4f worst 5%% %.4f | size ratio max %.3f "
          "mean %.3f" % (label, q.n_inverted, ov, q.sj_min, q.sj_mean, q.sj_q05, s.size_ratio_max, s.size_ratio_mean))
    return q, s, ov


runs = {}
counts = res.counts()
print("hemisphere: %d quads, %d vertices; %d on the sphere, %d on the rim (frozen)"
      % (start.n_quads, start.n_points, counts["on_surface"], counts["frozen"]))
for note in res.notes:
    print("  note:", note)
movable = np.flatnonzero(res.kind != cn.FROZEN)
rng = np.random.default_rng(SEED)
P = np.array(start.points, dtype=float)
P[movable] += rng.uniform(-NOISE, NOISE, (movable.size, 3)) * cn.vertex_length(start.points, start.corners)[movable, None]
stressed = smooth.with_points(start, cn.snap(P, res))     # back onto the sphere: tangential perturbation only
for label, m0 in (("asis", start), ("stress", stressed)):
    t0 = time.perf_counter()
    nrm, nrep = cn.surface_normals(m0.points, m0.corners, cn.claimed_surfaces(res))
    print("\n%s: %s (%d quads disagree with the surface normal)"
          % (label, "as built" if label == "asis" else "perturbed by up to %.2f of the local edge" % NOISE, nrep.n_disagree))
    before = reading("input", m0, nrm)
    out, logs = smooth.run_schedule(m0, res, SCHEDULE, normals=nrm)
    after = reading("smoothed", out, nrm)
    on = cn.on_target_report(out.points, out.corners, res)
    ell = cn.vertex_length(m0.points, m0.corners)
    moved = np.linalg.norm(out.points - m0.points, axis=1) / ell
    print("  moved: max %.3f local edges (mean %.4f); rim moved %.1e" % (moved.max(), moved.mean(),
                                                                     np.abs(out.points[rim] - m0.points[rim]).max()))
    print("  on target: %s" % str(on))
    if label == "asis":
        print("  verdict: scaled J min %.4f -> %.4f, mean %.4f -> %.4f, largest move %.3f local edges" %
              (before[0].sj_min, after[0].sj_min, before[0].sj_mean, after[0].sj_mean, moved.max()))
    if after[2] > 0:
        print("  NOT CLEAN: %d quads still overlap after smoothing (%d before)" % (after[2], before[2]))
    runs[label] = dict(before=m0, out=out, logs=logs, q=(before[0], after[0]), ov=(before[2], after[2]), on=on,
                       moved=moved, size_ratio=(before[1].size_ratio_max, after[1].size_ratio_max))
    quad_to_vtu(m0, "cs_hemisphere_%s_before.vtu" % label, cell_data={"sj_min": per_quad_sj(m0, nrm)})
    quad_to_vtu(out, "cs_hemisphere_%s_after.vtu" % label, point_data={"displacement": out.points - m0.points},
                cell_data={"sj_min": per_quad_sj(out, nrm)})
    print("  (%.1f s)" % (time.perf_counter() - t0))
mesh = runs["asis"]["out"]
print("\ntotal %.1f s" % (time.perf_counter() - t_start))
