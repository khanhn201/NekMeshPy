"""Case study: the 2-D conjugate channel with a free-standing obstacle (examples/channel_obstacle_cht.py).

The real mesh of that example (a fluid channel between two solid walls, a rectangular obstacle floating in
it, clustered toward the obstacle) is built by running the example itself, then taken to its straight-sided
shadow (``highorder.linearize`` -- corners, connectivity and edge tags unchanged) because the smoother moves
vertices only.  The example's boundary tags drive the constraints:

* ``obstacle``, ``inlet``, ``outlet``, ``solid_end``  -- ``Frozen`` (the body outline and the ends are the
  design, not something to relax);
* ``solid_outer`` (the two outer walls) -- ``OnCurve`` on its straight line, sliding along it;
* the fluid/solid **interface** (an interior row of edges the tags do not name: found as the vertices on
  ``y = Y0`` and ``y = Y1``) -- ``OnCurve`` on its line, sliding, so the interface stays a straight line.

Two runs of the same schedule: (1) the mesh as built -- the honest result is "almost nothing changes, it is
already a good mesh" -- and (2) the same mesh with every movable vertex perturbed by ``NOISE`` of its
local edge length.  For each: inverted and overlapping quads, scaled Jacobian min/mean/worst 5 %, the
largest size jump between neighbouring quads, how far anything moved (in local edge lengths) and the
on-target distances.  If a stressed run cannot end with zero overlaps, it says so.

    PYTHONPATH=. python examples/smoother/quad/cs_channel_obstacle.py

Produces ``cs_channel_<asis|stress>_{before,after}.vtu`` (``sj_min`` per cell, ``displacement`` per point)
and ``cs_channel_<asis|stress>_frames.json``.  ``mesh`` is the smoothed as-built mesh.
"""

import logging
import os
import runpy
import time

import numpy as np

from nekmeshpy import quadmesh
from nekmeshpy.io.writer import quad_to_vtu
from nekmeshpy.quadmesh.smoother import constrain as cn
from nekmeshpy.quadmesh.smoother import frames, highorder, smooth

# -- parameters --------------------------------------------------------------
NOISE = 0.2                  # stress variant: tangential perturbation, fraction of the local edge length
SEED = 5
WINDOW = 0.2                 # playback JSON: only the quads within |x| <= WINDOW of the obstacle (the whole mesh is ~1 MB a variant)
SCHEDULE = smooth.user_schedule(outer=2, n_opt=10)

t_start = time.perf_counter()
example = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir, "channel_obstacle_cht.py")
cwd0 = os.getcwd()
ns = runpy.run_path(example, run_name="__main__")          # the example's own mesh (order 2, tagged)
logging.getLogger().setLevel(logging.ERROR)
Y0, Y1 = ns["Y0"], ns["Y1"]                                # the fluid/solid interface rows
start = highorder.linearize(ns["mesh"])
P0 = np.array(start.points, dtype=float)
nrm = smooth.reference_normals(start.points, start.corners)


# -- constraints from the example's own edge tags ---------------------------------------------------
def tag_vertices(m, tag):
    return np.unique(m.line_mesh.lines[m.edge_tags.flatnonzero(tag)])


def line_at(ids, axis_point, direction):
    return cn.OnCurve(ids, cn.line(axis_point, direction))


frozen_ids = np.unique(np.concatenate([tag_vertices(start, t) for t in ("obstacle", "inlet", "outlet", "solid_end")]))
outer_ids = np.setdiff1d(tag_vertices(start, "solid_outer"), frozen_ids)
y_wall = (P0[:, 1].min(), P0[:, 1].max())
constraints = [cn.Frozen(frozen_ids)]
for y in y_wall:
    constraints.append(line_at(outer_ids[np.isclose(P0[outer_ids, 1], y)], [0, y, 0], [1, 0, 0]))
interface = {y: np.flatnonzero(np.isclose(P0[:, 1], y)) for y in (Y0, Y1)}
for y, ids in interface.items():
    ids = np.setdiff1d(ids, frozen_ids)
    constraints.append(line_at(ids, [0, y, 0], [1, 0, 0]))
res = cn.resolve(start, constraints)                       # remaining boundary vertices (none) would freeze
fixed = res.kind == cn.FROZEN
free_ids = np.flatnonzero(res.kind == cn.FREE)
slide_ids = np.flatnonzero(res.kind == cn.ON_CURVE)
print("channel_obstacle_cht: %d quads (%d fluid, %d solid), %d vertices; %d frozen (obstacle, inlet/outlet, "
      "solid ends, corners), %d sliding on a line (outer walls + interface), %d free"
      % (start.n_quads, int((start.element_tags.to_dense() == "fluid").sum()),
         int((start.element_tags.to_dense() == "solid").sum()), start.n_points, fixed.sum(), slide_ids.size,
         free_ids.size))
for note in res.notes:
    print("  note:", note)


def per_quad_sj(m):
    return smooth.corner_metrics(m.points, m.corners, nrm).sj.min(axis=1)


def reading(label, m):
    q = smooth.quality(m.points, m.corners, nrm)
    s = smooth.shape_stats(m.points, m.corners)
    ov = int(quadmesh.overlapping_quads(m).size)
    print("  %-8s inverted %d | overlapping %d | scaled J min %.4f mean %.4f worst 5%% %.4f | size ratio max %.3f "
          "mean %.3f | aspect max %.1f" % (label, q.n_inverted, ov, q.sj_min, q.sj_mean, q.sj_q05,
                                           s.size_ratio_max, s.size_ratio_mean, s.aspect_max))
    return q, s, ov


def perturbed(m, seed):
    rng = np.random.default_rng(seed)
    P = np.array(m.points, dtype=float)
    ids = np.flatnonzero(res.kind != cn.FROZEN)
    ell = cn.vertex_length(m.points, m.corners)[ids, None]
    P[ids, :2] += rng.uniform(-NOISE, NOISE, (ids.size, 2)) * ell
    return smooth.with_points(m, cn.snap(P, res))          # sliding vertices stay on their lines


runs = {}
for label, m0 in (("asis", start), ("stress", perturbed(start, SEED))):
    t0 = time.perf_counter()
    print("\n%s: %s" % (label, "the mesh as built by the example" if label == "asis" else
                          "free and sliding vertices perturbed by up to %.2f of their local edge" % NOISE))
    before = reading("input", m0)
    out, logs = smooth.run_schedule(m0, res, SCHEDULE, normals=nrm, record=True)
    after = reading("smoothed", out)
    on = cn.on_target_report(out.points, out.corners, res)
    ell = cn.vertex_length(m0.points, m0.corners)
    moved = np.linalg.norm(out.points - m0.points, axis=1) / ell
    dev = np.linalg.norm(out.points - P0, axis=1) / ell     # distance from the example's own mesh
    print("  moved: max %.3f local edges (mean %.4f); from the example's mesh: max %.3f edges" %
          (moved.max(), moved.mean(), dev.max()))
    print("  on target: %s" % str(on))
    if label == "asis":
        print("  verdict: %s" % ("the input is already a perfect tensor grid (every quad a rectangle, scaled J %.4f): the "
                                 "smoother cannot improve it; min scaled J %s from %.4f to %.4f and it moved vertices up to "
                                 "%.2f local edges (mean %.3f) by relaxing the designed clustering -- the shape metric knows "
                                 "angles, not the sizes the design asked for"
                                 % (before[0].sj_min, "fell" if after[0].sj_min < before[0].sj_min else "held",
                                    before[0].sj_min, after[0].sj_min, moved.max(), moved.mean())))
    if after[2] > 0:
        print("  NOT CLEAN: %d quads still overlap after smoothing (known open issue: a folded input can leave a "
              "persistent local overlap); %d overlapped before" % (after[2], before[2]))
    runs[label] = dict(before=m0, out=out, logs=logs, q=(before[0], after[0]), ov=(before[2], after[2]), on=on,
                       moved=moved, size_ratio=(before[1].size_ratio_max, after[1].size_ratio_max))
    quad_to_vtu(m0, "cs_channel_%s_before.vtu" % label, cell_data={"sj_min": per_quad_sj(m0)})
    quad_to_vtu(out, "cs_channel_%s_after.vtu" % label, point_data={"displacement": out.points - m0.points},
                cell_data={"sj_min": per_quad_sj(out)})
    cut = np.abs(m0.points[m0.corners][:, :, 0]).max(axis=1) <= WINDOW        # quads of the window
    keep = np.unique(m0.corners[cut])
    remap = np.full(m0.n_points, -1)
    remap[keep] = np.arange(keep.size)
    win = quadmesh.QuadMesh.from_corners(m0.points[keep], remap[m0.corners[cut]])
    frames.write_frames("cs_channel_%s_frames.json" % label, win, [lg._replace(points=lg.points[keep]) for lg in logs],
                        nrm[cut], decimals=3, title="channel with obstacle, window |x| <= %.1f (%s)" % (WINDOW, label))
    print("  (%.1f s)" % (time.perf_counter() - t0))
mesh = runs["asis"]["out"]
print("\ntotal %.1f s" % (time.perf_counter() - t_start))
