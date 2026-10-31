"""Case study: the O-grid cross-section of the circular pipe (examples/circular_pipe.py).

The pipe is a swept section, so smoothing the section is smoothing the pipe's cells.  The section is rebuilt here
with the example's own parameters and calls (``linemesh.circle`` + ``quadmesh.ogrid``, same ``RADIUS``,
``N_SIDE``, ``N_RADIAL``, ``CENTER_SCALE``, ``RADIAL_GRADING``), at order 1 because the smoother moves vertices
only; the example itself builds a 40-layer hex pipe this study does not need.

Constraints:

* the wall is declared ``OnCurve`` on the pipe circle (the target the vertices stay on);
* the boundary layers are protected with ``KeepLayer(wall, rings=KEEP_RINGS)``: the wall and the first rings off
  it are frozen as meshed, so the graded wall spacing (the reason the example grades the section radially, a first
  layer of 0.0006 under a 0.13 wall segment) cannot inflate.

**The two do not combine into a sliding wall**: ``KeepLayer`` freezes its seeds, so the wall it protects cannot slide
(``resolve`` lets a frozen claim win).  The alternative, freezing only rings 1..K and letting the wall slide, was
tried: the wall vertices then shear against the frozen first ring (a layer 0.0006 thick), and a wall slip of 0.2
segment leaves inverted quads that the smoother does not repair.  A sliding wall needs ``SlideColumns`` (rigid
columns across the layers) and a single slide direction, which a circle does not have.  So here the wall is
``OnCurve`` *and* frozen, and the printed on-target distance proves it is still on the circle.

Two runs of the same schedule: (1) the section as built (honest answer below) and (2) the same section with
the unprotected interior (everything beyond the kept rings) perturbed by ``NOISE`` of the
local edge length.  Inverted and overlapping quads, scaled Jacobian min/mean/worst 5 %, size jump between
neighbours, the layer thickness before/after, the wall's distance to the circle and what moved are printed.

    PYTHONPATH=. python examples/smoother/cs_pipe_section.py

Produces ``cs_pipe_<asis|stress>_{before,after}.vtu`` (``sj_min`` per cell, ``displacement`` per point) and
``cs_pipe_<asis|stress>_frames.json``.  ``mesh`` is the smoothed as-built section.
"""

import logging
import time

import numpy as np

from nekmeshpy import linemesh, quadmesh
from nekmeshpy.core.fields import geometric_spacing
from nekmeshpy.io.writer import quad_to_vtu
from nekmeshpy.quadmesh.smoother import constrain as cn
from nekmeshpy.quadmesh.smoother import frames, smooth

# -- parameters (circular_pipe.py's) -------------------------------------------
RADIUS = 0.5
N_SIDE = 6                   # central square block cells per side
N_RADIAL = 8                 # O-ring layers out to the wall
CENTER_SCALE = 0.7
RADIAL_GRADING = 0.5         # <1 clusters cells toward the wall
# -- this study's ---------------------------------------------------------------
KEEP_RINGS = 3               # rings off the wall that are protected (the three thinnest layers)
NOISE = 0.2                  # stress: interior perturbation, fraction of the local edge
SEED = 7
SCHEDULE = smooth.user_schedule(outer=5)

logging.getLogger().setLevel(logging.ERROR)
t_start = time.perf_counter()
wall_line = linemesh.circle(RADIUS, 4 * N_SIDE, element_tag="wall")
start = quadmesh.ogrid(wall_line, N_SIDE, geometric_spacing(N_RADIAL, RADIAL_GRADING),
                       center_scale=CENTER_SCALE, quadrant_scale=CENTER_SCALE)
P0 = np.array(start.points, dtype=float)
nrm = smooth.reference_normals(start.points, start.corners)
wall = np.unique(start.line_mesh.lines[start.edge_tags.flatnonzero("wall")])
circle = cn.circle([0, 0, 0], RADIUS)
res = cn.resolve(start, [cn.OnCurve(wall, circle), cn.KeepLayer(wall, rings=KEEP_RINGS)])
layer = np.flatnonzero(res.kind == cn.FROZEN)                                   # wall + rings 1..K
free = np.flatnonzero(res.kind == cn.FREE)
first_ring = np.setdiff1d(cn.layer_rings(start, wall, 1), wall)                 # ring 1: the thinnest layer's far side


def first_layer(P):
    """Radial distance wall -> ring 1, per wall vertex's nearest (by angle) ring-1 vertex."""
    a_w = np.arctan2(P[wall, 1], P[wall, 0])
    a_r = np.arctan2(P[first_ring, 1], P[first_ring, 0])
    near = first_ring[np.argmin(np.abs(np.angle(np.exp(1j * (a_w[:, None] - a_r[None, :])))), axis=1)]
    return np.hypot(P[wall, 0], P[wall, 1]) - np.hypot(P[near, 0], P[near, 1])


print("circular_pipe section: %d quads, %d vertices; wall of %d vertices on the circle r = %.2f; "
      "%d vertices (wall + rings 1..%d) kept as meshed; %d free"
      % (start.n_quads, start.n_points, wall.size, RADIUS, layer.size, KEEP_RINGS, free.size))
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
    ell = cn.vertex_length(m.points, m.corners)
    P[free, :2] += rng.uniform(-NOISE, NOISE, (free.size, 2)) * ell[free, None]
    return smooth.with_points(m, P)


runs = {}
for label, m0 in (("asis", start), ("stress", perturbed(start, SEED))):
    t0 = time.perf_counter()
    print("\n%s: %s" % (label, "the section as built" if label == "asis" else
                          "the free vertices perturbed by up to %.2f of their local edge" % NOISE))
    before = reading("input", m0)
    out, logs = smooth.run_schedule(m0, res, SCHEDULE, normals=nrm, record=True)
    after = reading("smoothed", out)
    on = cn.on_target_report(out.points, out.corners, res)
    ell = cn.vertex_length(m0.points, m0.corners)
    moved = np.linalg.norm(out.points - m0.points, axis=1) / ell
    radial_err = np.abs(np.hypot(out.points[wall, 0], out.points[wall, 1]) - RADIUS).max()
    print("  moved: max %.3f local edges (mean %.4f); wall off the circle %.1e"
          % (moved.max(), moved.mean(), radial_err))
    print("  layer protected: rings 1..%d moved max %.1e; first-layer thickness min/max %.4f/%.4f -> %.4f/%.4f"
          % (KEEP_RINGS, np.abs(out.points[layer] - m0.points[layer]).max(), first_layer(m0.points).min(),
             first_layer(m0.points).max(), first_layer(out.points).min(), first_layer(out.points).max()))
    print("  on target: %s" % str(on))
    if label == "asis":
        print("  verdict: the free rings were redistributed (the shape metric wants squares, the example graded them radially): "
              "scaled J min %.4f -> %.4f, mean %.4f -> %.4f, but largest neighbour size ratio %.3f -> %.3f and "
              "%.3f local edges moved on average" % (before[0].sj_min, after[0].sj_min, before[0].sj_mean,
                                                    after[0].sj_mean, before[1].size_ratio_max, after[1].size_ratio_max,
                                                    moved.mean()))
    if after[2] > 0:
        print("  NOT CLEAN: %d quads still overlap after smoothing (%d before)" % (after[2], before[2]))
    runs[label] = dict(before=m0, out=out, logs=logs, q=(before[0], after[0]), ov=(before[2], after[2]), on=on, moved=moved,
                       size_ratio=(before[1].size_ratio_max, after[1].size_ratio_max))
    quad_to_vtu(m0, "cs_pipe_%s_before.vtu" % label, cell_data={"sj_min": per_quad_sj(m0)})
    quad_to_vtu(out, "cs_pipe_%s_after.vtu" % label, point_data={"displacement": out.points - m0.points},
                cell_data={"sj_min": per_quad_sj(out)})
    frames.write_frames("cs_pipe_%s_frames.json" % label, m0, logs, nrm, decimals=5,
                        title="pipe section, wall sliding, layers kept (%s)" % label)
    print("  (%.1f s)" % (time.perf_counter() - t0))
mesh = runs["asis"]["out"]
print("\ntotal %.1f s" % (time.perf_counter() - t_start))
