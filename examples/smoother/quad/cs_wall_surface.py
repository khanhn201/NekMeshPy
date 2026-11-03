"""Case study: the wall of the carotid hex mesh, smoothed as a surface (examples/carotid.py).

``examples/carotid.py`` builds a 7200-hex vessel.  Its ``wall`` faces are extracted with
``hexmesh.boundary_mesh(mesh, "wall")`` (1440 curved quads carrying the hexes' own nodes) and taken to the
straight-sided shadow (``highorder.linearize``) the smoother works on.

**THERE IS NO PROPAGATION INTO THE HEXES.**  The smoothed wall is a free-standing surface mesh: the hexes are not
touched, and their interior nodes do not follow the moved wall vertices (the volume route -- carrying a wall
move into the hexes -- is not built).  Do not read these numbers as the quality of the carotid *volume* mesh.
What this shows is the surface smoother on a real, strongly curved, open wall.

**Normals and surface: from a tessellation, not from the quads.**  The wall is a CAD vessel known only as the
triangle mesh ``examples/data/car.vtx`` / ``car.tri`` (the file the example itself projects onto).  So every
non-rim vertex is ``OnSurface(constrain.surface_from_trimesh(...))`` -- the closest point on that tessellation --
and the metric's normals are the tessellation's own, blended continuous across triangles
(``constrain.surface_normals``), not the quads' winding.  Why: the winding normal of a quad is exactly what a
folded quad corrupts; the surface's is independent of the mesh.  Caveat of the choice: the tessellation is itself
an approximation (chord error remains, sharp edges are smoothed over by the blended normal).  The open rim of the
wall (inlet / outlets / flux planes, where the wall meets the other faces) is ``Frozen``.

Two runs of the same schedule: (1) the wall as built and (2) with every movable vertex perturbed
tangentially by ``NOISE`` of its local edge (and snapped back onto the tessellation).  Inverted and overlapping
quads, scaled Jacobian min/mean/worst 5 %, the size jump between neighbours, how far the as-built vertices sit
from the tessellation, the on-target distance and what moved are printed.

    PYTHONPATH=. python examples/smoother/quad/cs_wall_surface.py

Produces ``cs_wall_<asis|stress>_{before,after}.vtu`` (``sj_min`` per cell, ``displacement`` per point).  No
playback JSON (curved surface).  ``mesh`` is the smoothed as-built wall (a ``QuadMesh``, not a ``HexMesh``).
"""

import logging
import os
import runpy
import time

import numpy as np

from nekmeshpy import QuadMesh, TriMesh, hexmesh, quadmesh
from nekmeshpy.io.writer import quad_to_vtu
from nekmeshpy.quadmesh.smoother import constrain as cn
from nekmeshpy.quadmesh.smoother import highorder, smooth

# -- parameters --------------------------------------------------------------
NOISE = 0.2                  # stress: tangential perturbation, fraction of the local edge
SEED = 13
SCHEDULE = smooth.user_schedule(outer=2, n_opt=10)

t_start = time.perf_counter()
examples = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir)
ns = runpy.run_path(os.path.join(examples, "carotid.py"), run_name="__main__")      # the hex mesh (builds in ~4 s)
logging.getLogger().setLevel(logging.ERROR)
hexes = ns["mesh"]
wall = hexmesh.boundary_mesh(hexes, "wall")
start = highorder.linearize(wall)
tess = TriMesh.from_files(ns["VTX"], ns["TRI"])                                       # the example's own tessellation
surface = cn.surface_from_trimesh(tess.points, tess.tris, name="carotid tessellation")
rim = quadmesh.boundary_points(start)
inside = np.setdiff1d(np.arange(start.n_points), rim)
res = cn.resolve(start, [cn.Frozen(rim), cn.OnSurface(inside, surface)])
print("carotid wall (surface of a %d-hex mesh): %d quads, %d vertices; %d on the tessellation (%d triangles), %d on the "
      "open rim (frozen)" % (hexes.n_hexes, start.n_quads, start.n_points, res.counts()["on_surface"], tess.n_tris,
                             res.counts()["frozen"]))
print("  ** NO PROPAGATION INTO THE HEXES: the %d hexes are not touched; only this free-standing surface is smoothed **"
      % hexes.n_hexes)
for note in res.notes:
    print("  note:", note)
d_in = np.linalg.norm(surface.project(start.points[inside]) - start.points[inside], axis=1) / \
    cn.vertex_length(start.points, start.corners)[inside]
print("  as-built vertices sit %.3f local edges (max, mean %.4f) off the tessellation; the first snap pulls them on"
      % (d_in.max(), d_in.mean()))


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
movable = np.flatnonzero(res.kind != cn.FROZEN)
rng = np.random.default_rng(SEED)
P = np.array(start.points, dtype=float)
P[movable] += rng.uniform(-NOISE, NOISE, (movable.size, 3)) * cn.vertex_length(start.points, start.corners)[movable, None]
stressed = smooth.with_points(start, cn.snap(P, res))           # back onto the tessellation: tangential perturbation
for label, m0 in (("asis", start), ("stress", stressed)):
    t0 = time.perf_counter()
    nrm, nrep = cn.surface_normals(m0.points, m0.corners, cn.claimed_surfaces(res))
    print("\n%s: %s (%d quads disagree with the tessellation's normal, min cos %.3f)"
          % (label, "as built" if label == "asis" else "perturbed by up to %.2f of the local edge" % NOISE,
             nrep.n_disagree, nrep.min_cos))
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
        print("  verdict: scaled J min %.4f -> %.4f, mean %.4f -> %.4f, largest move %.3f local edges (hexes NOT updated)"
              % (before[0].sj_min, after[0].sj_min, before[0].sj_mean, after[0].sj_mean, moved.max()))
    if after[2] > 0:
        print("  NOT CLEAN: %d quads still overlap after smoothing (%d before)" % (after[2], before[2]))
    runs[label] = dict(before=m0, out=out, logs=logs, q=(before[0], after[0]), ov=(before[2], after[2]), on=on,
                       moved=moved, size_ratio=(before[1].size_ratio_max, after[1].size_ratio_max))
    quad_to_vtu(m0, "cs_wall_%s_before.vtu" % label, cell_data={"sj_min": per_quad_sj(m0, nrm)})
    quad_to_vtu(out, "cs_wall_%s_after.vtu" % label, point_data={"displacement": out.points - m0.points},
                cell_data={"sj_min": per_quad_sj(out, nrm)})
    print("  (%.1f s)" % (time.perf_counter() - t0))
assert isinstance(runs["asis"]["out"], QuadMesh)
mesh = runs["asis"]["out"]
print("\ntotal %.1f s" % (time.perf_counter() - t_start))
