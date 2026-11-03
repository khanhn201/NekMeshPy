"""Case study: the helical wire surface of the wire-coil hex mesh, smoothed as a surface (examples/wire_coil.py).

``examples/wire_coil.py`` builds a conjugate hex mesh of a wire wound helically in a pipe.  The wire's own surface
is the face tag ``coil`` (1344 quads).  It is extracted with ``hexmesh.boundary_mesh(mesh, "coil")`` and taken to
the straight-sided shadow (``highorder.linearize``) the smoother works on.  ``coil`` is an interior interface (fluid
on one side, wire on the other), so ``boundary_mesh`` returns every quad twice, once per side; the copies are
dropped (same four corners) before smoothing, otherwise every quad would read as overlapping its twin.

**THERE IS NO PROPAGATION INTO THE HEXES.**  The hexes are not touched; the smoothed wire surface is free-standing.

**The target surface is circular.**  The wire has no CAD description, so the surface the vertices are pulled onto is
a tessellation built from the mesh's own curved quads (each order-2 quad read on a finer lattice and split into
triangles).  It is the mesh's own geometry, not an independent reference: the smoother is held to the surface it
started on, and its normals are that surface's.  The open rim of the wire surface (where it meets the pipe wall and
the end planes) is ``Frozen``; every other vertex is ``OnSurface``.

**Why vertices move a lot.**  The helical template skews its quads by design, so the smoother moves vertices by
several local edges.  The minimum scaled Jacobian barely changes; the mean and the worst 5 % improve.

Two runs of the same schedule: (1) as built and (2) with every movable vertex perturbed tangentially by
``NOISE`` of its local edge (and snapped back onto the target).  Printed: inverted and overlapping quads, scaled
Jacobian min/mean/worst 5 %, the size jump between neighbours, how far vertices moved, the on-target check.

    PYTHONPATH=. python examples/smoother/quad/cs_wire_coil.py

Produces ``cs_coil_<asis|stress>_{before,after}.vtu`` (``sj_min`` per cell, ``displacement`` per point).  ``mesh`` is
the smoothed as-built surface (a ``QuadMesh``).
"""

import logging
import os
import runpy
import time

import numpy as np

from nekmeshpy import QuadMesh, hexmesh, quadmesh
from nekmeshpy.core import conform, interp
from nekmeshpy.io.writer import quad_to_vtu
from nekmeshpy.quadmesh.smoother import constrain as cn
from nekmeshpy.quadmesh.smoother import highorder, smooth

# -- parameters --------------------------------------------------------------
TAG = "coil"                 # face tag of the wire surface
NOISE = 0.2                  # stress: tangential perturbation, fraction of the local edge
SEED = 13
TESS_ORDER = 6               # the curved quads are read on this lattice to build the target tessellation
OUTER = int(os.environ.get("NEKMESHPY_SMOOTH_OUTER", 5))      # outer cycles; the test suite sets 1 to stay fast
SCHEDULE = smooth.user_schedule(outer=OUTER, n_opt=20)

t_start = time.perf_counter()
examples = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir)
ns = runpy.run_path(os.path.join(examples, "wire_coil.py"), run_name="__main__")      # the hex mesh
logging.getLogger().setLevel(logging.ERROR)
hexes = ns["mesh"]
both = hexmesh.boundary_mesh(hexes, TAG)
_, first = np.unique(np.sort(both.corners, axis=1), axis=0, return_index=True)       # one quad per physical face
wall = quadmesh.select(both, np.sort(first))
start = highorder.linearize(wall)

# the target: a triangulation of the mesh's own curved quads
nodes, conn = conform.conformal_quad(wall.points, wall.corners, wall.quads, wall.orient, wall.line_mesh.interior,
                                     wall.interior, wall.order)
fine = interp.resample_block(nodes[conn], wall.order, TESS_ORDER, 2)                  # (Q, (n+1)^2, 3)
m = TESS_ORDER + 1
idx = np.arange(m * m).reshape(m, m)
a, b, c, d = idx[:-1, :-1].ravel(), idx[1:, :-1].ravel(), idx[1:, 1:].ravel(), idx[:-1, 1:].ravel()
local = np.r_[np.c_[a, b, c], np.c_[a, c, d]]
tess_tris = (np.arange(fine.shape[0])[:, None, None] * m * m + local[None]).reshape(-1, 3)
surface = cn.surface_from_trimesh(fine.reshape(-1, 3), tess_tris, name="wire surface, own curved quads")

rim = quadmesh.boundary_points(start)
inside = np.setdiff1d(np.arange(start.n_points), rim)
res = cn.resolve(start, [cn.Frozen(rim), cn.OnSurface(inside, surface)])
print("wire-coil wire surface (tag %r of a %d-hex mesh): %d quads (%d before dropping the twin side), %d vertices; "
      "%d on the target (%d triangles), %d on the open rim (frozen)"
      % (TAG, hexes.n_hexes, start.n_quads, both.n_quads, start.n_points, res.counts()["on_surface"],
         tess_tris.shape[0], res.counts()["frozen"]))
print("  ** NO PROPAGATION INTO THE HEXES: the %d hexes are not touched; only this free-standing surface is smoothed **"
      % hexes.n_hexes)
print("  ** the target is a tessellation of the mesh's own curved quads: circular, not an independent reference **")
for note in res.notes:
    print("  note:", note)
d_in = np.linalg.norm(surface.project(start.points[inside]) - start.points[inside], axis=1) / \
    cn.vertex_length(start.points, start.corners)[inside]
print("  as-built vertices sit %.3f local edges (max, mean %.4f) off the target; the first snap pulls them on"
      % (d_in.max(), d_in.mean()))


def per_quad_sj(mesh_, nrm):
    return smooth.corner_metrics(mesh_.points, mesh_.corners, nrm).sj.min(axis=1)


def reading(label, mesh_, nrm):
    q = smooth.quality(mesh_.points, mesh_.corners, nrm)
    s = smooth.shape_stats(mesh_.points, mesh_.corners)
    ov = int(quadmesh.overlapping_quads(mesh_).size)
    print("  %-8s inverted %d | overlapping %d | scaled J min %.4f mean %.4f worst 5%% %.4f | size ratio max %.3f "
          "mean %.3f" % (label, q.n_inverted, ov, q.sj_min, q.sj_mean, q.sj_q05, s.size_ratio_max, s.size_ratio_mean))
    return q, s, ov


runs = {}
movable = np.flatnonzero(res.kind != cn.FROZEN)
rng = np.random.default_rng(SEED)
P = np.array(start.points, dtype=float)
P[movable] += rng.uniform(-NOISE, NOISE, (movable.size, 3)) * cn.vertex_length(start.points, start.corners)[movable, None]
stressed = smooth.with_points(start, cn.snap(P, res))           # back onto the target: tangential perturbation
for label, m0 in (("asis", start), ("stress", stressed)):
    t0 = time.perf_counter()
    nrm, nrep = cn.surface_normals(m0.points, m0.corners, cn.claimed_surfaces(res))
    print("\n%s: %s (%d quads disagree with the target's normal, min cos %.3f)"
          % (label, "as built" if label == "asis" else "perturbed by up to %.2f of the local edge" % NOISE,
             nrep.n_disagree, nrep.min_cos))
    before = reading("input", m0, nrm)
    out, logs = smooth.run_schedule(m0, res, SCHEDULE, normals=nrm)
    after = reading("smoothed", out, nrm)
    on = cn.on_target_report(out.points, out.corners, res)
    ell = cn.vertex_length(m0.points, m0.corners)
    moved = np.linalg.norm(out.points - m0.points, axis=1) / ell
    print("  moved: mean %.2f local edges, max %.2f; rim moved %.1e" % (moved[movable].mean(), moved.max(),
                                                                   np.abs(out.points[rim] - m0.points[rim]).max()))
    print("  on target: %s" % str(on))
    if after[2] > 0:
        print("  NOT CLEAN: %d quads still overlap after smoothing (%d before)" % (after[2], before[2]))
    runs[label] = dict(before=m0, out=out, logs=logs, q=(before[0], after[0]), ov=(before[2], after[2]), on=on,
                       moved=moved[movable], size_ratio=(before[1].size_ratio_max, after[1].size_ratio_max))
    quad_to_vtu(m0, "cs_coil_%s_before.vtu" % label, cell_data={"sj_min": per_quad_sj(m0, nrm)})
    quad_to_vtu(out, "cs_coil_%s_after.vtu" % label, point_data={"displacement": out.points - m0.points},
                cell_data={"sj_min": per_quad_sj(out, nrm)})
    print("  (%.1f s)" % (time.perf_counter() - t0))
assert isinstance(runs["asis"]["out"], QuadMesh)
mesh = runs["asis"]["out"]
print("\ntotal %.1f s" % (time.perf_counter() - t_start))
