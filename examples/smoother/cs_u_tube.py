"""Case study: the U-tube wall of the u_tube_box hex mesh, smoothed as a surface (examples/u_tube_box.py).

``examples/u_tube_box.py`` builds a conjugate hex mesh of a U-shaped tube cut out of a box.  The tube wall is the
face tag ``interface`` (2040 quads).  It is extracted with ``hexmesh.boundary_mesh(mesh, "interface")`` and taken to
the straight-sided shadow (``highorder.linearize``) the smoother works on.  ``interface`` is an interior face (fluid
on one side, solid on the other), so ``boundary_mesh`` returns every quad twice, once per side; the copies are
dropped (same four corners) before smoothing, otherwise every quad would read as overlapping its twin.

**THERE IS NO PROPAGATION INTO THE HEXES.**  The hexes are not touched; the smoothed wall is free-standing.  Carrying
a smoothed wall into the hexes would damage them: the inner stair-step side of the wall layer is unchanged, so the
layers between the two sheared when the wall was moved, which inverted hexes in a trial.

**The target surface is analytic.**  The tube is exactly a radius ``R_TUBE`` about the U's centreline, and the
example's own ``project_to_wall`` is the closest-point map onto it; the normal is the radial direction from
``axis_point``.  Both come from the example, nothing is tessellated.  The open rim of the wall (the tube's two
ends) is ``Frozen``; every other vertex is ``OnSurface``.

**The wall is already good**, so the gain is modest: the smoother has little to repair.

**The outer box** (face tag ``outer``, 5000 quads) is smoothed the same way, as a second surface: its six faces are
planes (``OnSurface(plane)``), its twelve creases lines (``OnCurve(line)``), its eight corners and the rim of the
tube's two openings in ``x = 0`` are ``Frozen``.  As built it is planar and almost all squares (mean scaled J
0.998), so there is little to gain; the perturbed-input run shows the smoother recovering.

Two runs of the same schedule per surface: (1) as built and (2) with every movable vertex perturbed tangentially by
``NOISE`` of its local edge (and snapped back onto its surface).  Printed: inverted and overlapping quads, scaled
Jacobian min/mean/worst 5 %, the size jump between neighbours, how far vertices moved, the on-target check.  The
box smoothing roughly doubles the run time; ``NEKMESHPY_SMOOTH_OUTER`` (outer cycles, default 5) shortens it.

    PYTHONPATH=. python examples/smoother/cs_u_tube.py

Produces ``cs_utube_<asis|stress>_{before,after}.vtu`` and ``cs_utube_box_<asis|stress>_{before,after}.vtu`` (``sj_min``
per cell, ``displacement`` per point).  ``mesh`` is the smoothed as-built wall (a ``QuadMesh``); ``runs`` are its runs
and ``box_runs`` the box's.
"""

import logging
import os
import runpy
import time

import numpy as np

from nekmeshpy import QuadMesh, hexmesh, quadmesh
from nekmeshpy.io.writer import quad_to_vtu
from nekmeshpy.quadmesh.smoother import constrain as cn
from nekmeshpy.quadmesh.smoother import highorder, smooth

# -- parameters --------------------------------------------------------------
TAG = "interface"            # face tag of the tube wall
OUTER = "outer"              # face tag of the box walls
NOISE = 0.2                  # stress: tangential perturbation, fraction of the local edge
SEED = 13
N_OUTER = int(os.environ.get("NEKMESHPY_SMOOTH_OUTER", 5))      # outer cycles; the test suite sets 1 to stay fast
SCHEDULE = smooth.user_schedule(outer=N_OUTER, n_opt=20)

t_start = time.perf_counter()
examples = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir)
ns = runpy.run_path(os.path.join(examples, "u_tube_box.py"), run_name="__main__")      # the hex mesh
logging.getLogger().setLevel(logging.ERROR)
hexes = ns["mesh"]
both = hexmesh.boundary_mesh(hexes, TAG)
_, first = np.unique(np.sort(both.corners, axis=1), axis=0, return_index=True)       # one quad per physical face
wall = quadmesh.select(both, np.sort(first))
start = highorder.linearize(wall)

# the target: the example's own analytic tube (closest point on the wall, radial normal)
axis_point, R_TUBE = ns["axis_point"], ns["R_TUBE"]


def radial(p):
    v = p - axis_point(p)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


surface = cn.Surface(ns["project_to_wall"], radial, "U-tube wall, radius %g about the centreline" % R_TUBE)

rim = quadmesh.boundary_points(start)
inside = np.setdiff1d(np.arange(start.n_points), rim)
res = cn.resolve(start, [cn.Frozen(rim), cn.OnSurface(inside, surface)])
print("U-tube wall (tag %r of a %d-hex mesh): %d quads (%d before dropping the twin side), %d vertices; "
      "%d on the analytic tube, %d on the open rim (frozen)"
      % (TAG, hexes.n_hexes, start.n_quads, both.n_quads, start.n_points, res.counts()["on_surface"],
         res.counts()["frozen"]))
print("  ** NO PROPAGATION INTO THE HEXES: the %d hexes are not touched; only this free-standing surface is smoothed **"
      % hexes.n_hexes)
for note in res.notes:
    print("  note:", note)
d_in = np.linalg.norm(surface.project(start.points[inside]) - start.points[inside], axis=1) / \
    cn.vertex_length(start.points, start.corners)[inside]
print("  as-built vertices sit %.3f local edges (max, mean %.4f) off the tube; the first snap pulls them on"
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


def run_pair(prefix, what, start, res, rim):
    """As built and perturbed, the same schedule: the printed readings, the ``.vtu`` files, one dict per run."""
    out_runs = {}
    movable = np.flatnonzero(res.kind != cn.FROZEN)
    rng = np.random.default_rng(SEED)
    P = np.array(start.points, dtype=float)
    P[movable] += rng.uniform(-NOISE, NOISE, (movable.size, 3)) * cn.vertex_length(start.points, start.corners)[movable, None]
    stressed = smooth.with_points(start, cn.snap(P, res))           # back onto its surface: tangential perturbation
    for label, m0 in (("asis", start), ("stress", stressed)):
        t0 = time.perf_counter()
        nrm, nrep = cn.surface_normals(m0.points, m0.corners, cn.claimed_surfaces(res))
        print("\n%s, %s: %s (%d quads disagree with the surface's normal, min cos %.3f)"
              % (what, label, "as built" if label == "asis" else "perturbed by up to %.2f of the local edge" % NOISE,
                 nrep.n_disagree, nrep.min_cos))
        before = reading("input", m0, nrm)
        out, logs = smooth.run_schedule(m0, res, SCHEDULE, normals=nrm)
        after = reading("smoothed", out, nrm)
        on = cn.on_target_report(out.points, out.corners, res)
        ell = cn.vertex_length(m0.points, m0.corners)
        moved = np.linalg.norm(out.points - m0.points, axis=1) / ell
        print("  moved: mean %.2f local edges, max %.2f; rim moved %.1e"
              % (moved[movable].mean(), moved.max(), np.abs(out.points[rim] - m0.points[rim]).max()))
        print("  on target: %s" % str(on))
        if after[2] > 0:
            print("  NOT CLEAN: %d quads still overlap after smoothing (%d before)" % (after[2], before[2]))
        out_runs[label] = dict(before=m0, out=out, logs=logs, q=(before[0], after[0]), ov=(before[2], after[2]), on=on,
                               moved=moved[movable], size_ratio=(before[1].size_ratio_max, after[1].size_ratio_max))
        quad_to_vtu(m0, "%s_%s_before.vtu" % (prefix, label), cell_data={"sj_min": per_quad_sj(m0, nrm)})
        quad_to_vtu(out, "%s_%s_after.vtu" % (prefix, label), point_data={"displacement": out.points - m0.points},
                    cell_data={"sj_min": per_quad_sj(out, nrm)})
        print("  (%.1f s)" % (time.perf_counter() - t0))
    return out_runs


runs = run_pair("cs_utube", "interface", start, res, rim)

# -- the outer box: planes, creases, corners -----------------------------------
# The outer faces (tag OUTER) are the six planes of the box, 5000 quads, closed except for the tube's two openings
# in x = 0 (the rim of each hole is the tube wall's rim, so it is held).  ``outer`` is a domain boundary, one quad per
# face, so there is no twin side to drop.  A vertex on one plane is OnSurface, on two a crease (OnCurve, its line), on
# three a corner (Frozen).  As with the 03_constraints box, the as-built box is planar and nearly all squares.
box_start = highorder.linearize(hexmesh.boundary_mesh(hexes, OUTER))
BP = np.array(box_start.points, dtype=float)
lo, hi = BP.min(axis=0), BP.max(axis=0)                         # the box: x 0..BOX_X, y +-BOX_HALF_Y, z +-BOX_HALF_Z
on_plane = np.stack([np.isclose(BP, lo), np.isclose(BP, hi)], axis=-1)       # (P, 3, 2): on the low / high plane of an axis
n_plane = on_plane.any(axis=2).sum(axis=1)
box_rim = quadmesh.boundary_points(box_start)                   # the two openings
corner_ids = np.flatnonzero(n_plane == 3)
crease_ids = np.setdiff1d(np.flatnonzero(n_plane == 2), box_rim)
face_ids = np.setdiff1d(np.flatnonzero(n_plane == 1), box_rim)
box_constraints = [cn.Frozen(np.union1d(corner_ids, box_rim))]
for axis in range(3):
    for side, bound in enumerate((lo, hi)):                      # six planes
        ids = face_ids[on_plane[face_ids, axis, side]]
        normal = np.eye(3)[axis]
        box_constraints.append(cn.OnSurface(ids, cn.plane(bound[axis] * normal, normal)))
creases = {}
for v in crease_ids:                                              # twelve lines, keyed by the two planes
    a1, a2 = np.flatnonzero(on_plane[v].any(axis=1))
    creases.setdefault((a1, a2, int(on_plane[v, a1, 1]), int(on_plane[v, a2, 1])), []).append(v)
for (a1, a2, s1, s2), ids in creases.items():
    origin = np.zeros(3)
    origin[a1], origin[a2] = (lo, hi)[s1][a1], (lo, hi)[s2][a2]
    box_constraints.append(cn.OnCurve(np.array(ids), cn.line(origin, np.eye(3)[3 - a1 - a2])))
box_res = cn.resolve(box_start, box_constraints)
print("\nouter box (tag %r): %d quads, %d vertices; %d corners and %d opening-rim points frozen, %d crease points on %d "
      "lines, %d face points on 6 planes" % (OUTER, box_start.n_quads, box_start.n_points, corner_ids.size, box_rim.size,
                                              crease_ids.size, len(creases), face_ids.size))
box_runs = run_pair("cs_utube_box", "outer box", box_start, box_res, np.union1d(corner_ids, box_rim))

assert isinstance(runs["asis"]["out"], QuadMesh)
mesh = runs["asis"]["out"]
print("\ntotal %.1f s" % (time.perf_counter() - t_start))
