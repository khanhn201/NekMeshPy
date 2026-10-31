"""Case study: the two closed surfaces of the flow-past-sphere shell (examples/flow_past_sphere.py).

The example fills a hex shell between an inner sphere (``quadmesh.sphere``) and a far-field cube
(``quadmesh.box``).  Those two **closed quad surfaces** are rebuilt here with the example's own calls and
parameters (``R``, ``S``, ``N_FACE``) at order 1, and smoothed as surfaces:

* the sphere: every vertex ``OnSurface`` the exact sphere (sliding, snapped back every step);
* the cube: face vertices ``OnSurface`` on the six planes, the 12 creases ``OnCurve`` on their own lines (they
  slide along the crease and never leave it), the 8 corners ``Frozen`` -- as in ``03_constraints.py (b)``.

The metric's normals are the **surfaces' own** (``constrain.surface_normals``), not the quads' winding, so a
folded input quad cannot poison them.  There is **no propagation into the hex shell**: the shell is not touched
(the volume route is not built); these are the two surfaces its inner and outer faces would come from.

Two runs of the same schedule per surface: (1) as built (a cubed-sphere is already a good mesh, so little
changes -- the numbers say how little) and (2) with every movable vertex perturbed tangentially by ``NOISE``
of its local edge.  Inverted and overlapping quads, scaled Jacobian min/mean/worst 5 %, the size jump between
neighbours, the on-target distances (and that the corners and creases hold) and what moved are printed.

    PYTHONPATH=. python examples/smoother/cs_sphere_shell.py

Produces ``cs_shell_<sphere|cube>_<asis|stress>_{before,after}.vtu`` (``sj_min`` per cell, ``displacement`` per
point).  No playback JSON (curved surfaces: the docs viewer for them is separate).  ``mesh`` is the smoothed
as-built sphere.
"""

import logging
import time

import numpy as np

from nekmeshpy import quadmesh
from nekmeshpy.io.writer import quad_to_vtu
from nekmeshpy.quadmesh.smoother import constrain as cn
from nekmeshpy.quadmesh.smoother import smooth

# -- parameters (flow_past_sphere.py's) -----------------------------------------
R = 0.5                      # sphere radius
S = 4.0                      # far-field cube half-side
N_FACE = 12                  # cells per direction on each cube face
# -- this study's ---------------------------------------------------------------
NOISE = 0.2                  # stress: tangential perturbation, fraction of the local edge
SEED = 11
SCHEDULE = smooth.user_schedule(outer=3, n_opt=10)

logging.getLogger().setLevel(logging.ERROR)
t_start = time.perf_counter()


def constrain_sphere(m):
    return [cn.OnSurface(np.arange(m.n_points), cn.sphere([0, 0, 0], R))]


def constrain_cube(m):                                    # planes, creases, corners (as 03_constraints.py (b))
    B = np.asarray(m.points)
    on_face = np.isclose(np.abs(B), S)
    n_face = on_face.sum(axis=1)
    corner_ids, crease_ids, face_ids = (np.flatnonzero(n_face == k) for k in (3, 2, 1))
    cons = [cn.Frozen(corner_ids)]
    for axis in range(3):
        for sign in (-1.0, 1.0):
            ids = face_ids[on_face[face_ids, axis] & (np.sign(B[face_ids, axis]) == sign)]
            cons.append(cn.OnSurface(ids, cn.plane(sign * S * np.eye(3)[axis], np.eye(3)[axis])))
    creases = {}
    for v in crease_ids:
        a1, a2 = np.flatnonzero(on_face[v])
        creases.setdefault((a1, a2, np.sign(B[v, a1]), np.sign(B[v, a2])), []).append(v)
    for (a1, a2, s1, s2), ids in creases.items():
        origin = np.zeros(3)
        origin[a1], origin[a2] = s1 * S, s2 * S
        cons.append(cn.OnCurve(np.array(ids), cn.line(origin, np.eye(3)[3 - a1 - a2])))
    return cons


SURFACES = {"sphere": (quadmesh.sphere(R, N_FACE, element_tag="sphere"), constrain_sphere),
            "cube": (quadmesh.box(S, N_FACE), constrain_cube)}


def reading(label, m, nrm):
    q = smooth.quality(m.points, m.corners, nrm)
    s = smooth.shape_stats(m.points, m.corners)
    ov = int(quadmesh.overlapping_quads(m).size)
    print("  %-8s inverted %d | overlapping %d | scaled J min %.4f mean %.4f worst 5%% %.4f | size ratio max %.3f "
          "mean %.3f" % (label, q.n_inverted, ov, q.sj_min, q.sj_mean, q.sj_q05, s.size_ratio_max, s.size_ratio_mean))
    return q, s, ov


def per_quad_sj(m, nrm):
    return smooth.corner_metrics(m.points, m.corners, nrm).sj.min(axis=1)


runs = {}
for name, (start, build) in SURFACES.items():
    res = cn.resolve(start, build(start), boundary="free")      # closed: there is no boundary to freeze
    counts = res.counts()
    print("\n%s: %d quads, %d vertices; %d on a surface, %d on a curve, %d frozen"
          % (name, start.n_quads, start.n_points, counts["on_surface"], counts["on_curve"], counts["frozen"]))
    for note in res.notes:
        print("  note:", note)
    movable = np.flatnonzero((res.kind != cn.FROZEN))
    rng = np.random.default_rng(SEED)
    P = np.array(start.points, dtype=float)
    P[movable] += rng.uniform(-NOISE, NOISE, (movable.size, 3)) * cn.vertex_length(start.points, start.corners)[movable, None]
    stressed = smooth.with_points(start, cn.snap(P, res))     # onto the surface / its crease line: tangential perturbation only
    for label, m0 in (("asis", start), ("stress", stressed)):
        t0 = time.perf_counter()
        nrm, nrep = cn.surface_normals(m0.points, m0.corners, cn.claimed_surfaces(res))
        print(" %s, %s: %s (%d quads disagree with the surface normal)"
              % (name, label, "as built" if label == "asis" else "perturbed by up to %.2f of the local edge" % NOISE,
                 nrep.n_disagree))
        before = reading("input", m0, nrm)
        out, logs = smooth.run_schedule(m0, res, SCHEDULE, normals=nrm)
        after = reading("smoothed", out, nrm)
        on = cn.on_target_report(out.points, out.corners, res)
        ell = cn.vertex_length(m0.points, m0.corners)
        moved = np.linalg.norm(out.points - m0.points, axis=1) / ell
        print("  moved: max %.3f local edges (mean %.4f); frozen moved %.1e" % (moved.max(), moved.mean(), on.max_frozen_move))
        print("  on target: %s" % str(on))
        if label == "asis":
            print("  verdict: scaled J min %.4f -> %.4f, mean %.4f -> %.4f, largest move %.3f local edges" %
                  (before[0].sj_min, after[0].sj_min, before[0].sj_mean, after[0].sj_mean, moved.max()))
        if after[2] > 0:
            print("  NOT CLEAN: %d quads still overlap after smoothing (%d before)" % (after[2], before[2]))
        runs[name, label] = dict(before=m0, out=out, logs=logs, q=(before[0], after[0]), ov=(before[2], after[2]), on=on,
                                 moved=moved, size_ratio=(before[1].size_ratio_max, after[1].size_ratio_max), res=res)
        quad_to_vtu(m0, "cs_shell_%s_%s_before.vtu" % (name, label), cell_data={"sj_min": per_quad_sj(m0, nrm)})
        quad_to_vtu(out, "cs_shell_%s_%s_after.vtu" % (name, label), point_data={"displacement": out.points - m0.points},
                    cell_data={"sj_min": per_quad_sj(out, nrm)})
        print("  (%.1f s)" % (time.perf_counter() - t0))
mesh = runs["sphere", "asis"]["out"]
print("\ntotal %.1f s" % (time.perf_counter() - t_start))
