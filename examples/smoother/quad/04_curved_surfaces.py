"""Smoothing quads on curved surfaces: how well it works, and where it stops.

Three closed surfaces -- a sphere, an ellipsoid, a torus -- each meshed with quads, perturbed
tangentially, and smoothed with every vertex ``OnSurface`` (sliding, snapped back to the surface
each step) and the *surface's own normals* for the metric.  Each is judged against the exact surface,
not only against itself:

* **angle defect** (``quadmesh.angle_defect``, the discrete Gaussian curvature at a vertex) against the
  exact integral of ``K`` over the vertex's dual cell -- closed-form ``K`` for each surface is written
  below -- as rms and max error per vertex;
* **sagitta**: distance of between-vertex samples (3 x 3 per quad) to the surface, which vertices
  exactly on the surface cannot see;
* **on-target distance** (``on_target_report``) and **area drift**.

Then the limits, shown and not hidden:

(a) the error does not go to zero: angle defects sum to 2 pi chi whatever the mesh, so the curvature
    is carried by the few irregular vertices of the connectivity, and the *undisturbed* mesh of the
    same connectivity has an error of its own.  Smoothing gets close to that floor, not below it.
    Element size also does not follow curvature (the ellipsoid's tips stay as coarse as its sides).
(b) the normals.  The same ellipsoid with one *folded* input quad, smoothed twice -- once with the
    surface's normals, once with the normals read off the input quads' winding (what the metric used
    before the session took them from the surface).  The folded quad poisons the second.
    A folded input is NOT part of the main comparison above: the inputs there have no folded quad
    (checked: ``n_disagree`` is printed).  Some folded inputs leave a persistent overlap that the
    smoother cannot remove, with either normals; that is a known open issue under investigation, and
    (b) prints how many of its folds each choice repairs rather than showing only the best.

    PYTHONPATH=. python examples/smoother/quad/04_curved_surfaces.py

Produces ``04_<name>_{before,after}.vtu`` (``sj_min`` per cell, ``displacement`` and
``defect_error`` per point) and ``04_b_ellipsoid_fold<q>_{surface,winding}_normals.vtu``.  ``mesh`` is the
smoothed ellipsoid.  (No playback JSON: the docs viewer for curved surfaces is separate.)
"""

import logging
import time

import numpy as np

from nekmeshpy import quadmesh
from nekmeshpy.io.writer import quad_to_vtu
from nekmeshpy.quadmesh.smoother import constrain as cn
from nekmeshpy.quadmesh.smoother import smooth

# -- parameters --------------------------------------------------------------
SPHERE_R = 1.0
ELLIPSOID_AXES = (1.0, 1.5, 2.0)
TORUS_R, TORUS_r = 1.0, 0.35         # major, minor radius
CUBE_N = 5                           # cells per cube edge (sphere, ellipsoid): 6 * N^2 quads
TORUS_NU, TORUS_NV = 20, 10          # cells around / through the tube
NOISE = 0.25                         # tangential jitter, fraction of the local edge length
FOLD_QUADS = (52, 13, 104, 143)      # (b): quads whose first corner is thrown past the opposite one, one fold each
NEWTON_ITERS = 8                     # Newton steps of the implicit ellipsoid's projection
SEED = 1
SCHEDULE = smooth.user_schedule(outer=5)
logging.getLogger().setLevel(logging.ERROR)     # the session-level normal warnings are shown by (b), by number


# -- the surfaces: projection, normal, exact Gaussian curvature ----------------
def _unit(v):
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-300)


def ellipsoid(a, b, c):
    def f(X):
        return (X[:, 0] / a) ** 2 + (X[:, 1] / b) ** 2 + (X[:, 2] / c) ** 2 - 1.0

    def grad(X):
        return 2.0 * np.stack([X[:, 0] / a ** 2, X[:, 1] / b ** 2, X[:, 2] / c ** 2], 1)

    def project(P):                     # Newton along the gradient: lands on the surface
        X = np.array(P, float)
        for _ in range(NEWTON_ITERS):
            g = grad(X)
            X = X - (f(X) / np.maximum((g * g).sum(1), 1e-300))[:, None] * g
        return X

    def K(X):                           # Gaussian curvature of the ellipsoid at X on it
        s = (X[:, 0] / a ** 2) ** 2 + (X[:, 1] / b ** 2) ** 2 + (X[:, 2] / c ** 2) ** 2
        return 1.0 / ((a * b * c) ** 2 * s ** 2)

    return cn.Surface(project, lambda P: _unit(grad(P)), "ellipsoid"), K


def torus(R, r):
    def axis_point(P):
        rho = np.maximum(np.linalg.norm(P[:, :2], axis=1), 1e-300)
        return np.c_[R * P[:, 0] / rho, R * P[:, 1] / rho, np.zeros(len(P))]

    def K(P):                           # cos(v) / (r (R + r cos v)), written with rho = R + r cos v
        rho = np.linalg.norm(P[:, :2], axis=1)
        return (rho - R) / (r ** 2 * rho)

    return (cn.Surface(lambda P: axis_point(P) + r * _unit(P - axis_point(P)),
                       lambda P: _unit(P - axis_point(P)), "torus"), K)


# -- meshes: a cube projected on the surface, a structured torus ---------------
def oriented_outward(P, q, surface):
    n = np.cross(P[q[:, 1]] - P[q[:, 0]], P[q[:, 3]] - P[q[:, 0]])
    flip = np.einsum("ij,ij->i", n, surface.normal(P[q].mean(1))) < 0
    q = q.copy()
    q[flip] = q[flip][:, ::-1]
    return q


def cube_surface_mesh(surface, scale):
    cube = quadmesh.box(1.0, CUBE_N)                 # six patches welded: 8 valence-3 corners
    P = surface.project(np.asarray(cube.points) / np.linalg.norm(cube.points, axis=1, keepdims=True) * scale)
    return quadmesh.QuadMesh.from_corners(P, oriented_outward(P, cube.corners, surface))


def torus_mesh(surface):
    u = np.linspace(0, 2 * np.pi, TORUS_NU, endpoint=False)
    v = np.linspace(0, 2 * np.pi, TORUS_NV, endpoint=False)
    U, V = np.meshgrid(u, v, indexing="ij")
    P = np.stack([(TORUS_R + TORUS_r * np.cos(V)) * np.cos(U), (TORUS_R + TORUS_r * np.cos(V)) * np.sin(U),
                  TORUS_r * np.sin(V)], -1).reshape(-1, 3)
    i = np.arange(TORUS_NU * TORUS_NV).reshape(TORUS_NU, TORUS_NV)
    q = np.stack([i, np.roll(i, -1, 0), np.roll(np.roll(i, -1, 0), -1, 1), np.roll(i, -1, 1)], -1).reshape(-1, 4)
    return quadmesh.QuadMesh.from_corners(P, oriented_outward(P, q, surface))


def jittered(ref, surface, seed):
    rng = np.random.default_rng(seed)
    ell = cn.vertex_length(ref.points, ref.corners)[:, None]
    return smooth.with_points(ref, surface.project(ref.points + rng.normal(size=ref.points.shape) * NOISE * ell))


# -- the measurements ----------------------------------------------------------
def dual_cell_curvature(m, surface, K, n=5):
    """Exact integral of K over each vertex's dual cell (the union of the four ``[vertex, edge midpoint,
    quad centre, edge midpoint]`` sectors of its quads): every sector is cut n x n bilinearly, the
    sub-corners are carried onto the surface, each sub-cell contributes K at its foot point times its area."""
    P = np.asarray(m.points, float)
    q = m.corners
    Q = P[q]
    g = Q.mean(1)
    out = np.zeros(len(P))
    t = np.linspace(0.0, 1.0, n + 1)
    for k in range(4):
        c = [Q[:, k], (Q[:, k] + Q[:, (k + 1) % 4]) / 2, g, (Q[:, k] + Q[:, (k - 1) % 4]) / 2]
        grid = np.empty((len(q), n + 1, n + 1, 3))
        for i, a in enumerate(t):
            for j, b in enumerate(t):
                grid[:, i, j] = (1 - a) * (1 - b) * c[0] + a * (1 - b) * c[1] + a * b * c[2] + (1 - a) * b * c[3]
        S = surface.project(grid.reshape(-1, 3)).reshape(len(q), n + 1, n + 1, 3)
        p00, p10, p11, p01 = S[:, :-1, :-1], S[:, 1:, :-1], S[:, 1:, 1:], S[:, :-1, 1:]
        area = 0.5 * (np.linalg.norm(np.cross(p10 - p00, p11 - p00), axis=-1)
                      + np.linalg.norm(np.cross(p11 - p00, p01 - p00), axis=-1))
        foot = surface.project(((p00 + p10 + p11 + p01) / 4).reshape(-1, 3))
        out += np.bincount(q[:, k], weights=(area.reshape(len(q), -1) * K(foot).reshape(len(q), -1)).sum(1),
                           minlength=len(P))
    return out


def sagitta(m, surface, n=3):
    Q = np.asarray(m.points, float)[m.corners]
    t = (np.arange(n) + 0.5) / n
    X = np.concatenate([(1 - a) * (1 - b) * Q[:, 0] + a * (1 - b) * Q[:, 1] + a * b * Q[:, 2] + (1 - a) * b * Q[:, 3]
                        for a in t for b in t])
    return np.linalg.norm(surface.project(X) - X, axis=1)


def area(m):
    Q = np.asarray(m.points, float)[m.corners]
    return float(0.5 * np.linalg.norm(np.cross(Q[:, 2] - Q[:, 0], Q[:, 3] - Q[:, 1]), axis=1).sum())


def measure(m, surface, K):
    err = quadmesh.angle_defect(m) - dual_cell_curvature(m, surface, K)
    sg = sagitta(m, surface)
    return {"rms": float(np.sqrt((err ** 2).mean())), "max": float(np.abs(err).max()), "err": err,
            "sag_max": float(sg.max()), "area": area(m)}


def per_quad_sj(m, nrm):
    return smooth.corner_metrics(m.points, m.corners, nrm).sj.min(axis=1)


def smooth_on(m, surface, normals_from="surface"):
    """Smooth ``m`` with every vertex OnSurface; the metric's normals from the surface or from the winding."""
    r = cn.resolve(m, [cn.OnSurface(np.arange(m.n_points), surface)], boundary="free")
    nrm, rep = cn.surface_normals(m.points, m.corners, cn.claimed_surfaces(r))
    used = nrm if normals_from == "surface" else smooth.reference_normals(m.points, m.corners)
    out, logs = smooth.run_schedule(m, r, SCHEDULE, normals=used)
    return out, logs, r, nrm, rep


# ============================================================================
# the main comparison: three closed surfaces, inputs without a folded quad
# ============================================================================
sph_surf = cn.sphere([0, 0, 0], SPHERE_R)
ell_surf, ell_K = ellipsoid(*ELLIPSOID_AXES)
tor_surf, tor_K = torus(TORUS_R, TORUS_r)
CASES = {
    "sphere": (sph_surf, lambda P: np.full(len(P), 1.0 / SPHERE_R ** 2), cube_surface_mesh(sph_surf, SPHERE_R), 2,
               4 * np.pi * SPHERE_R ** 2),
    "ellipsoid": (ell_surf, ell_K, cube_surface_mesh(ell_surf, np.array(ELLIPSOID_AXES)), 2, None),
    "torus": (tor_surf, tor_K, torus_mesh(tor_surf), 0, 4 * np.pi ** 2 * TORUS_R * TORUS_r),
}

rows = {}
for name, (surf, K, ref, chi, exact_area) in CASES.items():
    t0 = time.perf_counter()
    m = jittered(ref, surf, SEED)
    out, logs, res, nrm, nrep = smooth_on(m, surf)
    mr, mi, mo = (measure(x, surf, K) for x in (ref, m, out))
    on = cn.on_target_report(out.points, out.corners, res)
    val = np.bincount(m.corners.ravel(), minlength=m.n_points)
    rows[name] = dict(ref=ref, input=m, out=out, logs=logs, res=res, nrm=nrm, ref_m=mr, in_m=mi, out_m=mo,
                      on=on, chi=chi, K=K, surf=surf, n_irregular=int((val != 4).sum()))
    print("%s: %d quads, %d vertices (%d irregular, valence != 4), chi = %d, jitter %.2f edge, %d quads disagree "
          "with the surface normal" % (name, m.n_quads, m.n_points, rows[name]["n_irregular"], chi, NOISE, nrep.n_disagree))
    q0, q1 = logs[0].before, logs[-1].after
    print("  inverted %d -> %d | scaled J min %.3f -> %.3f | F/quad %.4f -> %.4f"
          % (q0.n_inverted, q1.n_inverted, q0.sj_min, q1.sj_min, q0.objective_mean, q1.objective_mean))
    print("  angle-defect error vs exact dual-cell K:  rms %.4f -> %.4f  (undisturbed mesh %.4f) | max %.4f -> %.4f  (undisturbed %.4f)"
          % (mi["rms"], mo["rms"], mr["rms"], mi["max"], mo["max"], mr["max"]))
    print("  sum of angle defects / 2 pi = %.10f (chi = %d)" % (quadmesh.angle_defect(out).sum() / (2 * np.pi), chi))
    print("  sagitta max %.4f -> %.4f (undisturbed %.4f) | on-target max distance %.1e (passed %s)"
          % (mi["sag_max"], mo["sag_max"], mr["sag_max"], on.max_distance, on.passed))
    ex = "" if exact_area is None else " | vs exact area %.3f %%" % (100 * (mo["area"] / exact_area - 1))
    print("  area drift %.3f %% (%.4f -> %.4f)%s  (%.1f s)"
          % (100 * (mo["area"] / mi["area"] - 1), mi["area"], mo["area"], ex, time.perf_counter() - t0))
    quad_to_vtu(m, "04_%s_before.vtu" % name, point_data={"defect_error": mi["err"]},
                cell_data={"sj_min": per_quad_sj(m, nrm)})
    quad_to_vtu(out, "04_%s_after.vtu" % name,
                point_data={"defect_error": mo["err"], "displacement": out.points - m.points},
                cell_data={"sj_min": per_quad_sj(out, nrm)})

# ============================================================================
# (a) what the smoothing does not do
# ============================================================================
print("\n(a) limits that smoothing does not remove")
e = rows["ellipsoid"]
Pm, Po = e["input"].points, e["out"].points
k_vertex = e["K"](e["out"].points)
hi, lo = k_vertex >= np.quantile(k_vertex, 0.8), k_vertex <= np.quantile(k_vertex, 0.2)
h_in, h_out = cn.vertex_length(Pm, e["input"].corners), cn.vertex_length(Po, e["out"].corners)
print("  ellipsoid K spans %.3f .. %.3f (ratio %.1f); mean local edge at the most curved 20 %% of vertices / the least: "
      "%.3f -> %.3f   (a mesh that followed curvature would have a ratio well below 1)"
      % (k_vertex.min(), k_vertex.max(), k_vertex.max() / k_vertex.min(),
         h_in[hi].mean() / h_in[lo].mean(), h_out[hi].mean() / h_out[lo].mean()))
for name, r in rows.items():
    print("  %-9s error floor: the undisturbed regular mesh reads rms %.4f; the smoothed one %.4f (%.1fx of it); "
          "%d irregular vertices carry what the topology forces" % (name, r["ref_m"]["rms"], r["out_m"]["rms"],
                                                                   r["out_m"]["rms"] / r["ref_m"]["rms"], r["n_irregular"]))
mesh = rows["ellipsoid"]["out"]

# ============================================================================
# (b) the same ellipsoid with ONE folded input quad: surface normals versus winding normals
# ============================================================================
print("\n(b) one folded input quad at a time: metric normals from the surface vs from the quads' winding (ellipsoid)")
t0 = time.perf_counter()
base = rows["ellipsoid"]["input"]
print("  fold at quad: input angle-defect rms | surface normals: inverted left, rms | winding normals: inverted left, rms"
      "   (inverted counted against the surface normals)")
fold_out = {}
for fq in FOLD_QUADS:
    Pf = np.array(base.points)
    c = base.corners[fq]
    Pf[c[0]] = ell_surf.project((2 * Pf[c[2]] - Pf[c[0]])[None])[0]        # corner 0 thrown past corner 2: a bow-tie
    folded = smooth.with_points(base, Pf)
    cells = []
    for label in ("surface", "winding"):
        out, logs, res, nrm, nrep = smooth_on(folded, ell_surf, normals_from=label)
        me = measure(out, ell_surf, ell_K)
        left = smooth.quality(out.points, out.corners, nrm).n_inverted
        fold_out[fq, label] = (out, me, left)
        cells.append("%d, %.3f" % (left, me["rms"]))
        quad_to_vtu(out, "04_b_ellipsoid_fold%d_%s_normals.vtu" % (fq, label), point_data={"defect_error": me["err"]},
                    cell_data={"sj_min": per_quad_sj(out, nrm)})
    print("  %3d: %.3f (%d quads disagree with the surface) | %-10s | %-10s" % (fq, measure(folded, ell_surf, ell_K)["rms"],
                                                                         nrep.n_disagree, *cells))
fq0 = FOLD_QUADS[0]
w, s_ = fold_out[fq0, "winding"], fold_out[fq0, "surface"]
n_clean = {lab: sum(fold_out[fq, lab][2] == 0 for fq in FOLD_QUADS) for lab in ("surface", "winding")}
print("  fold at quad %d: winding normals leave an error %.1fx the surface-normal one -- the metric was fitting the "
      "fold, not the ellipsoid.  Folds fully repaired: %d of %d with surface normals, %d of %d with winding normals."
      % (fq0, w[1]["rms"] / s_[1]["rms"], n_clean["surface"], len(FOLD_QUADS), n_clean["winding"], len(FOLD_QUADS)))
print("  Caveat: surface normals do not make every fold repairable -- a fold that leaves an inverted quad above is the "
      "persistent-overlap issue (open, being investigated), not a claim of robustness.  (%.1f s)" % (time.perf_counter() - t0))
