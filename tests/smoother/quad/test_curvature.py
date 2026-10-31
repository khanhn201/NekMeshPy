"""Smoothing a quad mesh that lives on a curved 3-D surface: exactly on target, and what it does to curvature.

The Gauss-Bonnet sum (on the triangulated quads) is a topological identity every mesh satisfies, so it
guards against a corrupted mesh, not against bad smoothing.  The informative checks are the local ones:
the angle defect of each vertex against the *exact* integral of the Gaussian curvature over its dual cell,
and the sagitta (distance of between-vertex samples to the surface).
"""

import numpy as np
import pytest
import surfaces as S

from nekmeshpy import quadmesh
from nekmeshpy.quadmesh.smoother import constrain as cn
from nekmeshpy.quadmesh.smoother import smooth
from nekmeshpy.quadmesh.smoother.session import SmoothSession

CLOSED = ["sphere", "ellipsoid", "capsule", "torus"]


def _smooth(name, **kw):
    an, m, ref = S.jittered_case(name, **kw)
    ids = np.arange(m.n_points)
    if name == "saddle":                                  # an open patch: the boundary stays where it is
        ids = np.setdiff1d(ids, quadmesh.query.boundary_points(m))
    r = cn.resolve(m, [cn.OnSurface(ids, an.surface)], boundary="frozen" if name == "saddle" else "free")
    nrm, _ = cn.surface_normals(m.points, m.corners, cn.claimed_surfaces(r))      # the surface's own normals
    out, logs = smooth.run_schedule(m, r, smooth.suggest_schedule(smooth.quality(m.points, m.corners, nrm)), normals=nrm)
    return an, m, ref, out, logs, r


@pytest.fixture(scope="module", params=CLOSED + ["saddle"])
def case(request):
    return (request.param,) + _smooth(request.param)


def test_gauss_bonnet_identity_holds_before_and_after(case):
    name, an, m, ref, out, logs, r = case
    for mesh in (ref, m, out):
        assert abs(quadmesh.gauss_bonnet_residual(mesh)) < 1e-10
    chi = {"sphere": 2, "ellipsoid": 2, "capsule": 2, "torus": 0}.get(name)
    if chi is not None:                                   # closed: the defects themselves sum to 2 pi chi
        assert abs(quadmesh.angle_defect(out).sum() - 2 * np.pi * chi) < 1e-10


def test_smoothed_mesh_is_valid_and_exactly_on_the_surface(case):
    name, an, m, ref, out, logs, r = case
    assert logs[0].before.n_inverted > 0 and logs[-1].after.n_inverted == 0
    rep = cn.on_target_report(out.points, out.corners, r)
    assert rep.passed and rep.max_distance < 1e-9
    assert logs[-1].after.objective_mean < logs[0].before.objective_mean


def test_area_and_volume_do_not_drift(case):
    name, an, m, ref, out, logs, r = case
    assert abs(S.area(out) - S.area(m)) / S.area(m) < 0.01
    if name == "torus":
        exact = 2 * np.pi ** 2 * 1.0 * 0.35 ** 2
        assert abs(abs(S.volume(out)) - exact) / exact < 0.08          # chordal faceting, not drift
        assert abs(abs(S.volume(out)) - abs(S.volume(m))) / exact < 0.02


@pytest.mark.parametrize("name", ["sphere", "capsule", "torus"])
def test_local_curvature_error_improves(name):
    """On these meshes shape smoothing also brings the angle defect back toward the exact curvature."""
    an, m, ref, out, logs, r = _smooth(name)
    before, after = S.curvature_metrics(m, an), S.curvature_metrics(out, an)
    assert after["rms"] < before["rms"]


def test_ellipsoid_local_curvature_error_does_not_grow():
    """Was an xfail while the metric took its normals from the input mesh: one folded input quad forced a
    4-vertex dipole (0.157 -> 0.231).  With the surface's own normals it is 0.157 -> ~0.017."""
    an, m, ref, out, logs, r = _smooth("ellipsoid")
    assert S.curvature_metrics(out, an)["rms"] <= S.curvature_metrics(m, an)["rms"]


def test_saddle_interior_curvature_and_frozen_boundary():
    an, m, ref, out, logs, r = _smooth("saddle")
    b = quadmesh.query.boundary_points(m)
    np.testing.assert_array_equal(out.points[b], m.points[b])
    after = S.curvature_metrics(out, an, interior_only=True)
    assert after["rms"] < 0.05                                # K < 0 everywhere; compare with the unit of curvature


def test_a_long_session_on_the_sphere_settles(tmp_path):
    """Rounds keep improving until a plateau; the plateau is the mesh's own valence-limited shape quality."""
    an, m, ref = S.jittered_case("sphere")
    nrm = smooth.reference_normals(ref.points, ref.corners)
    s = SmoothSession(m, [cn.OnSurface(np.arange(m.n_points), an.surface)], boundary="free", normals=nrm)
    import contextlib
    import io
    f = []
    for k in range(3):
        with contextlib.redirect_stdout(io.StringIO()):
            rep = s.run(1, review="print", schedule=None if k == 0 else smooth.user_schedule(outer=6))[0]
        f.append(rep.after.objective_mean)
    assert f[1] <= f[0] + 1e-9 and abs(f[2] - f[1]) < 0.01 * f[0]
    assert s.undo(3) is m


def test_gauss_bonnet_residual_fails_only_on_an_inconsistent_edge_count_and_is_blind_to_the_rest():
    """residual = 2 pi (E - 2Q) on a closed mesh: it jumps by 2 pi per non-manifold fin and by pi for a quad naming a
    vertex twice, and stays at rounding for an unwelded seam, flipped winding, a self-intersection, a hole... which
    `check_connectivity` (not this identity) is there to catch.  Scripted version: scratch/quad_smooth/gb_failure_examples.py"""
    P, q = S.random_quad_surface(S.sphere(), 40, seed=1)
    q = S.orient_outward(P, q)

    def residual(Pc, qc):
        return quadmesh.gauss_bonnet_residual(quadmesh.QuadMesh.from_corners(Pc, qc))

    assert abs(residual(P, q)) < 1e-10
    n = len(P)
    e = (q[0, 0], q[0, 1])
    fin = (np.vstack([P, P[list(e)] + [0, 0, 0.3]]), np.vstack([q, [[e[0], e[1], n + 1, n]]]))
    assert residual(*fin) == pytest.approx(-2 * np.pi, abs=1e-9)                  # fails badly: one fin = 2 pi
    q8 = q.copy()
    q8[3] = [q[3, 0], q[3, 1], q[3, 1], q[3, 3]]
    assert residual(P, q8) == pytest.approx(np.pi, abs=1e-9)                       # a quad naming a vertex twice = pi
    flipped = q.copy()
    flipped[::2] = flipped[::2, ::-1]
    pushed = P.copy()
    pushed[:20] *= -0.3
    holed = np.delete(q, [0, 1, 2], axis=0)
    for blind in (residual(P, flipped), residual(pushed, q), residual(P, holed)):  # orientation, self-intersection, a hole
        assert abs(blind) < 1e-9
    zc = P[q].mean(1)[:, 2]
    up = zc > 0
    seam = np.intersect1d(np.unique(q[up]), np.unique(q[~up]))
    remap = -np.ones(len(P), int)
    remap[seam] = np.arange(len(P), len(P) + len(seam))
    cut = q.copy()
    for i in np.flatnonzero(up):
        cut[i] = np.where(remap[q[i]] >= 0, remap[q[i]], q[i])
    cut_mesh = quadmesh.QuadMesh.from_corners(np.vstack([P, P[seam]]), cut)
    assert abs(quadmesh.gauss_bonnet_residual(cut_mesh)) < 1e-9                     # an unwelded seam: invisible to the identity ...
    assert not quadmesh.check_connectivity(cut_mesh).passed                        # ... and caught by the gap check
