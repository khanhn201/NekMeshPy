"""M4 of the quad smoother: the condition-number optimizer and the schedule runner."""

import numpy as np
import pytest
from grids import grid

from nekmeshpy import linemesh, quadmesh
from nekmeshpy.quadmesh.query import boundary_points
from nekmeshpy.quadmesh.smoother import constrain as cn
from nekmeshpy.quadmesh.smoother import smooth


def _mesh(n=10, jitter=0.0, seed=5):
    return quadmesh.QuadMesh.from_corners(*grid(n, jitter=jitter, seed=seed))


def _disk(noise=0.03, seed=3):
    sec = quadmesh.ogrid(linemesh.circle(1.0, 24, element_tag="wall"), 6, 4, wall_tag="wall")
    P = sec.points + np.random.default_rng(seed).normal(size=sec.points.shape) * [noise, noise, 0]
    sec = smooth.with_points(sec, P)
    return sec, cn.resolve(sec, [cn.OnCurve("wall", cn.circle([0, 0, 0], 1.0))])


def test_untangles_a_mesh_that_starts_inverted():
    m = _mesh(10, jitter=0.45)
    out, log = smooth.optimize(m, cn.resolve(m, []), iters=60)
    assert log.before.n_inverted == 2 and log.after.n_inverted == 0
    assert log.after.sj_min > 0.9 and log.after.objective_mean < log.before.objective_mean


def test_without_the_barrier_removal_it_lowers_f_but_never_untangles():
    """Why `delta` exists: the exact phi is a wall at zero Jacobian that no line search crosses."""
    m = _mesh(10, jitter=0.45)
    _, log = smooth.optimize(m, cn.resolve(m, []), iters=60, delta=0.0)
    assert log.after.objective_mean < log.before.objective_mean
    assert log.after.n_inverted == log.before.n_inverted > 0


def test_objective_never_rises_on_a_valid_start():
    m = _mesh(12, jitter=0.2)
    assert smooth.quality(m.points, m.corners, smooth.reference_normals(m.points, m.corners)).n_inverted == 0
    _, log = smooth.optimize(m, cn.resolve(m, []), iters=40)
    f = np.array([log.before.objective_mean] + [h[3] for h in log.history])
    assert (np.diff(f) <= 1e-9 * np.abs(f[:-1]).max()).all()
    assert log.after.n_inverted == 0


def test_never_accepts_a_step_that_adds_an_inverted_quad():
    m = _mesh(10, jitter=0.45)
    _, log = smooth.optimize(m, cn.resolve(m, []), iters=40)
    counts = [log.before.n_inverted] + [h[1] for h in log.history]
    assert (np.diff(counts) <= 0).all()


def test_an_already_ideal_mesh_stops_at_once():
    m = _mesh(6)
    out, log = smooth.optimize(m, cn.resolve(m, []), iters=10)
    assert log.stopped == "converged" and log.passes_done == 0
    np.testing.assert_array_equal(out.points, m.points)


def test_boundary_slides_tangentially_and_stays_on_the_circle_every_run():
    sec, r = _disk()
    out, log = smooth.optimize(sec, r, iters=60)
    rep = cn.on_target_report(out.points, out.corners, r)
    assert rep.passed and rep.max_distance < 1e-13
    w = r.kind == cn.ON_CURVE
    a0 = np.arctan2(sec.points[w, 1], sec.points[w, 0])
    a1 = np.arctan2(out.points[w, 1], out.points[w, 0])
    assert np.abs(a1 - a0).max() > 1e-3                      # it really slid
    assert log.after.n_inverted == 0


def test_frozen_and_surface_vertices_obey_their_constraints():
    P, q = grid(8, jitter=0.4, seed=2)
    u, v = (P[:, 0] - 0.5), (P[:, 1] - 0.5)
    P = np.stack([np.sin(u) * np.cos(v), np.sin(v), np.cos(u) * np.cos(v)], 1)
    m = quadmesh.QuadMesh.from_corners(P, q)
    if np.einsum("qj,qj->q", smooth.reference_normals(m.points, m.corners), m.points[m.corners].mean(1)).mean() < 0:
        m = quadmesh.QuadMesh.from_corners(P, q[:, ::-1].copy())
    bd = boundary_points(m)
    inside = np.setdiff1d(np.arange(m.n_points), bd)
    r = cn.resolve(m, [cn.OnSurface(inside, cn.sphere([0, 0, 0], 1.0))])    # boundary frozen
    m = smooth.with_points(m, cn.snap(m.points, r))
    out, log = smooth.optimize(m, r, iters=40)
    np.testing.assert_array_equal(out.points[bd], m.points[bd])
    assert cn.on_target_report(out.points, out.corners, r).passed
    assert log.after.n_inverted <= log.before.n_inverted


def test_local_variant_moves_only_the_vertices_of_the_chosen_quads():
    m = _mesh(10, jitter=0.4)
    nrm = smooth.reference_normals(m.points, m.corners)
    sel = smooth.failing_elements(m.points, m.corners, nrm, sj_target=0.4)
    assert sel.any()
    out, log = smooth.optimize(m, cn.resolve(m, [], boundary="free"), iters=30, elements=sel)
    touched = np.unique(m.corners[sel])
    rest = np.setdiff1d(np.arange(m.n_points), touched)
    np.testing.assert_array_equal(out.points[rest], m.points[rest])
    assert not np.allclose(out.points[touched], m.points[touched])
    assert log.method == "optimize_local"
    assert smooth.optimize(m, cn.resolve(m, []), elements=np.zeros(m.n_quads, bool))[1].stopped == "no_elements"


@pytest.mark.parametrize("scale", [1e-3, 1e3])
def test_no_hidden_units_scaling_the_whole_model_changes_nothing(scale):
    """Steps, finite scales and tolerances are in local edge lengths, so the history is identical."""
    m = _mesh(8, jitter=0.4)
    big = quadmesh.QuadMesh.from_corners(m.points * scale, m.corners)
    o1, l1 = smooth.optimize(m, cn.resolve(m, []), iters=25)
    o2, l2 = smooth.optimize(big, cn.resolve(big, []), iters=25)
    assert l1.passes_done == l2.passes_done and l1.stopped == l2.stopped
    np.testing.assert_allclose(np.array(l1.history)[:, 3], np.array(l2.history)[:, 3], rtol=1e-6)
    np.testing.assert_allclose(o2.points, o1.points * scale, rtol=1e-6, atol=1e-9 * scale)


def test_both_cg_restart_variants_converge():
    m = _mesh(10, jitter=0.4)
    r = cn.resolve(m, [])
    a = smooth.optimize(m, r, iters=60, restart="halve")[1].after
    b = smooth.optimize(m, r, iters=60, restart="none")[1].after
    assert a.n_inverted == b.n_inverted == 0 and a.objective_mean < 0.05 and b.objective_mean < 0.05


def test_input_untouched_and_bad_arguments_refused():
    m = _mesh(6, jitter=0.3)
    before = m.points.copy()
    smooth.optimize(m, cn.resolve(m, []), iters=5)
    np.testing.assert_array_equal(m.points, before)
    with pytest.raises(ValueError, match="iters"):
        smooth.optimize(m, cn.resolve(m, []), iters=0)
    with pytest.raises(ValueError, match="restart"):
        smooth.optimize(m, cn.resolve(m, []), restart="x")


def test_run_schedule_chains_stages_logs_each_and_refuses_unknown_methods():
    m = _mesh(14, jitter=0.45)
    before = m.points.copy()
    out, logs = smooth.run_schedule(m, cn.resolve(m, []), smooth.user_schedule(outer=2, n_opt=10))
    assert [l.method for l in logs][:4] == ["laplace", "optimize", "laplace", "optimize"]
    assert logs[-1].method == "laplace_local" and len(logs) == 6
    assert logs[0].before.n_inverted == 4 and logs[-1].after.n_inverted == 0
    np.testing.assert_array_equal(m.points, before)
    with pytest.raises(ValueError, match="unknown"):
        smooth.run_schedule(m, cn.resolve(m, []), [("magic", {})])


def test_shape_weights_scale_only_the_shape_term_and_keep_the_inversion_penalty():
    from grids import grid as _g
    P, q = _g(6, jitter=0.4, seed=2)
    nrm = smooth.reference_normals(P, q)
    f1 = smooth.objective(P, q, nrm)
    w0 = np.zeros(len(q))
    f0 = smooth.objective(P, q, nrm, weights=w0)
    assert np.isclose(smooth.objective(P, q, nrm, weights=np.ones(len(q))), f1)
    assert f0 < f1                                      # no shape term left ...
    bad = P.copy()
    bad[10] = [0.9, 0.9, 0.0]                           # ... but an inversion is still costly
    assert smooth.objective(bad, q, nrm, weights=w0) > 10 * smooth.EPS
    half = np.full(len(q), 0.5)
    g = smooth.gradient(P, q, nrm, weights=half)
    h = 1e-7
    gf = np.zeros_like(P)
    for i in range(P.shape[0]):
        for k in range(2):
            a = P.copy()
            a[i, k] += h
            b = P.copy()
            b[i, k] -= h
            gf[i, k] = (smooth.objective(a, q, nrm, weights=half)
                        - smooth.objective(b, q, nrm, weights=half)) / (2 * h)
    np.testing.assert_allclose(g[:, :2], gf[:, :2], rtol=1e-4, atol=1e-5)


def test_a_zero_weight_region_is_left_alone_while_the_weighted_one_improves():
    """Jittered unit square, free vertices: with shape weight 0 on the lower half the optimizer spends its effort on
    the upper half, so the upper half ends up squarer than the lower one; with unit weights both improve."""
    from grids import grid as _g
    P, q = _g(10, jitter=0.4, seed=6)
    m = quadmesh.QuadMesh.from_corners(P, q)
    r = cn.resolve(m, [], boundary="frozen")
    nrm = smooth.reference_normals(m.points, m.corners)
    lower = m.points[m.corners].mean(1)[:, 1] < 0.5
    out_w, _ = smooth.optimize(m, r, iters=60, normals=nrm, weights=np.where(lower, 0.0, 1.0))
    out_u, _ = smooth.optimize(m, r, iters=60, normals=nrm)

    def mean_phi(M, sel):
        return float(smooth.corner_metrics(M.points, M.corners, nrm).phi.max(1)[sel].mean())

    assert mean_phi(out_w, ~lower) < mean_phi(m, ~lower)                 # the weighted half improved
    assert mean_phi(out_w, lower) > mean_phi(out_u, lower)               # the zero-weight half improved less than with unit weights
