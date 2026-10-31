"""The hex element-shrink Laplace: the guard, the boundary rules, local passes."""

import numpy as np
from hexblocks import block

from nekmeshpy import hexmesh, linemesh, quadmesh
from nekmeshpy.hexmesh.smoother import constrain as cn
from nekmeshpy.hexmesh.smoother import smooth


def _box_classes(m, tol=1e-12):
    P = m.points
    lo, hi = P.min(axis=0), P.max(axis=0)
    return ((np.abs(P - lo) < tol) | (np.abs(P - hi) < tol))


def test_w_equal_one_is_the_identity_and_a_regular_grid_a_fixed_point():
    m = block(4, jitter=0.3)
    out, log = smooth.laplace(m, cn.resolve(m, [], boundary="free"), w=1.0, passes=2)
    np.testing.assert_allclose(out.points, m.points, atol=1e-14)
    assert log.stopped == "completed"
    g = block(4)
    out, _ = smooth.laplace(g, cn.resolve(g, [], boundary="free"), w=0.9, passes=4)
    np.testing.assert_allclose(out.points, g.points, atol=1e-13)


def test_a_perturbed_block_improves_and_a_frozen_boundary_never_moves():
    m = block(6, jitter=0.3, seed=1)
    r = cn.resolve(m, [])
    out, log = smooth.laplace(m, r, passes=5)
    assert log.after.sj_min > log.before.sj_min and log.after.objective_mean < log.before.objective_mean
    assert log.after.n_inverted <= log.before.n_inverted
    fro = r.kind == cn.FROZEN
    np.testing.assert_array_equal(out.points[fro], m.points[fro])


def test_slide_keeps_every_boundary_vertex_on_its_face_plane_and_corners_still():
    m = block(5, jitter=0.25, seed=2)
    on = _box_classes(m)
    # move the face and edge vertices within their planes / lines, so sliding has something to fix
    d = np.random.default_rng(0).uniform(-0.05, 0.05, m.points.shape)
    d[on] = 0.0
    m = smooth.with_points(m, m.points + d)
    r = cn.resolve(m, [], boundary="slide")
    out, log = smooth.laplace(m, r, passes=6)
    assert log.after.objective_mean < log.before.objective_mean
    Q = out.points
    for ax in range(3):
        for side in (0.0, 1.0):
            sel = np.abs(m.points[:, ax] - side) < 1e-12
            np.testing.assert_allclose(Q[sel, ax], side, atol=1e-14)
    corner = on.sum(axis=1) == 3
    np.testing.assert_array_equal(Q[corner], m.points[corner])
    edge = on.sum(axis=1) == 2
    assert np.abs(Q[edge] - m.points[edge]).max() > 1e-6          # edge vertices did slide


def test_tangent_rule_keeps_a_free_box_and_a_free_cylinder_volume():
    m = block(4, jitter=0.25, seed=3)
    out, _ = smooth.laplace(m, cn.resolve(m, [], boundary="tangent"), passes=5)
    assert abs(hexmesh.volume(out) - 1.0) < 1e-12
    np.testing.assert_allclose(out.points[_box_classes(m).sum(1) == 3], m.points[_box_classes(m).sum(1) == 3])
    sec = quadmesh.ogrid(linemesh.circle(1.0, 16), 4, 2)
    cyl = hexmesh.extrude(sec, 1.0, 2)
    bnd = np.zeros(cyl.n_points, bool)
    bnd[hexmesh.boundary_points(cyl)] = True
    P = cyl.points.copy()
    P[~bnd] += np.random.default_rng(1).normal(size=(int((~bnd).sum()), 3)) * 0.02
    cyl = smooth.with_points(cyl, P)
    V0 = hexmesh.volume(cyl)
    dv = {fb: abs(hexmesh.volume(smooth.laplace(cyl, cn.resolve(cyl, [], boundary=fb), passes=6)[0]) - V0) / V0
          for fb in ("tangent", "free")}
    assert dv["tangent"] < 1e-3 and dv["tangent"] < 0.2 * dv["free"]


def test_a_refused_pass_rolls_back_and_says_why():
    m = block(4, jitter=0.3, seed=5)
    r = cn.resolve(m, [])
    out, log = smooth.laplace(m, r, passes=3, max_sj_drop=-10.0)     # demand an impossible gain
    assert log.stopped == "rolled_back" and log.passes_done == 0 and "refused" in log.reason
    np.testing.assert_array_equal(out.points, m.points)


def test_laplace_local_moves_only_the_selected_hexes_vertices():
    m = block(5, jitter=0.3, seed=6)
    r = cn.resolve(m, [])
    sel = np.zeros(m.n_hexes, bool)
    sel[[31, 62]] = True
    out, log = smooth.laplace(m, r, passes=3, elements=sel)
    assert log.method == "laplace_local"
    moved = np.flatnonzero((out.points != m.points).any(axis=1))
    assert moved.size and set(moved) <= set(np.unique(m.corners[sel]))
    none, log0 = smooth.laplace(m, r, elements=np.zeros(m.n_hexes, bool))
    assert log0.stopped == "no_elements" and none is m


def test_tangent_keeps_box_face_vertices_in_their_planes_and_corners_fixed():
    m = block(5, jitter=0.25, seed=2)
    on = _box_classes(m)
    d = np.random.default_rng(0).uniform(-0.05, 0.05, m.points.shape)
    d[on] = 0.0
    m = smooth.with_points(m, m.points + d)
    r = cn.resolve(m, [], boundary="tangent")
    corner, edge, face = on.sum(1) == 3, on.sum(1) == 2, on.sum(1) == 1
    assert (r.kind[corner] == cn.FROZEN).all() and corner.sum() == 8
    assert (r.kind[edge | face] == cn.TANGENT).all()
    assert len({tuple(sorted(e)) for v in np.flatnonzero(edge) for e in [tuple(r.nbr[v])]}) > 0
    for run in (smooth.laplace(m, r, passes=6)[0], smooth.optimize(m, r, iters=15)[0]):
        Q = run.points
        np.testing.assert_array_equal(Q[corner], m.points[corner])
        for ax in range(3):
            for side in (0.0, 1.0):
                sel = np.abs(m.points[:, ax] - side) < 1e-12
                assert np.abs(Q[sel, ax] - side).max() < 1e-12            # planar patches: exact to rounding
    assert np.abs(smooth.laplace(m, r, passes=6)[0].points[edge | face] - m.points[edge | face]).max() > 1e-6


def test_hex_box_has_eight_corners_and_twelve_edges_and_oncurve_onsurface_fit_the_input():
    m = block(4)
    bp = cn.boundary_patches(m)
    assert bp.planar.size == 6
    assert int((bp.n_patches >= 3).sum()) == 8
    r = cn.resolve(m, [], boundary="slide")
    on = _box_classes(m)
    assert (r.kind[on.sum(1) == 3] == cn.FROZEN).all()
    assert (r.kind[on.sum(1) == 2] == cn.ON_CURVE).all() and (r.kind[on.sum(1) == 1] == cn.ON_SURFACE).all()
    bnd = np.flatnonzero(on.any(axis=1))
    r2 = cn.resolve(m, [cn.OnCurve(bnd), cn.OnSurface(bnd)])
    np.testing.assert_array_equal(r.kind, r2.kind)
    assert len(np.unique(r2.target[r2.kind == cn.ON_CURVE])) == 12
