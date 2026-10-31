"""M3 of the quad smoother: the element-shrink Laplace method."""

import numpy as np
import pytest
from grids import grid

from nekmeshpy import linemesh, quadmesh
from nekmeshpy.quadmesh.query import boundary_points
from nekmeshpy.quadmesh.smoother import constrain as cn
from nekmeshpy.quadmesh.smoother import smooth


def _mesh(n=8, jitter=0.0, seed=1):
    P, q = grid(n, jitter=jitter, seed=seed)
    return quadmesh.QuadMesh.from_corners(P, q)


def test_edge_k_joins_corner_k_and_k_plus_1():
    m = _mesh(4)
    got = np.sort(np.stack([m.corners, np.roll(m.corners, -1, axis=1)], -1), axis=-1)
    want = np.sort(m.edges[m.quads], axis=-1)
    np.testing.assert_array_equal(got, want)      # the shrink's side bookkeeping relies on this


def test_w_equal_one_is_the_identity():
    m = _mesh(6, jitter=0.4)
    r = cn.resolve(m, [], boundary="free")
    out, log = smooth.laplace(m, r, w=1.0, passes=2)
    np.testing.assert_allclose(out.points, m.points, atol=1e-14)
    assert log.stopped == "completed"


def test_a_regular_grid_is_a_fixed_point_even_with_a_free_boundary():
    m = _mesh(6)
    r = cn.resolve(m, [], boundary="free")
    out, _ = smooth.laplace(m, r, w=0.9, passes=5)
    np.testing.assert_allclose(out.points, m.points, atol=1e-13)


def test_jittered_square_gets_better_everywhere_that_matters_and_frozen_never_moves():
    m = _mesh(10, jitter=0.45)
    r = cn.resolve(m, [])                          # boundary frozen by default
    out, log = smooth.laplace(m, r, w=0.9, passes=10)
    assert log.stopped == "completed"
    assert log.after.sj_min > log.before.sj_min
    assert log.after.sj_mean > log.before.sj_mean
    assert log.after.phi_mean < log.before.phi_mean
    assert log.after.n_inverted == 0
    b = boundary_points(m)
    np.testing.assert_array_equal(out.points[b], m.points[b])
    assert out.points is not m.points and not np.array_equal(out.points, m.points)


def test_input_mesh_is_untouched():
    m = _mesh(6, jitter=0.4)
    before = m.points.copy()
    smooth.laplace(m, cn.resolve(m, []), passes=3)
    np.testing.assert_array_equal(m.points, before)


def test_boundary_only_slides_along_the_circle_and_evens_its_spacing():
    ang = np.sort(np.random.default_rng(4).uniform(0, 2 * np.pi, 24))
    ang[::4] = np.arange(6) * np.pi / 3          # keep the four-fold structure ogrid wants
    ring = linemesh.circle(1.0, 24, element_tag="wall")
    sec = quadmesh.ogrid(ring, 6, 3, wall_tag="wall")
    P = np.asarray(sec.points, float).copy()
    wall = np.unique(sec.edges[sec.line_mesh.element_tags.flatnonzero("wall")])
    th = np.arctan2(P[wall, 1], P[wall, 0])
    th2 = th + 0.08 * np.sin(5 * th)             # uneven boundary spacing, still on the circle
    P[wall, 0], P[wall, 1] = np.cos(th2), np.sin(th2)
    sec = smooth.with_points(sec, P)
    r = cn.resolve(sec, [cn.OnCurve("wall", cn.circle([0, 0, 0], 1.0))])

    def spread(M):
        t = np.sort(np.arctan2(M.points[wall, 1], M.points[wall, 0]) % (2 * np.pi))
        d = np.diff(np.r_[t, t[0] + 2 * np.pi])
        return d.std() / d.mean()

    out, log = smooth.laplace(sec, r, w=0.9, passes=10)
    rep = cn.on_target_report(out.points, out.corners, r)
    assert rep.passed and rep.max_distance < 1e-13
    assert spread(out) < spread(sec)             # tangential motion evened the spacing
    assert log.after.n_inverted == 0


def test_surface_vertices_end_exactly_on_the_surface_after_the_snap_ramp():
    P, q = grid(8, jitter=0.4, seed=2)
    u, v = (P[:, 0] - 0.5) * 1.0, (P[:, 1] - 0.5) * 1.0
    P = np.stack([np.sin(u) * np.cos(v), np.sin(v), np.cos(u) * np.cos(v)], 1)
    m = quadmesh.QuadMesh.from_corners(P, q[:, ::-1].copy())
    nrm = smooth.reference_normals(m.points, m.corners)
    if np.einsum("qj,qj->q", nrm, m.points[m.corners].mean(1)).mean() < 0:
        m = quadmesh.QuadMesh.from_corners(P, q)
    sph = cn.sphere([0, 0, 0], 1.0)
    r = cn.resolve(m, [cn.OnSurface(np.arange(m.n_points), sph)], boundary="free")
    out, log = smooth.laplace(m, r, w=0.9, passes=6)
    rep = cn.on_target_report(out.points, out.corners, r)
    assert rep.passed and rep.max_distance < 1e-13
    assert log.after.n_inverted == 0 and log.passes_done == 6


def test_a_refused_pass_rolls_back_to_the_last_accepted_iterate_and_says_why():
    m = _mesh(6, jitter=0.3)
    r = cn.resolve(m, [], boundary="free")
    out, log = smooth.laplace(m, r, w=0.2, passes=6, max_sj_drop=0.0)
    if log.stopped == "rolled_back":
        assert "refused" in log.reason and log.passes_done < 6
    assert log.after.n_inverted <= log.before.n_inverted
    assert log.after.sj_min >= log.before.sj_min - 1e-12 or log.stopped == "completed"


def test_local_variant_moves_only_the_vertices_of_the_chosen_quads():
    m = _mesh(8, jitter=0.4)
    r = cn.resolve(m, [], boundary="free")
    sel = np.zeros(m.n_quads, bool)
    sel[[10, 11]] = True
    out, log = smooth.laplace(m, r, w=0.8, passes=4, elements=sel)
    touched = np.unique(m.corners[sel])
    rest = np.setdiff1d(np.arange(m.n_points), touched)
    np.testing.assert_array_equal(out.points[rest], m.points[rest])
    assert not np.allclose(out.points[touched], m.points[touched])
    assert log.method == "laplace_local"
    assert smooth.laplace(m, r, elements=np.zeros(m.n_quads, bool))[1].stopped == "no_elements"


def test_bad_arguments_are_refused():
    m = _mesh(4)
    r = cn.resolve(m, [])
    for w in (0.0, -0.1, 1.2):
        with pytest.raises(ValueError, match="w must be"):
            smooth.laplace(m, r, w=w)
    with pytest.raises(ValueError, match="passes"):
        smooth.laplace(m, r, passes=0)


def test_quality_reads_known_values_and_the_worst_tail():
    m = _mesh(10)
    nrm = smooth.reference_normals(m.points, m.corners)
    q = smooth.quality(m.points, m.corners, nrm)
    assert q.n_inverted == 0 and q.sj_min == pytest.approx(1.0) and q.phi_max == pytest.approx(1.0)
    assert q.objective_mean == pytest.approx(0.0, abs=1e-12)
    P = m.points.copy()
    P[45] += [0.03, 0.02, 0]
    q2 = smooth.quality(P, m.corners, nrm)
    assert q2.sj_q01 <= q2.sj_q05 <= q2.sj_q10 <= q2.sj_median <= 1.0
    assert q2.sj_min < 1.0 and "inverted" in str(q2)


def test_failing_elements_finds_a_collapsed_corner():
    m = _mesh(8)
    P = m.points.copy()
    P[40] = P[41] + [1e-4, 0, 0]       # an almost-collapsed edge
    nrm = smooth.reference_normals(m.points, m.corners)
    bad = smooth.failing_elements(P, m.corners, nrm)
    assert bad.any() and not bad.all()
    assert smooth.failing_elements(m.points, m.corners, nrm).sum() == 0


def test_neighbour_size_ratio_reads_the_grading_that_the_scaled_jacobian_cannot_see():
    P, q = grid(6)
    uniform = smooth.shape_stats(P, q)
    assert uniform.size_ratio_max == pytest.approx(1.0) and uniform.size_ratio_mean == pytest.approx(1.0)
    G = P.copy()
    G[:, 0] = G[:, 0] ** 2                                    # graded in x: perfect rectangles (scaled J = 1), unequal sizes
    nrm = smooth.reference_normals(G, q)
    assert smooth.quality(G, q, nrm).sj_min == pytest.approx(1.0)
    graded = smooth.shape_stats(G, q)
    assert graded.size_ratio_max == pytest.approx(np.sqrt(3.0))   # widths 1,3,5,... -> sqrt(area) ratio sqrt(3) at the first neighbours
    assert graded.size_ratio_mean > 1.1
    pairs = smooth.quad_neighbours(q)
    assert pairs.shape == (2 * 6 * 5, 2)                      # a 6 x 6 grid has 60 interior edges
