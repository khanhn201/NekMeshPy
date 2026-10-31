"""The Laplace boundary stack: source (no normal shrink), movement (``free_boundary``), constraint
(``resolve(..., boundary="slide")``)."""

import numpy as np
import pytest
from grids import grid

from nekmeshpy import linemesh, quadmesh
from nekmeshpy.quadmesh.query import boundary_points
from nekmeshpy.quadmesh.smoother import constrain as cn
from nekmeshpy.quadmesh.smoother import smooth


def _area(mesh):
    P = np.asarray(mesh.points)[mesh.corners][:, :, :2]
    x, y = P[..., 0], P[..., 1]
    return 0.5 * float((x * np.roll(y, -1, 1) - np.roll(x, -1, 1) * y).sum())


def _disk(wall_n=24, jitter=0.3, seed=3, warp=None):
    m0 = quadmesh.ogrid(linemesh.circle(1.0, wall_n, element_tag="wall"), 6, 4, wall_tag="wall")
    P = np.array(m0.points, float)
    if warp is not None:
        P = warp(P)
    wall = np.zeros(len(P), bool)
    wall[boundary_points(m0)] = True
    m0 = smooth.with_points(m0, P)
    ell = cn.vertex_length(m0.points, m0.corners)
    P = P.copy()
    P[~wall, :2] += np.random.default_rng(seed).uniform(-jitter, jitter, ((~wall).sum(), 2)) * ell[~wall, None]
    return smooth.with_points(m0, P)


def _dist_to_curves(mesh, r):
    """Largest distance of a constrained vertex to its own target curve."""
    d = 0.0
    for t, c in enumerate(r.curves):
        m = (r.kind == cn.ON_CURVE) & (r.target == t)
        if m.any():
            d = max(d, float(np.linalg.norm(c.project(mesh.points[m]) - mesh.points[m], axis=1).max()))
    return d


def test_source_a_boundary_side_is_not_shrunk_across_its_own_normal():
    P = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], float)
    c = np.array([[0, 1, 2, 3]])
    out = smooth._shrunk_corners(P, c, np.array([[True, False, False, False]]), 0.8)
    np.testing.assert_allclose(out[0, :2, 1], 0.0, atol=1e-15)        # side 0 stays on y = 0 ...
    np.testing.assert_allclose(out[0, 2:, 1], 0.9, atol=1e-15)        # ... the opposite one shrinks
    np.testing.assert_allclose(out[0, :2, 0], [0.1, 0.9], atol=1e-15)  # and slides inward along it


def test_tangent_keeps_a_free_disk_area_and_chain_shrinks_it():
    m = _disk()
    r = cn.resolve(m, [], boundary="free")
    a0 = _area(m)
    tan, _ = smooth.laplace(m, r, passes=100, w=0.95)
    chain, _ = smooth.laplace(m, r, passes=100, w=0.95, free_boundary="chain")
    assert abs(_area(tan) / a0 - 1) < 1e-3
    assert _area(chain) / a0 - 1 < -0.10


def test_chain_reproduces_the_unprojected_pass_bit_for_bit():
    m = _disk(jitter=0.2)
    r = cn.resolve(m, [], boundary="free")
    passes, w = 4, 0.95
    nrm = smooth.reference_normals(m.points, m.corners)
    bnd = quadmesh.query.boundary_edge_ids(m)[m.quads]
    mult = np.bincount(m.corners.ravel(), minlength=m.n_points).astype(float)
    P = np.asarray(m.points, float).copy()
    for i in range(1, passes + 1):
        sh = smooth._shrunk_corners(P, m.corners, bnd, w)
        P = cn.snap(smooth._scatter(m.n_points, m.corners, sh) / mult[:, None], r, weight=i / passes)
    out, log = smooth.laplace(m, r, passes=passes, w=w, free_boundary="chain")
    assert log.stopped == "completed"
    np.testing.assert_array_equal(out.points, P)
    del nrm


def test_free_boundary_is_validated_and_passes_through_the_schedule():
    m = _disk()
    r = cn.resolve(m, [], boundary="free")
    with pytest.raises(ValueError, match="free_boundary"):
        smooth.laplace(m, r, free_boundary="nope")
    sched = [("laplace", {"passes": 5, "w": 0.95})]
    a, _ = smooth.run_schedule(m, r, sched)
    b, _ = smooth.run_schedule(m, r, [("laplace", {**sched[0][1], "free_boundary": "chain"})])
    assert abs(_area(a) - _area(m)) < 1e-3 * _area(m) < abs(_area(b) - _area(m))


def test_tangent_leaves_frozen_and_oncurve_vertices_alone():
    m = _disk()
    for cons, bd in (([], "frozen"), ([cn.OnCurve(boundary_points(m), cn.circle([0, 0, 0], 1.0))], "free")):
        r = cn.resolve(m, cons, boundary=bd)
        a, _ = smooth.laplace(m, r, passes=6, w=0.95)
        b, _ = smooth.laplace(m, r, passes=6, w=0.95, free_boundary="chain")
        np.testing.assert_array_equal(a.points, b.points)


def test_slide_keeps_area_under_laplace_and_optimize_and_stays_on_the_curve():
    m = _disk()
    r = cn.resolve(m, [], boundary="slide")
    assert any("boundary='slide'" in n and "0 corner" in n for n in r.notes)
    a0 = _area(m)
    for out in (smooth.laplace(m, r, passes=30, w=0.95)[0], smooth.optimize(m, r, iters=40)[0]):
        assert abs(_area(out) / a0 - 1) < 1e-3
        edge = np.linalg.norm(m.points[m.edges[:, 0]] - m.points[m.edges[:, 1]], axis=1).mean()
        assert _dist_to_curves(out, r) < 1e-9 * edge


def _hexagon():
    def warp(P):
        th = np.arctan2(P[:, 1], P[:, 0])
        rho = np.cos(np.pi / 6) / np.cos(np.mod(th, np.pi / 3) - np.pi / 6)
        return P * np.c_[rho, rho, np.ones(len(P))]
    return _disk(warp=warp)


def test_slide_freezes_the_corners_of_a_square_and_a_hexagon():
    P, q = grid(6, jitter=0.3, seed=2)
    sq = quadmesh.QuadMesh.from_corners(P, q)
    for m, n_corner in ((sq, 4), (_hexagon(), 6)):
        r = cn.resolve(m, [], boundary="slide")
        assert "%d corner(s)" % n_corner in r.notes[-1]
        assert int((r.kind == cn.FROZEN).sum()) == n_corner
        out, _ = smooth.laplace(m, r, passes=10, w=0.9)
        moved = np.linalg.norm(out.points - m.points, axis=1)
        assert moved[r.kind == cn.FROZEN].max() == 0.0
        assert _dist_to_curves(out, r) < 1e-9
        assert abs(_area(out) / _area(m) - 1) < 1e-3


def test_slide_holds_an_ellipse_boundary_close_to_the_true_curve():
    a, b = 1.0, 0.5
    m = _disk(wall_n=24, warp=lambda P: P * [a, b, 1.0])
    r = cn.resolve(m, [], boundary="slide")
    out, _ = smooth.laplace(m, r, passes=100, w=0.95)
    B = boundary_points(m)
    x = out.points[B]
    h = np.linalg.norm(m.points[m.edges[:, 0]] - m.points[m.edges[:, 1]], axis=1)[:1].mean()
    drift = np.abs(np.hypot(x[:, 0] / a, x[:, 1] / b) - 1.0)
    assert drift.max() < 5e-3 * h + 1e-3


def test_slide_leaves_a_claimed_vertex_to_its_own_constraint():
    m = _disk()
    B = boundary_points(m)
    r = cn.resolve(m, [cn.Frozen(B[:3])], boundary="slide")
    assert (r.kind[B[:3]] == cn.FROZEN).all()
    assert (r.kind[B[3:]] == cn.ON_CURVE).sum() + (r.kind[B[3:]] == cn.FROZEN).sum() == len(B) - 3


def test_slide_works_on_a_surface_patch_in_3d():
    n = 8
    u = np.linspace(0, 1, n + 1)
    U, V = np.meshgrid(u, u, indexing="ij")
    ang = 0.5 * np.pi * U
    P = np.stack([np.cos(ang).ravel(), np.sin(ang).ravel(), V.ravel()], 1)
    inner = ((U > 0) & (U < 1) & (V > 0) & (V < 1)).ravel()
    P[inner, 2] += np.random.default_rng(1).uniform(-0.3, 0.3, inner.sum()) / n
    idx = np.arange((n + 1) ** 2).reshape(n + 1, n + 1)
    q = np.stack([idx[:-1, :-1], idx[1:, :-1], idx[1:, 1:], idx[:-1, 1:]], -1).reshape(-1, 4)
    m = quadmesh.QuadMesh.from_corners(P, q)
    r = cn.resolve(m, [], boundary="slide")
    assert "4 corner(s)" in r.notes[-1]
    out, _ = smooth.laplace(m, r, passes=10, w=0.9)
    assert _dist_to_curves(out, r) < 1e-9
    assert out.points[:, :2].shape == (len(P), 2) and np.abs(np.hypot(out.points[:, 0], out.points[:, 1]) - 1).max() < 1e-2


def test_boundary_argument_is_validated():
    with pytest.raises(ValueError, match="slide"):
        cn.resolve(_disk(), [], boundary="nope")
