"""The Laplace boundary stack: source (no normal shrink), movement (``resolve(..., boundary=)``), constraint
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
    m0 = quadmesh.ogrid(linemesh.circle(1.0, wall_n, element_tag="wall"), wall_n // 4, 4, wall_tag="wall")
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


def test_tangent_keeps_a_free_disk_area_and_free_shrinks_it():
    m = _disk()
    a0 = _area(m)
    tan, _ = smooth.laplace(m, cn.resolve(m, [], boundary="tangent"), passes=100, w=0.95)
    free, _ = smooth.laplace(m, cn.resolve(m, [], boundary="free"), passes=100, w=0.95)
    assert abs(_area(tan) / a0 - 1) < 1e-3
    assert _area(free) / a0 - 1 < -0.10


def test_circle_tangent_is_the_analytic_tangent_on_an_uneven_arc():
    rng = np.random.default_rng(0)
    th = np.sort(rng.uniform(0.0, 3.0, 40))
    R, c = 2.5, np.array([0.3, -0.7, 1.1])
    for e1, e2 in (((1, 0, 0), (0, 1, 0)), ((0, 1, 0), (0, 0, 1)), ((1, 1, 0), (0, 0, 1))):
        e1, e2 = np.array(e1, float) / np.linalg.norm(e1), np.array(e2, float)
        X = c + R * (np.cos(th)[:, None] * e1 + np.sin(th)[:, None] * e2)
        t = cn.circle_tangent(X[:-2], X[1:-1], X[2:])
        exact = -np.sin(th[1:-1])[:, None] * e1 + np.cos(th[1:-1])[:, None] * e2
        np.testing.assert_allclose(t, exact, atol=1e-12)
    line = np.array([[0.0, 0, 0], [1, 0, 0], [3.5, 0, 0]])           # collinear: the edge direction
    np.testing.assert_allclose(cn.circle_tangent(line[:1], line[1:2], line[2:]), [[1.0, 0, 0]], atol=1e-15)


def test_a_free_disk_with_uneven_wall_spacing_keeps_its_area():
    m = _disk(wall_n=24)
    P = np.array(m.points, float)
    wall = np.asarray(boundary_points(m))
    ang = np.arctan2(P[wall, 1], P[wall, 0])
    P[wall, 0], P[wall, 1] = np.cos(ang + 0.12 * np.sin(3 * ang)), np.sin(ang + 0.12 * np.sin(3 * ang))
    m = smooth.with_points(m, P)
    r = cn.resolve(m, [], boundary="tangent")
    a0 = _area(m)
    # evening the wall spacing alone raises the inscribed polygon's area a little (here a few tenths of a percent); a wall
    # pulled onto its chords would lose >10 %
    for out in (smooth.laplace(m, r, passes=100, w=0.95)[0], smooth.winslow(m, r, sweeps=60)[0],
                smooth.optimize(m, r, iters=40)[0]):
        assert abs(_area(out) / a0 - 1) < 6e-3


def test_free_reproduces_the_unprojected_pass_bit_for_bit():
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
    out, log = smooth.laplace(m, r, passes=passes, w=w)
    assert log.stopped == "completed"
    np.testing.assert_array_equal(out.points, P)
    del nrm


def test_boundary_mode_is_validated_and_passes_through_the_schedule():
    m = _disk()
    with pytest.raises(ValueError, match="boundary"):
        cn.resolve(m, [], boundary="chain")
    sched = [("laplace", {"passes": 5, "w": 0.95})]
    a, _ = smooth.run_schedule(m, cn.resolve(m, [], boundary="tangent"), sched)
    b, _ = smooth.run_schedule(m, cn.resolve(m, [], boundary="free"), sched)
    assert abs(_area(a) - _area(m)) < 1e-3 * _area(m) < abs(_area(b) - _area(m))


def _wall_normal_drift(m, out, ids):
    """Largest distance a wall vertex moved along the circle's outward normal."""
    P0, P1 = np.asarray(m.points)[ids], np.asarray(out.points)[ids]
    n = P0 / np.linalg.norm(P0, axis=1, keepdims=True)
    return float(np.abs(np.einsum("ij,ij->i", P1 - P0, n)).max())


def test_tangent_moves_the_wall_along_it_in_every_method_and_the_schedule():
    m = _disk(wall_n=48)
    ids = np.asarray(boundary_points(m))
    r = cn.resolve(m, [], boundary="tangent")
    assert r.counts()["tangent"] == ids.size and r.counts()["frozen"] == 0
    a0 = _area(m)
    outs = [smooth.laplace(m, r, passes=20, w=0.95)[0], smooth.winslow(m, r, sweeps=20)[0],
            smooth.optimize(m, r, iters=20)[0],
            smooth.run_schedule(m, r, [("laplace", {"passes": 10}), ("winslow", {"sweeps": 10}),
                                       ("optimize", {"iters": 10})])[0]]
    assert max(np.linalg.norm(np.asarray(o.points)[ids] - np.asarray(m.points)[ids], axis=1).max()
               for o in outs) > 1e-4                          # it does slide ...
    for out in outs:
        moved = np.linalg.norm(np.asarray(out.points)[ids] - np.asarray(m.points)[ids], axis=1).max()
        assert _wall_normal_drift(m, out, ids) < 5e-3 * moved + 2e-4   # ... and only along the wall (to second order)
        assert abs(_area(out) / a0 - 1) < 2e-3


def test_free_wall_pushes_the_optimizer_off_the_circle_where_tangent_does_not():
    m = _disk(wall_n=24)
    a0 = _area(m)
    free, _ = smooth.optimize(m, cn.resolve(m, [], boundary="free"), iters=40)
    tan, _ = smooth.optimize(m, cn.resolve(m, [], boundary="tangent"), iters=40)
    assert abs(_area(tan) / a0 - 1) < abs(_area(free) / a0 - 1) / 5


def test_tangent_constraint_freezes_corners_and_rejects_interior_vertices():
    P0, q0 = grid(4)
    sq = quadmesh.QuadMesh.from_corners(P0, q0)
    ids = np.asarray(boundary_points(sq))
    r = cn.resolve(sq, [cn.Tangent(ids)])
    P = np.asarray(sq.points)
    on_x = (np.abs(P[:, 0] - P[:, 0].min()) < 1e-12) | (np.abs(P[:, 0] - P[:, 0].max()) < 1e-12)
    on_y = (np.abs(P[:, 1] - P[:, 1].min()) < 1e-12) | (np.abs(P[:, 1] - P[:, 1].max()) < 1e-12)
    corner = (on_x & on_y)[ids]
    assert (r.kind[ids[corner]] == cn.FROZEN).all() and corner.sum() == 4
    assert (r.kind[ids[~corner]] == cn.TANGENT).all()
    interior = np.setdiff1d(np.arange(sq.n_points), ids)
    with pytest.raises(ValueError, match="not on the boundary"):
        cn.resolve(sq, [cn.Tangent(interior[:1])])


def test_corner_rule_circle_square_ellipse_and_sphere_patch():
    def ring(P):
        n = len(P)
        return {i: [(i - 1) % n, (i + 1) % n] for i in range(n)}
    th = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    circle8 = np.c_[np.cos(th), np.sin(th), 0 * th]
    assert cn.boundary_corners(circle8, ring(circle8)) == set()                 # 45 deg turns everywhere
    side = np.linspace(0, 1, 5)[:-1]
    sq = np.r_[np.c_[side, 0 * side], np.c_[1 + 0 * side, side], np.c_[1 - side, 1 + 0 * side],
               np.c_[0 * side, 1 - side]]
    sq = np.c_[sq, np.zeros(len(sq))]
    assert cn.boundary_corners(sq, ring(sq)) == {0, 4, 8, 12}                   # 4 points per side
    ph = np.linspace(0, 2 * np.pi, 12, endpoint=False)
    ell = np.c_[3.0 * np.cos(ph), np.sin(ph), 0 * ph]                           # coarse tips: turns 70 vs ~35
    assert cn.boundary_corners(ell, ring(ell)) == set()
    # a patch of a sphere with a circular rim: the surface's own bending is not a corner
    rim = quadmesh.ogrid(linemesh.circle(1.0, 16, element_tag="wall"), 4, 2, wall_tag="wall")
    P = np.array(rim.points, float)
    r2 = np.hypot(P[:, 0], P[:, 1])
    P[:, 2] = np.sqrt(np.maximum(1.0 - 0.9 * r2 ** 2, 0.0))                     # a cap, strongly curved
    cap = smooth.with_points(rim, P)
    assert (cn.resolve(cap, [], boundary="tangent").kind[boundary_points(cap)] == cn.TANGENT).all()
    adj = cn.boundary_adjacency(cap)
    nrm = cn.vertex_normals(P, cap.corners)
    flat = cn.boundary_corners(P, adj, 10.0, None)
    geo = cn.boundary_corners(P, adj, 10.0, nrm)
    assert len(geo) <= len(flat) and geo == set()


def test_oncurve_without_a_curve_is_the_old_slide_and_onsurface_stays_on_the_input_faces():
    m = _disk(wall_n=24)
    ids = np.asarray(boundary_points(m))
    r = cn.resolve(m, [], boundary="slide")
    r2 = cn.resolve(m, [cn.OnCurve(ids)])
    np.testing.assert_array_equal(r.kind, r2.kind)
    out, _ = smooth.optimize(m, r, iters=15)
    chord = 1.0 - np.cos(np.pi / 24)
    assert np.abs(np.hypot(*np.asarray(out.points)[ids, :2].T) - 1.0).max() < 3 * chord
    # a fitted surface holds vertices on the input faces, to the tessellation's chord error
    rim = quadmesh.ogrid(linemesh.circle(1.0, 16, element_tag="wall"), 4, 2, wall_tag="wall")
    P = np.array(rim.points, float)
    P[:, 2] = np.hypot(P[:, 0], P[:, 1]) ** 2
    bowl = smooth.with_points(rim, P)
    allv = np.arange(bowl.n_points)
    rs = cn.resolve(bowl, [cn.OnSurface(allv)])
    assert rs.counts()["on_surface"] == bowl.n_points
    Q = cn.snap(np.asarray(bowl.points) + np.array([0.0, 0.0, 0.05]), rs, 1.0)
    assert np.abs(Q - np.asarray(bowl.points)).max() < 0.05
