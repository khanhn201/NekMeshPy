"""The remaining planned tests: numbering invariance (T1b), a graded boundary layer (T1e), a box surface with its
creases and corners (T4e), and the negative cases (T6)."""

import numpy as np
import pytest
import surfaces as S
from grids import grid

from nekmeshpy import quadmesh
from nekmeshpy.quadmesh.smoother import constrain as cn
from nekmeshpy.quadmesh.smoother import smooth
from nekmeshpy.quadmesh.smoother.session import SmoothSession


# ------------------------------------------------------------------ T1b: nothing may depend on the numbering
def _permuted(P, q, seed=0):
    """The same mesh with the vertex ids shuffled, the quads shuffled and every quad's start corner rotated."""
    rng = np.random.default_rng(seed)
    pv = rng.permutation(len(P))                          # old id i becomes pv[i]
    P2 = np.empty_like(P)
    P2[pv] = P
    q2 = pv[q]
    rot = rng.integers(0, 4, size=len(q))
    q2 = np.stack([np.roll(row, -r) for row, r in zip(q2, rot)])
    q2 = q2[rng.permutation(len(q2))]
    return P2, q2, pv


def test_smoothing_does_not_depend_on_vertex_quad_or_start_corner_numbering():
    P, q = grid(10, jitter=0.4, seed=5)
    P2, q2, pv = _permuted(P, q, seed=1)
    m1 = quadmesh.QuadMesh.from_corners(P, q)
    m2 = quadmesh.QuadMesh.from_corners(P2, q2)
    sched = smooth.user_schedule(outer=3, n_opt=15)
    n1 = smooth.reference_normals(m1.points, m1.corners)
    n2 = smooth.reference_normals(m2.points, m2.corners)
    o1, l1 = smooth.run_schedule(m1, cn.resolve(m1, []), sched, normals=n1)
    o2, l2 = smooth.run_schedule(m2, cn.resolve(m2, []), sched, normals=n2)
    np.testing.assert_allclose(o2.points[pv], o1.points, atol=1e-7)             # same vertices end in the same places
    assert l1[-1].after.n_inverted == l2[-1].after.n_inverted
    assert l1[-1].after.sj_min == pytest.approx(l2[-1].after.sj_min, abs=1e-7)
    assert l1[-1].after.objective_mean == pytest.approx(l2[-1].after.objective_mean, rel=1e-6, abs=1e-9)


# ------------------------------------------------------------------ T1e: a graded boundary layer
def _graded_square(n_thin=3, n=10, seed=2):
    ys = np.r_[0.0, np.cumsum([0.001, 0.0015, 0.0022][:n_thin]), np.linspace(0.02, 1.0, n + 1)[1:]]
    xs = np.linspace(0, 1, n + 1)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    P = np.stack([X.ravel(), Y.ravel(), 0 * X.ravel()], 1)
    idx = np.arange(len(xs) * len(ys)).reshape(len(xs), len(ys))
    q = np.stack([idx[:-1, :-1], idx[1:, :-1], idx[1:, 1:], idx[:-1, 1:]], -1).reshape(-1, 4)
    inner = (P[:, 0] > 1e-9) & (P[:, 0] < 1 - 1e-9) & (P[:, 1] > 1e-9) & (P[:, 1] < 1 - 1e-9)
    P[inner, 0] += np.random.default_rng(seed).uniform(-0.3, 0.3, inner.sum()) / n          # unevenness along the layer
    return quadmesh.QuadMesh.from_corners(P, q), idx


def test_a_thin_boundary_layer_is_kept_exactly_by_keep_layer_and_inflated_without_it():
    m, idx = _graded_square()
    seeds = np.flatnonzero(np.isclose(m.points[:, 1], 0.0))                     # the wall row
    nrm = smooth.reference_normals(m.points, m.corners)
    prot = cn.resolve(m, [cn.KeepLayer(seeds, rings=3)])
    out, logs = smooth.run_schedule(m, prot, smooth.user_schedule(outer=4), normals=nrm)
    keep = cn.layer_rings(m, seeds, 3)
    np.testing.assert_array_equal(out.points[keep], m.points[keep])             # thickness AND spacing, bit for bit
    assert logs[-1].after.n_inverted == 0
    free = cn.resolve(m, [])                                                    # only the outer boundary frozen
    out2, _ = smooth.run_schedule(m, free, smooth.user_schedule(outer=4), normals=nrm)
    thick0 = m.points[idx[:, 1], 1] - m.points[idx[:, 0], 1]
    thick1 = out2.points[idx[:, 1], 1] - out2.points[idx[:, 0], 1]
    assert np.abs(thick1[1:-1] / thick0[1:-1] - 1).max() > 0.5                  # unprotected, the layer is inflated by > 50 %


# ------------------------------------------------------------------ T4e: a box surface keeps its 8 corners and 12 creases
def _box_case(slide_edges):
    m0 = quadmesh.box(1.0, 6)
    P = np.asarray(m0.points, float)
    P = np.where(np.isclose(np.abs(P), 1.0, atol=1e-4), np.sign(P), P)          # quadmesh.box is a cube only to ~1e-6: make it exact
    m = smooth.with_points(m0, P)
    on = np.isclose(np.abs(P), 1.0)                                              # (P,3): which faces a vertex lies on
    nface = on.sum(1)
    rng = np.random.default_rng(3)
    Pj = P.copy()
    for v in range(len(P)):
        if nface[v] == 1:
            ax = int(np.flatnonzero(on[v])[0])
            free = [a for a in range(3) if a != ax]
            Pj[v, free] += rng.uniform(-0.05, 0.05, 2)
        elif nface[v] == 2 and slide_edges:
            ax = int(np.flatnonzero(~on[v])[0])                                  # the axis along the crease
            Pj[v, ax] += rng.uniform(-0.05, 0.05)
    mj = smooth.with_points(m, np.clip(Pj, -1.0, 1.0))
    cons = []
    for ax in range(3):
        for side in (-1.0, 1.0):
            ids = np.flatnonzero(on[:, ax] & (np.sign(P[:, ax]) == side) & ((nface == 1) if slide_edges else (nface >= 1)))
            normal = np.zeros(3)
            normal[ax] = 1.0
            cons.append(cn.OnSurface(ids, cn.plane(np.where(normal > 0, side, 0.0), normal)))
    if slide_edges:
        for ax in range(3):                                                      # a crease runs along `ax` where the other two coordinates are +-1
            o = [a for a in range(3) if a != ax]
            for s0 in (-1.0, 1.0):
                for s1 in (-1.0, 1.0):
                    ids = np.flatnonzero((nface >= 2) & (P[:, o[0]] == s0) & (P[:, o[1]] == s1))        # corners lie on three creases
                    pt = np.zeros(3)
                    pt[o[0]], pt[o[1]] = s0, s1
                    d = np.zeros(3)
                    d[ax] = 1.0
                    cons.append(cn.OnCurve(ids, cn.line(pt, d)))
    return m, mj, cons, nface, on


@pytest.mark.parametrize("slide_edges", [False, True])
def test_a_box_surface_keeps_flat_faces_straight_creases_and_its_eight_corners(slide_edges):
    m, mj, cons, nface, on = _box_case(slide_edges)
    # default: every face claims all its vertices, so a crease vertex (two surfaces) and a corner (three) are frozen by the
    # multi-surface rule; with slide_edges the creases are lines that slide and each corner (three lines) is frozen
    r = cn.resolve(mj, cons, boundary="free")
    corner = nface == 3
    assert corner.sum() == 8 and (r.kind[corner] == cn.FROZEN).all()
    assert (r.kind[nface == 2] == (cn.ON_CURVE if slide_edges else cn.FROZEN)).all()
    out, logs = smooth.run_schedule(mj, r, smooth.user_schedule(outer=4))
    P = out.points
    face = nface == 1
    for ax in range(3):                                                          # faces stay exactly flat, creases exactly straight
        sel = face & on[:, ax]
        assert np.abs(np.abs(P[sel, ax]) - 1.0).max() < 1e-12
    assert np.abs(P[corner]).min() > 1 - 1e-12 and np.abs(np.abs(P[corner]) - 1).max() < 1e-12
    edge = nface == 2
    assert (np.isclose(np.abs(P[edge]), 1.0).sum(1) >= 2).all()                 # still on two faces: the crease did not round off
    if slide_edges:
        assert np.abs(P[edge] - mj.points[edge]).max() > 1e-3                    # the creases' vertices really slid
    else:
        np.testing.assert_array_equal(P[edge], mj.points[edge])                  # default: multi-surface vertices frozen
    np.testing.assert_array_equal(P[corner], mj.points[corner])
    # curvature stays where the geometry has it: only the 8 corners carry angle defect, each pi/2, total 4 pi
    d = quadmesh.angle_defect(out)
    assert np.abs(d[~corner]).max() < 1e-9
    np.testing.assert_allclose(d[corner], np.pi / 2, atol=1e-9)
    assert abs(quadmesh.gauss_bonnet_residual(out)) < 1e-10
    assert logs[-1].after.n_inverted == 0 and logs[-1].after.objective_mean <= logs[0].before.objective_mean + 1e-12


# ------------------------------------------------------------------ T6: failures must be loud, not silent
def test_a_target_that_cannot_project_fails_loudly():
    m = quadmesh.QuadMesh.from_corners(*grid(4, jitter=0.2))
    bad = cn.Surface(lambda P: np.full_like(P, np.nan), lambda P: np.tile([0.0, 0.0, 1.0], (len(P), 1)), "broken")
    r = cn.resolve(m, [cn.OnSurface(np.arange(m.n_points), bad)], boundary="free")
    with pytest.raises(ValueError, match="non-finite coordinates"):
        cn.snap(m.points, r)
    with pytest.raises(ValueError, match="non-finite"):
        smooth.laplace(m, r, passes=1)


def test_a_retraction_that_is_not_a_projection_makes_the_round_fail_and_blocks_acceptance():
    m = quadmesh.QuadMesh.from_corners(*grid(6, jitter=0.2))
    drift = cn.Surface(lambda P: P + np.array([0.0, 0.0, 1e-3]), lambda P: np.tile([0.0, 0.0, 1.0], (len(P), 1)), "drifting")
    inside = np.setdiff1d(np.arange(m.n_points), quadmesh.query.boundary_points(m))
    s = SmoothSession(m, [cn.OnSurface(inside, drift)])
    p = s.propose(smooth.user_schedule(outer=1, n_opt=3))
    assert not p.report.passed and any("geometry" in f for f in p.report.failures)
    with pytest.raises(ValueError, match="failed round"):
        s.accept(p)
    assert s.round == 0 and s.current is m


def test_a_round_that_adds_inverted_quads_is_a_failure():
    m = quadmesh.QuadMesh.from_corners(*grid(6))
    s = SmoothSession(m, [])
    nrm = s.normals
    good = smooth.quality(m.points, m.corners, nrm)
    worse = good._replace(n_inverted=good.n_inverted + 2)
    warns, fails = s._verdict([], good, worse, smooth.shape_stats(m.points, m.corners),
                              smooth.shape_stats(m.points, m.corners), cn.on_target_report(m.points, m.corners, s.resolved))
    assert any("inverted quads rose" in f for f in fails)


def test_geometry_that_the_input_violates_is_reported_before_and_fixed_after():
    """An input off its target is not an error: the round reports the 'before' distance and the strict snap fixes it."""
    an, m, _ = S.jittered_case("torus")
    s = SmoothSession(m, [cn.OnSurface(np.arange(m.n_points), an.surface)], boundary="free")
    jitter = np.random.default_rng(0).normal(size=m.points.shape) * 0.002
    off = smooth.with_points(m, m.points + jitter)
    s2 = SmoothSession(off, [cn.OnSurface(np.arange(off.n_points), an.surface)], boundary="free")
    p = s2.propose(smooth.user_schedule(outer=1, n_opt=5))
    assert p.report.geometry_before.max_distance > 1e-4 and p.report.geometry_after.passed
    assert s.round == 0


def test_degenerate_corners_count_as_inverted_and_are_never_nan():
    """A quad naming a vertex twice (or a collapsed edge) used to give a NaN signed Jacobian, so the inversion gate
    read 0 inverted; now the corner reads 0 (inverted) and the exact phi is infinite."""
    from grids import grid

    from nekmeshpy.quadmesh.smoother import smooth
    P, q = grid(3)
    qd = q.copy()
    qd[0] = [q[0, 0], q[0, 1], q[0, 1], q[0, 3]]
    nrm = np.tile([0.0, 0.0, 1.0], (len(q), 1))
    m = smooth.corner_metrics(P, qd, nrm)
    assert np.isfinite(m.sj).all() and not np.isnan(m.phi).any() and np.isinf(m.phi[0]).any()
    qa = smooth.quality(P, qd, nrm)
    assert qa.n_inverted >= 1 and qa.sj_min <= 0
    assert np.isfinite(smooth.corner_metrics(P, qd, nrm, delta=0.1).phi).all()
