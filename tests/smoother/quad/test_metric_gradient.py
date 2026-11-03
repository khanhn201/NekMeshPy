"""M1 of the quad smoother: the corner metric, the objective and its analytic gradient."""

import numpy as np
import pytest
from grids import grid as _grid

from nekmeshpy import quadmesh
from nekmeshpy.quadmesh.smoother import smooth
from nekmeshpy.tags import Tags


def _fd(P, quads, normals, **kw):
    g = np.zeros_like(P)
    h = 1e-6
    for i in range(P.shape[0]):
        for k in range(3):
            q = P.copy()
            q[i, k] += h
            m = P.copy()
            m[i, k] -= h
            g[i, k] = (smooth.objective(q, quads, normals, **kw)
                       - smooth.objective(m, quads, normals, **kw)) / (2 * h)
    return g


def test_square_is_the_ideal():
    P, q = _grid(4)
    n = smooth.reference_normals(P, q)
    m = smooth.corner_metrics(P, q, n)
    np.testing.assert_allclose(m.phi, 1.0, atol=1e-12)
    np.testing.assert_allclose(m.sj, 1.0, atol=1e-12)
    assert smooth.objective(P, q, n) == pytest.approx(0.0, abs=1e-10)
    np.testing.assert_allclose(smooth.gradient(P, q, n), 0.0, atol=1e-10)


@pytest.mark.parametrize("r", [2.0, 3.0, 10.0])
def test_rectangle_and_rhombus_closed_forms(r):
    P = np.array([[0, 0, 0], [r, 0, 0], [r, 1, 0], [0, 1, 0]], float)
    q = np.array([[0, 1, 2, 3]])
    m = smooth.corner_metrics(P, q, smooth.reference_normals(P, q))
    np.testing.assert_allclose(m.phi, ((r + 1 / r) / 2) ** 2, rtol=1e-12)
    th = np.deg2rad(60.0)
    R = np.array([[0, 0, 0], [1, 0, 0], [1 + np.cos(th), np.sin(th), 0], [np.cos(th), np.sin(th), 0]])
    m = smooth.corner_metrics(R, q, smooth.reference_normals(R, q))
    np.testing.assert_allclose(m.phi, 1 / np.sin(th) ** 2, rtol=1e-12)


def test_shape_only_invariances_and_orientation_blindness():
    P, q = _grid(5, jitter=0.3)
    n = smooth.reference_normals(P, q)
    base = smooth.corner_metrics(P, q, n)
    c, s = np.cos(0.7), np.sin(0.7)
    Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    Q = 3.7 * P @ Rz.T + np.array([1.0, -2.0, 5.0])
    moved = smooth.corner_metrics(Q, q, smooth.reference_normals(Q, q))
    np.testing.assert_allclose(moved.phi, base.phi, rtol=1e-10)
    np.testing.assert_allclose(moved.sj, base.sj, rtol=1e-10)
    flipped = smooth.corner_metrics(P, q, -n)          # reversed reference normal
    np.testing.assert_allclose(flipped.phi, base.phi, rtol=1e-12)
    np.testing.assert_allclose(flipped.sj, -base.sj, rtol=1e-12)


def test_inverted_quad_is_signed_negative_and_penalised():
    P, q = _grid(3)
    n = smooth.reference_normals(P, q)
    bad = P.copy()
    bad[5] = [0.9, 0.9, 0.0]                           # push an interior vertex past its neighbours
    m = smooth.corner_metrics(bad, q, n)
    assert m.sj.min() < 0
    assert smooth.objective(bad, q, n) > 100 * smooth.EPS


@pytest.mark.parametrize("kind", ["plane", "pushed", "sphere"])
def test_gradient_matches_central_differences(kind):
    P, q = _grid(5, jitter=0.5, seed=3)
    if kind == "pushed":                               # penalty branch on
        P[8] = [0.55, 0.52, 0.0]
        P[9] = [0.5, 0.5, 0.0]
    normals = smooth.reference_normals(P, q)
    if kind == "sphere":
        u, v = (P[:, 0] - 0.5) * 1.2, (P[:, 1] - 0.5) * 1.2
        P = np.stack([np.sin(u) * np.cos(v), np.sin(v), np.cos(u) * np.cos(v)], 1)
        normals = P[q].mean(axis=1)
        normals /= np.linalg.norm(normals, axis=1, keepdims=True)
        # CCW seen from outside needs the winding to agree with the radial normal
        if np.einsum("qj,qj->q", smooth.reference_normals(P, q), normals).mean() < 0:
            q = q[:, ::-1].copy()
    if kind == "pushed":
        assert (smooth.corner_metrics(P, q, normals).sj < smooth.EPS).any()
    ga = smooth.gradient(P, q, normals)
    gf = _fd(P, q, normals)
    np.testing.assert_allclose(ga, gf, rtol=1e-5, atol=1e-6 * max(1.0, np.abs(gf).max()))


@pytest.mark.parametrize("delta", [0.01, 0.1])
def test_barrier_free_gradient_matches_central_differences(delta):
    P, q = _grid(5, jitter=0.5, seed=3)
    P[8] = [0.55, 0.52, 0.0]
    P[9] = [0.5, 0.5, 0.0]
    n = smooth.reference_normals(P, q)
    ga = smooth.gradient(P, q, n, delta=delta)
    g = np.zeros_like(P)
    h = 1e-7
    for i in range(P.shape[0]):
        for k in range(3):
            a = P.copy()
            a[i, k] += h
            b = P.copy()
            b[i, k] -= h
            g[i, k] = (smooth.objective(a, q, n, delta=delta)
                       - smooth.objective(b, q, n, delta=delta)) / (2 * h)
    np.testing.assert_allclose(ga, g, rtol=1e-4, atol=1e-5 * max(1.0, np.abs(g).max()))


def test_barrier_free_metric_is_finite_through_inversion_and_one_at_a_square():
    P, q = _grid(4)
    n = smooth.reference_normals(P, q)
    np.testing.assert_allclose(smooth.corner_metrics(P, q, n, delta=0.01).phi, 1.0, atol=1e-12)
    bad = P.copy()
    bad[6] = [0.9, 0.9, 0.0]                  # inverts neighbouring quads
    m = smooth.corner_metrics(bad, q, n, delta=0.01)
    assert (smooth.corner_metrics(bad, q, n).sj < 0).any()
    assert np.isfinite(m.phi).all() and m.phi.min() >= 1.0 - 1e-12
    exact = smooth.corner_metrics(P + 0.0, q, n)
    pos = smooth.corner_metrics(_grid(4, jitter=0.2, seed=1)[0], q, n)
    d = smooth.corner_metrics(_grid(4, jitter=0.2, seed=1)[0], q, n, delta=1e-6)
    ok = pos.sj > 0.2                                          # away from the wall the two agree as delta -> 0
    np.testing.assert_allclose(d.phi[ok], pos.phi[ok], rtol=1e-4)
    assert exact.phi.min() >= 1.0 - 1e-12


def test_gradient_obeys_the_two_symmetries():
    """F is invariant to translation and to uniform scaling, so sum(g)=0 and sum(x.g)=0."""
    P, q = _grid(6, jitter=0.5, seed=7)
    n = smooth.reference_normals(P, q)
    g = smooth.gradient(P, q, n)
    np.testing.assert_allclose(g.sum(axis=0), 0.0, atol=1e-9 * np.abs(g).max())
    assert abs((P * g).sum()) < 1e-9 * np.abs(P * g).sum()


def test_objective_gradient_scale_does_not_depend_on_element_count():
    """A vertex's gradient depends only on its incident quads, so refining a patch of the same shape
    leaves the *per-vertex* gradient of a corresponding interior vertex unchanged."""
    def vertex_grad(n):
        P, q = _grid(n)
        P[(n + 1) * (n // 2) + n // 2, :2] += [0.2 / n, 0.1 / n]
        nm = smooth.reference_normals(P, q)
        return smooth.gradient(P, q, nm)[(n + 1) * (n // 2) + n // 2] / n   # dF/dx scales as 1/h, so g*h is the invariant
    np.testing.assert_allclose(vertex_grad(6), vertex_grad(12), rtol=1e-8)


def test_with_points_is_a_new_mesh_and_keeps_every_tag():
    P, q = _grid(3)
    m = quadmesh.QuadMesh.from_corners(P, q, Tags.full(q.shape[0], "fluid"))
    before = m.points.copy()
    moved = smooth.with_points(m, P + 0.01)
    np.testing.assert_array_equal(m.points, before)             # input untouched
    np.testing.assert_allclose(moved.points, P + 0.01)
    np.testing.assert_array_equal(moved.corners, m.corners)
    assert list(moved.element_tags.to_dense()) == list(m.element_tags.to_dense())
    assert moved.points is not m.points


def test_with_points_refuses_high_order():
    ring = quadmesh.rectangle(np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], float), 2, 2, order=2)
    with pytest.raises(ValueError, match="vertices only"):
        smooth.with_points(ring, ring.points)
