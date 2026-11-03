"""The hex smoother's corner metric, its objective and its analytic gradient."""

import numpy as np
import pytest
from hexblocks import block, block_arrays

from nekmeshpy import hexmesh, linemesh, quadmesh
from nekmeshpy.hexmesh.quality import corner_scaled_jacobian
from nekmeshpy.hexmesh.smoother import smooth

_UNIT = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
                  [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]], float)
_ONE = np.arange(8)[None, :]


def _fd(P, H, h=1e-6, **kw):
    g = np.zeros_like(P)
    for i in range(P.shape[0]):
        for k in range(3):
            a, b = P.copy(), P.copy()
            a[i, k] += h
            b[i, k] -= h
            g[i, k] = (smooth.objective(a, H, **kw) - smooth.objective(b, H, **kw)) / (2 * h)
    return g


def test_a_cube_is_the_ideal():
    P, H = block_arrays(3)
    m = smooth.corner_metrics(P, H)
    np.testing.assert_allclose(m.phi, 1.0, atol=1e-12)
    np.testing.assert_allclose(m.sj, 1.0, atol=1e-12)
    assert smooth.objective(P, H) == pytest.approx(0.0, abs=1e-10)
    np.testing.assert_allclose(smooth.gradient(P, H), 0.0, atol=1e-10)


@pytest.mark.parametrize("r", [2.0, 3.0, 10.0])
def test_box_and_sheared_closed_forms(r):
    P = _UNIT * [r, 1.0, 1.0]
    m = smooth.corner_metrics(P, _ONE)
    np.testing.assert_allclose(m.phi, ((r * r + 2.0) / 3.0) ** 3 / r ** 2, rtol=1e-12)
    np.testing.assert_allclose(m.sj, 1.0, rtol=1e-12)
    th = np.deg2rad(60.0)
    S = _UNIT.copy()
    S[:, 0] += S[:, 1] * np.cos(th)
    S[:, 1] *= np.sin(th)                      # a rhombic prism: unit edges, 60 deg in the base
    m = smooth.corner_metrics(S, _ONE)
    np.testing.assert_allclose(np.abs(m.sj), np.sin(th), rtol=1e-12)
    np.testing.assert_allclose(m.phi, 1.0 / np.sin(th) ** 2, rtol=1e-12)


def test_the_signed_reading_is_hexmesh_corner_scaled_jacobian():
    P, H = block_arrays(3, jitter=0.4, seed=2)
    np.testing.assert_allclose(smooth.corner_metrics(P, H).sj.min(axis=1), corner_scaled_jacobian(P, H),
                               rtol=1e-12, atol=1e-14)


def test_shape_only_invariances_and_orientation_blindness():
    P, H = block_arrays(3, jitter=0.3, seed=5)
    base = smooth.corner_metrics(P, H)
    ax = np.array([1.0, 2.0, -0.5]) / np.linalg.norm([1.0, 2.0, -0.5])
    K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
    R = np.eye(3) + np.sin(0.9) * K + (1 - np.cos(0.9)) * K @ K
    moved = smooth.corner_metrics(4.2 * P @ R.T + [1.0, -3.0, 2.0], H)
    np.testing.assert_allclose(moved.phi, base.phi, rtol=1e-9)
    np.testing.assert_allclose(moved.sj, base.sj, rtol=1e-9, atol=1e-13)
    mirrored = smooth.corner_metrics(P * [-1.0, 1.0, 1.0], H)       # a reflection inverts every corner
    np.testing.assert_allclose(mirrored.phi, base.phi, rtol=1e-10)
    np.testing.assert_allclose(mirrored.sj, -base.sj, rtol=1e-10, atol=1e-13)


def test_an_inverted_hex_is_signed_negative_and_penalised():
    P, H = block_arrays(2)
    bad = P.copy()
    bad[13] = [0.95, 0.9, 0.9]                 # the centre vertex pushed past its neighbours
    assert smooth.corner_metrics(bad, H).sj.min() < 0
    assert smooth.objective(bad, H) > 100 * smooth.EPS


@pytest.mark.parametrize("kind", ["jittered", "penalty"])
def test_gradient_matches_central_differences(kind):
    P, H = block_arrays(2, jitter=0.35, seed=3)
    if kind == "penalty":
        P[13] = [0.8, 0.75, 0.7]
        assert (smooth.corner_metrics(P, H).sj < smooth.EPS).any()
    ga = smooth.gradient(P, H)
    gf = _fd(P, H)
    np.testing.assert_allclose(ga, gf, rtol=1e-5, atol=1e-6 * max(1.0, np.abs(gf).max()))


@pytest.mark.parametrize("delta", [0.01, 0.1])
def test_barrier_free_gradient_matches_central_differences(delta):
    P, H = block_arrays(2, jitter=0.35, seed=3)
    P[13] = [0.95, 0.9, 0.9]                   # inverted: only the barrier-free metric is finite here
    ga = smooth.gradient(P, H, delta=delta)
    gf = _fd(P, H, h=1e-7, delta=delta)
    np.testing.assert_allclose(ga, gf, rtol=1e-4, atol=1e-5 * max(1.0, np.abs(gf).max()))


def test_barrier_free_metric_is_finite_through_inversion_and_one_at_a_cube():
    P, H = block_arrays(2)
    np.testing.assert_allclose(smooth.corner_metrics(P, H, delta=0.01).phi, 1.0, atol=1e-12)
    bad = P.copy()
    bad[13] = [0.95, 0.9, 0.9]
    m = smooth.corner_metrics(bad, H, delta=0.01)
    assert np.isfinite(m.phi).all() and m.phi.min() >= 1.0 - 1e-12
    Q = block_arrays(2, jitter=0.2, seed=1)[0]
    exact, near = smooth.corner_metrics(Q, H), smooth.corner_metrics(Q, H, delta=1e-6)
    ok = exact.sj > 0.2
    np.testing.assert_allclose(near.phi[ok], exact.phi[ok], rtol=1e-4)


def test_gradient_obeys_translation_and_scale_symmetry():
    P, H = block_arrays(3, jitter=0.3, seed=7)
    g = smooth.gradient(P, H)
    np.testing.assert_allclose(g.sum(axis=0), 0.0, atol=1e-9 * np.abs(g).max())
    assert abs((P * g).sum()) < 1e-9 * np.abs(P * g).sum()


def test_with_points_is_a_new_mesh_and_keeps_every_tag():
    m = block(2, tags=["fluid"] * 4 + ["solid"] * 4)
    m = hexmesh.tag_faces(m, np.flatnonzero(hexmesh.boundary_face_ids(m))[:3], "wall")
    before = m.points.copy()
    moved = smooth.with_points(m, m.points + 0.01)
    np.testing.assert_array_equal(m.points, before)
    np.testing.assert_allclose(moved.points, before + 0.01)
    np.testing.assert_array_equal(moved.corners, m.corners)
    assert list(moved.element_tags.to_dense()) == list(m.element_tags.to_dense())
    assert list(moved.face_tags.to_dense()) == list(m.face_tags.to_dense())


def test_high_order_is_refused_with_the_linear_shadow_route():
    from nekmeshpy.hexmesh.smoother import linearize, resolve
    sec = quadmesh.ogrid(linemesh.circle(1.0, 8, order=2), 2, 1)
    cyl = hexmesh.extrude(sec, 1.0, 2)
    assert cyl.order == 2
    with pytest.raises(ValueError, match="linearize"):
        smooth.with_points(cyl, cyl.points)
    lin = linearize(cyl)
    assert lin.order == 1 and np.array_equal(lin.corners, cyl.corners)
    np.testing.assert_array_equal(lin.points, cyl.points)
    r = resolve(cyl, [])
    with pytest.raises(ValueError, match="linear"):
        smooth.laplace(cyl, r)
    out, _ = smooth.laplace(lin, r, passes=1)
    assert out.order == 1
