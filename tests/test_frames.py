"""Moving frames along a sampled curve (``nekmeshpy.core.frames``).

The load-bearing check here is ``test_rmf_matches_fixed_up_on_planar_arc``: a planar
curve has zero torsion, so the rotation-minimizing frame *must* coincide with the
fixed-up frame seeded by the plane normal.  A double-reflection implementation that is
subtly wrong (reflections in the wrong order, the wrong bisecting plane) still produces
orthonormal right-handed frames and still looks smooth -- it just twists.  That test is
what catches it.
"""

from __future__ import annotations

import numpy as np
import pytest

from nekmeshpy.core import affine, frames

# ---------------------------------------------------------------- sample curves


def _helix(k: int, turns: float = 1.5, pitch: float = 0.7) -> np.ndarray:
    t = np.linspace(0.0, 2.0 * np.pi * turns, k)
    return np.stack([np.cos(t), np.sin(t), pitch * t], axis=1)


def _arc(k: int, sweep: float = 0.5 * np.pi, radius: float = 2.0) -> np.ndarray:
    """Planar circular arc in the ``xy`` plane (``sweep = pi`` gives the U-turn)."""
    t = np.linspace(0.0, sweep, k)
    return np.stack([radius * np.cos(t), radius * np.sin(t), np.zeros_like(t)], axis=1)


def _straight(k: int) -> np.ndarray:
    s = np.linspace(0.0, 3.0, k)
    d = np.array([1.0, 2.0, -2.0]) / 3.0
    return s[:, None] * d[None, :] + np.array([0.5, -1.0, 4.0])


def _circle(k: int, radius: float = 1.0) -> np.ndarray:
    """Closed planar circle: ``k`` distinct points, no repeated wrap sample."""
    t = np.linspace(0.0, 2.0 * np.pi, k, endpoint=False)
    return np.stack([radius * np.cos(t), radius * np.sin(t), np.zeros_like(t)], axis=1)


def _wavy_loop(k: int) -> np.ndarray:
    """Closed and genuinely non-planar, with a **non-zero** holonomy.

    A symmetric lift (``z = a sin 2t`` over a circle) is closed and non-planar but its
    holonomy cancels exactly by symmetry -- measured at 1e-16, which would make this a
    vacuous test.  Breaking the symmetry in the *xy* radius as well leaves a residual of
    about -1.23 rad.
    """
    t = np.linspace(0.0, 2.0 * np.pi, k, endpoint=False)
    return np.stack([1.5 * np.cos(t), np.sin(t), 0.6 * np.sin(2.0 * t)], axis=1)


CURVES = {
    "helix": _helix(120),
    "arc90": _arc(60),
    "uturn": _arc(80, sweep=np.pi),
    "straight": _straight(30),
}


def _assert_frames(R: np.ndarray, k: int) -> None:
    assert R.shape == (k, 3, 3)
    eye = np.eye(3)[None, :, :]
    assert np.max(np.abs(np.einsum("kji,kjl->kil", R, R) - eye)) < 1e-14
    assert np.max(np.abs(np.linalg.det(R) - 1.0)) < 1e-14


# ---------------------------------------------------------------- tangents


@pytest.mark.parametrize("name", sorted(CURVES))
def test_tangents_unit_and_accurate(name: str) -> None:
    P = CURVES[name]
    T = frames.tangents(P)
    assert T.shape == P.shape
    assert np.allclose(np.linalg.norm(T, axis=1), 1.0, atol=1e-15)
    # every tangent points forward along its own chord
    fwd = np.einsum("kj,kj->k", T[:-1], np.diff(P, axis=0))
    assert np.all(fwd > 0.0)


def test_tangents_exact_on_a_straight_line() -> None:
    P = _straight(11)
    T = frames.tangents(P)
    d = np.array([1.0, 2.0, -2.0]) / 3.0
    # the three-point end rule is exact for collinear samples in exact arithmetic (its
    # weights sum to 0 and its first moment to 1); what is left is pure cancellation
    assert np.max(np.abs(T - d[None, :])) < 1e-14


def test_tangents_are_exact_on_a_uniformly_sampled_circle() -> None:
    """A chord of a uniformly sampled circle is exactly parallel to the tangent at its
    midpoint, so the interior central differences have *no* direction error at all."""
    t = np.linspace(0.0, 1.0, 41)
    P = np.stack([np.cos(t), np.sin(t), np.zeros_like(t)], axis=1)
    exact = np.stack([-np.sin(t), np.cos(t), np.zeros_like(t)], axis=1)
    assert np.max(np.linalg.norm(frames.tangents(P)[1:-1] - exact[1:-1], axis=1)) < 1e-14


def test_tangents_second_order_on_a_twisted_cubic() -> None:
    err = []
    for k in (41, 81, 161):
        t = np.linspace(0.0, 1.0, k)
        P = np.stack([t, t ** 2, t ** 3], axis=1)
        d = np.stack([np.ones_like(t), 2.0 * t, 3.0 * t ** 2], axis=1)
        exact = d / np.linalg.norm(d, axis=1)[:, None]
        # the ends are included: the three-point one-sided rule is second order too
        err.append(np.max(np.linalg.norm(frames.tangents(P) - exact, axis=1)))
    assert err[0] / err[1] > 3.5
    assert err[1] / err[2] > 3.5


def test_tangents_wrap_on_a_closed_circle() -> None:
    k = 64
    P = _circle(k)
    T = frames.tangents(P, loop=True)
    ang = np.linspace(0.0, 2.0 * np.pi, k, endpoint=False)
    exact = np.stack([-np.sin(ang), np.cos(ang), np.zeros_like(ang)], axis=1)
    # central differences on a circle give the exact direction, only the chord shortens
    assert np.max(np.linalg.norm(T - exact, axis=1)) < 1e-14
    # the wrapped field is index-equivariant: no point is an endpoint any more
    rolled = frames.tangents(np.roll(P, 7, axis=0), loop=True)
    assert np.max(np.abs(rolled - np.roll(T, 7, axis=0))) < 1e-14
    # ... which the open field is not, at either end
    open_T = frames.tangents(P, loop=False)
    assert np.linalg.norm(open_T[0] - T[0]) > 1e-5
    assert np.linalg.norm(open_T[-1] - T[-1]) > 1e-5


def test_tangents_rejects_a_repeated_point() -> None:
    P = np.array([[0.0, 0, 0], [1.0, 0, 0], [1.0, 0, 0], [2.0, 0, 0]])
    with pytest.raises(ValueError, match=r"points\[1\] and points\[2\] coincide"):
        frames.tangents(P)


def test_tangents_rejects_an_exact_cusp() -> None:
    P = np.array([[0.0, 0, 0], [1.0, 0, 0], [0.0, 0, 0]])
    with pytest.raises(ValueError, match=r"central difference at points\[1\] vanishes"):
        frames.tangents(P)


def test_tangents_rejects_bad_shapes() -> None:
    with pytest.raises(ValueError, match=r"\(K,3\)"):
        frames.tangents(np.zeros((5, 2)))
    with pytest.raises(ValueError, match="at least 2 points"):
        frames.tangents(np.zeros((1, 3)))


# ---------------------------------------------------------------- orthonormality


@pytest.mark.parametrize("name", sorted(CURVES))
def test_all_generators_are_orthonormal_right_handed(name: str) -> None:
    P = CURVES[name]
    T = frames.tangents(P)
    up = np.array([0.0, 0.0, 1.0]) if name in ("arc90", "uturn") else np.array([1.0, 0.0, 0.0])

    _assert_frames(frames.fixed_up(T, up), len(P))
    _assert_frames(frames.parallel_transport(P, T, up), len(P))
    if name != "straight":
        _assert_frames(frames.frenet(P, T), len(P))


@pytest.mark.parametrize("name", sorted(CURVES))
def test_third_column_is_the_tangent(name: str) -> None:
    P = CURVES[name]
    T = frames.tangents(P)
    up = np.array([0.0, 0.0, 1.0]) if name in ("arc90", "uturn") else np.array([1.0, 0.0, 0.0])
    for R in (frames.fixed_up(T, up), frames.parallel_transport(P, T, up)):
        assert np.max(np.abs(R[:, :, 2] - T)) < 1e-14


# ---------------------------------------------------------------- the RMF is an RMF


@pytest.mark.parametrize("sweep", [0.5 * np.pi, np.pi])
def test_rmf_matches_fixed_up_on_planar_arc(sweep: float) -> None:
    """Zero torsion => the rotation-minimizing frame *is* the fixed-up frame."""
    P = _arc(97, sweep=sweep)
    T = frames.tangents(P)
    up = np.array([0.0, 0.0, 1.0])
    err = np.max(np.abs(frames.parallel_transport(P, T, up) - frames.fixed_up(T, up)))
    assert err < 1e-12, err


def test_rmf_matches_fixed_up_on_a_tilted_planar_arc() -> None:
    """Same statement, in a plane that is not a coordinate plane."""
    P = _arc(85, sweep=np.pi)
    R, off = affine.rotation(0.7, axis=[1.0, 1.0, 0.3], center=[0.2, -0.4, 1.1])
    P = affine.apply(P, R, off)
    T = frames.tangents(P)
    up = R @ np.array([0.0, 0.0, 1.0])
    err = np.max(np.abs(frames.parallel_transport(P, T, up) - frames.fixed_up(T, up)))
    assert err < 1e-12, err


def test_rmf_is_constant_on_a_straight_segment() -> None:
    P = _straight(40)
    T = frames.tangents(P)
    R = frames.parallel_transport(P, T, np.array([0.0, 1.0, 0.0]))
    assert np.max(np.abs(R - R[0][None, :, :])) < 1e-13


def test_frenet_is_undefined_on_a_straight_segment() -> None:
    """Documented behaviour: raise, rather than emit a round-off direction or NaN."""
    P = _straight(40)
    T = frames.tangents(P)
    with pytest.raises(ValueError, match="curvature vanishes"):
        frames.frenet(P, T)


def test_frenet_flips_through_an_inflection() -> None:
    """The other Frenet failure a sweep hits: an S-bend reverses the normal by 180deg."""
    # an even sample count straddles the inflection instead of landing on it; landing on
    # it exactly is the *other* documented failure and raises (see the test above)
    t = np.linspace(-1.0, 1.0, 100)
    P = np.stack([t, t ** 3, np.zeros_like(t)], axis=1)
    T = frames.tangents(P)
    with pytest.raises(ValueError, match="curvature vanishes"):
        frames.frenet(P, np.stack([t * 0.0 + 1.0, t * 0.0, t * 0.0], axis=1))
    R = frames.frenet(P, T)
    u = R[:, :, 0]
    dots = np.einsum("kj,kj->k", u[:-1], u[1:])
    assert np.min(dots) < -0.999     # adjacent normals antiparallel across the inflection
    assert int(np.argmin(dots)) in (48, 49, 50)
    # the RMF, on the same curve, turns smoothly
    Rr = frames.parallel_transport(P, T, np.array([0.0, 0.0, 1.0]))
    ur = Rr[:, :, 0]
    assert np.min(np.einsum("kj,kj->k", ur[:-1], ur[1:])) > 0.999


def test_rmf_agrees_with_a_coarse_integrated_reference_on_a_helix() -> None:
    """First-order projection transport (project the previous normal onto the next
    normal plane) is the naive RMF integrator; double reflection must converge to it."""
    up = np.array([1.0, 0.0, 0.0])

    def gap(k: int) -> float:
        P = _helix(k)
        T = frames.tangents(P)
        got = frames.parallel_transport(P, T, up)[:, :, 0]
        ref = np.empty_like(P)
        ref[0] = got[0]
        for i in range(1, len(P)):
            w = ref[i - 1] - (ref[i - 1] @ T[i]) * T[i]
            ref[i] = w / np.linalg.norm(w)
        return float(np.max(np.linalg.norm(got - ref, axis=1)))

    g1, g2, g4 = gap(1001), gap(2001), gap(4001)
    assert g1 < 3e-3
    # the reference integrator is first order, so the two agree to O(h): halving h
    # halves the gap.  If the double reflection were transporting something *else* the
    # gap would plateau at an O(1) value instead.
    assert g1 / g2 > 1.8 and g2 / g4 > 1.8


def test_rmf_twists_less_than_frenet_on_a_helix() -> None:
    """A helix has constant non-zero torsion, so the Frenet frame spins about the
    tangent while the RMF does not -- that spin is exactly what RMF minimizes."""
    P = _helix(801)
    T = frames.tangents(P)
    up = np.array([1.0, 0.0, 0.0])

    def total_twist(R: np.ndarray) -> float:
        u = R[:, :, 0]
        # angle from u[i] to the parallel-transported u[i] measured against u[i+1]
        proj = u[:-1] - np.einsum("kj,kj->k", u[:-1], T[1:])[:, None] * T[1:]
        proj /= np.linalg.norm(proj, axis=1)[:, None]
        s = np.einsum("kj,kj->k", np.cross(proj, u[1:]), T[1:])
        c = np.einsum("kj,kj->k", proj, u[1:])
        return float(np.sum(np.abs(np.arctan2(s, c))))

    rmf = total_twist(frames.parallel_transport(P, T, up))
    fre = total_twist(frames.frenet(P, T))
    # Frenet accumulates the curve's torsion integral (~5.4 rad here) and does not
    # shrink under refinement; the RMF's residual is pure discretization and does.
    assert fre > 5.0
    assert rmf < 0.02
    assert rmf < 5e-3 * fre

    fine = _helix(3201)
    Tf = frames.tangents(fine)
    P, T = fine, Tf
    assert total_twist(frames.parallel_transport(fine, Tf, up)) < 0.4 * rmf
    assert total_twist(frames.frenet(fine, Tf)) > 5.0


def _exact_helix_tangents(k: int, turns: float = 1.5, pitch: float = 0.7) -> np.ndarray:
    t = np.linspace(0.0, 2.0 * np.pi * turns, k)
    d = np.stack([-np.sin(t), np.cos(t), pitch * np.ones_like(t)], axis=1)
    return d / np.linalg.norm(d, axis=1)[:, None]


def test_rmf_converges_under_refinement_with_exact_tangents() -> None:
    """The transport itself, isolated from the tangent estimator: sample the same helix
    at K, 2K, 4K intervals and compare at the shared parameters.  Double reflection is
    fourth-order accurate per step, and that is what shows up here (~16x per halving)."""
    up = np.array([1.0, 0.0, 0.0])

    def solve(n: int) -> np.ndarray:
        return frames.parallel_transport(_helix(n + 1), _exact_helix_tangents(n + 1), up)

    e1 = np.max(np.abs(solve(50) - solve(100)[::2]))
    e2 = np.max(np.abs(solve(100) - solve(200)[::2]))
    e3 = np.max(np.abs(solve(200) - solve(400)[::2]))
    assert e1 > e2 > e3
    assert e1 / e2 > 8.0 and e2 / e3 > 8.0, (e1, e2, e3)


def test_rmf_converges_second_order_with_estimated_tangents() -> None:
    """End to end, tangent estimator included.  The interior central difference and the
    three-point end rule are both second order, so the whole pipeline is -- which is why
    the end rule has to be three-point: the plain forward difference at ``points[0]``
    seeds the transport with an O(h) error that is then carried along the *entire* curve
    and pins the result at first order no matter how fine the sampling."""
    up = np.array([1.0, 0.0, 0.0])

    def solve(n: int) -> np.ndarray:
        P = _helix(n + 1)
        return frames.parallel_transport(P, frames.tangents(P), up)

    e1 = np.max(np.abs(solve(50) - solve(100)[::2]))
    e2 = np.max(np.abs(solve(100) - solve(200)[::2]))
    e3 = np.max(np.abs(solve(200) - solve(400)[::2]))
    assert e1 > e2 > e3
    assert e1 / e2 > 3.4 and e2 / e3 > 3.4, (e1, e2, e3)


# ---------------------------------------------------------------- loops / holonomy


def test_holonomy_vanishes_on_a_planar_closed_circle() -> None:
    P = _circle(96)
    T = frames.tangents(P, loop=True)
    assert abs(frames.holonomy(P, T, np.array([0.0, 0.0, 1.0]))) < 1e-13


def test_holonomy_is_real_on_a_non_planar_closed_curve() -> None:
    P = _wavy_loop(400)
    T = frames.tangents(P, loop=True)
    theta = frames.holonomy(P, T, np.array([0.0, 0.0, 1.0]))
    assert abs(theta) > 1e-2
    # it is a property of the curve, not of the sampling: refining converges it
    fine = _wavy_loop(1600)
    theta_f = frames.holonomy(fine, frames.tangents(fine, loop=True),
                              np.array([0.0, 0.0, 1.0]))
    assert abs(theta - theta_f) < 1e-3 * abs(theta) + 1e-4
    # ... nor of the seed
    alt = frames.holonomy(P, T, np.array([0.3, -0.9, 0.25]))
    assert abs(theta - alt) < 1e-10


def _per_element_twist(P: np.ndarray, T: np.ndarray, R: np.ndarray) -> np.ndarray:
    """Signed twist of each of the ``K`` loop segments: the angle between the
    double-reflection transport of ``u[i]`` and ``u[i+1 mod K]``, about the tangent."""
    k = len(P)
    out = np.empty(k)
    for i in range(k):
        j = (i + 1) % k
        v1 = P[j] - P[i]
        c1 = v1 @ v1
        u = R[i, :, 0]
        rl = u - (2.0 / c1) * (v1 @ u) * v1
        tl = T[i] - (2.0 / c1) * (v1 @ T[i]) * v1
        v2 = T[j] - tl
        c2 = v2 @ v2
        carried = rl if c2 == 0.0 else rl - (2.0 / c2) * (v2 @ rl) * v2
        nxt = R[j, :, 0]
        out[i] = np.arctan2(np.cross(carried, nxt) @ T[j], carried @ nxt)
    return out


def test_loop_distribution_spreads_the_holonomy_evenly() -> None:
    """``distribute=True`` does not remove the holonomy -- nothing can, it is the
    curve's geometry.  It moves it: every one of the ``K`` elements carries ``-theta/K``
    instead of the seam element carrying all of it."""
    P = _wavy_loop(300)
    T = frames.tangents(P, loop=True)
    up = np.array([0.0, 0.0, 1.0])
    theta = frames.holonomy(P, T, up)
    k = len(P)

    raw = frames.parallel_transport(P, T, up, loop=True, distribute=False)
    dist = frames.parallel_transport(P, T, up, loop=True, distribute=True)
    _assert_frames(raw, k)
    _assert_frames(dist, k)

    # raw: twist-free everywhere, with the whole residual dumped into the seam element
    tw_raw = _per_element_twist(P, T, raw)
    assert np.max(np.abs(tw_raw[:-1])) < 1e-13
    assert abs(tw_raw[-1] + theta) < 1e-12
    assert abs(theta) > 1.0

    # distributed: the same total, spread uniformly -- the seam is no longer special
    tw = _per_element_twist(P, T, dist)
    assert np.max(np.abs(tw + theta / k)) < 1e-12
    assert abs(np.sum(tw) + theta) < 1e-11
    assert np.max(np.abs(tw)) < 1.01 * np.max(np.abs(tw_raw))

    # the frame field itself is single-valued: it is the same array either way at k=0
    assert np.max(np.abs(dist[0] - raw[0])) < 1e-15


def test_loop_distribution_is_a_no_op_on_a_planar_circle() -> None:
    P = _circle(64)
    T = frames.tangents(P, loop=True)
    up = np.array([0.0, 0.0, 1.0])
    raw = frames.parallel_transport(P, T, up, loop=True, distribute=False)
    dist = frames.parallel_transport(P, T, up, loop=True, distribute=True)
    assert np.max(np.abs(raw - dist)) < 1e-14
    assert np.max(np.abs(dist - frames.fixed_up(T, up))) < 1e-13


# ---------------------------------------------------------------- error cases


def test_up0_parallel_to_the_first_tangent_is_rejected() -> None:
    P = _straight(10)
    T = frames.tangents(P)
    with pytest.raises(ValueError, match="parallel to tangents"):
        frames.parallel_transport(P, T, T[0])
    with pytest.raises(ValueError, match="parallel to tangents"):
        frames.parallel_transport(P, T, -T[0])


def test_fixed_up_rejects_a_tangent_parallel_to_up() -> None:
    """A path that turns into the ``up`` direction: the elbow's own axis becomes 'up'."""
    # a quarter turn in the xz plane sampled so that t = pi/2 lands on point 10, where
    # the path points straight along +z: the central difference is exact there
    t = np.linspace(0.0, np.pi, 21)
    P = np.stack([np.sin(t), np.zeros_like(t), -np.cos(t)], axis=1)
    T = frames.tangents(P)
    assert np.max(np.abs(T[10] - np.array([0.0, 0.0, 1.0]))) < 1e-15
    with pytest.raises(ValueError, match=r"tangents\[10\] .* is parallel to up"):
        frames.fixed_up(T, np.array([0.0, 0.0, 1.0]))
    # ... and the RMF handles the same path fine
    _assert_frames(frames.parallel_transport(P, T, np.array([0.0, 1.0, 0.0])), len(P))


def test_generators_reject_non_unit_or_mismatched_tangents() -> None:
    P = _arc(10)
    T = frames.tangents(P)
    with pytest.raises(ValueError, match="must be unit vectors"):
        frames.fixed_up(2.0 * T, np.array([0.0, 0.0, 1.0]))
    with pytest.raises(ValueError, match=r"\(K,3\) matching the 10 points"):
        frames.parallel_transport(P, T[:5], np.array([0.0, 0.0, 1.0]))


def test_zero_and_malformed_up_vectors_are_rejected() -> None:
    P = _arc(10)
    T = frames.tangents(P)
    with pytest.raises(ValueError, match="non-zero direction"):
        frames.fixed_up(T, np.zeros(3))
    with pytest.raises(ValueError, match=r"\(3,\) direction"):
        frames.fixed_up(T, np.array([0.0, 0.0]))


# ---------------------------------------------------------------- frame_transform


def test_frame_transform_identity() -> None:
    P = _helix(20)
    R = frames.parallel_transport(P, frames.tangents(P), np.array([1.0, 0.0, 0.0]))
    M, off = frames.frame_transform(R[7], P[7], R[7], P[7])
    assert np.max(np.abs(M - np.eye(3))) < 1e-15
    # ``offset = o - (R R^T) o`` cancels to round-off *relative to the coordinate*
    assert np.max(np.abs(off)) < 1e-15 * (1.0 + np.max(np.abs(P[7])))
    pts = np.array([[0.3, -1.2, 5.0], [1.0, 1.0, 1.0]])
    assert np.max(np.abs(affine.apply(pts, M, off) - pts)) < 1e-14


def test_frame_transform_places_local_coordinates() -> None:
    """The consumer's actual call: a section authored in the world frame at the origin,
    placed onto station ``k`` of the path."""
    P = _helix(60)
    T = frames.tangents(P)
    R = frames.parallel_transport(P, T, np.array([1.0, 0.0, 0.0]))
    world = np.eye(3)
    origin = np.zeros(3)

    for k in (0, 17, 59):
        M, off = frames.frame_transform(world, origin, R[k], P[k])
        # the local axes land on the frame's columns, offset to the station
        assert np.max(np.abs(affine.apply(np.eye(3), M, off) - (R[k].T + P[k]))) < 1e-14
        # the local origin lands on the station itself
        assert np.max(np.abs(affine.apply(origin, M, off) - P[k])) < 1e-14
        # a section in the local xy plane lands in the cross-section plane, i.e.
        # perpendicular to the tangent
        sect = np.array([[0.4, 0.0, 0.0], [0.0, -0.9, 0.0], [0.25, 0.25, 0.0]])
        placed = affine.apply(sect, M, off)
        assert np.max(np.abs((placed - P[k]) @ T[k])) < 1e-14


def test_frame_transform_is_rigid() -> None:
    P = _wavy_loop(50)
    T = frames.tangents(P, loop=True)
    R = frames.parallel_transport(P, T, np.array([0.0, 0.0, 1.0]), loop=True)
    M, off = frames.frame_transform(R[3], P[3], R[29], P[29])

    assert np.max(np.abs(M @ M.T - np.eye(3))) < 1e-14
    assert abs(np.linalg.det(M) - 1.0) < 1e-14
    rng = np.random.default_rng(0)
    pts = rng.normal(size=(12, 3))
    moved = affine.apply(pts, M, off)
    d0 = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=2)
    d1 = np.linalg.norm(moved[:, None, :] - moved[None, :, :], axis=2)
    assert np.max(np.abs(d0 - d1)) < 1e-13


def test_frame_transform_composes_back() -> None:
    P = _helix(30)
    R = frames.parallel_transport(P, frames.tangents(P), np.array([1.0, 0.0, 0.0]))
    fwd = frames.frame_transform(R[2], P[2], R[21], P[21])
    bwd = frames.frame_transform(R[21], P[21], R[2], P[2])
    pts = np.array([[1.0, 2.0, 3.0], [-0.5, 0.0, 0.25]])
    back = affine.apply(affine.apply(pts, *fwd), *bwd)
    assert np.max(np.abs(back - pts)) < 1e-14


def test_frame_transform_rejects_bad_shapes() -> None:
    R = np.eye(3)
    o = np.zeros(3)
    with pytest.raises(ValueError, match=r"R_from must be a \(3,3\)"):
        frames.frame_transform(np.eye(2), o, R, o)
    with pytest.raises(ValueError, match=r"origin_to must be a \(3,\)"):
        frames.frame_transform(R, o, R, np.zeros(2))
