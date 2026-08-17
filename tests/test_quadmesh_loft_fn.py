"""Unit tests for ``QuadMesh.loft_fn`` -- ``QuadMesh.loft`` with the profiles
**evaluated** from a parametrization instead of handed in -- and for the
``sweep_nodes`` / per-layer ``element_tags`` arguments it delegates through.

The contract, one rung up from ``LineMesh.loft_fn``: a plain ``loft`` sees only the
corner-level profiles, so at ``order > 1`` it subdivides the *sweep* straight, and a
swept curved surface comes out high-order in storage and linear in geometry.  Lofting a
torus from **exact** rings still puts its interior nodes tens of percent of the tube
radius off the true surface.  ``loft_fn`` evaluates the profiles at the intermediate
GLL levels too, so every node is a genuine profile point and the surface is exact.

The load-bearing regression here is the *other* direction: ``loft`` with
``sweep_nodes=None`` must stay bit-identical to what it has always produced, which is
what keeps the goldens frozen.
"""

import numpy as np
import pytest

from nekmeshpy import ElementTags, linemesh, quadmesh
from nekmeshpy.core import conform

R, RT, NU, NV = 2.0, 0.6, 8, 6


def _tube_ring(order):
    """One tube cross-section of the torus, sitting at ``theta = 0``."""
    return linemesh.circle(RT, NU, center=(R, 0.0, 0.0), normal=(0, 1, 0),
                           order=order)


def _torus_f(order):
    """The torus as a family of profiles: the same ring, *placed* by rotation.

    Placing rather than rebuilding is the idiom the docstring recommends -- the
    affine ops move no index, so every profile stays index-paired with the first.
    """
    base = _tube_ring(order)
    return lambda t: linemesh.rotate(base, t, axis=(0, 0, 1))


def _ring_fractions(n=NV):
    """``n+1`` values whose last is the wrap back onto the first profile."""
    return np.linspace(0.0, 2.0 * np.pi, n + 1)


def _nodes(qm):
    """Every conformal node of the section, corners and high-order alike."""
    nodes, _ = conform.conformal_quad(qm.points, qm.corners, qm.quads, qm.orient,
                                      qm.line_mesh.interior, qm.interior, qm.order)
    return nodes


def _tube_deviation(qm):
    """Max distance of any node from the true torus tube surface."""
    x, y, z = _nodes(qm).T
    return float(np.max(np.abs(np.hypot(np.hypot(x, y) - R, z) - RT)))


# -- the defect this factory closes -----------------------------------------

@pytest.mark.parametrize("order", [2, 3, 4])
def test_plain_loft_of_exact_rings_is_straight_along_the_sweep(order):
    """The baseline: exact input profiles do *not* give an exact swept surface."""
    f = _torus_f(order)
    fr = _ring_fractions()
    straight = quadmesh.loft([f(t) for t in fr[:-1]], loop=True)
    # off by a sizeable fraction of the tube radius -- not float noise
    assert _tube_deviation(straight) > 0.1 * RT


@pytest.mark.parametrize("order", [1, 2, 3, 4])
def test_every_node_lies_on_the_true_torus(order):
    torus = quadmesh.loft_fn(_torus_f(order), _ring_fractions(),
                             loop=True, order=order)
    assert _tube_deviation(torus) < 1e-13


@pytest.mark.parametrize("order", [1, 2, 3])
def test_corners_are_exact_at_every_order(order):
    """Corners were always exact; the fix must not disturb them."""
    torus = quadmesh.loft_fn(_torus_f(order), _ring_fractions(),
                             loop=True, order=order)
    x, y, z = torus.points.T
    assert np.max(np.abs(np.hypot(np.hypot(x, y) - R, z) - RT)) < 1e-14


# -- the closed sweep --------------------------------------------------------

@pytest.mark.parametrize("order", [1, 2, 3])
def test_loop_gives_a_closed_surface_with_no_duplicated_layer(order):
    torus = quadmesh.loft_fn(_torus_f(order), _ring_fractions(),
                             loop=True, order=order)
    assert torus.n_points == NU * NV        # no seam profile duplicated
    assert torus.n_quads == NU * NV         # NV layers, not NV-1
    # a torus is closed: every edge is shared by exactly two quads
    edges, elem_edges, _ = conform.unique_edges(torus.corners, 2)
    counts = np.zeros(edges.shape[0], dtype=np.int64)
    np.add.at(counts, elem_edges.ravel(), 1)
    assert np.all(counts == 2)


def test_loop_rejects_a_family_that_does_not_close():
    base = _tube_ring(2)
    # a full turn *plus* a bit: f(fr[-1]) does not land back on f(fr[0])
    fr = np.linspace(0.0, 2.0 * np.pi + 0.3, NV + 1)
    with pytest.raises(ValueError, match="map back to the first profile"):
        quadmesh.loft_fn(lambda t: linemesh.rotate(base, t, axis=(0, 0, 1)), fr,
                         loop=True, order=2)


def test_loop_places_end_caps_on_the_seam():
    """A closed sweep has no free end, but its seam is a real side -- a cap tag names
    that side of it rather than being refused."""
    qm = quadmesh.loft_fn(_torus_f(1), _ring_fractions(), loop=True, first_tag="in")
    assert qm.edge_tags.count("in") > 0


def test_loop_rejects_fewer_than_three_fractions():
    with pytest.raises(ValueError, match="at least 3 fractions"):
        quadmesh.loft_fn(_torus_f(1), np.array([0.0, 2.0 * np.pi]), loop=True)


def test_needs_at_least_two_fractions():
    with pytest.raises(ValueError, match="at least 2 fractions"):
        quadmesh.loft_fn(_torus_f(1), np.array([0.0]))


# -- order 1 and the open sweep are the plain loft ---------------------------

@pytest.mark.parametrize("loop", [False, True])
def test_order_one_equals_a_plain_loft_of_the_same_profiles(loop):
    f = _torus_f(1)
    fr = _ring_fractions()
    got = quadmesh.loft_fn(f, fr, loop=loop, order=1)
    want = quadmesh.loft([f(t) for t in (fr[:-1] if loop else fr)], loop=loop)
    assert np.array_equal(got.points, want.points)
    assert np.array_equal(got.corners, want.corners)
    assert np.array_equal(got.quads, want.quads)
    assert np.array_equal(got.orient, want.orient)


def test_open_sweep_has_the_expected_shape_and_caps():
    f = _torus_f(2)
    fr = np.linspace(0.0, np.pi / 2, 4)
    sec = quadmesh.loft_fn(f, fr, order=2,
                           first_tag="start", last_tag="end")
    assert sec.n_points == NU * 4
    assert sec.n_quads == NU * 3
    assert _tube_deviation(sec) < 1e-13
    assert set(sec.edge_group_tags) == {"start", "end"}


# -- grading -----------------------------------------------------------------

def test_grading_is_honored_per_layer():
    """A non-uniform sweep grading must place each layer's interior nodes inside
    that layer's own parameter span -- and still on the true surface."""
    order = 3
    f = _torus_f(order)
    fr = np.array([0.0, 0.3, 2.4, 2.0 * np.pi])
    sec = quadmesh.loft_fn(f, fr, order=order)
    assert _tube_deviation(sec) < 1e-13
    # the corner levels sit exactly at the requested parameters
    theta = np.arctan2(sec.points[:, 1], sec.points[:, 0])
    theta[theta < -1e-12] += 2.0 * np.pi
    for want in fr:
        assert np.min(np.abs(theta - (want % (2.0 * np.pi)))) < 1e-12


# -- tags --------------------------------------------------------------------

def test_element_tags_name_the_swept_column_of_each_profile_line():
    """``element_tags`` is per *profile line*: line l's tag lands on every quad swept
    from it, at every layer -- and a single string names the whole section."""
    base = linemesh.circle(RT, NU, center=(R, 0.0, 0.0), normal=(0, 1, 0))
    f = lambda t: linemesh.rotate(base, t, axis=(0, 0, 1))                   # noqa: E731
    per_line = ElementTags.from_dense(["hot"] + [""] * (NU - 1))
    sec = quadmesh.loft_fn(f, np.linspace(0.0, 1.0, 4), element_tags=per_line)
    tags = sec.element_tags.dense(sec.n_quads).reshape(3, NU)      # quad (layer i, line l) = i*NU + l
    assert list(np.unique(tags[:, 0])) == ["hot"]
    assert list(np.unique(tags[:, 1:])) == [""]
    assert quadmesh.loft_fn(f, np.linspace(0.0, 1.0, 4),
                            element_tags="fluid").element_group_tags == ["fluid"]


def test_loft_rejects_element_tags_naming_a_line_the_profile_lacks():
    base = _tube_ring(1)
    slices = [linemesh.translate(base, (0.0, 0.0, z)) for z in (0.0, 1.0, 2.0)]
    with pytest.raises(ValueError, match="only %d elements" % base.n_lines):
        quadmesh.loft(slices, element_tags=ElementTags([base.n_lines], ["off"]))
    with pytest.raises(TypeError, match="element_tags must be"):
        quadmesh.loft(slices, element_tags=["a", "b"])


# -- validation of what f returns --------------------------------------------

def test_rejects_a_profile_of_the_wrong_order():
    base = _tube_ring(1)
    with pytest.raises(ValueError, match="order-1 profile"):
        quadmesh.loft_fn(lambda t: linemesh.rotate(base, t, axis=(0, 0, 1)),
                         np.linspace(0.0, 1.0, 3), order=2)


def test_rejects_profiles_that_are_not_index_paired():
    a = linemesh.circle(RT, NU, center=(R, 0.0, 0.0), normal=(0, 1, 0))
    b = linemesh.circle(RT, NU + 1, center=(R, 0.0, 0.0), normal=(0, 1, 0))
    f = lambda t: (a if t < 0.5 else b)                            # noqa: E731
    with pytest.raises(ValueError, match="index-paired and conformal"):
        quadmesh.loft_fn(f, np.linspace(0.0, 1.0, 3))


# -- sweep_nodes on loft itself ----------------------------------------------

def test_sweep_nodes_must_be_sized_per_layer():
    base = _tube_ring(3)
    slices = [linemesh.translate(base, (0.0, 0.0, z)) for z in (0.0, 1.0, 2.0)]
    mid = [linemesh.translate(base, (0.0, 0.0, z)) for z in (0.3, 0.7)]
    with pytest.raises(ValueError, match="one entry per layer"):
        quadmesh.loft(slices, sweep_nodes=[mid])
    with pytest.raises(ValueError, match="order-1"):
        quadmesh.loft(slices, sweep_nodes=[mid[:1], mid[:1]])


def test_sweep_nodes_at_order_one_is_ignored_not_an_error():
    """Order 1 has no interior level, so an empty stack is simply a no-op."""
    base = _tube_ring(1)
    slices = [linemesh.translate(base, (0.0, 0.0, z)) for z in (0.0, 1.0, 2.0)]
    got = quadmesh.loft(slices, sweep_nodes=[[], []])
    want = quadmesh.loft(slices)
    assert np.array_equal(got.points, want.points)
    assert np.array_equal(got.corners, want.corners)


def test_straight_sweep_nodes_reproduce_the_plain_loft():
    """Handing in exactly the profiles the lerp would have invented must give back
    the plain loft -- so ``sweep_nodes`` changes geometry only when the geometry
    genuinely is not straight."""
    from nekmeshpy.core.fields import gll_nodes
    order = 3
    base = _tube_ring(order)
    zs = [0.0, 1.0, 2.0]
    slices = [linemesh.translate(base, (0.0, 0.0, z)) for z in zs]
    g = gll_nodes(order)[1:order]
    sweep = [[linemesh.translate(base, (0.0, 0.0, zs[i] + t * (zs[i + 1] - zs[i])))
              for t in g] for i in range(2)]
    got = quadmesh.loft(slices, sweep_nodes=sweep)
    want = quadmesh.loft(slices)
    assert np.allclose(got.interior, want.interior, atol=1e-14)
    assert np.allclose(got.line_mesh.interior, want.line_mesh.interior, atol=1e-14)
