"""Unit tests for ``HexMesh.loft_fn`` -- ``HexMesh.loft`` with the sections
**evaluated** from a parametrization instead of handed in -- and for the
``sweep_nodes`` / per-layer ``element_tags`` arguments it delegates through.

The contract, one rung up from ``QuadMesh.loft_fn``: a plain ``loft`` sees only the
corner-level sections, so at ``order > 1`` every node between two of them is a plain GLL
lerp of their in-plane blocks and the swept solid comes out high-order in storage and
linear in geometry.  Lofting a solid torus from **exact** disc sections still puts its
interior nodes tens of percent of the tube radius off the true shape.  ``loft_fn``
evaluates the sections at the intermediate GLL levels too, so every node is a genuine
section point and the solid is exact.

The load-bearing regression here is the *other* direction: ``loft`` with
``sweep_nodes=None`` and ``element_tags=None`` must stay bit-identical to what it has
always produced, which is what keeps the goldens frozen.
"""

import numpy as np
import pytest
from conftest import conformal, face_rows

from nekmeshpy import ElementTags, hexmesh, linemesh, quadmesh

R, RT, NS, NV = 2.0, 0.6, 2, 6
RADIAL = np.array([0.5, 1.0])


def _tube_disc(order, *, flipped=False, wall_tag=""):
    """One solid tube cross-section of the torus, an O-grid disc at ``theta = 0``.

    It sits in the ``y = 0`` plane centred on ``(R, 0, 0)``, so rotating it about
    ``z`` sweeps the torus.  ``flipped`` reverses the boundary loop, which reverses
    the section winding and drives ``loft`` down its left-handed branch.
    """
    ring = linemesh.circle(RT, 4 * NS, center=(R, 0.0, 0.0), normal=(0, 1, 0),
                           order=order,
                           element_tag="wall" if wall_tag else "")
    if flipped:
        ring = linemesh.reverse(ring)
    return quadmesh.ogrid(ring, NS, RADIAL, wall_tag=wall_tag)


def _flat_disc(order, *, flipped=False):
    """The same O-grid disc, but in the ``z = 0`` plane -- the section for the
    straight ``translate``-along-``z`` stacks the ``loft`` argument tests use."""
    ring = linemesh.circle(RT, 4 * NS, order=order)
    if flipped:
        ring = linemesh.reverse(ring)
    return quadmesh.ogrid(ring, NS, RADIAL)


def _torus_f(order, **kw):
    """The torus as a family of sections: the same disc, *placed* by rotation.

    Placing rather than rebuilding is the idiom the docstring recommends -- the
    affine ops move no index, so every section stays index-paired with the first.
    """
    base = _tube_disc(order, **kw)
    return lambda t: quadmesh.rotate(base, t, axis=(0, 0, 1))


def _ring_fractions(n=NV):
    """``n+1`` values whose last is the wrap back onto the first section."""
    return np.linspace(0.0, 2.0 * np.pi, n + 1)


def _section_profile(order, **kw):
    """The section's own conformal nodes as ``(rho, z)`` pairs -- the exact generator
    of the solid of revolution."""
    nodes, _ = conformal(_tube_disc(order, **kw))
    return np.column_stack([nodes[:, 0], nodes[:, 2]])


def _revolution_deviation(block, order, **kw):
    """Max distance of any conformal node of ``block`` from the true solid torus.

    Every node of an exact sweep is some section node rotated about ``z``, so in the
    ``(rho, z)`` half-plane it must coincide with a node of the generating section.
    """
    nodes, _ = conformal(block)
    q = np.column_stack([np.hypot(nodes[:, 0], nodes[:, 1]), nodes[:, 2]])
    p = _section_profile(order, **kw)
    return float(np.max(np.min(
        np.linalg.norm(q[:, None, :] - p[None, :, :], axis=2), axis=1)))


# -- the defect this factory closes -----------------------------------------

@pytest.mark.parametrize("order", [2, 3, 4])
def test_plain_loft_of_exact_sections_is_straight_along_the_sweep(order):
    """The baseline: exact input sections do *not* give an exact swept solid."""
    f = _torus_f(order)
    fr = _ring_fractions()
    straight = hexmesh.loft([f(t) for t in fr[:-1]], loop=True)
    # off by a sizeable fraction of the tube radius -- not float noise
    assert _revolution_deviation(straight, order) > 0.1 * RT


@pytest.mark.parametrize("order", [1, 2, 3, 4])
def test_every_node_lies_on_the_true_solid_torus(order):
    torus = hexmesh.loft_fn(_torus_f(order), _ring_fractions(),
                            loop=True, order=order)
    assert _revolution_deviation(torus, order) < 1e-13


@pytest.mark.parametrize("order", [1, 2, 3])
def test_corners_are_exact_at_every_order(order):
    """Corners were always exact; the fix must not disturb them."""
    torus = hexmesh.loft_fn(_torus_f(order), _ring_fractions(),
                            loop=True, order=order)
    x, y, z = torus.points.T
    rho = np.hypot(x, y)
    # the outermost corner ring rides the tube surface exactly
    d = np.abs(np.hypot(rho - R, z) - RT)
    assert np.min(d) < 1e-14
    assert np.max(np.abs(rho - R)) <= RT + 1e-14


# -- the closed solid --------------------------------------------------------

@pytest.mark.parametrize("order", [1, 2, 3])
def test_loop_gives_a_closed_solid_with_no_duplicated_layer(order):
    torus = hexmesh.loft_fn(_torus_f(order), _ring_fractions(),
                            loop=True, order=order)
    sec = _tube_disc(order)
    assert torus.n_points == sec.n_points * NV      # no seam section duplicated
    assert torus.n_hexes == sec.n_quads * NV        # NV layers, not NV-1
    assert hexmesh.is_conforming(torus)
    # the seam faces are genuine shared entities, so the only free surface is the
    # swept tube wall -- and that surface is closed, i.e. the solid is watertight
    assert hexmesh.is_watertight(torus)
    faces = hexmesh.boundary_faces(torus)
    assert faces.shape[0] == 4 * NS * NV            # only the outer ring's wall
    # no boundary face is a sweep-direction cap (faces 5/6): the sweep is periodic
    assert set(np.unique(faces[:, 1])) <= {1, 2, 3, 4}


def test_loop_rejects_a_family_that_does_not_close():
    base = _tube_disc(2)
    # a full turn *plus* a bit: f(fr[-1]) does not land back on f(fr[0])
    fr = np.linspace(0.0, 2.0 * np.pi + 0.3, NV + 1)
    with pytest.raises(ValueError, match="map back to the first section"):
        hexmesh.loft_fn(lambda t: quadmesh.rotate(base, t, axis=(0, 0, 1)), fr,
                        loop=True, order=2)


def test_loop_places_end_caps_on_the_seam():
    """A closed sweep has no free end, but its seam is a real side -- a cap tag names
    that side of it rather than being refused."""
    hm = hexmesh.loft_fn(_torus_f(1), _ring_fractions(), loop=True, first_tag="in")
    assert hm.face_tags.count("in") > 0


def test_loop_rejects_fewer_than_three_fractions():
    with pytest.raises(ValueError, match="at least 3 fractions"):
        hexmesh.loft_fn(_torus_f(1), np.array([0.0, 2.0 * np.pi]), loop=True)


def test_needs_at_least_two_fractions():
    with pytest.raises(ValueError, match="at least 2 fractions"):
        hexmesh.loft_fn(_torus_f(1), np.array([0.0]))


# -- order 1 and the open sweep are the plain loft ---------------------------

@pytest.mark.parametrize("loop", [False, True])
def test_order_one_equals_a_plain_loft_of_the_same_sections(loop):
    f = _torus_f(1)
    fr = _ring_fractions()
    got = hexmesh.loft_fn(f, fr, loop=loop, order=1)
    want = hexmesh.loft([f(t) for t in (fr[:-1] if loop else fr)], loop=loop)
    assert np.array_equal(got.points, want.points)
    assert np.array_equal(got.corners, want.corners)
    assert np.array_equal(got.hexes, want.hexes)
    assert np.array_equal(got.orient, want.orient)
    assert np.array_equal(got.element_tags.ids, want.element_tags.ids)
    assert np.array_equal(got.element_tags.tags, want.element_tags.tags)


def test_open_sweep_has_the_expected_shape_and_caps():
    order = 2
    f = _torus_f(order)
    fr = np.linspace(0.0, np.pi / 2, 4)
    blk = hexmesh.loft_fn(f, fr, order=order,
                          first_tag="start", last_tag="end")
    sec = _tube_disc(order)
    assert blk.n_points == sec.n_points * 4
    assert blk.n_hexes == sec.n_quads * 3
    assert _revolution_deviation(blk, order) < 1e-13
    assert set(blk.face_group_tags) == {"start", "end"}


# -- grading -----------------------------------------------------------------

def test_grading_is_honored_per_layer():
    """A non-uniform sweep grading must place each layer's interior nodes inside
    that layer's own parameter span -- and still on the true solid."""
    order = 3
    f = _torus_f(order)
    fr = np.array([0.0, 0.3, 2.4, 2.0 * np.pi])
    blk = hexmesh.loft_fn(f, fr, order=order)
    assert _revolution_deviation(blk, order) < 1e-13
    # the corner levels sit exactly at the requested parameters
    theta = np.arctan2(blk.points[:, 1], blk.points[:, 0])
    theta[theta < -1e-12] += 2.0 * np.pi
    for want in fr:
        assert np.min(np.abs(theta - (want % (2.0 * np.pi)))) < 1e-12


# -- tags --------------------------------------------------------------------

def test_element_tags_name_the_swept_column_of_each_section_quad():
    """``element_tags`` is per *section element*: quad q's tag lands on every hex
    swept from it, at every layer -- and a single string names the whole block."""
    base = quadmesh.ogrid(
        linemesh.circle(RT, 4 * NS, center=(R, 0.0, 0.0), normal=(0, 1, 0)),
        NS, RADIAL)
    f = lambda t: quadmesh.rotate(base, t, axis=(0, 0, 1))                   # noqa: E731
    per_quad = ElementTags.from_dense(["hot"] + [""] * (base.n_quads - 1))
    blk = hexmesh.loft_fn(f, np.linspace(0.0, 1.0, 4), element_tags=per_quad)
    tags = blk.element_tags.dense(blk.n_hexes).reshape(3, base.n_quads)   # hex (layer i, quad q)
    assert list(np.unique(tags[:, 0])) == ["hot"]
    assert list(np.unique(tags[:, 1:])) == [""]
    assert hexmesh.loft_fn(f, np.linspace(0.0, 1.0, 4),
                           element_tags="fluid").element_group_tags == ["fluid"]


def test_side_and_cap_boundary_tags_survive_the_element_tags():
    f = _torus_f(1, wall_tag="wall")
    blk = hexmesh.loft_fn(f, np.linspace(0.0, 1.0, 3), order=1,
                          element_tags="a", first_tag="inlet", last_tag="outlet")
    assert set(blk.face_group_tags) == {"wall", "inlet", "outlet"}
    rows = face_rows(blk)
    # caps land on faces 5/6 of the first / last layer, the wall on side faces
    sides = lambda nm: {f for _, f, t in rows if t == nm}  # noqa: E731
    assert sides("inlet") == {5}
    assert sides("outlet") == {6}
    assert sides("wall") <= {1, 2, 3, 4}
    assert blk.element_tags.group_tags == ["a"]


def test_loft_rejects_element_tags_naming_a_quad_the_section_lacks():
    base = _flat_disc(1)
    slices = [quadmesh.translate(base, (0.0, 0.0, z)) for z in (0.0, 1.0, 2.0)]
    with pytest.raises(ValueError, match="only %d elements" % base.n_quads):
        hexmesh.loft(slices, element_tags=ElementTags([base.n_quads], ["off"]))
    with pytest.raises(TypeError, match="element_tags must be"):
        hexmesh.loft(slices, element_tags=["a", "b"])


# -- validation of what f returns --------------------------------------------

def test_rejects_a_section_of_the_wrong_order():
    base = _tube_disc(1)
    with pytest.raises(ValueError, match="order-1 section"):
        hexmesh.loft_fn(lambda t: quadmesh.rotate(base, t, axis=(0, 0, 1)),
                        np.linspace(0.0, 1.0, 3), order=2)


def test_rejects_sections_that_are_not_index_paired():
    a = quadmesh.ogrid(
        linemesh.circle(RT, 4 * NS, center=(R, 0.0, 0.0), normal=(0, 1, 0)),
        NS, RADIAL)
    b = quadmesh.ogrid(
        linemesh.circle(RT, 4 * (NS + 2), center=(R, 0.0, 0.0), normal=(0, 1, 0)),
        NS + 2, RADIAL)
    f = lambda t: (a if t < 0.5 else b)                            # noqa: E731
    with pytest.raises(ValueError, match="index-paired and conformal"):
        hexmesh.loft_fn(f, np.linspace(0.0, 1.0, 3))


# -- sweep_nodes on loft itself ----------------------------------------------

def test_sweep_nodes_must_be_sized_per_layer():
    base = _flat_disc(3)
    slices = [quadmesh.translate(base, (0.0, 0.0, z)) for z in (0.0, 1.0, 2.0)]
    mid = [quadmesh.translate(base, (0.0, 0.0, z)) for z in (0.3, 0.7)]
    with pytest.raises(ValueError, match="one entry per layer"):
        hexmesh.loft(slices, sweep_nodes=[mid])
    with pytest.raises(ValueError, match="order-1"):
        hexmesh.loft(slices, sweep_nodes=[mid[:1], mid[:1]])


def test_sweep_nodes_must_match_the_slices():
    base = _flat_disc(2)
    other = quadmesh.ogrid(linemesh.circle(RT, 4 * (NS + 2), order=2),
                           NS + 2, RADIAL)
    slices = [quadmesh.translate(base, (0.0, 0.0, z)) for z in (0.0, 1.0, 2.0)]
    with pytest.raises(ValueError, match="must match the slices"):
        hexmesh.loft(slices, sweep_nodes=[[other], [other]])


def test_sweep_nodes_at_order_one_is_ignored_not_an_error():
    """Order 1 has no interior level, so an empty stack is simply a no-op."""
    base = _flat_disc(1)
    slices = [quadmesh.translate(base, (0.0, 0.0, z)) for z in (0.0, 1.0, 2.0)]
    got = hexmesh.loft(slices, sweep_nodes=[[], []])
    want = hexmesh.loft(slices)
    assert np.array_equal(got.points, want.points)
    assert np.array_equal(got.corners, want.corners)


@pytest.mark.parametrize("flipped", [False, True])
def test_straight_sweep_nodes_reproduce_the_plain_loft(flipped):
    """Handing in exactly the sections the lerp would have invented must give back
    the plain loft -- so ``sweep_nodes`` changes geometry only when the geometry
    genuinely is not straight.  Parametrized over the winding, because the
    left-handed branch transposes the in-plane grid and the intermediate levels
    must take that transpose too."""
    from nekmeshpy.core.fields import gll_nodes
    order = 3
    base = _flat_disc(order, flipped=flipped)
    zs = [0.0, 1.0, 2.0]
    slices = [quadmesh.translate(base, (0.0, 0.0, z)) for z in zs]
    g = gll_nodes(order)[1:order]
    sweep = [[quadmesh.translate(base, (0.0, 0.0, zs[i] + t * (zs[i + 1] - zs[i])))
              for t in g] for i in range(2)]
    got = hexmesh.loft(slices, sweep_nodes=sweep)
    want = hexmesh.loft(slices)
    assert np.allclose(got.interior, want.interior, atol=1e-14)
    assert np.allclose(got.quad_mesh.interior, want.quad_mesh.interior, atol=1e-14)
    assert np.allclose(got.quad_mesh.line_mesh.interior, want.quad_mesh.line_mesh.interior,
                       atol=1e-14)


# -- the reverse-wound section -----------------------------------------------

@pytest.mark.parametrize("order", [2, 3])
def test_reverse_wound_section_sweeps_exactly_too(order):
    """A left-handed section takes ``loft``'s ``flip`` branch, which transposes the
    in-plane block; the supplied sweep levels must be transposed identically or the
    column's grid is scrambled against its own corner table."""
    f = _torus_f(order, flipped=True)
    fr = _ring_fractions()
    torus = hexmesh.loft_fn(f, fr, loop=True, order=order)
    assert _revolution_deviation(torus, order, flipped=True) < 1e-13
    assert hexmesh.is_conforming(torus)
    straight = hexmesh.loft([f(t) for t in fr[:-1]], loop=True)
    assert _revolution_deviation(straight, order, flipped=True) > 0.1 * RT
