"""The metric queries: ``bounds`` / ``centroid`` and the size each rung names for
itself -- ``linemesh.length`` / ``quadmesh.area`` / ``hexmesh.volume``.

One kernel serves all three (``core.measure`` integrates a node block), so the
properties to pin are the same at each rung: the linear reading measures the
straight-sided element the corners describe, the ``high_order=True`` reading measures
the curved element the mesh actually stores, and the two differ exactly where the mesh
is curved -- which is the trap ``docs/user/concepts`` warns about, asserted from the
measuring side.  The sizes are pinned against geometry with a closed form, so a wrong
quadrature or a transposed lattice cannot pass.
"""

import numpy as np
import pytest

from nekmeshpy import hexmesh, linemesh, quadmesh
from nekmeshpy.core import measure

RADIAL = np.linspace(0.5, 1.0, 3)


def _rungs(order):
    """One mesh per rung with its package, all curved at ``order > 1``: a unit circle,
    the O-grid filling it, and that extruded to height 2."""
    ring = linemesh.circle(1.0, 8, element_tag="wall", order=order)
    section = quadmesh.ogrid(ring, 2, RADIAL, wall_tag="wall")
    block = hexmesh.extrude(section, length=2.0, layers=2,
                            first_tag="inlet", last_tag="outlet")
    return ((ring, linemesh, "length", "element_lengths"),
            (section, quadmesh, "area", "element_areas"),
            (block, hexmesh, "volume", "element_volumes"))


def _size(pkg, name, mesh, **kw):
    return getattr(pkg, name)(mesh, **kw)


# -- the linear reading is exact on straight-sided geometry -------------
def test_linear_size_is_exact_on_a_polygon():
    """An 8-gon inscribed in the unit circle has a closed form at every rung, and the
    order-1 mesh *is* that polygon -- so these are equalities, not tolerances."""
    ring, section, block = (r[0] for r in _rungs(1))
    perimeter = 8 * 2 * np.sin(np.pi / 8)
    area = 8 * 0.5 * np.sin(2 * np.pi / 8)
    assert linemesh.length(ring) == pytest.approx(perimeter, rel=1e-12)
    assert quadmesh.area(section) == pytest.approx(area, rel=1e-12)
    assert hexmesh.volume(block) == pytest.approx(2.0 * area, rel=1e-12)


def test_a_unit_cube_measures_one_at_any_order():
    """The volume quadrature is taken high enough to integrate an order-N Jacobian
    exactly, so a cube reads 1.0 at every order and on both readings."""
    for order in (1, 2, 4):
        corners = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=float)
        sq = quadmesh.rectangle(corners, 2, 2, order=order)
        cube = hexmesh.extrude(sq, length=1.0, layers=2)
        assert hexmesh.volume(cube) == pytest.approx(1.0, abs=1e-13)
        assert hexmesh.volume(cube, high_order=True) == pytest.approx(1.0, abs=1e-13)
        assert quadmesh.area(sq, high_order=True) == pytest.approx(1.0, abs=1e-13)


# -- the high-order reading is the curved one ---------------------------
def test_high_order_recovers_the_curve_the_linear_reading_misses():
    """The order-1 mesh measures the inscribed polygon; the order-4 mesh stores nodes
    on the true arc, and reading it curved recovers the circle to quadrature accuracy.

    This is the storage-vs-geometry trap from the other side: same eight elements, and
    the number moves only because the *nodes between the corners* are being used."""
    ring, section, block = (r[0] for r in _rungs(4))
    assert linemesh.length(ring) == pytest.approx(8 * 2 * np.sin(np.pi / 8), rel=1e-12)
    assert linemesh.length(ring, high_order=True) == pytest.approx(2 * np.pi, rel=1e-8)
    assert quadmesh.area(section, high_order=True) == pytest.approx(np.pi, rel=1e-8)
    assert hexmesh.volume(block, high_order=True) == pytest.approx(2 * np.pi, rel=1e-8)


def test_the_two_readings_agree_at_order_one():
    """At order 1 there are no interior nodes to find, so the flag cannot matter."""
    for mesh, pkg, size, _per in _rungs(1):
        assert _size(pkg, size, mesh) == pytest.approx(
            _size(pkg, size, mesh, high_order=True), rel=1e-14)


def test_element_sizes_sum_to_the_total():
    """The total is defined as the per-element measure summed -- pinned so the two can
    never drift apart."""
    for order in (1, 3):
        for mesh, pkg, size, per in _rungs(order):
            for ho in (False, True):
                parts = getattr(pkg, per)(mesh, high_order=ho)
                assert parts.shape == (mesh.points[getattr(
                    mesh, {"length": "lines", "area": "quads",
                           "volume": "hexes"}[size])].shape[0],)
                assert float(parts.sum()) == pytest.approx(
                    _size(pkg, size, mesh, high_order=ho), rel=1e-12)


# -- signs, and what they catch -----------------------------------------
def test_volume_is_signed_so_an_inverted_mesh_reads_negative():
    """The hex measure is the signed Jacobian integral, so a mesh wound inside out --
    what a bare reflection produces -- comes back negative rather than plausible."""
    from nekmeshpy.core import affine
    block = _rungs(1)[2][0]
    M, off = affine.reflection((1.0, 0.0, 0.0))
    flipped = hexmesh.transform(block, M, off)
    assert hexmesh.volume(block) > 0
    assert hexmesh.volume(flipped) == pytest.approx(-hexmesh.volume(block), rel=1e-12)
    assert (hexmesh.element_volumes(flipped) < 0).all()


def test_length_and_area_are_never_negative():
    """A curve or a surface embedded in 3-D has no orientation to be signed by."""
    from nekmeshpy.core import affine
    ring, section = _rungs(1)[0][0], _rungs(1)[1][0]
    M, off = affine.reflection((1.0, 0.0, 0.0))
    assert linemesh.length(linemesh.transform(ring, M, off)) > 0
    assert quadmesh.area(quadmesh.transform(section, M, off)) > 0


# -- centroid is a mass property, not a node average --------------------
def test_centroid_is_measure_weighted_not_a_node_mean():
    """A graded O-grid piles nodes near the wall, so the mean of the points is pulled
    off the axis while the area-weighted centroid stays on it."""
    section = quadmesh.ogrid(linemesh.circle(1.0, 8), 2, np.linspace(0.2, 1.0, 4))
    moved = quadmesh.translate(section, (3.0, -1.0, 0.0))
    assert quadmesh.centroid(moved) == pytest.approx([3.0, -1.0, 0.0], abs=1e-12)


def test_centroid_of_a_block_sits_at_its_mid_height():
    block = _rungs(3)[2][0]
    assert hexmesh.centroid(block, high_order=True) == pytest.approx(
        [0.0, 0.0, 1.0], abs=1e-9)


def test_centroid_follows_a_rigid_placement():
    """A centroid is a point in space, so it must ride any placement exactly."""
    for mesh, pkg, _size_name, _per in _rungs(2):
        before = pkg.centroid(mesh, high_order=True)
        after = pkg.centroid(pkg.translate(mesh, (1.0, 2.0, 3.0)), high_order=True)
        assert after == pytest.approx(before + np.array([1.0, 2.0, 3.0]), abs=1e-12)


# -- bounds --------------------------------------------------------------
def test_bounds_and_its_derived_readings():
    block = _rungs(1)[2][0]
    b = hexmesh.bounds(block)
    assert b.min == pytest.approx([-1.0, -1.0, 0.0], abs=1e-12)
    assert b.max == pytest.approx([1.0, 1.0, 2.0], abs=1e-12)
    assert b.size == pytest.approx([2.0, 2.0, 2.0], abs=1e-12)
    assert b.center == pytest.approx([0.0, 0.0, 1.0], abs=1e-12)
    assert b.diagonal == pytest.approx(np.sqrt(12.0), rel=1e-12)


def test_bounds_high_order_sees_nodes_the_corners_miss():
    """Rolled off the axes, a coarse circle's corners all sit inside ``x = 1`` while
    the arc between them reaches it -- so the curved box is the wider one, and only the
    interior nodes know that."""
    ring = linemesh.rotate(linemesh.circle(1.0, 4, order=4), np.pi / 4)
    corner_box = linemesh.bounds(ring)
    node_box = linemesh.bounds(ring, high_order=True)
    assert (node_box.max >= corner_box.max - 1e-15).all()
    assert node_box.max[0] > corner_box.max[0] + 1e-3
    assert node_box.max[0] == pytest.approx(1.0, rel=1e-6)


def test_bounds_of_nothing_is_an_error():
    with pytest.raises(ValueError, match="no points"):
        measure.bounds_of(np.zeros((0, 3)))


# -- composition across the rungs ---------------------------------------
def test_wetted_area_is_the_boundary_mesh_measured_one_rung_down():
    """There is no ``hexmesh.area``: a block's wall area is the quad-rung measure of
    the wall's own mesh, which is the toolkit's answer to a query that changes rung."""
    block = _rungs(1)[2][0]
    wall = hexmesh.boundary_mesh(block, "wall")
    assert quadmesh.area(wall) == pytest.approx(
        2.0 * 8 * 2 * np.sin(np.pi / 8), rel=1e-12)


def test_a_measure_survives_a_round_trip_through_select():
    """Splitting a mesh and measuring the pieces must add back up -- the property that
    ties the new subset operations to the new measures."""
    block = _rungs(3)[2][0]
    ids = np.arange(block.n_hexes)
    a = hexmesh.select(block, ids[: block.n_hexes // 2])
    b = hexmesh.remove(block, ids[: block.n_hexes // 2])
    assert (hexmesh.volume(a, high_order=True) + hexmesh.volume(b, high_order=True)
            == pytest.approx(hexmesh.volume(block, high_order=True), rel=1e-12))
