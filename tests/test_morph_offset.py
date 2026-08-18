"""``linemesh.offset`` / ``quadmesh.offset``: a rung-preserving displacement along the
mesh's own tangent(s), averaged across every element sharing a node.

Both rungs share the same contract this file pins: connectivity and tags ride through
verbatim, every node moves by exactly ``distance`` (unit direction), and a shared node
(a chain's interior corners, a quad's shared edge/corner) gets the *average* of every
incident element's own direction rather than one element's alone.
"""

import numpy as np
import pytest
from conftest import assert_same_side_tags

from nekmeshpy import linemesh, quadmesh


# -- LineMesh -------------------------------------------------------------------------
def test_line_offset_moves_a_straight_segment_by_exactly_distance():
    """A single straight segment: the offset direction is unambiguous (any in-plane
    perpendicular to the one tangent), and every point moves exactly ``distance``."""
    P = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    lm = linemesh.line(P[0], P[1], [0.0, 1.0])
    out = linemesh.offset(lm, 0.5, normal=(0.0, 0.0, 1.0))
    d = out.points - lm.points
    assert np.allclose(np.linalg.norm(d, axis=1), 0.5)
    # perpendicular to the segment and to the given plane normal
    assert np.allclose(d @ np.array([1.0, 0.0, 0.0]), 0.0, atol=1e-12)
    assert np.allclose(d @ np.array([0.0, 0.0, 1.0]), 0.0, atol=1e-12)


@pytest.mark.parametrize("order", [1, 3])
def test_line_offset_keeps_high_order_nodes_on_the_offset_circle(order):
    """A circle offset outward by ``d`` lands every node -- corners and GLL interior
    nodes alike -- on the exact circle of radius ``R+d``."""
    R = 2.0
    ring = linemesh.circle(R, 8, order=order)
    out = linemesh.offset(ring, 0.3, normal=(0.0, 0.0, 1.0))
    good = np.vstack([out.points, out.interior.reshape(-1, 3)])
    r = np.linalg.norm(good, axis=1)
    # sign of the auto/explicit plane normal is a convention: the magnitude of the
    # displacement is what's pinned, in either direction
    assert np.allclose(r, R + 0.3, atol=1e-12) or np.allclose(r, R - 0.3, atol=1e-12)


def test_line_offset_auto_detects_the_plane_when_normal_is_omitted():
    ring = linemesh.circle(1.0, 8, order=2)
    explicit = linemesh.offset(ring, 0.2, normal=(0.0, 0.0, 1.0))
    auto = linemesh.offset(ring, 0.2)
    r_expl = np.linalg.norm(explicit.points, axis=1)
    r_auto = np.linalg.norm(auto.points, axis=1)
    assert np.allclose(r_expl, r_auto, atol=1e-12)


def test_line_offset_averages_at_a_shared_corner():
    """Two collinear segments sharing an endpoint: the shared corner's tangent is the
    average of both segments' own tangent, which here is the same direction as each --
    so the corner still moves by exactly ``distance`` along it."""
    P = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    chain = linemesh.loft(P)
    out = linemesh.offset(chain, 0.4, normal=(0.0, 0.0, 1.0))
    d = out.points - chain.points
    assert np.allclose(np.linalg.norm(d, axis=1), 0.4)
    # all three points offset in the same direction (straight chain -> no cancellation)
    assert np.allclose(d[0], d[1]) and np.allclose(d[1], d[2])


def test_line_offset_miters_at_a_bend():
    """A right-angle bend is a crease, and across a crease the offset is a **miter**, not
    an averaged normal -- the property that matters is that every *segment* ends up
    ``distance`` from where it was, which is what a CAD offset guarantees.

    Stepping ``distance`` along the averaged normal instead would move the corner only
    ``0.3``, leaving each segment at ``0.3*cos(45deg) = 0.212`` and pinching the layer to
    71% of its thickness exactly at the corner."""
    P = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    chain = linemesh.loft(P)
    out = linemesh.offset(chain, 0.3, normal=(0.0, 0.0, 1.0))

    # each segment is axis-aligned, so its offset distance is one coordinate
    assert np.isclose(abs(out.points[0][1] - P[0][1]), 0.3)      # y=0 segment
    assert np.isclose(abs(out.points[2][0] - P[2][0]), 0.3)      # x=0 segment
    # the corner sits 0.3 clear of *both* segment lines at once
    assert np.isclose(abs(out.points[1][1]), 0.3)
    assert np.isclose(abs(out.points[1][0]), 0.3)
    # so it travels 0.3*sqrt(2), not 0.3
    assert np.isclose(np.linalg.norm(out.points[1] - P[1]), 0.3 * np.sqrt(2.0))


def test_line_offset_keeps_averaging_where_the_curve_is_smooth():
    """The miter must not leak into smooth stretches: a circle's nodes are all below the
    crease angle, so they step exactly ``distance`` and stay on the true offset circle.
    Mitering every facet instead would push them to ``R + d/cos(pi/n)``."""
    R, d, n = 2.0, 0.3, 8
    out = linemesh.offset(linemesh.circle(R, n), d, normal=(0.0, 0.0, 1.0))
    r = np.linalg.norm(out.points, axis=1)
    assert np.allclose(r, R + d, atol=1e-12) or np.allclose(r, R - d, atol=1e-12)
    assert not np.isclose(r[0], R + d / np.cos(np.pi / n))


def test_line_offset_preserves_topology_and_tags():
    ring = linemesh.circle(1.0, 8, element_tag="wall", order=3)
    out = linemesh.offset(ring, 0.1)
    assert np.array_equal(out.lines, ring.lines)
    assert out.order == ring.order
    assert np.array_equal(out.element_tags.ids, ring.element_tags.ids)
    assert np.array_equal(out.element_tags.tags, ring.element_tags.tags)
    assert_same_side_tags(out.point_tags, ring.point_tags)


# -- QuadMesh -------------------------------------------------------------------------
def test_quad_offset_moves_a_flat_section_by_exactly_distance_along_z():
    ring = linemesh.circle(1.0, 8)
    section = quadmesh.ogrid(ring, 2, np.linspace(0.4, 1.0, 3))
    out = quadmesh.offset(section, 0.25)
    d = out.points - section.points
    assert np.allclose(np.linalg.norm(d, axis=1), 0.25)
    # a flat section in the z=0 plane offsets purely along +-z
    assert np.allclose(d[:, :2], 0.0, atol=1e-12)
    assert np.allclose(np.abs(d[:, 2]), 0.25, atol=1e-12)


def _radii(mesh):
    good = np.vstack([mesh.points, mesh.line_mesh.interior.reshape(-1, 3),
                      mesh.interior.reshape(-1, 3)])
    return np.linalg.norm(good, axis=1)


@pytest.mark.parametrize("order", [1, 4])
def test_quad_offset_keeps_high_order_nodes_near_the_offset_sphere(order):
    """A closed cubed-sphere surface offset by ``d`` along its own normal (every quad's
    normal is already radial, and every shared node's neighbours agree, so no
    cancellation) lands every node -- corners, shared edge nodes, and private interior
    nodes -- close to the radius ``R+d`` (or ``R-d``, a winding convention).

    The averaged direction is a *discrete* tangent estimate off the mesh's own GLL
    nodes, not the sphere's exact analytic normal (the cubed-sphere's cube-edge nodes
    sit where three differently-parametrized patches meet), so this converges with
    order rather than landing exactly -- pinned by :func:`test_quad_offset_sphere_error_shrinks_with_order`."""
    R = 2.0
    sph = quadmesh.sphere(R, 3, order=order)
    out = quadmesh.offset(sph, 0.3)
    r = _radii(out)
    assert np.allclose(r, R + 0.3, atol=5e-4) or np.allclose(r, R - 0.3, atol=5e-4)


def test_quad_offset_sphere_error_shrinks_with_order():
    R = 2.0
    err = []
    for order in (1, 2, 4):
        sph = quadmesh.sphere(R, 3, order=order)
        r = _radii(quadmesh.offset(sph, 0.3))
        err.append(min(np.abs(r - (R + 0.3)).max(), np.abs(r - (R - 0.3)).max()))
    assert err[0] > err[1] > err[2]


def test_quad_offset_averages_across_incident_quads_at_a_shared_corner():
    """A structured 2x2 grid: the interior point shared by all four quads gets the
    (renormalized) average of their four individual normals -- here all identical by
    symmetry, so it reduces to a single normal, but the averaging path is exercised."""
    corners = [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [2.0, 2.0, 0.0], [0.0, 2.0, 0.0]]
    section = quadmesh.rectangle(corners, 2, 2)
    out = quadmesh.offset(section, 0.2)
    d = out.points - section.points
    assert np.allclose(np.linalg.norm(d, axis=1), 0.2)
    assert np.allclose(d, d[0])                      # flat plate: every normal agrees


def test_quad_offset_preserves_topology_and_tags():
    ring = linemesh.circle(1.0, 8, element_tag="wall")
    section = quadmesh.ogrid(ring, 2, np.linspace(0.4, 1.0, 3), wall_tag="wall")
    out = quadmesh.offset(section, 0.1)
    assert np.array_equal(out.corners, section.corners)
    assert np.array_equal(out.quads, section.quads)
    assert np.array_equal(out.orient, section.orient)
    assert out.order == section.order
    assert_same_side_tags(out.edge_tags, section.edge_tags)
