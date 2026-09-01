"""Uniform H-refinement: ``linemesh.refine`` / ``quadmesh.refine`` / ``hexmesh.refine``
-- split every line into 2, every quad into 4, every hex into 8.

Built bottom-up like the ladder itself: a hex's refine calls quadmesh.refine on its
shared faces, which calls linemesh.refine on its shared edges, so a face or edge
shared between two elements is refined exactly once and both neighbours land on the
identical result. Exact at any polynomial order -- every new midpoint/center/
cell-center point, and every child's own curved interior, is read off the parent's
*stored* polynomial map via ``core.interp.resample_block_at``, not a straight-line or
bilinear guess -- so refining a curved mesh does not facet it.
"""

import numpy as np
import pytest

from nekmeshpy import hexmesh, linemesh, quadmesh
from nekmeshpy.core.fields import gll_nodes
from nekmeshpy.core.interp import resample_block, resample_block_at
from nekmeshpy.core.tags import ElementTags
from nekmeshpy.hexmesh.hexmesh import HexMesh
from nekmeshpy.hexmesh.query import element_blocks as hex_blocks
from nekmeshpy.linemesh.query import element_blocks as line_blocks
from nekmeshpy.quadmesh.query import element_blocks as quad_blocks

ORDERS = [1, 2, 3, 4]


# -- fixtures -------------------------------------------------------------------
def _line(order):
    return linemesh.line([0.0, 0.0, 0.0], [2.0, 0.0, 0.0], np.linspace(0.0, 1.0, 5),
                         order=order)


def _rect(nx=2, ny=2, order=1):
    corners = [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [2.0, 2.0, 0.0], [0.0, 2.0, 0.0]]
    return quadmesh.rectangle(corners, nx, ny, order=order)


def _ogrid(order=3):
    circ = linemesh.circle(1.0, 8, order=order)
    return quadmesh.ogrid(circ, 2, 2)


def _box(nx=2, ny=2, nz=2):
    return hexmesh.extrude(_rect(nx, ny), 2.0, nz)


def _curved_hex():
    return hexmesh.extrude(_ogrid(order=3), 3.0, 2)


# -- linemesh.refine --------------------------------------------------------------
def test_linemesh_refine_count_arithmetic():
    m = _line(1)
    r = linemesh.refine(m)
    assert r.n_lines == 2 * m.n_lines
    assert r.n_points == m.n_points + m.n_lines
    assert r.order == m.order


def test_linemesh_refine_linear_midpoint_is_exact():
    m = _line(1)
    r = linemesh.refine(m)
    xs = np.sort(r.points[:, 0])
    assert xs == pytest.approx(np.linspace(0.0, 2.0, 9))


def test_linemesh_refine_curved_circle_stays_on_the_circle():
    m = linemesh.circle(1.0, 8, order=4)
    r = linemesh.refine(m)
    radius = np.hypot(r.points[:, 0], r.points[:, 1])
    assert radius == pytest.approx(1.0, abs=1e-12)


def test_linemesh_refine_tags_propagate_to_both_children():
    m = _line(1)
    m = linemesh.LineMesh(m.point_mesh, m.lines, m.interior,
                          ElementTags.from_dense(["a", "", "", ""]))
    r = linemesh.refine(m)
    dense = r.element_tags.dense(r.n_lines)
    assert list(dense) == ["a", "a", "", "", "", "", "", ""]


@pytest.mark.parametrize("order", ORDERS)
def test_linemesh_refine_geometry_fidelity(order):
    m = linemesh.circle(1.0, 8, order=order)
    blocks = line_blocks(m)
    r = linemesh.refine(m)
    rblocks = line_blocks(r)
    n = 6
    direct = resample_block_at(blocks[0:1], order, [0.5 * gll_nodes(n) + 0.5], 1)
    via_child = resample_block(rblocks[1:2], order, n, 1)
    assert direct == pytest.approx(via_child, abs=1e-10)


def test_linemesh_refine_does_not_mutate_input():
    m = _line(2)
    pts = m.points.copy()
    linemesh.refine(m)
    assert np.array_equal(pts, m.points)


def test_linemesh_refine_composes():
    m = _line(1)
    r2 = linemesh.refine(linemesh.refine(m))
    assert r2.n_lines == 4 * m.n_lines


# -- quadmesh.refine --------------------------------------------------------------
def test_quadmesh_refine_count_arithmetic():
    m = _rect(2, 2)
    r = quadmesh.refine(m)
    assert r.n_quads == 4 * m.n_quads
    assert r.order == m.order
    # 3x3 corner grid (9 points) -> 5x5 (25 points), one center per quad
    assert r.n_points == 25


def test_quadmesh_refine_region_tags_propagate_to_all_four_children():
    m = _rect(2, 2)                                      # 4 quads
    tags = ElementTags.from_dense(["a", "a", "b", "b"])
    m = quadmesh.QuadMesh(m.line_mesh, m.quads, m.orient, m.interior, tags)
    r = quadmesh.refine(m)
    dense = r.element_tags.dense(r.n_quads)
    assert list(dense[:8]) == ["a"] * 8
    assert list(dense[8:]) == ["b"] * 8


def test_quadmesh_refine_edge_tags_propagate_and_new_edges_stay_untagged():
    m = _rect(2, 2)
    m = quadmesh.tag_edges(m, [(0, 4)], ["wall"])
    r = quadmesh.refine(m)
    assert r.edge_tags.group_tags == ["wall"]
    assert len(r.edge_tags) == 2                          # the one edge's two halves


def test_quadmesh_refine_watertight_across_a_multi_quad_mesh():
    m = _rect(3, 3)
    r = quadmesh.refine(m)
    assert r.n_quads == 4 * m.n_quads
    # every original boundary edge splits into 2; no new (interior spoke) edge is
    # ever a boundary edge
    be = quadmesh.boundary_edges(r)
    assert be.shape[0] == 2 * quadmesh.boundary_edges(m).shape[0]


@pytest.mark.parametrize("order", ORDERS)
def test_quadmesh_refine_geometry_fidelity(order):
    m = _ogrid(order)
    blocks = quad_blocks(m)
    r = quadmesh.refine(m)
    rblocks = quad_blocks(r)
    n = 6
    g = gll_nodes(n)
    direct = resample_block_at(blocks[0:1], order, [0.5 * g, 0.5 * g], 2)
    via_child = resample_block(rblocks[0:1], order, n, 2)
    assert direct == pytest.approx(via_child, abs=1e-10)


def test_quadmesh_refine_curved_survival():
    m = _ogrid(order=4)
    r = quadmesh.refine(m)
    outer = np.hypot(r.points[:, 0], r.points[:, 1]).max()
    assert outer == pytest.approx(1.0, abs=1e-12)


@pytest.mark.parametrize("order", ORDERS)
def test_quadmesh_refine_preserves_order(order):
    m = _rect(2, 2, order=order)
    assert quadmesh.refine(m).order == order


def test_quadmesh_refine_does_not_mutate_input():
    m = _rect(2, 2)
    pts = m.points.copy()
    quadmesh.refine(m)
    assert np.array_equal(pts, m.points)


def test_quadmesh_refine_composes():
    m = _rect(2, 2)
    r2 = quadmesh.refine(quadmesh.refine(m))
    assert r2.n_quads == 16 * m.n_quads


# -- hexmesh.refine ---------------------------------------------------------------
def test_hexmesh_refine_count_arithmetic():
    m = _box(2, 2, 2)
    r = hexmesh.refine(m)
    assert r.n_hexes == 8 * m.n_hexes
    assert r.order == m.order
    # 3x3x3 corner grid (27 points) -> 5x5x5 (125 points)
    assert r.n_points == 125


def test_hexmesh_refine_region_tags_propagate_to_all_eight_children():
    m = _box(2, 2, 1)                                    # 4 hexes
    tags = ElementTags.from_dense(["a", "a", "b", "b"])
    m = HexMesh(m.quad_mesh, m.hexes, m.orient, m.interior, tags)
    r = hexmesh.refine(m)
    dense = r.element_tags.dense(r.n_hexes)
    assert list(dense[:16]) == ["a"] * 16
    assert list(dense[16:]) == ["b"] * 16


def test_hexmesh_refine_face_tags_propagate_and_new_faces_stay_untagged():
    m = _box(1, 1, 1)
    face_id = int(np.flatnonzero(hexmesh.boundary_face_ids(m))[0])
    m = hexmesh.tag_faces(m, [face_id], "wall")
    r = hexmesh.refine(m)
    assert r.face_tags.group_tags == ["wall"]
    assert len(r.face_tags) == 4                          # that one face's 4 children


def test_hexmesh_refine_watertight_and_conforming_across_a_multi_hex_mesh():
    m = _box(2, 2, 2)
    r = hexmesh.refine(m)
    assert hexmesh.is_watertight(r)
    assert hexmesh.is_conforming(r)


def test_hexmesh_refine_volume_is_exact_for_a_linear_mesh():
    m = _box(2, 2, 2)
    r = hexmesh.refine(m)
    assert hexmesh.volume(r) == pytest.approx(hexmesh.volume(m))


@pytest.mark.parametrize("order", [1, 2, 3])
def test_hexmesh_refine_geometry_fidelity(order):
    m = hexmesh.extrude(_ogrid(order), 3.0, 2)
    blocks = hex_blocks(m)
    r = hexmesh.refine(m)
    rblocks = hex_blocks(r)
    n = 6
    g = 0.5 * gll_nodes(n) + 0.5                            # octant 6 = bits (1,1,1)
    direct = resample_block_at(blocks[0:1], order, [g, g, g], 3)
    via_child = resample_block(rblocks[6:7], order, n, 3)   # child 6: octant (1,1,1)
    assert direct == pytest.approx(via_child, abs=1e-10)


def test_hexmesh_refine_curved_survival():
    m = _curved_hex()
    r = hexmesh.refine(m)
    assert hexmesh.is_watertight(r)
    assert hexmesh.is_conforming(r)
    outer = np.hypot(r.points[:, 0], r.points[:, 1]).max()
    assert outer == pytest.approx(1.0, abs=1e-12)
    assert hexmesh.quality_summary(r).n_inverted == 0


@pytest.mark.parametrize("order", [1, 2, 3])
def test_hexmesh_refine_preserves_order(order):
    m = hexmesh.extrude(_ogrid(order), 3.0, 2)
    assert hexmesh.refine(m).order == order


def test_hexmesh_refine_does_not_mutate_input():
    m = _box(2, 2, 2)
    pts = m.points.copy()
    hexmesh.refine(m)
    assert np.array_equal(pts, m.points)


def test_hexmesh_refine_composes_and_stays_conforming():
    """The case that exposed the real bug: a hex-interior split's own new edge (an
    edge-midpoint to a face-center) *is* one of that face's own quadmesh.refine
    spokes -- deduplicating it after the fact, rather than in the same pass, tears
    the mesh in a way invisible to a single level (it only shows up once the result
    is itself refined and the wrong edge count changes which faces the new octants
    can find)."""
    m = _curved_hex()
    r = hexmesh.refine(m)
    r2 = hexmesh.refine(r)
    assert r2.n_hexes == 64 * m.n_hexes
    assert hexmesh.is_watertight(r2)
    assert hexmesh.is_conforming(r2)
    assert hexmesh.quality_summary(r2).n_inverted == 0


def test_hexmesh_refine_quality_matches_the_parent():
    """Refining should not manufacture a fold that was not already there."""
    m = _curved_hex()
    parent_quality = hexmesh.quality_summary(m)
    r = hexmesh.refine(m)
    refined_quality = hexmesh.quality_summary(r)
    assert refined_quality.n_inverted == 0
    assert refined_quality.min == pytest.approx(parent_quality.min, abs=1e-6)
