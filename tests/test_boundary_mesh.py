"""Unit tests for the delta -1 rung: ``hexmesh.boundary_mesh`` (a block's boundary as a
``QuadMesh``) and ``quadmesh.boundary_mesh`` (a section's as a ``LineMesh``).

The property that matters, and the reason this direction stopped being empty: the
extracted surface carries the **parent's own nodes**, bit for bit, at any order.  A
connector built off a port has to start from that port's real nodes -- re-deriving them
from the recipe that built the block lands close, and at ``order > 1``
``HexMesh.merge`` verifies shared high-order edge and face nodes against
``conform.entity_tol`` (~1e-9 of the model extent), where close fails.

So the assertions here are ``== 0.0``, not tolerances, and they are made against the
parent's **conformal node set** rather than its corners: a corner-only check passes on
a mesh that is high-order in storage and linear in geometry, which is exactly the
failure worth guarding.
"""

import numpy as np
import pytest
from scipy.spatial import cKDTree

from nekmeshpy import hexmesh, linemesh, quadmesh
from nekmeshpy.core import conform
from nekmeshpy.core.fields import uniform_spacing

ORDERS = [1, 2, 3]


def _hex_nodes(m):
    return conform.conformal_hex(m.points, m.hexes, m._elem_edges, m._edge_flip,
                                 m.quad_mesh.line_mesh.interior, m.hex, m.orient,
                                 m.quad_mesh.interior, m.interior, m.order)[0]


def _quad_nodes(m):
    return conform.conformal_quad(m.points, m.quads, m.quad, m.orient,
                                  m.line_mesh.interior, m.interior, m.order)[0]


def _line_nodes(m):
    return conform.conformal_line(m.points, m.lines, m.interior, m.order)[0]


def _section(order):
    ring = linemesh.circle(0.5, 8, element_tag="wall", order=order)
    return quadmesh.ogrid(ring, 2, uniform_spacing(2), wall_tag="wall")


def _block(order):
    return hexmesh.extrude(_section(order), 2.0, 3,
                           first_tag="inlet", last_tag="outlet")


def _all_on(parent_nodes, child_nodes):
    """Worst distance from any child node to the nearest parent node."""
    t = cKDTree(np.asarray(parent_nodes).reshape(-1, 3))
    return float(t.query(np.asarray(child_nodes).reshape(-1, 3))[0].max())


# -- hexmesh.boundary_mesh ----------------------------------------------------
@pytest.mark.parametrize("order", ORDERS)
@pytest.mark.parametrize("tag", ["inlet", "wall", None])
def test_extracted_surface_sits_exactly_on_the_parent(order, tag):
    block = _block(order)
    surf = hexmesh.boundary_mesh(block, tag)
    assert _all_on(_hex_nodes(block), _quad_nodes(surf)) == 0.0


@pytest.mark.parametrize("order", ORDERS)
def test_extracted_surface_inherits_the_parents_order(order):
    assert hexmesh.boundary_mesh(_block(order), "inlet").order == order


def test_the_whole_boundary_is_every_tagged_group_together():
    block = _block(2)
    whole = hexmesh.boundary_mesh(block)
    parts = sum(hexmesh.boundary_mesh(block, t).n_quads
                for t in ("inlet", "outlet", "wall"))
    assert whole.n_quads == parts == len(hexmesh.boundary_faces(block))
    assert sorted(whole.element_group_tags) == ["inlet", "outlet", "wall"]


def test_a_named_group_is_tagged_with_its_own_name():
    surf = hexmesh.boundary_mesh(_block(2), "inlet")
    assert surf.element_group_tags == ["inlet"]


def test_the_extracted_surface_has_its_own_index_space():
    """Compacted from the parent's ids, not a view of them."""
    block = _block(2)
    surf = hexmesh.boundary_mesh(block, "inlet")
    assert surf.n_points < block.n_points
    assert surf.quads.max() == surf.n_points - 1


def test_the_whole_boundary_of_a_watertight_block_is_a_closed_surface():
    surf = hexmesh.boundary_mesh(_block(2))
    assert len(quadmesh.boundary_edges(surf)) == 0


def test_extracted_surface_can_be_built_back_onto_its_parent():
    """The point of the operation: a piece grown off the port welds at order > 1."""
    block = _block(2)
    port = hexmesh.boundary_mesh(block, "outlet")
    stub = hexmesh.extrude(port, 0.5, 2, axis=(0.0, 0.0, 1.0), last_tag="outlet")
    block = hexmesh.retag_face(block, {"outlet": ""})
    rep = hexmesh.topology_report(hexmesh.merge([block, stub]))
    assert rep.watertight and rep.conformal and rep.n_components == 1


def test_unknown_tag_names_what_is_available():
    with pytest.raises(ValueError, match="no face carries the tag 'nope'"):
        hexmesh.boundary_mesh(_block(1), "nope")


# -- the template form --------------------------------------------------------
@pytest.mark.parametrize("order", ORDERS)
def test_template_keeps_its_numbering_and_takes_the_parents_coordinates(order):
    block = _block(order)
    template = quadmesh.translate(_section(order), (0.0, 0.0, 2.0))   # the outlet plane
    surf = hexmesh.boundary_mesh(block, "outlet", template=template)
    assert np.array_equal(surf.quad, template.quad)
    assert np.array_equal(surf.orient, template.orient)
    assert _all_on(_hex_nodes(block), _quad_nodes(surf)) == 0.0


def test_template_result_pairs_index_for_index_with_its_template():
    """Why the template form exists: ``adapter`` / ``bridge`` / ``blend`` all require
    identical connectivity paired by index.  The template form *guarantees* that; the
    plain form only numbers the surface consistently with itself, and whether that
    happens to agree with some other section is not something a caller can rely on."""
    block = _block(2)
    template = quadmesh.translate(_section(2), (0.0, 0.0, 2.0))
    surf = hexmesh.boundary_mesh(block, "outlet", template=template)
    mid = quadmesh.blend(template, surf, [0.0, 1.0])      # would raise on a mismatch
    assert np.array_equal(mid[-1].points, surf.points)
    assert np.array_equal(surf.quads, template.quads)


def test_template_that_is_not_this_ports_pattern_is_refused():
    block = _block(2)
    tiny = quadmesh.translate(_section(2), (0.0, 0.0, 2.0))
    tiny.points[:] = tiny.points.mean(axis=0)             # collapse it onto one point
    with pytest.raises(ValueError, match="does not pair one-for-one"):
        hexmesh.boundary_mesh(block, "outlet", template=tiny)


# -- quadmesh.boundary_mesh ---------------------------------------------------
@pytest.mark.parametrize("order", ORDERS)
def test_extracted_loop_sits_exactly_on_the_parent_section(order):
    sec = _section(order)
    loop = quadmesh.boundary_mesh(sec, "wall")
    assert _all_on(_quad_nodes(sec), _line_nodes(loop)) == 0.0
    assert loop.order == order


def test_extracted_loop_of_a_disc_is_closed():
    loop = quadmesh.boundary_mesh(_section(2))
    assert len(linemesh.boundary_points(loop)) == 0        # a cycle has no free end


def test_extracted_loop_carries_its_edge_tag():
    assert quadmesh.boundary_mesh(_section(2), "wall").element_group_tags == ["wall"]


def test_lower_is_reachable_as_a_namespace_module():
    assert hexmesh.lower.boundary_mesh is hexmesh.boundary_mesh
    assert quadmesh.lower.boundary_mesh is quadmesh.boundary_mesh
