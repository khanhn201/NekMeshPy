"""Tests for the toolkit generalization layer: physical groups, the shared-point
Mesh view, quality, and the section-smoothing registry (exercised on the
mesh the carotid example builds)."""

import numpy as np

from nekmeshpy import (
    SECTION_METHODS,
    PhysicalGroup,
    PhysicalGroups,
    export,
    hexmesh,
    linemesh,
    quadmesh,
    register_section_smoothing,
    set_section_smoothing,
)
from nekmeshpy.hexmesh import quality


def test_physical_groups_is_a_registry_with_no_presets():
    """The registry stores what a mesher tells it and knows nothing on its own: a
    name-to-code table is a statement about one piece of geometry, so it lives in the
    mesher next to the tags it names."""
    g = PhysicalGroups([PhysicalGroup("wall", 1, 2, "W  "),
                        PhysicalGroup("outlet", 4, 2, "O  ")])
    assert g.code_for(1) == "W  "
    assert g.name_for(4) == "outlet"
    assert g.tag_for("wall") == 1
    assert len(g) == 2
    assert not [m for m in dir(PhysicalGroups) if m in
                ("nek_default", "duct", "from_tags")]


def test_physical_group_pads_code():
    g = PhysicalGroups()
    grp = g.define("inlet", 9, code="v")
    assert grp.code == "v  "


def test_to_mesh_groups(built_mesh):
    m = export.to_mesh(built_mesh["mesh"], built_mesh["groups"])
    assert m.cells["hexahedron"].shape == (7200, 8)
    assert m.cells["quad"].shape == (1840, 4)
    assert set(m.cell_sets) >= {"wall", "trunk_outlet", "top_outlet_1", "top_outlet_2"}
    assert m.cell_sets["wall"]["quad"].size == 1440


def test_to_mesh_without_groups_cannot_orient_an_interior_plane(built_mesh):
    """With no registry there is no side rule, so a named *interior* face contributes
    the row each of its two hexes carries -- 160 more than the directed export. Which
    side a measurement plane belongs to is a property of the groups, not the mesh."""
    m = export.to_mesh(built_mesh["mesh"])
    assert m.cells["quad"].shape == (2000, 4)


def test_section_smoothing_registry_extensible():
    calls = {}

    @register_section_smoothing("noop_test")
    def _noop(qm, **opts):
        calls["hit"] = True
        return qm

    assert "noop_test" in SECTION_METHODS
    qm = quadmesh.structured(
        [linemesh.loft([(0, 0, 0), (1, 0, 0)]), linemesh.loft([(1, 0, 0), (1, 1, 0)]),
         linemesh.loft([(1, 1, 0), (0, 1, 0)]), linemesh.loft([(0, 1, 0), (0, 0, 0)])])
    set_section_smoothing(qm, "noop_test")
    assert calls.get("hit") is True
    del SECTION_METHODS["noop_test"]


def test_quality_module_matches_mesh(built_mesh):
    mesh = built_mesh["mesh"]
    X, HC, _ = hexmesh.weld(mesh)
    sj = quality.scaled_jacobian(X, HC)
    assert np.allclose(sj, hexmesh.scaled_jacobian(mesh))
    stats = quality.summary(X, HC)
    assert stats.n_inverted == 0
    # ``quality.summary`` reads corners -- it is handed points and connectivity, and a
    # welded array carries no curved nodes to read.  ``hexmesh.quality_summary``
    # defaults to the curved reading, so the two agree only when asked the same
    # question, and on this order-3 carotid they genuinely differ (0.121 vs 0.093).
    assert hexmesh.quality_summary(mesh, high_order=False) == stats
    assert hexmesh.quality_summary(mesh).min <= stats.min


# -- validate_layers: an int is n uniform layers ------------------------------

def test_int_layer_count_is_uniform_spacing():
    from nekmeshpy.core.fields import uniform_spacing, validate_layers
    # an int counts *layers* (cells), not positions -- 3 -> 4 positions
    assert np.array_equal(validate_layers(3, "who"), uniform_spacing(3))
    assert validate_layers(3, "who").size == 4


def test_int_layer_count_reaches_the_factories_bit_identically():
    from nekmeshpy.core.fields import uniform_spacing
    circ = linemesh.circle(1.0, 8)
    a = quadmesh.ogrid(circ, 2, uniform_spacing(2))
    b = quadmesh.ogrid(circ, 2, 2)
    assert a.points.tobytes() == b.points.tobytes()
    ha = hexmesh.extrude(a, axis=(0, 0, 1), length=1.0, layers=uniform_spacing(3))
    hb = hexmesh.extrude(b, axis=(0, 0, 1), length=1.0, layers=3)
    assert ha.points.tobytes() == hb.points.tobytes()


def test_layer_count_rejects_zero_and_floats():
    import pytest

    from nekmeshpy.core.fields import validate_layers
    with pytest.raises(ValueError):
        validate_layers(0, "who")            # zero layers is not a mesh
    with pytest.raises(ValueError):
        validate_layers(2.0, "who")          # a lone float is not a position array


# -- repr on the quad / hex / tri containers ----------------------------------

def test_repr_of_each_container_names_counts_and_tag_groups():
    from nekmeshpy import TriMesh
    section = quadmesh.ogrid(linemesh.circle(1.0, 8, element_tag="wall"), 2, 2)
    block = hexmesh.extrude(section, axis=(0, 0, 1), length=1.0, layers=2,
                            first_tag="inlet", last_tag="outlet")
    assert repr(section).startswith("<QuadMesh ")
    assert "order 1" in repr(section) and "edge_tags={wall}" in repr(section)
    assert repr(block).startswith("<HexMesh ")
    assert "face_tags={inlet,outlet,wall}" in repr(block)
    tri = TriMesh(np.zeros((4, 3)), [[0, 1, 2], [0, 2, 3]])
    assert repr(tri) == "<TriMesh 4 points, 2 tris>"
