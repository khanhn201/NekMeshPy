"""Tests for the toolkit generalization layer: boundary conditions, the shared-point
Mesh view, quality, and the section-smoothing registry (exercised on the
mesh the carotid example builds)."""

import numpy as np
import pytest

from nekmeshpy import (
    BoundaryConditions,
    hexmesh,
    linemesh,
    quadmesh,
    writer,
)
from nekmeshpy.hexmesh import quality


def test_boundary_conditions_is_a_registry_with_no_presets():
    """The registry stores what a mesher tells it and knows nothing on its own: a
    name-to-code table is a statement about one piece of geometry, so it lives in the
    mesher next to the tags it names."""
    g = BoundaryConditions({"wall": "W", "outlet": "O  "})
    assert g.fields == ("velocity",)
    assert g.side("velocity", "wall", "") == ("W  ", 0)
    assert len(g) == 2 and "wall" in g and "inlet" not in g
    assert not [m for m in dir(BoundaryConditions) if m in
                ("nek_default", "duct", "from_tags")]


def test_cbc_and_boundary_id_default_when_the_table_states_none():
    side = lambda g, n: g.side("velocity", n, "")   # noqa: E731
    only_cbc = BoundaryConditions({"a": "W", "b": ("v", None)})
    assert side(only_cbc, "a") == ("W  ", 0) and side(only_cbc, "b") == ("v  ", 0)
    only_id = BoundaryConditions({"a": 1, "b": (None, 2)})
    assert side(only_id, "a") == ("E  ", 1) and side(only_id, "b") == ("E  ", 2)
    both = BoundaryConditions({"a": ("W", 1), "b": ("v", 1)})   # names may share an id
    assert side(both, "a") == ("W  ", 1) and side(both, "b") == ("v  ", 1)
    assert BoundaryConditions({"a": (None, None)}).side("velocity", "a", "") == ("E  ", 0)


def test_cbc_and_boundary_id_are_all_or_none_per_table():
    with pytest.raises(ValueError, match="cbc.*Missing: b"):
        BoundaryConditions({"a": ("W", 1), "b": (None, 2)})
    with pytest.raises(ValueError, match="boundaryID.*Missing: b"):
        BoundaryConditions({"a": ("W", 1), "b": "v"})
    # the temperature table is judged on its own: it may state none at all
    BoundaryConditions({"a": ("W", 1)}, temperature={"a": "I"})


def test_boundary_ids_run_from_one_without_a_gap_per_field():
    with pytest.raises(ValueError, match="velocity.*1..k.*\\[2, 3\\]"):
        BoundaryConditions({"a": 2, "b": 3})
    with pytest.raises(ValueError, match="temperature.*\\[1, 3\\]"):
        BoundaryConditions({"a": 1}, temperature={"a": 1, "b": 3})
    BoundaryConditions({"a": 1, "b": 2}, temperature={"a": 1, "b": 1})   # separate sets


def test_a_stated_zero_and_E_clear_the_boundary():
    g = BoundaryConditions({"a": ("W", 1), "b": ("E", 0), "c": ("E  ", 2)})
    assert g.side("velocity", "a", "") == ("W  ", 1)
    assert g.side("velocity", "b", "") == ("E  ", 0)
    assert g.numbered("velocity")        # b's 0 is stated, so no positional fallback
    assert not BoundaryConditions({"a": "W"}).numbered("velocity")


def test_boundary_condition_per_region_sides():
    g = BoundaryConditions({"iface": {"fluid": ("W", 1), "solid": None}, "x": ("v", 2)})
    assert g.side("velocity", "iface", "fluid") == ("W  ", 1)
    assert g.side("velocity", "iface", "solid") is None
    assert g.names_with("velocity", "W  ") == {"iface"}
    with pytest.raises(ValueError, match="region"):
        g.side("velocity", "iface", "")


def test_boundary_conditions_fields_are_separate_tables():
    g = BoundaryConditions({"inlet": "v", "wall": "W"},
                           temperature={"wall": ("I", 1), "outer": ("f", 2)})
    assert g.fields == ("velocity", "temperature")
    assert list(g) == ["inlet", "wall", "outer"]
    assert g.has("temperature", "outer") and not g.has("velocity", "outer")
    assert g.side("temperature", "wall", "") == ("I  ", 1)
    assert g.side("velocity", "wall", "") == ("W  ", 0)
    assert g.tag_of("velocity", "wall") == 2 and g.tag_of("temperature", "wall") == 1


def test_to_mesh_groups(built_mesh):
    m = writer.to_mesh(built_mesh["mesh"], built_mesh["groups"])
    assert m.cells["hexahedron"].shape == (7200, 8)
    assert m.cells["quad"].shape == (1840, 4)
    assert set(m.cell_sets) >= {"wall", "trunk_outlet", "top_outlet_1", "top_outlet_2"}
    assert m.cell_sets["wall"]["quad"].size == 1440


def test_to_mesh_without_groups_cannot_orient_an_interior_plane(built_mesh):
    """With no registry there is no side rule, so a named *interior* face contributes
    the row each of its two hexes carries -- 160 more than the directed export. Which
    side a measurement plane belongs to is a property of the groups, not the mesh."""
    m = writer.to_mesh(built_mesh["mesh"])
    assert m.cells["quad"].shape == (2000, 4)


def test_quality_module_matches_mesh(built_mesh):
    mesh = built_mesh["mesh"]
    X, HC = mesh.points, mesh.corners
    # the public readings are curved-only now, so the module twin to compare against is
    # the mesh-taking pair, not the bare-array corner one
    assert np.allclose(quality.scaled_jacobian(mesh, mesh.order),
                       hexmesh.scaled_jacobian(mesh))
    assert quality.summary(mesh, mesh.order) == hexmesh.quality_summary(mesh)
    # the corner reading still exists inside ``quality`` and still disagrees -- on this
    # order-3 carotid it is the more flattering of the two, which is the whole reason it
    # is no longer reachable from the rung namespace
    corner = quality.corner_summary(X, HC)
    assert corner.n_inverted == 0
    assert hexmesh.quality_summary(mesh).min <= corner.min


# -- validate_layers: an int is n uniform layers ------------------------------

def test_int_layer_count_is_uniform_spacing():
    from nekmeshpy.core.fields import uniform_spacing, validate_layers
    # an int counts *layers* (cells), not positions -- 3 -> 4 positions
    assert np.array_equal(validate_layers(3), uniform_spacing(3))
    assert validate_layers(3).size == 4


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
        validate_layers(0)            # zero layers is not a mesh
    with pytest.raises(ValueError):
        validate_layers(2.0)          # a lone float is not a position array


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
