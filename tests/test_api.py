"""Tests for the toolkit generalization layer: physical groups, the shared-point
Mesh view, quality, and the section-smoothing registry (exercised on the
mesh the bifurcation example builds)."""

import numpy as np

from nekmeshpy import (
    SECTION_METHODS,
    LineMesh,
    PhysicalGroups,
    QuadMesh,
    export,
    register_section_smoothing,
    set_section_smoothing,
)
from nekmeshpy.hexmesh import quality


def test_physical_groups_default_codes():
    g = PhysicalGroups.nek_default()
    assert g.code_for(1) == "W  "
    assert g.name_for(4) == "top_outlet_2"
    assert g.tag_for("wall") == 1
    assert len(g) == 6


def test_physical_group_pads_code():
    g = PhysicalGroups()
    grp = g.define("inlet", 9, code="v")
    assert grp.code == "v  "


def test_to_mesh_groups(built_mesh):
    m = export.to_mesh(built_mesh["mesh"])
    assert m.cells["hexahedron"].shape == (4800, 8)
    assert m.cells["quad"].shape == (1360, 4)
    assert set(m.cell_sets) >= {"wall", "trunk_outlet", "top_outlet_1", "top_outlet_2"}
    assert m.cell_sets["wall"]["quad"].size == 960


def test_section_smoothing_registry_extensible():
    calls = {}

    @register_section_smoothing("noop_test")
    def _noop(qm, **opts):
        calls["hit"] = True
        return qm

    assert "noop_test" in SECTION_METHODS
    qm = QuadMesh.structured(
        [LineMesh.loft([(0, 0, 0), (1, 0, 0)]), LineMesh.loft([(1, 0, 0), (1, 1, 0)]),
         LineMesh.loft([(1, 1, 0), (0, 1, 0)]), LineMesh.loft([(0, 1, 0), (0, 0, 0)])])
    set_section_smoothing(qm, "noop_test")
    assert calls.get("hit") is True
    del SECTION_METHODS["noop_test"]


def test_quality_module_matches_mesh(built_mesh):
    mesh = built_mesh["mesh"]
    X, HC, _ = mesh.weld()
    sj = quality.scaled_jacobian(X, HC)
    assert np.allclose(sj, mesh.scaled_jacobian())
    stats = quality.summary(X, HC)
    assert stats["n_inverted"] == 0
    assert mesh.quality_summary() == stats
