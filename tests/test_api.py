"""Tests for the generalization layer: config (de)serialization + validation,
physical groups, the shared-node Mesh view, and the interior-method registry."""

import numpy as np
import pytest

from nekmeshpy import (
    INTERIOR_METHODS,
    Config,
    PhysicalGroups,
    export,
    quality,
    register_interior,
    set_interior,
)


def test_config_roundtrip_dict():
    cfg = Config()
    cfg.interior_method = "winslow"
    cfg2 = Config.from_dict(cfg.to_dict())
    assert cfg2.interior_method == "winslow"
    assert cfg2.radial == cfg.radial


def test_config_from_dict_rejects_unknown():
    with pytest.raises(ValueError):
        Config.from_dict({"not_a_field": 1})


def test_config_file_roundtrip(tmp_path):
    cfg = Config()
    cfg.n_slices = 12
    for ext in ("yaml", "json"):
        p = tmp_path / ("c." + ext)
        cfg.save(str(p))
        back = Config.from_file(str(p))
        assert back.n_slices == 12


@pytest.mark.parametrize("mutate,msg", [
    (lambda c: setattr(c, "n_half", 6), "n_half"),
    (lambda c: setattr(c, "radial", [0.4, 0.8, 0.9]), "radial"),
    (lambda c: setattr(c, "radial", [0.8, 0.4, 1.0]), "increasing"),
    (lambda c: setattr(c, "center_scale", 1.5), "center_scale"),
])
def test_config_validation_catches(mutate, msg):
    cfg = Config()
    mutate(cfg)
    with pytest.raises(ValueError) as e:
        cfg.validate()
    assert msg in str(e.value)


def test_default_config_valid():
    Config().validate()


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
    assert m.cells["hexahedron"].shape == (3840, 8)
    assert m.cells["quad"].shape == (1280, 4)
    assert set(m.cell_sets) >= {"wall", "trunk_outlet", "top_outlet_1", "top_outlet_2"}
    assert m.cell_sets["wall"]["quad"].size == 960


def test_interior_registry_extensible(built_mesh):
    calls = {}

    @register_interior("noop_test")
    def _noop(mesh, twall, **opts):
        calls["hit"] = True
        return mesh

    assert "noop_test" in INTERIOR_METHODS
    set_interior(built_mesh["mesh"], "noop_test", 1)
    assert calls.get("hit") is True
    del INTERIOR_METHODS["noop_test"]


def test_quality_module_matches_mesh(built_mesh):
    mesh = built_mesh["mesh"]
    X, HC, _ = mesh.weld()
    sj = quality.scaled_jacobian(X, HC)
    assert np.allclose(sj, quality.scaled_jacobian(*mesh.weld()[:2]))
    stats = quality.summary(X, HC)
    assert stats["n_inverted"] == 0
