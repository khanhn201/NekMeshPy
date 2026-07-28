"""Run the flat pipe example scripts and check the meshes they produce."""

import numpy as np
import pytest
from conftest import run_example

from nekmeshpy import topology
from nekmeshpy.hexmesh import quality


def _scaled_jac(mesh):
    return quality.scaled_jacobian(*mesh.weld()[:2])


def _tag_count(mesh, name):
    return int(np.sum(mesh.boundary_tags == name))


def test_circular_pipe(tmp_path):
    mesh = run_example("circular_pipe.py", tmp_path)["mesh"]
    # wall sides + inlet / outlet caps all present
    assert _tag_count(mesh, "wall") > 0
    assert (_tag_count(mesh, "inlet") > 0
            and _tag_count(mesh, "inlet") == _tag_count(mesh, "outlet"))
    # butterfly O-grid: no collapsed centre cell, all positive Jacobian
    assert float(np.min(_scaled_jac(mesh))) > 0.5
    assert mesh.is_watertight() and mesh.is_conforming()
    assert set(mesh.boundary_group_tags) >= {"wall", "inlet", "outlet"}


def test_rectangular_pipe(tmp_path):
    mesh = run_example("rectangular_pipe.py", tmp_path)["mesh"]
    # a structured axis-aligned duct is exact -> scaled Jacobian 1 everywhere
    assert float(np.min(_scaled_jac(mesh))) == pytest.approx(1.0, abs=1e-9)
    assert mesh.is_watertight() and mesh.is_conforming()
    assert _tag_count(mesh, "inlet") == _tag_count(mesh, "outlet")  # caps match
    assert set(mesh.boundary_group_tags) >= {"wall", "inlet", "outlet"}


def test_circular_pipe_tjunction(tmp_path):
    mesh = run_example("circular_pipe_tjunction.py", tmp_path)["mesh"]
    # three legs welded into one conformal, watertight block at the junction
    assert mesh.is_watertight() and mesh.is_conforming()
    assert topology.hex_report(*mesh.weld()[:2])["n_components"] == 1
    # wall + three circular openings, each opening the same face count
    assert _tag_count(mesh, "wall") > 0
    for name in ("outlet", "branch"):
        assert _tag_count(mesh, name) == _tag_count(mesh, "inlet")
    # analytic O-grid junction stays well away from degenerate
    assert float(np.min(_scaled_jac(mesh))) > 0.3
    assert set(mesh.boundary_group_tags) >= {"wall", "inlet", "outlet", "branch"}
