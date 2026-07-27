"""Run the flat flow-past-a-body example scripts and check the meshes they build.

These are tolerance-only (no goldens, like the pipe examples): each script is a
sensible external-flow domain around a body, and the checks assert the mesh is a
single watertight/conformal all-hex block with positive Jacobians and the expected
named boundary groups (inlet/outlet + far field + the body)."""

import numpy as np
import pytest
from conftest import run_example

from nekmeshpy import quality, topology


def _scaled_jac(mesh):
    return quality.scaled_jacobian(*mesh.weld()[:2])


def _tag_count(mesh, name):
    return int(np.sum(mesh.boundary_names == name))


def _assert_valid_flow_block(mesh, *, body, jac_floor, groups):
    # one watertight, conformal, positively-oriented block
    assert mesh.is_watertight() and mesh.is_conforming()
    assert topology.hex_report(*mesh.weld()[:2])["n_components"] == 1
    assert float(np.min(_scaled_jac(mesh))) > jac_floor
    # exactly the expected named groups, all non-empty
    assert set(mesh.boundary_group_names) == groups
    for name in groups:
        assert _tag_count(mesh, name) > 0
    # inlet and outlet flow openings are present and the body is embedded
    assert _tag_count(mesh, "inlet") > 0 and _tag_count(mesh, "outlet") > 0
    assert _tag_count(mesh, body) > 0


def test_backward_facing_step(tmp_path):
    mesh = run_example("backward_facing_step.py", tmp_path)["mesh"]
    # axis-aligned structured blocks -> exact unit Jacobian
    assert float(np.min(_scaled_jac(mesh))) == pytest.approx(1.0, abs=1e-9)
    _assert_valid_flow_block(
        mesh, body="wall", jac_floor=0.99,
        groups={"inlet", "outlet", "wall", "front", "back"})


def test_flow_past_cylinder(tmp_path):
    mesh = run_example("flow_past_cylinder.py", tmp_path)["mesh"]
    _assert_valid_flow_block(
        mesh, body="cylinder", jac_floor=0.3,
        groups={"inlet", "outlet", "cylinder", "top", "bottom", "front", "back"})
    # the ring is azimuthally symmetric -> inlet/outlet/top/bottom face counts match
    for name in ("outlet", "top", "bottom"):
        assert _tag_count(mesh, name) == _tag_count(mesh, "inlet")


def test_flow_past_plate(tmp_path):
    mesh = run_example("flow_past_plate.py", tmp_path)["mesh"]
    # thin-ellipse O-grid: sharp ends are skewed but stay well clear of inverted
    _assert_valid_flow_block(
        mesh, body="plate", jac_floor=0.1,
        groups={"inlet", "outlet", "plate", "top", "bottom", "front", "back"})


def test_flow_past_half_cylinder(tmp_path):
    mesh = run_example("flow_past_half_cylinder.py", tmp_path)["mesh"]
    _assert_valid_flow_block(
        mesh, body="wall", jac_floor=0.2,
        groups={"inlet", "outlet", "wall", "top", "front", "back"})


def test_flow_past_sphere(tmp_path):
    mesh = run_example("flow_past_sphere.py", tmp_path)["mesh"]
    _assert_valid_flow_block(
        mesh, body="sphere", jac_floor=0.4,
        groups={"inlet", "outlet", "sphere", "top", "bottom", "front", "back"})
    # cubed-sphere shell: all six box faces carry the same face count
    for name in ("outlet", "top", "bottom", "front", "back"):
        assert _tag_count(mesh, name) == _tag_count(mesh, "inlet")


def test_flow_past_hemisphere(tmp_path):
    mesh = run_example("flow_past_hemisphere.py", tmp_path)["mesh"]
    _assert_valid_flow_block(
        mesh, body="hemisphere", jac_floor=0.4,
        groups={"inlet", "outlet", "hemisphere", "ground", "top", "front", "back"})
    # the ground annulus (z=0) is present and distinct from the body
    assert _tag_count(mesh, "ground") > 0
