"""Run the flat flow-past-a-body example scripts and check the meshes they build.

These are tolerance-only (no goldens, like the pipe examples): each script is a
sensible external-flow domain around a body, and the checks assert the mesh is a
single watertight/conformal all-hex block with positive Jacobians and the expected
named boundary groups (inlet/outlet + far field + the body)."""

import numpy as np
import pytest
from conftest import conformal, run_example

from nekmeshpy import hexmesh, topology
from nekmeshpy.hexmesh import quality
from nekmeshpy.model.interp import hex_face_indices


def _scaled_jac(mesh):
    return quality.scaled_jacobian(*hexmesh.query.weld(mesh)[:2])


def _wall_nodes(mesh, name):
    """Every high-order node on the boundary faces tagged ``name``, ``(K,3)``.

    Walks the conformal (welded) node array, so these are the nodes a high-order
    ``.vtu`` export actually writes -- corners *and* the shared edge/face nodes.
    A wall built from an analytic factory must have all of them on the true
    surface; one built by sampling points and subdividing straight will not.
    """
    nodes, conn_ho = conformal(mesh)
    order = mesh.order
    faces = mesh.face_tags.select(mesh.face_tags.mask_for(name))
    assert len(faces) > 0, "no boundary faces tagged %r" % name
    idx = {f: hex_face_indices(f, order) for f in range(1, 7)}
    picked = np.concatenate([conn_ho[e][idx[int(f)]] for e, f, _t in faces])
    return nodes[np.unique(picked)]


def _tag_count(mesh, name):
    return mesh.face_tags.count(name)


def _assert_valid_flow_block(mesh, *, body, jac_floor, groups):
    # one watertight, conformal, positively-oriented block
    assert hexmesh.query.is_watertight(mesh) and hexmesh.query.is_conforming(mesh)
    assert topology.hex_report(*hexmesh.query.weld(mesh)[:2]).n_components == 1
    assert float(np.min(_scaled_jac(mesh))) > jac_floor
    # exactly the expected named groups, all non-empty
    assert set(mesh.face_group_tags) == groups
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
    ns = run_example("flow_past_cylinder.py", tmp_path)
    mesh = ns["mesh"]
    # the body is LineMesh.circle, so *every* high-order wall node is on the exact
    # circle -- not just the corners (a straight-subdivided wall would be off by the
    # chord sagitta, R*(1-cos(pi/N_THETA)) ~ 1e-3 here)
    assert ns["ORDER"] > 1
    w = _wall_nodes(mesh, "cylinder")
    dev = np.abs(np.hypot(w[:, 0], w[:, 1]) - ns["R"])
    assert float(np.max(dev)) < 1e-12
    _assert_valid_flow_block(
        mesh, body="cylinder", jac_floor=0.3,
        groups={"inlet", "outlet", "cylinder", "top", "bottom", "front", "back"})
    # the ring is azimuthally symmetric -> inlet/outlet/top/bottom face counts match
    for name in ("outlet", "top", "bottom"):
        assert _tag_count(mesh, name) == _tag_count(mesh, "inlet")


def test_flow_past_plate(tmp_path):
    ns = run_example("flow_past_plate.py", tmp_path)
    mesh = ns["mesh"]
    # the ellipse has no analytic LineMesh factory, so the example places the
    # high-order nodes itself -- check they really landed on the exact ellipse
    assert ns["ORDER"] > 1
    w = _wall_nodes(mesh, "plate")
    dev = np.abs((w[:, 0] / ns["A"]) ** 2 + (w[:, 1] / ns["B"]) ** 2 - 1.0)
    assert float(np.max(dev)) < 1e-12
    # thin-ellipse O-grid: sharp ends are skewed but stay well clear of inverted
    _assert_valid_flow_block(
        mesh, body="plate", jac_floor=0.1,
        groups={"inlet", "outlet", "plate", "top", "bottom", "front", "back"})


def test_flow_past_half_cylinder(tmp_path):
    ns = run_example("flow_past_half_cylinder.py", tmp_path)
    mesh = ns["mesh"]
    # the bottom is two LineMesh.line runs welded to a LineMesh.arc, and
    # QuadMesh.structured stamps each edge's own high-order nodes onto the block --
    # so every "wall" node is either exactly on the floor or exactly on the bump
    assert ns["ORDER"] > 1
    w = _wall_nodes(mesh, "wall")
    r = np.hypot(w[:, 0], w[:, 1])
    on_floor = np.abs(w[:, 1])
    on_bump = np.abs(r - ns["R"])
    assert float(np.max(np.minimum(on_floor, on_bump))) < 1e-12
    # and the bump really is resolved: nodes strictly above the floor exist and all
    # of them sit on the exact circle (straight subdivision would miss by ~1e-3)
    lifted = w[w[:, 1] > 1e-9]
    assert lifted.shape[0] > 2 * ns["N_BUMP"]
    assert float(np.max(np.abs(np.hypot(lifted[:, 0], lifted[:, 1]) - ns["R"]))) < 1e-12
    _assert_valid_flow_block(
        mesh, body="wall", jac_floor=0.2,
        groups={"inlet", "outlet", "wall", "top", "front", "back"})


def test_flow_past_sphere(tmp_path):
    ns = run_example("flow_past_sphere.py", tmp_path)
    mesh = ns["mesh"]
    assert ns["ORDER"] > 1
    w = _wall_nodes(mesh, "sphere")
    assert float(np.max(np.abs(np.linalg.norm(w, axis=1) - ns["R"]))) < 1e-12
    _assert_valid_flow_block(
        mesh, body="sphere", jac_floor=0.4,
        groups={"inlet", "outlet", "sphere", "top", "bottom", "front", "back"})
    # cubed-sphere shell: all six box faces carry the same face count
    for name in ("outlet", "top", "bottom", "front", "back"):
        assert _tag_count(mesh, name) == _tag_count(mesh, "inlet")


def test_flow_past_hemisphere(tmp_path):
    ns = run_example("flow_past_hemisphere.py", tmp_path)
    mesh = ns["mesh"]
    # QuadMesh.hemisphere projects every node -- corners, edge and face nodes -- so
    # the whole high-order body wall is on the exact sphere, and it sits on z = 0
    assert ns["ORDER"] > 1
    w = _wall_nodes(mesh, "hemisphere")
    assert float(np.max(np.abs(np.linalg.norm(w, axis=1) - ns["R"]))) < 1e-12
    assert float(np.min(w[:, 2])) > -1e-14
    assert float(np.max(np.abs(_wall_nodes(mesh, "ground")[:, 2]))) < 1e-14
    _assert_valid_flow_block(
        mesh, body="hemisphere", jac_floor=0.4,
        groups={"inlet", "outlet", "hemisphere", "ground", "top", "front", "back"})
    # the ground annulus (z=0) is present and distinct from the body
    assert _tag_count(mesh, "ground") > 0
