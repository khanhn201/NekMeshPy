"""High-order rebuild: curved edge/face nodes carried to the smoothed vertices and re-projected onto the targets."""

import numpy as np
import pytest
import surfaces as S

from nekmeshpy import linemesh, quadmesh
from nekmeshpy.core import interp
from nekmeshpy.quadmesh.smoother import constrain as cn
from nekmeshpy.quadmesh.smoother import highorder as ho
from nekmeshpy.quadmesh.smoother import smooth
from nekmeshpy.quadmesh.smoother.session import SmoothSession


def _disk(order):
    return quadmesh.ogrid(linemesh.circle(1.0, 24, element_tag="wall", order=order), 6, 4, wall_tag="wall")


def _smoothed_disk(order=4, seed=3):
    sec = _disk(order)
    lin = ho.linearize(sec)
    jit = smooth.with_points(lin, lin.points + np.random.default_rng(seed).normal(size=lin.points.shape) * [0.03, 0.03, 0])
    r = cn.resolve(jit, [cn.OnCurve("wall", cn.circle([0, 0, 0], 1.0))])
    nrm = smooth.reference_normals(jit.points, jit.corners)
    sm, logs = smooth.run_schedule(jit, r, smooth.suggest_schedule(smooth.quality(jit.points, jit.corners, nrm)))
    return sec, sm, r, logs


@pytest.mark.parametrize("order", [2, 3, 4])
def test_unchanged_corners_give_back_the_input_bit_for_bit(order):
    sec = _disk(order)
    out, rep = ho.rebuild_high_order(sec, sec.points)
    np.testing.assert_array_equal(quadmesh.element_blocks(out), quadmesh.element_blocks(sec))
    assert rep.max_untouched_move == 0.0 and out.order == order


def test_a_rigid_move_of_the_corners_carries_every_node_with_its_element():
    sec = _disk(4)
    v = np.array([0.3, -0.2, 0.0])
    out, _ = ho.rebuild_high_order(sec, sec.points + v)
    np.testing.assert_allclose(quadmesh.element_blocks(out), quadmesh.element_blocks(sec) + v, atol=1e-13)


def test_linearize_is_order_one_and_keeps_connectivity_and_every_tag():
    sec = _disk(4)
    lin = ho.linearize(sec)
    assert lin.order == 1 and sec.order == 4
    np.testing.assert_array_equal(lin.corners, sec.corners)
    np.testing.assert_array_equal(lin.points, sec.points)
    assert list(lin.edge_tags.flatnonzero("wall")) == list(sec.edge_tags.flatnonzero("wall"))
    assert lin.points is not sec.points


def test_the_curved_boundary_comes_back_onto_the_circle_after_smoothing():
    sec, sm, r, logs = _smoothed_disk(4)
    assert logs[0].before.n_inverted > 0 and logs[-1].after.n_inverted == 0
    out, rep = ho.rebuild_high_order(sec, sm.points, r)
    wall = out.edge_tags.flatnonzero("wall")
    assert len(wall) == 24 and list(wall) == list(out.edge_tags.flatnonzero("wall"))
    rad = np.linalg.norm(out.line_mesh.interior[wall].reshape(-1, 3), axis=1)
    np.testing.assert_allclose(rad, 1.0, atol=1e-13)                              # on the circle
    carry, _ = ho.rebuild_high_order(sec, sm.points, None)
    rad_c = np.linalg.norm(carry.line_mesh.interior[carry.edge_tags.flatnonzero("wall")].reshape(-1, 3), axis=1)
    assert np.abs(rad_c - 1).max() > 1e-4                                         # the carry alone is off: projection is needed
    np.testing.assert_array_equal(out.points, sm.points)                          # corners are exactly the smoothed ones
    assert quadmesh.is_conforming(out) and rep.n_inverted_sampled == 0
    assert rep.sj_min_rebuilt > 0.5 and rep.max_distance_after < 1e-13 and rep.max_distance_before > 1e-4
    assert list(out.element_tags.to_dense()) == list(sec.element_tags.to_dense())


def test_elements_whose_corners_did_not_move_are_untouched_bit_for_bit():
    sec = _disk(3)
    lin = ho.linearize(sec)
    sel = np.zeros(lin.n_quads, bool)
    sel[[5, 6, 7]] = True
    touched = np.unique(lin.corners[sel])
    P = lin.points.copy()
    P[touched] += np.random.default_rng(1).normal(size=(len(touched), 3)) * [0.02, 0.02, 0]   # only these vertices are perturbed
    jit = smooth.with_points(lin, P)
    r = cn.resolve(jit, [], boundary="free")
    nrm = smooth.reference_normals(lin.points, lin.corners)
    sm, _ = smooth.optimize(jit, r, iters=10, normals=nrm, elements=sel)
    out, rep = ho.rebuild_high_order(sec, sm.points, r)
    same = (sm.points[sm.corners] == sec.points[sec.corners]).all(axis=(1, 2))
    assert same.any() and not same.all()
    assert rep.max_untouched_move == 0.0
    np.testing.assert_array_equal(quadmesh.element_blocks(out)[same], quadmesh.element_blocks(sec)[same])


@pytest.mark.parametrize("name,order", [("sphere", 4), ("torus", 3)])
def test_lifting_a_smoothed_linear_surface_puts_every_node_on_the_surface(name, order):
    an, m, ref = S.jittered_case(name)
    nrm = smooth.reference_normals(ref.points, ref.corners)
    r = cn.resolve(m, [cn.OnSurface(np.arange(m.n_points), an.surface)], boundary="free")
    sm, _ = smooth.run_schedule(m, r, smooth.suggest_schedule(smooth.quality(m.points, m.corners, nrm)), normals=nrm)
    out, rep = ho.rebuild_high_order(sm, sm.points, r, order=order)
    assert out.order == order and quadmesh.is_conforming(out)
    nodes = np.concatenate([out.line_mesh.interior.reshape(-1, 3), out.interior.reshape(-1, 3)])
    assert np.linalg.norm(an.surface.project(nodes) - nodes, axis=1).max() < 1e-10    # every node on the surface
    assert rep.n_edge_nodes_projected > 0 and rep.n_face_nodes_projected > 0 and rep.n_inverted_sampled == 0
    if name == "sphere":                                      # the curved mesh represents the sphere far better than the chords
        exact = 4 * np.pi
        assert abs(quadmesh.area(out, high_order=True) - exact) < 0.2 * abs(quadmesh.area(sm) - exact)


def test_raising_the_order_is_a_change_of_basis_and_keeps_the_curvature():
    sec = _disk(3)
    out, rep = ho.rebuild_high_order(sec, sec.points, order=5)
    want = interp.resample_block(quadmesh.element_blocks(sec), 3, 5, 2)
    np.testing.assert_allclose(quadmesh.element_blocks(out), want, atol=1e-13)
    assert out.order == 5 and any("lifted" in n for n in rep.notes)


def test_a_straight_sided_mesh_can_be_lifted_and_projected():
    sec = _disk(1)
    r = cn.resolve(sec, [cn.OnCurve("wall", cn.circle([0, 0, 0], 1.0))])
    out, rep = ho.rebuild_high_order(sec, sec.points, r, order=4)
    rad = np.linalg.norm(out.line_mesh.interior[out.edge_tags.flatnonzero("wall")].reshape(-1, 3), axis=1)
    np.testing.assert_allclose(rad, 1.0, atol=1e-13)
    assert out.order == 4 and rep.n_edge_nodes_projected == 24 * 3


def test_bad_arguments_are_refused():
    sec = _disk(3)
    with pytest.raises(ValueError, match="below the mesh's own order"):
        ho.rebuild_high_order(sec, sec.points, order=2)
    with pytest.raises(ValueError, match="new_points must be"):
        ho.rebuild_high_order(sec, sec.points[:-1])


def test_a_session_on_a_curved_mesh_smooths_the_straight_shadow_and_rebuilds_on_request():
    sec = _disk(4)
    jitter = np.random.default_rng(3).normal(size=sec.points.shape) * [0.03, 0.03, 0]
    sec_j = smooth.with_points(ho.linearize(sec), sec.points + jitter)          # a jittered linear start
    s = SmoothSession(sec, [cn.OnCurve("wall", cn.circle([0, 0, 0], 1.0))])
    assert s.original is sec and s.current.order == 1
    assert s.original is not None and sec_j.order == 1
    p = s.propose()
    s.accept(p)
    out, rep = s.rebuild_high_order()
    assert out.order == 4 and rep.n_inverted_sampled == 0
    np.testing.assert_array_equal(out.points, s.current.points)
    assert s.undo(1).order == 1 and s.original is sec                           # the original is never replaced
    lifted, _ = s.rebuild_high_order(order=6)
    assert lifted.order == 6
