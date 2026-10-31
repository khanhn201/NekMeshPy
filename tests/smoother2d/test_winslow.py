"""The Winslow stage: an alternative to Laplace with the same boundary treatment."""

import numpy as np
import pytest
from grids import grid
from test_boundary_stack import _area, _disk

from nekmeshpy import quadmesh
from nekmeshpy.quadmesh.query import boundary_points
from nekmeshpy.quadmesh.smoother import constrain as cn
from nekmeshpy.quadmesh.smoother import smooth


def _polyline(pts, cells):
    out = [np.asarray(pts[0], float)]
    for a, b, n in zip(pts[:-1], pts[1:], cells):
        out += [(1 - t) * np.asarray(a, float) + t * np.asarray(b, float) for t in np.arange(1, n + 1) / n]
    return np.array(out)


def _notch():
    """A Coons patch of a rectangle with a deep notch cut in its bottom: folded where the notch is."""
    bottom = _polyline([(0, 0), (1.35, 0), (1.35, 0.75), (1.65, 0.75), (1.65, 0), (3, 0)], [9, 5, 2, 5, 9])
    top, left, right = (_polyline(p, [n]) for p, n in (([(0, 1), (3, 1)], 30), ([(0, 0), (0, 1)], 10),
                                                       ([(3, 0), (3, 1)], 10)))
    nx, ny = len(bottom) - 1, len(left) - 1
    u = (np.arange(nx + 1) / nx)[:, None, None]
    v = (np.arange(ny + 1) / ny)[None, :, None]
    X = ((1 - v) * bottom[:, None] + v * top[:, None] + (1 - u) * left[None] + u * right[None]
         - ((1 - u) * (1 - v) * bottom[0] + u * (1 - v) * bottom[-1] + (1 - u) * v * top[0] + u * v * top[-1]))
    idx = np.arange((nx + 1) * (ny + 1)).reshape(nx + 1, ny + 1)
    q = np.stack([idx[:-1, :-1], idx[1:, :-1], idx[1:, 1:], idx[:-1, 1:]], -1).reshape(-1, 4)
    P = np.zeros((idx.size, 3))
    P[:, :2] = X.reshape(-1, 2)
    return quadmesh.QuadMesh.from_corners(P, q)


def test_winslow_untangles_a_concave_domain_where_laplace_cannot():
    m = _notch()
    r = cn.resolve(m, [])
    _, lap = smooth.laplace(m, r, passes=100, w=0.95)
    _, win = smooth.winslow(m, r, sweeps=100)
    assert lap.before.n_inverted == win.before.n_inverted > 0
    assert win.after.n_inverted < lap.after.n_inverted


def test_winslow_keeps_a_free_disk_area_with_the_tangent_rule():
    m = _disk()
    r = cn.resolve(m, [], boundary="free")
    tan, log = smooth.winslow(m, r, sweeps=60)
    chain, _ = smooth.winslow(m, r, sweeps=60, free_boundary="chain")
    assert abs(_area(tan) / _area(m) - 1) < 1e-3
    assert _area(chain) / _area(m) - 1 < -0.05
    assert log.after.sj_min > log.before.sj_min
    assert log.after.phi_mean < log.before.phi_mean


def test_winslow_respects_frozen_and_oncurve_vertices_exactly():
    m = _disk()
    out, _ = smooth.winslow(m, cn.resolve(m, []), sweeps=30)             # boundary frozen
    b = boundary_points(m)
    np.testing.assert_array_equal(out.points[b], m.points[b])
    circ = cn.circle([0, 0, 0], 1.0)
    r = cn.resolve(m, [cn.OnCurve(b, circ)], boundary="free")
    out, _ = smooth.winslow(m, r, sweeps=30)
    np.testing.assert_allclose(np.hypot(out.points[b, 0], out.points[b, 1]), 1.0, atol=1e-12)
    assert not np.array_equal(out.points, m.points)


def test_winslow_with_boundary_slide_keeps_a_free_disk_area():
    m = _disk()
    r = cn.resolve(m, [], boundary="slide")
    out, _ = smooth.winslow(m, r, sweeps=60)
    assert abs(_area(out) / _area(m) - 1) < 1e-3


def test_winslow_stagelog_contract():
    m = _disk()
    before = m.points.copy()
    r = cn.resolve(m, [], boundary="free")
    out, log = smooth.winslow(m, r, sweeps=7, tol=0.0)
    np.testing.assert_array_equal(m.points, before)                      # the input is untouched
    assert out.points is not m.points
    assert (log.method, log.passes_requested, log.passes_done, log.stopped) == ("winslow", 7, 7, "completed")
    assert [h[0] for h in log.history] == list(range(1, 8))
    assert log.after.sj_min == log.history[-1][2]
    _, local = smooth.winslow(m, r, sweeps=3, elements=np.zeros(m.n_quads, bool))
    assert (local.method, local.stopped, local.passes_done) == ("winslow_local", "no_elements", 0)
    _, conv = smooth.winslow(m, r, sweeps=3000)
    assert conv.stopped == "completed" and conv.passes_done < 3000 and "converged" in conv.reason


def test_winslow_local_moves_only_the_selected_quads_vertices():
    P, q = grid(8, jitter=0.4, seed=3)
    m = quadmesh.QuadMesh.from_corners(P, q)
    r = cn.resolve(m, [])
    sel = np.zeros(m.n_quads, bool)
    sel[:10] = True
    out, log = smooth.winslow(m, r, sweeps=10, elements=sel)
    touched = np.zeros(m.n_points, bool)
    touched[m.corners[sel].ravel()] = True
    assert log.method == "winslow_local"
    np.testing.assert_array_equal(out.points[~touched], m.points[~touched])
    assert not np.array_equal(out.points[touched], m.points[touched])


def test_winslow_runs_in_a_schedule_and_does_not_change_the_default_schedules():
    P, q = grid(8, jitter=0.45, seed=2)
    m = quadmesh.QuadMesh.from_corners(P, q)
    r = cn.resolve(m, [])
    sched = [("winslow", {"sweeps": 8}), ("laplace", {"passes": 3}), ("winslow_local", {"sweeps": 4})]
    out, logs = smooth.run_schedule(m, r, sched)
    assert [lg.method for lg in logs][:2] == ["winslow", "laplace"]
    assert logs[0].after.sj_min > logs[0].before.sj_min
    assert out.n_points == m.n_points
    used = {s[0] for s in smooth.user_schedule() + smooth.suggest_schedule(logs[0].before)}
    assert not {"winslow", "winslow_local"} & used


def test_winslow_guard_stops_a_closed_surface_drifting_over_many_sweeps():
    surf = cn.sphere([0, 0, 0], 1.0)
    cube = quadmesh.box(1.0, 5)
    P = surf.project(np.asarray(cube.points) / np.linalg.norm(cube.points, axis=1, keepdims=True))
    c = cube.corners
    flip = np.einsum("ij,ij->i", np.cross(P[c[:, 1]] - P[c[:, 0]], P[c[:, 3]] - P[c[:, 0]]),
                     surf.normal(P[c].mean(1))) < 0
    c = c.copy()
    c[flip] = c[flip][:, ::-1]
    ref = quadmesh.QuadMesh.from_corners(P, c)
    ell = cn.vertex_length(ref.points, ref.corners)[:, None]
    m = smooth.with_points(ref, surf.project(ref.points + np.random.default_rng(1).normal(size=P.shape) * 0.25 * ell))
    r = cn.resolve(m, [cn.OnSurface(np.arange(m.n_points), surf)], boundary="free")
    nrm = cn.surface_normals(m.points, m.corners, cn.claimed_surfaces(r))[0]
    out, log = smooth.winslow(m, r, sweeps=1500, normals=nrm, tol=0.0)
    assert np.abs(np.linalg.norm(out.points, axis=1) - 1).max() < 1e-9      # stays on the sphere
    assert log.after.n_inverted == 0
    assert log.stopped == "rolled_back" and log.passes_done < 1500
    assert log.after.sj_min > max(h[2] for h in log.history) - 0.05 - 1e-12


def test_winslow_arguments_are_validated():
    m = _disk()
    r = cn.resolve(m, [], boundary="free")
    for kw in ({"free_boundary": "x"}, {"omega": 0.0}, {"sweeps": 0}, {"elements": np.ones(3, bool)}):
        with pytest.raises(ValueError, match="winslow"):
            smooth.winslow(m, r, **kw)
