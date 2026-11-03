"""Sliding along STRAIGHT boundary edges (plan T1d): every side of a square is an OnCurve(line) that slides,
the four corners lie on two lines each and are therefore frozen by the two-curve rule."""

import numpy as np
from grids import grid

from nekmeshpy import quadmesh
from nekmeshpy.quadmesh.query import boundary_points
from nekmeshpy.quadmesh.smoother import constrain as cn
from nekmeshpy.quadmesh.smoother import smooth


def _square_with_uneven_sides(n=10, seed=4):
    P, q = grid(n, jitter=0.35, seed=seed)
    b = np.isclose(P[:, 1], 0) | np.isclose(P[:, 1], 1) | np.isclose(P[:, 0], 0) | np.isclose(P[:, 0], 1)
    on_x = b & (np.isclose(P[:, 1], 0) | np.isclose(P[:, 1], 1)) & ~(np.isclose(P[:, 0], 0) | np.isclose(P[:, 0], 1))
    on_y = b & (np.isclose(P[:, 0], 0) | np.isclose(P[:, 0], 1)) & ~(np.isclose(P[:, 1], 0) | np.isclose(P[:, 1], 1))
    rng = np.random.default_rng(seed)
    P[on_x, 0] = np.clip(P[on_x, 0] + rng.uniform(-0.04, 0.04, on_x.sum()), 0.02, 0.98)     # uneven spacing along each side
    P[on_y, 1] = np.clip(P[on_y, 1] + rng.uniform(-0.04, 0.04, on_y.sum()), 0.02, 0.98)
    return quadmesh.QuadMesh.from_corners(P, q)


def _side_constraints(m):
    P = m.points
    sides = [cn.OnCurve(np.flatnonzero(np.isclose(P[:, 1], 0)), cn.line([0, 0, 0], [1, 0, 0])),
             cn.OnCurve(np.flatnonzero(np.isclose(P[:, 1], 1)), cn.line([0, 1, 0], [1, 0, 0])),
             cn.OnCurve(np.flatnonzero(np.isclose(P[:, 0], 0)), cn.line([0, 0, 0], [0, 1, 0])),
             cn.OnCurve(np.flatnonzero(np.isclose(P[:, 0], 1)), cn.line([1, 0, 0], [0, 1, 0]))]
    return sides


def _gaps(M, side):
    P = M.points
    ids = np.flatnonzero(np.isclose(P[:, 1], 0)) if side == "bottom" else np.flatnonzero(np.isclose(P[:, 0], 0))
    t = np.sort(P[ids, 0] if side == "bottom" else P[ids, 1])
    d = np.diff(t)
    return d.std() / d.mean()


def test_corners_are_frozen_and_sides_slide_along_their_lines():
    m = _square_with_uneven_sides()
    r = cn.resolve(m, _side_constraints(m))
    P = m.points
    corner = np.isclose(P[:, 0], 0) & np.isclose(P[:, 1], 0) | np.isclose(P[:, 0], 1) & np.isclose(P[:, 1], 0) \
        | np.isclose(P[:, 0], 0) & np.isclose(P[:, 1], 1) | np.isclose(P[:, 0], 1) & np.isclose(P[:, 1], 1)
    assert corner.sum() == 4 and (r.kind[corner] == cn.FROZEN).all()            # two curves -> frozen
    side = np.setdiff1d(boundary_points(m), np.flatnonzero(corner))
    assert (r.kind[side] == cn.ON_CURVE).all()
    out, logs = smooth.run_schedule(m, r, smooth.user_schedule(outer=4))
    np.testing.assert_array_equal(out.points[corner], m.points[corner])         # corners did not move
    rep = cn.on_target_report(out.points, out.corners, r)
    assert rep.passed and rep.max_distance < 1e-13                              # still exactly on their lines
    moved = np.abs(out.points[side] - m.points[side]).max()
    assert moved > 1e-3                                                         # they really slid
    assert _gaps(out, "bottom") < _gaps(m, "bottom") and _gaps(out, "left") < _gaps(m, "left")   # spacing evened
    assert logs[-1].after.n_inverted == 0


def test_sliding_beats_a_frozen_boundary_on_quality():
    m = _square_with_uneven_sides()
    nrm = smooth.reference_normals(m.points, m.corners)
    slide = smooth.run_schedule(m, cn.resolve(m, _side_constraints(m)), smooth.user_schedule(outer=4), normals=nrm)[1][-1].after
    frozen = smooth.run_schedule(m, cn.resolve(m, []), smooth.user_schedule(outer=4), normals=nrm)[1][-1].after
    assert slide.objective_mean < frozen.objective_mean
    assert slide.sj_min >= frozen.sj_min - 1e-9
