"""Overlapping quads: the reading the inverted-quad count cannot give."""
import logging

import numpy as np
from grids import grid

from nekmeshpy import quadmesh
from nekmeshpy.core.smoother.session import Count, verdict
from nekmeshpy.quadmesh.smoother import constrain as cn
from nekmeshpy.quadmesh.smoother import smooth
from nekmeshpy.quadmesh.smoother.session import RoundReport, SmoothSession


def _overlap(before, after):
    return (Count("overlapping quads", before, after, "%d overlapping quad(s) remain (positive Jacobians do not rule this out)"),)


def _nested():
    """Quad B sits inside quad A, both wound the same way: every Jacobian is positive."""
    P = np.array([[0, 0, 0], [2, 0, 0], [2, 2, 0], [0, 2, 0], [1, 0.2, 0], [1, 1, 0], [0.2, 1, 0]], float)
    q = np.array([[0, 1, 2, 3], [0, 4, 5, 6]])
    return quadmesh.QuadMesh.from_corners(P, q)


def test_nested_quads_are_found_although_no_jacobian_is_negative():
    m = _nested()
    nrm = np.tile([0.0, 0.0, 1.0], (2, 1))
    assert smooth.quality(m.points, m.corners, nrm).n_inverted == 0       # the old count is blind to it
    assert list(quadmesh.overlapping_quads(m)) == [0, 1]
    assert not quadmesh.is_overlap_free(m)


def test_clean_meshes_have_none():
    P, q = grid(6)
    assert quadmesh.overlapping_quads(quadmesh.QuadMesh.from_corners(P, q)).size == 0


def test_a_folded_vertex_is_found():
    P, q = grid(4)
    P = P.copy()
    P[12] = [0.6, 0.1, 0.0]                                                # an interior vertex dragged into a far quad
    assert quadmesh.overlapping_quads(quadmesh.QuadMesh.from_corners(P, q)).size > 0


def test_the_round_report_carries_the_overlap_row_and_can_be_switched_off():
    P, q = grid(5)
    m = quadmesh.QuadMesh.from_corners(P, q)
    s = SmoothSession(m, [], boundary="frozen")
    assert s.input_overlaps == 0
    rep = s.propose([("laplace", {"passes": 1})]).report
    assert (rep.overlaps_before, rep.overlaps_after) == (0, 0)
    assert any(r[0] == "overlapping quads" for r in rep.metric_rows())
    off = SmoothSession(m, [], boundary="frozen", check_overlaps=False)
    rep2 = off.propose([("laplace", {"passes": 1})]).report
    assert off.input_overlaps == -1 and rep2.overlaps_before == -1
    assert all(r[0] != "overlapping quads" for r in rep2.metric_rows())


def test_overlapping_input_warns_and_a_rise_fails_the_round(caplog):
    s = SmoothSession(_nested(), [], boundary="frozen", allow_gaps=True)      # the pinch is overridden on purpose
    assert s.input_overlaps == 2
    P, q = grid(4)
    m = quadmesh.QuadMesh.from_corners(P, q)
    with caplog.at_level(logging.WARNING):
        SmoothSession(_nested(), [], allow_gaps=True)
    assert "overlap another quad" in caplog.text
    ok = SmoothSession(m, [], boundary="frozen")
    nrm = ok.normals
    qa = smooth.quality(m.points, m.corners, nrm)
    sh = smooth.shape_stats(m.points, m.corners)
    geom = cn.on_target_report(m.points, m.corners, ok.resolved, tol=1e-6)
    _, fails = verdict(RoundReport, [], qa, qa, sh, sh, geom, _overlap(0, 3))
    assert any("overlapping quads rose 0 -> 3" in f for f in fails)
    warns, fails = verdict(RoundReport, [], qa, qa, sh, sh, geom, _overlap(3, 3))
    assert not fails and any("3 overlapping quad(s) remain" in w for w in warns)


def test_it_is_cheap():
    import time
    P, q = grid(300)                                                       # 90,000 quads
    m = quadmesh.QuadMesh.from_corners(P, q)
    t = time.perf_counter()
    assert quadmesh.overlapping_quads(m).size == 0
    assert time.perf_counter() - t < 10.0


def test_a_coarse_closed_pipe_wall_does_not_overlap_itself():
    """Opposite wall quads of a tight tube project into each other's footprint without touching: the in-plane test
    alone read every quad of a real swept pipe wall as overlapping.  The out-of-plane cutoff must clear it."""
    n_around, n_along = 8, 6
    th = np.linspace(0, 2 * np.pi, n_around, endpoint=False)
    P = np.array([[np.cos(t), np.sin(t), z] for z in np.linspace(0, 3, n_along + 1) for t in th])
    q = []
    for k in range(n_along):
        for j in range(n_around):
            a, b = k * n_around + j, k * n_around + (j + 1) % n_around
            q.append([a, b, b + n_around, a + n_around])
    m = quadmesh.QuadMesh.from_corners(P, np.array(q))
    assert quadmesh.overlapping_quads(m).size == 0
