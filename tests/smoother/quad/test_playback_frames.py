"""Recorded smoothing frames and the playback JSON."""
import json

import numpy as np
import pytest
from grids import grid

from nekmeshpy import quadmesh
from nekmeshpy.quadmesh.smoother import constrain as cn
from nekmeshpy.quadmesh.smoother import frames, smooth


def _case():
    P, q = grid(8)
    rng = np.random.default_rng(3)
    inner = np.setdiff1d(np.arange(len(P)), quadmesh.query.boundary_points(quadmesh.QuadMesh.from_corners(P, q)))
    P = P.copy()
    P[inner, :2] += rng.uniform(-0.04, 0.04, (inner.size, 2))
    m = quadmesh.QuadMesh.from_corners(P, q)
    return m, cn.resolve(m, [], boundary="frozen")


def test_record_keeps_a_copy_after_every_stage_and_changes_nothing_else():
    m, r = _case()
    sched = [("laplace", {"passes": 2}), ("optimize", {"iters": 5})]
    plain, logs0 = smooth.run_schedule(m, r, sched)
    rec, logs1 = smooth.run_schedule(m, r, sched, record=True)
    assert all(lg.points is None for lg in logs0) and all(lg.points is not None for lg in logs1)
    assert np.array_equal(rec.points, plain.points)                       # recording does not perturb the run
    assert np.array_equal(logs1[-1].points, rec.points)
    assert not np.array_equal(logs1[0].points, logs1[1].points)
    logs1[0].points[:] = 0.0                                              # a copy: later stages never overwrite it
    assert np.array_equal(logs1[-1].points, rec.points)


def test_write_frames_schema_and_numbers(tmp_path):
    m, r = _case()
    out, logs = smooth.run_schedule(m, r, [("laplace", {"passes": 2}), ("optimize", {"iters": 5})], record=True)
    f = str(tmp_path / "run.json")
    doc = frames.write_frames(f, m, logs, title="demo")
    back = json.load(open(f))
    assert back["title"] == "demo" and len(back["frames"]) == 3 == len(doc["frames"])
    assert back["quads"] == np.asarray(m.corners).tolist()
    assert [fr["label"] for fr in back["frames"]] == ["input", "1. laplace", "2. optimize"]
    nrm = smooth.reference_normals(m.points, m.corners)
    for fr, pts in zip(back["frames"], [m.points, logs[0].points, logs[1].points]):
        assert np.allclose(fr["points"], pts, atol=1e-6) and len(fr["sj"]) == m.n_quads
        assert fr["sj_min"] == pytest.approx(smooth.quality(pts, m.corners, nrm).sj_min, abs=1e-4)
    assert back["frames"][-1]["sj_min"] >= back["frames"][0]["sj_min"]    # smoothing helped here


def test_write_frames_needs_a_recorded_run(tmp_path):
    m, r = _case()
    _, logs = smooth.run_schedule(m, r, [("laplace", {"passes": 1})])
    with pytest.raises(ValueError, match="record=True"):
        frames.write_frames(str(tmp_path / "x.json"), m, logs)
