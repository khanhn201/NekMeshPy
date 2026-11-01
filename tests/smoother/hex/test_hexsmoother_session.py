"""The hex session: propose / accept / undo, the verdict, per-round files, the registry."""

import meshio
import numpy as np
import pytest
from hexblocks import block

from nekmeshpy import hexmesh, linemesh, quadmesh
from nekmeshpy.core.smoother.schedule import StageLog
from nekmeshpy.hexmesh.smoother import SmoothSession, linearize, smooth
from nekmeshpy.hexmesh.smoother import constrain as cn

SCHED = [("laplace", {"passes": 2}), ("optimize", {"iters": 5})]


def _session(**kw):
    m = block(3, jitter=0.3, seed=2, tags=["a"] * 13 + ["b"] * 14)
    return SmoothSession(m, [], boundary="frozen", sample_order=2, **kw)


def test_propose_runs_on_a_copy_and_accept_undo_are_bit_for_bit():
    s = _session()
    start = s.current
    p0 = start.points.copy()
    prop = s.propose(SCHED)
    assert s.round == 0 and prop.report.passed and prop.report.round == 1
    np.testing.assert_array_equal(s.current.points, p0)               # the session did not move
    assert prop.report.after.objective_mean < prop.report.before.objective_mean
    s.accept(prop)
    assert s.round == 1 and s.current is prop.mesh
    cand = s.current.points.copy()
    prop2 = s.propose(SCHED)
    s.accept(prop2)
    assert s.round == 2
    assert s.undo() is prop.mesh and s.round == 1
    np.testing.assert_array_equal(s.current.points, cand)
    assert s.undo() is start and s.round == 0
    np.testing.assert_array_equal(s.current.points, p0)
    with pytest.raises(ValueError, match="only 0"):
        s.undo()
    assert "input" in s.history()


def test_a_failed_round_cannot_be_accepted_and_a_stale_proposal_is_refused():
    s = _session()
    p = s.propose(SCHED)
    bad = p._replace(report=p.report._replace(failures=("geometry: broken",)))
    with pytest.raises(ValueError, match="failed round"):
        s.accept(bad)
    assert s.round == 0
    q = s.propose(SCHED)
    s.accept(p)
    with pytest.raises(ValueError, match="stale"):
        s.accept(q)
    with pytest.raises(ValueError, match="stale"):
        s.reject(q)
    s.reject(s.propose(SCHED))                                         # a current one is simply dropped
    assert s.round == 1


def test_run_print_and_auto(capsys):
    s = _session()
    reps = s.run(2, review="print", schedule=SCHED)
    out = capsys.readouterr().out
    assert len(reps) == 2 and s.round == 2 and out.count("=== round") == 2 and "per region" in out
    s2 = _session()
    reps = s2.run(5, review="auto", schedule=SCHED, stall=0.5)         # stops once a round gains < 50%
    assert 1 <= len(reps) < 5 and s2.round == len(reps)
    with pytest.raises(ValueError, match="review"):
        s2.run(1, review="ask")


def test_report_per_region_and_files_per_round(tmp_path):
    s = _session(out_dir=str(tmp_path))
    prop = s.propose(SCHED)
    rep = prop.report
    assert [g.region for g in rep.regions_after] == ["a", "b"] and rep.sample_order == 2
    assert rep.regions_after[0].n_hexes == 13 and rep.n_frozen > 0 and rep.n_frozen_moved == 0
    assert "1. constraints" in str(rep) and "6. undo" in str(rep) and "5." not in str(rep)
    after = meshio.read(rep.files["after"])
    assert after.points.shape[0] == 8 * 27 and {"sj_min", "displacement"} <= set(
        list(after.cell_data) + list(after.point_data))
    assert (tmp_path / "round_01" / "before.vtu").exists()
    assert hexmesh.smoother.report.round_report(s.current, prop.mesh, s.resolved, sample_order=2).passed


def test_a_high_order_input_is_refused_with_the_way_out():
    cyl = hexmesh.extrude(quadmesh.ogrid(linemesh.circle(1.0, 8, order=2), 2, 1), 1.0, 1)
    with pytest.raises(ValueError, match="linearize"):
        SmoothSession(cyl, [])
    s = SmoothSession(linearize(cyl), [], sample_order=2)
    with pytest.raises(NotImplementedError, match="cannot rebuild"):
        s.rebuild_high_order()


def test_constraints_are_a_state_and_a_registered_stage_runs_through_the_session():
    s = _session()
    s.set_constraints([], boundary="slide")
    assert s.round == 1 and s.boundary == "slide"
    s.undo()

    def nudge(mesh, resolved, *, scale=0.0):                           # a stand-in for a third-party backend
        P = np.array(mesh.points, copy=True)
        P += scale * (resolved.kind == cn.FREE)[:, None]
        q = smooth.quality(mesh.points, mesh.corners)
        out = smooth.with_points(mesh, P)
        return out, StageLog("nudge", 1, 1, "completed", "dummy", q, smooth.quality(P, mesh.corners), ())

    smooth.register_method("nudge", nudge)
    try:
        prop = s.propose([("nudge", {"scale": 0.0})])
        assert [lg.method for lg in prop.report.stages] == ["nudge"] and prop.report.passed
        np.testing.assert_array_equal(prop.mesh.points, s.current.points)
    finally:
        smooth.REGISTRY.entries.pop("nudge")
    with pytest.raises(ValueError, match="unknown smoothing method"):
        s.propose([("nudge", {})])
