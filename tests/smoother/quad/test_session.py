"""M5: the session -- gap gate, rounds, report, undo, schedule adjustment, fast paths."""

import numpy as np
import pytest
from grids import grid

from nekmeshpy import linemesh, quadmesh
from nekmeshpy.quadmesh.smoother import constrain as cn
from nekmeshpy.quadmesh.smoother import smooth
from nekmeshpy.quadmesh.smoother.session import (
    ConnectivityError,
    Proposal,
    SmoothSession,
    adjust_schedule,
)


def _mesh(n=10, jitter=0.45, seed=5):
    return quadmesh.QuadMesh.from_corners(*grid(n, jitter=jitter, seed=seed))


def _gapped():
    A, qa = grid(4, jitter=0.2, seed=1)
    B, qb = grid(4, jitter=0.2, seed=2)
    B = B + [1.0 + 1e-3 / 4, 0, 0]
    return quadmesh.QuadMesh.from_corners(np.vstack([A, B]), np.vstack([qa, qb + len(A)]))


def test_the_gap_gate_refuses_an_unwelded_seam_and_the_override_is_logged():
    m = _gapped()
    with pytest.raises(ConnectivityError, match="gap check failed"):
        SmoothSession(m, [])
    s = SmoothSession(m, [], allow_gaps=True)
    assert not s.connectivity.passed
    assert "GAP CHECK FAILED and overridden" in str(s.propose(smooth.user_schedule(outer=1)).report)


def test_propose_runs_on_a_copy_and_the_report_answers_all_six_questions(tmp_path):
    s = SmoothSession(_mesh(), [], out_dir=str(tmp_path))
    before = s.current.points.copy()
    prop = s.propose()
    assert isinstance(prop, Proposal) and s.round == 0
    np.testing.assert_array_equal(s.current.points, before)
    text = str(prop.report)
    for head in ("1. constraints", "2. what ran", "3. quality", "4. geometry", "5. propagate", "6. undo"):
        assert head in text
    names = [r[0] for r in prop.report.metric_rows()]
    for need in ("inverted quads", "min scaled J", "worst 1% scaled J", "worst 5% scaled J",
                 "worst 10% scaled J", "median scaled J", "mean scaled J", "max aspect ratio",
                 "95th pct aspect", "node spacing min"):
        assert need in names
    assert prop.report.passed and prop.report.before.n_inverted > 0 == prop.report.after.n_inverted
    import meshio
    after = meshio.read(prop.report.files["after"])
    assert {"constraint_kind", "dist_to_target", "displacement"} <= set(after.point_data)
    assert {"sj_min", "phi_max"} <= set(after.cell_data)


def test_accept_pushes_and_undo_restores_the_input_bit_for_bit():
    m = _mesh()
    s = SmoothSession(m, [])
    p1 = s.propose()
    s.accept(p1)
    assert s.round == 1 and s.current is p1.mesh
    p2 = s.propose(smooth.user_schedule(outer=2))
    s.accept(p2)
    assert s.round == 2
    back = s.undo(2)
    assert back is m and s.round == 0
    np.testing.assert_array_equal(back.points, m.points)
    with pytest.raises(ValueError, match="only 0"):
        s.undo(1)
    assert "input" in s.history() and s.history().count("\n") == 1


def test_a_failed_or_stale_proposal_cannot_be_accepted():
    s = SmoothSession(_mesh(), [])
    p = s.propose()
    bad = Proposal(p.mesh, p.report._replace(failures=("geometry: broken",)), p.schedule)
    with pytest.raises(ValueError, match="failed round"):
        s.accept(bad)
    s.accept(p)
    with pytest.raises(ValueError, match="stale"):
        s.accept(p)


def test_changing_constraints_is_a_state_and_undo_restores_the_old_ones():
    m = _mesh(8, jitter=0.2)
    s = SmoothSession(m, [])
    old_kinds = s.resolved.kind.copy()
    s.set_constraints([], boundary="free")
    assert s.round == 1 and (s.resolved.kind == cn.FREE).all() and (old_kinds == cn.FROZEN).any()
    s.undo(1)
    np.testing.assert_array_equal(s.resolved.kind, old_kinds)


def test_geometry_is_reported_and_enforced_on_a_curved_target():
    ring = linemesh.circle(1.0, 24, element_tag="wall")
    sec = quadmesh.ogrid(ring, 6, 4, wall_tag="wall")
    P = sec.points + np.random.default_rng(3).normal(size=sec.points.shape) * [0.03, 0.03, 0]
    sec = smooth.with_points(sec, P)
    s = SmoothSession(sec, [cn.OnCurve("wall", cn.circle([0, 0, 0], 1.0))])
    p = s.propose()
    assert p.report.geometry_before.max_distance > 1e-3          # displaced off the circle ...
    assert p.report.geometry_after.passed and p.report.geometry_after.max_distance < 1e-13
    assert p.report.after.n_inverted == 0 and "1 on-curve" not in str(p.report) and "24 on-curve" in str(p.report)


def test_adjust_schedule_is_one_table_never_a_loosened_constraint():
    base = smooth.user_schedule(outer=2)
    cons = adjust_schedule(base, "more_conservative")
    aggr = adjust_schedule(base, "more_aggressive")
    same = adjust_schedule(base, "same")
    assert same == base and same is not base and same[0][1] is not base[0][1]
    assert cons[0][1] == {"passes": 2, "w": 0.99} and aggr[0][1] == {"passes": 6, "w": 0.92}
    assert cons[1][1]["iters"] == 10 and aggr[1][1]["iters"] == 40 and cons[1][1]["alpha_max"] == 0.25
    unt = adjust_schedule(smooth.untangle_schedule(), "more_aggressive")
    assert all(k["iters"] == 100 and "delta" in k for _, k in unt)
    assert base[0][1] == {"passes": 3, "w": 0.97}                  # input untouched
    for _, k in aggr + cons:
        assert "kappa" not in k and "eps" not in k                # constraints/limits never change
    with pytest.raises(ValueError, match="mode must be"):
        adjust_schedule(base, "yolo")


def test_fast_paths_print_runs_n_rounds_and_auto_stops_when_it_stalls(capsys):
    s = SmoothSession(_mesh(), [])
    reps = s.run(3, review="print", schedule=smooth.user_schedule(outer=2, n_opt=15))
    assert len(reps) == 3 and s.round == 3 and capsys.readouterr().out.count("=== round") == 3
    s2 = SmoothSession(_mesh(), [])
    reps2 = s2.run(6, review="auto", schedule=smooth.user_schedule(outer=3, n_opt=20))
    assert 1 <= len(reps2) < 6                                    # converged, so auto stopped early
    with pytest.raises(ValueError, match="interactive"):
        s.run(1, review="interactive")


def test_zigzag_between_laplace_and_optimizer_is_flagged():
    s = SmoothSession(_mesh(6, jitter=0.1), [])
    P, q = grid(6, jitter=0.1, seed=5)
    nrm = smooth.reference_normals(P, q)
    lo = smooth.quality(P, q, nrm)
    hi = lo._replace(objective_mean=lo.objective_mean * 2)
    logs = [smooth.StageLog("laplace", 3, 3, "completed", "", lo, hi, ()) for _ in range(4)]
    warns, fails = s._verdict(logs, lo, lo, smooth.shape_stats(P, q), smooth.shape_stats(P, q),
                              cn.on_target_report(P, q, s.resolved))
    assert any("fighting" in w for w in warns) and not fails


def test_a_long_session_on_a_disk_converges_round_over_round(tmp_path):
    """Three accepted rounds: the first does the work, the later ones barely move anything."""
    ring = linemesh.circle(1.0, 24, element_tag="wall")
    sec = quadmesh.ogrid(ring, 6, 4, wall_tag="wall")
    sec = smooth.with_points(sec, sec.points + np.random.default_rng(3).normal(size=sec.points.shape) * [0.03, 0.03, 0])
    s = SmoothSession(sec, [cn.OnCurve("wall", cn.circle([0, 0, 0], 1.0))], out_dir=str(tmp_path))
    reps = s.run(3, review="print", schedule=smooth.user_schedule(outer=4, n_opt=20))
    f = [reps[0].before.objective_mean] + [r.after.objective_mean for r in reps]
    assert f[1] < 0.1 * f[0] and abs(f[3] - f[2]) < 0.05 * abs(f[1] - f[0])
    assert s.undo(3) is sec
