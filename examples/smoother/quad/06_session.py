"""The reviewed-round flow of ``SmoothSession``, and the checks that refuse a bad mesh.

Nothing is accepted silently.  A round is *proposed* (run on a copy, session unchanged), its
``RoundReport`` is read, and only then accepted or dropped:

(a) propose -> read the report -> reject a proposal (the session does not change; a stale proposal is
    refused) -> accept -> ``adjust_schedule`` for the next round -> accept -> ``undo``, and the input
    comes back **bit for bit** (compared here, array against array).
(b) the **gap gate**: two patches meeting along a seam whose points were stored twice (never welded)
    are refused before any smoothing, with the ``ConnectivityError`` text; ``allow_gaps=True``
    overrides and the override is written into every round's report.
(c) the fast paths that do not wait for a person: ``review="print"`` (accept every passing round, stop at
    the first failure) and ``review="auto"`` (also stop when a round gains less than ``stall``).
(d) the structural refusals, one line of mesh each: a collapsed edge, a pinch vertex.

    PYTHONPATH=. python examples/smoother/quad/06_session.py

Produces ``06_rounds/round_NN/{before,after}.vtu`` (``out_dir``: ``sj_min``, ``phi_max``,
``constraint_kind``, ``dist_to_target``, ``displacement``), ``06_{input,smoothed}.vtu`` and
``06_frames.json``.  ``mesh`` is the smoothed disk of (a).
"""

import contextlib
import io
import os
import shutil
import time

import numpy as np

from nekmeshpy import linemesh, quadmesh
from nekmeshpy.io.writer import quad_to_vtu
from nekmeshpy.quadmesh.smoother import (
    ConnectivityError,
    SmoothSession,
    adjust_schedule,
    frames,
    smooth,
)
from nekmeshpy.quadmesh.smoother import constrain as cn

# -- parameters --------------------------------------------------------------
RADIUS = 1.0
WALL_N = 24
JITTER = 0.1                 # interior jitter, fraction of the wall spacing
SLIP = 0.25                  # wall slip along the circle, fraction of the wall spacing
SEED = 3
OUT_DIR = "06_rounds"
SCHEDULE = smooth.user_schedule(outer=2)       # deliberately short: leaves room for round 2 to gain
GAP_N = 4                    # cells per side of each patch of the open-seam mesh
AUTO_ROUNDS = 8              # most rounds the 'auto' path may run
AUTO_STALL = 1e-2            # 'auto' stops when a round gains less than this fraction


def disk(seed=SEED):
    base = quadmesh.ogrid(linemesh.circle(RADIUS, WALL_N, element_tag="wall"), 6, 4, wall_tag="wall")
    P = np.array(base.points, dtype=float)
    wall = np.flatnonzero(np.isclose(np.hypot(P[:, 0], P[:, 1]), RADIUS, atol=1e-9))
    inner = np.setdiff1d(np.arange(base.n_points), wall)
    h = 2 * np.pi * RADIUS / WALL_N
    rng = np.random.default_rng(seed)
    P[inner, :2] += rng.uniform(-JITTER, JITTER, (inner.size, 2)) * h
    th = np.arctan2(P[wall, 1], P[wall, 0]) + rng.uniform(-SLIP, SLIP, wall.size) * h
    P[wall, 0], P[wall, 1] = RADIUS * np.cos(th), RADIUS * np.sin(th)
    return smooth.with_points(base, P)


def quiet(fn, *a, **k):
    """Run ``fn`` with its prints captured (the fast paths print every report); return (result, text)."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        res = fn(*a, **k)
    return res, buf.getvalue()


def refusal(mesh, **kw):
    try:
        SmoothSession(mesh, [], **kw)
    except ConnectivityError as err:
        return str(err)
    return None


def per_quad_sj(m, nrm):
    return smooth.corner_metrics(m.points, m.corners, nrm).sj.min(axis=1)


t_start = time.perf_counter()
shutil.rmtree(OUT_DIR, ignore_errors=True)
WALL = [cn.OnCurve("wall", cn.circle([0, 0, 0], RADIUS))]

# ============================================================================
# (a) propose -> read -> reject / accept -> adjust -> undo
# ============================================================================
print("(a) reviewed rounds on a jittered disk")
start = disk()
saved_points = np.array(start.points, copy=True)               # what undo must give back
s = SmoothSession(start, WALL, out_dir=OUT_DIR)
p1 = s.propose(SCHEDULE)
print(p1.report)                                               # the six answers, every metric before / after
print("\nsession after propose: round %d, current is the input mesh: %s   (a proposal changes nothing)"
      % (s.round, s.current is start))

# a rejection is simply not accepting; a proposal made for round 1 cannot be accepted later as if it were new
p_conservative = s.propose(adjust_schedule(p1.schedule, "more_conservative"))
print("a second proposal, more conservative: F/quad %.4f vs %.4f for the first -> keep the first, reject this one"
      % (p_conservative.report.after.objective_mean, p1.report.after.objective_mean))
assert p1.report.passed
s.accept(p1)
try:
    s.accept(p_conservative)
except ValueError as err:
    print("accepting the rejected one now is refused (stale): %s" % err)

p2 = s.propose(adjust_schedule(p1.schedule, "more_aggressive"))
gain = p1.report.after.objective_mean - p2.report.after.objective_mean
print("round 2 (more aggressive): F/quad %.4f -> %.4f, scaled J min %.4f -> %.4f, verdict %s, %d warning(s)"
      % (p2.report.before.objective_mean, p2.report.after.objective_mean, p2.report.before.sj_min,
         p2.report.after.sj_min, "PASS" if p2.report.passed else "FAIL", len(p2.report.warnings)))
if p2.report.passed and gain > 0:
    s.accept(p2)
print("\n" + s.history())
smoothed = s.current
n_accepted = s.round
undone = s.undo(n_accepted)
print("\nundo(%d): back at round %d, the input mesh object is back: %s, points identical bit for bit: %s"
      % (n_accepted, s.round, undone is start, np.array_equal(undone.points, saved_points)))
print("  the smoothed mesh was not touched by the undo: scaled J min %.4f vs input %.4f"
      % (smooth.quality(smoothed.points, smoothed.corners, s.normals).sj_min,
         smooth.quality(start.points, start.corners, s.normals).sj_min))
rounds = sorted(os.listdir(OUT_DIR))
print("per-round files in %s/: %s (%d .vtu)" % (OUT_DIR, ", ".join(rounds),
                                                 sum(f.endswith(".vtu") for r in rounds for f in os.listdir(os.path.join(OUT_DIR, r)))))
print("  (%.1f s)" % (time.perf_counter() - t_start))
mesh = smoothed
quad_to_vtu(start, "06_input.vtu", cell_data={"sj_min": per_quad_sj(start, s.normals)})
quad_to_vtu(smoothed, "06_smoothed.vtu", point_data={"displacement": smoothed.points - start.points},
            cell_data={"sj_min": per_quad_sj(smoothed, s.normals)})
res_a = cn.resolve(start, WALL)
_, logs = smooth.run_schedule(start, res_a, p1.schedule, normals=s.normals, record=True)
frames.write_frames("06_frames.json", start, logs, s.normals, title="06 session: round 1 of the reviewed flow", decimals=5)

# ============================================================================
# (b) the gap gate: an open seam
# ============================================================================
t0 = time.perf_counter()
print("\n(b) gap gate: two %dx%d patches whose shared seam was stored twice" % (GAP_N, GAP_N))


def patch(x0):
    xs, ys = np.linspace(x0, x0 + 1.0, GAP_N + 1), np.linspace(0.0, 1.0, GAP_N + 1)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    i = np.arange((GAP_N + 1) ** 2).reshape(GAP_N + 1, GAP_N + 1)
    q = np.stack([i[:-1, :-1], i[1:, :-1], i[1:, 1:], i[:-1, 1:]], -1).reshape(-1, 4)
    return np.stack([X.ravel(), Y.ravel(), np.zeros(X.size)], 1), q


(Pa, qa), (Pb, qb) = patch(0.0), patch(1.0)
open_seam = quadmesh.QuadMesh.from_corners(np.concatenate([Pa, Pb]), np.concatenate([qa, qb + len(Pa)]))
gap_text = refusal(open_seam)
print("SmoothSession(open_seam, ...) raises ConnectivityError:\n" + "\n".join("  | " + ln for ln in gap_text.splitlines()))
s_gap = SmoothSession(open_seam, [], allow_gaps=True)
rep_gap = s_gap.propose(smooth.user_schedule(outer=1)).report
note = [ln.strip() for ln in rep_gap.constraints.splitlines() if "GAP CHECK" in ln]
print("with allow_gaps=True it runs, and every report says so:\n  | %s" % note[0])
print("  (the seam stayed open: %d stored points for %d distinct positions; %.1f s)"
      % (open_seam.n_points, len(np.unique(np.round(open_seam.points, 9), axis=0)), time.perf_counter() - t0))

# ============================================================================
# (c) the fast paths
# ============================================================================
t0 = time.perf_counter()
print("\n(c) fast paths on a fresh copy of the disk (reports captured; one line per round shown)")
fast_reports = {}
for review, kw in (("print", dict(rounds=3)), ("auto", dict(rounds=AUTO_ROUNDS, stall=AUTO_STALL))):
    fast = SmoothSession(disk(), WALL)
    reports, printed = fast_reports[review] = quiet(fast.run, review=review, schedule=SCHEDULE, **kw)
    print("  review=%r: %d round(s) run, %d report(s) printed, all passed: %s"
          % (review, len(reports), printed.count("=== round"), all(r.passed for r in reports)))
    for r in reports:
        print("    round %d: F/quad %.4f -> %.4f (gain %.1f %%), scaled J min %.4f"
              % (r.round, r.before.objective_mean, r.after.objective_mean,
                 100 * (r.before.objective_mean - r.after.objective_mean) / r.before.objective_mean, r.after.sj_min))
    if review == "auto":
        last = reports[-1]
        gained = (last.before.objective_mean - last.after.objective_mean) / last.before.objective_mean
        regress = [w for w in last.warnings if "WORSE" in w or "fight" in w]
        why = ("raised a regression warning (%s)" % regress[0] if regress else
               "gained %.2g, below stall %g" % (gained, AUTO_STALL) if gained < AUTO_STALL else "was the last allowed")
        print("    stopped after %d of at most %d rounds: the last round %s" % (len(reports), AUTO_ROUNDS, why))
print("  (%.1f s)" % (time.perf_counter() - t0))

# ============================================================================
# (d) refusals the gap check adds: a collapsed edge, a pinch vertex
# ============================================================================
print("\n(d) structural refusals")
collapsed = quadmesh.QuadMesh.from_corners(
    np.array([[0, 0, 0], [0, 0, 0], [1, 1, 0], [0, 1, 0]], float),
    np.array([[0, 1, 2, 3]]))                                  # one quad whose edge 0-1 joins two points at one place
pinch = quadmesh.QuadMesh.from_corners(
    np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [2, 1, 0], [2, 2, 0], [1, 2, 0]], float),
    np.array([[0, 1, 2, 3], [2, 4, 5, 6]]))                    # two quads touching only at point 2
refusals = {}
for label, m in (("collapsed edge", collapsed), ("pinch vertex", pinch)):
    text = refusals[label] = refusal(m)
    notes = [] if text is None else [ln.strip(" -") for ln in text.splitlines() if ln.strip().startswith("- ")]
    print("  %-14s -> %s" % (label, "NOT refused" if text is None else "ConnectivityError: " + "; ".join(notes)))
