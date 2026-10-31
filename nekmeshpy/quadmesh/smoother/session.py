"""The interactive layer of the quad smoother: one :class:`SmoothSession` holds the mesh, its
constraints and a stack of *accepted* states, and every round answers the same six questions
(:class:`RoundReport`) before anything is accepted.

The session never edits in place and never accepts silently: :meth:`SmoothSession.propose` runs a
schedule on a copy and returns a :class:`Proposal` (report + candidate mesh + before/after
``.vtu``); the caller -- a person at the prompt, or an assistant reading the report to a person --
decides; :meth:`~SmoothSession.accept` pushes it, :meth:`~SmoothSession.undo` pops it.  Meshes are
immutable by contract, so a state is a reference and undo costs nothing.  ``review="print"`` /
``"auto"`` in :meth:`~SmoothSession.run` are the fast paths that do not wait for an answer.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, NamedTuple

import numpy as np

from ..._typing import PointArray
from ...io import writer
from ..quadmesh import QuadMesh
from ..query import ConnectivityReport, check_connectivity, overlapping_quads
from . import constrain as cn
from . import smooth
from .highorder import HighOrderReport, linearize, rebuild_high_order

__all__ = ["ConnectivityError", "RoundReport", "Proposal", "SmoothSession", "adjust_schedule"]


class ConnectivityError(RuntimeError):
    """The gap check failed and ``allow_gaps`` was not set."""


#: ``(metric, getter source, better)``: ``+1`` higher is better, ``-1`` lower, ``0`` neutral.
_METRICS: tuple[tuple[str, str, int], ...] = (
    ("inverted quads", "n_inverted", -1), ("min scaled J", "sj_min", +1),
    ("worst 1% scaled J", "sj_q01", +1), ("worst 5% scaled J", "sj_q05", +1),
    ("worst 10% scaled J", "sj_q10", +1), ("median scaled J", "sj_median", +1),
    ("mean scaled J", "sj_mean", +1), ("mean phi", "phi_mean", -1), ("max phi", "phi_max", -1),
    ("objective / quad", "objective_mean", -1),
    ("max aspect ratio", "aspect_max", -1), ("95th pct aspect", "aspect_p95", -1),
    ("min corner angle (deg)", "angle_min_deg", +1), ("max corner angle (deg)", "angle_max_deg", -1),
    ("neighbour size ratio max", "size_ratio_max", -1), ("neighbour size ratio 95th", "size_ratio_p95", -1),
    ("neighbour size ratio mean", "size_ratio_mean", -1),
    ("node spacing min", "spacing_min", 0), ("node spacing max", "spacing_max", 0),
)


def _val(q: smooth.Quality, sh: smooth.ShapeStats, key: str) -> float:
    return float(getattr(q, key) if hasattr(q, key) else getattr(sh, key))


class RoundReport(NamedTuple):
    """The six answers a round owes, plus the verdict.  ``failures`` block acceptance;
    ``warnings`` do not.  Printing it gives the full, honest before/after table."""

    round: int
    constraints: str                   # 1. what were the constraints
    stages: tuple[smooth.StageLog, ...]  # 2. what ran
    before: smooth.Quality             # 3. quality, every metric before and after
    after: smooth.Quality
    shape_before: smooth.ShapeStats
    shape_after: smooth.ShapeStats
    geometry_before: cn.OnTargetReport  # 4. still on the target?
    geometry_after: cn.OnTargetReport
    propagate: str                     # 5. propagate to the hex blocks?
    undo: str                          # 6. undo state
    warnings: tuple[str, ...]
    failures: tuple[str, ...]
    files: dict[str, str]
    #: Quads taking part in an overlap before / after (``-1``: not checked, ``check_overlaps=False``).
    overlaps_before: int = -1
    overlaps_after: int = -1

    @property
    def passed(self) -> bool:
        return not self.failures

    def metric_rows(self) -> list[tuple[str, float, float, str]]:
        rows = []
        for name, key, better in _METRICS:
            b, a = _val(self.before, self.shape_before, key), _val(self.after, self.shape_after, key)
            tol = 1e-9 * max(abs(b), abs(a), 1.0)
            mark = "same" if abs(a - b) <= tol else ("changed" if better == 0 else
                                                     ("better" if (a - b) * better > 0 else "WORSE"))
            rows.append((name, b, a, mark))
            if key == "n_inverted" and self.overlaps_before >= 0:     # next to it: overlap sees what it cannot
                ob, oa = float(self.overlaps_before), float(self.overlaps_after)
                rows.append(("overlapping quads", ob, oa, "same" if ob == oa else ("better" if oa < ob else "WORSE")))
        return rows

    def __str__(self) -> str:
        out = ["=== round %d: %s ===" % (self.round, "PASS" if self.passed else "FAIL"),
               "1. constraints", *("   " + ln for ln in self.constraints.splitlines()),
               "2. what ran"]
        shown = list(self.stages)
        skipped = 0
        if len(shown) > 10:                       # keep the head and tail, count the middle
            mid = shown[4:-2]
            skipped, shown = len(mid), shown[:4] + shown[-2:]
            now = sum(lg.passes_done == 0 or lg.stopped == "converged" for lg in mid)
        for i, lg in enumerate(shown):
            if skipped and i == 4:
                out.append("   ... %d further stages (%d of them converged or did nothing)" % (skipped, now))
            out.append("   %-14s %3d/%-3d %-11s %s" % (lg.method, lg.passes_done, lg.passes_requested,
                                                       lg.stopped, lg.reason))
        out.append("3. quality, before -> after (all metrics; negatives first)")
        out.append("   %-24s %14s %14s   %s" % ("metric", "before", "after", ""))
        for name, b, a, mark in self.metric_rows():
            out.append("   %-24s %14.6g %14.6g   %s" % (name, b, a, mark))
        out.append("4. geometry  before: %s" % (self.geometry_before,))
        out.append("             after:  %s" % (self.geometry_after,))
        out += ["5. propagate to hex: " + self.propagate, "6. undo: " + self.undo]
        if self.warnings:
            out += ["warnings:"] + ["   ! " + w for w in self.warnings]
        if self.failures:
            out += ["FAILURES (round cannot be accepted):"] + ["   X " + f for f in self.failures]
        if self.files:
            out.append("files: " + ", ".join("%s=%s" % kv for kv in self.files.items()))
        return "\n".join(out)


@dataclass(frozen=True)
class Proposal:
    """A round that has been run but not accepted: the candidate mesh, its report, the schedule."""

    mesh: QuadMesh
    report: RoundReport
    schedule: list[tuple[str, dict[str, Any]]]


def adjust_schedule(schedule: smooth.Schedule, mode: str) -> list[tuple[str, dict[str, Any]]]:
    """The "more aggressive / more conservative / same" step, as one table of knobs and never as
    a loosened constraint: Laplace shrink ``w`` (0.99 / keep / 0.92), Laplace passes (2 / keep / 6),
    optimizer iterations (10 / keep / 40) and optimizer step cap ``alpha_max`` (0.25 / keep / keep)."""
    if mode not in ("more_aggressive", "more_conservative", "same"):
        raise ValueError("mode must be more_aggressive, more_conservative or same, got %r" % (mode,))
    if mode == "same":
        return [(m, dict(k)) for m, k in schedule]
    aggr = mode == "more_aggressive"
    out = []
    for method, knobs in schedule:
        k = dict(knobs)
        if method.startswith("laplace"):
            k["w"], k["passes"] = (0.92, 6) if aggr else (0.99, 2)
        elif method.startswith("optimize") and "delta" not in k:
            k["iters"] = 40 if aggr else 10
            if not aggr:
                k["alpha_max"] = 0.25
        elif method.startswith("optimize"):        # untangling continuation: only the length changes
            k["iters"] = 100 if aggr else 30
        out.append((method, k))
    return out


@dataclass
class _State:
    mesh: QuadMesh
    constraints: list[cn.Constraint]
    resolved: cn.Resolved
    report: RoundReport | None
    schedule: list[tuple[str, dict[str, Any]]] | None


@dataclass
class SmoothSession:
    """One smoothing job on one linear quad mesh.  See the module docstring for the flow."""

    mesh: QuadMesh
    constraints: list[cn.Constraint]
    boundary: str = "frozen"
    normals: PointArray | None = None
    #: The surface the quads sit on, for the metric's normals (see ``__post_init__``); optional.
    surface: cn.Surface | None = None
    #: Compare the mesh's own winding with the surface normals and warn on disagreement (cheap; opt out here).
    check_normals: bool = True
    #: Count overlapping quads each round (a branched star can have every Jacobian positive); opt out here.
    check_overlaps: bool = True
    normal_cos_tol: float = 0.5
    sj_target: float = smooth.EPS
    geom_tol: float = 1e-6
    connectivity_tol: float = 0.2
    allow_gaps: bool = False
    weights: PointArray | None = None
    out_dir: str | None = None
    name: str = "smooth"
    connectivity: ConnectivityReport = field(init=False)
    #: Set when the normals were read from a surface (``None`` for explicit or reference normals).
    normal_report: cn.NormalReport | None = field(init=False, default=None)
    #: Quads of the *input* that overlap another quad (``-1`` when ``check_overlaps=False``).
    input_overlaps: int = field(init=False, default=-1)
    #: The high-order input, kept when a curved mesh is given: smoothing runs on its straight-sided shadow.
    original: QuadMesh | None = field(init=False, default=None)
    _states: list[_State] = field(init=False, default_factory=list)

    def __post_init__(self) -> None:
        self.connectivity = check_connectivity(self.mesh, connectivity_tol=self.connectivity_tol)
        if not self.connectivity.passed and not self.allow_gaps:
            raise ConnectivityError("the gap check failed; patch the connectivity first, or pass "
                                    "allow_gaps=True to smooth anyway (it will be logged):\n%s"
                                    % (self.connectivity,))
        if self.mesh.order > 1:
            self.original = self.mesh
            self.mesh = linearize(self.mesh)
        res = cn.resolve(self.mesh, self.constraints, boundary=self.boundary)
        if self.normals is None:
            # precedence: explicit normals > ``surface`` (every vertex) > the OnSurface constraints > the
            # quads' own winding.  The winding is the weakest source: one folded input quad poisons it.
            groups = ([(np.arange(self.mesh.n_points), self.surface)] if self.surface is not None
                      else cn.claimed_surfaces(res))
            if groups:
                self.normals, self.normal_report = cn.surface_normals(
                    self.mesh.points, self.mesh.corners, groups, check=self.check_normals,
                    cos_tol=self.normal_cos_tol)
                if self.normal_report.n_disagree:
                    logging.warning(
                        "%d of %d quads disagree with the surface normal by more than cos %.2f (min cos %.2f): "
                        "a folded or reversed input quad, or the wrong surface; the surface normals are used",
                        self.normal_report.n_disagree, self.normal_report.n_quads, self.normal_cos_tol,
                        self.normal_report.min_cos)
            else:
                self.normals = smooth.reference_normals(self.mesh.points, self.mesh.corners)
        if self.check_overlaps:
            self.input_overlaps = int(overlapping_quads(self.mesh).size)
            if self.input_overlaps:
                logging.warning("%d input quad(s) overlap another quad: smoothing can leave a persistent local "
                                "overlap (a branched star) around them, invisible to the inverted-quad count; "
                                "check_overlaps reports it each round", self.input_overlaps)
        self._states = [_State(self.mesh, list(self.constraints), res, None, None)]

    # -- state -----------------------------------------------------------------------------
    @property
    def current(self) -> QuadMesh:
        return self._states[-1].mesh

    @property
    def round(self) -> int:
        return len(self._states) - 1

    @property
    def resolved(self) -> cn.Resolved:
        return self._states[-1].resolved

    def rebuild_high_order(self, order: int | None = None) -> tuple[QuadMesh, HighOrderReport]:
        """The curved mesh of the *current accepted* vertices: the original high-order nodes carried along and
        re-projected onto the targets (see :mod:`nekmeshpy.quadmesh.smoother.highorder`).  With a linear input and
        ``order=N`` the mesh is lifted to order ``N``.  The session state is not changed."""
        base = self.original if self.original is not None else self.current
        return rebuild_high_order(base, self.current.points, self.resolved, order=order)

    def quality(self) -> smooth.Quality:
        assert self.normals is not None
        return smooth.quality(self.current.points, self.current.corners, self.normals)

    def suggest(self) -> list[tuple[str, dict[str, Any]]]:
        """The schedule to *propose* for the current state (never run silently)."""
        return smooth.suggest_schedule(self.quality())

    def set_constraints(self, constraints: list[cn.Constraint], *, boundary: str | None = None) -> None:
        """Change the constraints -- a state of its own, so :meth:`undo` restores the old ones."""
        b = self.boundary if boundary is None else boundary
        res = cn.resolve(self.current, constraints, boundary=b)
        self.boundary = b
        self._states.append(_State(self.current, list(constraints), res, None, None))

    # -- a round ---------------------------------------------------------------------------
    def propose(self, schedule: smooth.Schedule | None = None) -> Proposal:
        """Run ``schedule`` (default: :meth:`suggest`) on a copy; the session is unchanged."""
        sched = [(m, dict(k)) for m, k in (self.suggest() if schedule is None else schedule)]
        assert self.normals is not None
        cur, res = self.current, self.resolved
        cand, logs = smooth.run_schedule(cur, res, sched, normals=self.normals, sj_target=self.sj_target,
                                         weights=self.weights)
        before = smooth.quality(cur.points, cur.corners, self.normals)
        after = smooth.quality(cand.points, cand.corners, self.normals)
        shb, sha = smooth.shape_stats(cur.points, cur.corners), smooth.shape_stats(cand.points, cand.corners)
        gb = cn.on_target_report(cur.points, cur.corners, res, tol=self.geom_tol)
        ga = cn.on_target_report(cand.points, cand.corners, res, tol=self.geom_tol)
        rnd = self.round + 1
        ob = int(overlapping_quads(cur).size) if self.check_overlaps else -1
        oa = int(overlapping_quads(cand).size) if self.check_overlaps else -1
        warns, fails = self._verdict(logs, before, after, shb, sha, ga, ob, oa)
        files = self._write(rnd, cur, cand, res)
        prev = self._states[-1].report
        undo = ("accepting makes this round %d; undo(1) returns to round %d (%s)"
                % (rnd, self.round, "input mesh" if prev is None else
                   "min scaled J %.4f, objective/quad %.4f" % (prev.after.sj_min, prev.after.objective_mean)))
        rep = RoundReport(rnd, self._constraint_text(res), tuple(logs), before, after, shb, sha, gb, ga,
                          "not applicable: quad-only (no hex provenance recorded yet)", undo,
                          tuple(warns), tuple(fails), files, ob, oa)
        return Proposal(cand, rep, sched)

    def accept(self, proposal: Proposal) -> None:
        """Push an accepted proposal (refused if it failed or is stale)."""
        if not proposal.report.passed:
            raise ValueError("a failed round cannot be accepted:\n" + "\n".join(proposal.report.failures))
        if proposal.report.round != self.round + 1:
            raise ValueError("stale proposal: it was made for round %d, the session is at round %d"
                             % (proposal.report.round - 1, self.round))
        st = self._states[-1]
        self._states.append(_State(proposal.mesh, st.constraints, st.resolved, proposal.report,
                                   proposal.schedule))

    def undo(self, n: int = 1) -> QuadMesh:
        """Pop ``n`` accepted states (constraint changes count as states); the input is the floor."""
        if n < 1:
            raise ValueError("undo: n must be >= 1")
        if n > self.round:
            raise ValueError("undo: only %d state(s) can be undone" % self.round)
        del self._states[-n:]
        return self.current

    def history(self) -> str:
        """The trend over accepted states: inverted quads, min scaled J, objective per quad."""
        rows = ["round  inverted  min scaled J  worst 5%%  objective/quad   stages"]
        for i, st in enumerate(self._states):
            if st.report is None:
                q = smooth.quality(st.mesh.points, st.mesh.corners, self.normals)   # type: ignore[arg-type]
                what = "input" if i == 0 else "constraints changed"
            else:
                q, what = st.report.after, " ".join(m for m, _ in (st.schedule or []))[:40] or "-"
            rows.append("%5d  %8d  %12.4f  %8.4f  %14.4f   %s" % (i, q.n_inverted, q.sj_min, q.sj_q05,
                                                                  q.objective_mean, what))
        return "\n".join(rows)

    def run(self, rounds: int, *, review: str = "print", adjust: str = "same",
            schedule: smooth.Schedule | None = None, stall: float = 1e-3,
            ideal: float = 1e-6) -> list[RoundReport]:
        """The fast paths, which do not wait for an answer.  ``"print"`` runs ``rounds`` rounds on
        the same trajectory (or ``adjust``-ed each time), accepting every passing round, printing
        each report, and stopping at the first FAIL.  ``"auto"`` additionally stops when the mean
        objective is already below ``ideal`` (a square lattice is 0; nothing left to gain), when a
        round improves it by less than ``stall`` (relative), or when it raises a regression warning --
        the interactive path is :meth:`propose` + a human + :meth:`accept`."""
        if review not in ("print", "auto"):
            raise ValueError("review must be 'print' or 'auto' here (interactive = propose/accept)")
        reports: list[RoundReport] = []
        sched = None if schedule is None else [(m, dict(k)) for m, k in schedule]
        for _ in range(rounds):
            prop = self.propose(sched)
            print(prop.report)
            reports.append(prop.report)
            if not prop.report.passed:
                break
            self.accept(prop)
            sched = adjust_schedule(prop.schedule, adjust)
            rel = (prop.report.before.objective_mean - prop.report.after.objective_mean) / max(
                abs(prop.report.before.objective_mean), 1e-300)
            if review == "auto" and (prop.report.after.objective_mean < ideal or rel < stall or any("WORSE" in w or "fight" in w for w in prop.report.warnings)):
                break
        return reports

    # -- internals -------------------------------------------------------------------------
    def _constraint_text(self, res: cn.Resolved) -> str:
        c = res.counts()
        lines = ["vertices: %s" % ", ".join("%d %s" % (v, k.replace("_", "-")) for k, v in c.items()),
                 "targets: %d surface(s), %d curve(s); geometry mode: strict (snap + target-normal gradient every step)"
                 % (len(res.surfaces), len(res.curves)),
                 "limits: scaled-J target %.3g (penalty eps), on-target tol %.0e x local edge, connectivity tol %.2f"
                 % (self.sj_target, self.geom_tol, self.connectivity_tol)]
        lines += ["note: " + n for n in res.notes]
        if not self.connectivity.passed:
            lines.append("note: GAP CHECK FAILED and overridden (allow_gaps): %d gap cluster(s)"
                         % self.connectivity.n_gap_clusters)
        return "\n".join(lines)

    def _verdict(self, logs: list[smooth.StageLog], before: smooth.Quality, after: smooth.Quality,
                 shb: smooth.ShapeStats, sha: smooth.ShapeStats,
                 geom: cn.OnTargetReport, ob: int = -1, oa: int = -1) -> tuple[list[str], list[str]]:
        warns: list[str] = []
        fails: list[str] = []
        if not geom.passed:
            fails.append("geometry: %s" % (geom,))
        if after.n_inverted > before.n_inverted:
            fails.append("inverted quads rose %d -> %d" % (before.n_inverted, after.n_inverted))
        if after.n_inverted > 0:
            warns.append("%d inverted quad(s) remain" % after.n_inverted)
        if oa > max(ob, 0):
            fails.append("overlapping quads rose %d -> %d" % (ob, oa))
        if oa > 0:
            warns.append("%d overlapping quad(s) remain (positive Jacobians do not rule this out)" % oa)
        for name, b, a, mark in RoundReport(0, "", (), before, after, shb, sha, geom, geom, "", "", (),
                                            (), {}).metric_rows():
            if mark == "WORSE" and name != "inverted quads":
                warns.append("%s got WORSE: %.6g -> %.6g" % (name, b, a))
        rolled: dict[str, int] = {}
        for lg in logs:
            if lg.stopped == "rolled_back":
                key = "%s stage rolled back: %s" % (lg.method, lg.reason)
                rolled[key] = rolled.get(key, 0) + 1
        for key, n in rolled.items():                       # identical refusals are one warning
            warns.append(key if n == 1 else "%s  (x%d)" % (key, n))
        lap = [lg for lg in logs if lg.method == "laplace"]
        raised = sum(lg.after.objective_mean > lg.before.objective_mean * (1 + 1e-6) for lg in lap)
        if len(lap) >= 3 and raised >= max(3, len(lap) // 2):
            warns.append("Laplace raised the objective in %d of %d passes while the optimizer lowered it "
                         "again: the two methods are fighting; try fewer Laplace passes or a larger w"
                         % (raised, len(lap)))
        return warns, fails

    def _write(self, rnd: int, cur: QuadMesh, cand: QuadMesh, res: cn.Resolved) -> dict[str, str]:
        if self.out_dir is None:
            return {}
        d = os.path.join(self.out_dir, "round_%02d" % rnd)
        os.makedirs(d, exist_ok=True)
        files = {}
        assert self.normals is not None
        for tag, M in (("before", cur), ("after", cand)):
            m = smooth.corner_metrics(M.points, M.corners, self.normals)
            dist = np.zeros(M.n_points)
            for k, objs in cn._targets(res):
                for t, o in enumerate(objs):
                    sel = (res.kind == k) & (res.target == t)
                    if sel.any():
                        dist[sel] = np.linalg.norm(o.project(M.points[sel]) - M.points[sel], axis=1)
            path = os.path.join(d, tag + ".vtu")
            writer.quad_to_vtu(M, path,
                               point_data={"constraint_kind": res.kind.astype(float), "dist_to_target": dist,
                                           "displacement": M.points - cur.points},
                               cell_data={"sj_min": m.sj.min(1), "phi_max": m.phi.max(1),
                                          **({"overlap": np.isin(np.arange(M.n_quads),
                                                                 overlapping_quads(M)).astype(int)}
                                             if self.check_overlaps else {})})
            files[tag] = path
        return files
