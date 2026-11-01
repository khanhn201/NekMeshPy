"""The interactive layer of the smoother, shared by every rung: one :class:`SmoothSession` holds the mesh, its
constraints and a stack of *accepted* states, and every round answers the same questions
(:class:`RoundReport`) before anything is accepted.

The session never edits in place and never accepts silently: :meth:`SmoothSession.propose` runs a schedule on a
copy and returns a :class:`Proposal` (report + candidate mesh + before/after files); the caller -- a person at
the prompt, or an assistant reading the report to a person -- decides; :meth:`~SmoothSession.accept` pushes it,
:meth:`~SmoothSession.undo` pops it.  Meshes are immutable by contract, so a state is a reference and undo costs
nothing.  ``review="print"`` / ``"auto"`` in :meth:`~SmoothSession.run` are the fast paths that do not wait for
an answer.

Everything that depends on the mesh type sits behind a :class:`RungAdapter` -- how a mesh is linearised and its
curved nodes rebuilt (a curved mesh stays curved through a round: see the rung's ``run_schedule``), how constraints resolve, which input checks run, how quality is read, how a round's files
are written, and which stages exist (the rung's own registry, so a stage registered there -- including a third
party's -- runs through the session with no change here).  The session itself is only the state stack, the round
counter and the verdict.  ``core/`` imports no rung, so the adapter is the only door to one.
"""

from __future__ import annotations

import os
from typing import Any, NamedTuple

from .metric import EPS
from .schedule import Schedule, StageLog, suggest_schedule

__all__ = ["Count", "Audit", "RoundFields", "RoundReport", "Proposal", "RungAdapter", "SmoothSession", "adjust_schedule",
           "build_report", "verdict"]


class Count(NamedTuple):
    """An integer a rung counts per mesh that must not rise in a round (quads in an overlap, ...).  ``note`` is
    the ``%d`` warning printed while ``after`` is above zero."""

    label: str
    before: int
    after: int
    note: str


class Audit(NamedTuple):
    """A rung's own part of a verdict: extra ``warnings`` / ``failures``, structured ``data`` the rung's report
    type reads back, and text lines appended to section 3 (quality) and section 4 (geometry) of the printout."""

    warnings: tuple[str, ...] = ()
    failures: tuple[str, ...] = ()
    data: dict[str, Any] = {}
    detail3: tuple[str, ...] = ()
    detail4: tuple[str, ...] = ()


#: ``(metric template, field, better)``: ``+1`` higher is better, ``-1`` lower, ``0`` neutral.
#: ``%s`` is the report's ``plural`` / ``angle_word``.
_METRICS: tuple[tuple[str, str, int], ...] = (
    ("inverted %(p)s", "n_inverted", -1), ("min scaled J", "sj_min", +1),
    ("worst 1%% scaled J", "sj_q01", +1), ("worst 5%% scaled J", "sj_q05", +1),
    ("worst 10%% scaled J", "sj_q10", +1), ("median scaled J", "sj_median", +1),
    ("mean scaled J", "sj_mean", +1), ("mean phi", "phi_mean", -1), ("max phi", "phi_max", -1),
    ("objective / %(n)s", "objective_mean", -1),
    ("max aspect ratio", "aspect_max", -1), ("95th pct aspect", "aspect_p95", -1),
    ("min %(a)s angle (deg)", "angle_min_deg", +1), ("max %(a)s angle (deg)", "angle_max_deg", -1),
    ("neighbour size ratio max", "size_ratio_max", -1), ("neighbour size ratio 95th", "size_ratio_p95", -1),
    ("neighbour size ratio mean", "size_ratio_mean", -1),
    ("node spacing min", "spacing_min", 0), ("node spacing max", "spacing_max", 0),
)


def _val(q: Any, sh: Any, key: str) -> float:
    return float(getattr(q, key) if hasattr(q, key) else getattr(sh, key))


class RoundFields(NamedTuple):
    """The fields of a :class:`RoundReport`, in order."""

    round: int
    constraints: str                   # 1. what were the constraints
    stages: tuple[StageLog, ...]       # 2. what ran
    before: Any                        # 3. quality, every metric before and after (a rung's ``Quality``)
    after: Any
    shape_before: Any                  #    and its ``ShapeStats``
    shape_after: Any
    geometry_before: Any               # 4. still on the target?  (an ``OnTargetReport``)
    geometry_after: Any
    propagate: str                     # 5. propagate upward?  (empty: the question does not arise)
    undo: str                          # 6. undo state (empty: no session around it)
    warnings: tuple[str, ...]
    failures: tuple[str, ...]
    files: dict[str, str]
    counts: tuple[Count, ...] = ()
    data: dict[str, Any] = {}
    detail3: tuple[str, ...] = ()
    detail4: tuple[str, ...] = ()


class RoundReport(RoundFields):
    """What a round owes, plus the verdict.  ``failures`` block acceptance; ``warnings`` do not.  Printing it gives
    the full, honest before/after table.  A rung subclasses this (no new fields) to set the words
    (``noun`` / ``plural`` / ``angle_word``) and to read its :class:`Audit` ``data`` back as attributes."""

    __slots__ = ()

    noun = "element"
    plural = "elements"
    angle_word = "corner"
    propagate_label = "propagate"

    @property
    def passed(self) -> bool:
        return not self.failures

    def metric_rows(self) -> list[tuple[str, float, float, str]]:
        words = {"n": self.noun, "p": self.plural, "a": self.angle_word}
        rows = []
        for template, key, better in _METRICS:
            name = template % words
            b, a = _val(self.before, self.shape_before, key), _val(self.after, self.shape_after, key)
            tol = 1e-9 * max(abs(b), abs(a), 1.0)
            mark = "same" if abs(a - b) <= tol else ("changed" if better == 0 else
                                                     ("better" if (a - b) * better > 0 else "WORSE"))
            rows.append((name, b, a, mark))
            if key == "n_inverted":            # next to it: a count sees what the Jacobian cannot
                for c in self.counts:
                    rows.append((c.label, float(c.before), float(c.after),
                                 "same" if c.before == c.after else ("better" if c.after < c.before else "WORSE")))
        return rows

    def __str__(self) -> str:
        out = ["=== round %d: %s ===" % (self.round, "PASS" if self.passed else "FAIL")
               if self.round else "=== round: %s ===" % ("PASS" if self.passed else "FAIL"),
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
        out += ["   " + ln for ln in self.detail3]
        out.append("4. geometry  before: %s" % (self.geometry_before,))
        out.append("             after:  %s" % (self.geometry_after,))
        out += ["   " + ln for ln in self.detail4]
        if self.propagate:
            out.append("5. %s: %s" % (self.propagate_label, self.propagate))
        if self.undo:
            out.append("6. undo: " + self.undo)
        if self.warnings:
            out += ["warnings:"] + ["   ! " + w for w in self.warnings]
        if self.failures:
            out += ["FAILURES (round cannot be accepted):"] + ["   X " + f for f in self.failures]
        if self.files:
            out.append("files: " + ", ".join("%s=%s" % kv for kv in self.files.items()))
        return "\n".join(out)


class Proposal(NamedTuple):
    """A round that has been run but not accepted: the candidate mesh, its report, the schedule."""

    mesh: Any
    report: RoundReport
    schedule: list[tuple[str, dict[str, Any]]]


class RungAdapter:
    """Everything the session needs from one mesh type.  Subclass it per rung; the defaults are the "nothing to do"
    answers, so a rung overrides only what it has.  ``report_type`` is the :class:`RoundReport` subclass it fills."""

    report_type: type[RoundReport] = RoundReport
    #: Section 5 of the report ("propagate to the blocks above?"); empty when the question does not arise.
    propagate = ""
    #: The explanation printed in place of a rebuilt high-order mesh when the rung has none.
    no_rebuild = "this rung cannot rebuild a high-order mesh yet"

    # -- input ---------------------------------------------------------------------------
    def preflight(self, mesh: Any) -> None:
        """Refuse or warn about the *input* mesh (a gap gate, a curved mesh the rung cannot take).  Raises."""

    def linearize(self, mesh: Any) -> Any:
        """The straight-sided order-1 shadow of a curved ``mesh`` (what the linear stages of a schedule run on)."""
        raise NotImplementedError("linearize")

    def resolve(self, mesh: Any, constraints: list[Any], boundary: str) -> Any:
        """The per-vertex mobility (``Resolved``) of ``constraints`` on ``mesh``."""
        raise NotImplementedError("resolve")

    def prepare(self, mesh: Any, resolved: Any) -> None:
        """Called once with the linear mesh and its first resolution (reference normals, input checks)."""

    def rebuild(self, base: Any, points: Any, resolved: Any, order: int | None) -> tuple[Any, Any]:
        """The curved mesh of the smoothed vertices ``points`` carried on ``base`` (see
        :meth:`SmoothSession.rebuild_high_order`).  Returns ``(mesh, report)``."""
        raise NotImplementedError(self.no_rebuild)

    # -- readings ------------------------------------------------------------------------
    def quality(self, mesh: Any) -> Any:
        raise NotImplementedError("quality")

    def shape_stats(self, mesh: Any) -> Any:
        raise NotImplementedError("shape_stats")

    def on_target(self, mesh: Any, resolved: Any, tol: float) -> Any:
        raise NotImplementedError("on_target")

    def counts(self, mesh: Any) -> tuple[tuple[str, int, str], ...]:
        """``(label, value, note)`` per integer the verdict watches (it may not rise); empty by default."""
        return ()

    def audit(self, before: Any, after: Any, resolved: Any) -> Audit:
        """The rung's own warnings / failures and report sections for the round ``before -> after``."""
        return Audit()

    def describe(self, resolved: Any, *, sj_target: float, geom_tol: float) -> str:
        """Section 1 of the report: what the constraints were."""
        raise NotImplementedError("describe")

    # -- running -------------------------------------------------------------------------
    def run(self, mesh: Any, resolved: Any, schedule: Schedule, *, sj_target: float,
            weights: Any) -> tuple[Any, list[StageLog]]:
        """Run ``schedule`` on ``mesh`` through the rung's registry (``mesh`` is untouched)."""
        raise NotImplementedError("run")

    def write(self, directory: str, before: Any, after: Any, resolved: Any) -> dict[str, str]:
        """Write the round's files into ``directory`` (exists); returns ``{"before": path, "after": path}``."""
        return {}


def verdict(report_type: type[RoundReport], logs: list[StageLog], before: Any, after: Any,
            shape_before: Any, shape_after: Any, geometry: Any, counts: tuple[Count, ...] = (),
            audit: Audit | None = None) -> tuple[list[str], list[str]]:
    """``(warnings, failures)`` of a round -- the one place it is decided.  Fails: off target (``geometry`` is the
    *after* reading), more inverted elements, a watched count that rose, anything the rung's ``audit`` fails.
    Warns: what remains, any metric that got worse, a stage rolled back, Laplace and the optimizer fighting."""
    audit = Audit() if audit is None else audit
    rep = report_type(0, "", (), before, after, shape_before, shape_after, geometry, geometry, "", "", (), (), {},
                      counts)
    warns: list[str] = []
    fails: list[str] = []
    if not geometry.passed:
        fails.append("geometry: %s" % (geometry,))
    inv = "inverted %s" % rep.plural
    if after.n_inverted > before.n_inverted:
        fails.append("%s rose %d -> %d" % (inv, before.n_inverted, after.n_inverted))
    if after.n_inverted > 0:
        warns.append("%d inverted %s(s) remain" % (after.n_inverted, rep.noun))
    for c in counts:
        if c.after > max(c.before, 0):
            fails.append("%s rose %d -> %d" % (c.label, c.before, c.after))
        if c.after > 0:
            warns.append(c.note % c.after)
    fails += audit.failures
    warns += audit.warnings
    skip = {inv, *(c.label for c in counts)}
    for name, b, a, mark in rep.metric_rows():
        if mark == "WORSE" and name not in skip:
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


def build_report(adapter: RungAdapter, before: Any, after: Any, resolved: Any, logs: list[StageLog], *,
                 rnd: int = 0, undo: str = "", propagate: str = "", files: dict[str, str] | None = None,
                 sj_target: float = EPS, geom_tol: float = 1e-6) -> RoundReport:
    """The :class:`RoundReport` of smoothing ``before`` into ``after`` under ``resolved``, every reading taken
    through ``adapter`` and the verdict from :func:`verdict`."""
    qb, qa = adapter.quality(before), adapter.quality(after)
    cb, ca = adapter.counts(before), adapter.counts(after)
    counts = tuple(Count(lb, vb, va, note) for (lb, vb, note), (_, va, _n) in zip(cb, ca))
    gb, ga = adapter.on_target(before, resolved, geom_tol), adapter.on_target(after, resolved, geom_tol)
    aud = adapter.audit(before, after, resolved)
    shb, sha = adapter.shape_stats(before), adapter.shape_stats(after)
    warns, fails = verdict(adapter.report_type, logs, qb, qa, shb, sha, ga, counts, aud)
    return adapter.report_type(rnd, adapter.describe(resolved, sj_target=sj_target, geom_tol=geom_tol), tuple(logs),
                               qb, qa, shb, sha, gb, ga, propagate, undo, tuple(warns), tuple(fails),
                               dict(files or {}), counts, dict(aud.data), aud.detail3, aud.detail4)


def adjust_schedule(schedule: Schedule, mode: str) -> list[tuple[str, dict[str, Any]]]:
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


class _State(NamedTuple):
    mesh: Any
    constraints: list[Any]
    resolved: Any
    report: RoundReport | None
    schedule: list[tuple[str, dict[str, Any]]] | None


class SmoothSession:
    """One smoothing job on one mesh, through a :class:`RungAdapter`.  A rung's session subclasses this with its
    own constructor (the options its adapter takes) and binds the adapter; see the module docstring for the flow.
    A curved mesh stays curved (``keep_curved``); pass ``keep_curved=False`` to smooth its straight-sided shadow."""

    def __init__(self, mesh: Any, constraints: list[Any], adapter: RungAdapter, *, boundary: str = "frozen",
                 sj_target: float = EPS, geom_tol: float = 1e-6, weights: Any = None,
                 out_dir: str | None = None, name: str = "smooth", keep_curved: bool = True) -> None:
        self.adapter = adapter
        self.constraints = constraints
        self.boundary = boundary
        self.sj_target = sj_target
        self.geom_tol = geom_tol
        self.weights = weights
        self.out_dir = out_dir
        self.name = name
        # The high-order input, kept when a curved mesh is given.  By default the session smooths the curved mesh
        # itself: the rung's schedule runs ``tmop`` on it and carries its nodes through the linear stages.
        # ``keep_curved=False`` smooths its straight-sided shadow instead (``rebuild_high_order`` lifts it back).
        self.original: Any = None
        adapter.preflight(mesh)
        if mesh.order > 1:
            self.original = mesh
            if not keep_curved:
                mesh = adapter.linearize(mesh)
        res = adapter.resolve(mesh, constraints, boundary)
        adapter.prepare(mesh, res)
        self._states: list[_State] = [_State(mesh, list(constraints), res, None, None)]

    # -- state -----------------------------------------------------------------------------
    @property
    def current(self) -> Any:
        return self._states[-1].mesh

    @property
    def round(self) -> int:
        return len(self._states) - 1

    @property
    def resolved(self) -> Any:
        return self._states[-1].resolved

    def rebuild_high_order(self, order: int | None = None) -> tuple[Any, Any]:
        """The curved mesh of the *current accepted* vertices: the original high-order nodes carried along and
        re-projected onto the targets.  With a linear input and ``order=N`` the mesh is lifted to order ``N``.
        The session state is not changed.  Raises ``NotImplementedError`` at a rung without a rebuild."""
        base = self.original if self.original is not None else self.current
        return self.adapter.rebuild(base, self.current.points, self.resolved, order)

    def quality(self) -> Any:
        return self.adapter.quality(self.current)

    def suggest(self) -> list[tuple[str, dict[str, Any]]]:
        """The schedule to *propose* for the current state (never run silently)."""
        return suggest_schedule(self.quality())

    def set_constraints(self, constraints: list[Any], *, boundary: str | None = None) -> None:
        """Change the constraints -- a state of its own, so :meth:`undo` restores the old ones."""
        b = self.boundary if boundary is None else boundary
        res = self.adapter.resolve(self.current, constraints, b)
        self.boundary = b
        self._states.append(_State(self.current, list(constraints), res, None, None))

    # -- a round ---------------------------------------------------------------------------
    def propose(self, schedule: Schedule | None = None) -> Proposal:
        """Run ``schedule`` (default: :meth:`suggest`) on a copy; the session is unchanged."""
        sched = [(m, dict(k)) for m, k in (self.suggest() if schedule is None else schedule)]
        cur, res = self.current, self.resolved
        cand, logs = self.adapter.run(cur, res, sched, sj_target=self.sj_target, weights=self.weights)
        rnd = self.round + 1
        files = self._write(rnd, cur, cand, res)
        prev = self._states[-1].report
        noun = self.adapter.report_type.noun
        undo = ("accepting makes this round %d; undo(1) returns to round %d (%s)"
                % (rnd, self.round, "input mesh" if prev is None else
                   "min scaled J %.4f, objective/%s %.4f" % (prev.after.sj_min, noun, prev.after.objective_mean)))
        rep = build_report(self.adapter, cur, cand, res, logs, rnd=rnd, undo=undo,
                           propagate=self.adapter.propagate, files=files, sj_target=self.sj_target,
                           geom_tol=self.geom_tol)
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

    def reject(self, proposal: Proposal) -> None:
        """Drop a proposal.  Nothing was pushed, so there is nothing to undo; the method exists so a reviewer's
        answer has a name (and is checked for staleness like :meth:`accept`)."""
        if proposal.report.round != self.round + 1:
            raise ValueError("stale proposal: it was made for round %d, the session is at round %d"
                             % (proposal.report.round - 1, self.round))

    def undo(self, n: int = 1) -> Any:
        """Pop ``n`` accepted states (constraint changes count as states); the input is the floor."""
        if n < 1:
            raise ValueError("undo: n must be >= 1")
        if n > self.round:
            raise ValueError("undo: only %d state(s) can be undone" % self.round)
        del self._states[-n:]
        return self.current

    def history(self) -> str:
        """The trend over accepted states: inverted elements, min scaled J, objective per element."""
        rows = ["round  inverted  min scaled J  worst 5%%  objective/" + self.adapter.report_type.noun + "   stages"]
        for i, st in enumerate(self._states):
            if st.report is None:
                q = self.adapter.quality(st.mesh)
                what = "input" if i == 0 else "constraints changed"
            else:
                q, what = st.report.after, " ".join(m for m, _ in (st.schedule or []))[:40] or "-"
            rows.append("%5d  %8d  %12.4f  %8.4f  %14.4f   %s" % (i, q.n_inverted, q.sj_min, q.sj_q05,
                                                                  q.objective_mean, what))
        return "\n".join(rows)

    def run(self, rounds: int, *, review: str = "print", adjust: str = "same",
            schedule: Schedule | None = None, stall: float = 1e-3,
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
            if review == "auto" and (prop.report.after.objective_mean < ideal or rel < stall or any(
                    "WORSE" in w or "fight" in w for w in prop.report.warnings)):
                break
        return reports

    # -- internals -------------------------------------------------------------------------
    def _write(self, rnd: int, cur: Any, cand: Any, res: Any) -> dict[str, str]:
        if self.out_dir is None:
            return {}
        d = os.path.join(self.out_dir, "round_%02d" % rnd)
        os.makedirs(d, exist_ok=True)
        return self.adapter.write(d, cur, cand, res)

