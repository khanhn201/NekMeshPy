"""Smoothing stages, schedules and the method registry, shared by every rung.

A *stage* is one smoothing method run once and described by a :class:`StageLog`.  A *schedule* is an
ordered list of ``(method, knobs)``.  Each rung owns one :class:`Registry` -- the names its
``run_schedule`` knows (``laplace``, ``optimize``, ...) -- and a third party plugs another backend in with
``register`` under a new name.  The three preset schedules are the same at every rung.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, NamedTuple

import numpy as np

from ..._typing import BoolArray, FloatArray, PointArray

__all__ = ["StageLog", "Schedule", "Method", "Registry", "user_schedule", "untangle_schedule", "suggest_schedule"]


class StageLog(NamedTuple):
    """What one smoothing stage did -- the part of the round report this stage owns."""

    method: str
    passes_requested: int
    passes_done: int
    stopped: str               # "completed" | "converged" | "rolled_back" | "no_elements"
    reason: str
    #: The rung's ``Quality`` before and after the stage.
    before: Any
    after: Any
    #: Per pass ``(pass, n_inverted, sj_min, objective_mean)`` of the *accepted* iterates.
    history: tuple[tuple[int, int, float, float], ...]
    #: The vertex positions after this stage, only when ``run_schedule(..., record=True)`` asked for them.
    points: PointArray | None = None


#: A schedule is an ordered list of ``(method, knobs)``; the methods are names in a registry.
Schedule = Sequence[tuple[str, dict[str, Any]]]


class Method(NamedTuple):
    """One registered smoothing method: ``fn(mesh, resolved, **knobs) -> (mesh, StageLog)``.
    ``local``: the schedule runner hands it ``elements=`` (the failing elements) when it runs;
    ``weights``: it accepts ``weights=`` and receives the runner's."""

    fn: Callable[..., tuple[Any, StageLog]]
    local: bool
    weights: bool


class Registry:
    """The stage names one rung's schedule runner knows.  ``entries`` is the live name -> :class:`Method` table."""

    def __init__(self, entries: dict[str, Method] | None = None) -> None:
        self.entries: dict[str, Method] = {} if entries is None else entries

    def register(self, name: str, fn: Callable[..., tuple[Any, StageLog]], *, local: bool = False,
                 weights: bool = False, replace: bool = False) -> None:
        """Make ``fn`` available to :meth:`run` as the stage ``name`` -- the hook another backend plugs in by.
        ``fn(mesh, resolved, **knobs)`` must return a **new** mesh and a :class:`StageLog`, leave ``mesh``
        untouched and honour ``resolved``.  An existing name is refused unless ``replace=True``."""
        if not name or not isinstance(name, str):
            raise ValueError("register_method: name must be a non-empty string, got %r" % (name,))
        if name in self.entries and not replace:
            raise ValueError("register_method: %r is already registered (pass replace=True)" % name)
        if not callable(fn):
            raise TypeError("register_method: fn must be callable, got %r" % (fn,))
        self.entries[name] = Method(fn, local, weights)

    def names(self) -> list[str]:
        """The stage names, sorted."""
        return sorted(self.entries)

    def run(self, mesh: Any, resolved: Any, schedule: Schedule, *,
            failing: Callable[[Any], BoolArray], weights: FloatArray | None = None,
            common: dict[str, Any] | None = None, record: bool = False) -> tuple[Any, list[StageLog]]:
        """Run the stages in order, each on the previous stage's result.  ``failing(mesh)`` picks the elements a
        *local* stage works on, at the moment it runs; ``common`` knobs (e.g. the reference normals) go to every
        stage unless the schedule overrides them.  Returns the final mesh and one :class:`StageLog` per stage;
        ``mesh`` is untouched.  ``record=True`` keeps a copy of the vertex positions after every stage in
        ``StageLog.points``."""
        unknown = [m for m, _ in schedule if m not in self.entries]
        if unknown:
            raise ValueError("unknown smoothing method(s) %s; registered: %s" % (unknown, self.names()))
        logs: list[StageLog] = []
        cur = mesh
        for name, knobs in schedule:
            m = self.entries[name]
            kw = dict(knobs)
            if m.local:
                kw["elements"] = failing(cur)
            if m.weights and weights is not None:
                kw.setdefault("weights", weights)
            for k, v in (common or {}).items():
                kw.setdefault(k, v)
            cur, log = m.fn(cur, resolved, **kw)
            logs.append(log._replace(points=np.array(cur.points, copy=True)) if record else log)
        return cur, logs


def user_schedule(outer: int = 10, n_lap: int = 3, n_opt: int = 20, *, w: float = 0.97,
                  final_local: bool = True) -> list[tuple[str, dict[str, Any]]]:
    """The tuned shape -- *loops of loops*: ``outer`` times ``laplace(n_lap)`` then ``optimize(n_opt)``,
    closed by two local Laplace passes that lift the smallest edges.  Only a starting suggestion; the
    caller proposes and the user edits it."""
    sched: list[tuple[str, dict[str, Any]]] = []
    for _ in range(outer):
        sched.append(("laplace", {"passes": n_lap, "w": w}))
        sched.append(("optimize", {"iters": n_opt}))
    if final_local:
        sched += [("laplace_local", {"passes": 5, "w": w})] * 2
    return sched


def untangle_schedule(deltas: Sequence[float] = (1.0, 0.3, 0.1, 0.03, 0.01),
                      iters: int = 60) -> list[tuple[str, dict[str, Any]]]:
    """Optimizer stages with a shrinking barrier width: a wide ``delta`` first smooths the
    landscape enough for a folded vertex to be pulled through, then each narrower one
    sharpens it back toward the exact metric."""
    return [("optimize", {"iters": iters, "delta": d}) for d in deltas]


def suggest_schedule(start: Any, *, outer: int = 10) -> list[tuple[str, dict[str, Any]]]:
    """The schedule to *propose* for a mesh in the state ``start`` (a rung's ``Quality``): the tuned shape,
    preceded by the untangling continuation when any element starts inverted -- a Laplace pass on a
    tangled mesh has nothing to repair it with.  A suggestion, never silently run."""
    base = user_schedule(outer=outer)
    return (untangle_schedule() + base) if start.n_inverted > 0 else base
