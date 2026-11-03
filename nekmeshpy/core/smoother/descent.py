"""One nonlinear conjugate-gradient driver with a line search, for every rung's shape optimizer.

The optimizer is the same loop at the quad and the hex rung: Fletcher-Reeves on the objective, the constraints
enforced **every step**.  What differs -- how the objective and its gradient are evaluated on the elements in
play, how a gradient or a step is cut down to what the targets allow, how a trial is snapped back onto the
targets -- is passed in as callables, so this module imports no mesh type.

Everything is measured in one unit, the local edge length ``l_v`` (shortest incident edge, frozen at stage
start): the variables are ``u_v = dx_v / l_v``, the search direction is normalised so its largest move is one
``l_v``, and the line-search variable ``alpha`` is that move in edge lengths, capped at ``alpha_max``.  Nothing
depends on the element count or the model's units.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import NamedTuple

import numpy as np

from ..._typing import BoolArray, FloatArray, PointArray

__all__ = ["DescentResult", "minimize"]


class DescentResult(NamedTuple):
    """What :func:`minimize` ended with: the last accepted positions ``P`` (not yet snapped a final time), the
    objective ``f`` and inverted count ``ninv`` there, one ``(iteration, n_inverted, sj_min, objective_mean)``
    per accepted iterate, why it stopped and how many iterations were accepted."""

    P: PointArray
    f: float
    ninv: int
    history: list[tuple[int, int, float, float]]
    stopped: str
    reason: str
    done: int


def minimize(P: PointArray, *, evaluate: Callable[[PointArray], tuple[float, int]],
             gradient: Callable[[PointArray], PointArray],
             project: Callable[[PointArray, PointArray], PointArray],
             snap: Callable[[PointArray], PointArray],
             summary: Callable[[PointArray], tuple[int, float, float]],
             ell: FloatArray, movable: BoolArray, tangent: BoolArray, n_used: int,
             iters: int, alpha_max: float = 1.0, restart: str = "halve",
             tol_step: float = 1e-4, tol_rel: float = 1e-6) -> DescentResult:
    """Conjugate-gradient descent from ``P`` (already snapped onto the targets).

    ``evaluate(Q)`` is ``(F, n_inverted)`` over the elements in play; ``gradient(P)`` the analytic ``dF/dx``
    ``(P, 3)`` over the same; ``project(d, P)`` removes from a gradient or step ``d`` whatever the targets forbid
    at positions ``P``; ``snap(Q)`` puts a trial back on the targets; ``summary(P)`` is the
    ``(n_inverted, sj_min, objective_mean)`` of the whole mesh logged per accepted iterate.  ``ell`` ``(P, 1)`` is
    the local length unit, ``movable`` the vertices that may move, ``tangent`` those whose direction is re-cut
    after the CG mix, ``n_used`` the element count for the mean-excess test.

    Each step: take the gradient, project it, form the CG direction, and line-search ``alpha`` on
    ``F(snap(x + alpha l s))`` -- the objective **after** the snap, so the search sees what the mesh will
    actually become.  A trial that would add an inverted element is never accepted.

    Stops on ``iters``, a step below ``tol_step`` edge lengths, a relative drop of ``F`` below ``tol_rel``, a mean
    excess below ``1e-10`` (already ideal), a zero projected gradient, or a line search that finds no step
    lowering ``F``.  ``restart="halve"`` halves ``beta`` on every second iteration, ``"none"`` is plain
    Fletcher-Reeves."""
    f, ninv = evaluate(P)
    history: list[tuple[int, int, float, float]] = []
    g_old = s = None
    alpha_prev = 0.1
    stopped, reason, done = "completed", "all %d iterations accepted" % iters, 0
    for it in range(1, iters + 1):
        if f / n_used < 1e-10:
            stopped, reason = "converged", "mean excess below 1e-10 (already ideal)"
            break
        gx = project(gradient(P), P)
        gx[~movable] = 0.0
        g = ell * gx                                     # gradient in u = dx / l
        gg = float((g * g).sum())
        if gg == 0.0:
            stopped, reason = "converged", "projected gradient is zero"
            break
        if s is None or g_old is None:
            s = -g
        else:
            beta = gg / float((g_old * g_old).sum())
            if restart == "halve" and it % 2 == 0:
                beta *= 0.5
            s = -g + beta * s
            if float((g * s).sum()) >= 0.0:              # lost descent: restart on steepest descent
                s = -g
        top = float(np.linalg.norm(s, axis=1).max())
        dirx = ell * s / top                             # largest move = 1 local edge per unit alpha
        if tangent.any():                                # a CG mix of older tangents: re-cut to today's
            dirx[tangent] = project(dirx, P)[tangent]

        def trial(a: float, P: PointArray = P, dirx: PointArray = dirx,
                  ninv: int = ninv) -> tuple[float, int, PointArray]:
            Q = snap(P + a * dirx)
            ff, nn = evaluate(Q)
            return (ff if nn <= ninv else np.inf), nn, Q

        # bracket by doubling from the last accepted step, then refine by golden section
        a = min(alpha_prev * 2.0, alpha_max)
        pts = [(0.0, f)]
        while True:
            fa = trial(a)[0]
            pts.append((a, fa))
            if (fa > pts[-2][1] and len(pts) > 2) or a >= alpha_max or not np.isfinite(fa) and len(pts) > 2:
                break
            if fa > f and len(pts) == 2:                 # first trial already worse: shrink
                a *= 0.25
                pts = [(0.0, f)]
                if a < tol_step * 0.1:
                    break
                continue
            a = min(a * 2.0, alpha_max)
        best = min(pts, key=lambda t: t[1])
        i = pts.index(best)
        lo = pts[max(i - 1, 0)][0]
        hi = pts[min(i + 1, len(pts) - 1)][0]
        gr = 0.5 * (np.sqrt(5.0) - 1.0)
        for _ in range(12):
            if hi - lo < 1e-3 * max(hi, 1e-12):
                break
            x1, x2 = hi - gr * (hi - lo), lo + gr * (hi - lo)
            f1, f2 = trial(x1)[0], trial(x2)[0]
            # an infeasible probe is inf: when both are, the feasible region lies below x1, so shrink from above
            # (``inf < inf`` is false and would walk ``lo`` up toward larger, infeasible steps)
            if f1 < f2 or (np.isinf(f1) and np.isinf(f2)):
                hi = x2
            else:
                lo = x1
        cand = [(trial(x)[0], x) for x in (best[0], 0.5 * (lo + hi))]
        f_new, a_best = min(cand)
        if not np.isfinite(f_new) or a_best <= 0.0 or f_new >= f:
            stopped, reason = "converged", "no step lowers the objective (stalled at %.3e)" % f
            break
        _, ninv_new, Pn = trial(a_best)
        rel = (f - f_new) / max(abs(f), 1e-300)
        P, f, ninv = Pn, f_new, ninv_new
        g_old, alpha_prev, done = g, max(a_best, 1e-3), it
        n_inv, sj_min, obj_mean = summary(P)
        history.append((it, n_inv, sj_min, obj_mean))
        if a_best < tol_step:
            stopped, reason = "converged", "step %.2e edge lengths below %.0e" % (a_best, tol_step)
            break
        if rel < tol_rel:
            stopped, reason = "converged", "relative drop %.1e below %.0e" % (rel, tol_rel)
            break
    return DescentResult(P, f, ninv, history, stopped, reason, done)
