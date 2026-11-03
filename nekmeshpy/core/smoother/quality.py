"""Quality readings of a smoothed mesh, from per-corner and per-element numbers any rung can supply.

The rung builds the corner metrics (:mod:`nekmeshpy.core.smoother.metric`), its edge lengths and its
neighbour table; the reductions here turn them into the numbers a report prints.  The rung keeps its own
``Quality`` tuple (its element count is named ``n_quads`` / ``n_hexes``) and fills it from :func:`reduce_corners`;
:class:`ShapeStats` has the same fields at every rung and is shared.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import numpy as np

from ..._typing import BoolArray, FloatArray, IntArray
from .metric import EPS, CornerMetrics

__all__ = ["ShapeStats", "history_row", "with_scaled_jacobian", "reduce_corners", "failing_mask", "corner_angles", "pair_ratios", "shape_stats"]


def reduce_corners(m: CornerMetrics, objective: float) -> tuple[
        int, int, float, float, float, float, float, float, float, float, float]:
    """The fields of a rung's ``Quality`` from corner readings ``m`` and the objective ``F``, in order: element
    count, inverted count, then the worst-corner scaled Jacobian's min / mean / median / 1% / 5% / 10% quantiles,
    the worst-corner ``phi``'s mean and max, and ``F`` per element.  Per-element values are the *worst corner*
    (``sj`` its minimum, ``phi`` its maximum)."""
    sj = m.sj.min(axis=1)
    phi = m.phi.max(axis=1)
    q = np.asarray(np.quantile(sj, [0.01, 0.05, 0.10])) if sj.size else np.zeros(3)
    return (int(sj.size), int((sj <= 0).sum()), float(sj.min(initial=np.inf)),
            float(sj.mean()) if sj.size else 0.0, float(np.median(sj)) if sj.size else 0.0,
            float(q[0]), float(q[1]), float(q[2]),
            float(phi.mean()) if phi.size else 0.0, float(phi.max(initial=0.0)),
            objective / max(sj.size, 1))


def with_scaled_jacobian(q: Any, sj: FloatArray) -> Any:
    """A rung's ``Quality`` ``q`` (a corner reading) with its scaled-Jacobian fields -- inverted count, min, mean, median
    and the 1% / 5% / 10% quantiles -- replaced by those of the per-element values ``sj`` ``(n,)``: how a curved mesh is
    read on the nodes it stores.  The shape metric ``phi`` and the objective stay the corner reading (the metric is
    defined on corners)."""
    qs = np.asarray(np.quantile(sj, [0.01, 0.05, 0.10])) if sj.size else np.zeros(3)
    return q._replace(n_inverted=int((sj <= 0).sum()), sj_min=float(sj.min(initial=np.inf)),
                      sj_mean=float(sj.mean()) if sj.size else 0.0, sj_median=float(np.median(sj)) if sj.size else 0.0,
                      sj_q01=float(qs[0]), sj_q05=float(qs[1]), sj_q10=float(qs[2]))


def history_row(q: Any) -> tuple[int, float, float]:
    """``(n_inverted, sj_min, objective_mean)`` of a rung's ``Quality`` -- what a stage's history logs."""
    return int(q.n_inverted), float(q.sj_min), float(q.objective_mean)


def failing_mask(m: CornerMetrics, edge: FloatArray, *, sj_target: float = EPS,
                 short_edge: float = 0.1) -> BoolArray:
    """``(n_elements,)`` the elements a local stage should work on: a corner with signed scaled Jacobian below
    ``sj_target``, or an edge (``edge`` ``(n_elements, k)`` lengths) shorter than ``short_edge`` times the median."""
    sj = m.sj.min(axis=1)
    return np.asarray((sj < sj_target) | (edge.min(axis=1) < short_edge * np.median(edge)))


class ShapeStats(NamedTuple):
    """Edge- and size-based readings that complement a rung's ``Quality``: aspect ratio (longest over
    shortest edge of an element), node spacing (the shortest and longest edge anywhere), corner angles, and
    the **size jump between neighbouring elements** (``size_ratio_*``: the larger over the smaller of the
    two characteristic sizes of every pair of elements sharing an edge or face, so ``1`` is a perfectly
    graded mesh).  The scaled Jacobian does not see that last one at all."""

    aspect_max: float
    aspect_p95: float
    spacing_min: float
    spacing_max: float
    angle_min_deg: float
    angle_max_deg: float
    size_ratio_max: float
    size_ratio_p95: float
    size_ratio_mean: float


def corner_angles(u: FloatArray, v: FloatArray) -> FloatArray:
    """Angle in degrees between the edge vectors ``u`` and ``v`` (``(n, k, 3)`` each) at every corner."""
    cosang = np.einsum("ekj,ekj->ek", u, v) / np.maximum(
        np.linalg.norm(u, axis=2) * np.linalg.norm(v, axis=2), 1e-300)
    return np.asarray(np.degrees(np.arccos(np.clip(cosang, -1.0, 1.0))))


def pair_ratios(size: FloatArray, pairs: IntArray) -> FloatArray:
    """``(M,)`` larger-over-smaller ratio of the element sizes ``size`` for every neighbour pair in ``pairs``
    ``(M, 2)``."""
    if pairs.size == 0:
        return np.ones(0)
    ra = size[pairs[:, 0]] / size[pairs[:, 1]]
    return np.asarray(np.maximum(ra, 1.0 / ra))


def shape_stats(edge: FloatArray, angles: FloatArray, ratios: FloatArray) -> ShapeStats:
    """:class:`ShapeStats` from ``edge`` ``(n, k)`` lengths, corner ``angles`` (degrees, any shape) and the
    neighbour size ``ratios`` (:func:`pair_ratios`)."""
    aspect = edge.max(axis=1) / np.maximum(edge.min(axis=1), 1e-300)
    return ShapeStats(float(aspect.max(initial=1.0)), float(np.quantile(aspect, 0.95)) if aspect.size else 1.0,
                      float(edge.min(initial=np.inf)), float(edge.max(initial=0.0)),
                      float(angles.min(initial=180.0)), float(angles.max(initial=0.0)),
                      float(ratios.max(initial=1.0)), float(np.quantile(ratios, 0.95)) if ratios.size else 1.0,
                      float(ratios.mean()) if ratios.size else 1.0)
