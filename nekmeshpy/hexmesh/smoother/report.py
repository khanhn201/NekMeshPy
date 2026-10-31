"""The before/after reading of one hex smoothing round: :class:`HexRoundReport`.

The quad smoother's round report, for hexes, without the interactive session around it: the
caller runs :func:`~nekmeshpy.hexmesh.smoother.smooth.run_schedule` and hands the input, the
result and the resolved constraints to :func:`round_report`.  It answers, in order:

1. what were the constraints (vertex counts per kind, notes);
2. what ran (each stage and why it stopped);
3. quality before -> after: the corner reading (:class:`~nekmeshpy.hexmesh.smoother.smooth.Quality`,
   shape statistics), **per region** (``element_tags``) with counts below thresholds, and a
   **sampled** reading -- the trilinear map ``.re2`` exports, read on the GLL lattice of the
   solver's order (:func:`nekmeshpy.hexmesh.quality.linear_order_scan`), plus the curved map of a
   high-order input (:func:`nekmeshpy.hexmesh.quality.order_scan`); the order is stated;
4. geometry: is every constrained vertex still on its target -- per target, so a plane that held
   is visible as such -- and is every frozen vertex (a kept layer, a frozen region) bit-identical.

``failures`` block acceptance (off target, a frozen vertex moved, more inverted hexes);
``warnings`` do not.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import NamedTuple

import numpy as np

from ..._typing import PointArray
from ...core import quality as _core
from ...core.quality import OrderScan
from ..hexmesh import HexMesh
from ..quality import linear_order_scan, order_scan
from . import constrain as cn
from . import smooth as sm

__all__ = ["THRESHOLDS", "RegionQuality", "TargetHeld", "HexRoundReport", "region_quality", "targets_held",
           "round_report"]

#: Scaled-Jacobian thresholds the per-region rows count hexes below.
THRESHOLDS = (0.1, 0.2, 0.3)

_UNTAGGED = "(untagged)"


class RegionQuality(NamedTuple):
    """The corner reading of one region (one ``element_tags`` value)."""

    region: str
    n_hexes: int
    n_inverted: int
    sj_min: float
    sj_mean: float
    #: Hexes whose worst-corner scaled Jacobian is below each of the report's thresholds.
    below: tuple[int, ...]


class TargetHeld(NamedTuple):
    """How far the vertices held by one target sit from it."""

    name: str
    kind: str                  # "surface" | "curve"
    n_vertices: int
    max_distance: float


def region_quality(mesh: HexMesh, thresholds: Sequence[float] = THRESHOLDS) -> tuple[RegionQuality, ...]:
    """One :class:`RegionQuality` per element-tag value (untagged hexes as ``"(untagged)"``)."""
    sj = sm.corner_metrics(mesh.points, mesh.corners).sj.min(axis=1)
    dense = mesh.element_tags.to_dense()
    out = []
    for name in sorted(set(dense.tolist())):
        v = sj[dense == name]
        out.append(RegionQuality(name or _UNTAGGED, int(v.size), int((v <= 0).sum()), float(v.min()),
                                 float(v.mean()), tuple(int((v < t).sum()) for t in thresholds)))
    return tuple(out)


def targets_held(points: PointArray, r: cn.Resolved) -> tuple[TargetHeld, ...]:
    """One :class:`TargetHeld` per surface and curve that holds vertices in ``r``."""
    out = []
    groups: tuple[tuple[int, str, Sequence[cn.Surface | cn.Curve]], ...] = (
        (cn.ON_SURFACE, "surface", r.surfaces), (cn.ON_CURVE, "curve", r.curves))
    for kind, label, objs in groups:
        for t, obj in enumerate(objs):
            m = (r.kind == kind) & (r.target == t)
            if m.any():
                d = np.linalg.norm(obj.project(points[m]) - points[m], axis=1)
                out.append(TargetHeld(obj.name, label, int(m.sum()), float(d.max())))
    return tuple(out)


#: ``(metric, field, better)``: ``+1`` higher is better, ``-1`` lower, ``0`` neutral.
_METRICS: tuple[tuple[str, str, int], ...] = (
    ("inverted hexes", "n_inverted", -1), ("min scaled J", "sj_min", +1),
    ("worst 1% scaled J", "sj_q01", +1), ("worst 5% scaled J", "sj_q05", +1),
    ("worst 10% scaled J", "sj_q10", +1), ("median scaled J", "sj_median", +1),
    ("mean scaled J", "sj_mean", +1), ("mean phi", "phi_mean", -1), ("max phi", "phi_max", -1),
    ("objective / hex", "objective_mean", -1),
    ("max aspect ratio", "aspect_max", -1), ("95th pct aspect", "aspect_p95", -1),
    ("min edge angle (deg)", "angle_min_deg", +1), ("max edge angle (deg)", "angle_max_deg", -1),
    ("neighbour size ratio max", "size_ratio_max", -1), ("neighbour size ratio 95th", "size_ratio_p95", -1),
    ("neighbour size ratio mean", "size_ratio_mean", -1),
    ("node spacing min", "spacing_min", 0), ("node spacing max", "spacing_max", 0),
)


def _scan_text(scan: OrderScan | None) -> str:
    if scan is None:
        return "-"
    if not scan.orders:
        return "not checked (order %s declined by the scan budget)" % ", ".join(map(str, scan.skipped))
    return "  ".join("N=%d min %+.4f, %d inverted" % (n, m, i)
                     for n, m, i in zip(scan.orders, scan.min_sj, scan.n_inverted))


class HexRoundReport(NamedTuple):
    """The answers a hex smoothing round owes, plus the verdict.  Printing it gives the full
    before/after table."""

    constraints: str
    stages: tuple[sm.StageLog, ...]
    before: sm.Quality
    after: sm.Quality
    shape_before: sm.ShapeStats
    shape_after: sm.ShapeStats
    thresholds: tuple[float, ...]
    regions_before: tuple[RegionQuality, ...]
    regions_after: tuple[RegionQuality, ...]
    #: The order the sampled readings were taken at.
    sample_order: int
    #: The trilinear (``.re2``) map read at ``sample_order``, before and after.
    sampled_before: OrderScan
    sampled_after: OrderScan
    #: A high-order input's own curved map at ``sample_order`` (``None`` for a linear input).
    curved_before: OrderScan | None
    geometry_before: cn.OnTargetReport
    geometry_after: cn.OnTargetReport
    targets: tuple[TargetHeld, ...]
    #: Frozen vertices (kept layers, frozen regions, corners) and how many are not bit-identical
    #: to where they were held.
    n_frozen: int
    n_frozen_moved: int
    warnings: tuple[str, ...]
    failures: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return not self.failures

    def metric_rows(self) -> list[tuple[str, float, float, str]]:
        rows = []
        for name, key, better in _METRICS:
            src = (self.before, self.after) if key in sm.Quality._fields else (self.shape_before, self.shape_after)
            b, a = float(getattr(src[0], key)), float(getattr(src[1], key))
            tol = 1e-9 * max(abs(b), abs(a), 1.0)
            mark = "same" if abs(a - b) <= tol else ("changed" if better == 0 else
                                                     ("better" if (a - b) * better > 0 else "WORSE"))
            rows.append((name, b, a, mark))
        return rows

    def __str__(self) -> str:
        out = ["=== hex smoothing round: %s ===" % ("PASS" if self.passed else "FAIL"),
               "1. constraints", *("   " + ln for ln in self.constraints.splitlines()), "2. what ran"]
        for lg in self.stages:
            out.append("   %-14s %3d/%-3d %-11s %s" % (lg.method, lg.passes_done, lg.passes_requested,
                                                       lg.stopped, lg.reason))
        out.append("3. quality, before -> after (corner reading)")
        out.append("   %-26s %14s %14s" % ("metric", "before", "after"))
        for name, b, a, mark in self.metric_rows():
            out.append("   %-26s %14.6g %14.6g   %s" % (name, b, a, mark))
        th = "/".join("%g" % t for t in self.thresholds)
        out.append("   per region (hexes, inverted, min scaled J, below %s):" % th)
        for rb, ra in zip(self.regions_before, self.regions_after):
            out.append("   %-16s %5d  inv %d -> %d  min %.4f -> %.4f  below %s -> %s"
                       % (rb.region, rb.n_hexes, rb.n_inverted, ra.n_inverted, rb.sj_min, ra.sj_min,
                          "/".join(map(str, rb.below)), "/".join(map(str, ra.below))))
        out.append("   sampled at order %d (.re2 trilinear map): before %s" % (self.sample_order,
                                                                             _scan_text(self.sampled_before)))
        out.append("   %s after  %s" % (" " * 38, _scan_text(self.sampled_after)))
        if self.curved_before is not None:
            out.append("   high-order input, its curved map at order %d: %s (the result is linear)"
                       % (self.sample_order, _scan_text(self.curved_before)))
        out.append("4. geometry  before: %s" % (self.geometry_before,))
        out.append("             after:  %s" % (self.geometry_after,))
        for t in self.targets:
            out.append("   %-8s %-10s %5d vertices, max distance %.2e" % (t.kind, t.name, t.n_vertices,
                                                                         t.max_distance))
        out.append("   frozen: %d vertices, %d not bit-identical" % (self.n_frozen, self.n_frozen_moved))
        if self.warnings:
            out += ["warnings:"] + ["   ! " + w for w in self.warnings]
        if self.failures:
            out += ["FAILURES (round cannot be accepted):"] + ["   X " + f for f in self.failures]
        return "\n".join(out)


def _constraint_text(r: cn.Resolved) -> str:
    c = r.counts()
    lines = ["vertices: %s" % ", ".join("%d %s" % (v, k.replace("_", "-")) for k, v in c.items()),
             "targets: %d surface(s), %d curve(s)" % (len(r.surfaces), len(r.curves))]
    return "\n".join(lines + ["note: " + n for n in r.notes])


def round_report(before: HexMesh, after: HexMesh, resolved: cn.Resolved, *,
                 stages: Sequence[sm.StageLog] = (), original: HexMesh | None = None,
                 thresholds: Sequence[float] = THRESHOLDS, sample_order: int | None = None,
                 geom_tol: float = 1e-6) -> HexRoundReport:
    """The :class:`HexRoundReport` of smoothing ``before`` into ``after`` (both linear, same
    connectivity) under ``resolved``.  ``original`` is the high-order mesh ``before`` is the
    shadow of, when there is one: its curved map is sampled too.  ``sample_order`` defaults to
    :data:`SCAN_ORDER <nekmeshpy.core.quality.SCAN_ORDER>` (the solver's); a mesh above the scan
    budget is declined there and reported ``not checked`` -- unchecked, not clean."""
    if before.n_hexes != after.n_hexes or not np.array_equal(before.corners, after.corners):
        raise ValueError("round_report: before and after must share their connectivity")
    n = _core.SCAN_ORDER if sample_order is None else int(sample_order)
    th = tuple(float(t) for t in thresholds)
    qb = sm.quality(before.points, before.corners)
    qa = sm.quality(after.points, after.corners)
    shb = sm.shape_stats(before.points, before.corners)
    sha = sm.shape_stats(after.points, after.corners)
    gb = cn.on_target_report(before.points, before.corners, resolved, tol=geom_tol)
    ga = cn.on_target_report(after.points, after.corners, resolved, tol=geom_tol)
    sb = linear_order_scan(before, (n,))
    sa = linear_order_scan(after, (n,))
    cb = order_scan(original, (max(n, original.order),)) if original is not None and original.order > 1 else None
    fro = resolved.kind == cn.FROZEN
    moved = int((np.asarray(after.points)[fro] != resolved.anchor[fro]).any(axis=1).sum())
    warns: list[str] = []
    fails: list[str] = []
    if not ga.passed:
        fails.append("geometry: %s" % (ga,))
    if moved:
        fails.append("%d frozen vertices are not bit-identical to where they were held" % moved)
    if qa.n_inverted > qb.n_inverted:
        fails.append("inverted hexes rose %d -> %d" % (qb.n_inverted, qa.n_inverted))
    if qa.n_inverted > 0:
        warns.append("%d inverted hex(es) remain (corner reading)" % qa.n_inverted)
    if sa.orders and not sa.clean:
        warns.append("%d hex(es) fold in the .re2 trilinear map sampled at order %d (min %.4f)"
                     % (sa.n_inverted[0], n, sa.min_sj[0]))
    if sa.skipped:
        warns.append("sampled reading at order %d declined by the scan budget: unchecked" % n)
    rep = HexRoundReport(_constraint_text(resolved), tuple(stages), qb, qa, shb, sha, th,
                         region_quality(before, th), region_quality(after, th), n, sb, sa, cb, gb, ga,
                         targets_held(np.asarray(after.points, float), resolved), int(fro.sum()), moved,
                         (), ())
    for name, b, a, mark in rep.metric_rows():
        if mark == "WORSE" and name != "inverted hexes":
            warns.append("%s got WORSE: %.6g -> %.6g" % (name, b, a))
    for lg in stages:
        if lg.stopped == "rolled_back":
            warns.append("%s stage rolled back: %s" % (lg.method, lg.reason))
    return rep._replace(warnings=tuple(warns), failures=tuple(fails))
