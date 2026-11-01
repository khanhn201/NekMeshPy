"""The before/after reading of one hex smoothing round: :class:`HexRoundReport`, and the hex rung's
:class:`HexAdapter` behind the shared session.

The report is the shared :class:`~nekmeshpy.core.smoother.session.RoundReport` with the hex words and the
hex sections (:class:`~nekmeshpy.core.smoother.session.Audit`): the caller either runs a
:class:`~nekmeshpy.hexmesh.smoother.session.SmoothSession`, or runs
:func:`~nekmeshpy.hexmesh.smoother.smooth.run_schedule` itself and hands the input, the result and the
resolved constraints to :func:`round_report`.  It answers, in order:

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

import os
from collections.abc import Sequence
from typing import Any, NamedTuple

import numpy as np

from ..._typing import PointArray
from ...core import quality as _core
from ...core.quality import OrderScan
from ...core.smoother import session as _session
from ...core.smoother.constraints import targets
from ...core.smoother.quality import with_scaled_jacobian
from ...core.smoother.schedule import Schedule, StageLog
from ...io import writer
from ..hexmesh import HexMesh
from ..quality import corner_scaled_jacobian, linear_order_scan, order_scan, scaled_jacobian
from ..query import element_blocks
from . import constrain as cn
from . import highorder
from . import smooth as sm

__all__ = ["THRESHOLDS", "RegionQuality", "TargetHeld", "HexRoundReport", "HexAdapter", "region_quality",
           "targets_held", "round_report"]

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


def _scan_text(scan: OrderScan | None) -> str:
    if scan is None:
        return "-"
    if not scan.orders:
        return "not checked (order %s declined by the scan budget)" % ", ".join(map(str, scan.skipped))
    return "  ".join("N=%d min %+.4f, %d inverted" % (n, m, i)
                     for n, m, i in zip(scan.orders, scan.min_sj, scan.n_inverted))


class HexRoundReport(_session.RoundReport):
    """The answers a hex smoothing round owes, plus the verdict (the shared report, with hexes' words).  Printing
    it gives the full before/after table.  The hex sections are read back from ``data`` as attributes."""

    __slots__ = ()
    noun = "hex"
    plural = "hexes"
    angle_word = "edge"

    @property
    def thresholds(self) -> tuple[float, ...]:
        """The scaled-Jacobian thresholds the per-region rows count below."""
        return tuple(self.data["thresholds"])

    @property
    def regions_before(self) -> tuple[RegionQuality, ...]:
        return tuple(self.data["regions_before"])

    @property
    def regions_after(self) -> tuple[RegionQuality, ...]:
        return tuple(self.data["regions_after"])

    @property
    def sample_order(self) -> int:
        """The order the sampled readings were taken at."""
        return int(self.data["sample_order"])

    @property
    def sampled_before(self) -> OrderScan:
        """The trilinear (``.re2``) map read at ``sample_order``, before."""
        return self.data["sampled_before"]  # type: ignore[no-any-return]

    @property
    def sampled_after(self) -> OrderScan:
        return self.data["sampled_after"]  # type: ignore[no-any-return]

    @property
    def curved_before(self) -> OrderScan | None:
        """A high-order input's own curved map at ``sample_order`` (``None`` for a linear input)."""
        return self.data["curved_before"]  # type: ignore[no-any-return]

    @property
    def curved_after(self) -> OrderScan | None:
        """The result's own curved map at ``sample_order`` (``None`` for a linear result)."""
        return self.data["curved_after"]  # type: ignore[no-any-return]

    @property
    def corners_before(self) -> tuple[int, float]:
        """``(inverted, min scaled J)`` of the straight-sided hexes through the 8 corners alone -- what ``.re2``
        writes -- of a curved input (``(0, nan)`` for a linear one: the ordinary reading is already that)."""
        return self.data["corners_before"]  # type: ignore[no-any-return]

    @property
    def corners_after(self) -> tuple[int, float]:
        return self.data["corners_after"]  # type: ignore[no-any-return]

    @property
    def targets(self) -> tuple[TargetHeld, ...]:
        return tuple(self.data["targets"])

    @property
    def n_frozen(self) -> int:
        """Frozen vertices (kept layers, frozen regions, corners)."""
        return int(self.data["n_frozen"])

    @property
    def n_frozen_moved(self) -> int:
        """How many of them are not bit-identical to where they were held."""
        return int(self.data["n_frozen_moved"])


class HexAdapter(_session.RungAdapter):
    """The hex rung behind the shared session: hex readings, per-region and sampled sections, ``.vtu`` files.
    ``original`` is the high-order mesh the linear input is the shadow of, when there is one (its curved map is
    sampled too); ``sample_order`` defaults to :data:`SCAN_ORDER <nekmeshpy.core.quality.SCAN_ORDER>` (the
    solver's); a mesh above the scan budget is declined there and reported ``not checked`` -- unchecked, not
    clean.  There is no input gate: a hex mesh is conforming by construction.  A curved mesh stays curved through a
    round and is read on the nodes it stores (and, in the audit, on the ``sample_order`` lattice and at the corners
    ``.re2`` writes)."""

    report_type = HexRoundReport
    no_rebuild = ("hexmesh.smoother cannot rebuild the curved nodes of a smoothed mesh yet: the result of "
                  "smoothing a curved mesh is the linear mesh")

    def __init__(self, *, thresholds: Sequence[float] = THRESHOLDS, sample_order: int | None = None,
                 original: HexMesh | None = None) -> None:
        self.thresholds = tuple(float(t) for t in thresholds)
        self.sample_order = sample_order
        self.original = original
        self.curved = 0                # the order of a curved input the session is smoothing (0: linear)

    def linearize(self, mesh: HexMesh) -> HexMesh:
        return highorder.linearize(mesh)

    def resolve(self, mesh: HexMesh, constraints: list[Any], boundary: str) -> cn.Resolved:
        return cn.resolve(mesh, constraints, boundary=boundary)

    def quality(self, mesh: HexMesh) -> sm.Quality:
        q = sm.quality(mesh.points, mesh.corners)
        if mesh.order > 1:             # a curved mesh is read on the nodes it stores, not on its corners alone
            q = with_scaled_jacobian(q, scaled_jacobian(mesh, mesh.order))
        return q

    def shape_stats(self, mesh: HexMesh) -> sm.ShapeStats:
        return sm.shape_stats(mesh.points, mesh.corners)

    def on_target(self, mesh: HexMesh, resolved: cn.Resolved, tol: float) -> cn.OnTargetReport:
        return cn.on_target_report(mesh.points, mesh.corners, resolved, tol=tol)

    def describe(self, resolved: cn.Resolved, *, sj_target: float, geom_tol: float) -> str:
        c = resolved.counts()
        lines = ["vertices: %s" % ", ".join("%d %s" % (v, k.replace("_", "-")) for k, v in c.items()),
                 "targets: %d surface(s), %d curve(s)" % (len(resolved.surfaces), len(resolved.curves))]
        if self.curved:
            lines.append("curved mesh (order %d): tmop optimises every node; the linear stages move the vertices and "
                         "the nodes follow; edge and face nodes of a held vertex are held" % self.curved)
        return "\n".join(lines + ["note: " + n for n in resolved.notes])

    def audit(self, before: HexMesh, after: HexMesh, resolved: cn.Resolved) -> _session.Audit:
        n = _core.SCAN_ORDER if self.sample_order is None else int(self.sample_order)
        th = self.thresholds
        sb, sa = linear_order_scan(before, (n,)), linear_order_scan(after, (n,))
        orig = self.original
        if before.order > 1:
            cb: OrderScan | None = order_scan(before, (max(n, before.order),))
        else:
            cb = order_scan(orig, (max(n, orig.order),)) if orig is not None and orig.order > 1 else None
        ca = order_scan(after, (max(n, after.order),)) if after.order > 1 else None
        kb = ka = (0, float("nan"))
        if before.order > 1:           # .re2 has no curved format: what it writes is the corners alone
            csb, csa = corner_scaled_jacobian(before.points, before.corners), corner_scaled_jacobian(after.points, after.corners)
            kb, ka = (int((csb <= 0).sum()), float(csb.min())), (int((csa <= 0).sum()), float(csa.min()))
        rb, ra = region_quality(before, th), region_quality(after, th)
        tg = targets_held(np.asarray(after.points, float), resolved)
        fro = resolved.kind == cn.FROZEN
        moved = int((np.asarray(after.points)[fro] != resolved.anchor[fro]).any(axis=1).sum())
        warns: list[str] = []
        fails: list[str] = []
        if moved:
            fails.append("%d frozen vertices are not bit-identical to where they were held" % moved)
        if sa.orders and not sa.clean:
            warns.append("%d hex(es) fold in the .re2 trilinear map sampled at order %d (min %.4f)"
                         % (sa.n_inverted[0], n, sa.min_sj[0]))
        if sa.skipped:
            warns.append("sampled reading at order %d declined by the scan budget: unchecked" % n)
        if ca is not None and ca.orders and not ca.clean:
            warns.append("%d hex(es) fold in the curved map sampled at order %d (min %.4f)"
                         % (ca.n_inverted[0], n, ca.min_sj[0]))
        if ka[0] > kb[0]:
            warns.append("%d hex(es) invert at the corners alone (%d before): .re2 writes the corners only"
                         % (ka[0], kb[0]))
        d3 = ["per region (hexes, inverted, min scaled J, below %s):" % "/".join("%g" % t for t in th)]
        for x, y in zip(rb, ra):
            d3.append("%-16s %5d  inv %d -> %d  min %.4f -> %.4f  below %s -> %s"
                      % (x.region, x.n_hexes, x.n_inverted, y.n_inverted, x.sj_min, y.sj_min,
                         "/".join(map(str, x.below)), "/".join(map(str, y.below))))
        d3.append("sampled at order %d (.re2 trilinear map): before %s" % (n, _scan_text(sb)))
        d3.append("%s after  %s" % (" " * 38, _scan_text(sa)))
        if cb is not None and before.order > 1:
            d3.append("curved map at order %d: before %s" % (n, _scan_text(cb)))
            d3.append("%s after  %s" % (" " * 24, _scan_text(ca)))
            d3.append(".re2 corners alone: min scaled J %.4f -> %.4f, inverted %d -> %d" % (kb[1], ka[1], kb[0], ka[0]))
        elif cb is not None:
            d3.append("high-order input, its curved map at order %d: %s (the result is linear)"
                      % (n, _scan_text(cb)))
        d4 = ["%-8s %-10s %5d vertices, max distance %.2e" % (t.kind, t.name, t.n_vertices, t.max_distance)
              for t in tg]
        d4.append("frozen: %d vertices, %d not bit-identical" % (int(fro.sum()), moved))
        data = {"thresholds": th, "regions_before": rb, "regions_after": ra, "sample_order": n,
                "sampled_before": sb, "sampled_after": sa, "curved_before": cb, "curved_after": ca,
                "corners_before": kb, "corners_after": ka, "targets": tg,
                "n_frozen": int(fro.sum()), "n_frozen_moved": moved}
        return _session.Audit(tuple(warns), tuple(fails), data, tuple(d3), tuple(d4))

    def run(self, mesh: HexMesh, resolved: cn.Resolved, schedule: Schedule, *, sj_target: float,
            weights: Any) -> tuple[HexMesh, list[StageLog]]:
        return sm.run_schedule(mesh, resolved, schedule, sj_target=sj_target, weights=weights)

    def write(self, directory: str, before: HexMesh, after: HexMesh, resolved: cn.Resolved) -> dict[str, str]:
        files = {}
        for tag, M in (("before", before), ("after", after)):
            m = sm.corner_metrics(M.points, M.corners)
            if M.order > 1:            # a curved mesh has more nodes than points: per-element data only
                move = np.abs(element_blocks(M) - element_blocks(before)).max(axis=(1, 2)) if before.order == M.order \
                    else np.zeros(M.n_hexes)
                path = os.path.join(directory, tag + ".vtu")
                writer.to_vtu(M, path, cell_data={"sj_min": scaled_jacobian(M, M.order), "sj_corner_min": m.sj.min(1),
                                                  "node_move_max": move})
                files[tag] = path
                continue
            dist = np.zeros(M.n_points)
            for k, objs in targets(resolved):
                for t, o in enumerate(objs):
                    sel = (resolved.kind == k) & (resolved.target == t)
                    if sel.any():
                        dist[sel] = np.linalg.norm(o.project(M.points[sel]) - M.points[sel], axis=1)
            path = os.path.join(directory, tag + ".vtu")
            writer.to_vtu(M, path, point_data={"constraint_kind": resolved.kind.astype(float),
                                               "dist_to_target": dist, "displacement": M.points - before.points},
                          cell_data={"sj_min": m.sj.min(1), "phi_max": m.phi.max(1)})
            files[tag] = path
        return files


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
    ad = HexAdapter(thresholds=thresholds, sample_order=sample_order, original=original)
    rep = _session.build_report(ad, before, after, resolved, list(stages), geom_tol=geom_tol)
    assert isinstance(rep, HexRoundReport)
    return rep
