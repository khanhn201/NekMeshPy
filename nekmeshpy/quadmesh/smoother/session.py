"""The interactive layer of the quad smoother: :class:`SmoothSession` is the shared
:class:`~nekmeshpy.core.smoother.session.SmoothSession` bound to the quad rung's adapter, so a round is run,
reported (:class:`RoundReport`, the six answers), accepted and undone exactly as at the hex rung.

What is quad-specific lives here: the gap gate (:class:`ConnectivityError`), the overlap count, the surface
normals the metric reads, the high-order rebuild and the per-round ``.vtu`` with its quad diagnostics.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from ..._typing import PointArray
from ...core.smoother import session as _core
from ...core.smoother.constraints import targets
from ...core.smoother.schedule import Schedule, StageLog
from ...io import writer
from ..quadmesh import QuadMesh
from ..query import ConnectivityReport, check_connectivity, overlapping_quads
from . import constrain as cn
from . import smooth
from .highorder import HighOrderReport, linearize, rebuild_high_order

__all__ = ["ConnectivityError", "RoundReport", "Proposal", "SmoothSession", "adjust_schedule"]

Proposal = _core.Proposal
adjust_schedule = _core.adjust_schedule


class ConnectivityError(RuntimeError):
    """The gap check failed and ``allow_gaps`` was not set."""


class RoundReport(_core.RoundReport):
    """The six answers a quad round owes, plus the verdict (see the shared base).  ``overlaps_before`` /
    ``overlaps_after``: quads taking part in an overlap (``-1``: not checked, ``check_overlaps=False``)."""

    __slots__ = ()
    noun = "quad"
    plural = "quads"
    propagate_label = "propagate to hex"

    def _overlaps(self, side: int) -> int:
        for c in self.counts:
            if c.label == "overlapping quads":
                return c.after if side else c.before
        return -1

    @property
    def overlaps_before(self) -> int:
        return self._overlaps(0)

    @property
    def overlaps_after(self) -> int:
        return self._overlaps(1)


class QuadAdapter(_core.RungAdapter):
    """The quad rung behind the shared session: gap gate, surface normals, overlaps, high-order rebuild, ``.vtu``."""

    report_type = RoundReport
    propagate = "not applicable: quad-only (no hex provenance recorded yet)"

    def __init__(self, *, normals: PointArray | None, surface: cn.Surface | None, check_normals: bool,
                 check_overlaps: bool, normal_cos_tol: float, connectivity_tol: float, allow_gaps: bool) -> None:
        self.normals = normals
        self.surface = surface
        self.check_normals = check_normals
        self.check_overlaps = check_overlaps
        self.normal_cos_tol = normal_cos_tol
        self.connectivity_tol = connectivity_tol
        self.allow_gaps = allow_gaps
        self.connectivity: ConnectivityReport | None = None
        self.normal_report: cn.NormalReport | None = None
        self.input_overlaps = -1

    def preflight(self, mesh: QuadMesh) -> None:
        self.connectivity = check_connectivity(mesh, connectivity_tol=self.connectivity_tol)
        if not self.connectivity.passed and not self.allow_gaps:
            raise ConnectivityError("the gap check failed; patch the connectivity first, or pass "
                                    "allow_gaps=True to smooth anyway (it will be logged):\n%s"
                                    % (self.connectivity,))

    def linearize(self, mesh: QuadMesh) -> QuadMesh:
        return linearize(mesh)

    def resolve(self, mesh: QuadMesh, constraints: list[Any], boundary: str) -> cn.Resolved:
        return cn.resolve(mesh, constraints, boundary=boundary)

    def prepare(self, mesh: QuadMesh, resolved: cn.Resolved) -> None:
        if self.normals is None:
            # precedence: explicit normals > ``surface`` (every vertex) > the OnSurface constraints > the
            # quads' own winding.  The winding is the weakest source: one folded input quad poisons it.
            groups = ([(np.arange(mesh.n_points), self.surface)] if self.surface is not None
                      else cn.claimed_surfaces(resolved))
            if groups:
                self.normals, self.normal_report = cn.surface_normals(
                    mesh.points, mesh.corners, groups, check=self.check_normals, cos_tol=self.normal_cos_tol)
                if self.normal_report.n_disagree:
                    logging.warning(
                        "%d of %d quads disagree with the surface normal by more than cos %.2f (min cos %.2f): "
                        "a folded or reversed input quad, or the wrong surface; the surface normals are used",
                        self.normal_report.n_disagree, self.normal_report.n_quads, self.normal_cos_tol,
                        self.normal_report.min_cos)
            else:
                self.normals = smooth.reference_normals(mesh.points, mesh.corners)
        if self.check_overlaps:
            self.input_overlaps = int(overlapping_quads(mesh).size)
            if self.input_overlaps:
                logging.warning("%d input quad(s) overlap another quad: smoothing can leave a persistent local "
                                "overlap (a branched star) around them, invisible to the inverted-quad count; "
                                "check_overlaps reports it each round", self.input_overlaps)

    def rebuild(self, base: QuadMesh, points: PointArray, resolved: cn.Resolved,
                order: int | None) -> tuple[QuadMesh, HighOrderReport]:
        return rebuild_high_order(base, points, resolved, order=order)

    def quality(self, mesh: QuadMesh) -> smooth.Quality:
        assert self.normals is not None
        return smooth.quality(mesh.points, mesh.corners, self.normals)

    def shape_stats(self, mesh: QuadMesh) -> smooth.ShapeStats:
        return smooth.shape_stats(mesh.points, mesh.corners)

    def on_target(self, mesh: QuadMesh, resolved: cn.Resolved, tol: float) -> cn.OnTargetReport:
        return cn.on_target_report(mesh.points, mesh.corners, resolved, tol=tol)

    def counts(self, mesh: QuadMesh) -> tuple[tuple[str, int, str], ...]:
        if not self.check_overlaps:
            return ()
        return (("overlapping quads", int(overlapping_quads(mesh).size),
                 "%d overlapping quad(s) remain (positive Jacobians do not rule this out)"),)

    def describe(self, resolved: cn.Resolved, *, sj_target: float, geom_tol: float) -> str:
        c = resolved.counts()
        lines = ["vertices: %s" % ", ".join("%d %s" % (v, k.replace("_", "-")) for k, v in c.items()),
                 "targets: %d surface(s), %d curve(s); geometry mode: strict (snap + target-normal gradient every step)"
                 % (len(resolved.surfaces), len(resolved.curves)),
                 "limits: scaled-J target %.3g (penalty eps), on-target tol %.0e x local edge, connectivity tol %.2f"
                 % (sj_target, geom_tol, self.connectivity_tol)]
        lines += ["note: " + n for n in resolved.notes]
        assert self.connectivity is not None
        if not self.connectivity.passed:
            lines.append("note: GAP CHECK FAILED and overridden (allow_gaps): %d gap cluster(s)"
                         % self.connectivity.n_gap_clusters)
        return "\n".join(lines)

    def run(self, mesh: QuadMesh, resolved: cn.Resolved, schedule: Schedule, *, sj_target: float,
            weights: Any) -> tuple[QuadMesh, list[StageLog]]:
        return smooth.run_schedule(mesh, resolved, schedule, normals=self.normals, sj_target=sj_target,
                                   weights=weights)

    def write(self, directory: str, before: QuadMesh, after: QuadMesh, resolved: cn.Resolved) -> dict[str, str]:
        import os

        files = {}
        assert self.normals is not None
        for tag, M in (("before", before), ("after", after)):
            m = smooth.corner_metrics(M.points, M.corners, self.normals)
            dist = np.zeros(M.n_points)
            for k, objs in targets(resolved):
                for t, o in enumerate(objs):
                    sel = (resolved.kind == k) & (resolved.target == t)
                    if sel.any():
                        dist[sel] = np.linalg.norm(o.project(M.points[sel]) - M.points[sel], axis=1)
            path = os.path.join(directory, tag + ".vtu")
            writer.quad_to_vtu(M, path,
                               point_data={"constraint_kind": resolved.kind.astype(float), "dist_to_target": dist,
                                           "displacement": M.points - before.points},
                               cell_data={"sj_min": m.sj.min(1), "phi_max": m.phi.max(1),
                                          **({"overlap": np.isin(np.arange(M.n_quads),
                                                                 overlapping_quads(M)).astype(int)}
                                             if self.check_overlaps else {})})
            files[tag] = path
        return files


class SmoothSession(_core.SmoothSession):
    """One smoothing job on one quad mesh (a curved one is smoothed as its straight-sided shadow).  The shared
    session's flow -- :meth:`propose`, :meth:`accept`, :meth:`undo`, :meth:`run` -- with the quad options:
    ``surface`` (the surface the quads sit on, for the metric's normals), ``normals`` (explicit, overriding it),
    ``check_normals``, ``check_overlaps`` (count overlapping quads each round), ``allow_gaps`` (override the gap
    gate; ``connectivity_tol`` its tolerance)."""

    def __init__(self, mesh: QuadMesh, constraints: list[cn.Constraint], boundary: str = "frozen",
                 normals: PointArray | None = None, surface: cn.Surface | None = None,
                 check_normals: bool = True, check_overlaps: bool = True, normal_cos_tol: float = 0.5,
                 sj_target: float = smooth.EPS, geom_tol: float = 1e-6, connectivity_tol: float = 0.2,
                 allow_gaps: bool = False, weights: PointArray | None = None, out_dir: str | None = None,
                 name: str = "smooth") -> None:
        self._quad = QuadAdapter(normals=normals, surface=surface, check_normals=check_normals,
                                 check_overlaps=check_overlaps, normal_cos_tol=normal_cos_tol,
                                 connectivity_tol=connectivity_tol, allow_gaps=allow_gaps)
        super().__init__(mesh, constraints, self._quad, boundary=boundary, sj_target=sj_target,
                         geom_tol=geom_tol, weights=weights, out_dir=out_dir, name=name)

    @property
    def connectivity(self) -> ConnectivityReport:
        assert self._quad.connectivity is not None
        return self._quad.connectivity

    @property
    def normals(self) -> PointArray | None:
        return self._quad.normals

    @property
    def normal_report(self) -> cn.NormalReport | None:
        """Set when the normals were read from a surface (``None`` for explicit or reference normals)."""
        return self._quad.normal_report

    @property
    def input_overlaps(self) -> int:
        """Quads of the *input* that overlap another quad (``-1`` when ``check_overlaps=False``)."""
        return self._quad.input_overlaps
