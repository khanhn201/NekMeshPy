"""The interactive layer of the hex smoother: :class:`SmoothSession` is the shared
:class:`~nekmeshpy.core.smoother.session.SmoothSession` bound to the hex rung's adapter
(:class:`~nekmeshpy.hexmesh.smoother.report.HexAdapter`), so a round is run, reported
(:class:`~nekmeshpy.hexmesh.smoother.report.HexRoundReport`), accepted and undone exactly as at the quad rung.

The hex session takes a **linear** mesh (``linearize`` a curved one: the curved nodes are not rebuilt yet),
has no input gate (a hex mesh is conforming by construction) and no surface normals (a hex has a volume
Jacobian).  Each round's report adds, per region, the corner reading and counts below thresholds, and the
trilinear ``.re2`` map sampled at the solver's order.
"""

from __future__ import annotations

from collections.abc import Sequence

from ..._typing import FloatArray
from ...core.smoother import session as _core
from ...core.smoother.session import Proposal, RoundReport, adjust_schedule
from ..hexmesh import HexMesh
from . import constrain as cn
from . import smooth
from .report import THRESHOLDS, HexAdapter, HexRoundReport

__all__ = ["Proposal", "RoundReport", "HexRoundReport", "SmoothSession", "adjust_schedule"]


class SmoothSession(_core.SmoothSession):
    """One smoothing job on one linear hex mesh.  The shared session's flow -- :meth:`propose`, :meth:`accept`,
    :meth:`undo`, :meth:`run` -- with the hex options: ``sample_order`` (the order the sampled readings are taken
    at; default the solver's, :data:`~nekmeshpy.core.quality.SCAN_ORDER`) and ``thresholds`` (the scaled-Jacobian
    levels the per-region rows count below)."""

    def __init__(self, mesh: HexMesh, constraints: list[cn.Constraint], boundary: str = "frozen",
                 sj_target: float = smooth.EPS, geom_tol: float = 1e-6, weights: FloatArray | None = None,
                 out_dir: str | None = None, name: str = "smooth", *, sample_order: int | None = None,
                 thresholds: Sequence[float] = THRESHOLDS) -> None:
        super().__init__(mesh, constraints, HexAdapter(thresholds=thresholds, sample_order=sample_order),
                         boundary=boundary, sj_target=sj_target, geom_tol=geom_tol, weights=weights,
                         out_dir=out_dir, name=name)
