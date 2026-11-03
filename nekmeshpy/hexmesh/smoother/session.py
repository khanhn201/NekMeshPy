"""The interactive layer of the hex smoother: :class:`SmoothSession` is the shared
:class:`~nekmeshpy.core.smoother.session.SmoothSession` bound to the hex rung's adapter
(:class:`~nekmeshpy.hexmesh.smoother.report.HexAdapter`), so a round is run, reported
(:class:`~nekmeshpy.hexmesh.smoother.report.HexRoundReport`), accepted and undone exactly as at the quad rung.

The hex session takes a linear or a **curved** mesh and keeps its order: ``tmop`` stages optimise every node of a
curved mesh, the linear stages run on its straight-sided shadow and the nodes follow the vertices (see
:func:`~nekmeshpy.hexmesh.smoother.smooth.run_schedule`); a round's readings are taken on the stored nodes.  There is
no input gate (a hex mesh is conforming by construction) and no surface normals (a hex has a volume
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
    """One smoothing job on one hex mesh, linear or curved.  The shared session's flow -- :meth:`propose`, :meth:`accept`,
    :meth:`undo`, :meth:`run` -- with the hex options: ``sample_order`` (the order the sampled readings are taken
    at; default the solver's, :data:`~nekmeshpy.core.quality.SCAN_ORDER`) and ``thresholds`` (the scaled-Jacobian
    levels the per-region rows count below).  ``keep_curved=False`` smooths a curved input's straight-sided shadow
    instead of the curved mesh (the result is then linear)."""

    def __init__(self, mesh: HexMesh, constraints: list[cn.Constraint], boundary: str = "frozen",
                 sj_target: float = smooth.EPS, geom_tol: float = 1e-6, weights: FloatArray | None = None,
                 out_dir: str | None = None, name: str = "smooth", *, sample_order: int | None = None,
                 thresholds: Sequence[float] = THRESHOLDS, keep_curved: bool = True) -> None:
        shadow = mesh.order > 1 and not keep_curved
        adapter = HexAdapter(thresholds=thresholds, sample_order=sample_order, original=mesh if shadow else None)
        adapter.curved = mesh.order if mesh.order > 1 and keep_curved else 0
        super().__init__(mesh, constraints, adapter, boundary=boundary, sj_target=sj_target, geom_tol=geom_tol, weights=weights,
                         out_dir=out_dir, name=name, keep_curved=keep_curved)
