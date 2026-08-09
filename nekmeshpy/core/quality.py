"""The shared schema of the per-rung scaled-Jacobian quality summaries."""

from __future__ import annotations

from typing import NamedTuple

#: Scaled Jacobian below which an element is counted as *poor*.  Part of the public
#: schema: it names both the ``n_poor`` field and the ``poor (<...)`` report line.
POOR_THRESHOLD = 0.2


class QualitySummary(NamedTuple):
    """Aggregate scaled-Jacobian statistics over one mesh's elements."""

    #: Number of elements the statistics cover.
    n_elements: int
    #: Smallest per-element scaled Jacobian; ``<= 0`` means inverted / degenerate.
    min: float
    #: Largest per-element scaled Jacobian; ``1`` is a perfect corner.
    max: float
    #: Mean over the elements.
    mean: float
    #: Median over the elements.
    median: float
    #: Elements with a non-positive scaled Jacobian (inverted or degenerate).
    n_inverted: int
    #: Elements below :data:`POOR_THRESHOLD` -- includes ``n_inverted``.
    n_poor: int
