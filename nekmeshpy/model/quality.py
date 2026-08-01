"""The shared schema of the per-rung scaled-Jacobian quality summaries.

``quadmesh.quality`` and ``hexmesh.quality`` measure the same thing one rung apart,
so *what a summary is* -- its fields, and the threshold that decides which elements
count as poor -- is a single thing.  It lives here, container-free, and both rungs
import it.

That is the point of ``POOR_THRESHOLD``: the counted field and the ``poor (<...)``
line of the formatted report are both **derived** from it, so the number cannot drift
out of step with the text describing it.  As a ``"n_below_0.2"`` dict key it was
baked into the public schema four times over, and it was not even a valid Python
identifier -- which is why the summary could not be a ``NamedTuple`` until the
threshold became a constant.

Only the schema is shared.  The two ``quality`` modules still own their own
``_summary`` / ``format_report`` (byte-identical today; deduplicating them is a
separate change).
"""

from __future__ import annotations

from typing import NamedTuple

#: Scaled Jacobian below which an element is counted as *poor*.  Part of the public
#: schema: it names both the ``n_poor`` field and the ``poor (<...)`` report line.
POOR_THRESHOLD = 0.2


class QualitySummary(NamedTuple):
    """Aggregate scaled-Jacobian statistics over one mesh's elements.

    Produced by ``quadmesh.quality.summary`` / ``hexmesh.quality.summary`` (and
    their order-N ``summary_ho`` twins), and reached from a container as
    ``QuadMesh.quality_summary()`` / ``HexMesh.quality_summary()``.  Attribute
    access only -- it replaced a ``dict[str, Any]`` that type-checked nothing.
    """

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
