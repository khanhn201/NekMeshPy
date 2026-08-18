"""The shared schema of the per-rung scaled-Jacobian quality summaries."""

from __future__ import annotations

from typing import NamedTuple

#: Scaled Jacobian below which an element is counted as *poor*.  Part of the public
#: schema: it names both the ``n_poor`` field and the ``poor (<...)`` report line.
POOR_THRESHOLD = 0.2


#: The order a report double-checks the mesh at. This is the polynomial order the
#: *solver* will run it at, which is the only reading that predicts anything, so there
#: is no universal right value -- change it to match your solver.
SCAN_ORDER = 7

#: Sampled points a default ``order_scan`` may spend, summed over its orders.
#: The work is ``n_elements * (order+1)**dim``, and at :data:`SCAN_ORDER` that is 512
#: points a hex -- so a mesh too big for the cap is reported as *not checked* rather
#: than quietly costing minutes inside a routine report. Raise it (or call
#: ``order_scan`` directly) when you want the answer anyway.
SCAN_BUDGET = 15_000_000


class OrderScan(NamedTuple):
    """What the stored map looks like read on lattices finer than the mesh's own.

    An element's Jacobian is *sampled*, exact at its ``(order+1)**dim`` GLL nodes and
    silent between them, so a clean :class:`QualitySummary` at the mesh's own order is
    not a certificate. This is the same map at more points -- what a solver running at
    its own polynomial order computes."""

    #: Sampling orders actually read, ascending.
    orders: tuple[int, ...]
    #: Smallest scaled Jacobian at each of :attr:`orders`.
    min_sj: tuple[float, ...]
    #: Elements with a non-positive scaled Jacobian at each of :attr:`orders`.
    n_inverted: tuple[int, ...]
    #: Orders the budget refused, ascending -- unchecked, not clean.
    skipped: tuple[int, ...]

    @property
    def clean(self) -> bool:
        """True when at least one order was read and none found an inverted element.

        A scan that read *nothing* is not clean -- it is unchecked, which is the same
        reason :attr:`skipped` exists rather than the budget dropping orders in
        silence."""
        return bool(self.orders) and not any(self.n_inverted)

    @property
    def worst(self) -> tuple[int, float]:
        """``(order, min_sj)`` of the harshest reading."""
        if not self.orders:
            raise ValueError("OrderScan.worst: no order was sampled")
        i = min(range(len(self.orders)), key=lambda k: self.min_sj[k])
        return self.orders[i], self.min_sj[i]


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
