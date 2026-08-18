"""Hex-element quality metrics."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .._typing import FloatArray, IntArray, PointArray
from ..core import quality as _core
from ..core.quality import POOR_THRESHOLD, OrderScan, QualitySummary
from .hexmesh import HexMesh

# corner -> [corner, +xi, +eta, +zeta] neighbour point positions
_CN = np.array([[0, 1, 3, 4], [1, 2, 0, 5], [2, 3, 1, 6], [3, 0, 2, 7],
                [4, 7, 5, 0], [5, 4, 6, 1], [6, 5, 7, 2], [7, 6, 4, 3]],
               dtype=np.int64)


def corner_scaled_jacobian(points: PointArray, hexes: IntArray) -> FloatArray:
    """Per-hex minimum corner scaled Jacobian, shape ``(N,)``."""
    X = np.asarray(points, dtype=float)
    HC = np.asarray(hexes, dtype=np.int64).reshape(-1, 8)
    N = HC.shape[0]
    sj = np.ones(N)
    for c in range(8):
        o = X[HC[:, _CN[c, 0]], :]
        e1 = X[HC[:, _CN[c, 1]], :] - o
        e2 = X[HC[:, _CN[c, 2]], :] - o
        e3 = X[HC[:, _CN[c, 3]], :] - o
        L = (np.sqrt(np.sum(e1 ** 2, axis=1)) * np.sqrt(np.sum(e2 ** 2, axis=1))
             * np.sqrt(np.sum(e3 ** 2, axis=1)))
        j = np.sum(np.cross(e1, e2) * e3, axis=1)
        ok = L > 0
        j = np.where(ok, np.divide(j, L, out=np.zeros_like(j), where=ok), 0.0)
        sj = np.minimum(sj, j)
    return sj


def _curved_block(mesh: HexMesh) -> PointArray:
    """The per-hex ``(N,(mesh.order+1)**3,3)`` node block, at the mesh's **own**
    order -- the only order its stored nodes describe.

    Reading it on a finer lattice is :func:`sampled_scaled_jacobian
    <nekmeshpy.core.interp.sampled_scaled_jacobian>`'s job, which does it in chunks;
    resampling the whole block here would put the peak memory back."""
    from ..core import conform
    nodes, conn_ho = conform.conformal_hex(
        mesh.points, mesh.corners, mesh._elem_edges, mesh._edge_flip,
        mesh.quad_mesh.line_mesh.interior, mesh.hexes, mesh.orient,
        mesh.quad_mesh.interior, mesh.interior, mesh.order)
    block: PointArray = nodes[conn_ho]
    return block


def scaled_jacobian(mesh: HexMesh, order: int) -> FloatArray:
    """Per-hex minimum scaled Jacobian sampled at the ``(order+1)**3`` GLL nodes of
    the curved element block, shape ``(N,)`` -- the order-N generalization of
    :func:`scaled_jacobian <nekmeshpy.hexmesh.query.scaled_jacobian>`.

    ``order`` may exceed ``mesh.order``: the stored map is read on the finer lattice
    rather than re-meshed, so this answers "what will the solver see at ``lx1 = N``"
    without leaving the toolkit."""
    from ..core.interp import sampled_scaled_jacobian
    return sampled_scaled_jacobian(_curved_block(mesh), mesh.order, order, 3)


def order_scan(mesh: HexMesh, orders: Sequence[int] | None = None, *,
               budget: int | None = None) -> OrderScan:
    """Read the mesh's own map on finer GLL lattices and report what each one sees.

    ``orders`` defaults to :data:`SCAN_ORDER <nekmeshpy.core.quality.SCAN_ORDER>` alone
    -- the order the solver runs, which is the only reading that predicts what the
    solver will do. A sweep of intermediate orders is not worth its cost and does not
    certify anything either: the reading is **not monotone** in the sampling order, and
    a real mesh has read positive at 2, negative at 3 and 4, positive again at 5, and
    negative at 8 and 11. Checking the order you actually run is the whole point.

    ``budget`` defaults to :data:`SCAN_BUDGET <nekmeshpy.core.quality.SCAN_BUDGET>`,
    read *here* rather than captured as a default argument, so assigning the module
    constant at runtime actually takes effect. It caps the sampled points. At
    ``SCAN_ORDER`` that is 512 points a hex, so
    a large mesh runs to minutes; rather than spend it inside a routine report, the
    order is declined and returned in :attr:`OrderScan.skipped
    <nekmeshpy.core.quality.OrderScan.skipped>`. Declined is *unchecked*, not clean.

"""
    from ..core.interp import sampled_scaled_jacobian
    # read through the module, not a ``from`` import: the latter copies the value
    # at import and would ignore an assignment to the constant
    cap = _core.SCAN_BUDGET if budget is None else budget
    p = mesh.order
    want = sorted({int(n) for n in
                   (orders if orders is not None
                    else ((_core.SCAN_ORDER,)
                          if _core.SCAN_ORDER > p else ()))})
    below = [n for n in want if n < p]
    if below:
        raise ValueError("order_scan: cannot sample below the mesh's own order "
                         + str(p) + ", got " + repr(below))
    keep: list[int] = []
    spent = 0
    for n in want:
        cost = mesh.n_hexes * (n + 1) ** 3
        if spent + cost > cap:
            break
        keep.append(n)
        spent += cost
    # one conformal block for the whole sweep: assembling it is the expensive part and
    # it does not depend on the sampling order, so building it per order made the scan
    # cost three block assemblies instead of one
    block = _curved_block(mesh)
    mins: list[float] = []
    invs: list[int] = []
    for n in keep:
        sj = sampled_scaled_jacobian(block, p, n, 3)
        mins.append(float(np.min(sj)))
        invs.append(int(np.sum(sj <= 0)))
    return OrderScan(tuple(keep), tuple(mins), tuple(invs),
                     tuple(n for n in want if n not in keep))


def format_scan(scan: OrderScan, mesh_order: int) -> str:
    """The :func:`order_scan` block of a report, warning line included.

    The warning is the point of the whole thing: a mesh can be clean at its own order
    and folded at the one the solver runs, and the two readings are of the *same*
    geometry -- only the sampling differs."""
    if not scan.orders:
        why = ("order %s declined -- scan budget; call order_scan with a larger one"
               % ", ".join(str(n) for n in scan.skipped)) if scan.skipped else \
              "nothing above order %d asked for" % mesh_order
        return "sampling     : not checked (%s)" % why
    cells = "  ".join("N=%d %+.4f%s" % (n, m, "" if i == 0 else "(%d inv)" % i)
                      for n, m, i in zip(scan.orders, scan.min_sj, scan.n_inverted))
    lines = ["sampling     : " + cells]
    if scan.skipped:
        lines.append("               (order %s not checked -- scan budget)"
                     % ", ".join(str(n) for n in scan.skipped))
    if not scan.clean:
        n, m = scan.worst
        lines.append("  ** WARNING ** inverted at sampling order %d (min %.4f) though "
                     "clean at order %d." % (n, m, mesh_order))
        lines.append("               The element is folded between its own nodes; a "
                     "solver at that order will see it.")
    return "\n".join(lines)


def _summary(sj: FloatArray) -> QualitySummary:
    """Aggregate statistics from a per-element scaled-Jacobian array."""
    return QualitySummary(
        n_elements=int(sj.size),
        min=float(np.min(sj)),
        max=float(np.max(sj)),
        mean=float(np.mean(sj)),
        median=float(np.median(sj)),
        n_inverted=int(np.sum(sj <= 0)),
        n_poor=int(np.sum(sj < POOR_THRESHOLD)),
    )


def corner_summary(points: PointArray, hexes: IntArray) -> QualitySummary:
    """Aggregate quality statistics for a hex mesh."""
    return _summary(corner_scaled_jacobian(points, hexes))


def summary(mesh: HexMesh, order: int) -> QualitySummary:
    """Aggregate statistics for the order-N :func:`scaled_jacobian` metric."""
    return _summary(scaled_jacobian(mesh, order))


def histogram(points: PointArray, hexes: IntArray, bins: int = 10,
              lo: float = 0.0, hi: float = 1.0) -> tuple[IntArray, FloatArray]:
    """``(counts, edges)`` histogram of the scaled Jacobian distribution."""
    sj = corner_scaled_jacobian(points, hexes)
    return np.histogram(sj, bins=bins, range=(lo, hi))


def format_report(stats: QualitySummary,
                  hist: tuple[IntArray, FloatArray] | None = None) -> str:
    """Human-readable multi-line quality report from :func:`summary` output."""
    lines = [
        "elements     : %d" % stats.n_elements,
        "scaled Jac   : min=%.4f  mean=%.4f  median=%.4f  max=%.4f"
        % (stats.min, stats.mean, stats.median, stats.max),
        "inverted(<=0): %d" % stats.n_inverted,
        ("poor (<%g)" % POOR_THRESHOLD).ljust(13) + ": %d" % stats.n_poor,
    ]
    if hist is not None:
        counts, edges = hist
        lines.append("distribution :")
        peak = max(int(counts.max()), 1)
        for i in range(len(counts)):
            bar = "#" * int(40 * counts[i] / peak)
            lines.append("  [%.2f,%.2f) %6d %s"
                         % (edges[i], edges[i + 1], int(counts[i]), bar))
    return "\n".join(lines)
