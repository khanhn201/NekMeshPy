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


def _linear_block(mesh: HexMesh) -> PointArray:
    """The mesh's 8 corners alone, in the **tensor-lattice** order
    :func:`core.interp.sampled_scaled_jacobian <nekmeshpy.core.interp.sampled_scaled_jacobian>`
    expects -- not ``mesh.corners``' own Nek-hex order, which is a different
    permutation. This is the geometry ``.re2`` actually exports at any stored order:
    the trilinear hex through the 8 corners, nothing else."""
    from ..core.interp import corner_indices
    ci = corner_indices(1, 3)
    block: PointArray = np.empty((mesh.n_hexes, 8, 3))
    block[:, ci, :] = mesh.points[mesh.corners]
    return block


def linear_scaled_jacobian(mesh: HexMesh, order: int) -> FloatArray:
    """Per-hex minimum scaled Jacobian of the **trilinear** hex through the 8 corners
    alone, resampled at ``order`` -- what ``.re2`` exports, read the way a solver
    actually builds and checks its own working geometry: from the corners, at its
    own polynomial order, not at nekmeshpy's stored (possibly curved) one.

    ``order=1`` is exactly :func:`corner_scaled_jacobian` -- the 8 vertices alone.
    That is *not* sufficient in general: a trilinear hex's Jacobian is a polynomial
    in each direction, so it can fold **between** the 8 corners while every corner
    itself reads positive, the same lesson :func:`scaled_jacobian`'s own sampling
    split teaches about the curved map -- except here there is no curvature to
    blame, because ``.re2`` never carried any. Confirmed against a real Nek5000 run:
    a mesh clean at every corner still failed the solver's own Neg-Jacobian check at
    its working order, and reads exactly the same negative minimum here once
    resampled at that order."""
    from ..core.interp import sampled_scaled_jacobian
    return sampled_scaled_jacobian(_linear_block(mesh), 1, order, 3)


def linear_order_scan(mesh: HexMesh, orders: Sequence[int] | None = None, *,
                      budget: int | None = None) -> OrderScan:
    """:func:`linear_scaled_jacobian` read at several orders, :func:`order_scan`'s
    report shape and budget -- but of the **trilinear** (corners-only) map, not the
    curved one. Unlike ``order_scan`` there is no "mesh's own order" floor: the
    trilinear map is not stored at any particular order, so any ``order >= 1`` is a
    legitimate reading of it, ``order=1`` included.

    ``orders`` defaults to :data:`SCAN_ORDER <nekmeshpy.core.quality.SCAN_ORDER>`
    alone, the same solver-order reasoning as ``order_scan``: it is the one reading
    that predicts what a real run's own geometry generation will find, because it is
    built from the same corners the solver reads out of ``.re2``."""
    from ..core.interp import sampled_scaled_jacobian
    cap = _core.SCAN_BUDGET if budget is None else budget
    want = sorted({int(n) for n in
                   (orders if orders is not None else (_core.SCAN_ORDER,))})
    below = [n for n in want if n < 1]
    if below:
        raise ValueError("linear_order_scan: order must be >= 1, got " + repr(below))
    keep: list[int] = []
    spent = 0
    for n in want:
        cost = mesh.n_hexes * (n + 1) ** 3
        if spent + cost > cap:
            break
        keep.append(n)
        spent += cost
    block = _linear_block(mesh)
    mins: list[float] = []
    invs: list[int] = []
    for n in keep:
        sj = sampled_scaled_jacobian(block, 1, n, 3)
        mins.append(float(np.min(sj)))
        invs.append(int(np.sum(sj <= 0)))
    return OrderScan(tuple(keep), tuple(mins), tuple(invs),
                     tuple(n for n in want if n not in keep))


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


def format_linear_scan(scan: OrderScan) -> str:
    """The :func:`linear_order_scan` block of a report, warning line included.

    Unlike :func:`format_scan`, there is no "mesh's own order" to compare against --
    the trilinear map is not stored at any order, so a disagreement here is not
    "clean at N, folded at M" but simply "folds once resampled at all", which is
    exactly what a real solver's own geometry generation would find building its
    working GLL nodes from the same corners."""
    if not scan.orders:
        why = ("order %s declined -- scan budget; call linear_order_scan with a "
               "larger one" % ", ".join(str(n) for n in scan.skipped)) \
              if scan.skipped else "nothing asked for"
        return "linear sampling: not checked (%s)" % why
    cells = "  ".join("N=%d %+.4f%s" % (n, m, "" if i == 0 else "(%d inv)" % i)
                      for n, m, i in zip(scan.orders, scan.min_sj, scan.n_inverted))
    lines = ["linear sampling: " + cells]
    if scan.skipped:
        lines.append("               (order %s not checked -- scan budget)"
                     % ", ".join(str(n) for n in scan.skipped))
    if not scan.clean:
        n, m = scan.worst
        lines.append("  ** WARNING ** %d element(s) fold once the .re2 trilinear "
                     "geometry is resampled at order %d (min %.4f) -- this is what "
                     "a real solver's own geometry generation builds and checks at "
                     "its working order." % (scan.n_inverted[scan.orders.index(n)],
                                             n, m))
    return "\n".join(lines)


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


def format_linear(stats: QualitySummary, mesh_order: int) -> str:
    """The linear (``.re2``) reading of a report, warning line included.

    ``.re2`` has no curved format -- every mesh, at any stored order, exports as the
    straight-sided hex through its 8 corners alone (``corner_scaled_jacobian``), which
    is a genuinely *different* map from the curved one the mesh stores above order 1,
    not a coarser sampling of the same one.  An element can be clean at its own order
    only *because* of the curvature its interior nodes carry -- valid as a mesh,
    inverted as the file the solver actually reads."""
    lines = ["linear (.re2): min=%+.4f  (%d inverted of %d)"
            % (stats.min, stats.n_inverted, stats.n_elements)]
    if stats.n_inverted and mesh_order > 1:
        lines.append("  ** WARNING ** %d element(s) invert once flattened to .re2's "
                     "linear corners, though clean at order %d."
                     % (stats.n_inverted, mesh_order))
        lines.append("               .re2 has no curved format -- this is the "
                     "geometry the solver actually reads.")
    return "\n".join(lines)


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
