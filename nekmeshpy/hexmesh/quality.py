"""Hex-element quality metrics."""

from __future__ import annotations

import numpy as np

from .._typing import FloatArray, IntArray, PointArray
from ..core.quality import POOR_THRESHOLD, QualitySummary
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


def _curved_block(mesh: HexMesh, order: int) -> PointArray:
    """The per-hex ``(N,(order+1)**3,3)`` node block the order-N metrics sample."""
    from ..core import conform
    nodes, conn_ho = conform.conformal_hex(
        mesh.points, mesh.corners, mesh._elem_edges, mesh._edge_flip,
        mesh.quad_mesh.line_mesh.interior, mesh.hexes, mesh.orient,
        mesh.quad_mesh.interior, mesh.interior, order)
    block: PointArray = nodes[conn_ho]
    return block


def scaled_jacobian(mesh: HexMesh, order: int) -> FloatArray:
    """Per-hex minimum scaled Jacobian sampled at the ``(order+1)**3`` GLL nodes of the
    curved element block, shape ``(N,)`` -- the order-N generalization of
    :func:`scaled_jacobian <nekmeshpy.hexmesh.query.scaled_jacobian>`."""
    from ..core.interp import scaled_jacobian as _sj
    return _sj(_curved_block(mesh, order), order, dim=3)


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
