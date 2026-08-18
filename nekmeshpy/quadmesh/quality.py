"""Quad-element quality metrics."""

from __future__ import annotations

import numpy as np

from .._typing import FloatArray, IntArray, PointArray
from ..core.quality import POOR_THRESHOLD, QualitySummary
from .quadmesh import QuadMesh

# corner -> [corner, next, prev] neighbour point positions (CCW quad)
_CN = np.array([[0, 1, 3], [1, 2, 0], [2, 3, 1], [3, 0, 2]], dtype=np.int64)


def corner_scaled_jacobian(points: PointArray, quads: IntArray) -> FloatArray:
    """Per-quad minimum corner scaled Jacobian, shape ``(N,)``."""
    X = np.asarray(points, dtype=float)
    QC = np.asarray(quads, dtype=np.int64).reshape(-1, 4)
    N = QC.shape[0]
    cross = np.zeros((N, 4, 3))
    L = np.zeros((N, 4))
    for c in range(4):
        o = X[QC[:, _CN[c, 0]], :]
        e1 = X[QC[:, _CN[c, 1]], :] - o
        e2 = X[QC[:, _CN[c, 2]], :] - o
        cross[:, c, :] = np.cross(e1, e2)
        L[:, c] = np.sqrt(np.sum(e1 ** 2, axis=1)) * np.sqrt(np.sum(e2 ** 2, axis=1))
    nref = np.sum(cross, axis=1)                      # (N,3) mean-direction normal
    nmag = np.sqrt(np.sum(nref ** 2, axis=1))         # (N,)
    good = nmag > 0
    nu = np.divide(nref, nmag[:, None], out=np.zeros_like(nref), where=good[:, None])
    sj = np.ones(N)
    for c in range(4):
        j = np.sum(cross[:, c, :] * nu, axis=1)       # signed corner area / |n|
        ok = L[:, c] > 0
        j = np.where(ok, np.divide(j, L[:, c], out=np.zeros_like(j), where=ok), 0.0)
        sj = np.minimum(sj, j)
    return np.where(good, sj, 0.0)


def _curved_block(mesh: QuadMesh, order: int) -> PointArray:
    """The per-quad ``(Q,(order+1)**2,3)`` node block the order-N metrics sample."""
    from ..core import conform
    nodes, conn_ho = conform.conformal_quad(
        mesh.points, mesh.corners, mesh.quads, mesh.orient, mesh.line_mesh.interior,
        mesh.interior, order)
    block: PointArray = nodes[conn_ho]
    return block


def scaled_jacobian(mesh: QuadMesh, order: int) -> FloatArray:
    """Per-quad minimum scaled Jacobian sampled at the ``(order+1)**2`` GLL nodes of the
    curved element block, shape ``(N,)`` -- the order-N generalization of
    :func:`scaled_jacobian <nekmeshpy.quadmesh.query.scaled_jacobian>`."""
    from ..core.interp import scaled_jacobian as _sj
    return _sj(_curved_block(mesh, order), order, dim=2)


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


def corner_summary(points: PointArray, quads: IntArray) -> QualitySummary:
    """Aggregate quality statistics for a quad mesh."""
    return _summary(corner_scaled_jacobian(points, quads))


def summary(mesh: QuadMesh, order: int) -> QualitySummary:
    """Aggregate statistics for the order-N :func:`scaled_jacobian` metric."""
    return _summary(scaled_jacobian(mesh, order))


def histogram(points: PointArray, quads: IntArray, bins: int = 10,
              lo: float = 0.0, hi: float = 1.0) -> tuple[IntArray, FloatArray]:
    """``(counts, edges)`` histogram of the scaled Jacobian distribution."""
    sj = corner_scaled_jacobian(points, quads)
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
