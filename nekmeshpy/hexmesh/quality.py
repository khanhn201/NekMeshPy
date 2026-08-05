"""Hex-element quality metrics.

All metrics operate on a shared-point representation ``(points, hexes)`` where
``points`` is ``(P,3)`` and ``hexes`` is ``(N,8)`` in Nek corner order.
"""

from __future__ import annotations

import numpy as np

from .._typing import FloatArray, IntArray, PointArray
from ..model.quality import POOR_THRESHOLD, QualitySummary
from .hexmesh import HexMesh

# corner -> [corner, +xi, +eta, +zeta] neighbour point positions
_CN = np.array([[0, 1, 3, 4], [1, 2, 0, 5], [2, 3, 1, 6], [3, 0, 2, 7],
                [4, 7, 5, 0], [5, 4, 6, 1], [6, 5, 7, 2], [7, 6, 4, 3]],
               dtype=np.int64)


def scaled_jacobian(points: PointArray, hexes: IntArray) -> FloatArray:
    """Per-hex minimum corner scaled Jacobian, shape ``(N,)``.

    1 is a perfect cube corner; <= 0 is degenerate / inverted.
    """
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


def _ho_block(mesh: HexMesh, order: int) -> PointArray:
    """The per-hex ``(N,(order+1)**3,3)`` node block the order-N metrics sample.

    The mesh's entity B-rep is walked
    (:func:`~nekmeshpy.model.conform.conformal_hex`) and the block gathered
    **transiently** as ``nodes[conn_ho]`` -- nothing is stored.
    """
    from ..model import conform
    nodes, conn_ho = conform.conformal_hex(
        mesh.points, mesh.hexes, mesh._elem_edges, mesh._edge_flip,
        mesh.quads.lines.interior, mesh.hex, mesh.face_orient,
        mesh.quads.interior, mesh.interior, order)
    block: PointArray = nodes[conn_ho]
    return block


def scaled_jacobian_ho(mesh: HexMesh, order: int) -> FloatArray:
    """Per-hex minimum scaled Jacobian sampled at the ``(order+1)**3`` GLL nodes of the
    curved element block, shape ``(N,)`` -- the order-N generalization of
    :func:`scaled_jacobian <nekmeshpy.hexmesh.query.scaled_jacobian>`.

    ``mesh`` is a ``HexMesh``; its high-order nodes are gathered from the entity B-rep
    on the fly.

    Each node's ``det(J) / prod(|tangent|)`` is formed from the mapping's parametric
    tangents there.  This is the **opt-in** metric: the default corner-based
    :func:`scaled_jacobian <nekmeshpy.hexmesh.query.scaled_jacobian>` keeps the pinned linear numbers, and at ``order == 1`` (GLL
    nodes == corners) this reduces to it.
    """
    from ..model.interp import scaled_jacobian_ho as _sj
    return _sj(_ho_block(mesh, order), order, dim=3)


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


def summary(points: PointArray, hexes: IntArray) -> QualitySummary:
    """Aggregate quality statistics for a hex mesh."""
    return _summary(scaled_jacobian(points, hexes))


def summary_ho(mesh: HexMesh, order: int) -> QualitySummary:
    """Aggregate statistics for the order-N :func:`scaled_jacobian_ho` metric."""
    return _summary(scaled_jacobian_ho(mesh, order))


def histogram(points: PointArray, hexes: IntArray, bins: int = 10,
              lo: float = 0.0, hi: float = 1.0) -> tuple[IntArray, FloatArray]:
    """``(counts, edges)`` histogram of the scaled Jacobian distribution."""
    sj = scaled_jacobian(points, hexes)
    return np.histogram(sj, bins=bins, range=(lo, hi))


def format_report(stats: QualitySummary,
                  hist: tuple[IntArray, FloatArray] | None = None) -> str:
    """Human-readable multi-line quality report from :func:`summary` output.

    The ``poor`` label is **derived** from ``POOR_THRESHOLD`` rather than spelling
    the number again, so the text can never disagree with the count it labels; the
    label is padded to the 13-column field the other rows use.
    """
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
