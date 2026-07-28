"""Hex-element quality metrics, decoupled from :class:`~nekmeshpy.hexmesh.HexMesh`.

All metrics operate on a shared-point representation ``(points, hexes)`` where
``points`` is ``(P,3)`` and ``hexes`` is ``(N,8)`` in Nek corner order, so they
work equally on a welded :class:`~nekmeshpy.hexmesh.HexMesh` or a :class:`~nekmeshpy.model.mesh.Mesh`.

:func:`scaled_jacobian` is the same computation previously inlined in
``HexMesh.scaled_jacobian`` (per-corner min over the eight trilinear corners),
kept numerically identical.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .._typing import FloatArray, IntArray, PointArray

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


def summary(points: PointArray, hexes: IntArray) -> dict[str, Any]:
    """Dict of aggregate quality statistics for a hex mesh."""
    sj = scaled_jacobian(points, hexes)
    return {
        "n_elements": int(sj.size),
        "min": float(np.min(sj)),
        "max": float(np.max(sj)),
        "mean": float(np.mean(sj)),
        "median": float(np.median(sj)),
        "n_inverted": int(np.sum(sj <= 0)),
        "n_below_0.2": int(np.sum(sj < 0.2)),
    }


def histogram(points: PointArray, hexes: IntArray, bins: int = 10,
              lo: float = 0.0, hi: float = 1.0) -> tuple[IntArray, FloatArray]:
    """``(counts, edges)`` histogram of the scaled Jacobian distribution."""
    sj = scaled_jacobian(points, hexes)
    return np.histogram(sj, bins=bins, range=(lo, hi))


def format_report(stats: dict[str, Any],
                  hist: tuple[IntArray, FloatArray] | None = None) -> str:
    """Human-readable multi-line quality report from :func:`summary` output."""
    lines = [
        "elements     : %d" % stats["n_elements"],
        "scaled Jac   : min=%.4f  mean=%.4f  median=%.4f  max=%.4f"
        % (stats["min"], stats["mean"], stats["median"], stats["max"]),
        "inverted(<=0): %d" % stats["n_inverted"],
        "poor (<0.2)  : %d" % stats["n_below_0.2"],
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
