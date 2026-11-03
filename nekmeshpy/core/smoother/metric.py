"""The smoother's shape metric, its objective and analytic gradient, for any element dimension.

Every reading is taken off the corner Jacobian ``J = [e_1 ... e_d]`` of an element corner -- the ``d`` edge
vectors leaving it, right-handed -- so one function serves quads (``d = 2``) and hexes (``d = 3``) and the rung
only builds the edge vectors from its connectivity and scatters the gradient back onto its vertices.

Signed scaled Jacobian (``s = 1`` only for ``d`` orthogonal edges, ``s <= 0`` inverted or degenerate)::

    s = det J / (|e_1| ... |e_d|)

Shape metric (Knupp's mean-ratio measure to the power ``d``; ``phi >= 1``, ``1`` only for a square or cube corner)::

    phi = (|J|_F^2 / d)^d / det J^2 = K / s^2,        K = ((sum |e_i|^2) / d)^d / prod |e_i|^2

``phi`` is blind to orientation, so inversion is a separate signed penalty ``kappa max(0, eps - s)``.  The
barrier-free ``phi_delta = K c / s_delta^2`` (``s_delta = (s + sqrt(s^2 + 4 delta^2)) / 2``, ``c = s_delta(1)^2``)
removes the wall at a zero Jacobian that a line search cannot cross; ``delta = 0`` is the exact metric.  The
objective is the **sum** over elements of the corner mean (``1 / 2^d``) of ``w (phi - 1) + kappa max(0, eps - s)``;
a vertex's gradient involves only its incident elements, so it does not depend on the element count.

A quad is a surface in 3-D with 2-D topology: ``det J`` is read through a reference normal ``n`` per element,
``det J = (e_1 x e_2) . n`` (``normal=``), so a mesh on a curved surface is never assumed planar.  For ``d = 2``
the exact metric is evaluated as ``(A + B)^2 / (4 |e_1 x e_2|^2)`` and for ``d = 3`` as ``K / s^2``; the two
are the same function, written the way each is best conditioned, and the tests check them against the
single determinant formula above.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import NamedTuple

import numpy as np

from ..._typing import FloatArray, IntArray, PointArray

__all__ = ["KAPPA", "EPS", "DELTA", "CornerMetrics", "regularised", "scatter", "corner_metrics", "corner_gradient",
           "objective_value", "evaluate", "assemble_gradient"]

#: Inversion-penalty weight per corner, on the *sum* objective (see module docstring).
KAPPA = 1000.0
#: Scaled-Jacobian threshold below which a corner is penalised.
EPS = 0.05
#: Barrier-removal width (in units of the signed sine ``s``) the optimizer uses to untangle:
#: ``phi`` blows up as a corner's Jacobian goes through zero, so a plain ``phi`` is a *wall*
#: between an inverted corner and a valid one that no line search crosses.  Knupp's remedy
#: replaces ``s`` by ``s_d = (s + sqrt(s^2 + 4 d^2)) / 2``, positive and finite for every ``s``
#: and decreasing smoothly through zero.  ``delta = 0`` is the exact metric.
DELTA = 0.01
#: Floor on ``|e_1 x e_2|^2 / (|e_1|^2 |e_2|^2)`` (the squared sine of a quad corner angle) that keeps a
#: collapsed corner finite instead of ``inf``/``nan``; far below any reading that matters.
_SIN2_FLOOR = 1e-14
#: The same floor on ``s^2`` at ``d = 3``.
_S2_FLOOR = 1e-14


def regularised(s: FloatArray, delta: float) -> tuple[FloatArray, float, FloatArray]:
    """``(s_d, c, ds_d/ds)``: the barrier-free sine, the constant ``s_d(1)^2`` that makes a
    perfect corner score exactly 1, and the derivative of ``s_d`` with respect to ``s``."""
    root = np.sqrt(s * s + 4.0 * delta * delta)
    one = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * delta * delta))
    return 0.5 * (s + root), float(one * one), 0.5 * (1.0 + s / root)


class CornerMetrics(NamedTuple):
    """Per-corner readings, each ``(n_elements, corners_per_element)`` in the element's own corner order."""

    #: ``phi >= 1``, ``1`` at a square / cube corner; blind to orientation.
    phi: FloatArray
    #: Signed scaled Jacobian in ``[-1, 1]``; ``<= 0`` is inverted or degenerate.
    sj: FloatArray


def scatter(n: int, index: IntArray, values: PointArray) -> PointArray:
    """Sum ``values (n_elements, k, 3)`` onto ``n`` vertices by ``index (n_elements, k)`` -- the vertex
    assembly a shared-id mesh needs no gather-scatter library for."""
    flat = index.reshape(-1)
    return np.stack([np.bincount(flat, weights=values[..., k].reshape(-1), minlength=n)
                     for k in range(3)], axis=1)


# -- d = 2: a quad on a surface, det J through the reference normal --------------------------------
def _terms2(a: PointArray, b: PointArray, normals: PointArray) -> tuple[
        PointArray, FloatArray, FloatArray, FloatArray, FloatArray, FloatArray]:
    """The shared per-corner pieces: ``cross, A, B, C, g, sqrt(AB)``."""
    c = np.cross(a, b)
    A = np.einsum("qkj,qkj->qk", a, a)
    B = np.einsum("qkj,qkj->qk", b, b)
    C = np.einsum("qkj,qkj->qk", c, c)
    C = np.maximum(C, _SIN2_FLOOR * A * B)
    g = np.einsum("qkj,qj->qk", c, normals)
    return c, A, B, C, g, np.sqrt(A * B)


def _metrics2(a: PointArray, b: PointArray, normals: PointArray, delta: float) -> CornerMetrics:
    _, A, B, C, g, ab = _terms2(a, b, normals)
    # a corner with a zero-length edge (a collapsed edge, a quad naming a vertex twice) has no
    # direction: its signed Jacobian is 0 -- counted as inverted, never NaN, so the gates see it --
    # and its exact phi is infinite
    sj = np.where(ab > 0.0, g / np.where(ab > 0.0, ab, 1.0), 0.0)
    if delta > 0.0:
        sd, c, _ = regularised(sj, delta)
        K = (A + B) ** 2 / (4.0 * np.maximum(A * B, 1e-300))
        return CornerMetrics(phi=K * c / sd ** 2, sj=sj)
    with np.errstate(divide="ignore", invalid="ignore"):
        phi = (A + B) ** 2 / (4.0 * C)
    return CornerMetrics(phi=np.where(np.isnan(phi), np.inf, phi), sj=sj)


def _gradient2(a: PointArray, b: PointArray, normals: PointArray, *, kappa: float, eps: float, delta: float,
               weights: FloatArray | None) -> list[PointArray]:
    """Per corner, with ``A=|a|^2, B=|b|^2, d=a.b, C=|a x b|^2 = AB - d^2, g=(a x b).n``::

        dphi/da = (A+B) a / C - (A+B)^2 (B a - d b) / (2 C^2)      (b: swap a<->b, A<->B)
        ds/da   = (b x n) / sqrt(AB) - g a / (A sqrt(AB))
        ds/db   = (n x a) / sqrt(AB) - g b / (B sqrt(AB))

    The penalty branch is on only where ``s < eps``.  With ``delta > 0`` ``phi_d = K c / s_d^2`` and::

        dphi_d/da = c [ dK/da / s_d^2 - 2 K s_d' ds/da / s_d^3 ],   dK/da = K (2/(A+B) - 1/A) 2a
    """
    _, A, B, C, g, ab = _terms2(a, b, normals)
    d = np.einsum("qkj,qkj->qk", a, b)
    n = normals[:, None, :]
    ab3 = ab[..., None]
    ds_a = np.cross(b, n) / ab3 - g[..., None] * a / (A[..., None] * ab3)
    ds_b = np.cross(n, a) / ab3 - g[..., None] * b / (B[..., None] * ab3)
    sv = g / ab
    if delta > 0.0:
        sd, c, dsd = regularised(sv, delta)
        AB = np.maximum(A * B, 1e-300)
        K = (A + B) ** 2 / (4.0 * AB)
        S2 = (sd ** 2)[..., None]
        S3 = (sd ** 3)[..., None]
        Kc = K[..., None]
        dphi_a = c * (Kc * (2.0 / (A + B) - 1.0 / A)[..., None] * 2.0 * a / S2
                      - 2.0 * Kc * dsd[..., None] * ds_a / S3)
        dphi_b = c * (Kc * (2.0 / (A + B) - 1.0 / B)[..., None] * 2.0 * b / S2
                      - 2.0 * Kc * dsd[..., None] * ds_b / S3)
    else:
        S = (A + B)[..., None]
        Cc = C[..., None]
        dphi_a = S * a / Cc - S ** 2 * (B[..., None] * a - d[..., None] * b) / (2.0 * Cc ** 2)
        dphi_b = S * b / Cc - S ** 2 * (A[..., None] * b - d[..., None] * a) / (2.0 * Cc ** 2)
    on = (sv < eps)[..., None]
    w = 1.0 if weights is None else np.asarray(weights, float)[:, None, None]
    ga = 0.25 * (w * dphi_a - kappa * on * ds_a)
    gb = 0.25 * (w * dphi_b - kappa * on * ds_b)
    return [ga, gb]


# -- d = 3: a hex ----------------------------------------------------------------------------------
def _terms3(a: PointArray, b: PointArray, c: PointArray) -> tuple[
        FloatArray, FloatArray, FloatArray, FloatArray, FloatArray]:
    """The shared per-corner pieces: ``A, B, C, L = |a||b||c|, s``."""
    A = np.einsum("ekj,ekj->ek", a, a)
    B = np.einsum("ekj,ekj->ek", b, b)
    C = np.einsum("ekj,ekj->ek", c, c)
    L = np.sqrt(A * B * C)
    g = np.einsum("ekj,ekj->ek", np.cross(a, b), c)
    # a corner with a zero-length edge has no direction: its signed Jacobian is 0 -- counted as
    # inverted, never NaN, so the gates see it
    s = np.where(L > 0.0, g / np.where(L > 0.0, L, 1.0), 0.0)
    return A, B, C, L, s


def _shape_factor3(A: FloatArray, B: FloatArray, C: FloatArray) -> FloatArray:
    """``K = ((A + B + C) / 3)^3 / (A B C) >= 1``; ``inf`` on a zero-length edge."""
    with np.errstate(divide="ignore", invalid="ignore"):
        K = ((A + B + C) / 3.0) ** 3 / (A * B * C)
    return np.asarray(np.where(np.isfinite(K), K, np.inf))


def _sd3(s: FloatArray, delta: float) -> tuple[FloatArray, float, FloatArray]:
    """``(s_d, c, ds_d/ds)``: the regularised sine for ``delta > 0``, else ``s`` itself with ``|s|`` floored at
    ``sqrt(_S2_FLOOR)`` (derivative ``0`` there)."""
    if delta > 0.0:
        return regularised(s, delta)
    small = s * s < _S2_FLOOR
    return (np.where(small, np.sqrt(_S2_FLOOR), s), 1.0,
            np.where(small, 0.0, 1.0))


def _metrics3(a: PointArray, b: PointArray, c: PointArray, delta: float) -> CornerMetrics:
    A, B, C, _, s = _terms3(a, b, c)
    K = _shape_factor3(A, B, C)
    sd, cst, _ = _sd3(s, delta)
    with np.errstate(invalid="ignore"):
        phi = K * cst / sd ** 2
    return CornerMetrics(phi=np.where(np.isnan(phi), np.inf, phi), sj=s)


def _gradient3(a: PointArray, b: PointArray, c: PointArray, *, kappa: float, eps: float, delta: float,
               weights: FloatArray | None) -> list[PointArray]:
    """Per corner, with ``S = A + B + C`` and ``L = |a||b||c|``::

        ds/da = (b x c) / L - s a / A        (b: c x a, B;  c: a x b, C)
        dK/da = K (6 / S - 2 / A) a
        dphi/da = c [ dK/da / s_d^2 - 2 K s_d' ds/da / s_d^3 ]

    (``delta = 0``: ``s_d = s``, ``c = s_d' = 1``).  The penalty branch is on only where ``s < eps``."""
    A, B, C, L, s = _terms3(a, b, c)
    K = _shape_factor3(A, B, C)
    S = A + B + C
    Lx = np.where(L > 0.0, L, 1.0)[..., None]
    ds = [np.cross(b, c) / Lx - (s / np.where(A > 0, A, 1.0))[..., None] * a,
          np.cross(c, a) / Lx - (s / np.where(B > 0, B, 1.0))[..., None] * b,
          np.cross(a, b) / Lx - (s / np.where(C > 0, C, 1.0))[..., None] * c]
    sd, cst, dsd = _sd3(s, delta)
    S2 = (sd ** 2)[..., None]
    S3 = (sd ** 3)[..., None]
    Kc = K[..., None]
    on = (s < eps)[..., None]
    w = 1.0 if weights is None else np.asarray(weights, float)[:, None, None]
    out = []
    for j, (v, X) in enumerate(((a, A), (b, B), (c, C))):
        dK = Kc * (6.0 / S - 2.0 / np.where(X > 0, X, 1.0))[..., None] * v
        dphi = cst * (dK / S2 - 2.0 * Kc * dsd[..., None] * ds[j] / S3)
        out.append(0.125 * (w * dphi - kappa * on * ds[j]))
    return out


# -- the dimension-generic entry points ------------------------------------------------------------
def corner_metrics(edges: Sequence[PointArray], normals: PointArray | None = None, *,
                   delta: float = 0.0) -> CornerMetrics:
    """``phi`` and the signed scaled Jacobian at every corner.  ``edges`` is the ``d`` edge vectors leaving each
    corner, each ``(n_elements, corners, 3)``; ``normals`` ``(n_elements, 3)`` is required for ``d = 2``.
    ``delta > 0`` returns the barrier-free ``phi_d`` the optimizer descends on; the default is the exact metric
    the report prints."""
    if len(edges) == 2:
        if normals is None:
            raise ValueError("corner_metrics: a d = 2 element needs its reference normals")
        return _metrics2(edges[0], edges[1], normals, delta)
    if len(edges) == 3:
        return _metrics3(edges[0], edges[1], edges[2], delta)
    raise ValueError("corner_metrics: %d edge vectors; expected 2 (quad) or 3 (hex)" % len(edges))


def objective_value(m: CornerMetrics, dim: int, *, kappa: float = KAPPA, eps: float = EPS,
                    weights: FloatArray | None = None) -> float:
    """``F = sum_e (1/2^d) sum_c [ w_e (phi - 1) + kappa max(0, eps - s) ]`` for the readings ``m`` --
    ``0`` only for an all-square (cube), all-positive mesh.  ``weights`` ``(n_elements,)`` scale the *shape* term
    per element (default 1): ``0`` makes an element's shape irrelevant while the inversion penalty still
    protects it.  Divide by the element count for the mean the report shows."""
    w = 1.0 if weights is None else np.asarray(weights, float)[:, None]
    return float(0.5 ** dim * (w * (m.phi - 1.0) + kappa * np.maximum(0.0, eps - m.sj)).sum())


def evaluate(m: CornerMetrics, dim: int, *, kappa: float = KAPPA, eps: float = EPS,
             weights: FloatArray | None = None) -> tuple[float, int]:
    """``(F, n_inverted)`` from one set of corner readings: the objective and the number of elements with a
    corner whose signed scaled Jacobian is ``<= 0``."""
    return (objective_value(m, dim, kappa=kappa, eps=eps, weights=weights),
            int((m.sj.min(axis=1) <= 0).sum()))


def corner_gradient(edges: Sequence[PointArray], normals: PointArray | None = None, *, kappa: float = KAPPA,
                    eps: float = EPS, delta: float = 0.0,
                    weights: FloatArray | None = None) -> list[PointArray]:
    """``dF/d e_j`` for every edge vector, each ``(n_elements, corners, 3)`` (reference normals held fixed): the
    gradient with respect to the ``d`` edge vectors leaving each corner.  The corner's own vertex receives
    minus their sum and each edge's far end its own term -- :func:`assemble_gradient` does that with the rung's
    connectivity."""
    if len(edges) == 2:
        if normals is None:
            raise ValueError("corner_gradient: a d = 2 element needs its reference normals")
        return _gradient2(edges[0], edges[1], normals, kappa=kappa, eps=eps, delta=delta, weights=weights)
    if len(edges) == 3:
        return _gradient3(edges[0], edges[1], edges[2], kappa=kappa, eps=eps, delta=delta, weights=weights)
    raise ValueError("corner_gradient: %d edge vectors; expected 2 (quad) or 3 (hex)" % len(edges))


def assemble_gradient(n_points: int, corner_index: IntArray, far_index: Sequence[IntArray],
                      grads: Sequence[PointArray]) -> PointArray:
    """The vertex gradient ``(n_points, 3)`` from per-corner edge gradients: ``corner_index`` ``(n_elements, k)``
    the vertex at each corner, ``far_index[j]`` the vertex at the far end of edge ``j`` of that corner, ``grads[j]``
    from :func:`corner_gradient`.  The far end receives ``grads[j]``, the corner itself ``-sum_j grads[j]``."""
    out = np.zeros((n_points, 3))
    total = np.zeros_like(grads[0])
    for idx, gj in zip(far_index, grads):
        total += gj
        out += scatter(n_points, idx, gj)
    return np.asarray(out - scatter(n_points, corner_index, total))
