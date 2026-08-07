"""Moving orthonormal frames along a sampled curve -- the placement half of a sweep."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import numpy as np

from .._typing import FloatArray, Point, PointArray, Vec3

#: Segments shorter than this fraction of the curve's overall extent are treated as
#: repeated points.  Relative, so the check is scale-free: a curve modelled in metres
#: and the same curve in millimetres are rejected (or accepted) identically.
SEGMENT_TOL: float = 1e-12

#: A tangent whose component along ``up`` exceeds this in magnitude leaves too little
#: of ``up`` in the cross-section plane for :func:`fixed_up` to normalize meaningfully;
#: at ``1 - 1e-12`` the surviving perpendicular component is ~1.4e-6 of ``up``, already
#: at the edge of what double precision resolves in a direction.
PARALLEL_TOL: float = 1e-12

#: A profile point this far off its own best-fit plane, **relative to the
#: profile's overall extent**, is not a planar cross-section.  Relative, so a
#: section modelled in metres and the same section in millimetres are judged
#: identically; loose enough to pass a section assembled from several factories,
#: tight enough that a genuinely bowed profile is caught rather than sheared.
PLANAR_TOL: float = 1e-9


#: Below this the unit-tangent turning per sample is indistinguishable from round-off
#: and :func:`frenet` has no principal normal to speak of.  Absolute, because the
#: quantity it bounds -- ``|t[i+1] - t[i-1]| / 2`` -- is already dimensionless.
CURVATURE_TOL: float = 1e-10


def _as_points(points: PointArray, who: str) -> PointArray:
    """Validate a ``(K,3)`` sampled curve, ``K >= 2``."""
    P: PointArray = np.asarray(points, dtype=float)
    if P.ndim != 2 or P.shape[1] != 3:
        raise ValueError("%s: points must be a (K,3) array of 3-D coordinates, got %s; "
                         "curves live honestly in 3-D, a (K,2) array is not padded"
                         % (who, (P.shape,)))
    if P.shape[0] < 2:
        raise ValueError("%s: need at least 2 points to define a tangent, got %d"
                         % (who, P.shape[0]))
    return P


def _as_frames_tangents(tangents: PointArray, k: int, who: str) -> PointArray:
    """Validate a ``(K,3)`` tangent field against a known point count and unit-check it."""
    T: PointArray = np.asarray(tangents, dtype=float)
    if T.shape != (k, 3):
        raise ValueError("%s: tangents must be (K,3) matching the %d points, got %s"
                         % (who, k, (T.shape,)))
    n: FloatArray = np.linalg.norm(T, axis=1)
    bad = int(np.argmax(np.abs(n - 1.0)))
    if abs(n[bad] - 1.0) > 1e-9:
        raise ValueError("%s: tangents must be unit vectors; |tangents[%d]| = %.17g. "
                         "Build them with frames.tangents(points)." % (who, bad, n[bad]))
    return T


def _unit_literal(vector: Vec3 | Sequence[float], who: str, name: str) -> Vec3:
    """Validate and normalize a ``(3,)`` direction literal."""
    v: Vec3 = np.asarray(vector, dtype=float).reshape(-1)
    if v.shape != (3,):
        raise ValueError("%s: %s must be a (3,) direction, got %s"
                         % (who, name, (np.shape(vector),)))
    n = float(np.linalg.norm(v))
    if n == 0.0:
        raise ValueError("%s: %s must be a non-zero direction" % (who, name))
    return v / n


def _assemble(u: PointArray, w: PointArray) -> FloatArray:
    """Stack ``(K,3)`` cross-section ``u`` and tangent ``w`` into ``(K,3,3)`` frames with
    ``(u, v, w)`` as **columns** and ``v = w x u``, hence ``det == +1``."""
    v: PointArray = np.cross(w, u)
    R: FloatArray = np.stack([u, v, w], axis=-1)
    return R


def _end_derivative(x0: Point, x1: Point, x2: Point, a: float, b: float) -> Vec3:
    """Derivative at ``x0`` of the quadratic through ``x0, x1, x2`` sampled at
    cumulative-chord parameters ``0, a, a+b``."""
    d: Vec3 = (-(2.0 * a + b) / (a * (a + b)) * x0
               + (a + b) / (a * b) * x1
               - a / (b * (a + b)) * x2)
    return d


def tangents(points: PointArray, *, loop: bool = False) -> PointArray:
    """Unit tangents ``(K,3)`` of the ``(K,3)`` sampled curve ``points``."""
    P = _as_points(points, "tangents")
    k = P.shape[0]

    seg: PointArray = np.diff(P, axis=0)
    if loop:
        seg = np.vstack([seg, (P[0] - P[-1])[None, :]])
    extent = float(np.max(np.linalg.norm(P - P.mean(axis=0), axis=1)))
    tol = SEGMENT_TOL * max(extent, 1.0)
    seg_len: FloatArray = np.linalg.norm(seg, axis=1)
    if np.any(seg_len <= tol):
        i = int(np.argmin(seg_len))
        raise ValueError(
            "tangents: points[%d] and points[%d] coincide (segment length %.17g <= %.17g); "
            "a repeated sample has no tangent. Drop the duplicate before sampling."
            % (i, (i + 1) % k, seg_len[i], tol))
    if loop and k < 3:
        raise ValueError("tangents: loop=True needs at least 3 distinct points, got %d" % k)

    D: PointArray = np.empty_like(P)
    if loop:
        D[:] = np.roll(P, -1, axis=0) - np.roll(P, 1, axis=0)
    elif k == 2:
        D[:] = (P[1] - P[0])[None, :]
    else:
        D[1:-1] = P[2:] - P[:-2]
        D[0] = _end_derivative(P[0], P[1], P[2], seg_len[0], seg_len[1])
        D[-1] = -_end_derivative(P[-1], P[-2], P[-3], seg_len[-1], seg_len[-2])

    n: FloatArray = np.linalg.norm(D, axis=1)
    if np.any(n <= tol):
        i = int(np.argmin(n))
        raise ValueError(
            "tangents: the central difference at points[%d] vanishes (|x[i+1] - x[i-1]| = "
            "%.17g <= %.17g) -- the curve reverses exactly onto itself there and has no "
            "single tangent. Split the curve at the cusp." % (i, n[i], tol))
    T: PointArray = D / n[:, None]
    return T


def fixed_up(tangents: PointArray, up: Vec3 | Sequence[float]) -> FloatArray:
    """``(K,3,3)`` frames from Gram-Schmidt of a **constant world** ``up`` against each
    unit tangent: ``u = normalize(up - (up.w) w)``, ``v = w x u``, ``w =`` the tangent.
    """
    T: PointArray = np.asarray(tangents, dtype=float)
    if T.ndim != 2 or T.shape[1] != 3:
        raise ValueError("fixed_up: tangents must be a (K,3) array, got %s" % (T.shape,))
    T = _as_frames_tangents(T, T.shape[0], "fixed_up")
    U: Vec3 = _unit_literal(up, "fixed_up", "up")

    dot: FloatArray = T @ U
    bad = int(np.argmax(np.abs(dot)))
    if abs(dot[bad]) >= 1.0 - PARALLEL_TOL:
        raise ValueError(
            "fixed_up: tangents[%d] = %s is parallel to up = %s (|cos| = %.17g); the "
            "cross-section plane then contains no image of 'up' and the frame is "
            "arbitrary. Pick an up perpendicular to the path's plane, or use "
            "frames.parallel_transport for a non-planar path."
            % (bad, np.array2string(T[bad], precision=6), np.array2string(U, precision=6),
               abs(dot[bad])))

    u: PointArray = U[None, :] - dot[:, None] * T
    u = u / np.linalg.norm(u, axis=1)[:, None]
    return _assemble(u, T)


def _transport_reference(points: PointArray, tangents: PointArray, r0: Vec3,
                         *, loop: bool) -> tuple[PointArray, Vec3]:
    """The double-reflection sweep itself."""
    k = points.shape[0]
    steps = k if loop else k - 1
    R: PointArray = np.empty((steps + 1, 3), dtype=float)
    R[0] = r0
    for i in range(steps):
        j = (i + 1) % k
        v1: Vec3 = points[j] - points[i]
        c1 = float(v1 @ v1)
        r_l: Vec3 = R[i] - (2.0 / c1) * float(v1 @ R[i]) * v1
        t_l: Vec3 = tangents[i] - (2.0 / c1) * float(v1 @ tangents[i]) * v1
        v2: Vec3 = tangents[j] - t_l
        c2 = float(v2 @ v2)
        if c2 <= 0.0:
            R[i + 1] = r_l
        else:
            R[i + 1] = r_l - (2.0 / c2) * float(v2 @ r_l) * v2
    if loop:
        return R[:k], R[k]
    return R, R[-1]


def _seed(tangent: Vec3, up0: Vec3 | Sequence[float], who: str) -> Vec3:
    """Orthonormalize the seed ``up0`` against the first tangent, or reject it."""
    U: Vec3 = _unit_literal(up0, who, "up0")
    d = float(U @ tangent)
    if abs(d) >= 1.0 - PARALLEL_TOL:
        raise ValueError(
            "%s: up0 = %s is parallel to tangents[0] = %s (|cos| = %.17g); the seed must "
            "have a component in the first cross-section plane. Pick an up0 transverse "
            "to the path's start direction."
            % (who, np.array2string(U, precision=6),
               np.array2string(np.asarray(tangent), precision=6), abs(d)))
    r: Vec3 = U - d * tangent
    return r / float(np.linalg.norm(r))


def parallel_transport(points: PointArray, tangents: PointArray,
                       up0: Vec3 | Sequence[float], *, loop: bool = False,
                       distribute: bool = True) -> FloatArray:
    """``(K,3,3)`` rotation-minimizing frames along ``points``, seeded by ``up0``."""
    P = _as_points(points, "parallel_transport")
    T = _as_frames_tangents(tangents, P.shape[0], "parallel_transport")
    r0 = _seed(T[0], up0, "parallel_transport")

    R, wrap = _transport_reference(P, T, r0, loop=loop)
    if loop and distribute:
        theta = _signed_angle(R[0], wrap, T[0])
        k = P.shape[0]
        R = _spin(R, T, -theta * np.arange(k, dtype=float) / k)
    return _assemble(R, T)


def _spin(u: PointArray, tangents: PointArray, angles: FloatArray) -> PointArray:
    """Rotate each cross-section vector ``u[k]`` by ``angles[k]`` about ``tangents[k]``.
    """
    c: FloatArray = np.cos(angles)[:, None]
    s: FloatArray = np.sin(angles)[:, None]
    out: PointArray = c * u + s * np.cross(tangents, u)
    return out / np.linalg.norm(out, axis=1)[:, None]


def _signed_angle(a: Vec3, b: Vec3, axis: Vec3) -> float:
    """The angle taking ``a`` to ``b`` about ``axis``, both perpendicular to it."""
    return float(np.arctan2(float(np.cross(a, b) @ axis), float(a @ b)))


def holonomy(points: PointArray, tangents: PointArray,
             up0: Vec3 | Sequence[float]) -> float:
    """The residual twist, in **radians**, of transporting a frame once around the
    closed curve ``points`` (the closing segment ``points[K-1] -> points[0]`` included).
    """
    P = _as_points(points, "holonomy")
    T = _as_frames_tangents(tangents, P.shape[0], "holonomy")
    r0 = _seed(T[0], up0, "holonomy")
    R, wrap = _transport_reference(P, T, r0, loop=True)
    return _signed_angle(R[0], wrap, T[0])


def frenet(points: PointArray, tangents: PointArray) -> FloatArray:
    """``(K,3,3)`` Frenet-Serret frames: ``u`` the principal normal, ``v`` the binormal,
    ``w`` the tangent."""
    P = _as_points(points, "frenet")
    T = _as_frames_tangents(tangents, P.shape[0], "frenet")

    D: PointArray = np.empty_like(T)
    D[1:-1] = 0.5 * (T[2:] - T[:-2])
    D[0] = T[1] - T[0]
    D[-1] = T[-1] - T[-2]

    n: FloatArray = np.linalg.norm(D, axis=1)
    bad = int(np.argmin(n))
    if n[bad] <= CURVATURE_TOL:
        raise ValueError(
            "frenet: the curvature vanishes at points[%d] (|dt| = %.17g <= %.17g) -- a "
            "straight (or inflecting) segment has no principal normal, so the "
            "Frenet frame is undefined there. Use frames.fixed_up for a planar path or "
            "frames.parallel_transport for a general one; both are well defined here."
            % (bad, n[bad], CURVATURE_TOL))

    u: PointArray = D / n[:, None]
    u = u - (np.einsum("kj,kj->k", u, T))[:, None] * T
    u = u / np.linalg.norm(u, axis=1)[:, None]
    return _assemble(u, T)


def frame_transform(R_from: FloatArray, origin_from: Point,
                    R_to: FloatArray, origin_to: Point) -> tuple[FloatArray, Vec3]:
    """The single rigid ``(matrix, offset)`` carrying the frame ``(R_from,
    origin_from)`` onto ``(R_to, origin_to)``, in the convention of
    :func:`nekmeshpy.model.affine.apply` -- ``p @ matrix.T + offset``."""
    A: FloatArray = np.asarray(R_from, dtype=float)
    B: FloatArray = np.asarray(R_to, dtype=float)
    for name, M in (("R_from", A), ("R_to", B)):
        if M.shape != (3, 3):
            raise ValueError("frame_transform: %s must be a (3,3) frame, got %s"
                             % (name, (M.shape,)))
    oa: Point = np.asarray(origin_from, dtype=float).reshape(-1)
    ob: Point = np.asarray(origin_to, dtype=float).reshape(-1)
    for name, o in (("origin_from", oa), ("origin_to", ob)):
        if o.shape != (3,):
            raise ValueError("frame_transform: %s must be a (3,) point, got %s"
                             % (name, (o.shape,)))
    matrix: FloatArray = B @ A.T
    offset: Vec3 = ob - matrix @ oa
    return matrix, offset


def plane_frame(points: PointArray, *,
                normal: Vec3 | Sequence[float] | None = None,
                origin: Point | Sequence[float] | None = None,
                hint: Vec3 | Sequence[float] | None = None,
                ) -> tuple[FloatArray, Point]:
    """The **local frame of a planar cross-section**: the ``(R, origin)`` pair naming
    the plane the profile was authored in, ready to hand to :func:`frame_transform` as
    the ``_from`` half of a sweep."""
    P = np.asarray(points, dtype=float)
    if P.ndim != 2 or P.shape[1] != 3 or P.shape[0] < 2:
        raise ValueError("plane_frame: points must be a (K,3) array with K >= 2, got %s"
                         % (P.shape,))
    c: Point = P.mean(axis=0)
    D: PointArray = P - c
    extent = float(np.max(np.linalg.norm(D, axis=1)))
    if extent == 0.0:
        raise ValueError("plane_frame: every point is coincident, so the section has "
                         "no plane and no size")
    if normal is not None:
        w: Vec3 = _unit_literal(normal, "plane_frame", "normal")
    elif P.shape[0] < 3:
        raise ValueError("plane_frame: %d points do not determine a plane; pass "
                         "normal= to name it" % (P.shape[0],))
    else:
        w = np.linalg.svd(D, full_matrices=True)[2][2]
        w = w / float(np.linalg.norm(w))
    if normal is None:
        off = float(np.max(np.abs(D @ w)))
        if off > PLANAR_TOL * extent:
            raise ValueError(
                "plane_frame: the section is not planar -- a point lies %.3g off the "
                "fitted plane, %.3g of the section's own extent (tolerance %.0e). Pass "
                "normal= to name the plane to sweep it from."
                % (off, off / extent, PLANAR_TOL))
    if hint is not None and float(w @ _unit_literal(hint, "plane_frame", "hint")) < 0.0:
        w = -w
    Q: PointArray = D - np.outer(D @ w, w)
    nq: FloatArray = np.linalg.norm(Q, axis=1)
    k = int(np.argmax(nq))
    if nq[k] <= PLANAR_TOL * extent:
        raise ValueError("plane_frame: the section collapses onto the normal "
                         "direction, so it has no in-plane extent to build a frame on")
    o: Point = c if origin is None else np.asarray(origin, dtype=float).reshape(-1)
    if o.shape != (3,):
        raise ValueError("plane_frame: origin must be a (3,) point, got %s"
                         % (np.shape(origin),))
    return _assemble((Q[k] / nq[k])[None, :], w[None, :])[0], o


def sweep_placements(profile_points: PointArray, path_points: PointArray, *,
                     orientation: Literal["transport", "fixed", "frenet"] = "transport",
                     up: Vec3 | Sequence[float] | PointArray | None = None,
                     twist: float = 0.0,
                     close_twist: bool = True,
                     loop: bool = False,
                     origin: Point | Sequence[float] | None = None,
                     normal: Vec3 | Sequence[float] | None = None,
                     path_tangents: PointArray | None = None,
                     ) -> list[tuple[FloatArray, Vec3]]:
    """One rigid ``(matrix, offset)`` per path station, carrying a planar profile from
    its own plane onto the moving frame of the sampled curve ``path_points`` ``(K,3)``.
    """
    P = _as_points(path_points, "sweep_placements")
    K = P.shape[0]
    T = (tangents(P, loop=loop) if path_tangents is None
         else _as_frames_tangents(path_tangents, K, "sweep_placements"))
    R_from, o_from = plane_frame(profile_points, normal=normal, origin=origin,
                                 hint=T[0])
    # ``up`` carries either a single ``(3,)`` world direction or a ``(K,3)`` per-station
    # field; rank alone separates them, so K == 3 is not a special case.
    per_station: PointArray | None = None
    if up is not None and np.ndim(up) > 1:
        per_station = np.asarray(up, dtype=float)
        if per_station.shape != (K, 3):
            raise ValueError("sweep_placements: a per-station up must be (%d,3) to "
                             "match the path, got %s" % (K, (per_station.shape,)))
        if orientation != "fixed":
            raise ValueError(
                "sweep_placements: a per-station (K,3) up needs orientation='fixed' -- "
                "it names one held direction per station, which is what fixed_up "
                "consumes. orientation=%r has no use for more than one up vector (a "
                "transported frame is fixed by its seed alone), so the rest would be "
                "silently ignored. Pass a single (3,) up, or switch to 'fixed'."
                % (orientation,))
    if orientation == "transport":
        R = parallel_transport(P, T, R_from[:, 0] if up is None else up,
                               loop=loop, distribute=close_twist)
    elif orientation == "fixed":
        if up is None:
            raise ValueError(
                "sweep_placements: orientation='fixed' needs up= -- the constant "
                "world direction the cross-section is held against (for a planar "
                "path, the plane's normal), or a (K,3) array naming one per station. "
                "Use orientation='transport' to have the frame carried along the "
                "curve instead.")
        if per_station is None:
            R = fixed_up(T, up)
        else:
            # fixed_up is pointwise, so a per-station field is just it applied one
            # station at a time -- each frame against its own tangent and its own up.
            R = np.stack([fixed_up(T[k:k + 1], per_station[k])[0] for k in range(K)])
    elif orientation == "frenet":
        R = frenet(P, T)
    else:
        raise ValueError(
            "sweep_placements: orientation must be 'transport', 'fixed' or 'frenet', "
            "got %r. Per-station up vectors are no longer passed here -- hand them to "
            "up= as a (K,3) array with orientation='fixed'." % (orientation,))
    # Hold the section in the orientation it was *authored* in.  Every generator
    # fixes the frame field only up to a constant roll about the tangent -- ``fixed_up``
    # picks the phase ``up`` implies, ``frenet`` the one the curvature does -- and none
    # of them knows which way round the caller drew the section.  Spinning all ``K``
    # frames by the single angle that lands station 0 back on ``R_from`` introduces no
    # *relative* twist (the field is rigid under it), and buys the contract that a
    # section swept along a straight path comes out exactly where ``extrude`` would put
    # it, and that the three generators agree with each other on a planar path.
    u0: Vec3 = R_from[:, 0] - float(R_from[:, 0] @ T[0]) * T[0]
    n0 = float(np.linalg.norm(u0))
    if n0 > PLANAR_TOL:
        phase = _signed_angle(R[0, :, 0], u0 / n0, T[0])
        R = _assemble(_spin(R[:, :, 0], T, np.full(K, phase)), T)
    if twist != 0.0:
        span = float(K if loop else max(K - 1, 1))
        R = _assemble(_spin(R[:, :, 0], T, twist * np.arange(K) / span), T)
    return [frame_transform(R_from, o_from, R[k], P[k]) for k in range(K)]
