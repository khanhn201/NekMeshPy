"""Moving orthonormal frames along a sampled curve -- the placement half of a sweep.

Sweeping a cross-section along a curved path is a *rigid placement* problem, not a
meshing one: at every station along the path you need a rotation and a translation that
carry a profile authored in some local frame onto the path.  This module answers only
the frame question, on plain ``(K,3)`` coordinate arrays, and hands the result to
``model/affine.py`` -- so, like ``model/conform.py`` and ``model/affine.py``, it
**imports no container**.  Nothing here knows what a ``LineMesh`` is.

**Frame convention: columns.**  Each frame is a ``(3,3)`` matrix ``R`` whose *columns*
are ``(u, v, w)``, with

* ``w = R[:, 2]`` the **unit tangent** -- the sweep direction,
* ``u = R[:, 0]``, ``v = R[:, 1]`` an orthonormal basis of the cross-section plane,
* ``v = w x u``, which makes ``det(R) == +1`` (right-handed) by construction.

Columns rather than rows because that is exactly the convention
:func:`nekmeshpy.model.affine.apply` wants: it evaluates ``p @ matrix.T + offset``,
i.e. ``matrix @ p`` for a column point, so a column-frame ``R`` maps *local* section
coordinates to *world* coordinates directly (``world = R @ local + origin``) and
``R.T`` maps back.  :func:`frame_transform` composes the two halves into the single
``(matrix, offset)`` pair the consumer feeds straight to ``apply`` or to a mesh's
``.transform(matrix, offset)``.

**Which frame generator to use.**  Three are offered, and the choice matters:

* :func:`fixed_up` -- Gram-Schmidt of a constant world ``up`` against each tangent.
  For a **planar** path (a 90-degree elbow in the ``xy`` plane with ``up=(0,0,1)``)
  this is the right default: it is *exact* (no integration, no accumulated drift, no
  dependence on the sampling density) and introduces exactly zero twist.  It fails --
  loudly -- the moment a tangent becomes parallel to ``up``, which is precisely when
  "up" stops naming a direction in the cross-section plane.
* :func:`parallel_transport` -- rotation-minimizing frames (RMF) by the double
  reflection method of Wang, Juttler, Zheng & Liu, *Computation of rotation minimizing
  frames*, ACM TOG 27(1), 2008.  Use it for a genuinely non-planar path.  It is
  second-order accurate in the sampling and, being a transport rather than a
  differential-geometric construction, it is perfectly well behaved on a straight
  segment and through an inflection -- the two places Frenet breaks.  Its cost is that
  it is *integrated*: the frame at station ``k`` depends on the whole prefix, so on a
  closed path it generally does not come back to itself (see *holonomy* below).
* :func:`frenet` -- the Frenet-Serret frame.  Included for completeness and for
  comparison, but **it is the wrong tool for a sweep**: its normal is defined by the
  curvature vector, so it is *undefined* on a straight segment (zero curvature -- this
  implementation raises rather than emit NaN) and it *flips sign* through an
  inflection point, which tears a swept mesh in half.  A U-turn threaded through a
  straight lead-in hits both.  Prefer :func:`fixed_up`, else :func:`parallel_transport`.

**Holonomy on a closed path.**  Transport a frame all the way around a closed
non-planar curve and it returns rotated about the tangent by a residual angle -- the
geometric (Berry) phase of the curve.  This is a property of the *curve*, not a bug and
not a discretization error, so it cannot be "fixed"; it can only be placed somewhere.
:func:`parallel_transport` with ``loop=True`` therefore **reports** it via
:func:`holonomy` and, by default (``distribute=True``), spreads it linearly over the
``K`` steps -- frame ``k`` is spun back by ``-theta * k / K`` about its own tangent.
The frame *field* is single-valued either way (there are only ``K`` frames and the
seam element joins ``K-1`` back to ``0``); what changes is where the twist sits.  Raw,
every element is twist-free except the seam, which carries the whole ``-theta``;
distributed, all ``K`` elements carry ``-theta / K`` each and the seam is no longer
special.  That is almost always what a periodic ``loft`` wants -- one sheared element
is a quality outlier, a uniform twist is not -- but pass ``distribute=False`` to get
the raw frames and place (or refuse) the residual yourself.  On a **planar** closed
curve the holonomy is exactly zero, so the distribution is a no-op there.
"""

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
    cumulative-chord parameters ``0, a, a+b``.  Reduces to ``(-3x0 + 4x1 - x2) / (2h)``
    for uniform ``a == b == h``; the direction is what the caller uses, so the
    parameter scale is immaterial, but the *ratio* ``a : b`` is not -- assuming uniform
    spacing on a graded curve reintroduces the first-order error this rule removes."""
    d: Vec3 = (-(2.0 * a + b) / (a * (a + b)) * x0
               + (a + b) / (a * b) * x1
               - a / (b * (a + b)) * x2)
    return d


def tangents(points: PointArray, *, loop: bool = False) -> PointArray:
    """Unit tangents ``(K,3)`` of the ``(K,3)`` sampled curve ``points``.

    Central differences on the interior (``x[i+1] - x[i-1]``, second-order accurate for
    a smooth curve even under mild non-uniform sampling), one-sided at the two ends --
    or wrapped, when ``loop=True``, which makes the closing segment ``x[K-1] -> x[0]``
    a real segment and the tangent field genuinely periodic.

    The one-sided ends use the **three-point** rule (the derivative at the end of the
    quadratic through the three nearest samples, in their own cumulative-chord
    parameter, so non-uniform spacing is handled exactly) rather than the plain forward
    difference.  That matters more than it looks: the end tangent seeds
    :func:`parallel_transport`, so a first-order end error is carried the whole way
    along the curve and would cap the frame field at first-order accuracy however fine
    the sampling.  With the three-point rule the whole pipeline is second order.
    (``K == 2`` has no third point and falls back to the chord, which is exact for the
    straight line it can only be describing.)

    A zero-length segment (a repeated point) makes the tangent undefined, so it is a
    loud ``ValueError`` naming the index rather than a NaN that propagates silently
    into every downstream frame.  Likewise a coincident ``x[i-1] == x[i+1]``, which is
    an exact 180-degree cusp: there is no tangent there, only two of them.
    """
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

    This is the right default whenever it is defined.  It is *pointwise* -- frame ``k``
    depends on ``tangents[k]`` alone -- so it neither integrates nor drifts, it is exact
    at any sampling density, and it introduces **zero** twist along the sweep.  On a
    planar path it coincides with :func:`parallel_transport` seeded by the same ``up``
    (a planar curve has zero torsion, so the rotation-minimizing frame *is* the
    fixed-up frame); the transport is only worth its cost once the path leaves a plane.

    Fails when a tangent turns parallel to ``up``: the projection then has no direction
    left to normalize and the cross-section would spin arbitrarily.  That is reported as
    a ``ValueError`` naming the index, not silently regularized -- the caller must
    choose a different ``up`` (one perpendicular to the path's plane, if it has one) or
    switch to :func:`parallel_transport`.
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
    """The double-reflection sweep itself.

    Returns the ``(K,3)`` transported reference vectors and, when ``loop`` is set, the
    extra vector obtained by transporting once more across the closing segment
    ``x[K-1] -> x[0]``.  Comparing that against ``r[0]`` is the holonomy measurement.

    One step of Wang et al. (2008): reflect the frame in the bisecting plane of the
    chord, then reflect again in the bisecting plane of the two tangents.  Two
    reflections compose to a rotation, so orthonormality is exact by construction
    rather than maintained by renormalization -- this is why the method neither drifts
    nor needs a Gram-Schmidt fixup, and why it survives a straight segment (the second
    reflection degenerates to the identity, which is exactly the right answer: no
    turning, no frame change).
    """
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
    """``(K,3,3)`` rotation-minimizing frames along ``points``, seeded by ``up0``.

    The double-reflection method of Wang, Juttler, Zheng & Liu (2008): the frame is
    carried from station to station by two reflections, which compose to a rotation, so
    each step is exactly orthogonal and the accumulated frame is second-order accurate
    in the sample spacing without any renormalization.  Unlike :func:`frenet` it has no
    dependence on curvature, so a straight segment transports the frame unchanged and an
    inflection point passes through with nothing to flip -- the two failure modes a
    U-turn with straight leads hits immediately.

    ``up0`` seeds the cross-section basis at ``points[0]``: it is orthonormalized
    against ``tangents[0]`` (so it need not already be perpendicular) and rejected as a
    ``ValueError`` if it is parallel to it.

    With ``loop=True`` the closing segment ``points[K-1] -> points[0]`` is transported
    too and the frame does not come back to itself -- the curve's holonomy.  By default
    that residual angle ``theta`` (see :func:`holonomy`) is **distributed linearly**,
    frame ``k`` spun back by ``-theta * k / K`` about its own tangent, so every element
    carries the same ``-theta / K`` of twist.  With ``distribute=False`` the raw
    transported frames are returned: every element is then twist-free except the seam,
    which carries the whole ``-theta``.  Measure it with :func:`holonomy` and decide.
    """
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

    Rodrigues, vectorized; ``u`` is perpendicular to its tangent, so the
    ``(1-cos)(k.u)k`` term vanishes identically and the result stays perpendicular
    (and unit) to machine precision.
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

    Exactly zero -- to round-off -- for a planar closed curve, and generally non-zero
    otherwise: it is the curve's geometric phase, a property of the geometry rather
    than of the discretization, so refining the sampling converges it to a non-zero
    limit instead of driving it to zero.  This is the number
    ``parallel_transport(..., loop=True)`` distributes; call it directly when the
    consumer needs to reject a path whose twist per element would be unacceptable, or
    to compare candidate ``up0`` seeds (it does not depend on the seed except through
    round-off -- holonomy is a property of the curve, not the frame).
    """
    P = _as_points(points, "holonomy")
    T = _as_frames_tangents(tangents, P.shape[0], "holonomy")
    r0 = _seed(T[0], up0, "holonomy")
    R, wrap = _transport_reference(P, T, r0, loop=True)
    return _signed_angle(R[0], wrap, T[0])


def frenet(points: PointArray, tangents: PointArray) -> FloatArray:
    """``(K,3,3)`` Frenet-Serret frames: ``u`` the principal normal, ``v`` the binormal,
    ``w`` the tangent.

    **Do not sweep with this** unless you know the curve is strictly convex.  The normal
    is the direction of ``dt/ds``, which means:

    * on a **straight segment** the curvature is zero and there is no normal at all.
      This implementation raises a ``ValueError`` naming the index rather than dividing
      by ~0 and returning NaN or an arbitrary round-off direction.
    * through an **inflection** the curvature vector passes through zero and reverses,
      so the frame **flips by 180 degrees** between two adjacent samples.  A section
      swept on such a frame is turned inside out at that element -- the mesh is not
      merely twisted, it is inverted.
    * the frame also spins with the curve's **torsion** even where it is well defined,
      which is exactly the twist :func:`parallel_transport` exists to minimize.

    It is provided because it is the textbook frame and because it is the useful
    reference to *measure* twist against -- not because a sweep should use it.  Reach
    for :func:`fixed_up` first and :func:`parallel_transport` second.
    """
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
    """The single rigid ``(matrix, offset)`` carrying the frame ``(R_from, origin_from)``
    onto ``(R_to, origin_to)``, in the convention of
    :func:`nekmeshpy.model.affine.apply` -- ``p @ matrix.T + offset``.

    With the column convention of this module, ``R.T @ (p - origin)`` reads a world
    point in the source frame's local coordinates and ``R_to @ . + origin_to`` writes it
    back out in the target's, so ``matrix = R_to @ R_from.T`` and
    ``offset = origin_to - matrix @ origin_from``.  ``matrix`` is a product of two
    orthogonal matrices and therefore orthogonal: the map is rigid, so a profile placed
    with it keeps its shape, its lengths and its element quality exactly -- which is the
    whole point of placing a section rather than re-meshing it.  ``matrix`` is returned
    as an explicit array (never ``None``) even when it is the identity; that case is a
    pure translation only when the two frames agree, and detecting it is the caller's
    business, not a silent branch here.

    ``frame_transform(R, o, R, o)`` is the identity to round-off.
    """
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
    the ``_from`` half of a sweep.

    ``origin`` defaults to the **centroid** -- the only defensible automatic choice,
    but not always the one a sweep wants: an O-grid disc's centroid misses its centre
    by the grid's own slight asymmetry, so pass ``origin=`` explicitly when the section
    has a distinguished axis point (the centre the boundary loop was built about) and
    you want *that* riding the path.  ``R``'s third column ``w`` defaults to the plane
    normal -- the smallest right singular vector of the centred coordinates, i.e. the
    exact least-squares plane, which for a genuinely planar profile is that plane to
    round-off.  The in-plane ``u`` axis is taken along the profile point *farthest* from
    the centroid, the best-conditioned in-plane direction available and a deterministic
    function of the point array (so two calls on the same profile agree bit-for-bit,
    which is what lets a caller re-derive the frame instead of threading it through).

    ``normal=`` overrides the fit **and the planarity check with it** -- naming the
    plane is the caller asserting one, which is what makes it the escape hatch for a
    profile with fewer than three points, or one that is genuinely not planar and whose
    sweep plane only the caller knows.  Without it, a profile more than ``PLANAR_TOL`` of its own extent off the fitted
    plane is a loud ``ValueError``: sweeping a non-planar section is not wrong, but the
    frame it should be read in is then genuinely ambiguous and guessing it silently
    would shear the whole block.  ``hint=`` flips ``w`` to agree with a direction (the
    path's start tangent, in a sweep) so the section is not swept back-to-front.
    """
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
    if normal is None and float(np.max(np.abs(D @ w))) > PLANAR_TOL * extent:
        off = float(np.max(np.abs(D @ w)))
        raise ValueError(
            "plane_frame: the section is not planar -- a point lies %.3g off the "
            "fitted plane, %.3g of the section's own extent (tolerance %.0e). Pass "
            "normal= to name the plane to sweep it from." % (off, off / extent,
                                                             PLANAR_TOL))
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

    This is the whole placement half of a swept block, and the reason a sweep is *not*
    a per-point offset of the profile: the profile is moved **rigidly**, so through a
    bend of radius ``Rb`` a node sitting ``d`` outboard of the centreline traverses
    radius ``Rb + d`` and one ``d`` inboard traverses ``Rb - d``.  They travel different
    distances and neither follows the path -- which is exactly right, and exactly what
    offsetting every profile point along its own copy of the curve would get wrong (it
    would shear the section and, on a tight bend, invert the inboard elements).

    ``orientation`` selects the frame generator and **only** that: ``"transport"``
    (the default -- :func:`parallel_transport`, seeded from the profile's own in-plane
    axis so the section does not spin at the start), ``"fixed"`` (:func:`fixed_up`,
    which needs ``up=`` and is the exact choice for a planar path), or ``"frenet"``.
    It is a mode name, never data: the per-station up vectors that used to be passed
    *as* ``orientation`` now ride ``up=`` instead, where the rest of the up-vector data
    already lives.

    ``up`` is that data, and it takes either shape.  A ``(3,)`` direction is the
    constant world up: with ``"fixed"`` it is held against every tangent, with
    ``"transport"`` it seeds the first frame.  A ``(K,3)`` array is **per station**,
    one up vector per path point, each orthonormalized against its own tangent -- that
    is a generalization of ``"fixed"`` and nothing else, so it requires
    ``orientation="fixed"`` and is a ``ValueError`` with any other generator (a
    transported frame has exactly one degree of freedom, its seed, and would silently
    ignore the other ``K-1`` rows).  The two are told apart by rank, not by ``K``:
    1-D is the constant direction, 2-D the per-station field, so a three-station sweep
    is not ambiguous.

    ``twist`` adds a total roll of that many **radians** about the tangent, spread
    linearly over the stations -- a helical sweep.  On a closed path it does not close
    unless it is a multiple of ``2*pi``; nothing here checks that, because a
    deliberately non-closing twist has no meaning to reject against.  ``close_twist``
    is :func:`parallel_transport`'s ``distribute``: on a closed non-planar path the
    holonomy is spread over every element rather than dumped in the seam.
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
        R = (np.stack([fixed_up(T[k:k + 1], per_station[k])[0] for k in range(K)])
             if per_station is not None else fixed_up(T, up))
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
