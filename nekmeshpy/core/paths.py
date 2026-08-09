"""A declarative 2-D turtle-walk path builder."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Callable, NamedTuple, Tuple

import numpy as np

from .._typing import BoolArray, FloatArray, Point, PointArray, Vec3

#: How far ``embed``'s two in-plane axes may stray from unit length and mutual
#: orthogonality before it refuses to build a frame from them.
AXIS_TOL = 1e-12

#: One move: ``("line", length, 0.0)`` or ``("arc", radius, signed_degrees)`` -- a
#: line's third slot is unused, kept only so every move is a uniform 3-tuple.
Move = Tuple[str, float, float]


class TurtlePath(NamedTuple):
    """A turtle-walked path, exposed as continuous callables of normalized arc
    length ``s in [0, 1]`` rather than as its own segment table."""

    #: ``(K,)`` in ``[0, 1]`` -> ``(K, 2)`` points, vectorized for
    #: :func:`hexmesh.lift.sweep <nekmeshpy.hexmesh.lift.sweep>`, which samples the whole node
    #: lattice in one call to build a moving frame along it.
    centerline: Callable[[FloatArray], FloatArray]
    #: ``(K,)`` in ``[0, 1]`` -> ``(K, 2)`` unit tangents, the analytic derivative
    #: of :attr:`centerline`.
    tangent: Callable[[FloatArray], FloatArray]
    #: The exact total arc length of the walk.
    total_length: float
    #: Normalized ``s`` of every straight<->arc junction, strictly increasing --
    #: feed these through :func:`LineMesh.sweep_fractions
    #: <nekmeshpy.linemesh.shape.sweep_fractions>` so a sweep grades each piece
    #: on its own and lands a station exactly on every junction.
    break_fractions: FloatArray


def turtle_path(moves: Sequence[Move], start: Sequence[float] = (0.0, 0.0),
                heading: float = 0.0) -> TurtlePath:
    """Walk ``moves`` from ``start`` heading ``heading`` (radians, measured from ``+x``)
    into a :class:`TurtlePath`."""
    is_arc, p0, direction, center, radius, theta0, dtheta, length = (
        [], [], [], [], [], [], [], [])
    p: FloatArray = np.asarray(start, dtype=float)
    phi = float(heading)
    for move in moves:
        if move[0] == "line":
            d = np.array([np.cos(phi), np.sin(phi)])
            is_arc.append(False)
            p0.append(p.copy())
            direction.append(d)
            center.append(np.zeros(2))
            radius.append(1.0)
            theta0.append(0.0)
            dtheta.append(0.0)
            length.append(float(move[1]))
            p = p + move[1] * d
        elif move[0] == "arc":
            r, turn = float(move[1]), float(np.deg2rad(move[2]))
            sgn = 1.0 if turn > 0.0 else -1.0
            # the centre lies to the left of the heading for a CCW (positive) turn
            nrm = np.array([np.cos(phi + sgn * 0.5 * np.pi),
                            np.sin(phi + sgn * 0.5 * np.pi)])
            c = p + r * nrm
            t0 = float(np.arctan2(p[1] - c[1], p[0] - c[0]))
            is_arc.append(True)
            p0.append(p.copy())
            direction.append(np.zeros(2))
            center.append(c)
            radius.append(r)
            theta0.append(t0)
            dtheta.append(turn)
            length.append(r * abs(turn))         # exact: radius * angle
            p = c + r * np.array([np.cos(t0 + turn), np.sin(t0 + turn)])
            phi = phi + turn
        else:
            raise ValueError("turtle_path: move must be 'line' or 'arc', got %r"
                             % (move[0],))

    is_arc_a: BoolArray = np.asarray(is_arc, dtype=bool)
    p0_a = np.asarray(p0, dtype=float)
    dir_a = np.asarray(direction, dtype=float)
    cen_a = np.asarray(center, dtype=float)
    rad_a: FloatArray = np.asarray(radius, dtype=float)
    th0_a: FloatArray = np.asarray(theta0, dtype=float)
    dth_a: FloatArray = np.asarray(dtheta, dtype=float)
    len_a: FloatArray = np.asarray(length, dtype=float)
    cum = np.concatenate([np.zeros(1), np.cumsum(len_a)])   # (S+1,)
    total = float(cum[-1])
    breaks = cum[1:-1] / total       # normalized s of every segment junction

    def locate(s: FloatArray) -> tuple[FloatArray, FloatArray]:
        """Dispatch normalized arc lengths onto ``(segment index, arc length into it)``
        -- both closed forms below start here, via a ``searchsorted`` of the cumulative
        table, so nothing is ever evaluated with a neighbour's geometry."""
        t = np.clip(np.asarray(s, dtype=float).ravel(), 0.0, 1.0) * total
        idx = np.clip(np.searchsorted(cum, t, side="right") - 1, 0, len_a.size - 1)
        return idx, t - cum[idx]

    def centerline(s: FloatArray) -> FloatArray:
        idx, loc = locate(s)
        line_xy = p0_a[idx] + dir_a[idx] * loc[:, None]
        theta = th0_a[idx] + np.sign(dth_a[idx]) * loc / rad_a[idx]
        arc_xy = cen_a[idx] + rad_a[idx][:, None] * np.stack(
            [np.cos(theta), np.sin(theta)], axis=1)
        return np.where(is_arc_a[idx][:, None], arc_xy, line_xy)

    def tangent(s: FloatArray) -> FloatArray:
        """The analytic derivative of :func:`centerline`: a straight's is its own
        constant direction; an arc's is the radius vector turned a quarter turn in the
        direction of travel."""
        idx, loc = locate(s)
        sgn = np.sign(dth_a[idx])
        theta = th0_a[idx] + sgn * loc / rad_a[idx]
        arc_xy = sgn[:, None] * np.stack([-np.sin(theta), np.cos(theta)], axis=1)
        return np.where(is_arc_a[idx][:, None], arc_xy, dir_a[idx])

    return TurtlePath(centerline, tangent, total, breaks)


class SpacePath(NamedTuple):
    """A path **in space**, exposed as continuous callables of normalized arc length ``s
    in [0, 1]`` -- the 3-D counterpart of :class:`TurtlePath` and what
    :func:`hexmesh.lift.sweep_path <nekmeshpy.hexmesh.lift.sweep_path>` consumes."""

    #: ``(K,)`` in ``[0, 1]`` -> ``(K, 3)`` points.
    centerline: Callable[[FloatArray], PointArray]
    #: ``(K,)`` in ``[0, 1]`` -> ``(K, 3)`` unit tangents, the analytic derivative of
    #: :attr:`centerline`.  A **direction**: no origin enters it.
    tangent: Callable[[FloatArray], PointArray]
    #: The exact total arc length.
    total_length: float
    #: Normalized ``s`` of every straight<->arc junction, strictly increasing.
    break_fractions: FloatArray


def _plane_axis(vector: Vec3 | Sequence[float], name: str) -> Vec3:
    a: FloatArray = np.asarray(vector, dtype=float).reshape(-1)
    if a.shape != (3,):
        raise ValueError("embed: %s must be a (3,) vector, got %s"
                         % (name, np.shape(vector)))
    off = abs(float(a @ a) - 1.0)
    if off > AXIS_TOL:
        raise ValueError(
            "embed: %s = %s is not a unit vector (|%s|^2 is %.17g off 1). The in-plane "
            "axes are used verbatim, so a non-unit one silently rescales the path "
            "rather than just naming its plane." % (name, np.array2string(a), name, off))
    return a


def embed(path: TurtlePath, *, u: Vec3 | Sequence[float], v: Vec3 | Sequence[float],
          origin: Point | Sequence[float] = (0.0, 0.0, 0.0)) -> SpacePath:
    """Lift a 2-D :class:`TurtlePath` onto the plane spanned by ``u`` and ``v`` through
    ``origin``: the walk's own ``(x, y)`` becomes ``origin + x*u + y*v``."""
    U = _plane_axis(u, "u")
    V = _plane_axis(v, "v")
    dot = float(U @ V)
    if abs(dot) > AXIS_TOL:
        raise ValueError(
            "embed: u and v are not orthogonal (u.v = %.17g); they span a sheared "
            "plane, so the lifted path would not preserve the walk's own lengths."
            % dot)
    O: Point = np.asarray(origin, dtype=float).reshape(-1)
    if O.shape != (3,):
        raise ValueError("embed: origin must be a (3,) point, got %s"
                         % (np.shape(origin),))

    def centerline(s: FloatArray) -> PointArray:
        xy = path.centerline(s)
        return O + (xy[:, 0, None] * U + xy[:, 1, None] * V)

    def tangent(s: FloatArray) -> PointArray:
        xy = path.tangent(s)
        return xy[:, 0, None] * U + xy[:, 1, None] * V

    return SpacePath(centerline, tangent, path.total_length, path.break_fractions)


__all__ = ["AXIS_TOL", "Move", "SpacePath", "TurtlePath", "embed", "turtle_path"]
