"""A declarative 3-D turtle-walk path builder.

A path is a **move table** -- :func:`line`, :func:`arc`, :func:`helix` -- walked by
:func:`walk` into a :class:`Path`.  The turtle carries a full orthonormal frame,
not just a heading, which is what makes the vocabulary three-dimensional: ``tilt`` rolls
a bend's own plane about the direction of travel, so one arc verb reaches every
direction out of the current frame, and ``roll`` spins the frame about that direction as
the move travels.  Every move starts at the previous one's end point *and* keeps its
frame, so a walk is C1 by construction -- no fillet fitting, no corner rounding.

Each move is a **constant screw**, closed-form in arc length: exact total length (no
quadrature), an exact analytic tangent (differencing a sampled path is only O(h**2), and
worst exactly at a junction where the curvature jumps), and an ``s in [0, 1]`` that is
true arc length, so an element length is a station count.

The frame the walk carried comes back on :attr:`Path.up`, and
:func:`hexmesh.sweep_path <nekmeshpy.hexmesh.lift.sweep_path>` holds it per station
unless told otherwise -- which is the only way an out-of-plane bend or a distributed
twist can reach the mesh, since a sweep's own generators fix the frame only up to how
the path itself was rolled.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Callable, Literal, NamedTuple, Union

import numpy as np

from .._typing import FloatArray, Point, PointArray, Vec3
from .frames import PARALLEL_TOL, spin

_log = logging.getLogger(__name__)

#: A move shorter than this carries no length to distribute a roll over, and would put
#: two junctions on one ``s``.  Absolute rather than relative to the walk: the walk's own
#: extent is not known until every move has been read, and the quantity is a length that
#: has to be divided by, so what matters is that it is above the reciprocal's noise.
LENGTH_TOL = 1e-12


class Line(NamedTuple):
    """A straight run of ``length``, spinning the frame ``roll`` degrees as it goes."""

    #: Arc length of the run.
    length: float
    #: Degrees of frame spin about the heading, spread over the whole run.
    roll: float = 0.0


class Arc(NamedTuple):
    """A circular bend of ``radius`` through ``angle`` degrees, in the plane ``tilt``
    degrees round from the frame's own up."""

    #: Radius of the bend.
    radius: float
    #: Signed turn in degrees -- positive turns toward the frame's left.
    angle: float
    #: Degrees the **bend plane** is rolled about the heading: 0 turns left, 180 right,
    #: 90 pitches up, anything between is oblique.
    tilt: float = 0.0
    #: Degrees of frame spin about the heading, spread over the whole bend -- *on top of*
    #: the rotation the bend itself carries.
    roll: float = 0.0


class Helix(NamedTuple):
    """A constant screw: a bend of ``radius`` through ``angle`` degrees while advancing
    ``rise`` per full turn along the bend axis."""

    #: Radius measured from the helix axis.
    radius: float
    #: Signed turn in degrees about the helix axis.
    angle: float
    #: Advance along the helix axis per full 360-degree turn.
    rise: float = 0.0
    #: Degrees the helix axis is rolled about the heading, as :attr:`Arc.tilt`.
    tilt: float = 0.0
    #: Degrees of frame spin about the heading, spread over the whole run.
    roll: float = 0.0


#: One row of a move table -- whatever :func:`line`, :func:`arc` or :func:`helix` returns.
Move = Union[Line, Arc, Helix]


def line(length: float, *, roll: float = 0.0) -> Line:
    """A straight run of ``length``, twisting ``roll`` degrees over its own length."""
    return Line(float(length), float(roll))


def arc(radius: float, angle: float, *, tilt: float = 0.0, roll: float = 0.0) -> Arc:
    """A bend of ``radius`` through ``angle`` degrees -- ``tilt`` picks which way it bends
    (0 left, 180 right, 90 up), ``roll`` twists the section as it travels."""
    return Arc(float(radius), float(angle), float(tilt), float(roll))


def helix(radius: float, angle: float, *, rise: float = 0.0, tilt: float = 0.0,
          roll: float = 0.0) -> Helix:
    """A screw of ``radius`` through ``angle`` degrees, climbing ``rise`` per full turn
    along the axis ``tilt`` names."""
    return Helix(float(radius), float(angle), float(rise), float(tilt), float(roll))


class Path(NamedTuple):
    """A curve in space, exposed as continuous callables of normalized arc length ``s in
    [0, 1]`` rather than as its own segment table -- what :func:`hexmesh.lift.sweep_path
    <nekmeshpy.hexmesh.lift.sweep_path>` consumes.  :func:`walk` builds one; so can a
    caller with a parametrization of their own, which is why every field is a callable
    and only :attr:`up` may be left out."""

    #: ``(K,)`` in ``[0, 1]`` -> ``(K, 3)`` points.
    centerline: Callable[[FloatArray], PointArray]
    #: ``(K,)`` in ``[0, 1]`` -> ``(K, 3)`` unit tangents, the analytic derivative of
    #: :attr:`centerline`.  A **direction**: no origin enters it.
    tangent: Callable[[FloatArray], PointArray]
    #: The exact total arc length.
    total_length: float
    #: Normalized ``s`` of every junction between moves, strictly increasing.
    break_fractions: FloatArray
    #: ``(K,)`` in ``[0, 1]`` -> ``(K, 3)`` unit vectors perpendicular to
    #: :attr:`tangent`: the **cross-section reference axis the path itself carries**, so a
    #: sweep along it needs no ``orientation``/``up`` of its own.  ``None`` on a path that
    #: names no frame, which leaves the sweep to build one.
    up: Callable[[FloatArray], PointArray] | None = None


def _direction(vector: Vec3 | Sequence[float], who: str, name: str) -> Vec3:
    """Validate and normalize a ``(3,)`` direction."""
    v: Vec3 = np.asarray(vector, dtype=float).reshape(-1)
    if v.shape != (3,):
        raise ValueError("%s: %s must be a (3,) direction, got %s"
                         % (who, name, (np.shape(vector),)))
    n = float(np.linalg.norm(v))
    if n == 0.0:
        raise ValueError("%s: %s must be a non-zero direction" % (who, name))
    return v / n


def _seed_frame(heading: Vec3 | Sequence[float],
                up: Vec3 | Sequence[float]) -> tuple[Vec3, Vec3]:
    """The walk's starting ``(u, w)``: ``up`` orthonormalized against the heading."""
    w = _direction(heading, "walk", "heading")
    U = _direction(up, "walk", "up")
    d = float(U @ w)
    if abs(d) >= 1.0 - PARALLEL_TOL:
        raise ValueError(
            "walk: up = %s is parallel to heading = %s (|cos| = %.17g); up names the "
            "cross-section axis the walk carries, so it must have a component across "
            "the direction of travel. Pick an up transverse to heading."
            % (np.array2string(U, precision=6), np.array2string(w, precision=6), abs(d)))
    u: Vec3 = U - d * w
    return u / float(np.linalg.norm(u)), w


def _bend_axis(u: Vec3, v: Vec3, tilt: float) -> Vec3:
    """The axis a bend of ``tilt`` degrees turns about, before the turn's own sign is
    applied: the frame's own up, rolled that far about the heading.  ``tilt = 0`` gives
    ``u``, so a positive turn goes toward ``u x w`` -- the frame's left."""
    th = np.deg2rad(tilt)
    a: Vec3 = np.cos(th) * u + np.sin(th) * v
    return a


def _rodrigues(axis: PointArray, angle: FloatArray, vec: PointArray) -> PointArray:
    """Rotate each ``vec[k]`` by ``angle[k]`` about the unit ``axis[k]``."""
    c: FloatArray = np.cos(angle)[:, None]
    s: FloatArray = np.sin(angle)[:, None]
    dot: FloatArray = np.einsum("kj,kj->k", axis, vec)[:, None]
    out: PointArray = (c * vec + s * np.cross(axis, vec) + (1.0 - c) * dot * axis)
    return out


def _segment(move: Move, p: Point, u: Vec3, w: Vec3, index: int
             ) -> tuple[float, Vec3, Point, float, float, float]:
    """One move as a **constant screw**: ``(length, axis, centre, kappa, advance,
    roll_rate)``, where the point at local arc length ``l`` is ``centre + Rot(axis,
    kappa*l) @ (p - centre) + advance*l*axis`` and the frame there is ``Rot`` of the frame
    here.

    A straight is the degenerate screw with ``kappa = 0``, travelling along its own axis;
    an arc is the one with ``advance = 0``; the helix is the general case.  Writing all
    three this way is what keeps the samplers below a single closed form rather than a
    ``where`` over three geometries."""
    v: Vec3 = np.cross(w, u)
    if isinstance(move, Line):
        length = move.length
        _check_length(length, index, move)
        return length, w, p, 0.0, 1.0, np.deg2rad(move.roll) / length
    if isinstance(move, Arc):
        _check_radius(move.radius, index, move)
        length = move.radius * abs(np.deg2rad(move.angle))
        _check_length(length, index, move)
        # the axis carries the turn's sign, so the centre lands on the inside of the
        # turn (left for a positive angle, right for a negative one) and every screw
        # below runs through a *positive* angle about its own axis.
        axis = float(np.sign(move.angle)) * _bend_axis(u, v, move.tilt)
        centre: Point = p + move.radius * np.cross(axis, w)
        return (length, axis, centre, 1.0 / move.radius, 0.0,
                np.deg2rad(move.roll) / length)
    if isinstance(move, Helix):
        _check_radius(move.radius, index, move)
        # ``rise_rad`` is the axial advance per radian, so the tangent's own speed is
        # hypot(radius, rise_rad) per radian -- constant, which is what makes the arc
        # length exact rather than an integral.
        rise_rad = move.rise / (2.0 * np.pi)
        speed = float(np.hypot(move.radius, rise_rad))
        length = speed * abs(np.deg2rad(move.angle))
        _check_length(length, index, move)
        sgn = float(np.sign(move.angle))
        axis = sgn * _bend_axis(u, v, move.tilt)
        centre = p + move.radius * np.cross(axis, w)
        # the sgn on the advance cancels the one on the axis, so a positive rise climbs
        # along the axis ``tilt`` named whichever way the turn goes.
        return (length, axis, centre, 1.0 / speed, sgn * rise_rad / speed,
                np.deg2rad(move.roll) / length)
    raise ValueError("walk: move %d must be a paths.line, paths.arc or paths.helix, "
                     "got %r" % (index, (move,)))


def _check_length(length: float, index: int, move: Move) -> None:
    """Reject a move that goes nowhere.  Every move advances the walk, because a roll is
    spread over the length it travels -- there is no zero-length spin."""
    if not length > LENGTH_TOL:
        raise ValueError(
            "walk: move %d, %r, has arc length %.17g, which is not positive. A roll is "
            "spread over the length its own move travels, so a zero-length move has "
            "nothing to spin over -- put the roll= on the move either side of it "
            "instead, where the frame stays continuous." % (index, move, length))


def _check_radius(radius: float, index: int, move: Move) -> None:
    if not radius > 0.0:
        raise ValueError("walk: move %d, %r, has radius %.17g; a bend needs a positive "
                         "radius, and which way it bends is the sign of its angle and "
                         "its tilt, not the radius" % (index, move, radius))


def walk(moves: Sequence[Move], *,
         start: Point | Sequence[float] = (0.0, 0.0, 0.0),
         heading: Vec3 | Sequence[float] = (1.0, 0.0, 0.0),
         up: Vec3 | Sequence[float] = (0.0, 0.0, 1.0)) -> Path:
    """Walk ``moves`` in space from ``start``, setting out along ``heading`` with the
    cross-section's own up along ``up``, into a :class:`Path` that carries its own
    moving frame.

    ``up`` is normalized and projected off ``heading`` -- it names the frame's phase, not
    a length, so only its component across the direction of travel is used, and only a
    seed parallel to ``heading`` is refused.  From there the moves steer: a positive
    angle turns toward the walk's left (``up x heading``), ``tilt`` rolls that bend plane
    about the heading (90 pitches up), and ``roll`` spins the frame right-handed about
    the heading over the move's own length."""
    o: Point = np.asarray(start, dtype=float).reshape(-1)
    if o.shape != (3,):
        raise ValueError("walk: start must be a (3,) point, got %s" % (np.shape(start),))
    u0, w0 = _seed_frame(heading, up)

    p: Point = o
    u: Vec3 = u0
    w: Vec3 = w0
    p0_l, u0_l, w0_l, axis_l, cen_l = [], [], [], [], []
    kap_l, adv_l, rol_l, len_l = [], [], [], []
    for i, move in enumerate(moves):
        length, axis, centre, kappa, advance, roll_rate = _segment(move, p, u, w, i)
        p0_l.append(p)
        u0_l.append(u)
        w0_l.append(w)
        axis_l.append(axis)
        cen_l.append(centre)
        kap_l.append(kappa)
        adv_l.append(advance)
        rol_l.append(roll_rate)
        len_l.append(length)
        # advance the walk by evaluating the segment's own closed form at its far end
        th = np.array([kappa * length])
        ax = axis[None, :]
        p = (centre + _rodrigues(ax, th, (p - centre)[None, :])[0]
             + advance * length * axis)
        w = _rodrigues(ax, th, w[None, :])[0]
        u = spin(_rodrigues(ax, th, u[None, :]), w[None, :],
                 np.array([roll_rate * length]))[0]
    if not len_l:
        raise ValueError("walk: needs at least one move")

    p0_a: PointArray = np.asarray(p0_l, dtype=float)
    u0_a: PointArray = np.asarray(u0_l, dtype=float)
    w0_a: PointArray = np.asarray(w0_l, dtype=float)
    axis_a: PointArray = np.asarray(axis_l, dtype=float)
    cen_a: PointArray = np.asarray(cen_l, dtype=float)
    kap_a: FloatArray = np.asarray(kap_l, dtype=float)
    adv_a: FloatArray = np.asarray(adv_l, dtype=float)
    rol_a: FloatArray = np.asarray(rol_l, dtype=float)
    len_a: FloatArray = np.asarray(len_l, dtype=float)
    cum: FloatArray = np.concatenate([np.zeros(1), np.cumsum(len_a)])
    total = float(cum[-1])
    breaks: FloatArray = cum[1:-1] / total

    def locate(s: FloatArray) -> tuple[FloatArray, FloatArray]:
        """Dispatch normalized arc lengths onto ``(segment index, arc length into it)``
        -- every sampler below starts here, via a ``searchsorted`` of the cumulative
        table, so nothing is ever evaluated with a neighbour's geometry."""
        t = np.clip(np.asarray(s, dtype=float).ravel(), 0.0, 1.0) * total
        idx = np.clip(np.searchsorted(cum, t, side="right") - 1, 0, len_a.size - 1)
        return idx, t - cum[idx]

    def centerline(s: FloatArray) -> PointArray:
        idx, loc = locate(s)
        ax = axis_a[idx]
        rel = _rodrigues(ax, kap_a[idx] * loc, p0_a[idx] - cen_a[idx])
        out: PointArray = cen_a[idx] + rel + (adv_a[idx] * loc)[:, None] * ax
        return out

    def tangent(s: FloatArray) -> PointArray:
        idx, loc = locate(s)
        return _rodrigues(axis_a[idx], kap_a[idx] * loc, w0_a[idx])

    def frame_up(s: FloatArray) -> PointArray:
        idx, loc = locate(s)
        ax = axis_a[idx]
        th = kap_a[idx] * loc
        return spin(_rodrigues(ax, th, u0_a[idx]), _rodrigues(ax, th, w0_a[idx]),
                    rol_a[idx] * loc)

    return Path(centerline, tangent, total, breaks, frame_up)


#: What a sweep accepts for ``up``: one constant world direction, one direction per
#: station, or -- what a walked path hands over -- a callable of normalized arc length
#: returning one per station.
UpSpec = Union[Vec3, Sequence[float], PointArray, Callable[[FloatArray], PointArray]]

#: The frame generators :func:`frames.sweep_placements
#: <nekmeshpy.core.frames.sweep_placements>` knows.
Orientation = Literal["transport", "fixed", "frenet"]


def sample_up(up: UpSpec | None, t: FloatArray) -> PointArray | UpSpec | None:
    """Evaluate a callable ``up`` on the station parameters ``t``, and pass anything else
    through untouched -- the one place a path's own frame becomes the ``(K,3)`` field
    :func:`frames.sweep_placements <nekmeshpy.core.frames.sweep_placements>` consumes."""
    if not callable(up):
        return up
    U: PointArray = np.asarray(up(t), dtype=float)
    if U.shape != (t.shape[0], 3):
        raise ValueError("sweep: a callable up must map the (%d,) station lattice to a "
                         "(%d,3) array of directions, got %s"
                         % (t.shape[0], t.shape[0], (U.shape,)))
    return U


def resolve_frame(path: Path, orientation: Orientation | None,
                  up: UpSpec | None) -> tuple[Orientation, UpSpec | None]:
    """The ``(orientation, up)`` a sweep along ``path`` should run with.

    A path built by :func:`walk` carries its own frame, and with no ``orientation`` asked
    for that frame *is* the answer -- held per station, which is what ``"fixed"`` means.
    An explicit ``orientation`` (or an explicit ``up``) overrides it, because a caller who
    names a generator is asking for that generator's frame, not the path's."""
    if orientation is not None:
        if path.up is not None and up is None:
            _log.debug("sweep: orientation=%r overrides the frame the path carries",
                       orientation)
        return orientation, up
    if up is not None or path.up is None:
        return "transport", up
    return "fixed", path.up


__all__ = ["Arc", "Helix", "LENGTH_TOL", "Line", "Move", "Orientation", "Path",
           "UpSpec", "arc", "helix", "line", "resolve_frame", "sample_up", "walk"]
