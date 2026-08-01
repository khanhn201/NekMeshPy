"""Serpentine (heat-exchanger coil) pipe: one O-grid disc swept along a bent path.

A single continuous round pipe folded into a planar serpentine -- ``N_PASSES``
vertical passes joined by ``N_PASSES - 1`` semicircular 180 deg U-bends alternating
bottom / top, with the first and last pass carried up over the arches by a **Z-offset**
(two opposed 90 deg bends) to tall inlet / outlet risers.  Left-right symmetric and
entirely in the x-z plane (``y == 0``).

The centerline is a declarative **turtle walk** of straight runs and circular arcs:
every move starts at the previous move's end point *and* keeps its heading, so the
path is C1 by construction -- no fillet fitting, no corner rounding, no numerical
inversion anywhere.  The cumulative length table is accumulated from exact closed
forms (a straight's length, ``radius * angle`` for an arc), so the arc-length
parametrization ``s in [0, 1]`` is exact to machine precision.

Measured facts of the path as shipped: total length 15.441150082346; 23 segments
(12 straights + 11 arcs); turns
``[-90, +90, +180, -180, +180, -180, +180, -180, +180, +90, -90]``; min bend radius
0.15 = ``3 * R_PIPE``; min non-adjacent self-distance 0.22267 vs a tube diameter of
``2 * R_PIPE = 0.10`` -- so the coil never touches itself.

:meth:`HexMesh.sweep` carries the cross-section along that curve by a moving
orthonormal frame, placing it at every station by a *rigid* motion -- the curved
generalization of :meth:`HexMesh.extrude`.

    PYTHONPATH=. python examples/serpentine_pipe.py

Produces ``serpentine_pipe.re2`` and ``serpentine_pipe.vtu``.
"""

import logging
import time
from collections import namedtuple

import numpy as np

from nekmeshpy import HexMesh, LineMesh, QuadMesh, export
from nekmeshpy.model.fields import uniform_spacing

logging.basicConfig(level=logging.INFO, format="%(message)s")

# -- path parameters ---------------------------------------------------------
R_PIPE = 0.05                 # pipe (tube) radius -- the swept cross-section
N_PASSES = 8                  # vertical straight runs, alternating down/up
PITCH = 0.30                  # horizontal pass-to-pass spacing (= 6 * R_PIPE)
R_BEND = PITCH / 2.0          # U-bend semicircle radius; PITCH/2 is forced by geometry
Z_BOT = 0.0                   # centerline level all four bottom bends turn around at
Z_TOP_OUTER = 1.00            # turn-around level of the flanking top bends (2 and 6)
Z_TOP_INNER = 1.35            # turn-around level of the middle top bend (4) -- taller
Z_STEP = 1.75                 # height of the horizontal run of the inlet/outlet Z-offset
Z_PLANE = 2.40                # inlet / outlet plane (top of the risers)
R_ZOFF = 0.15                 # radius of the four 90 deg Z-offset bends (= R_BEND)
DX_OFFSET = 0.45              # horizontal step of the Z-offset, inward from the end pass
AXIS_U = np.array([1.0, 0.0, 0.0])   # in-plane "right" axis (x)
AXIS_V = np.array([0.0, 0.0, 1.0])   # in-plane "up" axis (z); the normal is y
ORIGIN = np.array([0.0, 0.0, 0.0])   # 3-D image of the in-plane origin
PLANE_NORMAL = (0.0, 1.0, 0.0)       # the coil is planar: y is the plane normal

# -- mesh parameters ---------------------------------------------------------
N_SIDE = 5                   # central square block cells per side (loop = 4*N_SIDE pts)
N_RADIAL = 3                 # O-ring layers out to the wall
CENTER_SCALE = 0.5
TARGET_LEN = 0.08            # target hex length along the sweep (~1.6 * R_PIPE, so the
                             # elements are roughly cubic; ~16k hexes overall)
ORDER = 2                    # polynomial order; 1 = linear. Both smoothers stay off:
                             # a repositioning smoother moves corner nodes only and
                             # rejects order > 1. sweep evaluates the path at the
                             # intermediate GLL levels too, so the bend geometry is
                             # exact along the sweep at any order (the .vtu renders
                             # true arcs; the .re2 stays linear either way).
SMOOTHING_METHOD = "bilinear"   # no-op section fill -- allowed at any order
OUT_NAME = "serpentine_pipe"

# boundary name -> Nek BC code, applied only at export
GROUPS = {"wall": "W  ", "inlet": "v  ", "outlet": "O  "}

# -- derived placement: passes are centred on u = 0, so the coil is symmetric --
X_PASS_0 = -0.5 * (N_PASSES - 1) * PITCH      # u of pass 1 (leftmost)
Z_ZOFF_START = Z_STEP - R_ZOFF                # where pass 1 stops and the Z-offset begins
L_RISER = Z_PLANE - (Z_STEP + R_ZOFF)         # straight run up to the inlet/outlet plane
L_ZRUN = DX_OFFSET - 2.0 * R_ZOFF             # horizontal run between the two 90 deg bends

# the 9 turn-around levels bounding the 8 passes (pass k runs LEVELS[k] -> LEVELS[k+1])
LEVELS = (Z_ZOFF_START, Z_BOT, Z_TOP_OUTER, Z_BOT, Z_TOP_INNER, Z_BOT, Z_TOP_OUTER,
          Z_BOT, Z_ZOFF_START)

# -- the move table: ("line", length, 0.0) or ("arc", radius, signed turn in deg) --
# a positive turn is counter-clockwise in the (u, v) plane
MOVES = [
    ("line", L_RISER, 0.0),         # down from the inlet plane
    ("arc", R_ZOFF, -90.0),         # Z-offset: turn from -v to -u
    ("line", L_ZRUN, 0.0),          # horizontal step outward, onto pass 1
    ("arc", R_ZOFF, +90.0),         # turn back to -v
]
for k in range(N_PASSES):
    MOVES.append(("line", abs(LEVELS[k + 1] - LEVELS[k]), 0.0))
    if k < N_PASSES - 1:
        # U-bends alternate sense: bottom bends CCW (+180), top bends CW (-180)
        MOVES.append(("arc", R_BEND, +180.0 if k % 2 == 0 else -180.0))
MOVES += [
    ("arc", R_ZOFF, +90.0),         # outlet Z-offset: turn from +v to -u (inward)
    ("line", L_ZRUN, 0.0),
    ("arc", R_ZOFF, -90.0),         # turn back to +v
    ("line", L_RISER, 0.0),         # up to the outlet plane
]

# -- turtle-walk the table into per-segment arrays ---------------------------
# One row per segment. A line is described by its start point + unit direction, an arc
# by its centre, radius, start angle and signed sweep; each fills the other's fields
# with harmless dummies (a line gets radius 1, an arc a zero direction) so both closed
# forms can be evaluated unconditionally and selected with np.where.
Seg = namedtuple("Seg", "is_arc p0 dir cen rad th0 dth length")

_segs = []
_p = np.array([X_PASS_0 + DX_OFFSET, Z_PLANE])   # inlet, above and inward of pass 1
_phi = -0.5 * np.pi                              # heading straight down
for _kind, _a, _b in MOVES:
    if _kind == "line":
        _d = np.array([np.cos(_phi), np.sin(_phi)])
        _segs.append(Seg(False, _p.copy(), _d, np.zeros(2), 1.0, 0.0, 0.0, _a))
        _p = _p + _a * _d
    else:
        _r, _turn = _a, float(np.deg2rad(_b))
        _sgn = 1.0 if _turn > 0.0 else -1.0
        # the centre lies to the left of the heading for a CCW (positive) turn
        _nrm = np.array([np.cos(_phi + _sgn * 0.5 * np.pi),
                         np.sin(_phi + _sgn * 0.5 * np.pi)])
        _c = _p + _r * _nrm
        _t0 = float(np.arctan2(_p[1] - _c[1], _p[0] - _c[0]))
        # the length is exact: radius * angle
        _segs.append(Seg(True, _p.copy(), np.zeros(2), _c, _r, _t0, _turn,
                         _r * abs(_turn)))
        _p = _c + _r * np.array([np.cos(_t0 + _turn), np.sin(_t0 + _turn)])
        _phi = _phi + _turn

IS_ARC = np.asarray([s.is_arc for s in _segs], dtype=bool)
SEG_P0 = np.asarray([s.p0 for s in _segs], dtype=float)
SEG_DIR = np.asarray([s.dir for s in _segs], dtype=float)
SEG_CEN = np.asarray([s.cen for s in _segs], dtype=float)
SEG_RAD = np.asarray([s.rad for s in _segs], dtype=float)
SEG_TH0 = np.asarray([s.th0 for s in _segs], dtype=float)
SEG_DTH = np.asarray([s.dth for s in _segs], dtype=float)
SEG_LEN = np.asarray([s.length for s in _segs], dtype=float)
CUM = np.concatenate([np.zeros(1), np.cumsum(SEG_LEN)])   # (S+1,) cumulative length
TOTAL = float(CUM[-1])                                    # exact total arc length
S_BREAKS = CUM[1:-1] / TOTAL      # normalized s of every straight<->arc junction


def locate(s):
    """Dispatch normalized arc lengths onto ``(segment index, arc length into it)``.

    ``(K,)`` in ``[0, 1]`` -> two ``(K,)`` arrays.  Both closed forms below start
    here: a ``searchsorted`` of the cumulative table is what puts each sample in its
    own segment, so nothing is ever evaluated with a neighbour's geometry.
    """
    t = np.clip(np.asarray(s, dtype=float).ravel(), 0.0, 1.0) * TOTAL
    idx = np.clip(np.searchsorted(CUM, t, side="right") - 1, 0, SEG_LEN.size - 1)
    return idx, t - CUM[idx]


def in_plane(uv):
    """``(K,2)`` in-plane components to ``(K,3)``, on the coil's own plane axes."""
    return uv[:, 0, None] * AXIS_U + uv[:, 1, None] * AXIS_V


def centerline(s):
    """Points on the serpentine centerline at normalized arc length ``s``.

    Vectorized -- ``(K,)`` in ``[0, 1]`` -> ``(K,3)`` -- because that is what
    :meth:`HexMesh.sweep` wants: it samples the whole node lattice in one call so a
    moving frame can be built along it.
    """
    idx, loc = locate(s)
    line_uv = SEG_P0[idx] + SEG_DIR[idx] * loc[:, None]
    theta = SEG_TH0[idx] + np.sign(SEG_DTH[idx]) * loc / SEG_RAD[idx]
    arc_uv = SEG_CEN[idx] + SEG_RAD[idx][:, None] * np.stack(
        [np.cos(theta), np.sin(theta)], axis=1)
    return ORIGIN + in_plane(np.where(IS_ARC[idx][:, None], arc_uv, line_uv))


def tangent(s):
    """Unit tangents of the centerline at normalized arc length ``s``.

    The **analytic** derivative, and worth the twenty lines: without it
    :meth:`HexMesh.sweep` differences the sampled path, which is only ``O(h**2)``
    and -- because this path's curvature jumps at every junction -- worst exactly
    where a straight meets an arc.  Each frame inherits that tilt and the section
    stops being perpendicular to the path.  Measured on this coil: the wall drifts
    ``1.1e-4`` inside ``R_PIPE`` (0.2% of the tube radius) with differenced tangents,
    and with these it lands on the true tube to ``4e-11`` -- which is the measurement
    floor of the dense nearest-point probe, not a residual of the mesh.

    Differentiating the two closed forms of ``centerline`` w.r.t. arc length: a
    straight's is its own constant direction; an arc's is the radius vector turned
    a quarter turn in the direction of travel.  Both are already unit length.
    """
    idx, loc = locate(s)
    sgn = np.sign(SEG_DTH[idx])
    theta = SEG_TH0[idx] + sgn * loc / SEG_RAD[idx]
    arc_uv = sgn[:, None] * np.stack([-np.sin(theta), np.cos(theta)], axis=1)
    # a direction, not a point -- ORIGIN deliberately does not enter here
    return in_plane(np.where(IS_ARC[idx][:, None], arc_uv, SEG_DIR[idx]))


# -- sweep stations: a node exactly on every junction ------------------------
# The path's curvature is piecewise constant and jumps at every straight<->arc
# junction, so an element that straddled one would be fitted across two different
# geometries. sweep_fractions subdivides each piece on its own at ~TARGET_LEN, so
# every junction is reproduced exactly rather than approached by a global linspace.
FRACTIONS = LineMesh.sweep_fractions(S_BREAKS * TOTAL, TOTAL, TARGET_LEN)

assert np.all(np.diff(FRACTIONS) > 0.0), "sweep stations must be strictly increasing"
assert np.isin(S_BREAKS, FRACTIONS).all(), "every junction must carry a sweep station"

# -- the cross-section: an O-grid disc normal to the path's start tangent -----
START = centerline(np.array([0.0]))[0]
START_TANGENT = (0.0, 0.0, -1.0)     # the inlet riser runs straight down
# tag the wall at the lowest level (the boundary loop); ogrid copies it onto the
# outer ring and the sweep carries it onto the side faces. ogrid has no order=
# kwarg -- the order is inherited from the loop, which must carry exactly 4*N_SIDE
# points so the wall is meshed exactly.
section = QuadMesh.ogrid(
    LineMesh.circle(R_PIPE, 4 * N_SIDE, center=START, normal=START_TANGENT,
                    element_tags=["wall"] * (4 * N_SIDE), order=ORDER),
    N_SIDE, uniform_spacing(N_RADIAL),
    center_scale=CENTER_SCALE, smoothing_method=SMOOTHING_METHOD)

# -- sweep it along the coil --------------------------------------------------
# The path is planar, so orientation="fixed" against the plane normal is exact,
# torsion-free and independent of the sampling density. "frenet" would be wrong
# here: the path is C1 but not C2 -- curvature jumps 0 <-> 1/R_BEND at every
# junction and the Frenet normal is undefined on the straights. origin= names the
# circle's centre, because the O-grid's centroid misses it slightly. tangent= hands
# in the analytic derivative so the sections stay exactly perpendicular (see above).
_t0 = time.perf_counter()
mesh = HexMesh.sweep(section, centerline, FRACTIONS, tangent=tangent,
                     orientation="fixed", up=PLANE_NORMAL, origin=START,
                     first_tag="inlet", last_tag="outlet")
BUILD_SECONDS = time.perf_counter() - _t0

# -- checks -------------------------------------------------------------------
assert mesh.is_watertight(), "the swept coil must be a single watertight block"
assert mesh.is_conforming(), "the swept coil must be conforming"
assert set(mesh.boundary_group_tags) == {"wall", "inlet", "outlet"}, \
    "boundary groups must be exactly wall/inlet/outlet, got %s" % (
        list(mesh.boundary_group_tags),)
stats = mesh.quality_summary()
# Through a 3*R_PIPE bend the inner wall traverses 2/3 of the outer arc length, so
# the elements are strongly graded across the tube -- but the bend radius exceeds
# the tube radius, so nothing folds through the axis. Check it rather than assume it.
assert stats.min > 0.0, "inverted element: min scaled Jacobian %g" % stats.min

# -- report + export ----------------------------------------------------------
print("serpentine pipe: %d hex elements, %d points, order %d"
      % (mesh.n_hexes, mesh.n_points, mesh.order))
print("path: L=%.12f, %d segments (%d arcs), %d sweep stations, min bend radius %g"
      % (TOTAL, SEG_LEN.size, int(IS_ARC.sum()), FRACTIONS.size - 1,
         SEG_RAD[IS_ARC].min()))
print("scaled Jacobian: min=%.4f mean=%.4f" % (stats.min, stats.mean))
print("build time: %.2f s" % BUILD_SECONDS)

export.to_re2(mesh, OUT_NAME + ".re2", groups=GROUPS)
export.to_vtu(mesh, OUT_NAME + ".vtu", groups=GROUPS)  # XML: renders curved cells
print("groups:", ", ".join(mesh.boundary_group_tags))
