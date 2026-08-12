"""Serpentine (heat-exchanger coil) pipe: one O-grid disc swept along a bent path.

A single continuous round pipe folded into a planar serpentine: **8** vertical
passes joined by 7 semicircular 180 deg U-bends alternating bottom / top, with a
**hook** at each end (two opposed 90 deg bends and a short sideways jog) carrying
the inlet and outlet clear of the coil.  Planar in x-z (``y == 0``).

The coil is not symmetric top-to-bottom: passes 4 and 5 are ``RAISE`` longer than
their neighbours, so the wide middle bridge (radius ``U_R_MID``) that joins the two
half-coils sits **above** the two flanking hairpins (radius ``U_R``) rather than
level with them, and passes 1 and 8 are lengthened to match.

**The coil's shape is traced from a reference photo and is fixed** -- it is only
ever placed, never reshaped or rescaled.  ``chimera_full.py`` imports ``MOVES`` and
``TARGET_LEN`` straight off this module (guarded below, so that import costs nothing
beyond the two names) and sweeps its own copy between the branches of each pair of
T2 junctions rather than standing it alone the way this script does; the two are the
same physical part, so the numbers are defined here once rather than in both.

The centerline is a declarative **turtle walk** of straight runs and circular arcs:
every move starts at the previous move's end point *and* keeps its heading, so the
path is C1 by construction -- no fillet fitting, no corner rounding, no numerical
inversion anywhere.  The cumulative length table is accumulated from exact closed
forms (a straight's length, ``radius * angle`` for an arc), so the arc-length
parametrization ``s in [0, 1]`` is exact to machine precision.

Measured facts of the path as shipped: total length 1238.823001646924; 23 segments
(12 straights + 11 arcs); turns
``[+90, -90, -180, +180, -180, +180, -180, +180, -180, -90, +90]``; min bend radius
2.5 = ``5 * R_PIPE``; min non-adjacent self-distance 4.806 vs a tube diameter of
``2 * R_PIPE = 1.0`` -- so the coil never touches itself.

:meth:`HexMesh.sweep` carries the cross-section along that curve by a moving
orthonormal frame, placing it at every station by a *rigid* motion -- the curved
generalization of :meth:`HexMesh.extrude`.

    PYTHONPATH=. python examples/serpentine_pipe.py

Produces ``serpentine_pipe.re2`` and ``serpentine_pipe.vtu``.
"""

import logging
import time

import numpy as np

from nekmeshpy import export, hexmesh, linemesh, quadmesh
from nekmeshpy.core import paths
from nekmeshpy.core.fields import uniform_spacing
from nekmeshpy.core.paths import turtle_path

logging.basicConfig(level=logging.INFO, format="%(message)s")

# -- the coil's own fixed shape ------------------------------------------------
# Traced from a reference photo; do not reshape or rescale (see the module
# docstring).  ``R_PIPE`` is this script's own cross-section radius, not shared --
# ``chimera_full.py`` sizes its tube from the disc it is handed instead.
R_PIPE = 0.5      # tube radius -- the swept cross-section, not the path
PASS_LEN = 136.0  # length of a full vertical pass
U_R = 2.5         # tight U-turn radius: bottom turns + top hairpins
U_R_MID = 4.0     # wider radius of the raised middle bridge
R_HOOK = U_R_MID  # the two end hooks turn at the same radius
HOOK_JOG = 5.0    # the hook's short sideways step
HOOK_DROP = 20.0  # the hook's straight run out to the inlet / outlet
RAISE = 4.0       # extra length on passes 1/4/5/8 -- what lifts the middle bridge
                  # above the two flanking hairpins

#: The two end hooks, each the other's time reversal (reverse the order, negate every
#: turn) -- which is what lands both openings on the same ``v`` facing the same way.
HOOK_IN = [("line", HOOK_DROP, 0.0), ("arc", R_HOOK, 90.0),
           ("line", HOOK_JOG, 0.0), ("arc", R_HOOK, -90.0)]
HOOK_OUT = [("arc", R_HOOK, -90.0), ("line", HOOK_JOG, 0.0),
            ("arc", R_HOOK, 90.0), ("line", HOOK_DROP, 0.0)]

#: ``("line", length, 0.0)`` or ``("arc", radius, signed turn in degrees)``; a positive
#: turn is counter-clockwise in the ``(u, v)`` plane.  8 vertical passes joined by 7
#: semicircular 180-degree U-bends alternating bottom / top, hooked at both ends.
#:
#: The coil is deliberately **not** symmetric top to bottom: passes 4 and 5 are ``RAISE``
#: longer than their neighbours, so the wide middle bridge (``U_R_MID``) that joins the
#: two half-coils sits *above* the two flanking hairpins (``U_R``) rather than level with
#: them, and passes 1 and 8 are lengthened to match.
MOVES = (HOOK_IN
    + [("line", PASS_LEN + RAISE, 0.0), ("arc", U_R, -180.0)]      # pass 1 -> bottom
    + [("line", PASS_LEN, 0.0), ("arc", U_R, 180.0)]               # pass 2 -> top hairpin
    + [("line", PASS_LEN, 0.0), ("arc", U_R, -180.0)]              # pass 3 -> bottom
    + [("line", PASS_LEN + RAISE, 0.0), ("arc", U_R_MID, 180.0)]   # 4 -> RAISED bridge
    + [("line", PASS_LEN + RAISE, 0.0), ("arc", U_R, -180.0)]      # pass 5 -> bottom
    + [("line", PASS_LEN, 0.0), ("arc", U_R, 180.0)]               # pass 6 -> top hairpin
    + [("line", PASS_LEN, 0.0), ("arc", U_R, -180.0)]              # pass 7 -> bottom
    + [("line", PASS_LEN + RAISE, 0.0)]                            # pass 8
    + HOOK_OUT)

#: Target hex length along the sweep.  NOT cubic: the coil is slender (a pass is 272
#: tube radii long), so ~1.6*R_PIPE would cost ~100k hexes.  The real floor is the
#: tightest turn -- ``sweep_fractions`` rounds a segment's length/target to the NEAREST
#: station count, so a target near a U-turn's own arc length (``pi * U_R`` = 7.85) rounds
#: down to ONE station spanning the whole 180 degrees: two opposed sections lerped into
#: a near-zero-volume hex.  2.0 puts 4 stations in that turn; 6.0 or 8.0 put 1.
TARGET_LEN = 2.0

# -- everything below here is this script's own mesh build, guarded so that
# ``from serpentine_pipe import MOVES, TARGET_LEN`` (``chimera_full.py``) costs
# nothing beyond the coil data above -- an ordinary Python import runs a module
# top to bottom regardless of which names the caller wants, so without the guard
# every import would also build and export this script's own standalone mesh.
if __name__ == "__main__":
    # -- path parameters -------------------------------------------------------
    # The turtle walks in its own ``(u, v)``: ``u`` is the pass direction, ``v`` the
    # pass-to-pass stacking.  Mapping u -> world +z and v -> world -x reproduces
    # chimera_full.py's own placement of this coil (and the reference photo's
    # orientation, in which "up" is z-).
    AXIS_U = np.array([0.0, 0.0, 1.0])   # turtle +u -> world +z (along a pass)
    AXIS_V = np.array([-1.0, 0.0, 0.0])  # turtle +v -> world -x (pass to pass)
    ORIGIN = np.array([0.0, 0.0, 0.0])   # 3-D image of the in-plane origin
    PLANE_NORMAL = (0.0, 1.0, 0.0)       # the coil is planar: y is the plane normal

    # -- mesh parameters --------------------------------------------------------
    N_SIDE = 5                   # central square block cells per side (loop = 4*N_SIDE pts)
    N_RADIAL = 3                 # O-ring layers out to the wall
    CENTER_SCALE = 0.5
    # TARGET_LEN above is a property of this coil's own geometry (the tightest
    # turn), not of how any one script meshes it.
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

    # -- turtle-walk the move table, then lift it onto the coil's own plane -------
    # paths.embed maps the walk's own (u, v) to origin + u*AXIS_U + v*AXIS_V, and gives
    # back the vectorized centerline/tangent pair HexMesh.sweep wants: it samples the
    # whole node lattice in one call so a moving frame can be built along it.
    #
    # The tangent is the **analytic** derivative, and worth the twenty lines it costs
    # turtle_path: without it sweep differences the sampled path, which is only O(h**2)
    # and -- because this path's curvature jumps at every junction -- worst exactly where
    # a straight meets an arc.  Each frame inherits that tilt and the section stops being
    # perpendicular to the path.  Measured on this coil: the wall drifts 1.1e-4 inside
    # R_PIPE (0.2% of the tube radius) with differenced tangents, and with these it lands
    # on the true tube to 4e-11 -- the measurement floor of the dense nearest-point probe,
    # not a residual of the mesh.  (embed is also what keeps ORIGIN out of the tangent: a
    # tangent is a direction, and translating it would tilt every frame.)
    _WALK = turtle_path(MOVES, start=(0.0, 0.0), heading=0.0)
    PATH = paths.embed(_WALK, u=AXIS_U, v=AXIS_V, origin=ORIGIN)
    S_BREAKS = PATH.break_fractions
    TOTAL = PATH.total_length

    # -- sweep stations: a node exactly on every junction ------------------------
    # The path's curvature is piecewise constant and jumps at every straight<->arc
    # junction, so an element that straddled one would be fitted across two different
    # geometries. target_length subdivides each piece on its own at ~TARGET_LEN, so
    # every junction is reproduced exactly rather than approached by a global linspace.
    FRACTIONS = linemesh.path_fractions(PATH, target_length=TARGET_LEN)

    assert np.all(np.diff(FRACTIONS) > 0.0), "sweep stations must be strictly increasing"
    assert np.isin(S_BREAKS, FRACTIONS).all(), "every junction must carry a sweep station"

    # -- the cross-section: an O-grid disc normal to the path's start tangent -----
    START = PATH.centerline(np.array([0.0]))[0]
    # read the opening's own heading off the path rather than restating it: the
    # hook decides which way the inlet points, and a stale literal here would tilt
    # the whole cross-section without failing any assertion.
    START_TANGENT = tuple(PATH.tangent(np.array([0.0]))[0])
    # tag the wall at the lowest level (the boundary loop); ogrid copies it onto the
    # outer ring and the sweep carries it onto the side faces. ogrid has no order=
    # kwarg -- the order is inherited from the loop, which must carry exactly 4*N_SIDE
    # points so the wall is meshed exactly.
    section = quadmesh.ogrid(
        linemesh.circle(R_PIPE, 4 * N_SIDE, center=START, normal=START_TANGENT,
                        element_tag="wall", order=ORDER),
        N_SIDE, uniform_spacing(N_RADIAL),
        center_scale=CENTER_SCALE, smoothing_method=SMOOTHING_METHOD)

    # -- sweep it along the coil --------------------------------------------------
    # The path is planar, so orientation="fixed" against the plane normal is exact,
    # torsion-free and independent of the sampling density. "frenet" would be wrong
    # here: the path is C1 but not C2 -- curvature jumps 0 <-> 1/U_R at every
    # junction and the Frenet normal is undefined on the straights. origin= names the
    # circle's centre, because the O-grid's centroid misses it slightly. tangent= hands
    # in the analytic derivative so the sections stay exactly perpendicular (see above).
    _t0 = time.perf_counter()
    mesh = hexmesh.sweep_path(section, PATH, fractions=FRACTIONS,
                              orientation="fixed", up=PLANE_NORMAL, origin=START,
                              first_tag="inlet", last_tag="outlet")
    BUILD_SECONDS = time.perf_counter() - _t0

    # -- checks -------------------------------------------------------------------
    assert hexmesh.is_watertight(mesh), "the swept coil must be a single watertight block"
    assert hexmesh.is_conforming(mesh), "the swept coil must be conforming"
    assert set(mesh.face_group_tags) == {"wall", "inlet", "outlet"}, \
        "boundary groups must be exactly wall/inlet/outlet, got %s" % (
            list(mesh.face_group_tags),)
    stats = hexmesh.quality_summary(mesh)
    # Through the tightest (U_R = 5*R_PIPE) turn the inner wall traverses 4/5 of the
    # outer arc length, so the elements are graded across the tube -- but the bend
    # radius comfortably exceeds it, so nothing folds through the axis. Check rather
    # than assume.
    assert stats.min > 0.0, "inverted element: min scaled Jacobian %g" % stats.min

    # -- report + export ----------------------------------------------------------
    print("serpentine pipe: %d hex elements, %d points, order %d"
          % (mesh.n_hexes, mesh.n_points, mesh.order))
    _ARC_RADII = [m[1] for m in MOVES if m[0] == "arc"]
    print("path: L=%.12f, %d segments (%d arcs), %d sweep stations, min bend radius %g"
          % (TOTAL, len(MOVES), len(_ARC_RADII), FRACTIONS.size - 1, min(_ARC_RADII)))
    print("scaled Jacobian: min=%.4f mean=%.4f" % (stats.min, stats.mean))
    print("build time: %.2f s" % BUILD_SECONDS)

    export.to_re2(mesh, OUT_NAME + ".re2", groups=GROUPS)
    export.to_vtu(mesh, OUT_NAME + ".vtu", groups=GROUPS)  # XML: renders curved cells
    print("groups:", ", ".join(mesh.face_group_tags))
