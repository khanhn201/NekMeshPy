"""Two parallel manifolds joined by ``N_BR`` hairpin tube connectors -- a chain of
welded quadrant T-junctions along each main pipe, with every branch stub lengthened
and bent 180 degrees to reach the opposite pipe instead of ending in an open cap.

Run with::

    PYTHONPATH=. python examples/chimera.py

Produces ``chimera.re2`` and ``chimera.vtu``.

Each junction, on its own
-----------------------

Every individual junction is built one of two ways, selected by the module constant
``TOPOLOGY``.

``TOPOLOGY = "quadrant"`` (the default) builds it exactly as in
``quadrant_pipe_tjunction.py``: every cross-section is a full disc of four
``quadrant_ogrid`` blocks, one quadrant of the main pipe *is* a quadrant of the branch,
and the two crotches (the corner regions where three quadrant faces and a curved
triangle of wall meet at the junction centre ``O``) are filled by ``HexMesh.tetra`` as
octants of a 3-D O-grid. See that module's docstring for the full decomposition -- it
is unchanged here.

``TOPOLOGY = "half_ogrid"`` instead follows ``circular_pipe_tjunction.py``'s recipe:
two saddle points ``A1``/``A2`` on the branch--main intersection curve
(``footprint(t = +-90 deg)``, verified algebraically to lie on both cylinders even
though ``R_MAIN != R_BRANCH``) are shared corners of all three legs' seam rings, so
there is no centre point ``O`` and no crotch caps -- every cross-section is instead a
single :meth:`QuadMesh.spined_ogrid <nekmeshpy.quadmesh.QuadMesh.spined_ogrid>` split
along the ``A1 -> A2`` chord. ``SEAM_OFFSET`` (degrees) rotates ``A1``/``A2`` off their
default symmetric branch angle, moving the seam's radiating fan pattern off-centre
(matching a reference image of an asymmetric junction) rather than leaving it at the
geometric default. Because ``R_MAIN`` and ``R_BRANCH`` differ, the two saddle points
sit close together in ``phi`` on the main cylinder (near the branch, not diametrically
opposite it), so one seam-ring half (the far wall) is much longer than the other (the
branch collar); the O-grid tolerates that imbalance -- it stays watertight, conformal,
and free of inverted elements -- but its scaled Jacobian is noticeably worse than the
quadrant topology's (order-N smoothing does not help: it is corner-node-only, see
``nekmeshpy.quadmesh.smoothing``). This mode is exercised manually, not by
``tests/test_pipes.py::test_chimera``, which -- like the checked-in default -- always
runs ``TOPOLOGY = "quadrant"``.

Chaining junctions
------------------

All of that per-junction geometry -- the footprint curves, the seams, the two
composite junction faces -- is defined in a local frame centred on the junction's own
``O`` at ``z = 0`` and never refers to a global position. So a junction is built once
in that local frame and moved into place with :meth:`HexMesh.translate
<nekmeshpy.hexmesh.HexMesh.translate>`, a rigid motion that leaves every relative
distance -- and hence every weld the junction already carries internally -- untouched.

Each junction's two legs no longer run all the way to the main pipe's end: instead
each takes an explicit run length, from its transition's plain disc to wherever it
should stop. Interior junctions stop halfway to their neighbour; the two end junctions
run out to the true inlet/outlet planes. The plain disc a leg's composite face morphs
into depends only on ``(z, sign)`` -- not on which junction's curves fed the morph --
so two neighbouring junctions' facing leg ends land on identical points at their shared
midpoint even though each junction is built independently. ``HexMesh.merge``'s
tolerance weld closes that seam into one conformal component, exactly as it already
welds the ring-closure point within a single disc.

Hairpins to a second pipe
-------------------------

A second main pipe never gets its own geometry. Rotating one whole finished pipe by
``pi`` about a vertical line is a rigid, orientation-preserving map -- unlike a mirror,
which :meth:`HexMesh.transform <nekmeshpy.hexmesh.HexMesh.transform>`'s own docstring
warns inverts every element's Jacobian -- so :func:`HexMesh.rotate
<nekmeshpy.hexmesh.HexMesh.rotate>` turns pipe A bodily into pipe B: same chain of
junctions, same inlet/outlet tags, but shifted to ``x = -D_PIPES`` with every branch
now facing ``-x`` to meet it, exactly the reference photo's two-header hairpin-tube
bank.

``branch()`` no longer ends at an open, tagged cap: past the round opening it grows a
short straight arm (``HexMesh.extrude``), left untagged because it becomes an interior
face once a bend welds onto it. ``bend(z0)`` carries that arm's ending disc along a
declarative turtle-walk path (the same pattern ``examples/serpentine_pipe.py`` uses for
its coil) -- a 180 degree U-turn through ``+y``, a straight run ``LOOP_LEN`` back
toward ``-x``, and a second 180 degree U-turn through ``-y`` that restores heading
``+x``. ``LOOP_LEN`` is solved so that landing point is exactly pipe B's own (rotated)
arm end, offset purely along ``-x`` rather than diagonally, so ``HexMesh.merge``'s
tolerance weld closes each hairpin without any bespoke geometry for the "return" side.
"""

import logging
from collections import namedtuple

import numpy as np

from nekmeshpy import export, hexmesh, linemesh, quadmesh
from nekmeshpy.model.interp import coons_grid_fn as coons_fn
from nekmeshpy.model.paths import turtle_path

logging.basicConfig(level=logging.INFO, format="%(message)s")

# -- parameters ---------------------------------------------------------------
R_MAIN = 1.2                 # main pipe radius (axis = z)
R_BRANCH = 0.5               # branch pipe radius (axis = x), deliberately < R_MAIN
Z_NEAR = 1.2                 # where a leg has finished morphing to a plain disc
H_BRANCH = 8.0               # branch opening plane, x = H_BRANCH

N_QUAD = 3                   # cells per quadrant half-arc; a quadrant spans 2*N_QUAD
RADIAL = np.array([0.0, 0.45, 0.8, 1.0])   # O-ring positions, core perimeter -> wall
CENTER_SCALE = 0.55          # core corner at CENTER_SCALE * R along the arc midpoint

#: ``quadrant`` topology only: weight of the branch-facing arc (``ab`` in every
#: :func:`cap` call) when locating each crotch's wall-triangle "tip" -- see
#: ``quadrant_pipe_tjunction.py``'s own ``CAP_TIP_BIAS``. ``1/3`` is the plain
#: centroid of the three arc midpoints; raise toward ``1.0`` to pull the tip closer
#: to the branch. Keep inside ``(0, 1)``.
CAP_TIP_BIAS = 1.0 / 3.0

PHI_W = np.deg2rad(85)    # bypass edge: the two z = 0 wall corners, at +-PHI_W

N_TRANS = 5                  # layers from a leg's composite face to its plain disc
N_LEG = 6                    # reference layer count -- sets the axial cell target
N_BRANCH = 3                 # layers in the branch, footprint -> opening

ORDER = 2                    # exact at any order; see quadrant_pipe_tjunction.py

#: Which T-junction topology to build every junction with. ``"quadrant"`` (default) is
#: the four-``quadrant_ogrid``-block decomposition above, unchanged. ``"half_ogrid"``
#: instead splits every cross-section into two ``QuadMesh.spined_ogrid`` halves along a
#: single seam -- see the "Two junction topologies" docstring section.
TOPOLOGY = "quadrant"
#: ``half_ogrid`` only: degrees, rotates the seam's two saddle points off their default
#: symmetric position (+-90 degrees of branch angle). 0 = the geometric seam; nonzero
#: offsets it, matching a reference image of an asymmetric radiating seam pattern. Keep
#: within (-90, 90) so both saddle points stay past the branch opening.
SEAM_OFFSET = 0.0
N_HALF = 8                   # half_ogrid only: cells per seam half-arc, MULTIPLE OF 4

N_BR = 7                    # number of T-junctions along the main pipe
BR_SPACING = 5.0             # centre-to-centre spacing between junctions (> 2*Z_NEAR)
END_MARGIN = 2.5             # outermost junction -> inlet/outlet plane (> Z_NEAR)
#: Target hex length along a leg's straight run, set by the single-junction reference
#: geometry (``L_MAIN - Z_NEAR`` there over ``N_LEG`` layers) so a chain segment grades
#: to roughly the same element size regardless of how long the segment is.
AXIAL_CELL = (END_MARGIN - Z_NEAR) / N_LEG

ARM_LEN = 15.0 - H_BRANCH     # straight run from the branch opening to the bend
X_MID = H_BRANCH + ARM_LEN    # x of the arm's ending point -- the hairpin's start
D_PIPES = 15.0                 # centre-to-centre offset, pipe A to pipe B, along x only
R_BEND = 5.0                  # radius of each of the hairpin's two U-turns
#: Straight run between the two U-turns, forced by ``D_PIPES``: the hairpin starts at
#: ``x = X_MID`` heading ``+x``, so after two 180 degree U-turns (net heading restored
#: to ``+x``) it lands ``LOOP_LEN`` further ``-x``, at ``x = X_MID - LOOP_LEN``. Pipe
#: B's own arm reaches ``X_MID`` further ``-x`` still from its axis, so pipe B's axis
#: sits at ``2*X_MID - LOOP_LEN``; solving that for ``-D_PIPES`` gives this.
LOOP_LEN = 2.0 * X_MID + D_PIPES
N_ARM = 12                     # layers in the straight arm
BEND_CELL = ARM_LEN / N_ARM   # target hex length along the hairpin, matching the arm
N_BEND = max(1, round(R_BEND * np.pi / BEND_CELL))    # layers around each U-turn
N_LOOP = max(1, round(LOOP_LEN / BEND_CELL))          # layers along the straight run

OUT_NAME = "chimera"
GROUPS = {"wall": "W  ", "inlet": "v  ", "outlet": "O  "}

N = N_QUAD
ORIGIN = np.zeros(3)

#: The seam sampling ``quadrant_ogrid`` demands -- the same for every seam, because
#: every block shares ``N_QUAD`` / ``RADIAL`` / ``CENTER_SCALE``.
FR = quadmesh.quadrant_seam_fractions(N_QUAD, RADIAL, CENTER_SCALE)

#: Branch polar angles of the four footprint corners, measured from ``+z`` (the main
#: axis) and **descending**, which is the winding whose normal points along the
#: branch.  Quadrant 0 (``45 -> -45``) faces the ``+z`` leg and quadrant 2
#: (``-135 -> -225``) the ``-z`` leg, exactly as the reference decomposes it.
TQ = np.deg2rad(45.0 - 90.0 * np.arange(5))


# -- curves -------------------------------------------------------------------
def footprint(t):
    """The branch--main intersection curve at branch polar angle ``t``.

    A cylinder of radius ``Rb`` about ``x`` meets one of radius ``Rm`` about ``z``
    where ``y = Rb sin t``, ``z = Rb cos t``, ``x = sqrt(Rm**2 - y**2)`` -- a closed
    form, so every node sits on the true curve."""
    t = np.asarray(t, dtype=float)
    y = R_BRANCH * np.sin(t)
    z = R_BRANCH * np.cos(t)
    return np.stack([np.sqrt(R_MAIN**2 - y**2), y, z], axis=1)


def opening(t):
    """The branch's circular opening at ``x = H_BRANCH``, on the same parameter.

    ``footprint`` and ``opening`` share their ``y``/``z`` component exactly, so a
    :meth:`LineMesh.blend <nekmeshpy.linemesh.LineMesh.blend>` between them moves only
    ``x`` and every intermediate node stays on the true branch cylinder."""
    t = np.asarray(t, dtype=float)
    return np.stack([np.full(t.shape, H_BRANCH),
                     R_BRANCH * np.sin(t), R_BRANCH * np.cos(t)], axis=1)


def cyl(phi, z):
    """Points on the main cylinder from their ``(phi, z)`` surface parameters."""
    phi, z = np.asarray(phi, dtype=float), np.asarray(z, dtype=float)
    return np.stack([R_MAIN * np.cos(phi), R_MAIN * np.sin(phi),
                     np.broadcast_to(z, phi.shape)], axis=-1)


def cyl_params(p):
    """The inverse: ``(phi, z)`` of points already on the main cylinder."""
    p = np.asarray(p, dtype=float)
    return np.stack([np.arctan2(p[..., 1], p[..., 0]), p[..., 2]], axis=-1)


#: A wall curve, carried as its **surface parametrization** rather than as points:
#: ``g`` maps a ``(K,)`` array of curve parameters to the ``(K, 2)`` ``(phi, z)`` it
#: passes through, and ``fr`` is the ``2*N_QUAD+1`` parameter values of its nodes.
Wall = namedtuple("Wall", "g fr")


def cyl_pts(u):
    """``(K, 2)`` wall parameters to ``(K, 3)`` points."""
    return cyl(u[:, 0], u[:, 1])


def wall_mesh(w):
    """A :class:`Wall` as a ``LineMesh`` on the cylinder, exact at every node."""
    return linemesh.loft_fn(lambda x: cyl_pts(w.g(x)), w.fr, order=ORDER)


def ruled_wall(pa, pb):
    """The straight ``(phi, z)`` segment between two stations, on ``[0, 1]``."""
    pa, pb = np.asarray(pa, dtype=float), np.asarray(pb, dtype=float)

    def g(x):
        xi = np.asarray(x, dtype=float)[:, None]
        return (1.0 - xi) * pa + xi * pb

    return Wall(g, np.linspace(0.0, 1.0, 2 * N_QUAD + 1))


def foot_wall(fr):
    """A footprint quadrant, in the branch angle it is analytic in."""
    return Wall(lambda t: cyl_params(footprint(t)), np.asarray(fr, dtype=float))


def plain_wall(w, phi0, phi1, z):
    """The plain-disc arc ``phi0 -> phi1`` at height ``z``, reparametrized onto
    ``w``'s domain so the two can be blended station by station (see :func:`leg`)."""
    t0, t1 = float(w.fr[0]), float(w.fr[-1])

    def g(x):
        xi = (np.asarray(x, dtype=float) - t0) / (t1 - t0)
        return np.stack([phi0 + xi * (phi1 - phi0), np.full(xi.shape, z)], axis=1)

    return Wall(g, w.fr)


def blend_wall(w0, w1, lam):
    """The two curves interpolated **in parameter space**, hence still on the wall."""
    return Wall(lambda x: (1.0 - lam) * w0.g(x) + lam * w1.g(x), w0.fr)


def reverse_wall(w):
    """The same curve traversed the other way: ``loft_fn`` takes a descending
    parameter sequence for exactly this."""
    return Wall(w.g, w.fr[::-1])


def shift_wall(w, turns):
    """The same curve with ``phi`` shifted by whole turns -- the identical points,
    rebranched so that curves meeting at a corner can be blended in ``phi``."""
    d = np.array([2.0 * np.pi * turns, 0.0])
    return Wall(lambda x: w.g(x) + d, w.fr)


def seam(target, center=ORIGIN):
    """One of the radii ``O -> wall corner``, sampled where ``quadrant_ogrid`` wants
    its ``n+1 + Nradial`` seam points."""
    return linemesh.line(center, target, FR, order=ORDER)


# -- sections -----------------------------------------------------------------
def quadrant(arc, seam1, seam2, wall_tag=""):
    """One quadrant face.  Every quadrant in the mesh -- disc sections and crotch
    caps alike -- comes from here, so a cap shares its points with the leg or branch
    on the other side of it."""
    return quadmesh.quadrant_ogrid(arc, seam1, seam2, RADIAL,
                                   center_scale=CENTER_SCALE, wall_tag=wall_tag)


def disc(pieces):
    """A full-disc section: four ``(arc, seam1, seam2)`` quadrants merged."""
    return quadmesh.merge([quadrant(arc, s1, s2, wall_tag="wall")
                           for arc, s1, s2 in pieces])


def plain_walls(composite, z, sign):
    """The plain four-quadrant disc at height ``z`` that a leg morphs into: seams at
    ``+-45`` / ``+-135`` degrees so one quadrant faces the branch.  Depends only on
    ``(z, sign)`` -- not on ``composite``'s actual shape, only its parameter domain --
    which is what lets two independently built junctions' facing leg ends coincide."""
    ang = sign * np.deg2rad(-45.0 + 90.0 * np.arange(5))
    return [plain_wall(composite[q], ang[q], ang[q + 1], z) for q in range(4)]


if TOPOLOGY == "quadrant":
    # -- the junction geometry (local frame, O at the origin) ------------------
    #: ``P[q]`` is footprint corner ``q``: ``P1, P4, P3, P2`` in the winding's own
    #: order.
    P = [footprint(TQ[q:q + 1])[0] for q in range(4)]
    WP, WM = cyl(PHI_W, 0.0), cyl(-PHI_W, 0.0)          # the two bypass edge corners

    #: The parameter values of each footprint quadrant's nodes, even in arc length.
    FQ_FR = [linemesh.arclength_fractions(footprint, 2 * N_QUAD,
                                          t_range=(TQ[q], TQ[q + 1]))
             for q in range(4)]
    #: The footprint quadrant arcs: 0 = ``A`` faces ``+z``, 1 = ``D`` faces ``-y``,
    #: 2 = ``C`` faces ``-z``, 3 = ``B`` faces ``+y``.
    FQ = [linemesh.loft_fn(footprint, fr, order=ORDER) for fr in FQ_FR]

    #: ``(phi, z)`` of the four footprint corners and the two bypass edge corners.
    UP = [cyl_params(p) for p in P]
    UWP, UWM = np.array([PHI_W, 0.0]), np.array([-PHI_W, 0.0])

    #: The composite junction faces, as the four wall curves of each.
    TURN = np.array([2.0 * np.pi, 0.0])
    W_R = [foot_wall(FQ_FR[0][::-1]),                 # P4 -> P1, the welded quadrant
           ruled_wall(UP[0], UWP),                    # +y side: P1 -> W+
           ruled_wall(UWP, TURN - UWP),               # bypass:  W+ -> W- long way
           ruled_wall(TURN - UWP, UP[1] + TURN)]      # -y side: W- -> P4
    W_L = [foot_wall(FQ_FR[2][::-1]),                 # P2 -> P3, the welded quadrant
           ruled_wall(UP[2], UWM),                    # -y side: P3 -> W-
           ruled_wall(UWM, -TURN - UWM),              # bypass:  W- -> W+ long way
           ruled_wall(-TURN - UWM, UP[3] - TURN)]     # +y side: W+ -> P2

    SP = [seam(p) for p in P]                            # the four footprint radii
    SWP, SWM = seam(WP), seam(WM)                        # the two bypass edge radii

    SIDE_RP, SIDE_RM = wall_mesh(W_R[1]), wall_mesh(W_R[3])
    SIDE_LM, SIDE_LP = wall_mesh(W_L[1]), wall_mesh(W_L[3])
    BYPASS = wall_mesh(W_R[2])        # the shared leg-to-leg face, wound for +z

    COMPOSITE_R = disc([(linemesh.reverse(FQ[0]), SP[1], SP[0]),      # P4 -> P1
                        (SIDE_RP, SP[0], SWP),
                        (BYPASS, SWP, SWM),
                        (SIDE_RM, SWM, SP[1])])

    COMPOSITE_L = disc([(linemesh.reverse(FQ[2]), SP[3], SP[2]),      # P2 -> P3
                        (SIDE_LM, SP[2], SWM),
                        (linemesh.reverse(BYPASS), SWM, SWP),
                        (SIDE_LP, SWP, SP[3])])
else:
    # -- the junction geometry, half-ogrid alternative (local frame, z = 0) -----
    # Two saddle points A1/A2 -- footprint(t) at t = +-90 deg, offset by SEAM_OFFSET
    # -- are shared corners of all three legs' seam rings, so unlike the quadrant
    # topology above there is no centre point O and no crotch caps: the three
    # A1->A2 arcs (the far wall, and the branch collar's +z/-z halves) are each
    # built once and reused by every ring that touches them, exactly mirroring
    # circular_pipe_tjunction.py's equal-radius three-leg recipe.
    OFFSET = np.deg2rad(SEAM_OFFSET)
    TA1, TA2 = np.pi / 2 + OFFSET, -np.pi / 2 + OFFSET
    UA1 = cyl_params(footprint(np.array([TA1]))[0])
    UA2 = cyl_params(footprint(np.array([TA2]))[0])
    #: ``UA2`` moved a full turn so a linear (phi, z) ramp from ``UA1`` runs the
    #: *long* way around the main cylinder, through the far wall (phi = 180) and
    #: away from the branch opening (phi = 0).
    UA2_FAR = UA2 + np.array([2.0 * np.pi, 0.0]) if UA2[0] < UA1[0] else UA2

    def half_arc(t0, t1):
        """One ``A1 -> A2`` footprint half, arc-length-even, ``N_HALF + 1`` points
        -- the collar arc shared between the branch and one main leg."""
        fr = linemesh.arclength_fractions(footprint, N_HALF, t_range=(t0, t1))
        return linemesh.loft_fn(footprint, fr, order=ORDER)

    def join_arcs(p, q):
        """Two shared-endpoint ``A1 -> A2`` arcs into one closed ``2*N_HALF``-point
        ring, index 0 at ``A1``, index ``N_HALF`` at ``A2``."""
        return linemesh.merge([p, linemesh.reverse(q)])

    def wall_arc(pa, pb):
        """The straight ``(phi, z)`` segment ``pa -> pb`` on the main cylinder,
        sampled at ``half_arc``'s own ``N_HALF + 1`` points -- the far-wall
        analogue of ``ruled_wall``/``wall_mesh``, at a half-ogrid arc's point
        count rather than a quadrant's."""
        pa, pb = np.asarray(pa, dtype=float), np.asarray(pb, dtype=float)

        def f(x):
            xi = np.asarray(x, dtype=float)[:, None]
            return cyl_pts((1.0 - xi) * pa + xi * pb)

        return linemesh.loft_fn(f, np.linspace(0.0, 1.0, N_HALF + 1), order=ORDER)

    #: The branch collar, split at the two saddle points: ``A_PLUS`` via the ``+z``
    #: side (through ``t = OFFSET``), ``A_MINUS`` via the ``-z`` side. Both run
    #: ``A1 -> A2`` so ``join_arcs`` can reverse either uniformly.
    A_PLUS = half_arc(TA1, TA2)
    A_MINUS = half_arc(TA1, TA1 + np.pi)
    #: The far wall, away from the branch entirely -- shared by both legs' rings.
    A_WALL = wall_arc(UA1, UA2_FAR)

    SEAM_R = join_arcs(A_WALL, A_PLUS)     # +z leg: far wall + +z collar
    SEAM_L = join_arcs(A_WALL, A_MINUS)    # -z leg: far wall + -z collar
    SEAM_BRANCH = join_arcs(A_PLUS, A_MINUS)

    def opening_arc(t0, t1):
        """The branch opening's ``A1 -> A2`` half, on ``half_arc``'s own parameter
        -- ``opening`` is a plain circle, so uniform ``t`` is already arc-length
        even -- giving the same point count and winding as ``A_PLUS``/``A_MINUS``
        so ``SEAM_BRANCH`` blends into it index-for-index."""
        return linemesh.loft_fn(opening, np.linspace(t0, t1, N_HALF + 1),
                                order=ORDER)

    OPEN_PLUS = opening_arc(TA1, TA2)
    OPEN_MINUS = opening_arc(TA1, TA1 + np.pi)
    OPENING_LOOP = join_arcs(OPEN_PLUS, OPEN_MINUS)

    def opening_main(z0):
        """The plain main-pipe circle at height ``z0``, split into two ``phi(A1)
        -> phi(A2)`` halves -- the far way (matching ``A_WALL``) and the short way
        (matching the collar side) -- giving the same ``2*N_HALF`` point count and
        ``join_arcs`` structure as ``SEAM_R``/``SEAM_L`` so a leg's transition
        blends cleanly."""
        far = wall_arc((float(UA1[0]), z0), (float(UA2_FAR[0]), z0))
        near = wall_arc((float(UA1[0]), z0), (float(UA2[0]), z0))
        return join_arcs(far, near)


# -- the crotch caps ----------------------------------------------------------
def arc_mids(walls):
    """The ``(phi, z)`` mid-node of each of a crotch's three arcs -- the nodes the
    wall triangle splits at, taken from the arcs' own sampling rather than at the
    midpoint of the parameter range, which is not the same point on a graded arc."""
    return [w.g(w.fr[N:N + 1])[0] for w in walls]


def wall_patch(fn, tag):
    """One patch of the wall triangle, evaluated on the cylinder at every node."""
    fr = np.linspace(0.0, 1.0, N + 1)
    return quadmesh.loft_fn(
        lambda y: linemesh.loft_fn(
            lambda x: cyl_pts(fn(x, np.full(np.shape(x), y))), fr, order=ORDER),
        fr, order=ORDER, element_tags=[tag] * N)


def wall_triangle(w_ab, w_bc, w_ca, tag="wall", mids=None, tip_bias=CAP_TIP_BIAS):
    """The curved wall triangle between three arcs, as the three patches about its
    tip that :meth:`HexMesh.tetra <nekmeshpy.hexmesh.HexMesh.tetra>` wants.
    ``tip_bias`` is :data:`CAP_TIP_BIAS`'s weight on ``w_ab`` (always the branch arc,
    by this file's calling convention)."""
    u_ab, u_bc, u_ca = arc_mids((w_ab, w_bc, w_ca)) if mids is None else mids
    wc = tip_bias * u_ab + (1.0 - tip_bias) * 0.5 * (u_bc + u_ca)

    def half(w, i0, i1):
        """Half an arc on ``[0, 1]``, remapped **through its own node values**."""
        step = 1 if i1 > i0 else -1
        idx = np.arange(i0, i1 + step, step)
        fr, m = w.fr[idx], idx.size - 1

        def rep(s):
            s = np.clip(np.asarray(s, dtype=float), 0.0, 1.0) * m
            i = np.clip(np.floor(s).astype(int), 0, m - 1)
            return (1.0 - (s - i)) * fr[i] + (s - i) * fr[i + 1]

        return lambda s: w.g(rep(s))

    def spoke(mid):
        return lambda s: mid + np.asarray(s, dtype=float)[:, None] * (wc - mid)

    return quadmesh.merge([
        wall_patch(coons_fn(half(w_ab, 0, N), spoke(u_ca),
                            half(w_ca, 2 * N, N), spoke(u_ab)), tag),
        wall_patch(coons_fn(half(w_ab, 2 * N, N), spoke(u_bc),
                            half(w_bc, 0, N), spoke(u_ab)), tag),
        wall_patch(coons_fn(half(w_ca, 0, N), spoke(u_bc),
                            half(w_bc, 2 * N, N), spoke(u_ca)), tag)])


def cap(sa, sb, sc, ab, bc, ca, tip_bias=CAP_TIP_BIAS):
    """A crotch: the curvilinear tetrahedron whose four sides are the three quadrant
    faces meeting at ``O`` and the patch of pipe wall between their arcs. ``ab`` is
    always the branch's own footprint arc, which is what lets ``tip_bias``
    (:data:`CAP_TIP_BIAS`) pull the wall triangle's tip toward it."""
    (m_ab, w_ab), (m_bc, w_bc), (m_ca, w_ca) = ab, bc, ca
    mids = arc_mids((w_ab, w_bc, w_ca))
    u_ab, u_bc, u_ca = mids
    wc_param = tip_bias * u_ab + (1.0 - tip_bias) * 0.5 * (u_bc + u_ca)
    wc = cyl_pts(wc_param[None, :])[0]
    return hexmesh.tetra([quadrant(m_ab, sa, sb), quadrant(m_bc, sb, sc),
                          quadrant(m_ca, sc, sa),
                          wall_triangle(w_ab, w_bc, w_ca, mids=mids,
                                       tip_bias=tip_bias)],
                         center=ORIGIN + CENTER_SCALE * np.sqrt(1.5) * (wc - ORIGIN))


# -- build ----------------------------------------------------------------------
def leg(composite, walls, sign, run, end_tag):
    """One main-pipe leg: morph the composite junction face into a plain disc, then
    extrude straight over ``run`` to wherever this leg should stop -- the midpoint to
    the next junction, or the pipe's true inlet/outlet for an end junction."""
    z = sign * Z_NEAR
    w_plain = plain_walls(walls, z, sign)

    def station(s):
        return quadmesh.quadrant_disc(
            [wall_mesh(blend_wall(walls[q], w_plain[q], s)) for q in range(4)],
            np.array([0.0, 0.0, s * z]), RADIAL, center_scale=CENTER_SCALE,
            wall_tag="wall")

    plain = station(1.0)
    n_run = max(1, round(run / AXIAL_CELL))
    return [hexmesh.loft_fn(station, np.linspace(0.0, 1.0, N_TRANS + 1),
                            order=ORDER),
            hexmesh.extrude(plain, run, n_run, axis=(0.0, 0.0, float(sign)),
                            last_tag=end_tag)]


#: The branch's round opening disc, axis ``x``, centred at ``(H_BRANCH, 0, 0)`` --
#: shared by ``branch()``/``branch_half()`` (as its far cross-section) and every
#: junction's hairpin bend (as the disc the straight arm carries out to the bend, via
#: :meth:`QuadMesh.translate <nekmeshpy.quadmesh.QuadMesh.translate>`), so both start
#: from the identical points. Built once, per topology, exactly like the rest of the
#: junction geometry above; ``bend()`` below only needs it to be *a* valid disc.
if TOPOLOGY == "quadrant":
    OPEN_ARCS = [linemesh.loft_fn(opening, fr, order=ORDER) for fr in FQ_FR]
    C_OPEN = np.array([H_BRANCH, 0.0, 0.0])
    OPENING_DISC = quadmesh.quadrant_disc(OPEN_ARCS, C_OPEN, RADIAL,
                                          center_scale=CENTER_SCALE, wall_tag="wall")
else:
    OPENING_DISC = quadmesh.spined_ogrid(OPENING_LOOP, RADIAL,
                                         center_scale=CENTER_SCALE, wall_tag="wall")


def branch():
    """The branch stub: the footprint disc morphed to the circular opening, then a
    short straight arm out to where this junction's hairpin bend begins. Neither end
    is tagged -- both become interior faces once the bend welds onto the far one."""
    t = np.linspace(0.0, 1.0, N_BRANCH + 1)
    walls = [linemesh.blend(f, o, t) for f, o in zip(FQ, OPEN_ARCS)]
    sections = [quadmesh.quadrant_disc([w[i] for w in walls], t[i] * C_OPEN, RADIAL,
                                       center_scale=CENTER_SCALE, wall_tag="wall")
               for i in range(t.size)]
    return [hexmesh.loft(sections),
            hexmesh.extrude(OPENING_DISC, ARM_LEN, N_ARM, axis=(1.0, 0.0, 0.0))]


def leg_half(seam_ring, sign, run, end_tag, n_slices=N_TRANS):
    """One main-pipe leg, half-ogrid version: blend the seam ring into a plain
    disc, :meth:`QuadMesh.spined_ogrid <nekmeshpy.quadmesh.QuadMesh.spined_ogrid>`
    each station, loft the transition, then extrude straight over ``run`` -- the
    same two-piece transition/run split as :func:`leg`."""
    loops = linemesh.blend(seam_ring, opening_main(sign * Z_NEAR),
                           np.linspace(0.0, 1.0, n_slices + 1))
    stations = [quadmesh.spined_ogrid(loop, RADIAL, center_scale=CENTER_SCALE,
                                      wall_tag="wall") for loop in loops]
    plain = stations[-1]
    n_run = max(1, round(run / AXIAL_CELL))
    return [hexmesh.loft(stations),
            hexmesh.extrude(plain, run, n_run, axis=(0.0, 0.0, float(sign)),
                            last_tag=end_tag)]


def branch_half():
    """The branch stub, half-ogrid version: blend ``SEAM_BRANCH`` into the round
    opening at ``H_BRANCH``, ``spined_ogrid`` each station, loft, then the same
    straight ``ARM_LEN`` extrude as :func:`branch`."""
    loops = linemesh.blend(SEAM_BRANCH, OPENING_LOOP,
                           np.linspace(0.0, 1.0, N_BRANCH + 1))
    stations = [quadmesh.spined_ogrid(loop, RADIAL, center_scale=CENTER_SCALE,
                                      wall_tag="wall") for loop in loops]
    return [hexmesh.loft(stations),
            hexmesh.extrude(stations[-1], ARM_LEN, N_ARM, axis=(1.0, 0.0, 0.0))]


def build_junction(z0, run_minus, run_plus, tag_minus, tag_plus):
    """One full junction, built in the local frame above and moved to its place on
    the chain -- either the quadrant topology (two legs, the branch stub, both
    crotch caps) or the half-ogrid one (two legs and the branch stub, no caps)."""
    if TOPOLOGY == "quadrant":
        blocks = [*leg(COMPOSITE_R, W_R, 1, run_plus, tag_plus),
                  *leg(COMPOSITE_L, W_L, -1, run_minus, tag_minus),
                  *branch(),
                  # A crotch's three arcs must share one branch of phi, so the two
                  # that are authored a full turn away in a leg's unwrapped list
                  # are shifted back.
                  cap(SP[0], SP[3], SWP,                    # +y crotch: P1, P2, W+
                      (linemesh.reverse(FQ[3]), foot_wall(FQ_FR[3][::-1])),
                      (linemesh.reverse(SIDE_LP), shift_wall(reverse_wall(W_L[3]), 1)),
                      (linemesh.reverse(SIDE_RP), reverse_wall(W_R[1]))),
                  cap(SP[2], SP[1], SWM,                    # -y crotch: P3, P4, W-
                      (linemesh.reverse(FQ[1]), foot_wall(FQ_FR[1][::-1])),
                      (linemesh.reverse(SIDE_RM), shift_wall(reverse_wall(W_R[3]), -1)),
                      (linemesh.reverse(SIDE_LM), reverse_wall(W_L[1])))]
    else:
        blocks = [*leg_half(SEAM_R, 1, run_plus, tag_plus),
                  *leg_half(SEAM_L, -1, run_minus, tag_minus),
                  *branch_half()]
    return hexmesh.translate(hexmesh.merge(blocks), (0.0, 0.0, z0))


#: The hairpin path, in the plane ``z = z0``, as a declarative turtle walk exactly
#: like ``examples/serpentine_pipe.py``'s: starting at the arm's own ending point
#: ``(X_MID, 0)`` heading ``+x``, a 180 degree U-turn through ``+y``, a straight run
#: ``LOOP_LEN`` back toward ``-x``, and a second 180 degree U-turn through ``-y`` that
#: restores heading ``+x`` -- landing exactly on pipe B's own (rotated) arm end, so
#: pipe B ends up offset purely along ``-x``, not diagonally, with its arm facing
#: ``-x`` to meet it. Every move keeps the previous move's heading, so the path is C1
#: by construction.
_PATH = turtle_path([("arc", R_BEND, 180.0), ("line", LOOP_LEN, 0.0),
                    ("arc", R_BEND, 180.0)], start=(X_MID, 0.0), heading=0.0)
_BREAKS = _PATH.break_fractions

#: Sweep stations, one exactly on every straight<->arc junction, each of the three
#: pieces graded at its own layer count rather than one global linspace.
_STATION_FR = np.concatenate([
    np.linspace(0.0, _BREAKS[0], N_BEND + 1),
    np.linspace(_BREAKS[0], _BREAKS[1], N_LOOP + 1)[1:],
    np.linspace(_BREAKS[1], 1.0, N_BEND + 1)[1:]])

#: Pipe B's axis, at ``x = -D_PIPES`` (solved above); the rotation midway between it
#: and pipe A's own axis at the origin is what turns pipe A bodily into pipe B.
BEND_CENTER_X = -D_PIPES / 2.0


def bend(z0):
    """One junction's hairpin, at height ``z0``: the arm's ending disc carried along
    ``_PATH.centerline``/``.tangent``, landing on pipe B's own (rotated) arm end by
    construction -- no separate return arm is built."""
    def centerline(s):
        xy = _PATH.centerline(s)
        return np.concatenate([xy, np.full((xy.shape[0], 1), z0)], axis=1)

    def tangent(s):
        xy = _PATH.tangent(s)
        return np.concatenate([xy, np.zeros((xy.shape[0], 1))], axis=1)

    section = quadmesh.translate(OPENING_DISC, (ARM_LEN, 0.0, z0))
    return hexmesh.sweep(section, centerline, _STATION_FR,
                         tangent=tangent, orientation="fixed", up=(0.0, 0.0, 1.0),
                         origin=(X_MID, 0.0, z0))


#: Junction centres, evenly spaced and centred on ``z = 0``.
Z_J = (np.arange(N_BR) - (N_BR - 1) / 2.0) * BR_SPACING

junctions = []
for _i in range(N_BR):
    _run_minus = (END_MARGIN if _i == 0 else BR_SPACING / 2.0) - Z_NEAR
    _run_plus = (END_MARGIN if _i == N_BR - 1 else BR_SPACING / 2.0) - Z_NEAR
    _tag_minus = "inlet" if _i == 0 else ""
    _tag_plus = "outlet" if _i == N_BR - 1 else ""
    junctions.append(build_junction(Z_J[_i], _run_minus, _run_plus,
                                    _tag_minus, _tag_plus))

# pipe A, its own hairpin bends, and pipe B -- a rigid pi rotation of pipe A about
# BEND_CENTER_X (on the z axis, y = 0), which needs no new geometry: rotation
# preserves element orientation (unlike a mirror, which HexMesh.transform's own
# docstring warns inverts every element), so pipe B is a second complete pipe, offset
# purely along x, with branches facing away from pipe A's, for free.
pipe_a = hexmesh.merge(junctions)
pipe_b = hexmesh.rotate(pipe_a, np.pi, axis=(0.0, 0.0, 1.0), center=(BEND_CENTER_X, 0.0, 0.0))
bends = [bend(z0) for z0 in Z_J]

mesh = hexmesh.merge([pipe_a, pipe_b, *bends])

print(hexmesh.report(mesh))
print(hexmesh.topology_report(mesh))

export.to_re2(mesh, OUT_NAME + ".re2", groups=GROUPS)
export.to_vtu(mesh, OUT_NAME + ".vtu", groups=GROUPS)
