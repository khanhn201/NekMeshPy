"""``N_COPIES`` chimera units chained end to end along ``z``, alternating which of a
unit's two manifolds -- pipe A or pipe B -- carries the connection to the next unit.

Run with::

    PYTHONPATH=. python examples/chimera.py

Produces ``chimera.re2`` and ``chimera.vtu``.

One chimera unit
----------------

Every unit is two parallel manifolds joined by ``N_BR`` hairpin tube connectors: a
chain of ``N_BR`` welded quadrant T-junctions along pipe A, pipe B built for free as
``hexmesh.rotate(pipe_a, pi, ...)`` -- a rigid rotation, unlike a mirror, which would
invert every element's Jacobian -- and a hairpin ``hexmesh.sweep`` bend per junction
carrying each of pipe A's branch stubs round to its opposite number on pipe B.

Each junction is built as in ``quadrant_pipe_tjunction.py``: every cross-section is a
full disc of four ``quadrant_ogrid`` blocks, one quadrant of the main pipe *is* a
quadrant of the branch, and the two crotches -- where three quadrant faces and a
curved triangle of wall meet at the junction centre ``O`` -- are filled by
``hexmesh.tetra`` as octants of a 3-D O-grid. See that module's docstring for the full
decomposition; it is unchanged here.

Chaining units
--------------

Consecutive units are translated so unit ``k``'s plus end lands exactly on unit
``k + 1``'s minus end (``L_HALF``, the same ``Z_J``/``END_MARGIN`` half-length a lone
``chimera.py`` run already uses to size its own pipe), and joined the same way two
neighbouring junctions already weld within one unit: ``HexMesh.merge``'s tolerance
weld, no new geometry.

At every inter-unit link exactly one pipe is the connector -- alternating pipe A,
pipe B, pipe A, ... down the chain -- and the two pipes are genuinely built to
different lengths there, not just tagged differently at the same face: the
connector's facing ends meet flush at the link's ``z`` plane and weld away (``""``,
the tag that already welds a junction's own leg-to-leg seams into invisible interior
faces), while the other pipe stops ``GAP`` short of that plane and is capped
``"wall"`` right there, so there really are no elements at all between it and its
neighbour. Each unit is already a single connected component on its own (its hairpin
bends tie pipe A to pipe B), so the chain stays one component as long as *some* pipe
connects every pair of neighbours, which the alternation guarantees.

That is why the *outermost* stretch of each unit's two end legs (past the last/first
junction) is pulled out of the shared junction construction: it is precisely the
stretch whose length must differ between the connector pipe (reaching the joint) and
the dead-end pipe (stopping ``GAP`` short). :func:`build_chimera` builds the shared
interior chain with both ends left bare (:func:`leg`'s ``run <= 0`` case), then
attaches four independent end stubs -- one per (pipe, end) -- each its own
``HexMesh.extrude`` of the right length and end tag. A pipe B stub is built in pipe
A's own local frame and carried over by the same rotation that turns pipe A into
pipe B (:func:`to_b`), so the branch/crotch geometry is still never derived twice.

There is exactly one ``"inlet"`` -- pipe A's minus end on the first unit -- and
exactly one ``"outlet"``: pipe B's plus end on the last unit does not stop there at
all, but folds back through :func:`outlet_return`'s U-turn and runs alongside the
whole chain until it reaches the inlet's own ``z``, becoming ``"outlet"`` there
instead. The two ends this leaves stranded -- pipe A's plus end on the last unit,
pipe B's minus end on the first -- are capped ``"wall"`` at the same full
(non-gapped) length an ``"inlet"``/``"outlet"`` end would have used.
"""

import logging
from collections import namedtuple

import numpy as np

from nekmeshpy import export, hexmesh, linemesh, quadmesh
from nekmeshpy.model.interp import coons_grid_fn as coons_fn
from nekmeshpy.model.paths import turtle_path

logging.basicConfig(level=logging.INFO, format="%(message)s")

# -- parameters ---------------------------------------------------------------
R_MAIN = 1.2                  # main pipe radius (axis = z)
R_BRANCH = 0.5                # branch pipe radius (axis = x), deliberately < R_MAIN
Z_NEAR = 1.2                  # where a leg has finished morphing to a plain disc
H_BRANCH = 4.0                # branch opening plane, x = H_BRANCH

N_QUAD = 2                    # cells per quadrant half-arc; a quadrant spans 2*N_QUAD
RADIAL = np.array([0.0, 0.6, 1.0])   # O-ring positions, core perimeter -> wall
CENTER_SCALE = 0.7           # core corner at CENTER_SCALE * R along the arc midpoint

#: Weight of the branch-facing arc when locating each crotch's wall-triangle "tip" --
#: see ``quadrant_pipe_tjunction.py``'s own ``CAP_TIP_BIAS``.
CAP_TIP_BIAS = 1.0 / 3.0

PHI_W = np.deg2rad(100.0)     # bypass edge: the two z = 0 wall corners, at +-PHI_W

N_TRANS = 5                   # layers from a leg's composite face to its plain disc
N_LEG = 6                     # reference layer count -- sets the axial cell target
N_BRANCH = 4                  # layers in the branch, footprint -> opening

ORDER = 2                     # exact at any order; see quadrant_pipe_tjunction.py

N_BR = 7                      # number of T-junctions along one unit's main pipe
BR_SPACING = 5.0               # centre-to-centre spacing between junctions (> 2*Z_NEAR)
END_MARGIN = 2.5                # outermost junction -> unit's own end plane (> Z_NEAR + GAP)
GAP = 1.0                      # physical space left between a dead-end cap and the
                                # neighbouring unit's joint plane -- no elements fill it
#: Target hex length along a leg's straight run, set by the single-junction reference
#: geometry (see ``chimera.py``) so a chain segment grades to roughly the same
#: element size regardless of how long the segment is.
AXIAL_CELL = (END_MARGIN - Z_NEAR) / N_LEG

ARM_LEN = 15.0 - H_BRANCH        # straight run from the branch opening to the bend
X_MID = H_BRANCH + ARM_LEN     # x of the arm's ending point -- the hairpin's start
D_PIPES = 15.0                  # centre-to-centre offset, pipe A to pipe B, along x only
R_BEND = 5.0                   # radius of each of the hairpin's two U-turns
#: Straight run between the two U-turns, forced by ``D_PIPES`` -- see ``chimera.py``.
LOOP_LEN = 2.0 * X_MID + D_PIPES
N_ARM = 15                      # layers in the straight arm
BEND_CELL = ARM_LEN / N_ARM    # target hex length along the hairpin, matching the arm
N_BEND = max(1, round(R_BEND * np.pi / BEND_CELL))    # layers around each U-turn
N_LOOP = max(1, round(LOOP_LEN / BEND_CELL))          # layers along the straight run

RETURN_BEND_R = 3.3             # radius of the last copy's pipe-A return U-turn

N_COPIES = 5                    # number of chimera units chained along z

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


# -- the junction geometry (local frame, O at the origin) ----------------------
#: ``P[q]`` is footprint corner ``q``: ``P1, P4, P3, P2`` in the winding's own order.
P = [footprint(TQ[q:q + 1])[0] for q in range(4)]
WP, WM = cyl(PHI_W, 0.0), cyl(-PHI_W, 0.0)          # the two bypass edge corners

SP = [seam(p) for p in P]                            # the four footprint radii
SWP, SWM = seam(WP), seam(WM)                        # the two bypass edge radii

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
def leg(composite, walls, sign, run):
    """One main-pipe leg: morph the composite junction face into a plain disc, then
    extrude straight over ``run`` -- the midpoint to the next junction.

    Every leg here is interior to its unit, so none is ever tagged. ``run <= 0``
    stops at the plain disc itself, leaving the leg bare: what an outermost
    junction's outward-facing side uses, since the straight run past it is pulled
    out into an independent :func:`end_stub` built to its own length."""
    z = sign * Z_NEAR
    w_plain = plain_walls(walls, z, sign)

    def station(s):
        return quadmesh.quadrant_disc(
            [wall_mesh(blend_wall(walls[q], w_plain[q], s)) for q in range(4)],
            np.array([0.0, 0.0, s * z]), RADIAL, center_scale=CENTER_SCALE,
            wall_tag="wall")

    transition = hexmesh.loft_fn(station, np.linspace(0.0, 1.0, N_TRANS + 1),
                                 order=ORDER)
    if run <= 0.0:
        return [transition]

    n_run = max(1, round(run / AXIAL_CELL))
    return [transition,
            hexmesh.extrude(station(1.0), run, n_run,
                            axis=(0.0, 0.0, float(sign)))]


def end_disc(sign):
    """The plain four-quadrant disc a unit's outermost junction bares its ``sign``
    side to -- rebuilt independently of :func:`leg` (:func:`plain_walls` is a pure
    function of ``(walls, z, sign)``) so it lands bit-identically on that bare face
    and welds to it with zero extra geometric risk."""
    walls = W_R if sign > 0 else W_L
    z = sign * Z_NEAR
    face = quadmesh.quadrant_disc([wall_mesh(w) for w in plain_walls(walls, z, sign)],
                                  np.array([0.0, 0.0, z]), RADIAL,
                                  center_scale=CENTER_SCALE, wall_tag="wall")
    return quadmesh.translate(face, (0.0, 0.0, Z_J[-1] if sign > 0 else Z_J[0]))


def end_stub(sign, tag, run):
    """The independent straight run past a unit's outermost junction, on the
    ``sign`` end, tagged ``tag`` at its far face."""
    n_run = max(1, round(run / AXIAL_CELL))
    return hexmesh.extrude(end_disc(sign), run, n_run,
                           axis=(0.0, 0.0, float(sign)), last_tag=tag)


def outlet_return():
    """Pipe B's true outlet, only at the chain's last copy: a single U-turn toward
    ``+y`` (radius :data:`RETURN_BEND_R`) starting at the bare plain disc
    ``end_disc(1)``, folding the pipe back to run parallel to the whole chain until
    it reaches the same global ``z`` as the chain's own ``"inlet"`` -- where it
    becomes the new ``"outlet"``. Built in pipe A's own local frame, exactly like
    :func:`end_disc` itself, and carried onto pipe B by the caller's :func:`to_b`."""
    z0 = Z_J[-1] + Z_NEAR
    run = (Z_J[-1] - Z_J[0]) + Z_NEAR + END_MARGIN + 2.0 * L_HALF * (N_COPIES - 1)
    path = turtle_path([("arc", RETURN_BEND_R, -180.0), ("line", run, 0.0)],
                       start=(z0, 0.0), heading=0.0)
    n_bend = max(1, round(RETURN_BEND_R * np.pi / BEND_CELL))
    n_run = max(1, round(run / AXIAL_CELL))
    breaks = path.break_fractions
    station_fr = np.concatenate([np.linspace(0.0, breaks[0], n_bend + 1),
                                 np.linspace(breaks[0], 1.0, n_run + 1)[1:]])

    def centerline(s):
        pq = path.centerline(s)
        return np.stack([np.zeros(pq.shape[0]), pq[:, 1], pq[:, 0]], axis=1)

    def tangent(s):
        pq = path.tangent(s)
        return np.stack([np.zeros(pq.shape[0]), pq[:, 1], pq[:, 0]], axis=1)

    return hexmesh.sweep(end_disc(1), centerline, station_fr, tangent=tangent,
                         orientation="fixed", up=(1.0, 0.0, 0.0),
                         origin=(0.0, 0.0, z0), last_tag="outlet")


def to_b(m):
    """Carry a block built in pipe A's own frame onto pipe B, by the rotation that
    turns the whole unit's pipe A into pipe B."""
    return hexmesh.rotate(m, np.pi, axis=(0.0, 0.0, 1.0), center=(BEND_CENTER_X, 0.0, 0.0))


#: The branch's round opening disc, axis ``x``, centred at ``(H_BRANCH, 0, 0)`` --
#: shared by ``branch()`` (as its far cross-section) and every junction's hairpin
#: bend (as the disc the straight arm carries out to the bend).
OPEN_ARCS = [linemesh.loft_fn(opening, fr, order=ORDER) for fr in FQ_FR]
C_OPEN = np.array([H_BRANCH, 0.0, 0.0])
OPENING_DISC = quadmesh.quadrant_disc(OPEN_ARCS, C_OPEN, RADIAL,
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


def build_junction(z0, run_minus, run_plus):
    """One full junction, built in the local frame above and moved to its place on
    the chain: two legs, the branch stub, and both crotch caps. Nothing here is
    tagged -- every open face it leaves behind is welded shut by a neighbouring
    junction, a hairpin bend or an :func:`end_stub`."""
    blocks = [*leg(COMPOSITE_R, W_R, 1, run_plus),
              *leg(COMPOSITE_L, W_L, -1, run_minus),
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


#: Junction centres within one unit, evenly spaced and centred on ``z = 0``.
Z_J = (np.arange(N_BR) - (N_BR - 1) / 2.0) * BR_SPACING


def build_chimera(a_minus, a_plus, b_minus, b_plus, *, b_plus_return=False):
    """One full chimera unit -- pipe A, its rotation pipe B, and every hairpin bend
    between them -- centred on its own local ``z = 0``. Each argument is one open
    end's ``(tag, run)`` pair from :func:`end_spec`, so a dead-end and a
    through-connection at the same nominal joint genuinely differ in how far they
    reach, rather than only in how they are tagged. ``b_plus_return`` swaps pipe
    B's plain ``+z`` stub for :func:`outlet_return`'s U-turn, for the chain's last
    copy only -- ``b_plus`` is then unused.

    The junction chain is built once with both its outermost ends left bare
    (:func:`leg`'s ``run <= 0`` case), and pipe B still comes for free as a rotation
    of pipe A. Only then are the four end stubs added -- pipe A's directly, pipe B's
    built in pipe A's own frame and carried over by :func:`to_b` -- so this still
    never derives the branch/crotch geometry twice."""
    mid_run = BR_SPACING / 2.0 - Z_NEAR
    pipe_a = hexmesh.merge([
        build_junction(Z_J[i], 0.0 if i == 0 else mid_run,
                       0.0 if i == N_BR - 1 else mid_run)
        for i in range(N_BR)])
    pipe_b = to_b(pipe_a)
    bends = [bend(z0) for z0 in Z_J]
    b_plus_stub = to_b(outlet_return()) if b_plus_return else to_b(end_stub(1, *b_plus))
    stubs = [end_stub(-1, *a_minus), end_stub(1, *a_plus),
             to_b(end_stub(-1, *b_minus)), b_plus_stub]
    return hexmesh.merge([pipe_a, pipe_b, *bends, *stubs])


# -- the chain of copies --------------------------------------------------------
#: One unit's half-length along z (its own ``Z_J``/``END_MARGIN`` span) -- unit ``k``
#: is placed at ``2 * L_HALF * k`` so its minus end exactly coincides with unit
#: ``k - 1``'s plus end.
L_HALF = (N_BR - 1) / 2.0 * BR_SPACING + END_MARGIN


def end_spec(is_extreme, extreme_name, is_connector):
    """The ``(tag, run)`` for one (pipe, end) pair of one unit: the chain's own
    inlet/outlet name at an extreme end, run out the full ``END_MARGIN - Z_NEAR``;
    otherwise ``""`` at that same full length (welds into the neighbouring unit) if
    this pipe is this link's connector, else ``"wall"`` stopped ``GAP`` short of the
    joint plane -- genuine empty space, not a coincident closed cap."""
    full = END_MARGIN - Z_NEAR
    if is_extreme:
        return extreme_name, full
    if is_connector:
        return "", full
    return "wall", full - GAP


copies = []
for _k in range(N_COPIES):
    _is_first, _is_last = _k == 0, _k == N_COPIES - 1
    # Which pipe connects unit _k to its neighbour on each side, alternating down
    # the chain: link j (between units j and j + 1) connects at pipe A if j is
    # even, pipe B if j is odd.
    _connector_before = "A" if _k % 2 == 0 else "B"
    _connector_after = "A" if (_k - 1) % 2 == 0 else "B"

    _unit = build_chimera(end_spec(_is_first, "inlet", _connector_before == "A"),
                          end_spec(_is_last, "wall", _connector_after == "A"),
                          end_spec(_is_first, "wall", _connector_before == "B"),
                          end_spec(_is_last, "outlet", _connector_after == "B"),
                          b_plus_return=_is_last)
    copies.append(hexmesh.translate(_unit, (0.0, 0.0, 2.0 * L_HALF * _k)))

mesh = hexmesh.merge(copies)

print(hexmesh.report(mesh))
print(hexmesh.topology_report(mesh))

export.to_re2(mesh, OUT_NAME + ".re2", groups=GROUPS)
export.to_vtu(mesh, OUT_NAME + ".vtu", groups=GROUPS)
