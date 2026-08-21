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

Each junction is built by ``tjunction_lib.build_tjunction`` -- the same construction
``quadrant_pipe_tjunction.py`` uses as its own reference caller: every cross-section
is a full disc of four ``quadrant_ogrid`` blocks, one quadrant of the main pipe *is* a
quadrant of the branch, and the two crotches -- where three quadrant faces and a
curved triangle of wall meet at the junction centre ``O`` -- are filled by
``hexmesh.tetra`` as octants of a 3-D O-grid. See that module's docstring for the full
decomposition. :func:`build_junction` extends it with what this file needs beyond the
bare junction -- the branch's straight arm out to the hairpin, and (interior to a
unit) each leg's own run to the next junction.

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
neighbour. That lid is skinned like any other no-slip surface, which moves it ``T_BL``
further out -- a rounding error against a ``GAP`` of 1. Each unit is already a single connected component on its own (its hairpin
bends tie pipe A to pipe B), so the chain stays one component as long as *some* pipe
connects every pair of neighbours, which the alternation guarantees.

That is why the *outermost* stretch of each unit's two end legs (past the last/first
junction) is pulled out of the shared junction construction: it is precisely the
stretch whose length must differ between the connector pipe (reaching the joint) and
the dead-end pipe (stopping ``GAP`` short). :func:`build_chimera` builds the shared
interior chain with both ends left bare (:func:`build_junction`'s ``run <= 0``
case), then
attaches four independent end stubs -- one per (pipe, end) -- each its own
``HexMesh.extrude`` of the right length and end tag. A pipe B stub is built in pipe
A's own local frame and carried over by the same rotation that turns pipe A into
pipe B (:func:`to_b`), so the branch/crotch geometry is still never derived twice.

The solid jacket
----------------

Each hairpin's **straight run** -- the ``LOOP_LEN`` stretch between its two U-turns,
the only part of the loop that is parallel to nothing but itself -- carries a solid
block welded onto the outboard half of the tube wall: in cross-section, the region
between the pipe's own semicircle and half of a rectangle ``BR_SPACING`` tall.

The block overhangs the run by ``SOLID_EXT`` at each end. The tube is already turning
there, so over an overhang the semicircle no longer bounds fluid at all -- it bounds
an empty cylindrical void, named ``"insulated"`` like any other external face of the
jacket, which the departing tube passes through without ever touching the material
around it (:func:`solid_run` has the argument). That is why the jacket is three
extrusions and not one.

The two regions are ``element_tags`` -- ``"fluid"`` and ``"solid"``, every element
carrying exactly one, so ``hexmesh.select(mesh, "solid")`` and its complement
partition the mesh. That vocabulary is deliberately disjoint from the ``face_tags``
in ``GROUPS`` (``"wall"`` / ``"inlet"`` / ``"outlet"`` / ``"insulated"``), which are
boundary conditions: the two tables are different slots on the container and a name
shared between them would only read as if they were the same thing.

The height is the point. Junction ``i``'s jacket spans ``z in Z_J[i] +- BR_SPACING/2``
and junctions are ``BR_SPACING`` apart, so consecutive jackets meet face to face and
weld into one continuous slab -- and because ``END_MARGIN`` happens to equal
``BR_SPACING / 2``, a unit's slab reaches exactly its own end plane and the slab runs
unbroken across the whole chain too. (Those two must stay equal: a larger
``END_MARGIN`` would leave a gap between units, a smaller one would make neighbouring
slabs *overlap*.)

Fluid and solid are built independently and joined by one stated
``hexmesh.attach`` seam at the very end, after the fluid has been skinned: the jacket
sits against the *finished* tube wall, and where that wall ends up is where
``quadmesh.offset``'s averaged normals put it, not on an analytic circle. So the two
halves correspond node for node -- :data:`SOLID_FR` reads the branch angles off the
junction's own port disc rather than resampling them -- but they do not agree bit for
bit, and a stated seam is exactly the join that does not need them to. The section is
built in the arm's frame on the ``-y`` side and carried onto the straight run by
:func:`to_run`, the same half turn :func:`bend`'s sweep applies over its first U-turn;
that is what puts the jacket **outboard** (``+y``), the only side of the run that is
clear of the hairpin's own U-turns.

There is exactly one ``"inlet"`` -- pipe A's minus end on the first unit -- and
exactly one ``"outlet"``: pipe B's plus end on the last unit does not stop there at
all, but folds back through :func:`outlet_return`'s U-turn and runs alongside the
whole chain until it reaches the inlet's own ``z``, becoming ``"outlet"`` there
instead. The two ends this leaves stranded -- pipe A's plus end on the last unit,
pipe B's minus end on the first -- are capped ``"wall"`` at the same full
(non-gapped) length an ``"inlet"``/``"outlet"`` end would have used.
"""

import logging
import os
import sys

import numpy as np

from nekmeshpy import hexmesh, linemesh, quadmesh, writer
from nekmeshpy.core import paths
from nekmeshpy.core.tags import ElementTags
from nekmeshpy.hexmesh import Seam
from nekmeshpy.pointmesh import PointMesh

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tjunction_lib import build_cob, skin_wall  # noqa: E402  (needs the path above)

logging.basicConfig(level=logging.INFO, format="%(message)s")

# -- parameters ---------------------------------------------------------------
R_MAIN = 1.2                  # main pipe radius (axis = z), *finished* -- see BL
R_BRANCH = 0.5                # branch pipe radius (axis = x), deliberately < R_MAIN
Z_NEAR = 1.2                  # where a junction's own block ends and a plain run begins
H_BRANCH = 4.0                # branch opening plane, x = H_BRANCH

#: Main pipe azimuthal cells, radial layers and hub placement.  These three are not
#: free: they fix the **port pattern**, and ``chimera_full.py`` has to meet it.  A cob
#: port is the plain pipe section, so at 16/4-per-side/2 it comes out 48 quads over 57
#: points with `quads`/`orient`/`lines` *byte-identical* to ``build_eqtee``'s
#: ``spined_ogrid`` disc at ``n_half=8`` -- both bottom out in ``quadrant_ogrid``
#: quarters -- and ``CENTER_SCALE_MAIN`` is T1's own 0.75 so the two patterns agree in
#: shape as well as in connectivity, not merely in count.  At 24 the port is 84 quads and
#: nothing in that file can pair with it.
N_THETA_MAIN = 16             # main pipe azimuthal cells -- sets the wall cell size
RADIAL_MAIN = np.array([0.0, 0.6, 1.0])   # main pipe O-grid radial stations
                                          # -- T1's own, so the patterns coincide
CENTER_SCALE_MAIN = 0.75      # main pipe O-grid hub placement -- == chimera_full's T1
N_THETA_BRANCH = 16           # branch azimuthal cells; MULTIPLE OF 4, and even in
                              # N_THETA_BRANCH/4 so the ring has a node on the y = 0
                              # plane the solid jacket is cut on
CENTER_SCALE_BRANCH = 0.8     # branch O-grid hub placement

#: Boundary-layer stations, as distances out from the wall the fluid core is meshed at
#: -- so the core is built at ``R_MAIN - BL[-1]`` / ``R_BRANCH - BL[-1]`` and
#: :func:`tjunction_lib.skin_wall` grows the layers back out, landing the finished wall on
#: ``R_MAIN`` / ``R_BRANCH`` exactly.  An offset is a uniform thickness along the normal
#: rather than a scaling, which is why one number does both radii.
#:
#: **The thickness is bounded by the geometry, not by taste.**  An offset can only move a
#: surface by less than its own local feature size, and the smallest feature here is the
#: branch bore's imprint on the far pipe wall.  Measured on one junction at this
#: resolution: ``0.05 * R_MAIN`` folds four elements at the branch root (min scaled
#: Jacobian -0.707), and ``0.04`` is the thickest of this family that comes back clean.
#:
#: ``chimera_full.py`` grows the **same** layer over its own manifold, and is not what
#: bounds this: with every junction there meshed by ``build_cob`` too, the joined mesh's
#: worst element is a junction's own and the skin costs it nothing -- measured identical
#: at 0.025, 0.035 and 0.045.
T_BL = 0.035 * R_MAIN
BL = T_BL * np.array([0.0, 0.6, 1.0])
#: What the *fluid core* is meshed at.  Everything inside :func:`build_junction` and every
#: block lofted off its ports is built on these; only the jacket, which welds onto the
#: finished outside, uses the nominal radii above.
RC_MAIN = R_MAIN - T_BL
RC_BRANCH = R_BRANCH - T_BL

N_LEG = 6                     # reference layer count -- sets the axial cell target
N_BRANCH = 4                  # layers in the branch, footprint -> opening

ORDER = 2                     # exact at any order; see cob_tjunction.py

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

SOLID_W = BR_SPACING / 2.0      # how far the jacket reaches out past the pipe axis
N_SOLID = 4                     # jacket radial layers, tube wall -> outer face
SOLID_EXT = R_BEND + 2.0*R_BRANCH   # jacket overhang past each end of the straight run

#: The two ``element_tags`` -- **regions**, not boundary conditions. Every element
#: carries exactly one, so the conjugate split reads straight off ``element_tags``:
#: ``hexmesh.select(mesh, SOLID_TAG)`` and ``select(mesh, FLUID_TAG)`` partition the
#: mesh. Leaving the fluid untagged would have distinguished the two just as well, but
#: only by convention: ``element_tags`` is sparse, so a missing row cannot say whether
#: an element was classified and found to be fluid or simply never looked at.
#: Both are applied where the block is built, never stamped on afterwards, so a block
#: added later has to state which region it joins rather than defaulting into one.
FLUID_TAG = "fluid"
SOLID_TAG = "solid"

#: The jacket's own external faces. A ``face_tags`` name, kept distinct from
#: :data:`SOLID_TAG` because it is a boundary condition -- it belongs to :data:`GROUPS`
#: alongside ``"wall"`` / ``"inlet"`` / ``"outlet"``, not to the region vocabulary.
INSULATED_TAG = "insulated"

#: The conjugate interface: the tube wall the jacket welded onto, which is one shared
#: face between a ``"fluid"`` hex and a ``"solid"`` one. It has to be named apart from
#: the rest of the wall because its **two sides want different conditions** -- and a
#: face carries one name, so the asymmetry lives in :data:`GROUPS`, keyed by the region
#: of the element each exported row belongs to.
INTERFACE_TAG = "interface"

#: The tube's own no-slip wall -- and, with :data:`INTERFACE_TAG`, one of the two names
#: :func:`tjunction_lib.skin_wall` grows the boundary layer over.  **Every** no-slip
#: surface carries it, dead-end lids included: a lid is as much a no-slip wall as the tube
#: is, and ``offset_shift`` miters the 90-degree corner between the two rather than
#: averaging across it, so both faces keep their full thickness (measured: the rim lands
#: at ``R_MAIN`` exactly *and* ``T_BL`` past the lid).  What is left out of the skin is
#: only the flow openings -- ``inlet`` / ``outlet`` -- where a layer would be elements
#: across the opening itself.
WALL_TAG = "wall"

N_COPIES = 5                    # number of chimera units chained along z

OUT_NAME = "chimera"
GROUPS = {
    "wall": "W  ", "inlet": "v  ", "outlet": "O  ", "insulated": "I  ",
    # the fluid side keeps the tube's own no-slip wall; the solid side writes nothing,
    # so the interface exports exactly the rows it did when it was plain "wall"
    "interface": {"fluid": "W  ", "solid": None},
}

# -- the junction, built once ---------------------------------------------------
# Every junction in the chain is the same block at a different ``z``, so it is built once
# and translated.  ``build_cob`` -- the **cob** construction: the branch bored straight
# through the main pipe, no hub for a small radius ratio to degenerate at.  At this
# file's own ratio (0.42) it measures min scaled Jacobian 0.47 against
# ``build_tjunction``'s 0.007 with two poor elements, and its ports come back *bit
# identical to the plain pipe section*, so a leg is a plain extrude with no morph.
#
# The radii are the **core** ones: the boundary layer is grown back out over the finished
# assembly at the very end, not junction by junction, because the wall it has to cover is
# the whole fluid surface -- legs, arms and hairpin bends included.
JUNCTION = build_cob(RC_MAIN, RC_BRANCH, H_BRANCH, Z_NEAR=Z_NEAR, order=ORDER,
                     N_THETA_MAIN=N_THETA_MAIN, RADIAL_MAIN=RADIAL_MAIN,
                     CENTER_SCALE_MAIN=CENTER_SCALE_MAIN,
                     N_THETA_BRANCH=N_THETA_BRANCH,
                     CENTER_SCALE_BRANCH=CENTER_SCALE_BRANCH,
                     N_BRANCH=N_BRANCH, element_tag=FLUID_TAG, wall_tag=WALL_TAG)


def opening(t):
    """The branch's circular opening at ``x = H_BRANCH``, at the **finished** radius.

    Only the jacket uses it: the fluid tube is meshed at :data:`RC_BRANCH` and the skin
    carries it out to here, so this is where the two meet.  ``t`` is the branch's own
    polar angle measured from ``+z``, the same parameter :data:`SOLID_FR` reads off the
    junction's port disc."""
    t = np.asarray(t, dtype=float)
    return np.stack([np.full(t.shape, H_BRANCH),
                     R_BRANCH * np.sin(t), R_BRANCH * np.cos(t)], axis=1)


#: The branch angles of the port disc's **own** wall nodes, over the ``-y`` semicircle the
#: jacket wraps, descending ``0 -> -pi``.  Read off the disc rather than resampled: the
#: cob's bore ring is spaced by pairing it against the collar's square, not evenly in
#: angle, so an even sampling would land the jacket's interface nodes between the tube's
#: and :func:`hexmesh.attach <nekmeshpy.hexmesh.assemble.attach>` would refuse the seam
#: for not corresponding one for one.
#:
#: The two halves still do not agree bit for bit -- the skin's own outer ring is where
#: ``quadmesh.offset``'s averaged normals put it, which slips about 9e-5 rad round the
#: circle from the analytic one -- and that is exactly why the jacket is **attached**
#: rather than merged: a stated seam pairs by bijection and then adopts one side's nodes,
#: where a proximity weld would have to be told a tolerance it cannot safely be given.
_RING = quadmesh.boundary_mesh(JUNCTION.disc_branch, WALL_TAG).points
_ANG = np.arctan2(_RING[:, 1], _RING[:, 2])
_ANG[np.isclose(_ANG, np.pi)] = -np.pi          # the far end is one node, not two
SOLID_FR = np.sort(_ANG[_ANG <= 1e-12])[::-1]


# -- build ----------------------------------------------------------------------
def build_junction(z0, run_minus, run_plus):
    """One full junction -- the shared :data:`JUNCTION` block, extended here by the
    branch's straight arm and (where a chain neighbour needs it) each leg's own run, then
    moved to its place on the chain.

    Returns ``(block, disc_minus, disc_plus, disc_branch)``: the merged, translated hex
    block, and the three plain discs *also* translated to this junction's position --
    except ``disc_branch``, kept in the **local** frame, since :func:`bend` reuses one
    copy of it at every ``z0`` rather than rebuilding it per junction.

    ``run_minus`` / ``run_plus`` ``<= 0`` skip that leg's own extrusion, leaving it bare
    -- what an outermost junction's outward-facing side uses, since the straight run past
    it is pulled out into an independent :func:`end_stub` built to its own length.
    Nothing here is tagged beyond :data:`FLUID_TAG` and the wall the junction names
    itself -- every open fluid face this leaves behind is welded shut by a neighbouring
    junction, a hairpin bend or an :func:`end_stub`."""
    core, disc_minus, disc_plus, disc_branch = JUNCTION
    blocks = [core,
              hexmesh.extrude(disc_branch, ARM_LEN, N_ARM, axis=(1.0, 0.0, 0.0),
                              element_tags=FLUID_TAG)]
    if run_plus > 0.0:
        n_run = max(1, round(run_plus / AXIAL_CELL))
        blocks.append(hexmesh.extrude(disc_plus, run_plus, n_run, element_tags=FLUID_TAG,
                                      axis=(0.0, 0.0, 1.0)))
    if run_minus > 0.0:
        n_run = max(1, round(run_minus / AXIAL_CELL))
        blocks.append(hexmesh.extrude(disc_minus, run_minus, n_run, element_tags=FLUID_TAG,
                                      axis=(0.0, 0.0, -1.0)))
    z_off = (0.0, 0.0, z0)
    return (hexmesh.translate(hexmesh.merge(blocks), z_off),
            quadmesh.translate(disc_minus, z_off),
            quadmesh.translate(disc_plus, z_off),
            disc_branch)


def end_stub(disc, sign, tag, run):
    """The independent straight run past a unit's outermost junction, on the
    ``sign`` end starting from its bare plain ``disc`` (a translated
    ``disc_minus``/``disc_plus`` from :func:`build_junction`), tagged ``tag`` at its
    far face."""
    n_run = max(1, round(run / AXIAL_CELL))
    return hexmesh.extrude(disc, run, n_run, element_tags=FLUID_TAG,
                           axis=(0.0, 0.0, float(sign)), last_tag=tag)


def outlet_return(disc):
    """Pipe B's true outlet, only at the chain's last copy: a single U-turn toward
    ``+y`` (radius :data:`RETURN_BEND_R`) starting at the bare plain ``disc`` (the
    last junction's own ``disc_plus``), folding the pipe back to run parallel to the
    whole chain until it reaches the same global ``z`` as the chain's own
    ``"inlet"`` -- where it becomes the new ``"outlet"``. Built in pipe A's own local
    frame and carried onto pipe B by the caller's :func:`to_b`."""
    z0 = Z_J[-1] + Z_NEAR
    run = (Z_J[-1] - Z_J[0]) + Z_NEAR + END_MARGIN + 2.0 * L_HALF * (N_COPIES - 1)
    # the walk heads down the chain (+z) and folds toward +y, which pins its up to
    # heading x left = -x; the sweep below reads that frame off the path itself.
    path = paths.walk([paths.arc(RETURN_BEND_R, -180.0), paths.line(run)],
                      start=(0.0, 0.0, z0), heading=(0.0, 0.0, 1.0),
                      up=(-1.0, 0.0, 0.0))
    n_bend = max(1, round(RETURN_BEND_R * np.pi / BEND_CELL))
    n_run = max(1, round(run / AXIAL_CELL))
    breaks = path.break_fractions
    station_fr = np.concatenate([np.linspace(0.0, breaks[0], n_bend + 1),
                                 np.linspace(breaks[0], 1.0, n_run + 1)[1:]])
    return hexmesh.sweep_path(disc, path, fractions=station_fr,
                              origin=(0.0, 0.0, z0), last_tag="outlet",
                              element_tags=FLUID_TAG)


def to_b(m):
    """Carry a block built in pipe A's own frame onto pipe B, by the rotation that
    turns the whole unit's pipe A into pipe B."""
    return hexmesh.rotate(m, np.pi, axis=(0.0, 0.0, 1.0), center=(BEND_CENTER_X, 0.0, 0.0))


#: The hairpin path, in the plane ``z = z0``, as a declarative turtle walk exactly
#: like ``examples/serpentine_pipe.py``'s: starting at the arm's own ending point
#: ``(X_MID, 0)`` heading ``+x``, a 180 degree U-turn through ``+y``, a straight run
#: ``LOOP_LEN`` back toward ``-x``, and a second 180 degree U-turn through ``-y`` that
#: restores heading ``+x`` -- landing exactly on pipe B's own (rotated) arm end, so
#: pipe B ends up offset purely along ``-x``, not diagonally, with its arm facing
#: ``-x`` to meet it. Every move keeps the previous move's heading, so the path is C1
#: by construction.
_MOVES = [paths.arc(R_BEND, 180.0), paths.line(LOOP_LEN), paths.arc(R_BEND, 180.0)]

#: The junction plane ``z0`` only translates the walk, so one reference walk settles the
#: junction fractions for every copy.
_BREAKS = paths.walk(_MOVES, start=(X_MID, 0.0, 0.0), heading=(1.0, 0.0, 0.0),
                     up=(0.0, 0.0, 1.0)).break_fractions

#: Sweep stations, one exactly on every straight<->arc junction, each of the three
#: pieces graded at its own layer count rather than one global linspace.
_STATION_FR = np.concatenate([
    np.linspace(0.0, _BREAKS[0], N_BEND + 1),
    np.linspace(_BREAKS[0], _BREAKS[1], N_LOOP + 1)[1:],
    np.linspace(_BREAKS[1], 1.0, N_BEND + 1)[1:]])

#: Pipe B's axis, at ``x = -D_PIPES`` (solved above); the rotation midway between it
#: and pipe A's own axis at the origin is what turns pipe A bodily into pipe B.
BEND_CENTER_X = -D_PIPES / 2.0


def bend(z0, opening_disc):
    """One junction's hairpin, at height ``z0``: the arm's ending disc (``opening_disc``,
    a **local**-frame ``disc_branch`` from any one :func:`build_junction` call -- they
    are all bit-identical, so one is built and reused rather than rebuilt per
    junction) carried along ``_MOVES`` walked in the plane ``z = z0``, landing on
    pipe B's own (rotated) arm end by construction -- no separate return arm is
    built."""
    path = paths.walk(_MOVES, start=(X_MID, 0.0, z0), heading=(1.0, 0.0, 0.0),
                      up=(0.0, 0.0, 1.0))
    section = quadmesh.translate(opening_disc, (ARM_LEN, 0.0, z0))
    return hexmesh.sweep_path(section, path, fractions=_STATION_FR,
                              origin=(X_MID, 0.0, z0), element_tags=FLUID_TAG)


def to_run(m):
    """Carry a section built in the arm's frame onto the hairpin's straight run --
    the half turn :func:`bend`'s own sweep has applied by the time it gets there, so
    a section handed through here lands on exactly the frame the fluid disc does."""
    return quadmesh.rotate(m, np.pi, axis=(0.0, 0.0, 1.0), center=(X_MID, R_BEND, 0.0))


def solid_section(z0, minus_tag, plus_tag, inner_tag=quadmesh.NO_TAG):
    """The jacket's cross-section on the straight run at height ``z0``: the pipe's own
    semicircle lofted out to half a ``BR_SPACING`` tall rectangle.

    ``minus_tag`` / ``plus_tag`` name the two ``z`` faces -- where this junction's
    jacket meets its neighbours', so ``NO_TAG`` everywhere but at the chain's two
    extreme ends. The remaining external edges are named here: the outer face and, via
    the inner ring's ``point_tags``, the two strips left in the ``y`` plane the
    semicircle is cut on. ``inner_tag`` names the semicircle itself, which is
    ``NO_TAG`` over the straight run -- it welds onto the tube's own ``"wall"`` faces,
    which already name that side -- and :data:`INSULATED_TAG` over the two overhangs,
    where the tube has curved away and left the void's surface external."""
    ring = linemesh.translate(linemesh.loft_fn(opening, SOLID_FR, order=ORDER),
                              (ARM_LEN, 0.0, z0))
    # the semicircle's two ends -- the strips left in the y plane it is cut on.  A
    # point tag names the point itself now, so these are point ids, not (line, side).
    inner = linemesh.LineMesh(
        PointMesh(ring.points,
                  ElementTags(np.array([0, ring.n_points - 1]),
                              np.array([INSULATED_TAG, INSULATED_TAG]))),
        ring.lines, ring.interior)

    #: The ring's own segment count, split ``n/2 | n | n/2`` round the half-rectangle
    #: below.  It is read off :data:`SOLID_FR` rather than set, so the outer polyline is
    #: cut to match however finely the branch is resolved -- which is what keeps every
    #: radial line springing from a node of the tube's own wall.
    n = (SOLID_FR.size - 1) // 2
    h = BR_SPACING / 2.0
    #: The half-rectangle's four corners, in the ring's own order: the semicircle's
    #: two ends and its two quadrant corners map to them, so the outer polyline is cut
    #: ``n/2 | n | n/2`` and every radial line springs from a node of the fluid disc.
    corner = [(X_MID, 0.0, z0 + h), (X_MID, -SOLID_W, z0 + h),
              (X_MID, -SOLID_W, z0 - h), (X_MID, 0.0, z0 - h)]

    def seg(a, b, k):
        s = np.linspace(0.0, 1.0, k + 1)[:-1, None]
        return (1.0 - s) * np.asarray(a) + s * np.asarray(b)

    outer = linemesh.loft(np.vstack([seg(corner[0], corner[1], n // 2),
                                     seg(corner[1], corner[2], n),
                                     seg(corner[2], corner[3], n // 2),
                                     [corner[3]]]), order=ORDER)
    face = np.array([plus_tag] * (n // 2) + [INSULATED_TAG] * n
                    + [minus_tag] * (n // 2))
    tagged = np.flatnonzero(face != quadmesh.NO_TAG)

    rings = linemesh.blend(inner, outer, np.linspace(0.0, 1.0, N_SOLID + 1))
    return to_run(quadmesh.loft(rings, first_tag=inner_tag,
                                last_tag=ElementTags(tagged, face[tagged])))


#: Layers in each overhang, at the same target cell length as the hairpin itself.
N_SOLID_EXT = max(1, round(SOLID_EXT / BEND_CELL))


def solid_run(z0, minus_tag=quadmesh.NO_TAG, plus_tag=quadmesh.NO_TAG):
    """One junction's solid jacket: :func:`solid_section` run back along the straight
    stretch of the hairpin and ``SOLID_EXT`` past each of its ends.

    Three blocks rather than one graded extrude, because they differ in what the
    semicircle bounds. Over the middle stretch it is the fluid interface, so it is named
    :data:`INTERFACE_TAG` -- the one seam this whole block exists to make -- and lands on
    the same ``N_LOOP`` stations :data:`_STATION_FR` gives the tube there; the run is
    straight, so a plain extrude divides it the same way. Over
    the two overhangs the tube has peeled off into its U-turn and the semicircle
    bounds an empty cylindrical void instead, so it is named :data:`INSULATED_TAG`.

    The tube does re-enter the slab's ``y`` range on its way round, but only ever
    *inside* that void: a point on a circle of radius ``r <= R_BRANCH`` about a centre
    ``d`` short of the void's axis is at ``r**2 - 2 r d cos(theta) + d**2`` from that
    axis, which is below ``R_BRANCH**2`` whenever ``r cos(theta) > d`` puts it past
    the axis at all. So the overhangs never eat into the fluid."""
    mid = solid_section(z0, minus_tag, plus_tag, INTERFACE_TAG)
    ext = solid_section(z0, minus_tag, plus_tag, INSULATED_TAG)
    return [hexmesh.extrude(quadmesh.translate(ext, (SOLID_EXT, 0.0, 0.0)), SOLID_EXT,
                            N_SOLID_EXT, axis=(-1.0, 0.0, 0.0),
                            element_tags=SOLID_TAG, first_tag=INSULATED_TAG),
            hexmesh.extrude(mid, LOOP_LEN, N_LOOP, axis=(-1.0, 0.0, 0.0),
                            element_tags=SOLID_TAG),
            hexmesh.extrude(quadmesh.translate(ext, (-LOOP_LEN, 0.0, 0.0)), SOLID_EXT,
                            N_SOLID_EXT, axis=(-1.0, 0.0, 0.0),
                            element_tags=SOLID_TAG, last_tag=INSULATED_TAG)]


#: Junction centres within one unit, evenly spaced and centred on ``z = 0``.
Z_J = (np.arange(N_BR) - (N_BR - 1) / 2.0) * BR_SPACING


def build_chimera(a_minus, a_plus, b_minus, b_plus, *, b_plus_return=False):
    """One chimera unit's **fluid** -- pipe A, its rotation pipe B, and every hairpin
    bend between them -- centred on its own local ``z = 0``, meshed at the core radii
    :data:`RC_MAIN` / :data:`RC_BRANCH`. Each argument is one open end's ``(tag, run)``
    pair from :func:`end_spec`, so a dead-end and a through-connection at the same
    nominal joint genuinely differ in how far they reach, rather than only in how they
    are tagged. ``b_plus_return`` swaps pipe B's plain ``+z`` stub for
    :func:`outlet_return`'s U-turn, for the chain's last copy only -- ``b_plus`` is then
    unused.

    The solid jacket is **not** built here: it welds onto the *finished* tube wall, which
    does not exist until the whole chain has been skinned. :func:`build_jackets` builds
    it, and one stated seam at the end joins the two.

    The junction chain is built once with both its outermost ends left bare
    (:func:`build_junction`'s ``run <= 0`` case), and pipe B still comes for free as a
    rotation of pipe A. Only then are the four end stubs added -- pipe A's directly,
    pipe B's built in pipe A's own frame and carried over by :func:`to_b`."""
    mid_run = BR_SPACING / 2.0 - Z_NEAR
    junctions = [build_junction(Z_J[i], 0.0 if i == 0 else mid_run,
                                0.0 if i == N_BR - 1 else mid_run)
                for i in range(N_BR)]
    pipe_a = hexmesh.merge([j[0] for j in junctions])
    pipe_b = to_b(pipe_a)
    opening_disc = junctions[0][3]         # any junction's disc_branch will do
    bends = [bend(z0, opening_disc) for z0 in Z_J]
    disc_minus_first, disc_plus_last = junctions[0][1], junctions[-1][2]
    b_plus_stub = (to_b(outlet_return(disc_plus_last)) if b_plus_return
                  else to_b(end_stub(disc_plus_last, 1, *b_plus)))
    stubs = [end_stub(disc_minus_first, -1, *a_minus),
             end_stub(disc_plus_last, 1, *a_plus),
             to_b(end_stub(disc_minus_first, -1, *b_minus)), b_plus_stub]
    return hexmesh.merge([pipe_a, pipe_b, *bends, *stubs])


def build_jackets(solid_minus=quadmesh.NO_TAG, solid_plus=quadmesh.NO_TAG):
    """One unit's ``N_BR`` solid jackets, in the unit's own local frame.

    ``solid_minus`` / ``solid_plus`` name the outermost jackets' ``z`` faces, external
    only at the two ends of the whole chain -- everywhere else a jacket welds into its
    neighbour's, within the unit and across it."""
    return [block
            for i, z0 in enumerate(Z_J)
            for block in solid_run(z0,
                                   solid_minus if i == 0 else quadmesh.NO_TAG,
                                   solid_plus if i == N_BR - 1 else quadmesh.NO_TAG)]


# -- the chain of copies --------------------------------------------------------
#: One unit's half-length along z (its own ``Z_J``/``END_MARGIN`` span) -- unit ``k``
#: is placed at ``2 * L_HALF * k`` so its minus end exactly coincides with unit
#: ``k - 1``'s plus end.
L_HALF = (N_BR - 1) / 2.0 * BR_SPACING + END_MARGIN


def link_connector(j):
    """Which pipe carries link ``j``, the joint between units ``j`` and ``j + 1``.

    Alternating down the chain -- pipe A on the odd links, pipe B on the even ones. Which
    phase it starts on is arbitrary; that it *alternates* is not, since each unit is
    already one connected component on its own (its hairpin bends tie pipe A to pipe B),
    so the chain stays one component exactly as long as some pipe connects every
    neighbouring pair."""
    return "A" if j % 2 == 1 else "B"


def end_spec(is_extreme, extreme_name, is_connector):
    """The ``(tag, run)`` for one (pipe, end) pair of one unit: the chain's own
    inlet/outlet name at an extreme end, run out the full ``END_MARGIN - Z_NEAR``;
    otherwise ``""`` at that same full length (welds into the neighbouring unit) if
    this pipe is this link's connector, else :data:`WALL_TAG` stopped ``GAP`` short of the
    joint plane -- genuine empty space, not a coincident closed cap.

    A dead end is plain :data:`WALL_TAG`, so the boundary layer covers it like any other
    no-slip surface. The lid ends up ``T_BL`` further out than the number here, which the
    ``GAP`` has room for many times over."""
    full = END_MARGIN - Z_NEAR
    if is_extreme:
        return extreme_name, full
    if is_connector:
        return "", full
    return WALL_TAG, full - GAP


fluids, solids = [], []
for _k in range(N_COPIES):
    _is_first, _is_last = _k == 0, _k == N_COPIES - 1
    # Which pipe connects unit _k to its neighbour on each side.  The link before unit
    # _k is link _k - 1 and the link after it is link _k, so one predicate settles both --
    # which is what keeps a link's two sides from ever disagreeing about who connects it.
    _connector_before = link_connector(_k - 1)
    _connector_after = link_connector(_k)

    _shift = (0.0, 0.0, 2.0 * L_HALF * _k)
    _unit = build_chimera(end_spec(_is_first, "inlet", _connector_before == "A"),
                          end_spec(_is_last, WALL_TAG, _connector_after == "A"),
                          end_spec(_is_first, WALL_TAG, _connector_before == "B"),
                          end_spec(_is_last, "outlet", _connector_after == "B"),
                          b_plus_return=_is_last)
    fluids.append(hexmesh.translate(_unit, _shift))
    solids += [hexmesh.translate(b, _shift) for b in build_jackets(
        INSULATED_TAG if _is_first else quadmesh.NO_TAG,
        INSULATED_TAG if _is_last else quadmesh.NO_TAG)]

fluid = hexmesh.merge(fluids)

# -- name the conjugate interface, on the core, before it is skinned ------------
#: The jacket covers the outboard (``+y``) half of each hairpin's straight run, and
#: nothing else in the model reaches that far out: the run's own axis sits at
#: ``2 * R_BEND``, while the widest thing on either manifold -- the last copy's outlet
#: return -- folds only to ``2 * RETURN_BEND_R``. So ``x`` within the run and ``y`` above
#: its axis names exactly the tube faces the jacket will meet, and the count says so.
#: Naming it **here**, on the core, is what carries it through the skinning: the shell's
#: outer cap inherits each wall group's own name, so the seam survives to the end instead
#: of being flattened into one ``"wall"``.
_Y_RUN = 2.0 * R_BEND
_free = np.flatnonzero(hexmesh.boundary_face_ids(fluid))
_named = fluid.face_tags.dense(fluid.quad_mesh.n_quads)
_cand = _free[_named[_free] == WALL_TAG]
_mid = fluid.points[fluid.quad_mesh.corners[_cand]].mean(axis=1)
_hit = ((_mid[:, 0] > X_MID - LOOP_LEN) & (_mid[:, 0] < X_MID)
        & (_mid[:, 1] > _Y_RUN))
_want = (SOLID_FR.size - 1) * N_LOOP * N_BR * N_COPIES
assert _hit.sum() == _want, "run interface: %d faces, expected %d" % (_hit.sum(), _want)
fluid = hexmesh.tag_faces(fluid, _cand[_hit], INTERFACE_TAG)

# -- the boundary layer, over the whole fluid wall at once ----------------------
# Both wall groups are skinned -- every no-slip surface, dead-end lids included, since
# ``offset_shift`` miters the wall/lid corner rather than averaging across it.  Only the
# two flow openings are left alone, and their rims ride outward with the skin because a
# free edge of the surface offsets along its own averaged normal, which is radial there
# and so moves no node along the pipe.
fluid = skin_wall(fluid, BL, wall_tags=(WALL_TAG, INTERFACE_TAG),
                  element_tag=FLUID_TAG)

# -- the jacket, welded onto the finished wall ----------------------------------
# ``attach``, not ``merge``: the two halves correspond node for node but do not agree bit
# for bit (see :data:`SOLID_FR`), and a stated seam pairs by bijection and then adopts one
# side's nodes -- here the fluid's, which is the side whose geometry is the physics.
# ``attach_tag`` keeps the buried seam **named**, which is the whole point of it: the two
# sides want different conditions, and :data:`GROUPS` keys them by each row's own region.
mesh = hexmesh.attach([fluid, hexmesh.merge(solids)],
                      [Seam(0, INTERFACE_TAG, 1, INTERFACE_TAG, own="a",
                            attach_tag=INTERFACE_TAG)])

print(hexmesh.report(mesh))
print(hexmesh.topology_report(mesh))

writer.to_re2(mesh, OUT_NAME + ".re2", groups=GROUPS)
writer.to_vtu(mesh, OUT_NAME + ".vtu", groups=GROUPS)
