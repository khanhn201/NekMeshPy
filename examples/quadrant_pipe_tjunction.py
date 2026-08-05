"""T-junction of a smaller branch pipe on a main pipe, meshed as a **single welded
component** with the junction topology of the reference vascular mesher.

Run with::

    PYTHONPATH=. python examples/quadrant_pipe_tjunction.py

Produces ``quadrant_pipe_tjunction.re2`` and ``quadrant_pipe_tjunction.vtu``.

The decomposition
-----------------

Every cross-section is a **full disc of four** ``quadrant_ogrid`` **blocks**, and the
junction is built so that *one quadrant of the main pipe is a quadrant of the branch*.
That is what welds the branch to the main pipe instead of merely parking it against
the wall.

Four solid regions meet at the junction centre ``O``, the point where the two axes
cross, and every interface between them is a quadrant face radiating from ``O``:

=====================  =======================================================
region                 its near-junction face
=====================  =======================================================
branch stub            the **footprint disc** -- four quadrants ``A/D/C/B``
                       whose wall is the exact intersection curve
``+z`` (outlet) leg    ``[A, sideRP, bypass, sideRM]``
``-z`` (inlet) leg     ``[C, sideLM, bypass reversed, sideLP]``
``+y`` crotch cap      ``[B, sideLP, sideRP]``
``-y`` crotch cap      ``[D, sideRM, sideLM]``
=====================  =======================================================

so the nine internal faces are

* ``bypass`` -- between the two legs: the part of the cross-section far from the
  branch, spanning ``phi`` from ``+PHI_W`` round through ``pi`` to ``-PHI_W``;
* ``A`` / ``C`` -- between the branch and the ``+z`` / ``-z`` leg.  **These are the
  welded quadrant-to-quadrant interfaces**: quadrant ``A`` of the branch's footprint
  disc *is* the branch-facing quadrant of the outlet leg's face;
* ``B`` / ``D`` -- between the branch and the two caps;
* the four ``side*`` quadrants -- between a leg and a cap.  Their wall arcs are the
  ruled ``(phi, z)`` curves that carry a footprint corner round the main cylinder to
  a bypass edge corner at ``z = 0``.

Each leg then morphs from its composite face to a plain 45-degree-rotated
four-quadrant disc and extrudes straight to its end plane; the branch morphs from the
footprint disc to a plain circular disc at ``x = H_BRANCH``.  Adjacent quadrants are
handed the *same* seam ``LineMesh`` object -- one of the six radii ``O -> P1..P4``,
``O -> W+``, ``O -> W-`` -- so they weld bit-exactly rather than to a tolerance.

**Both morphs are done in a surface parameter, not on the points.**  A leg's
transition interpolates its wall in the cylinder's own ``(phi, z)``; the branch
interpolates in the branch angle, where ``footprint`` and ``opening`` differ only in
``x``.  Lerping the *points* instead draws a chord between two points of a cylinder,
which dips inside it, so every intermediate station would sit proud of the wall.

The two crotch caps
-------------------

A crotch is the corner region bounded by three quadrant faces meeting at ``O`` plus a
curved triangle of main-pipe wall.  That is exactly an **octant of a 3-D O-grid**:
index the corner by ``(i, j, k)`` along its three seams and it is an ``n x n x n``
core hex block -- whose three inner faces are the three quadrant *cores* -- plus three
``n x n x Nradial`` slabs carrying the core's three outer faces out to a third of the
wall triangle each.  That gives ``n**2 + 2*n*Nradial`` quads on each shared face,
which is exactly what a ``quadrant_ogrid`` has, and the slabs reuse that factory's own
ring formula ``(1-tau)*perimeter + tau*wall``, so the two halves of every shared ring
band agree and the cap welds.

The wall triangle's tip -- the point its three patches converge to, and the direction
the octant core's far corner is placed along -- defaults to the plain centroid of the
three arcs' midpoints. ``CAP_TIP_BIAS`` reweights that toward the branch-facing arc
(always the crotch's ``ab`` argument), sliding the tip -- and the visible fan of mesh
lines around it -- closer to the branch.

Order
-----

Exact at any ``ORDER``, which took some care: a junction like this is where the
straight-subdivision traps all bite at once.

* Every wall curve is carried as its **surface parametrization** (:class:`Wall`), not
  as sampled points, and meshed with ``LineMesh.loft_fn`` -- so the footprint, the
  side transitions and the bypass are all evaluated on the true curve at every node.
  Sampling them into arrays and calling ``LineMesh.loft`` chords the wall from
  ``order > 1`` on.
* A leg's transition is a ``HexMesh.loft_fn`` over the blend parameter, not a
  ``loft`` of a section stack.  ``loft`` is straight **along the sweep**, so its wall
  nodes would be chords between stations at different ``phi`` -- measured here at
  order 3, that alone put them 7.2e-4 off the cylinder.
* The cap blocks are ``evaluated_block``s -- nested ``loft_fn`` over an explicit
  map on the unit cube -- rather than ``HexMesh.from_grid``, which takes corners and
  blends straight.  A ``from_grid`` cap is fine at order 1 and at ``order > 1`` both
  leaves the wall and disagrees with ``quadrant_ogrid``'s bowed ring bands along the
  faces they share, which ``merge`` rightly rejects.

Order 1 remains a strict no-op: the corner coordinates are bit-identical at ``ORDER``
1, 2, 3 and 4, and every wall node lies on the main or branch cylinder to 2.2e-16 at
all of them.
"""

import logging
import os
import sys

import numpy as np

from nekmeshpy import export, hexmesh, linemesh, quadmesh
from nekmeshpy.model import surfaces

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tjunction_lib import auto_params  # noqa: E402  (needs the path above)

logging.basicConfig(level=logging.INFO, format="%(message)s")

# -- parameters ---------------------------------------------------------------
R_MAIN = 1.0                 # main pipe radius (axis = z)
R_BRANCH = 0.5               # branch pipe radius (axis = x), deliberately < R_MAIN
L_MAIN = 3.0                 # main pipe runs z = -L_MAIN .. +L_MAIN
Z_NEAR = 1.2                 # where a leg has finished morphing to a plain disc
H_BRANCH = 2.5               # branch opening plane, x = H_BRANCH

N_QUAD = 3                   # cells per quadrant half-arc; a quadrant spans 2*N_QUAD
RADIAL = np.array([0.0, 0.45, 0.8, 1.0])   # O-ring positions, core perimeter -> wall
CENTER_SCALE = 0.55          # core corner at CENTER_SCALE * R along the arc midpoint

#: Weight of the branch-facing arc (``ab`` in every :func:`cap` call, always the
#: branch's own footprint quadrant) when locating each crotch's wall-triangle "tip" --
#: the wall point its three patches converge to.  The remaining weight is split evenly
#: between the two side arcs.  ``1/3`` is the plain centroid of the three arc
#: midpoints (the original, symmetric placement); raise it toward ``1.0`` to pull the
#: tip closer to the branch, lower it toward ``0.0`` to push it toward the bypass wall.
#: Keep it inside ``(0, 1)`` -- either endpoint collapses one patch's spoke to zero
#: length.


N_TRANS = 5                  # layers from a leg's composite face to its plain disc
N_LEG = 6                    # layers from the plain disc to the end plane
N_BRANCH = 8                 # layers in the branch, footprint -> opening

ORDER = 3                    # exact at any order; see the module docstring

OUT_NAME = "quadrant_pipe_tjunction"
GROUPS = {"wall": "W  ", "inlet": "v  ", "outlet": "O  ", "branch": "O  "}

N = N_QUAD
#: The three shape parameters of the quadrant construction, chosen for this
#: junction's own radius ratio rather than fixed -- see
#: ``tjunction_lib.auto_params``. ``PHI_W`` is the bypass edge (the two z = 0
#: wall corners sit at +-PHI_W); ``CAP_TIP_BIAS`` weights the branch arc when
#: locating each crotch's wall-triangle tip; ``ORIGIN`` is the junction hub every
#: quadrant seam radiates from.
PHI_W, CAP_TIP_BIAS, ORIGIN = auto_params(R_MAIN, R_BRANCH)

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
#:
#: A function, not an array, because at ``order > 1`` every node -- the element
#: interiors as much as the corners -- has to be evaluated on the true curve.  Handing
#: ``LineMesh.loft`` a sampled array instead subdivides straight between the samples,
#: which chords the wall from ``order > 1`` on.  Parameters rather
#: than points because ``phi`` is only defined modulo ``2*pi`` and two curves
#: deliberately take the long way round (the bypass runs *behind* the pipe, from
#: ``+PHI_W`` to ``2*pi - PHI_W``), and because morphing a leg in ``(phi, z)`` is what
#: keeps its transition stations on the wall (see :func:`leg`).
#: A wall curve is a :class:`~nekmeshpy.model.surfaces.SurfaceCurve`, carried as its
#: parametrization rather than as points, which is what keeps every derived curve
#: exactly *on* the cylinder.


def cyl_pts(u):
    """``(K, 2)`` wall parameters to ``(K, 3)`` points -- this junction's surface map."""
    return cyl(u[:, 0], u[:, 1])


def wall_mesh(w):
    """A wall curve as a ``LineMesh`` on the cylinder, exact at every node."""
    return linemesh.on_surface(w, cyl_pts, order=ORDER)


def ruled_wall(pa, pb):
    """The straight ``(phi, z)`` segment between two stations, on ``[0, 1]``."""
    return surfaces.ruled(pa, pb, 2 * N_QUAD)


def foot_wall(fr):
    """A footprint quadrant, in the branch angle it is analytic in."""
    return surfaces.curve(lambda t: cyl_params(footprint(t)), fr)


def plain_wall(w, phi0, phi1, z):
    """The plain-disc arc ``phi0 -> phi1`` at height ``z``, reparametrized onto
    ``w``'s domain so the two can be blended station by station (see :func:`leg`)."""
    return surfaces.reparam(w, (phi0, z), (phi1, z))


def shift_wall(w, turns):
    """The same curve with ``phi`` shifted by whole turns -- the identical points,
    rebranched so that curves meeting at a corner can be blended in ``phi``."""
    return surfaces.shift(w, (2.0 * np.pi * turns, 0.0))


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
    """A full-disc section: four ``(arc, seam1, seam2)`` quadrants merged.

    The winding is load-bearing.  ``quadrant_ogrid`` orients a block by
    ``(A1 - O) x (A2 - O)``, which must point along the sweep direction or the swept
    hexes come out inverted -- so ``pieces`` must run around the disc the right way.
    Handing each shared radius in as the same ``LineMesh`` object is what makes
    neighbours weld bit-exactly instead of to a tolerance."""
    return quadmesh.merge([quadrant(arc, s1, s2, wall_tag="wall")
                           for arc, s1, s2 in pieces])


def plain_walls(composite, z, sign):
    """The plain four-quadrant disc at height ``z`` that a leg morphs into: seams at
    ``+-45`` / ``+-135`` degrees so one quadrant faces the branch, running in the
    ``sign`` direction and so in the cyclic order that pairs index-for-index with the
    ``composite`` junction face, each quadrant on that quadrant's own domain."""
    ang = sign * np.deg2rad(-45.0 + 90.0 * np.arange(5))
    return [plain_wall(composite[q], ang[q], ang[q + 1], z) for q in range(4)]


# -- the junction geometry ----------------------------------------------------
#: ``P[q]`` is footprint corner ``q``: ``P1, P4, P3, P2`` in the winding's own order.
P = [footprint(TQ[q:q + 1])[0] for q in range(4)]
WP, WM = cyl(PHI_W, 0.0), cyl(-PHI_W, 0.0)          # the two bypass edge corners

SP = [seam(p) for p in P]                            # the four footprint radii
SWP, SWM = seam(WP), seam(WM)                        # the two bypass edge radii

#: The parameter values of each footprint quadrant's nodes, even in arc length.  The
#: opening circle is sampled at the *same* values, which is what keeps the blended
#: branch wall exactly cylindrical.
FQ_FR = [linemesh.arclength_fractions(footprint, 2 * N_QUAD,
                                      t_range=(TQ[q], TQ[q + 1])) for q in range(4)]
#: The footprint quadrant arcs: 0 = ``A`` faces ``+z``, 1 = ``D`` faces ``-y``,
#: 2 = ``C`` faces ``-z``, 3 = ``B`` faces ``+y``.
FQ = [linemesh.loft_fn(footprint, fr, order=ORDER) for fr in FQ_FR]

#: ``(phi, z)`` of the four footprint corners and the two bypass edge corners.
UP = [cyl_params(p) for p in P]
UWP, UWM = np.array([PHI_W, 0.0]), np.array([-PHI_W, 0.0])

#: The composite junction faces, as the four wall curves of each.  ``phi`` is carried
#: unwrapped -- each list ascends (``+z`` leg) or descends (``-z`` leg) monotonically
#: through a full turn -- so that morphing it toward :func:`plain_walls` sweeps the
#: short way at every station.
TURN = np.array([2.0 * np.pi, 0.0])
W_R = [foot_wall(FQ_FR[0][::-1]),                 # P4 -> P1, the welded quadrant
       ruled_wall(UP[0], UWP),                    # +y side: P1 -> W+
       ruled_wall(UWP, TURN - UWP),               # bypass:  W+ -> W- the long way
       ruled_wall(TURN - UWP, UP[1] + TURN)]      # -y side: W- -> P4
W_L = [foot_wall(FQ_FR[2][::-1]),                 # P2 -> P3, the welded quadrant
       ruled_wall(UP[2], UWM),                    # -y side: P3 -> W-
       ruled_wall(UWM, -TURN - UWM),              # bypass:  W- -> W+ the long way
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
    return [surfaces.node(w, N) for w in walls]


def cap(sa, sb, sc, ab, bc, ca, tip_bias=CAP_TIP_BIAS):
    """A crotch: the curvilinear tetrahedron whose four sides are the three quadrant
    faces meeting at ``O`` and the patch of pipe wall between their arcs.

    Each of ``ab``/``bc``/``ca`` is that arc as a ``(LineMesh, SurfaceCurve)`` pair -- the mesh
    the quadrant face is built on, the parametrization the wall patch is evaluated on.
    ``ab`` is always the branch's own footprint arc, by this file's calling
    convention, which is what lets ``tip_bias`` (:data:`CAP_TIP_BIAS`) pull the wall
    triangle's tip toward it. :meth:`HexMesh.tetra <nekmeshpy.hexmesh.HexMesh.tetra>`
    does the rest: one hex block per corner, inheriting the split each face already
    carries, which is exactly the octant of a 3-D O-grid this region wants."""
    (m_ab, w_ab), (m_bc, w_bc), (m_ca, w_ca) = ab, bc, ca
    # The centre is the octant core's far corner: out along the ray from ``O`` through
    # the wall tip (quadmesh.tri_patch_tip), at ``CENTER_SCALE * sqrt(1.5) * R``.
    # Each quadrant face puts its own core corner at ``CENTER_SCALE * R`` along its arc
    # midpoint, and on the ideal (untipped) octant the cube corner spanned by those
    # three lies ``sqrt(3/2)`` times as far out -- so the core block meets all three
    # faces' cores squarely.  ``tetra``'s default (the centroid of the four face
    # centres) is well inside that and would flatten the corner cells.
    mids = arc_mids((w_ab, w_bc, w_ca))
    # the solid apex is placed against the patch's own tip, read from the same helper
    # tri_patch uses, so the two cannot drift apart
    wc_param = quadmesh.tri_patch_tip(*mids, tip_bias=tip_bias)
    wc = cyl_pts(wc_param[None, :])[0]
    return hexmesh.tetra([quadrant(m_ab, sa, sb), quadrant(m_bc, sb, sc),
                          quadrant(m_ca, sc, sa),
                          quadmesh.tri_patch(cyl_pts, w_ab, w_bc, w_ca, order=ORDER,
                                             tip_bias=tip_bias, mids=mids,
                                             element_tag="wall")],
                         center=ORIGIN + CENTER_SCALE * np.sqrt(1.5) * (wc - ORIGIN))


# -- build --------------------------------------------------------------------
def leg(composite, walls, sign, end_tag):
    """One main-pipe leg: morph the composite junction face into a plain disc, then
    extrude straight to the end plane.

    The morph is done on the wall's ``(phi, z)`` parameters rather than by lerping
    the *points*, which is what a plain
    :meth:`QuadMesh.blend <nekmeshpy.quadmesh.QuadMesh.blend>` between the two
    sections would do: a chord between two points of a cylinder dips inside it, so
    every intermediate station of the transition would sit a little proud of the
    wall.  Parameter space has no such chord -- each node lands on the cylinder
    exactly, at any number of layers."""
    z = sign * Z_NEAR
    w_plain = plain_walls(walls, z, sign)

    def station(s):
        return quadmesh.quadrant_disc(
            [wall_mesh(surfaces.blend(walls[q], w_plain[q], s)) for q in range(4)],
            (1.0 - s) * ORIGIN + s * np.array([0.0, 0.0, z]),
            RADIAL, center_scale=CENTER_SCALE,
            wall_tag="wall")

    plain = station(1.0)
    return [hexmesh.loft_fn(station, np.linspace(0.0, 1.0, N_TRANS + 1),
                            order=ORDER),
            hexmesh.extrude(plain, L_MAIN - Z_NEAR, N_LEG,
                            axis=(0.0, 0.0, float(sign)), last_tag=end_tag)]


def branch():
    """The branch stub: the footprint disc morphed to the circular opening."""
    open_arcs = [linemesh.loft_fn(opening, fr, order=ORDER) for fr in FQ_FR]
    t = np.linspace(0.0, 1.0, N_BRANCH + 1)
    walls = [linemesh.blend(f, o, t) for f, o in zip(FQ, open_arcs)]
    c_open = np.array([H_BRANCH, 0.0, 0.0])
    sections = [quadmesh.quadrant_disc([w[i] for w in walls],
                                       (1.0 - t[i]) * ORIGIN + t[i] * c_open, RADIAL,
                                       center_scale=CENTER_SCALE, wall_tag="wall")
               for i in range(t.size)]
    return hexmesh.loft(sections, last_tag="branch")


blocks = [*leg(COMPOSITE_R, W_R, 1, "outlet"),
          *leg(COMPOSITE_L, W_L, -1, "inlet"),
          branch(),
          # A crotch's three arcs must share one branch of phi, so the two that are
          # authored a full turn away in a leg's unwrapped list are shifted back.
          cap(SP[0], SP[3], SWP,                        # +y crotch: P1, P2, W+
              (linemesh.reverse(FQ[3]), foot_wall(FQ_FR[3][::-1])),
              (linemesh.reverse(SIDE_LP), shift_wall(surfaces.reverse(W_L[3]), 1)),
              (linemesh.reverse(SIDE_RP), surfaces.reverse(W_R[1]))),
          cap(SP[2], SP[1], SWM,                        # -y crotch: P3, P4, W-
              (linemesh.reverse(FQ[1]), foot_wall(FQ_FR[1][::-1])),
              (linemesh.reverse(SIDE_RM), shift_wall(surfaces.reverse(W_R[3]), -1)),
              (linemesh.reverse(SIDE_LM), surfaces.reverse(W_L[1])))]
mesh = hexmesh.merge(blocks)

print(hexmesh.report(mesh))
print(hexmesh.topology_report(mesh))

export.to_re2(mesh, OUT_NAME + ".re2", groups=GROUPS)
export.to_vtu(mesh, OUT_NAME + ".vtu", groups=GROUPS)
