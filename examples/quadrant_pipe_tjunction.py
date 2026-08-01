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

Order
-----

Exact at any ``ORDER``, which took some care: a junction like this is where the
straight-subdivision traps all bite at once.

* Every wall curve is carried as its **surface parametrization** (:class:`Wall`), not
  as sampled points, and meshed with ``LineMesh.loft_curve`` -- so the footprint, the
  side transitions and the bypass are all evaluated on the true curve at every node.
  Sampling them into arrays and calling ``LineMesh.loft`` chords the wall from
  ``order > 1`` on.
* A leg's transition is a ``HexMesh.loft_curve`` over the blend parameter, not a
  ``loft`` of a section stack.  ``loft`` is straight **along the sweep**, so its wall
  nodes would be chords between stations at different ``phi`` -- measured here at
  order 3, that alone put them 7.2e-4 off the cylinder.
* The cap blocks are ``evaluated_block``s -- nested ``loft_curve`` over an explicit
  map on the unit cube -- rather than ``HexMesh.from_grid``, which takes corners and
  blends straight.  A ``from_grid`` cap is fine at order 1 and at ``order > 1`` both
  leaves the wall and disagrees with ``quadrant_ogrid``'s bowed ring bands along the
  faces they share, which ``merge`` rightly rejects.

Order 1 remains a strict no-op: the corner coordinates are bit-identical at ``ORDER``
1, 2, 3 and 4, and every wall node lies on the main or branch cylinder to 2.2e-16 at
all of them.
"""

import logging
from collections import namedtuple

import numpy as np

from nekmeshpy import HexMesh, LineMesh, QuadMesh, export
from nekmeshpy.model.interp import coons_grid

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

PHI_W = np.deg2rad(112.5)    # bypass edge: the two z = 0 wall corners, at +-PHI_W

N_TRANS = 5                  # layers from a leg's composite face to its plain disc
N_LEG = 6                    # layers from the plain disc to the end plane
N_BRANCH = 8                 # layers in the branch, footprint -> opening

ORDER = 3                    # exact at any order; see the module docstring

OUT_NAME = "quadrant_pipe_tjunction"
GROUPS = {"wall": "W  ", "inlet": "v  ", "outlet": "O  ", "branch": "O  "}

N = N_QUAD
ORIGIN = np.zeros(3)

#: The seam sampling ``quadrant_ogrid`` demands -- the same for every seam, because
#: every block shares ``N_QUAD`` / ``RADIAL`` / ``CENTER_SCALE``.
FR = QuadMesh.quadrant_seam_fractions(N_QUAD, RADIAL, CENTER_SCALE)

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
Wall = namedtuple("Wall", "g fr")


def cyl_pts(u):
    """``(K, 2)`` wall parameters to ``(K, 3)`` points."""
    return cyl(u[:, 0], u[:, 1])


def wall_mesh(w):
    """A :class:`Wall` as a ``LineMesh`` on the cylinder, exact at every node."""
    return LineMesh.loft_curve(lambda x: cyl_pts(w.g(x)), w.fr, order=ORDER)


def ruled_wall(pa, pb):
    """The straight ``(phi, z)`` segment between two stations, on ``[0, 1]``."""
    pa, pb = np.asarray(pa, dtype=float), np.asarray(pb, dtype=float)
    return Wall(lambda x: (1.0 - np.asarray(x, dtype=float)[:, None]) * pa
                + np.asarray(x, dtype=float)[:, None] * pb,
                np.linspace(0.0, 1.0, 2 * N_QUAD + 1))


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
    """The same curve traversed the other way: ``loft_curve`` takes a descending
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
    return LineMesh.line(center, target, FR, order=ORDER)


# -- sections -----------------------------------------------------------------
def disc(pieces):
    """A full-disc section: four ``(arc, seam1, seam2)`` quadrants merged.

    The winding is load-bearing.  ``quadrant_ogrid`` orients a block by
    ``(A1 - O) x (A2 - O)``, which must point along the sweep direction or the swept
    hexes come out inverted -- so ``pieces`` must run around the disc the right way.
    Handing each shared radius in as the same ``LineMesh`` object is what makes
    neighbours weld bit-exactly instead of to a tolerance."""
    return QuadMesh.merge([
        QuadMesh.quadrant_ogrid(arc, s1, s2, RADIAL, center_scale=CENTER_SCALE,
                                wall_tag="wall")
        for arc, s1, s2 in pieces])


def arc_disc(arcs, z_center):
    """A main-pipe section from its four wall arcs and the height of its centre; each
    shared radius is built once and handed to both of its quadrants."""
    c = np.array([0.0, 0.0, z_center])
    seams = [seam(a.points[0], c) for a in arcs]
    seams.append(seams[0])
    return disc([(arcs[q], seams[q], seams[q + 1]) for q in range(4)])


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
FQ_FR = [LineMesh.arclength_fractions(footprint, 2 * N_QUAD,
                                      t_range=(TQ[q], TQ[q + 1])) for q in range(4)]
#: The footprint quadrant arcs: 0 = ``A`` faces ``+z``, 1 = ``D`` faces ``-y``,
#: 2 = ``C`` faces ``-z``, 3 = ``B`` faces ``+y``.
FQ = [LineMesh.loft_curve(footprint, fr, order=ORDER) for fr in FQ_FR]

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

COMPOSITE_R = disc([(FQ[0].reverse(), SP[1], SP[0]),      # P4 -> P1
                    (SIDE_RP, SP[0], SWP),
                    (BYPASS, SWP, SWM),
                    (SIDE_RM, SWM, SP[1])])

COMPOSITE_L = disc([(FQ[2].reverse(), SP[3], SP[2]),      # P2 -> P3
                    (SIDE_LM, SP[2], SWM),
                    (BYPASS.reverse(), SWM, SWP),
                    (SIDE_LP, SWP, SP[3])])


# -- the crotch caps: an octant of a 3-D O-grid -------------------------------
def core_grid(seam1, seam2, arc):
    """A quadrant's core patch and its far corner ``K``.

    :meth:`QuadMesh.quadrant_core <nekmeshpy.quadmesh.QuadMesh.quadrant_core>` is the
    same construction ``quadrant_ogrid`` builds its own core with, which is the whole
    point: the cap blocks below have to land on exactly those points to weld to the
    three quadrant faces around them."""
    o = seam1.points[0]
    return (QuadMesh.quadrant_core(arc, seam1, seam2, center_scale=CENTER_SCALE),
            o + CENTER_SCALE * (arc.points[N] - o))


def polyline(v):
    """The uniform piecewise-linear map ``[0,1] -> v`` through ``v``'s ``m+1`` rows.

    Piecewise-linear, not a global fit: it is what the toolkit puts on a
    straight-sided edge at order N, so a cap block built on this reproduces the
    quadrant's own node positions rather than merely passing through the same
    corners."""
    v = np.asarray(v, dtype=float)
    m = v.shape[0] - 1

    def f(s):
        s = np.clip(np.asarray(s, dtype=float), 0.0, 1.0) * m
        i = np.clip(np.floor(s).astype(int), 0, m - 1)
        w = s - i
        return (1.0 - w if v.ndim == 1 else (1.0 - w)[:, None]) * v[i] + (
            w if v.ndim == 1 else w[:, None]) * v[i + 1]

    return f


def bilinear(grid):
    """The uniform piecewise-bilinear map ``[0,1]**2 -> grid`` -- the two-dimensional
    twin of :func:`polyline`, and what a straight-sided quad patch's nodes are."""
    g = np.asarray(grid, dtype=float)
    m = g.shape[0] - 1

    def f(x, y):
        x = np.clip(np.asarray(x, dtype=float), 0.0, 1.0) * m
        y = np.clip(np.asarray(y, dtype=float), 0.0, 1.0) * m
        i = np.clip(np.floor(x).astype(int), 0, m - 1)
        j = np.clip(np.floor(y).astype(int), 0, m - 1)
        a, b = (x - i)[:, None], (y - j)[:, None]
        return ((1 - a) * (1 - b) * g[i, j] + a * (1 - b) * g[i + 1, j]
                + (1 - a) * b * g[i, j + 1] + a * b * g[i + 1, j + 1])

    return f


def coons_fn(cb, ct, cl, cr):
    """A transfinite patch of four boundary *functions*: the continuous form of
    ``model.interp.coons_grid``, evaluable at any node lattice."""
    def f(x, y):
        xa = np.asarray(x, dtype=float)[:, None]
        ya = np.asarray(y, dtype=float)[:, None]
        z, u = np.zeros(1), np.ones(1)
        p00, p10, p01, p11 = cb(z)[0], cb(u)[0], ct(z)[0], ct(u)[0]
        return ((1 - ya) * cb(x) + ya * ct(x) + (1 - xa) * cl(y) + xa * cr(y)
                - ((1 - xa) * (1 - ya) * p00 + xa * (1 - ya) * p10
                   + (1 - xa) * ya * p01 + xa * ya * p11))

    return f


def evaluated_block(x_map, fx, fy, fz, last_tag=""):
    """A structured block from a vectorized map on the unit cube, **evaluated at every
    node** rather than subdivided between corners.

    This is what ``from_grid`` cannot do: it takes corners and blends straight, so at
    ``order > 1`` its edges are chords.  Nesting the three rungs' ``loft_curve`` --
    each of which calls its ``f`` at the whole GLL-refined lattice, not just the
    corner levels -- puts every node on the true map instead, which is what lets the
    cap agree with ``quadrant_ogrid``'s bowed ring bands and stay on the wall."""
    def profile(y, z):
        return LineMesh.loft_curve(
            lambda x: x_map(x, np.full(np.shape(x), y), np.full(np.shape(x), z)),
            fx, order=ORDER)

    def section(z):
        return QuadMesh.loft_curve(lambda y: profile(y, z), fy, order=ORDER)

    return HexMesh.loft_curve(section, fz, order=ORDER, last_tag=last_tag)


def cap(sa, sb, sc, ab, bc, ca):
    """Fill the corner region bounded by the three quadrant faces ``(sa, sb, ab)``,
    ``(sb, sc, bc)``, ``(sc, sa, ca)`` and the wall triangle between their arcs.  Each
    of ``ab``/``bc``/``ca`` is that arc as a ``(LineMesh, Wall)`` pair -- the mesh to
    share points with, the parametrization to evaluate the wall on.

    Index the corner by ``(i, j, k)`` along the three seams.  The ``n x n x n`` core
    cube's inner faces ``k=0`` / ``j=0`` / ``i=0`` *are* the three quadrant cores;
    its three outer faces each carry an ``n x n x Nradial`` slab out to a third of the
    wall triangle, and every slab side face is half of a neighbouring quadrant's ring
    band computed by that factory's own blend."""
    (m_ab, w_ab), (m_bc, w_bc), (m_ca, w_ca) = ab, bc, ca
    core_ab, k_ab = core_grid(sa, sb, m_ab)
    core_bc, k_bc = core_grid(sb, sc, m_bc)
    core_ca, k_ca = core_grid(sc, sa, m_ca)
    t = np.arange(N + 1, dtype=float) / N
    fx = np.linspace(0.0, 1.0, N + 1)

    # the three arc midpoints and their centroid, read off the parametrizations so
    # that all three share one coherent branch of phi.
    u_ab, u_bc, u_ca = (w.g(w.fr[N:N + 1])[0] for w in (w_ab, w_bc, w_ca))
    wc = (u_ab + u_bc + u_ca) / 3.0

    # The core cube's far corner.  Scaling it like the K's -- or, worse, averaging
    # them -- puts it in their own plane, and three coplanar edges at a corner is a
    # flat cell.  A cube octant has its seam ends at d, its face corners at d*sqrt(2)
    # and its far corner at d*sqrt(3); ``quadrant_ogrid`` already places M at
    # ``center_scale/sqrt(2)`` and K at ``center_scale``, so the matching far corner
    # is ``center_scale*sqrt(3/2)`` along the wall triangle's centroid ray.
    o = sa.points[0]
    cc = o + CENTER_SCALE * np.sqrt(1.5) * (cyl_pts(wc[None, :])[0] - o)

    def chord(p, q):
        return p + t[:, None] * (q - p)

    # the cube's three outer faces: a Coons patch of two core perimeter halves and
    # two chords running into the far corner.
    f_i = coons_grid(core_ab[N], chord(k_ca, cc), core_ca[:, N], chord(k_ab, cc),
                     t, t)                                       # [j][k]
    f_j = coons_grid(core_ab[:, N], chord(k_bc, cc), core_bc[N], chord(k_ab, cc),
                     t, t)                                       # [i][k]
    f_k = coons_grid(core_ca[N], chord(k_bc, cc), core_bc[:, N], chord(k_ca, cc),
                     t, t)                                       # [i][j]

    cube = np.empty((N + 1, N + 1, N + 1, 3))
    for m in range(N + 1):                        # a Coons level per k; k = 0 and
        cube[:, :, m, :] = coons_grid(core_ca[m], f_j[:, m],   # k = n reproduce the
                                      core_bc[:, m], f_i[:, m], t, t)  # given faces

    # the wall triangle, split into three patches about that same centroid.  All of it
    # is done in the cylinder's own (phi, z) parameters, so no node leaves the wall.
    def half(w, i0, i1):
        """One half of an arc, reparametrized onto ``[0,1]`` **through its own node
        values** -- a plain linear remap would move the shared nodes off the arc when
        its spacing is graded, as the footprint's arc-length spacing is."""
        rep = polyline(w.fr[np.arange(i0, i1 + (1 if i1 > i0 else -1),
                                      1 if i1 > i0 else -1)])
        return lambda s: w.g(rep(s))

    def spoke(mid):
        return lambda s: mid + np.asarray(s, dtype=float)[:, None] * (wc - mid)

    w_a = coons_fn(half(w_ab, 0, N), spoke(u_ca), half(w_ca, 2 * N, N), spoke(u_ab))
    w_b = coons_fn(half(w_ab, 2 * N, N), spoke(u_bc), half(w_bc, 0, N), spoke(u_ab))
    w_c = coons_fn(half(w_ca, 0, N), spoke(u_bc), half(w_bc, 2 * N, N), spoke(u_ca))

    def slab(base, wall, sm, swap):
        """One radial band: the cube's outer face blended out to its wall patch over
        ``RADIAL``.  That blend *is* ``quadrant_ogrid``'s ring formula, so the two side
        faces it shares with the neighbouring quadrants come out node-for-node the
        same.  ``swap`` fixes the one slab whose natural axis order is left-handed."""
        b = bilinear(base)

        def x_map(x, y, tau):
            p, q = (y, x) if swap else (x, y)
            s = np.asarray(tau, dtype=float)[:, None]
            return (1.0 - s) * b(p, q) + s * cyl_pts(wall(p, q))

        # The slab's seam column is *derived*, not snapped.  ``quadrant_ogrid`` pins
        # each ring's two ends onto the seam's own stations; for a straight seam
        # sampled at ``quadrant_seam_fractions`` the blend already lands exactly
        # there, which is why nothing here has to reproduce that snap.  Bow a seam or
        # hand-sample one and the identity quietly stops holding, so check it.
        z1 = np.zeros(1)
        col = ((1.0 - RADIAL[:, None]) * b(z1, z1)
               + RADIAL[:, None] * cyl_pts(wall(z1, z1)))
        if not np.allclose(col, sm.points[N:], rtol=0.0, atol=1e-12):
            raise ValueError(
                "cap: the seam column does not reproduce the seam's own ring "
                "stations, so this slab would not weld to the quadrant faces beside "
                "it -- the seams must be straight and sampled at "
                "QuadMesh.quadrant_seam_fractions(N_QUAD, RADIAL, CENTER_SCALE)")
        return evaluated_block(x_map, fx, fx, RADIAL, last_tag="wall")

    return [HexMesh.from_grid(cube, order=ORDER),
            slab(f_i, w_a, sa, False), slab(f_j, w_b, sb, True),
            slab(f_k, w_c, sc, False)]


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
        return arc_disc([wall_mesh(blend_wall(walls[q], w_plain[q], s))
                         for q in range(4)], s * z)

    plain = station(1.0)
    return [HexMesh.loft_curve(station, np.linspace(0.0, 1.0, N_TRANS + 1),
                               order=ORDER),
            HexMesh.extrude(plain, L_MAIN - Z_NEAR, N_LEG,
                            axis=(0.0, 0.0, float(sign)), last_tag=end_tag)]


def branch():
    """The branch stub: the footprint disc morphed to the circular opening."""
    open_arcs = [LineMesh.loft_curve(opening, fr, order=ORDER) for fr in FQ_FR]
    t = np.linspace(0.0, 1.0, N_BRANCH + 1)
    walls = [LineMesh.blend(f, o, t) for f, o in zip(FQ, open_arcs)]
    c_open = np.array([H_BRANCH, 0.0, 0.0])
    sections = []
    for i in range(t.size):
        arcs = [w[i] for w in walls]
        seams = [seam(a.points[0], t[i] * c_open) for a in arcs]
        seams.append(seams[0])
        sections.append(disc([(arcs[q], seams[q], seams[q + 1]) for q in range(4)]))
    return [HexMesh.loft(sections, last_tag="branch")]


blocks = (leg(COMPOSITE_R, W_R, 1, "outlet")
          + leg(COMPOSITE_L, W_L, -1, "inlet") + branch()
          # A crotch's three arcs must share one branch of phi, so the two that are
          # authored a full turn away in a leg's unwrapped list are shifted back.
          + cap(SP[0], SP[3], SWP,                      # +y crotch: P1, P2, W+
                (FQ[3].reverse(), foot_wall(FQ_FR[3][::-1])),
                (SIDE_LP.reverse(), shift_wall(reverse_wall(W_L[3]), 1)),
                (SIDE_RP.reverse(), reverse_wall(W_R[1])))
          + cap(SP[2], SP[1], SWM,                      # -y crotch: P3, P4, W-
                (FQ[1].reverse(), foot_wall(FQ_FR[1][::-1])),
                (SIDE_RM.reverse(), shift_wall(reverse_wall(W_R[3]), -1)),
                (SIDE_LM.reverse(), reverse_wall(W_L[1]))))
mesh = HexMesh.merge(blocks)

print(mesh.report())
print(mesh.topology_report())

export.to_re2(mesh, OUT_NAME + ".re2", groups=GROUPS)
export.to_vtu(mesh, OUT_NAME + ".vtu", groups=GROUPS)
