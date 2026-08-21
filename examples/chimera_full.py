r"""Full chimera manifold: two chimera ports fed by a serpentine coil.

The whole assembly, inlet to outlet:

    riser -> T1 -> (main leg) elbow up into chimera's own opening
                \-- (branch, -y) --> T2 --> bridge inward, bend to +z
                                              --> the serpentine coil

Two of those, mirrored about the coil's midspan: the T1 on the **negative-x**
side feeds chimera's *outlet*, the one on the **positive-x** side its
*inlet*.

Each T1 branch does not carry one T2 but a **chain of ``N_T2``**, each hanging
off the previous one's own ``-y`` leg at ``T2_SPACING`` intervals; only the
last is capped.  The two chains are mirror images, so their ``k``-th junctions
face each other across their own copy of the serpentine coil -- ``N_T2``
parallel coils stacked down ``-y``, each planar in its own x-z plane.  The
coil's shape is traced from a reference photo and is **fixed**: it is only
placed, never rescaled (see ``serpentine_pipe.py``, whose ``MOVES`` move table
this file imports directly and which also builds the coil standing alone).  The
chimera ports sit exactly ``H1 + RUN_T1_T2`` = 10 above the first coil in y.

    PYTHONPATH=. python examples/chimera_full.py

Produces ``chimera_full.re2`` and ``chimera_full.vtu``.

``FAST`` (default ``True``) swaps the real ``chimera.py`` build -- 350k
elements, a couple of minutes -- for two short capped stubs carrying its exact
inlet/outlet disc pattern, so the manifold's own geometry can be checked in
seconds.  Set it ``False`` for the whole thing.

Meeting chimera: the boundary layer
-----------------------------------

``chimera.py`` grows a boundary layer over its whole fluid wall, so a chimera *port* is
no longer the plain pipe cross-section -- it is that section plus the ring the skin
leaves round an opening it deliberately does not skin: **80 quads over 89 points where
the bare section is 48 over 57**.  Nothing can be morphed across that difference.
``adapter``, ``bridge`` and ``loft_between`` all need the two patterns to pair
one-for-one, and 48 quads do not pair with 80 however gently they are blended.

So the manifold grows the *same* layer.  Every block here is meshed at the **core**
radii (``RC_MAIN`` / ``RC_BR``, inset by ``T_BL``), assembled exactly as before, and
skinned once at the end by ``tjunction_lib.skin_wall`` -- one surface, so the layer
crosses every seam this file welds rather than meeting itself at a different angle on
each side of one.  Its chimera-facing ends are *named* and therefore left unskinned, so
each comes out ringed by the shell exactly as chimera's own openings are, by the same
construction rather than by imitation.

That leaves only the pattern to match, and ``chimera.py`` picks its three main-pipe
parameters to make it exact: at ``N_THETA_MAIN=16`` over a 4-per-side core with T1's own
``CENTER_SCALE_MAIN=0.75`` and ``RADIAL_MAIN``, ``quadmesh.ogrid``'s section and
``build_eqtee``'s ``spined_ogrid`` disc come out with byte-identical
``quads``/``orient``/``lines`` -- both bottom out in ``quadrant_ogrid`` quarters -- and
node-for-node identical geometry.  Measured across the finished seam: **5.6e-15**.  (At
``N_THETA_MAIN=24`` the port is 84 quads and nothing here can pair with it at all; 16 is
what T2's own resolution pins it to, since T2 is left alone.)

What is still not shared is *numbering* and *phase*.  Each port was numbered by its own
``boundary_mesh`` extraction, and the manifold's arrives at whatever roll its sweep
frame carried it to.  ``template=`` settles the first -- the manifold's port is handed
chimera's own B-rep and filled with the manifold's real coordinates, so point ``i`` is
the same node on both sides -- and ``loft_between``'s twist settles the second, which is
why the last ``ADAPT`` is still a lofted block and not a plain weld.

Fluid and solid
---------------

Only ``chimera.py`` contributes solid: the jacket around each hairpin's straight
run, which arrives with the rest of the chain since this file runs that script
whole rather than reproducing it.  Nothing here is near it -- the manifold sits
below ``z = -17.5`` at ``y <= 6.6`` and the slab lives at ``y >= 10`` -- so it
comes through untouched.  Two things here have to not undo it: chimera's stale
inlet/outlet tags are dropped at the merge, and its ``"insulated"`` faces must
survive that; and every block this file builds names itself ``"fluid"`` at its
own call site, so the region partition covers the assembly rather than stopping
at chimera's boundary.  In ``FAST`` mode there is no chimera and so no solid.

Three seam techniques earn their keep here, all for the same reason -- pieces
built by *different* constructions have to meet exactly, and at ``order > 1``
``HexMesh.merge`` verifies shared high-order edge/face nodes to
``conform.entity_tol`` (~1e-9 x the model extent), far tighter than any
coordinate weld:

* ``quadmesh.reindex`` -- re-express one section's geometry through another's
  index labels.  A pure permutation, so it is exact by construction where a
  coordinate rotation is only approximate.
* ``hexmesh.adapter`` -- blend across a *small* pattern difference (the ~0.03
  between T1's own leg and chimera's), first slice and last slice both exact.
* ``hexmesh.bridge`` -- span a *large* one as a single ``HexMesh.loft``: rigid stubs
  off each side, a blend across the gap between them, no internal merge to fail.  What
  is left for it is the coil's own far end against T2_out's bend, two genuinely
  different patterns landing ``GAP_Z`` apart.

All three were written here first and now live in the toolkit; this file keeps
only the choice of which to use at each seam, which is the part that is about
this manifold rather than about meshes in general.

And where a seam can be removed rather than made exact, it is: the inbound
connector and the coil sweep as ONE turtle walk (see ``build_coil``).  They
used to be two pieces welded together, which broke as soon as the coils were
stacked: the weld then fused by rounding to a ``tol``-sized *bucket*, so the
~1 ULP disagreement between the connector's last layer and the coil's
recomputed first one failed to weld wherever a coordinate happened to land on a
bucket edge, pinching the surface open.  The weld is a plain radius now and
would survive that, but a seam that does not exist cannot open at all.
"""

import os
import sys

import numpy as np
from scipy.spatial import cKDTree

from nekmeshpy import hexmesh, linemesh, quadmesh, writer
from nekmeshpy.core import conform, paths
from nekmeshpy.quadmesh._helpers import _elevate as _quad_elevate
from nekmeshpy.quadmesh.quadmesh import QuadMesh

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# serpentine_pipe's own mesh build is guarded behind `if __name__ == "__main__"`,
# so importing it for these two names alone costs nothing beyond them.
from serpentine_pipe import MOVES as COIL_MOVES_LIB  # noqa: E402  (needs the path above)
from serpentine_pipe import TARGET_LEN as COIL_TARGET_LEN_LIB  # noqa: E402
from tjunction_lib import build_cob, build_eqtee, skin_wall  # noqa: E402


def merge_at(blocks, distance):
    """``hexmesh.merge`` welding at an absolute ``distance`` in model units.

    ``merge``'s ``tol`` is a *fraction* of the assembly's largest x/y/z range, which is
    the right default -- but every tolerance in this file is known absolutely, not
    proportionally: a measured ~0.03 residual to bridge and a real 0.05 feature not to
    fuse.  Writing those as pre-divided fractions would bury the two numbers the choice
    actually turns on, so divide here instead, by the same scale ``weld_points`` will."""
    return hexmesh.merge(
        blocks, tol=distance / conform.bbox_scale(
            np.vstack([b.points for b in blocks])))


def _twist_slices(a, b, fracs):
    """Index-paired intermediate sections between ``a`` and ``b`` (same ``quads``),
    each point carried along a helix about the a->b axis rather than the straight
    chord ``quadmesh.blend`` would use: radius and axial position interpolate
    linearly, but the angle *around* the axis takes the short way rather than
    cutting across the disc. Point ``i`` of ``a`` and point ``i`` of ``b`` are index-
    paired but not angularly aligned (the two mesher families' own conventions differ
    by a rotation no reindexing removes -- see loft_between), so the straight chord
    a plain blend takes between two out-of-phase points passes close to the axis,
    pinching the block; a rotation about the shared axis does not.

    Only the interior fractions are built here (0 and 1 are literally ``a``/``b``,
    unchanged, so the block's own two ends stay bit-exact for whatever the caller
    welds them to)."""
    ca, cb = a.points.mean(axis=0), b.points.mean(axis=0)
    axis = cb - ca
    axis = axis / np.linalg.norm(axis)
    up = np.array([0.0, 0.0, 1.0]) if abs(axis[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    e1 = up - axis * np.dot(up, axis)
    e1 = e1 / np.linalg.norm(e1)
    e2 = np.cross(axis, e1)

    def polar(pts, c):
        rel = pts - c
        return rel @ axis, np.hypot(rel @ e1, rel @ e2), np.arctan2(rel @ e2, rel @ e1)

    ax_a, r_a, th_a = polar(a.points, ca)
    ax_b, r_b, th_b = polar(b.points, cb)
    dth = np.mod(th_b - th_a + np.pi, 2.0 * np.pi) - np.pi   # shortest turn, signed

    slices = []
    for t in fracs:
        if t <= 0.0:
            slices.append(a)
            continue
        if t >= 1.0:
            slices.append(b)
            continue
        c_t = (1.0 - t) * ca + t * cb
        ax_t = (1.0 - t) * ax_a + t * ax_b
        r_t = (1.0 - t) * r_a + t * r_b
        th_t = th_a + t * dth
        pts = (c_t + ax_t[:, None] * axis
               + (r_t * np.cos(th_t))[:, None] * e1
               + (r_t * np.sin(th_t))[:, None] * e2)
        linear = QuadMesh.from_corners(pts, a.corners)
        slices.append(_quad_elevate(linear, a.order))
    return slices


def _align(a, b):
    """``b`` relabelled through ``a``'s own index labels, paired by nearest node once
    both are centred -- a pure relabelling, so ``b``'s geometry rides into the result
    untouched and whatever already carries it still welds.

    :func:`loft_between` twists point ``i`` of ``a`` onto point ``i`` of ``b``, which is
    only meaningful if the two indices name the *same node of the pattern*.  Two discs
    can share connectivity exactly and still not agree on that: the cob's
    ``quadmesh.ogrid`` section and ``build_eqtee``'s ``spined_ogrid`` disc have identical
    ``quads``/``orient``/``lines`` but number them from different starts, so index
    pairing them asks the twist for a roll of most of a turn -- and the nodes near the
    hub then sweep through the axis and the block folds (measured: both windings came
    back inverted).  Matching by position first leaves the twist only the residual phase.

    Returns ``b`` unchanged if the match is not a permutation, so a caller whose two
    sections genuinely are index-paired but geometrically far apart is no worse off."""
    if not (np.array_equal(a.quads, b.quads) and np.array_equal(a.orient, b.orient)
            and np.array_equal(a.line_mesh.lines, b.line_mesh.lines)):
        return b
    _, sigma = cKDTree(b.points - b.points.mean(axis=0)).query(
        a.points - a.points.mean(axis=0))
    if len(set(sigma.tolist())) != sigma.size:
        return b
    return quadmesh.reindex(a, b, sigma)


def loft_between(a, b, n_layers, element_tags=None):
    """Twist-loft two discs that share ``quads``/``orient`` connectivity even though
    they come from different mesher families -- eqtee's ``spined_ogrid`` disc, the cob's
    ``quadmesh.ogrid`` section and the quadrant construction's disc all bottom out in
    ``quadrant_ogrid`` quarters, so at matching resolution they turn out identical
    (verified, not assumed).  That is a weaker guarantee than it sounds: identical
    connectivity does not mean identical *numbering*, so :func:`_align` pairs the two by
    position first, unlike ``hexmesh.adapter``/``bridge`` (built for sections that only
    differ in *pattern*, not connectivity, and can only approximate curvature they
    cannot exactly reconcile).  What is left after that is phase, so
    :func:`_twist_slices` carries each point there by rotating about the connector's
    own axis instead of ``quadmesh.blend``'s straight chord, which would cut across
    the disc and pinch the block at the middle station (measured: a straight blend
    between out-of-phase discs waists to well under the wall radius; the same discs
    twist-lofted hold their radius across every station).

    The one thing phase does not fix is which way each disc's CCW winding faces -- the
    two families' own "outward" convention can come out opposite even on identical
    connectivity -- so this tries ``a`` as given and, if that comes out inverted, a
    copy of ``a`` flipped 180 degrees about an in-plane axis instead. Only ``a`` is
    ever touched: ``b``'s own points/interior ride into the result verbatim (the
    ``t=1`` end), which is what lets the caller weld the result to whatever else
    already carries ``b``'s own true geometry."""
    b = _align(a, b)
    fracs = np.linspace(0.0, 1.0, n_layers + 1)

    def _try(aa):
        slices = _twist_slices(aa, b, fracs)
        block = hexmesh.loft(slices, element_tags=element_tags)
        return block, float(hexmesh.scaled_jacobian(block).min())

    try:
        block, m = _try(a)
        if m > 0.0:
            return block
    except ValueError:
        m = -1.0
    na = quadmesh.plane_normal(a)
    axis = (0.0, 1.0, 0.0) if abs(na[1]) < 0.9 else (1.0, 0.0, 0.0)
    a_flip = quadmesh.rotate(a, np.pi, axis=axis, center=a.points.mean(axis=0))
    block2, m2 = _try(a_flip)
    if m2 <= 0.0:
        raise ValueError(
            "loft_between: both windings of a gave an inverted block (min sj %.4g, "
            "%.4g) -- a and b may not actually share connectivity" % (m, m2))
    return block2


FAST = False
ORDER = 2
N_HALF = 8
RADIAL = np.array([0.0, 0.4, 0.8, 1.0])
CENTER_SCALE = 0.5
n_slices = 3

R_MAIN = 1.2     # == chimera's R_MAIN, *finished*
R_BR = 0.5       # == chimera's R_BRANCH, finished

#: chimera's own boundary layer, restated here because this manifold has to grow the
#: **same** one.  A chimera port is not the plain pipe section any more: it is that
#: section plus the ring the skin leaves round an opening it deliberately does not skin,
#: 80 quads over 89 points where the bare section is 48 over 57.  Nothing can be morphed
#: across that difference -- ``adapter`` / ``bridge`` / ``loft_between`` all need the two
#: patterns to pair one-for-one -- so the manifold is meshed at the **core** radii and
#: skinned back out at the very end, exactly as chimera is, and its port comes out the
#: same 80 quads by the same construction rather than by imitation.
T_BL = 0.035 * R_MAIN
BL = T_BL * np.array([0.0, 0.6, 1.0])
RC_MAIN = R_MAIN - T_BL
RC_BR = R_BR - T_BL

# == chimera's own names.  The regions are element_tags (every element carries one,
# manifold included); SOLID_FACE_TAG is a face_tags / GROUPS name, deliberately not
# the same string as the "solid" region -- see chimera.py's own note.
FLUID_TAG = "fluid"
SOLID_FACE_TAG = "insulated"
#: chimera's conjugate interface, which arrives with the chain.  Its two sides want
#: different conditions and a face carries one name, so the asymmetry is keyed by the
#: region of the element each exported row belongs to -- see GROUPS below.
INTERFACE_TAG = "interface"

L1 = 2.5 * R_MAIN     # T1 main half-length
H1 = 2.5 * R_MAIN     # T1 branch length
L2 = 1.2               # T2's main-leg offset from its own centre (build_tjunction's Z_NEAR default)

kw_t1 = dict(n_half=N_HALF, order=ORDER, radial=RADIAL, center_scale=CENTER_SCALE,
             n_slices_a=n_slices, n_slices_b=n_slices, n_slices_branch=n_slices)
kw_t2 = kw_t1


#: chimera's own pipe cross-section, reproduced from its recipe rather than imitated:
#: ``build_cob`` meshes the main pipe as ``quadmesh.ogrid`` of a 16-cell circle over a
#: 4-per-side core, and chimera picks ``CENTER_SCALE_MAIN`` / ``RADIAL_MAIN`` to be T1's
#: own -- so this and ``build_eqtee``'s ``spined_ogrid`` disc at ``n_half=8`` come out
#: with byte-identical ``quads`` / ``orient`` / ``lines`` (both bottom out in
#: ``quadrant_ogrid`` quarters) *and* node-for-node identical geometry, measured 7.6e-16.
#: That equality is the whole reason this file can meet chimera at all.
_CHI_SECTION = quadmesh.ogrid(linemesh.circle(RC_MAIN, 2 * N_HALF, order=ORDER),
                              N_HALF // 2, np.array([0.0, 0.6, 1.0]),
                              center_scale=0.75, wall_tag="wall")


def _skinned_port():
    """The chimera **port** pattern: :data:`_CHI_SECTION` with the ring the boundary
    layer leaves round an unskinned opening, built by running a stub of pipe through the
    very same :func:`tjunction_lib.skin_wall` chimera uses. Derived, not described -- the
    ring's radial stations and its numbering both fall out of the skin rather than being
    restated here, so this cannot drift from what chimera actually builds."""
    stub = hexmesh.extrude(_CHI_SECTION, 1.0, 1, axis=(0.0, 0.0, 1.0),
                           element_tags=FLUID_TAG, last_tag="port")
    return hexmesh.boundary_mesh(skin_wall(stub, BL, element_tag=FLUID_TAG), "port")


#: Built once: every use is the same pattern at a different place.  Stripped of the
#: ``"port"`` name the extraction stamped on it -- this is a *pattern*, and every disc
#: templated from it inherits its ``element_tags``, which would then ride onto the morph
#: block's own cap and collide with the manifold's own port name at the weld.
_CHI_PORT = quadmesh.retag_element(_skinned_port(), {"port": ""})


def fake_chi_disc(center):
    """The port pattern centred on ``center``, normal ``+z`` -- chimera's own
    inlet/outlet convention."""
    return quadmesh.translate(_CHI_PORT,
                              np.asarray(center) - _CHI_PORT.points.mean(axis=0))


# chimera's REAL inlet/outlet disc centres (probed: both at z = -17.5,
# facing -z -- the chain body extends up to z = +160). Our whole manifold
# therefore sits BELOW that plane and every pipe that meets chimera arrives
# heading +z, face to face with chimera's own -z-facing openings.
CHI_IN = np.array([0.0, 0.0, -17.5])
CHI_OUT = np.array([-15.0, 6.6, -17.5])

# The real chain is built FIRST when it is wanted, so the connectors can be
# fitted to its own ports rather than to a stand-in (see port_disc).
chi_mesh = None
if not FAST:
    import runpy as _runpy
    print("building real chimera (slow)...")
    _chi_ns = _runpy.run_path(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                           "chimera.py"))
    chi_mesh = _chi_ns["mesh"]

if chi_mesh is None:
    chi_in_disc = fake_chi_disc(CHI_IN)
    chi_out_disc = fake_chi_disc(CHI_OUT)
else:
    # template= reuses fake_chi_disc's B-rep *structure* while every coordinate comes
    # off chimera itself.  A freshly numbered extraction would be exact too, but could
    # not pair index-for-index with T1's own end section, which loft_between needs
    # (a straight index-paired blend, so it never has to reindex either side).
    #
    # Reading the target off the real mesh is what removes the guess: a recipe that
    # reproduces one port exactly need not reproduce the other.  chimera builds its
    # inlet as a straight end_stub (which fake_chi_disc matches to 1e-15) but its
    # outlet through outlet_return()'s bend, whose end disc lands ~3.4e-3 away.  At
    # order 1 that just welds; at order > 1 HexMesh.merge checks shared high-order
    # edge nodes against conform.entity_tol (~2e-7 at this model's extent) and the
    # outlet seam failed on 24 edges by ~1e-3.  Measured residual now 0.0 at both.
    chi_in_disc = hexmesh.boundary_mesh(chi_mesh, "inlet",
                                        template=fake_chi_disc(CHI_IN))
    chi_out_disc = hexmesh.boundary_mesh(chi_mesh, "outlet",
                                         template=fake_chi_disc(CHI_OUT))

# -----------------------------------------------------------------------------
# T1: main and branch are genuinely equal radius here (R_MAIN both), which neither of
# the unequal-radius constructions can mesh -- the quadrant one's footprint curve
# degenerates as R_BRANCH -> R_MAIN (which used to force a 99.9%-of-R_MAIN fudge), and
# the cob's bore has to fit inside the cob, which it cannot at ratio 1. eqtee's collar
# construction is built for exactly this ratio, so T1 uses it, while T2 (genuinely
# unequal: main = R_MAIN, branch = R_BR, ratio 0.42) uses build_cob below.
# That puts the T1-branch <-> T2-main weld across families -- eqtee's spined_ogrid disc
# into the cob's quadmesh.ogrid section -- but at this resolution the two share
# identical quad/orient/line connectivity (verified; both bottom out in quadrant_ogrid
# quarters), so loft_between's blend spans that seam with both sides' true curvature
# intact.  They do not share a *numbering*, which is what _align is for.
# main axis -> x, branch -> -y.
# -----------------------------------------------------------------------------
ROT_T1 = -np.deg2rad(120.0)
AXIS_T1 = (1.0, -1.0, 1.0)


def build_t1(mirror=False):
    tj = build_eqtee(RC_MAIN, L1, H1, order=ORDER, n_half=N_HALF,
                     radial=np.array([0.0, 0.6, 1.0]), n_layers_main=n_slices,
                     n_layers_branch=n_slices, center_scale=0.75,
                     quadrant_scale=0.7, element_tag=FLUID_TAG)
    ang, axis = ROT_T1, AXIS_T1
    core, da, db, dbr = (hexmesh.rotate(tj.core, ang, axis=axis), quadmesh.rotate(tj.disc_minus, ang, axis=axis),
                        quadmesh.rotate(tj.disc_plus, ang, axis=axis), quadmesh.rotate(tj.disc_branch, ang, axis=axis))
    if mirror:
        # T1_out's main pipe "comes in from the opposite side": an extra 180
        # about the branch's own axis (world y) swaps disc_a/disc_b's world
        # positions (main -x <-> +x) while leaving the -y branch direction
        # itself untouched (a rotation, not a mirror -- element quality is
        # unaffected, unlike HexMesh.transform's own reflection warning).
        yax = (0.0, 1.0, 0.0)
        core, da, db, dbr = (hexmesh.rotate(core, np.pi, axis=yax), quadmesh.rotate(da, np.pi, axis=yax),
                            quadmesh.rotate(db, np.pi, axis=yax), quadmesh.rotate(dbr, np.pi, axis=yax))
    return tj._replace(core=core, disc_minus=da, disc_plus=db, disc_branch=dbr)


def elbow_backward(target_pt, target_dir_sign, start_heading_sign, bend_r, run_len,
                    vertical_run=0.0):
    """Solve, in the (x, z) plane, the start point of a [line, 90-arc,
    line] path that starts heading +x (start_heading_sign*x) and ends at
    `target_pt` heading (0,0,target_dir_sign) -- by building the path
    *backwards* from the known target. ``vertical_run`` is an extra straight
    segment *after* the arc (the actual descent/climb -- the arc alone only
    covers ``bend_r``), mirroring the riser's own line+arc+line shape so the
    two legs read as comparably long, not one long riser and a barely-there
    kink. Returns (start_xz, forward_moves, forward_heading)."""
    turn = 90.0 if (target_dir_sign > 0) == (start_heading_sign > 0) else -90.0
    # backward: start at target, heading -target_dir (heading angle pi/2 * sign),
    # first undo the final vertical run, then the arc, then the horizontal line.
    heading0 = np.pi / 2 * (1 if target_dir_sign > 0 else -1) + np.pi
    back = xz_path([paths.line(vertical_run), paths.arc(bend_r, -turn),
                    paths.line(run_len)],
                   (target_pt[0], target_pt[2]), heading0, target_pt[1])
    x0, _, v0 = back.centerline(np.array([1.0]))[0]
    fwd_heading = 0.0 if start_heading_sign > 0 else np.pi
    return ((x0, v0), [paths.line(run_len), paths.arc(bend_r, turn),
                       paths.line(vertical_run)], fwd_heading)


def xz_path(moves, start_xz, heading, y_fixed):
    """A turtle walk in the world ``(x, z)`` plane at fixed ``y`` -- every connector in
    this file lives in such a plane, so ``heading`` stays the 2-D angle from ``+x``
    toward ``+z`` the rest of this file solves in.  A positive turn goes toward the
    walk's left, which in this plane pins the frame's up to ``-y``; the sweeps read that
    frame off the path rather than restating it."""
    return paths.walk(moves, start=(start_xz[0], y_fixed, start_xz[1]),
                      heading=(np.cos(heading), 0.0, np.sin(heading)),
                      up=(0.0, -1.0, 0.0))


def build_bend_mesh(section, start_pt3, moves, heading2d, y_fixed, n_layers, last_tag=""):
    path = xz_path(moves, (start_pt3[0], start_pt3[2]), heading2d, y_fixed)
    return hexmesh.sweep_path(section, path, layers=n_layers, origin=start_pt3,
                              last_tag=last_tag, element_tags=FLUID_TAG)


BEND_R1 = 2.0 * R_MAIN
VERTICAL_DROP = 14.0   # matches VERTICAL_RISE -- the two legs read as comparable
ADAPT = 3.0            # length of the loft_between transition into chimera's own
                       # disc pattern (see loft_between) -- wider than a same-family
                       # adapter needs, since the twist that avoids pinching wants
                       # room to turn rather than being forced through a short gap
RUN_TO_RISER = 8.0
VERTICAL_RISE = 14.0

#: Centre-to-centre distance between the two y-pipes (each T1's branch running
#: down into its T2's main) -- the T1 centres are *placed* at this separation,
#: symmetric about the chimera targets' x midpoint, and each side's horizontal
#: chimera-run length is solved from that, instead of the other way round.
D_YPIPES = 52.0

#: The two manifold-side chimera ports, named per side so each can be paired against its
#: own chimera opening. They are openings, so the skin leaves them alone and instead
#: rings each with its own lateral layer -- which is exactly what makes them chimera's
#: pattern rather than the bare pipe section's.
PORT_TAG_IN = "chi_port_in"
PORT_TAG_OUT = "chi_port_out"

pieces = []


def place_t1(side_center, chi_target, chi_disc, tag, t1_x, port_tag,
             mirror=False):
    t1 = build_t1(mirror=mirror)
    # disc_b faces +x normally, -x when mirrored; disc_a always the opposite.
    b_sign = -1 if mirror else +1
    # Forward path: disc_b (at t1_x + b_sign*L1) runs b_sign*x, the 90-degree
    # arc adds b_sign*BEND_R1 more, then drops straight to chi -- so the run
    # is fixed by where the T1 centre must sit, not the reverse.
    run = b_sign * (chi_target[0] - t1_x) - L1 - BEND_R1
    assert run > 0.5, ("T1 at x=%.1f can't reach chimera at x=%.1f (run=%.2f); "
                       "widen D_YPIPES or move the chimera targets" %
                       (t1_x, chi_target[0], run))
    # target_dir_sign=+1: the connector *rises* into chimera's -z-facing
    # opening (T1 sits below the chimera plane), face to face -- descending
    # onto it (-1, as before the real-chimera flip) would put two -z-facing
    # faces back to back, which cannot weld.
    # the swept tube stops ADAPT short of the chimera plane; the last ADAPT is a
    # loft_between spanning that gap, built after the skin (see join_chimera) because
    # only then does this end carry chimera's own 80-quad port pattern rather than the
    # 48-quad bare section.
    # NOTE: elbow_backward always lands exactly *at* the target point it is
    # given (that is what "backward" means -- it solves the start position to
    # make that true), so the connector must be solved to a point ADAPT short
    # of chimera, not to chimera itself with a shortened run (which only
    # moves T1, not where the tube actually stops).
    chi_target_short = chi_target - np.array([0.0, 0.0, ADAPT])
    (x0, z0), moves, heading = elbow_backward(chi_target_short, +1, b_sign, BEND_R1, run,
                                              vertical_run=VERTICAL_DROP - ADAPT)
    t1_center = np.array([x0 - b_sign * L1, side_center[1], z0])
    assert abs(t1_center[0] - t1_x) < 1e-9
    core = hexmesh.translate(t1.core, t1_center)
    da = quadmesh.translate(t1.disc_minus, t1_center)
    db = quadmesh.translate(t1.disc_plus, t1_center)
    dbr = quadmesh.translate(t1.disc_branch, t1_center)
    db_c = db.points.mean(axis=0)

    def _end_section(disc):
        """The tube's exact end section for a given disc_plus, via the same
        placement machinery sweep() itself uses (not re-derived) -- a cheap
        QuadMesh.transform, no mesh build. Uses db_c[1], not the nominal
        t1_center[1], as the path's own y: T1's quadrant disc isn't perfectly
        centred on t1_center (a ~6e-4 residual, same order as the pattern's
        own near-4-fold-symmetry residual elsewhere), and origin=db_c already
        commits to db's *actual* centroid -- mixing that with the nominal
        t1_center[1] for the path itself put the sweep's start station 6e-4
        off db's own position (confirmed with a standalone HexMesh.sweep
        reproducer: it reproduces its input to machine precision when origin
        and the path's own start agree, and by exactly this residual when
        they don't)."""
        path = xz_path(moves, (db_c[0], db_c[2]), heading, db_c[1])
        end = quadmesh.place_on_path(disc, path, [0.0, 1.0], origin=db_c)[-1]
        return quadmesh.port(end, outward=path.tangent(np.array([1.0]))[0])

    # The tube stops ADAPT short of chimera and is *named* there.  It is not morphed
    # into chimera's pattern here and could not be: this connector is core-radius pipe,
    # and chimera's port is a skinned one.  The skin is grown over the whole manifold at
    # the end, which turns this named cap into the same 80-quad port chimera has -- and
    # only then is the last ADAPT lofted across (see ``join_chimera``).
    conn_chi = build_bend_mesh(db, db_c, moves, heading, db_c[1], n_slices*8,
                               last_tag=port_tag)
    end_port = _end_section(db)
    # chimera's own openings face -z and this connector rises +z into them, so the two
    # face each other; the twist loft below reads that from each port's own normal.
    chi_port = quadmesh.port(chi_disc, outward=(0.0, 0.0, -1.0))
    print("  [%s] end %s" % (tag, end_port))
    print("  [%s] tgt %s" % (tag, chi_port))
    # disc_a -> bends the opposite way, then climbs straight to the riser.
    # The inlet/outlet risers themselves bend to z- (away from chimera, which
    # conn_chi above reaches via +z) -- opposite sign from a naive a_sign-only
    # turn, so both flip regardless of which side is mirrored.
    a_sign = -b_sign
    moves_r = [paths.line(RUN_TO_RISER),
               paths.arc(BEND_R1, -90.0 if a_sign > 0 else 90.0),
               paths.line(VERTICAL_RISE)]
    da_c = da.points.mean(axis=0)
    # da_c[1], not the nominal t1_center[1] -- same reasoning as _end_section
    # above (da is no more exactly centred on t1_center than db is).
    riser = build_bend_mesh(da, da_c, moves_r, 0.0 if a_sign > 0 else np.pi,
                            da_c[1], n_slices*8, last_tag=tag)
    return core, [conn_chi], riser, dbr, t1_center


# T1 on the negative-x side (the assembly's own "inlet" riser) connects to
# chimera's OUTLET; T1 on the positive-x side ("outlet" riser) to its
# INLET.  Which physical port a side reaches is carried entirely by the
# (chi_target, chi_disc) pair -- place_t1 itself is agnostic.
X_MID = 0.5 * (CHI_IN[0] + CHI_OUT[0])
core_in, conn_chi_in, riser_in, br_in, t1c_in = place_t1(
    (0, CHI_OUT[1], 0), CHI_OUT, chi_out_disc, "inlet",
    t1_x=X_MID - D_YPIPES / 2.0, port_tag=PORT_TAG_IN)
print("t1c_in", t1c_in)
core_out, conn_chi_out, riser_out, br_out, t1c_out = place_t1(
    (0, CHI_IN[1], 0), CHI_IN, chi_in_disc, "outlet",
    t1_x=X_MID + D_YPIPES / 2.0, port_tag=PORT_TAG_OUT, mirror=True)
print("t1c_out", t1c_out, "| y-pipe separation:", t1c_out[0] - t1c_in[0])

for _nm, _m in [("core_in", core_in), ("conn_chi_in", conn_chi_in[0]), ("riser_in", riser_in),
               ("core_out", core_out), ("conn_chi_out", conn_chi_out[0]), ("riser_out", riser_out)]:
    print("  %s min_sj=%.4e" % (_nm, hexmesh.scaled_jacobian(_m).min()))

pieces += [core_in, *conn_chi_in, riser_in, core_out, *conn_chi_out, riser_out]

mesh1 = merge_at(pieces, 0.005)
# one report, read twice: is_watertight and is_conforming each recompute the
# whole thing, which is seconds apiece at this size.
_rep1 = hexmesh.topology_report(mesh1)
print("stage1:", mesh1.n_hexes, "hexes, watertight",
      _rep1.watertight and _rep1.n_components == 1,
      "conforming", _rep1.conformal, "min sj", hexmesh.scaled_jacobian(mesh1).min())

# -----------------------------------------------------------------------------
# T2: branches off T1's -y branch.  Unequal radius (main = R_MAIN, matching T1's
# branch; branch = R_BR, matching the serpentine) -- the **cob** construction, not the
# equal-radius eqtee and no longer the quadrant one either.  At this ratio (0.42) the
# quadrant junction's crotch cap is its weakest point, and the boundary layer grown over
# it was the worst element in the whole assembly: measured min scaled Jacobian 0.18
# against the cob's 0.47, which is the junction's own figure with the skin costing it
# nothing.  The cob also hands back its main legs as the *plain pipe section*, which is
# byte-identical to T1's own disc here -- so the chain's welds get simpler, not harder.
# Local main axis z, branch axis x
# (tjunction_lib's own convention); rotate -90 about world x so main -> y
# (one leg +y facing back to T1, the other -y a dead end) and branch stays x,
# then bends down (z-) to the serpentine.
# -----------------------------------------------------------------------------
ROT_T2 = -np.pi / 2
AXIS_T2 = (1.0, 0.0, 0.0)
#: T2's branch runs nearly all the way to the coil's own bend rather than
#: stopping a stub's length off the junction body, and carries layers about as
#: long as the coil's rather than being finely sliced right at the junction.
#: Whatever length is left over is swept as the connector's leading straight
#: (see build_coil), which solves its own run from wherever the branch ends --
#: so growing this shortens that automatically and the overall shape does not
#: move.
#:
#: Lengthening costs no shape fidelity: build_cob's branch is an exact cylinder off the
#: bore -- it holds the disc's own (x, z) and carries only the axial coordinate out, so
#: the far end merely relaxes from saddle-shaped to flat more gradually.  The same was
#: true of build_tjunction's branch(), which blended the footprint into the opening, and
#: those two differ *only* in the axial coordinate -- both carry the identical
#: R_BRANCH*(sin, cos) cross-section -- so the blend straightens the section's
#: axial position and never touches its radius.  Measured over the whole run:
#: wall radius = R_BR to 1.7e-15.  A long branch is a straight round pipe whose
#: end plane merely relaxes from saddle-shaped to flat more gradually.
H2_BRANCH = 15.0
N2_BRANCH = max(2, int(round(H2_BRANCH / 2.0)))   # ~2.0-long layers, as the coil uses


def build_t2(mirror=False):
    tj = build_cob(RC_MAIN, RC_BR, H2_BRANCH, order=ORDER, Z_NEAR=L2,
                   N_THETA_MAIN=2 * N_HALF, RADIAL_MAIN=np.array([0.0, 0.6, 1.0]),
                   CENTER_SCALE_MAIN=0.75, N_THETA_BRANCH=2 * N_HALF,
                   RADIAL_BRANCH=np.array([0.0, 0.6, 1.0]),
                   CENTER_SCALE_BRANCH=0.75, N_BRANCH=N2_BRANCH,
                   element_tag=FLUID_TAG)
    ang, axis = ROT_T2, AXIS_T2
    core, dm, dp, dbr = (hexmesh.rotate(tj.core, ang, axis=axis), quadmesh.rotate(tj.disc_minus, ang, axis=axis),
                        quadmesh.rotate(tj.disc_plus, ang, axis=axis), quadmesh.rotate(tj.disc_branch, ang, axis=axis))
    if mirror:
        # T2's branch always comes out world +x (unaffected by ROT_T2, which
        # only turns the main axis z -> y) -- so without this, T2_out's
        # branch points the *same* absolute direction as T2_in's rather than
        # back towards it, and forcing the downstream bend the other way
        # folds the pipe back into itself instead of turning it. An extra
        # 180 about the main axis (world y) flips branch x -> -x while
        # leaving the y-axis main legs on the rotation axis, invariant.
        yax = (0.0, 1.0, 0.0)
        core, dm, dp, dbr = (hexmesh.rotate(core, np.pi, axis=yax), quadmesh.rotate(dm, np.pi, axis=yax),
                            quadmesh.rotate(dp, np.pi, axis=yax), quadmesh.rotate(dbr, np.pi, axis=yax))
    return tj._replace(core=core, disc_minus=dm, disc_plus=dp, disc_branch=dbr)


# base margin below the lower of the two T1 branches' own y. T1's own y is
# pinned exactly to its chimera target's y (build_bend_mesh's connector lives
# in a single fixed-y plane), so T2_SHARED_Y = min(chi target y) - H1 -
# RUN_T1_T2 -- meaning chimera sits exactly (H1 + RUN_T1_T2) above serp's own
# y, for *any* absolute chimera y. Fixed at 10.0 - H1 so that gap is exactly
# the user's requested 10, without having to move chimera itself.
RUN_T1_T2 = 10.0 - H1


def place_t2(source_disc, t2_y, mirror=False):
    """One T2 at ``t2_y``, its ``+y`` leg loft_between-ed back to ``source_disc`` --
    T1's own branch for the first of a chain, the previous T2's ``-y`` leg for
    every one after that.  Returns that T2's own ``-y`` leg rather than capping
    it, so the caller decides whether it carries on to another T2 or dead-ends.

    The offset is pure ``-y`` and the run length is solved *per side*, so both
    sides' T2s land on the same ``t2_y`` even though T1_in and T1_out sit at
    different y (they follow chimera's own inlet/outlet).  A fixed run length
    instead would leave the two sides at different y -- which the coil build
    downstream cannot express, since it spans one shared x-z plane.  Pure -y
    also keeps the two discs facing each other squarely, avoiding the large
    in-plane rotation a diagonal offset would otherwise blend across."""
    t2 = build_t2(mirror=mirror)
    br_pos = source_disc.points.mean(axis=0)
    run = br_pos[1] - t2_y - L2
    assert run > 0.1, "t2_y too close to source disc y=%.2f (side)" % br_pos[1]
    t2_center = np.array([br_pos[0], t2_y, br_pos[2]])
    core = hexmesh.translate(t2.core, t2_center)
    da = quadmesh.translate(t2.disc_minus, t2_center)   # -y, on to the next T2 (or capped)
    db = quadmesh.translate(t2.disc_plus, t2_center)    # +y, faces back upstream
    dbr = quadmesh.translate(t2.disc_branch, t2_center)  # +/-x (mirror), out to a serpentine
    # source_disc is T1's own branch (eqtee's spined_ogrid) for the first T2 in a chain,
    # or the previous T2's own disc_minus (the cob's ogrid section, same recipe as db)
    # for every one after -- both share db's quad/orient/line connectivity either way, so
    # loft_between's blend applies uniformly.  Neither shares db's *numbering*, which is
    # what its own _align pass settles before the twist.
    conn = loft_between(source_disc, db, 6, element_tags=FLUID_TAG)
    # conn's db-facing end is db verbatim (loft_between's t=1), so this weld is exact --
    # but it is still worth stating locally rather than leaving to the assembly's own
    # 0.005: 0.04 is the window between the residual any of these seams has to bridge and
    # the nearest *real* feature separation here, which is 0.05 (measured
    # 0.04999999999999716) and must not be collapsed.
    core = merge_at([core, conn], 0.04)
    return core, da, dbr, t2_center


#: T2 junctions in series on each side, each hanging off the previous one's own
#: -y leg, and each feeding its own copy of the serpentine coil.
N_T2 = 4
T2_SPACING = 6.0        # centre-to-centre step along -y between successive T2s

# T2_in's branch faces world +x, T2_out's -x (mirrored) -- so as long as
# T2_out sits to the +x side of T2_in, the two branches point at each other
# by construction, not by a post-hoc sign guess on the downstream bend.
T2_SHARED_Y = min(br_in.points.mean(axis=0)[1], br_out.points.mean(axis=0)[1]) - RUN_T1_T2


def t2_chain(source_disc, mirror):
    """``N_T2`` T2s in series along -y off one T1 branch.  Only the last one's
    -y leg is capped; every other one carries the chain to its successor, so
    the whole run stays a single open flow path off that branch."""
    levels, src = [], source_disc
    for k in range(N_T2):
        core, da, dbr, ctr = place_t2(src, T2_SHARED_Y - k * T2_SPACING,
                                      mirror=mirror)
        levels.append({"core": core, "da": da, "dbr": dbr, "ctr": ctr})
        src = da
    levels[-1]["dead"] = hexmesh.extrude(levels[-1]["da"], 1.5 * R_BR, 2,
                                         axis=(0.0, -1.0, 0.0), last_tag="wall",
                                         element_tags=FLUID_TAG)
    return levels


chain_in = t2_chain(br_in, mirror=False)
chain_out = t2_chain(br_out, mirror=True)

pieces2 = pieces + [p for lv in (*chain_in, *chain_out)
                    for p in (lv["core"], *([lv["dead"]] if "dead" in lv else []))]
mesh2 = merge_at(pieces2, 0.005)
_rep2 = hexmesh.topology_report(mesh2)
print("stage2:", mesh2.n_hexes, "hexes,", 2 * N_T2, "T2 junctions, watertight",
      _rep2.watertight and _rep2.n_components == 1, "conforming", _rep2.conformal,
      "min sj", hexmesh.scaled_jacobian(mesh2).min())

# -----------------------------------------------------------------------------
# T2 branch -> serpentine.  The coil's own shape (COIL_MOVES below) is FIXED --
# traced directly from the reference photo and not to be reshaped or rescaled,
# only placed in space -- so the two T2 branches are brought to it instead of
# the other way around: each bridges INWARD by the same run length (so they
# meet the coil's own fixed cap-to-cap span symmetrically), then bends 90
# degrees to heading +z, exactly like elbow_backward/build_bend_mesh's other
# connectors elsewhere in this file.
#
# Why a bend is unavoidable: dbr_t2i/dbr_t2o face +x/-x (T2_out mirrored, see
# build_t2) while the coil's own passes run in z ("up is z-" -- the long
# PASS_LEN runs are the vertical passes, the short u_r/u_r_mid arcs are the
# horizontal pass-to-pass stacking). A first attempt tried to keep the coil
# in the (x,z) plane matching the branches' own x-heading and land the far
# end exactly on dbr_t2o's stored point pattern by construction -- but
# dbr_t2o's pattern is rotated 180 degrees (build_t2's own mirror) relative
# to dbr_t2i's, so an *exact* landing forces the coil to arrive from the same
# side as T2_out's own core, piercing straight through it. Bridging inward
# and bending to z+ instead approaches each T2 from its own free side, and
# the small pattern mismatch where the coil's own (continued) end meets
# T2_out's bridge is absorbed by hexmesh.bridge (same tool already used for the
# T1-to-T2 joins above), not by forcing an exact but colliding registration.
# -----------------------------------------------------------------------------

# -- the coil's own fixed shape --------------------------------------------
# Traced from a reference photo; do not reshape or rescale.  It lives in
# serpentine_pipe.py, which builds the same physical part standing alone.
COIL_MOVES = COIL_MOVES_LIB

_coil_local = xz_path(COIL_MOVES, (0.0, 0.0), 0.0, 0.0)
_coil_end = _coil_local.centerline(np.array([1.0]))[0]
COIL_DV = _coil_end[2]   # in-plane end offset is (0, COIL_DV) exactly

BEND_R_CONN = 3.0
VERTICAL_RUN = 16.0
#: How much of the outbound leg is left for hexmesh.bridge to span.  It has to be
#: generous: bridge spends a rigid stub (up to stub_max) at each end and
#: fits n_blend layers into whatever remains, so a token gap makes those layers
#: absurdly thin -- at GAP_Z = 1.0 the two stubs ate 0.6 of it and the six
#: blend layers were 0.067 each, about a seventh of the tube radius.  Must stay
#: below VERTICAL_RUN, which is the leg it is carved out of.
GAP_Z = 10.0
assert 0.0 < GAP_Z < VERTICAL_RUN, (
    "GAP_Z is carved out of VERTICAL_RUN -- at or above it the outbound leg's "
    "own straight run vanishes (or reverses) instead of just getting shorter")


def _dx_of(line_len, turn_sign, heading0):
    """Net x-displacement of [line(line_len), arc(BEND_R_CONN,turn_sign)]
    alone (the vertical run after it is pure +/-z and contributes zero) --
    affine in line_len, so two probes pin down the line length that lands
    exactly on a given total dx without re-deriving the arc's own x-throw by
    hand for each heading/turn-sign combination.  Both probes are positive:
    every move of a walk advances it, so there is no zero-length line to
    probe the intercept with directly."""
    mv = [paths.line(line_len), paths.arc(BEND_R_CONN, turn_sign)]
    return xz_path(mv, (0.0, 0.0), heading0, 0.0).centerline(np.array([1.0]))[0][0]


def _solve_line_len(turn_sign, heading0, target_dx):
    dx1, dx2 = _dx_of(1.0, turn_sign, heading0), _dx_of(2.0, turn_sign, heading0)
    return 1.0 + (target_dx - dx1) / (dx2 - dx1)


def _end_section(section, moves, heading, y_fixed):
    """The swept tube's own exact terminal cross-section, via the same
    frames.sweep_placements machinery sweep() uses internally (not
    re-derived) -- so a piece built to continue from it lands seamlessly."""
    c = section.points.mean(axis=0)
    path = xz_path(moves, (c[0], c[2]), heading, y_fixed)
    end = quadmesh.place_on_path(section, path, [0.0, 1.0], origin=c)[-1]
    # the tube's own downstream tangent is the port's outward direction; port() takes
    # it only as a sign hint and reads the precision off the section's fitted plane
    return quadmesh.port(end, outward=path.tangent(np.array([1.0]))[0])


TOTAL_COIL = _coil_local.total_length
# The sweep target is the coil's own (see serpentine_pipe.py): it must subdivide
# even the tightest U_R 180-degree turn into several stations.
COIL_TARGET_LEN = COIL_TARGET_LEN_LIB


def build_coil(dbr_i, dbr_o):
    """One serpentine coil spanning a facing pair of T2 branches, returned as
    ``[conn_in, coil, conn_out, bridge]``.

    Called once per level of the T2 chains: every level's pair sits at the
    same x and z as level 0's and differs only in y, so each level gets the
    same coil in its own plane.  Everything is derived from the two discs
    handed in -- nothing reaches back to a particular level's globals."""
    ci = dbr_i.points.mean(axis=0)
    co = dbr_o.points.mean(axis=0)
    gap = co[0] - ci[0]
    assert gap > 1.0, (
        "T2_out must sit meaningfully to the +x side of T2_in for the two "
        "(fixed-direction) branches to actually converge -- got x_dbr_i=%.2f, "
        "x_dbr_o=%.2f" % (ci[0], co[0]))
    # Both bridge inward by the SAME run, so they land symmetrically |COIL_DV|
    # apart -- exactly the coil's own fixed cap-to-cap span -- with no leftover
    # horizontal jog needed once they turn to +z.
    total_dx = (gap - abs(COIL_DV)) / 2.0
    assert total_dx > 0.5, "T2 branches too close together for this coil's own span"

    moves_in = [paths.line(_solve_line_len(90.0, 0.0, total_dx)),
                paths.arc(BEND_R_CONN, 90.0), paths.line(VERTICAL_RUN)]
    moves_out = [paths.line(_solve_line_len(-90.0, np.pi, -total_dx)),
                 paths.arc(BEND_R_CONN, -90.0), paths.line(VERTICAL_RUN - GAP_Z)]

    # The inbound connector and the coil are the SAME planar (x, z) turtle walk
    # -- the connector ends heading +z and the coil's own local +u is +z, with
    # the same turn handedness -- so their move tables simply concatenate and
    # the two sweep as ONE piece.  That is not just tidier, it removes a real
    # failure: built separately, the coil's start section is recomputed from
    # the path rather than taken from the connector's own last layer, and the
    # two agree only to ~1 ULP.  That was once fatal: merge welded by rounding to a
    # tol-sized bucket, and this geometry sits on a decimal lattice commensurate with
    # tol, so a coordinate could land exactly on a bucket edge (measured:
    # y = -15.7375 = -3147.5 * 0.005 at T2 level 1) and a 1.78e-15 disagreement then
    # failed to weld, pinching the surface into one open edge.  Coincidence is a plain
    # radius now, which has no edges to land on -- but one sweep has no such seam at
    # all, and that is still the better answer.
    inflow_moves = moves_in + COIL_MOVES
    path = xz_path(inflow_moves, (ci[0], ci[2]), 0.0, ci[1])
    inflow = hexmesh.sweep_path(dbr_i, path, target_length=COIL_TARGET_LEN,
                                origin=ci, element_tags=FLUID_TAG)
    assert hexmesh.is_conforming(inflow), "coil sweep produced a non-conforming mesh"
    conn_o = build_bend_mesh(dbr_o, co, moves_out, np.pi, co[1], n_slices)

    # conn_o's own end (dbr_o's pattern, through its own bend) and the coil's
    # own end (dbr_i's pattern, carried the whole way) are different patterns
    # landing GAP_Z apart by construction -- hexmesh.bridge (same tool as the
    # T1-to-T2 joins above) closes that last short gap.
    joint = hexmesh.bridge(_end_section(dbr_o, moves_out, np.pi, co[1]),
                           _end_section(dbr_i, inflow_moves, 0.0, ci[1]), layers=3,
                           element_tags=FLUID_TAG)
    return [inflow, conn_o, joint]


coils = [p for lv_i, lv_o in zip(chain_in, chain_out)
         for p in build_coil(lv_i["dbr"], lv_o["dbr"])]

pieces3 = pieces2 + coils
mesh3 = merge_at(pieces3, 0.005)
_rep3 = hexmesh.topology_report(mesh3)
print("stage3:", mesh3.n_hexes, "watertight",
      _rep3.watertight and _rep3.n_components == 1,
      "conforming", _rep3.conformal, "min sj", hexmesh.scaled_jacobian(mesh3).min())
# -- the boundary layer, over the whole manifold at once ----------------------
# Every no-slip face has to be named before this runs, and a few are not: the blend
# blocks (``loft_between``, ``bridge``, ``adapter``) build their intermediate sections
# through ``QuadMesh.from_corners``, which carries no edge tags, so their lateral faces
# come out unnamed.  Naming whatever free face is still unnamed closes that gap without
# having to teach each blend about tags -- and it is safe precisely because everything
# that is *not* wall here is already named: the two riser caps, and the two chimera
# ports named in ``place_t1``.
_named = mesh3.face_tags.dense(mesh3.quad_mesh.n_quads)
_free = np.flatnonzero(hexmesh.boundary_face_ids(mesh3))
mesh3 = hexmesh.tag_faces(mesh3, _free[_named[_free] == ""], "wall")

# Skinned as one surface, not block by block: the layer has to cross every seam this
# file just welded -- T1 into its connectors, T1 into T2, T2 into the coils -- and an
# offset computed per block would meet itself at a different angle on each side of one.
manifold = skin_wall(mesh3, BL, element_tag=FLUID_TAG)
print("skinned manifold:", manifold.n_hexes, "hexes")


def join_chimera(port_tag, chi_disc, name):
    """The last ``ADAPT`` into one chimera opening.

    Both sides are now the *same* pattern -- 80 quads over 89 points, chimera's port and
    this one built by the same ``skin_wall`` over the same cross-section -- but neither
    their numbering nor their phase agrees: each was numbered by its own
    ``boundary_mesh`` extraction, and the manifold's arrives at whatever roll the sweep's
    own frame carried it to.

    ``template=`` settles the numbering: the extracted port is handed
    :data:`_CHI_PORT`'s own B-rep and fills it with the manifold's real coordinates, so
    point ``i`` here and point ``i`` of ``chi_disc`` are the same node of one pattern.
    :func:`loft_between` then settles the phase, carrying each point round the connector
    axis on the *short* way rather than straight across the disc -- a chord between two
    out-of-phase discs passes near the axis and waists the block."""
    port = hexmesh.boundary_mesh(manifold, port_tag,
                                 template=fake_chi_disc(chi_disc.points.mean(axis=0)
                                                        - np.array([0.0, 0.0, ADAPT])))
    # in-plane only: the two are ADAPT apart along z by construction, and what is worth
    # reporting is how far the *patterns* sit from each other once that is removed
    a = port.points - port.points.mean(axis=0)
    b = chi_disc.points - chi_disc.points.mean(axis=0)
    print("chimera %s: port %d quads, index-paired residual %.3e (phase)"
          % (name, port.n_quads, np.linalg.norm(a[:, :2] - b[:, :2], axis=1).max()))
    block = loft_between(port, chi_disc, 6, element_tags=FLUID_TAG)
    print("chimera %s: morph min sj %.4e" % (name, hexmesh.scaled_jacobian(block).min()))
    return block


# conn_chi_in targets CHI_OUT and conn_chi_out targets CHI_IN (the swap at placement),
# so the discs paired here follow the same swap, not the "in"/"out" name.
morphs = [join_chimera(PORT_TAG_IN, chi_out_disc, "in"),
          join_chimera(PORT_TAG_OUT, chi_in_disc, "out")]

if chi_mesh is not None:
    # chimera's own inlet/outlet faces are welded away into interior planes here,
    # so their names must go (the combined mesh's inlet/outlet are the riser tops);
    # a stale tagged interior face would export as a bogus BC.  Retiring the two by
    # name leaves every other row alone -- including the jacket's own "insulated"
    # exterior, which nothing here touches and which a keep-only-"wall" filter would
    # have stripped, dropping 39k boundary faces to no BC at all.
    chi_mesh = hexmesh.retag_face(chi_mesh, {"inlet": "", "outlet": ""})
    mesh_out = merge_at([manifold, *morphs, chi_mesh], 0.005)
else:
    caps = [hexmesh.extrude(d, 0.3, 1, axis=(0, 0, 1), last_tag="wall",
                            element_tags=FLUID_TAG)
            for d in (chi_in_disc, chi_out_disc)]
    mesh_out = merge_at([manifold, *morphs, *caps], 0.005)

# the morph blocks' own lateral faces are unnamed for the same reason the blends' were
# -- and the two manifold-side port names have just been welded shut, so they go too
mesh_out = hexmesh.retag_face(mesh_out, {PORT_TAG_IN: "", PORT_TAG_OUT: ""})
_named = mesh_out.face_tags.dense(mesh_out.quad_mesh.n_quads)
_free = np.flatnonzero(hexmesh.boundary_face_ids(mesh_out))
mesh_out = hexmesh.tag_faces(mesh_out, _free[_named[_free] == ""], "wall")

_rep_out = hexmesh.topology_report(mesh_out)
print("mesh_out:", mesh_out.n_hexes, "hexes, watertight",
      _rep_out.watertight and _rep_out.n_components == 1,
      "conforming", _rep_out.conformal)
print(_rep_out)

mesh = mesh_out
OUT_NAME = "chimera_full"
GROUPS = {"wall": "W  ", "inlet": "v  ", "outlet": "O  ", SOLID_FACE_TAG: "I  ",
          INTERFACE_TAG: {"fluid": "W  ", "solid": None}}
writer.to_re2(mesh, OUT_NAME + ".re2", groups=GROUPS)
writer.to_vtu(mesh, OUT_NAME + ".vtu", groups=GROUPS)
writer.to_fld(mesh, OUT_NAME + ".f00000")
print("groups:", ", ".join(mesh.face_group_tags))

stats = hexmesh.quality_summary(mesh)
assert stats.min > 0.0, "inverted element: min scaled Jacobian %g" % stats.min
print("%d hex elements, %d points, order %d"
      % (mesh.n_hexes, mesh.n_points, mesh.order))
print("scaled Jacobian: min=%.4f mean=%.4f" % (stats.min, stats.mean))
