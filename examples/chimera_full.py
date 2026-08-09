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
placed, never rescaled (see ``coil_lib``, whose move table ``serpentine_pipe.py``
builds standing alone).  The chimera ports sit exactly ``H1 + RUN_T1_T2`` = 10
above the first coil in y.

    PYTHONPATH=. python examples/chimera_full.py

Produces ``chimera_full.re2`` and ``chimera_full.vtu``.

``FAST`` (default ``True``) swaps the real ``chimera.py`` build -- 350k
elements, a couple of minutes -- for two short capped stubs carrying its exact
inlet/outlet disc pattern, so the manifold's own geometry can be checked in
seconds.  Set it ``False`` for the whole thing.

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
* ``hexmesh.bridge`` -- span a *large* one (the ~0.94 median between T1's
  arc-length-stationed branch disc and T2's uniform-angle main leg) as a
  single ``HexMesh.loft``: rigid stubs off each side, a blend across the gap
  between them, no internal merge to fail.

All three were written here first and now live in the toolkit; this file keeps
only the choice of which to use at each seam, which is the part that is about
this manifold rather than about meshes in general.

And where a seam can be removed rather than made exact, it is: the inbound
connector and the coil sweep as ONE turtle walk (see ``build_coil``).  They
used to be two pieces welded together, which broke as soon as the coils were
stacked -- ``_weld`` fuses by rounding to a ``tol``-sized *bucket*, so the
~1 ULP disagreement between the connector's last layer and the coil's
recomputed first one fails to weld wherever a coordinate happens to land on a
bucket edge, pinching the surface open.
"""

import os
import sys

import numpy as np
from scipy.spatial import cKDTree

from nekmeshpy import export, hexmesh, quadmesh
from nekmeshpy.core import paths
from nekmeshpy.core.paths import turtle_path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from coil_lib import MOVES as COIL_MOVES_LIB  # noqa: E402  (needs the path above)
from coil_lib import TARGET_LEN as COIL_TARGET_LEN_LIB  # noqa: E402
from tjunction_lib import build_tjunction  # noqa: E402

FAST = False
ORDER = 2
N_HALF = 8
RADIAL = np.array([0.0, 0.4, 0.8, 1.0])
CENTER_SCALE = 0.5
n_slices = 3

R_MAIN = 1.2     # == chimera's R_MAIN
R_BR = 0.5       # == chimera's R_BRANCH

L1 = 2.5 * R_MAIN     # T1 main half-length
H1 = 2.5 * R_MAIN     # T1 branch length
L2 = 1.2               # T2's main-leg offset from its own centre (build_tjunction's Z_NEAR default)

kw_t1 = dict(n_half=N_HALF, order=ORDER, radial=RADIAL, center_scale=CENTER_SCALE,
             n_slices_a=n_slices, n_slices_b=n_slices, n_slices_branch=n_slices)
kw_t2 = kw_t1


# chimera's own port cross-section: the same quadrant_disc recipe its
# junctions are built from, so it reproduces a port's pattern exactly.
_chi_kw = dict(order=ORDER, N_QUAD=2, RADIAL=np.array([0.0, 0.6, 1.0]),
               CENTER_SCALE=0.7, N_TRANS=2, N_BRANCH=2)


def fake_chi_disc(center):
    tj = build_tjunction(1.2, 0.5, 3.0, **_chi_kw)
    d = tj.disc_plus  # normal +z, matches chimera's actual inlet/outlet convention
    return quadmesh.translate(d, np.asarray(center) - np.array([0.0, 0.0, 1.2]))


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
    # not pair index-for-index with T1's own end section, which hexmesh.adapter needs.
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
# T1: same quadrant pattern family as T2 (not eqtee) so the T1-branch <-> T2-
# main weld is a same-family tolerance weld, not a cross-family one. Equal
# main/branch radius is not directly supported by the quadrant construction
# (the footprint curve degenerates as R_BRANCH -> R_MAIN), so R_BRANCH is
# 99.9% of R_MAIN -- visually and functionally equal. PHI_W / CAP_TIP_BIAS /
# ORIGIN are left to tjunction_lib.auto_params, which picks them from the
# radius ratio; at this ratio a hand-tuned PHI_W=165 with the old fixed
# bias and hub gave 0.105 min scaled Jacobian and the automatic choice gives
# 0.253. CENTER_SCALE has no effect on this at all (checked).
# main axis -> x, branch -> -y.
# -----------------------------------------------------------------------------
T1_RATIO = 0.999
ROT_T1 = -np.deg2rad(120.0)
AXIS_T1 = (1.0, -1.0, 1.0)


def build_t1(mirror=False):
    tj = build_tjunction(R_MAIN, R_MAIN * T1_RATIO, H1, order=ORDER, N_QUAD=2,
                         RADIAL=np.array([0.0, 0.6, 1.0]), CENTER_SCALE=0.7,
                         N_TRANS=n_slices, N_BRANCH=n_slices, Z_NEAR=L1)
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
                    plane_v_index=2, vertical_run=0.0):
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
    back = turtle_path([("line", vertical_run, 0.0), ("arc", bend_r, -turn),
                        ("line", run_len, 0.0)],
                       start=(target_pt[0], target_pt[plane_v_index]), heading=heading0)
    x0, v0 = back.centerline(np.array([1.0]))[0]
    fwd_heading = 0.0 if start_heading_sign > 0 else np.pi
    return ((x0, v0), [("line", run_len, 0.0), ("arc", bend_r, turn),
                       ("line", vertical_run, 0.0)], fwd_heading)


def xz_path(moves, start_xz, heading, y_fixed):
    """A turtle walk lifted into the world ``(x, z)`` plane at fixed ``y`` -- every
    connector in this file lives in such a plane, so the walk's own ``+u`` is world
    ``+x`` and its ``+v`` world ``+z``.  ``paths.embed`` is what keeps ``y_fixed`` out
    of the tangent (a direction, not a point)."""
    return paths.embed(turtle_path(moves, start=start_xz, heading=heading),
                       u=(1.0, 0.0, 0.0), v=(0.0, 0.0, 1.0),
                       origin=(0.0, y_fixed, 0.0))


def build_bend_mesh(section, start_pt3, moves, heading2d, y_fixed, n_layers, last_tag=""):
    path = xz_path(moves, (start_pt3[0], start_pt3[2]), heading2d, y_fixed)
    return hexmesh.sweep_path(section, path, layers=n_layers, orientation="fixed",
                              up=(0.0, 1.0, 0.0), origin=start_pt3, last_tag=last_tag)


BEND_R1 = 2.0 * R_MAIN
VERTICAL_DROP = 14.0   # matches VERTICAL_RISE -- the two legs read as comparable
ADAPT = 1.0            # length of each pattern-adapter layer (see hexmesh.adapter)
RUN_TO_RISER = 8.0
VERTICAL_RISE = 14.0

#: Centre-to-centre distance between the two y-pipes (each T1's branch running
#: down into its T2's main) -- the T1 centres are *placed* at this separation,
#: symmetric about the chimera targets' x midpoint, and each side's horizontal
#: chimera-run length is solved from that, instead of the other way round.
D_YPIPES = 52.0

pieces = []


def place_t1(side_center, chi_target, chi_disc, tag, t1_x, mirror=False):
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
    # the swept tube stops ADAPT short of the chimera plane; the last ADAPT is
    # a hexmesh.adapter morphing T1's own disc pattern into chimera's own
    # (they differ ~2% in one quadrant's wall spacing -- different
    # branch-radius params -- so no plain weld across that seam can be exact).
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
        end = quadmesh.place_on_path(disc, path, [0.0, 1.0], orientation="fixed",
                                     up=(0.0, 1.0, 0.0), origin=db_c)[-1]
        return quadmesh.port(end, outward=path.tangent(np.array([1.0]))[0])

    conn_chi = build_bend_mesh(db, db_c, moves, heading, db_c[1], n_slices*8)
    end_port = _end_section(db)
    # chimera's own openings face -z and this connector rises +z into them, so the two
    # face each other; adapter takes its roll axis from end_port's own normal, which is
    # what the explicit axis=(0,0,1) always was.
    chi_port = quadmesh.port(chi_disc, outward=(0.0, 0.0, -1.0))
    print("  [%s] end %s" % (tag, end_port))
    print("  [%s] tgt %s" % (tag, chi_port))
    adapter = hexmesh.adapter(end_port, chi_port, layers=2)
    # disc_a -> bends the opposite way, then climbs straight to the riser.
    # The inlet/outlet risers themselves bend to z- (away from chimera, which
    # conn_chi above reaches via +z) -- opposite sign from a naive a_sign-only
    # turn, so both flip regardless of which side is mirrored.
    a_sign = -b_sign
    moves_r = [("line", RUN_TO_RISER, 0.0), ("arc", BEND_R1, -90.0 if a_sign > 0 else 90.0),
              ("line", VERTICAL_RISE, 0.0)]
    da_c = da.points.mean(axis=0)
    # da_c[1], not the nominal t1_center[1] -- same reasoning as _end_section
    # above (da is no more exactly centred on t1_center than db is).
    riser = build_bend_mesh(da, da_c, moves_r, 0.0 if a_sign > 0 else np.pi,
                            da_c[1], n_slices*8, last_tag=tag)
    # conn_chi's own end and the adapter's own start are the *same* physical
    # points (both derived from db via the same sweep_placements machinery),
    # but the adapter's internal 90-degree roll search (see hexmesh.adapter)
    # rotates them, and T1's own disc is only *near*-exactly 4-fold symmetric
    # (a ~0.03-unit residual, well under merge()'s global tol=0.005 default)
    # -- so weld *this one seam* locally, at a tolerance sized to that
    # specific residual, rather than loosening the tolerance for the whole
    # assembly (which welded an unrelated, closer-together pair by mistake
    # the one time this was tried globally).
    conn_chi = hexmesh.merge([conn_chi, adapter], tol=0.05)
    return core, [conn_chi], riser, dbr, t1_center


# T1 on the negative-x side (the assembly's own "inlet" riser) connects to
# chimera's OUTLET; T1 on the positive-x side ("outlet" riser) to its
# INLET.  Which physical port a side reaches is carried entirely by the
# (chi_target, chi_disc) pair -- place_t1 itself is agnostic.
X_MID = 0.5 * (CHI_IN[0] + CHI_OUT[0])
core_in, conn_chi_in, riser_in, br_in, t1c_in = place_t1(
    (0, CHI_OUT[1], 0), CHI_OUT, chi_out_disc, "inlet",
    t1_x=X_MID - D_YPIPES / 2.0)
print("t1c_in", t1c_in)
core_out, conn_chi_out, riser_out, br_out, t1c_out = place_t1(
    (0, CHI_IN[1], 0), CHI_IN, chi_in_disc, "outlet",
    t1_x=X_MID + D_YPIPES / 2.0, mirror=True)
print("t1c_out", t1c_out, "| y-pipe separation:", t1c_out[0] - t1c_in[0])

for _nm, _m in [("core_in", core_in), ("conn_chi_in", conn_chi_in[0]), ("riser_in", riser_in),
               ("core_out", core_out), ("conn_chi_out", conn_chi_out[0]), ("riser_out", riser_out)]:
    print("  %s min_sj=%.4e" % (_nm, hexmesh.scaled_jacobian(_m).min()))

pieces += [core_in, *conn_chi_in, riser_in, core_out, *conn_chi_out, riser_out]

mesh1 = hexmesh.merge(pieces, tol=0.005)
# one report, read twice: is_watertight and is_conforming each recompute the
# whole thing, which is seconds apiece at this size.
_rep1 = hexmesh.topology_report(mesh1)
print("stage1:", mesh1.n_hexes, "hexes, watertight",
      _rep1.watertight and _rep1.n_components == 1,
      "conforming", _rep1.conformal, "min sj", hexmesh.scaled_jacobian(mesh1).min())

# -----------------------------------------------------------------------------
# T2: branches off T1's -y branch.  Unequal radius (main = R_MAIN, matching
# T1's branch; branch = R_BR, matching the serpentine) -- the quadrant
# construction, not the equal-radius eqtee. Local main axis z, branch axis x
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
#: Lengthening costs no shape fidelity: build_tjunction's branch() blends the
#: footprint (the saddle where branch meets main) into the round opening, and
#: those two differ *only* in the axial coordinate -- both carry the identical
#: R_BRANCH*(sin, cos) cross-section -- so the blend straightens the section's
#: axial position and never touches its radius.  Measured over the whole run:
#: wall radius = R_BR to 1.7e-15.  A long branch is a straight round pipe whose
#: end plane merely relaxes from saddle-shaped to flat more gradually.
H2_BRANCH = 15.0
N2_BRANCH = max(2, int(round(H2_BRANCH / 2.0)))   # ~2.0-long layers, as the coil uses


def build_t2(mirror=False):
    tj = build_tjunction(R_MAIN, R_BR, H2_BRANCH, order=ORDER, N_QUAD=2,
                         RADIAL=np.array([0.0, 0.6, 1.0]), CENTER_SCALE=0.7,
                         N_TRANS=n_slices, N_BRANCH=N2_BRANCH)
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
    """One T2 at ``t2_y``, its ``+y`` leg bridged back to ``source_disc`` --
    T1's own branch for the first of a chain, the previous T2's ``-y`` leg for
    every one after that.  Returns that T2's own ``-y`` leg rather than capping
    it, so the caller decides whether it carries on to another T2 or dead-ends.

    The offset is pure ``-y`` and the run length is solved *per side*, so both
    sides' T2s land on the same ``t2_y`` even though T1_in and T1_out sit at
    different y (they follow chimera's own inlet/outlet).  A fixed run length
    instead would leave the two sides at different y -- which the coil build
    downstream cannot express, since it spans one shared x-z plane.  Pure -y
    also keeps the hexmesh.bridge stubs aligned with both discs' own normal,
    avoiding the large-rotation mismatch a diagonal offset causes."""
    t2 = build_t2(mirror=mirror)
    br_pos = source_disc.points.mean(axis=0)
    run = br_pos[1] - t2_y - L2
    assert run > 0.1, "t2_y too close to source disc y=%.2f (side)" % br_pos[1]
    t2_center = np.array([br_pos[0], t2_y, br_pos[2]])
    core = hexmesh.translate(t2.core, t2_center)
    da = quadmesh.translate(t2.disc_minus, t2_center)   # -y, on to the next T2 (or capped)
    db = quadmesh.translate(t2.disc_plus, t2_center)    # +y, faces back upstream
    dbr = quadmesh.translate(t2.disc_branch, t2_center)  # +/-x (mirror), out to a serpentine
    # both legs run along y: the source faces -y (on down the chain) and this T2's
    # +y leg faces back upstream at it.  Stating that rather than letting bridge infer
    # it from the centroid line is what lets it check the two really do face each
    # other, and that they are the same size.
    conn = hexmesh.bridge(quadmesh.port(source_disc, outward=(0.0, -1.0, 0.0)),
                          quadmesh.port(db, outward=(0.0, 1.0, 0.0)))
    return core, conn, da, dbr, t2_center


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
        core, conn, da, dbr, ctr = place_t2(src, T2_SHARED_Y - k * T2_SPACING,
                                            mirror=mirror)
        levels.append({"core": core, "conn": conn, "da": da, "dbr": dbr, "ctr": ctr})
        src = da
    levels[-1]["dead"] = hexmesh.extrude(levels[-1]["da"], 1.5 * R_BR, 2,
                                         axis=(0.0, -1.0, 0.0), last_tag="wall")
    return levels


chain_in = t2_chain(br_in, mirror=False)
chain_out = t2_chain(br_out, mirror=True)

pieces2 = pieces + [p for lv in (*chain_in, *chain_out)
                    for p in (lv["core"], lv["conn"], *( [lv["dead"]]
                                                         if "dead" in lv else []))]
mesh2 = hexmesh.merge(pieces2, tol=0.005)
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
# Traced from a reference photo; do not reshape or rescale.  It lives in coil_lib
# because serpentine_pipe.py builds the same physical part standing alone.
COIL_MOVES = COIL_MOVES_LIB

_coil_local = turtle_path(COIL_MOVES, start=(0.0, 0.0), heading=0.0)
_coil_end_uv = _coil_local.centerline(np.array([1.0]))[0]
COIL_DV = _coil_end_uv[1]   # local (u, v) end offset is (0, COIL_DV) exactly

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
    hand for each heading/turn-sign combination."""
    mv = [("line", line_len, 0.0), ("arc", BEND_R_CONN, turn_sign)]
    return turtle_path(mv, start=(0.0, 0.0), heading=heading0).centerline(
        np.array([1.0]))[0][0]


def _solve_line_len(turn_sign, heading0, target_dx):
    dx0, dx1 = _dx_of(0.0, turn_sign, heading0), _dx_of(1.0, turn_sign, heading0)
    return (target_dx - dx0) / (dx1 - dx0)


def _end_section(section, moves, heading, y_fixed):
    """The swept tube's own exact terminal cross-section, via the same
    frames.sweep_placements machinery sweep() uses internally (not
    re-derived) -- so a piece built to continue from it lands seamlessly."""
    c = section.points.mean(axis=0)
    path = xz_path(moves, (c[0], c[2]), heading, y_fixed)
    end = quadmesh.place_on_path(section, path, [0.0, 1.0], orientation="fixed",
                                 up=(0.0, 1.0, 0.0), origin=c)[-1]
    # the tube's own downstream tangent is the port's outward direction; port() takes
    # it only as a sign hint and reads the precision off the section's fitted plane
    return quadmesh.port(end, outward=path.tangent(np.array([1.0]))[0])


TOTAL_COIL = _coil_local.total_length
# The sweep target is the coil's own (see coil_lib): it must subdivide even the
# tightest U_R 180-degree turn into several stations.
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

    moves_in = [("line", _solve_line_len(90.0, 0.0, total_dx), 0.0),
                ("arc", BEND_R_CONN, 90.0), ("line", VERTICAL_RUN, 0.0)]
    moves_out = [("line", _solve_line_len(-90.0, np.pi, -total_dx), 0.0),
                 ("arc", BEND_R_CONN, -90.0), ("line", VERTICAL_RUN - GAP_Z, 0.0)]

    # The inbound connector and the coil are the SAME planar (x, z) turtle walk
    # -- the connector ends heading +z and the coil's own local +u is +z, with
    # the same turn handedness -- so their move tables simply concatenate and
    # the two sweep as ONE piece.  That is not just tidier, it removes a real
    # failure: built separately, the coil's start section is recomputed from
    # the path rather than taken from the connector's own last layer, and the
    # two agree only to ~1 ULP.  HexMesh.merge welds by rounding to a
    # tol-sized *bucket* (see _weld -- "two points on opposite sides of a
    # bucket edge do not weld however close they are"), and this geometry sits
    # on a decimal lattice commensurate with tol, so a coordinate can land
    # exactly on a bucket edge (measured: y = -15.7375 = -3147.5 * 0.005 at
    # T2 level 1) and a 1.78e-15 disagreement then fails to weld, pinching the
    # surface into one open edge.  One sweep has no such seam at all.
    inflow_moves = moves_in + COIL_MOVES
    path = xz_path(inflow_moves, (ci[0], ci[2]), 0.0, ci[1])
    inflow = hexmesh.sweep_path(dbr_i, path, target_length=COIL_TARGET_LEN,
                                orientation="fixed", up=(0.0, 1.0, 0.0), origin=ci)
    assert hexmesh.is_conforming(inflow), "coil sweep produced a non-conforming mesh"
    conn_o = build_bend_mesh(dbr_o, co, moves_out, np.pi, co[1], n_slices)

    # conn_o's own end (dbr_o's pattern, through its own bend) and the coil's
    # own end (dbr_i's pattern, carried the whole way) are different patterns
    # landing GAP_Z apart by construction -- hexmesh.bridge (same tool as the
    # T1-to-T2 joins above) closes that last short gap.
    joint = hexmesh.bridge(_end_section(dbr_o, moves_out, np.pi, co[1]),
                           _end_section(dbr_i, inflow_moves, 0.0, ci[1]), layers=3)
    return [inflow, conn_o, joint]


coils = [p for lv_i, lv_o in zip(chain_in, chain_out)
         for p in build_coil(lv_i["dbr"], lv_o["dbr"])]

pieces3 = pieces2 + coils
mesh3 = hexmesh.merge(pieces3, tol=0.005)
_rep3 = hexmesh.topology_report(mesh3)
print("stage3:", mesh3.n_hexes, "watertight",
      _rep3.watertight and _rep3.n_components == 1,
      "conforming", _rep3.conformal, "min sj", hexmesh.scaled_jacobian(mesh3).min())
# -- registration check: does the connector's rising end actually land point-
# for-point on chimera's own disc pattern (mod the quadrant disc's 90-degree
# symmetry)?  The fake stand-in discs use the identical pattern/params as the
# real chimera, so this check is valid in FAST mode too. conn_chi_in
# targets CHI_OUT and conn_chi_out targets CHI_IN (the swap above), so the
# discs paired here follow the same swap, not the "in"/"out" name.
for nm, conn, disc in (("in", conn_chi_in[-1], chi_out_disc),
                       ("out", conn_chi_out[-1], chi_in_disc)):
    d, _ = cKDTree(conn.points).query(disc.points)
    print("chimera %s registration: max dist %.3e" % (nm, d.max()))
    print("chimera %s adapter min sj: %.4e" % (nm, hexmesh.scaled_jacobian(conn).min()))

manifold = pieces3

if chi_mesh is not None:
    # chimera's own inlet/outlet faces are welded away into interior planes
    # here, so their tags must go (the combined mesh's inlet/outlet are the
    # riser tops); a stale tagged interior face would export as a bogus BC.
    chi_mesh.face_tags = chi_mesh.face_tags.select(
        chi_mesh.face_tags.mask_for("wall"))
    mesh_out = hexmesh.merge([*manifold, chi_mesh], tol=0.005)
else:
    chi_in_cap = hexmesh.extrude(chi_in_disc, 0.3, 1, axis=(0, 0, 1), last_tag="wall")
    chi_out_cap = hexmesh.extrude(chi_out_disc, 0.3, 1, axis=(0, 0, 1), last_tag="wall")
    mesh_out = hexmesh.merge([*manifold, chi_in_cap, chi_out_cap], tol=0.005)

_rep_out = hexmesh.topology_report(mesh_out)
print("mesh_out:", mesh_out.n_hexes, "hexes, watertight",
      _rep_out.watertight and _rep_out.n_components == 1,
      "conforming", _rep_out.conformal)
print(_rep_out)

mesh = mesh_out
OUT_NAME = "chimera_full"
GROUPS = {"wall": "W  ", "inlet": "v  ", "outlet": "O  "}
export.to_re2(mesh, OUT_NAME + ".re2", groups=GROUPS)
export.to_vtu(mesh, OUT_NAME + ".vtu", groups=GROUPS)
export.to_fld(mesh, OUT_NAME + ".f00000")
print("groups:", ", ".join(mesh.face_group_tags))

stats = hexmesh.quality_summary(mesh)
assert stats.min > 0.0, "inverted element: min scaled Jacobian %g" % stats.min
print("%d hex elements, %d points, order %d"
      % (mesh.n_hexes, mesh.n_points, mesh.order))
print("scaled Jacobian: min=%.4f mean=%.4f" % (stats.min, stats.mean))
