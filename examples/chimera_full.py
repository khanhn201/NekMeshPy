r"""Full chimera manifold: two chimera_chain ports fed by a serpentine coil.

The whole assembly, inlet to outlet:

    riser -> T1 -> (main leg) elbow up into chimera_chain's own opening
                \-- (branch, -y) --> T2 --> bridge inward, bend to +z
                                              --> the serpentine coil

Two of those, mirrored about the coil's midspan: the T1 on the **negative-x**
side feeds chimera_chain's *outlet*, the one on the **positive-x** side its
*inlet*.

Each T1 branch does not carry one T2 but a **chain of ``N_T2``**, each hanging
off the previous one's own ``-y`` leg at ``T2_SPACING`` intervals; only the
last is capped.  The two chains are mirror images, so their ``k``-th junctions
face each other across their own copy of the serpentine coil -- ``N_T2``
parallel coils stacked down ``-y``, each planar in its own x-z plane.  The
coil's shape is traced from a reference photo and is **fixed**: it is only
placed, never rescaled (see ``COIL_MOVES``, shared verbatim with
``serpentine_pipe.py``).  The chimera ports sit exactly ``H1 + RUN_T1_T2`` = 10
above the first coil in y.

    PYTHONPATH=. python examples/chimera_full.py

Produces ``chimera_full.re2`` and ``chimera_full.vtu``.

``FAST`` (default ``True``) swaps the real ``chimera_chain.py`` build -- 350k
elements, a couple of minutes -- for two short capped stubs carrying its exact
inlet/outlet disc pattern, so the manifold's own geometry can be checked in
seconds.  Set it ``False`` for the whole thing.

Three seam techniques earn their keep here, all for the same reason -- pieces
built by *different* constructions have to meet exactly, and at ``order > 1``
``HexMesh.merge`` verifies shared high-order edge/face nodes to
``conform.entity_tol`` (~1e-9 x the model extent), far tighter than any
coordinate weld:

* ``_reindex_geometry`` -- re-express one section's geometry through another's
  index labels.  A pure permutation, so it is exact by construction where a
  coordinate rotation is only approximate.
* ``pattern_adapter`` -- blend across a *small* pattern difference (the ~0.03
  between T1's own leg and chimera's), first slice and last slice both exact.
* ``weld_bridge`` -- span a *large* one (the ~0.94 median between T1's
  arc-length-stationed branch disc and T2's uniform-angle main leg) as a
  single ``HexMesh.loft``: rigid stubs off each side, a blend across the gap
  between them, no internal merge to fail.

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

from nekmeshpy import HexMesh, LineMesh, QuadMesh, export
from nekmeshpy.model import frames
from nekmeshpy.model.paths import turtle_path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tjunction_lib import build_tjunction  # noqa: E402  (needs the path above)


def _find_roll(sec_a, sec_b, axis):
    """The 90-degree roll k minimizing index-wise deviation between sec_a and
    sec_b about their own centres."""
    ca, cb = sec_a.points.mean(axis=0), sec_b.points.mean(axis=0)
    best_k, best_d = None, np.inf
    for k in range(4):
        cand = sec_a.rotate(k * np.pi / 2.0, axis=axis, center=ca)
        d = np.linalg.norm((cand.points - ca) - (sec_b.points - cb), axis=1).max()
        if d < best_d:
            best_k, best_d = k, d
    return best_k


def _reindex_geometry(sec_a, tgt, sigma):
    """Re-express tgt's own geometry through sec_a's own (shared) structural
    B-rep arrays: point i's coordinate becomes tgt's own point sigma[i], an
    edge's interior node becomes whichever of tgt's own edges connects the
    sigma-mapped endpoint pair (matching orientation), and a quad's private
    interior node similarly by its 4 sigma-mapped corners.

    This is a pure relabeling, not a geometric transform: the returned mesh's
    point/edge/quad SET is exactly tgt's own (bit-identical coordinates), just
    reached through sec_a's own index labels -- so a HexMesh.merge against
    the *actual* tgt-patterned mesh later (which is coordinate-proximity
    based, not index based) welds exactly regardless of this relabeling.
    Requires sec_a and tgt to already share identical quad/flip/lines.lines
    (the existing precondition for QuadMesh.blend, satisfied by construction
    since both come from the same quadrant-disc recipe)."""
    assert np.array_equal(sec_a.quad, tgt.quad) and np.array_equal(sec_a.flip, tgt.flip)
    assert np.array_equal(sec_a.lines.lines, tgt.lines.lines), (
        "_reindex_geometry: sec_a and tgt must share identical edge connectivity")
    sigma = np.asarray(sigma, dtype=np.int64)
    new_points = tgt.points[sigma]

    tgt_edges = tgt.lines.lines
    edge_lookup: dict[tuple[int, int], tuple[int, bool]] = {}
    for e in range(tgt_edges.shape[0]):
        u, v = int(tgt_edges[e, 0]), int(tgt_edges[e, 1])
        edge_lookup[(u, v)] = (e, False)
        edge_lookup[(v, u)] = (e, True)
    struct_edges = sec_a.lines.lines
    new_interior = np.empty_like(tgt.lines.interior)
    for e in range(struct_edges.shape[0]):
        u, v = int(struct_edges[e, 0]), int(struct_edges[e, 1])
        te, rev = edge_lookup[(int(sigma[u]), int(sigma[v]))]
        vals = tgt.lines.interior[te]
        new_interior[e] = vals[::-1] if rev else vals
    new_lines = LineMesh(new_points, struct_edges, new_interior,
                         tgt.lines.boundaries, tgt.lines.boundary_tags,
                         tgt.lines.element_tags, order=tgt.lines.order)

    quad_lookup: dict[frozenset[int], int] = {}
    tgt_quads = tgt.quads
    for q in range(tgt_quads.shape[0]):
        quad_lookup[frozenset(int(x) for x in tgt_quads[q])] = q
    struct_quads = sec_a.quads
    new_qinterior = np.empty_like(tgt.interior)
    for q in range(struct_quads.shape[0]):
        corners = frozenset(int(sigma[c]) for c in struct_quads[q])
        new_qinterior[q] = tgt.interior[quad_lookup[corners]]

    return QuadMesh(new_lines, sec_a.quad, sec_a.flip, new_qinterior,
                    tgt.boundaries, tgt.boundary_tags, tgt.element_tags,
                    order=tgt.order)


def pattern_adapter(sec_a, sec_b, axis, n_layers=2, name=""):
    """One short loft that morphs between two same-connectivity quadrant discs
    whose *node patterns* differ slightly (different quadrant-junction params
    put one quadrant's wall nodes ~2% off between the two) -- a plain
    coordinate weld can therefore never be exact across such a seam, but a
    blend is: its first slice IS sec_a's exact points and its last is sec_b's
    own geometry reached through sec_a's own labeling (see
    _reindex_geometry), so both end welds are bit-EXACT (not merely close
    within a coordinate tolerance) while the mismatch is absorbed smoothly
    inside the adapter. The index pairing between the two discs may be rolled
    by a multiple of 90 degrees (both patterns' seams live on the 45-degree
    family, but each disc arrives via its own chain of axis-permuting
    rotations); blending across a rolled pairing twists the adapter into
    inverted elements, so the roll is *measured* -- the k minimizing the
    index-wise deviation about each disc's own centre -- not assumed.

    An earlier version rotated sec_a's own *coordinates* by that roll to
    build the blend's first slice -- exact enough for a coordinate-tolerance
    weld at order 1, but at order > 1 HexMesh.merge also verifies high-order
    edge/face interior nodes to a strict, scale-relative tolerance
    (conform.entity_tol, ~1e-9 x scale) that a rotated *copy* of sec_a's
    corners -- close, not identical, to sec_a's own literal end -- cannot
    satisfy. Relabeling sec_b instead (a pure index permutation, zero
    residual by construction) keeps sec_a completely untouched, so whatever
    it is bit-identical to (e.g. a swept connector's own terminal section)
    stays bit-identical, and the reindexed sec_b's own coordinate SET is
    still exactly sec_b's, so it also welds exactly wherever sec_b's real
    pattern shows up later."""
    ca, cb = sec_a.points.mean(axis=0), sec_b.points.mean(axis=0)
    best_k = _find_roll(sec_a, sec_b, axis)
    cand = sec_a.rotate(best_k * np.pi / 2.0, axis=axis, center=ca)
    # sigma: sec_a's own near-4-fold self-map under the discovered roll (which
    # of sec_a's *own* corners does corner i land closest to, rotated) --
    # entirely about sec_a's own geometry, nothing to do with sec_b yet.
    _, sigma = cKDTree(cand.points).query(sec_a.points)
    sec_b_aligned = _reindex_geometry(sec_a, sec_b, sigma)
    best_d = np.linalg.norm((sec_b_aligned.points - cb) - (sec_a.points - ca), axis=1).max()
    print("pattern_adapter[%s]: roll k=%d, residual index-wise dev %.3e" %
          (name, best_k, best_d))
    assert best_d < 0.2, "no 90-degree roll aligns these two disc patterns"
    if name:
        d_per = np.linalg.norm((sec_b_aligned.points - cb) - (sec_a.points - ca), axis=1)
        worst = np.argsort(-d_per)[:6]
        for i in worst:
            print("  worst[%d] d=%.4f a=%s b=%s" % (
                i, d_per[i], np.round(sec_a.points[i] - ca, 3),
                np.round(sec_b_aligned.points[i] - cb, 3)))
    result = HexMesh.loft(QuadMesh.blend(sec_a, sec_b_aligned,
                                         np.linspace(0.0, 1.0, n_layers + 1)))
    if name:
        sj = result.scaled_jacobian()
        worst_hex = int(np.argmin(sj))
        print("  adapter min sj=%.4e at hex %d / %d" % (sj.min(), worst_hex, result.n_hexes))
        hc = result.hexes[worst_hex]
        print("  hex corner ids:", hc)
        print("  hex corner pts (abs):\n%s" % np.round(result.points[hc], 4))
    return result, best_k

FAST = False
ORDER = 2
N_HALF = 8
RADIAL = np.array([0.0, 0.4, 0.8, 1.0])
CENTER_SCALE = 0.5
n_slices = 3

R_MAIN = 1.2     # == chimera_chain's R_MAIN
R_BR = 0.5       # == chimera_chain's R_BRANCH

L1 = 2.5 * R_MAIN     # T1 main half-length
H1 = 2.5 * R_MAIN     # T1 branch length
L2 = 1.2               # T2's main-leg offset from its own centre (build_tjunction's Z_NEAR default)

kw_t1 = dict(n_half=N_HALF, order=ORDER, radial=RADIAL, center_scale=CENTER_SCALE,
             n_slices_a=n_slices, n_slices_b=n_slices, n_slices_branch=n_slices)
kw_t2 = kw_t1


def normal_of(qm):
    pts = qm.points
    c = pts.mean(axis=0)
    u, s, vt = np.linalg.svd(pts - c)
    return vt[-1]


# chimera_chain's own port cross-section: the same quadrant_disc recipe its
# junctions are built from, so it reproduces a port's pattern exactly.
_chi_kw = dict(order=ORDER, N_QUAD=2, RADIAL=np.array([0.0, 0.6, 1.0]),
               CENTER_SCALE=0.7, PHI_W=np.deg2rad(100.0), N_TRANS=2, N_BRANCH=2)


def fake_chi_disc(center):
    tj = build_tjunction(1.2, 0.5, 3.0, **_chi_kw)
    d = tj.disc_plus  # normal +z, matches chimera's actual inlet/outlet convention
    return d.translate(np.asarray(center) - np.array([0.0, 0.0, 1.2]))


def port_disc(hexmesh, tag, template):
    """``template``'s B-rep *structure* carrying ``hexmesh``'s **own**
    coordinates over its ``tag`` boundary-face group -- corners, shared
    edge-interior nodes and per-quad interior nodes all read straight out of
    ``hexmesh``, paired to the template by nearest corner.

    Needed because a recipe that reproduces one port exactly need not
    reproduce the other: chimera_chain builds its *inlet* as a straight
    ``end_stub`` (which ``fake_chi_disc`` matches to 1e-15) but its *outlet*
    through ``outlet_return()``'s bend, whose end disc lands ~3.4e-3 away.
    At order 1 that difference just welds; at order > 1 ``HexMesh.merge``
    checks shared high-order edge nodes against ``conform.entity_tol``
    (~2e-7 at this model's extent) and the outlet seam fails on 24 edges by
    ~1e-3.  Reading the target off the real mesh removes the guess entirely
    -- measured residual 0.0 at *both* ports."""
    rows = hexmesh.boundaries[hexmesh.boundary_tags == tag]
    poly = hexmesh.hexes[rows[:, 0][:, None], hexmesh.FACE_POINTS[rows[:, 1] - 1, :]]
    gids = np.unique(poly)
    dist, loc = cKDTree(hexmesh.points[gids]).query(template.points)
    g = gids[loc]                              # template point i -> hexmesh point id
    assert len(set(g.tolist())) == g.size, (
        "port_disc[%s]: template does not pair one-for-one with the port" % tag)

    he, hn = hexmesh.edges, hexmesh.edge_nodes
    ekey = {}
    for e in range(he.shape[0]):
        ekey[(int(he[e, 0]), int(he[e, 1]))] = (e, False)
        ekey[(int(he[e, 1]), int(he[e, 0]))] = (e, True)
    tl = template.lines
    new_ei = np.empty_like(tl.interior)
    for e in range(tl.lines.shape[0]):
        u, v = int(tl.lines[e, 0]), int(tl.lines[e, 1])
        idx, rev = ekey[(int(g[u]), int(g[v]))]
        vals = hn[idx]
        new_ei[e] = vals[::-1] if rev else vals
    new_lines = LineMesh(hexmesh.points[g], tl.lines, new_ei, tl.boundaries,
                         tl.boundary_tags, tl.element_tags, order=tl.order)

    hf, hfn = hexmesh.faces, hexmesh.face_nodes
    fkey = {frozenset(int(x) for x in hf[i]): i for i in range(hf.shape[0])}
    new_qi = np.empty_like(template.interior)
    for q in range(template.quads.shape[0]):
        new_qi[q] = hfn[fkey[frozenset(int(g[c]) for c in template.quads[q])]]

    print("  port_disc[%s]: template paired to %.3e, now exact on chimera's own nodes"
          % (tag, dist.max()))
    return QuadMesh(new_lines, template.quad, template.flip, new_qi,
                    template.boundaries, template.boundary_tags,
                    template.element_tags, order=template.order)


# chimera_chain's REAL inlet/outlet disc centres (probed: both at z = -17.5,
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
    print("building real chimera_chain (slow)...")
    _chi_ns = _runpy.run_path(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                           "chimera_chain.py"))
    chi_mesh = _chi_ns["mesh"]

if chi_mesh is None:
    chi_in_disc = fake_chi_disc(CHI_IN)
    chi_out_disc = fake_chi_disc(CHI_OUT)
else:
    chi_in_disc = port_disc(chi_mesh, "inlet", fake_chi_disc(CHI_IN))
    chi_out_disc = port_disc(chi_mesh, "outlet", fake_chi_disc(CHI_OUT))

# -----------------------------------------------------------------------------
# T1: same quadrant pattern family as T2 (not eqtee) so the T1-branch <-> T2-
# main weld is a same-family tolerance weld, not a cross-family one. Equal
# main/branch radius is not directly supported by the quadrant construction
# (the footprint curve degenerates as R_BRANCH -> R_MAIN) -- R_BRANCH is
# 99.9% of R_MAIN (visually/functionally equal) with PHI_W opened up to 172
# degrees, which was enough to pull the crotch caps back to a valid,
# reasonable-quality (~0.21 min scaled Jacobian) octant split; CENTER_SCALE
# has no effect on this at all (checked). main axis -> x, branch -> -y.
# -----------------------------------------------------------------------------
T1_PHI_W = np.deg2rad(172.0)
T1_RATIO = 0.999
ROT_T1 = -np.deg2rad(120.0)
AXIS_T1 = (1.0, -1.0, 1.0)


def build_t1(mirror=False):
    tj = build_tjunction(R_MAIN, R_MAIN * T1_RATIO, H1, order=ORDER, N_QUAD=2,
                         RADIAL=np.array([0.0, 0.6, 1.0]), CENTER_SCALE=0.7,
                         PHI_W=T1_PHI_W, N_TRANS=n_slices, N_BRANCH=n_slices,
                         Z_NEAR=L1)
    ang, axis = ROT_T1, AXIS_T1
    core, da, db, dbr = (tj.core.rotate(ang, axis=axis), tj.disc_minus.rotate(ang, axis=axis),
                        tj.disc_plus.rotate(ang, axis=axis), tj.disc_branch.rotate(ang, axis=axis))
    if mirror:
        # T1_out's main pipe "comes in from the opposite side": an extra 180
        # about the branch's own axis (world y) swaps disc_a/disc_b's world
        # positions (main -x <-> +x) while leaving the -y branch direction
        # itself untouched (a rotation, not a mirror -- element quality is
        # unaffected, unlike HexMesh.transform's own reflection warning).
        yax = (0.0, 1.0, 0.0)
        core, da, db, dbr = (core.rotate(np.pi, axis=yax), da.rotate(np.pi, axis=yax),
                            db.rotate(np.pi, axis=yax), dbr.rotate(np.pi, axis=yax))
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


def build_bend_mesh(section, start_pt3, moves, heading2d, y_fixed, n_layers, last_tag=""):
    path = turtle_path(moves, start=(start_pt3[0], start_pt3[2]), heading=heading2d)
    total = path.total_length

    def centerline(s):
        xz = path.centerline(s)
        return np.stack([xz[:, 0], np.full(xz.shape[0], y_fixed), xz[:, 1]], axis=1)

    def tangent(s):
        xz = path.tangent(s)
        return np.stack([xz[:, 0], np.zeros(xz.shape[0]), xz[:, 1]], axis=1)

    fr = LineMesh.sweep_fractions(path.break_fractions * total, total, total / n_layers)
    return HexMesh.sweep(section, centerline, fr, tangent=tangent, orientation="fixed",
                         up=(0.0, 1.0, 0.0), origin=start_pt3, last_tag=last_tag)


BEND_R1 = 2.0 * R_MAIN
VERTICAL_DROP = 14.0   # matches VERTICAL_RISE -- the two legs read as comparable
ADAPT = 1.0            # length of each pattern-adapter layer (see pattern_adapter)
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
    # a pattern_adapter morphing T1's own disc pattern into chimera's own
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
    core = t1.core.translate(t1_center)
    da = t1.disc_minus.translate(t1_center)
    db = t1.disc_plus.translate(t1_center)
    dbr = t1.disc_branch.translate(t1_center)
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
        p = turtle_path(moves, start=(db_c[0], db_c[2]), heading=heading)
        s = np.array([0.0, 1.0])
        P, T = p.centerline(s), p.tangent(s)
        P3 = np.stack([P[:, 0], np.full(2, db_c[1]), P[:, 1]], axis=1)
        T3 = np.stack([T[:, 0], np.zeros(2), T[:, 1]], axis=1)
        m, o = frames.sweep_placements(disc.points, P3, orientation="fixed",
                                       up=(0.0, 1.0, 0.0), origin=db_c,
                                       path_tangents=T3)[1]
        return disc.transform(m, o)

    conn_chi = build_bend_mesh(db, db_c, moves, heading, db_c[1], n_slices)
    end_sec = _end_section(db)
    print("  [%s] end_sec center=%s normal=%s" %
         (tag, end_sec.points.mean(axis=0), normal_of(end_sec)))
    print("  [%s] tgt center=%s normal=%s"
         % (tag, chi_disc.points.mean(axis=0), normal_of(chi_disc)))
    adapter, _ = pattern_adapter(end_sec, chi_disc, (0.0, 0.0, 1.0), n_layers=2,
                                 name="chi_" + tag)
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
                            da_c[1], n_slices, last_tag=tag)
    # conn_chi's own end and the adapter's own start are the *same* physical
    # points (both derived from db via the same sweep_placements machinery),
    # but the adapter's internal 90-degree roll search (see pattern_adapter)
    # rotates them, and T1's own disc is only *near*-exactly 4-fold symmetric
    # (a ~0.03-unit residual, well under merge()'s global tol=0.005 default)
    # -- so weld *this one seam* locally, at a tolerance sized to that
    # specific residual, rather than loosening the tolerance for the whole
    # assembly (which welded an unrelated, closer-together pair by mistake
    # the one time this was tried globally).
    conn_chi = HexMesh.merge([conn_chi, adapter], tol=0.05)
    return core, [conn_chi], riser, dbr, t1_center


# T1 on the negative-x side (the assembly's own "inlet" riser) connects to
# chimera_chain's OUTLET; T1 on the positive-x side ("outlet" riser) to its
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
    print("  %s min_sj=%.4e" % (_nm, _m.scaled_jacobian().min()))

pieces += [core_in, *conn_chi_in, riser_in, core_out, *conn_chi_out, riser_out]

mesh1 = HexMesh.merge(pieces, tol=0.005)
print("stage1:", mesh1.n_hexes, "hexes, watertight", mesh1.is_watertight(),
     "conforming", mesh1.is_conforming(), "min sj", mesh1.scaled_jacobian().min())

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
H2_BRANCH = 2.5 * R_BR


def build_t2(mirror=False):
    tj = build_tjunction(R_MAIN, R_BR, H2_BRANCH, order=ORDER, N_QUAD=2,
                         RADIAL=np.array([0.0, 0.6, 1.0]), CENTER_SCALE=0.7,
                         PHI_W=np.deg2rad(100.0), N_TRANS=n_slices, N_BRANCH=n_slices)
    ang, axis = ROT_T2, AXIS_T2
    core, dm, dp, dbr = (tj.core.rotate(ang, axis=axis), tj.disc_minus.rotate(ang, axis=axis),
                        tj.disc_plus.rotate(ang, axis=axis), tj.disc_branch.rotate(ang, axis=axis))
    if mirror:
        # T2's branch always comes out world +x (unaffected by ROT_T2, which
        # only turns the main axis z -> y) -- so without this, T2_out's
        # branch points the *same* absolute direction as T2_in's rather than
        # back towards it, and forcing the downstream bend the other way
        # folds the pipe back into itself instead of turning it. An extra
        # 180 about the main axis (world y) flips branch x -> -x while
        # leaving the y-axis main legs on the rotation axis, invariant.
        yax = (0.0, 1.0, 0.0)
        core, dm, dp, dbr = (core.rotate(np.pi, axis=yax), dm.rotate(np.pi, axis=yax),
                            dp.rotate(np.pi, axis=yax), dbr.rotate(np.pi, axis=yax))
    return tj._replace(core=core, disc_minus=dm, disc_plus=dp, disc_branch=dbr)


# base margin below the lower of the two T1 branches' own y. T1's own y is
# pinned exactly to its chimera target's y (build_bend_mesh's connector lives
# in a single fixed-y plane), so T2_SHARED_Y = min(chi target y) - H1 -
# RUN_T1_T2 -- meaning chimera sits exactly (H1 + RUN_T1_T2) above serp's own
# y, for *any* absolute chimera y. Fixed at 10.0 - H1 so that gap is exactly
# the user's requested 10, without having to move chimera_chain itself.
RUN_T1_T2 = 10.0 - H1


def _stub_sections(disc, c, n_dir, dist, n_sec):
    """n_sec sections of disc's own exact pattern, rigidly swept a total
    distance dist along n_dir from centre c (n_sec<2 or dist<=0 -> just
    [disc], unchanged) -- the building block weld_bridge uses on both a's and
    b's own side of its gap, each stub staying bit-exact to its own source
    disc throughout (a pure rigid transform, no shape change).

    n_dir is the disc's OWN true normal at both call sites, not the raw
    centroid-to-centroid direction: the two differ by a tiny angle (neither
    disc is perfectly centred on its own nominal translate target), and
    sweep's "fixed" orientation makes the section exactly perpendicular to
    whatever tangent it is handed -- even at s=0 -- so a tangent a hair off
    the disc's own normal makes the first station a hair off the disc itself
    (measured: 7e-6 with the centroid direction, exactly 0.0 with the
    normal)."""
    if n_sec < 2 or dist <= 0.0:
        return [disc]
    up = (0.0, 0.0, 1.0) if abs(n_dir[2]) < 0.9 else (1.0, 0.0, 0.0)
    s_all = np.linspace(0.0, 1.0, n_sec)

    def path(s):
        s = np.asarray(s, dtype=float)[:, None]
        return c + s * dist * n_dir

    def tang(s):
        return np.tile(n_dir, (np.asarray(s).size, 1))

    placements = frames.sweep_placements(disc.points, path(s_all), orientation="fixed",
                                         up=up, origin=c, path_tangents=tang(s_all))
    return [disc.transform(m, o) for m, o in placements]


def weld_bridge(a, b, n=4, stub_frac=0.3, stub_max=1.5, n_blend=6):
    """Connector between two same-radius, possibly very differently
    *patterned* discs (a T-junction's own leg vs. another T-junction's own
    leg, built by different algorithms -- or, at the T1-branch/T2-main joint,
    by different *functions* entirely).  A short rigid stub is extruded from
    **each** side along its own true normal (bit-exact to its own source
    disc, so both near ends stay exactly bonded to whatever a and b are
    themselves exactly bonded to), and the remaining gap is spanned by a
    straight ``QuadMesh.blend`` -- with the stubs and the blend lofted
    together as **one** ``HexMesh.loft``.

    Building the whole bridge as a single loft is what makes this exact at
    order > 1.  The old version built a rigid sweep and left the far seam to
    ``HexMesh.merge``'s tolerance weld: fine at order 1, but order > 1 also
    verifies shared high-order edge nodes against ``conform.entity_tol``
    (~1e-9 x scale), which an approximate weld cannot meet -- the original
    "non-conforming high-order edge" failure.  One loft has no internal seam
    to verify, so it comes out conformal by construction.

    The blend needs an honest point correspondence, and here the two patterns
    are genuinely far apart: T1's branch disc (``branch()``, stations spaced
    by arc length along the near-degenerate branch/main intersection curve)
    against T2's main leg (``leg()``, uniform angular stations) differ by a
    *median* 0.94 -- comparable to the disc radius itself, not the ~0.03
    ``pattern_adapter`` absorbs at the chimera seam -- and no 90-degree
    rotation improves it, because the mismatch is a difference in station
    *distribution*, not orientation.  Nearest-neighbour matching of the two
    centred point clouds cuts that to a median of 0.001.  Every section on
    b's side is then reindexed through that correspondence (see
    ``_reindex_geometry``), not just the one touching the blend: reindexing
    only the tip leaves the blend's last slice and b's own naturally-labelled
    stub disagreeing, which twists that seam into 96/480 inverted hexes.
    Reindexing the whole stub gives 0/480 inverted, min scaled Jacobian
    0.707."""
    ca, cb = a.points.mean(axis=0), b.points.mean(axis=0)
    length = np.linalg.norm(cb - ca)
    na = normal_of(a)
    na = na if np.dot(na, cb - ca) > 0 else -na
    nb = normal_of(b)
    nb = nb if np.dot(nb, ca - cb) > 0 else -nb
    stub = min(stub_max, stub_frac * length)

    n_stub_sec = max(2, n // 2)
    a_secs = _stub_sections(a, ca, na, stub, n_stub_sec)
    b_secs_raw = _stub_sections(b, cb, nb, stub, n_stub_sec)[::-1]

    a_end, b_end = a_secs[-1], b_secs_raw[0]
    ca_end, cb_end = a_end.points.mean(axis=0), b_end.points.mean(axis=0)
    _, sigma = cKDTree(b_end.points - cb_end).query(a_end.points - ca_end)
    assert len(set(sigma.tolist())) == sigma.size, (
        "weld_bridge: nearest-neighbour matching is not a permutation -- the "
        "two disc patterns are too dissimilar to pair one-for-one")
    b_secs = [_reindex_geometry(a_end, s, sigma) for s in b_secs_raw]

    blend_secs = QuadMesh.blend(a_end, b_secs[0], np.linspace(0.0, 1.0, n_blend + 1))
    return HexMesh.loft(a_secs[:-1] + blend_secs + b_secs[1:])


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
    also keeps the weld_bridge sweep aligned with both discs' own normal,
    avoiding the large-rotation mismatch a diagonal offset causes."""
    t2 = build_t2(mirror=mirror)
    br_pos = source_disc.points.mean(axis=0)
    run = br_pos[1] - t2_y - L2
    assert run > 0.1, "t2_y too close to source disc y=%.2f (side)" % br_pos[1]
    t2_center = np.array([br_pos[0], t2_y, br_pos[2]])
    core = t2.core.translate(t2_center)
    da = t2.disc_minus.translate(t2_center)   # -y, on to the next T2 (or capped)
    db = t2.disc_plus.translate(t2_center)    # +y, faces back upstream
    dbr = t2.disc_branch.translate(t2_center)  # +/-x (mirror), out to a serpentine
    conn = weld_bridge(source_disc, db)
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
    levels[-1]["dead"] = HexMesh.extrude(levels[-1]["da"], 1.5 * R_BR, 2,
                                         axis=(0.0, -1.0, 0.0), last_tag="wall")
    return levels


chain_in = t2_chain(br_in, mirror=False)
chain_out = t2_chain(br_out, mirror=True)

pieces2 = pieces + [p for lv in (*chain_in, *chain_out)
                    for p in (lv["core"], lv["conn"], *( [lv["dead"]]
                                                         if "dead" in lv else []))]
mesh2 = HexMesh.merge(pieces2, tol=0.005)
print("stage2:", mesh2.n_hexes, "hexes,", 2 * N_T2, "T2 junctions, watertight",
      mesh2.is_watertight(), "conforming", mesh2.is_conforming(),
      "min sj", mesh2.scaled_jacobian().min())

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
# T2_out's bridge is absorbed by weld_bridge (same tool already used for the
# T1-to-T2 joins above), not by forcing an exact but colliding registration.
# -----------------------------------------------------------------------------

# -- the coil's own fixed shape (traced from the reference photo; do not
# reshape or rescale -- see trace_serp3.py) ----------------------------------
PASS_LEN = 136.0
U_R = 2.5        # tight radius: bottom turns + top hairpins
U_R_MID = 4.0    # wider radius: middle bridge between the two half-coils
R_HOOK = U_R_MID
HOOK_JOG = 5.0
HOOK_DROP = 20.0
RAISE = 4.0      # extra length on the middle two passes -> raised middle bridge

_hook_in = [("line", HOOK_DROP, 0.0), ("arc", R_HOOK, 90.0),
            ("line", HOOK_JOG, 0.0), ("arc", R_HOOK, -90.0)]
_hook_out = [("arc", R_HOOK, -90.0), ("line", HOOK_JOG, 0.0),
             ("arc", R_HOOK, 90.0), ("line", HOOK_DROP, 0.0)]

COIL_MOVES = (_hook_in
    + [("line", PASS_LEN + RAISE, 0.0), ("arc", U_R, -180.0)]
    + [("line", PASS_LEN, 0.0), ("arc", U_R, 180.0)]
    + [("line", PASS_LEN, 0.0), ("arc", U_R, -180.0)]
    + [("line", PASS_LEN + RAISE, 0.0), ("arc", U_R_MID, 180.0)]
    + [("line", PASS_LEN + RAISE, 0.0), ("arc", U_R, -180.0)]
    + [("line", PASS_LEN, 0.0), ("arc", U_R, 180.0)]
    + [("line", PASS_LEN, 0.0), ("arc", U_R, -180.0)]
    + [("line", PASS_LEN + RAISE, 0.0)]
    + _hook_out)

_coil_local = turtle_path(COIL_MOVES, start=(0.0, 0.0), heading=0.0)
_coil_end_uv = _coil_local.centerline(np.array([1.0]))[0]
COIL_DV = _coil_end_uv[1]   # local (u, v) end offset is (0, COIL_DV) exactly

BEND_R_CONN = 3.0
VERTICAL_RUN = 10.0
GAP_Z = 1.0   # deliberate short leftover so weld_bridge has a nonzero span


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
    p = turtle_path(moves, start=(c[0], c[2]), heading=heading)
    s = np.array([0.0, 1.0])
    P = p.centerline(s)
    T = p.tangent(s)
    P3 = np.stack([P[:, 0], np.full(2, y_fixed), P[:, 1]], axis=1)
    T3 = np.stack([T[:, 0], np.zeros(2), T[:, 1]], axis=1)
    m, o = frames.sweep_placements(section.points, P3, orientation="fixed",
                                   up=(0.0, 1.0, 0.0), origin=c, path_tangents=T3)[1]
    return section.transform(m, o)


TOTAL_COIL = _coil_local.total_length
# The sweep target must subdivide even the tightest (U_R) 180-degree turn into
# several stations -- sweep_fractions rounds a segment's own length/target to
# the *nearest* station count, so a target only a little under the arc's own
# length (2*pi*U_R/2 = 7.85) rounds down to a single, unsubdivided station
# spanning the full 180 degrees: two opposite-facing cross-sections linearly
# interpolated into one wildly distorted (near-zero-volume) hex. Measured:
# 6.0 does exactly this (round(7.85/6.0) == 1); 2.0 does not.
COIL_TARGET_LEN = 2.0


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
    path = turtle_path(inflow_moves, start=(ci[0], ci[2]), heading=0.0)
    total = path.total_length
    fr = LineMesh.sweep_fractions(path.break_fractions * total, total, COIL_TARGET_LEN)

    def centerline(s):
        xz = path.centerline(s)
        return np.stack([xz[:, 0], np.full(xz.shape[0], ci[1]), xz[:, 1]], axis=1)

    def tangent(s):
        xz = path.tangent(s)
        return np.stack([xz[:, 0], np.zeros(xz.shape[0]), xz[:, 1]], axis=1)

    inflow = HexMesh.sweep(dbr_i, centerline, fr, tangent=tangent,
                           orientation="fixed", up=(0.0, 1.0, 0.0), origin=ci)
    assert inflow.is_conforming(), "coil sweep produced a non-conforming mesh"
    conn_o = build_bend_mesh(dbr_o, co, moves_out, np.pi, co[1], n_slices)

    # conn_o's own end (dbr_o's pattern, through its own bend) and the coil's
    # own end (dbr_i's pattern, carried the whole way) are different patterns
    # landing GAP_Z apart by construction -- weld_bridge (same tool as the
    # T1-to-T2 joins above) closes that last short gap.
    bridge = weld_bridge(_end_section(dbr_o, moves_out, np.pi, co[1]),
                         _end_section(dbr_i, inflow_moves, 0.0, ci[1]), n=3)
    return [inflow, conn_o, bridge]


coils = [p for lv_i, lv_o in zip(chain_in, chain_out)
         for p in build_coil(lv_i["dbr"], lv_o["dbr"])]

pieces3 = pieces2 + coils
mesh3 = HexMesh.merge(pieces3, tol=0.005)
print("stage3:", mesh3.n_hexes, "watertight", mesh3.is_watertight(),
     "conforming", mesh3.is_conforming(), "min sj", mesh3.scaled_jacobian().min())
# -- registration check: does the connector's rising end actually land point-
# for-point on chimera's own disc pattern (mod the quadrant disc's 90-degree
# symmetry)?  The fake stand-in discs use the identical pattern/params as the
# real chimera_chain, so this check is valid in FAST mode too. conn_chi_in
# targets CHI_OUT and conn_chi_out targets CHI_IN (the swap above), so the
# discs paired here follow the same swap, not the "in"/"out" name.
for nm, conn, disc in (("in", conn_chi_in[-1], chi_out_disc),
                       ("out", conn_chi_out[-1], chi_in_disc)):
    d, _ = cKDTree(conn.points).query(disc.points)
    print("chimera %s registration: max dist %.3e" % (nm, d.max()))
    print("chimera %s adapter min sj: %.4e" % (nm, conn.scaled_jacobian().min()))

manifold = pieces3

if chi_mesh is not None:
    # chimera's own inlet/outlet faces are welded away into interior planes
    # here, so their tags must go (the combined mesh's inlet/outlet are the
    # riser tops); a stale tagged interior face would export as a bogus BC.
    _keep = chi_mesh.boundary_tags == "wall"
    chi_mesh.boundaries = chi_mesh.boundaries[_keep]
    chi_mesh.boundary_tags = chi_mesh.boundary_tags[_keep]
    mesh_out = HexMesh.merge([*manifold, chi_mesh], tol=0.005)
else:
    chi_in_cap = HexMesh.extrude(chi_in_disc, 0.3, 1, axis=(0, 0, 1), last_tag="wall")
    chi_out_cap = HexMesh.extrude(chi_out_disc, 0.3, 1, axis=(0, 0, 1), last_tag="wall")
    mesh_out = HexMesh.merge([*manifold, chi_in_cap, chi_out_cap], tol=0.005)

print("mesh_out:", mesh_out.n_hexes, "hexes, watertight", mesh_out.is_watertight(),
      "conforming", mesh_out.is_conforming())
print(mesh_out.topology_report())

mesh = mesh_out
OUT_NAME = "chimera_full"
GROUPS = {"wall": "W  ", "inlet": "v  ", "outlet": "O  "}
export.to_re2(mesh, OUT_NAME + ".re2", groups=GROUPS)
export.to_vtu(mesh, OUT_NAME + ".vtu", groups=GROUPS)
print("groups:", ", ".join(mesh.boundary_group_tags))
