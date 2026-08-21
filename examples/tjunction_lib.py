"""Standalone, parametrized quadrant T-junction builder -- adapted from
examples/quadrant_pipe_tjunction.py, generalized into a function so it can be
called multiple times at different radii/positions.  Local frame: main pipe axis
= Z (legs at z = -Z_NEAR "minus" and z = +Z_NEAR "plus"), branch axis = X
(opening at x = H_BRANCH), exactly quadrant_pipe_tjunction.py's own convention.
``quadrant_pipe_tjunction.py`` is this function's reference caller: it passes
``branch_tag="branch"`` (the one boundary this function can tag that a caller
cannot reach after the fact -- the branch stub's far cap is already inside
``core``) and extrudes the two ``disc_minus``/``disc_plus`` legs onward itself.

Returns a TJunction(core, disc_minus, disc_plus, disc_branch): ``core`` is the
merged crotch/transition hex block (tagged ``element_tag`` throughout, and
``branch_tag`` on the branch's far cap if given), and the three discs are the
*plain* QuadMesh cross-sections at each leg's outward end -- for a caller to
continue building from (extrude/loft/sweep) rather than re-deriving the
junction's own geometry.  The two legs stop at the plain disc rather than being
capped here, since how far to carry them (and by what -- a straight extrude, a
sweep along a bend) is a per-caller choice; the branch is the one piece with
nothing left for a caller to add when it needs none, so it is capped inline.
"""
from __future__ import annotations

from collections import defaultdict, namedtuple

import numpy as np

from nekmeshpy import (
    ElementTags,
    LineMesh,
    QuadMesh,
    hexmesh,
    linemesh,
    quadmesh,
)
from nekmeshpy.core import surfaces
from nekmeshpy.core.fields import gll_nodes, lagrange_matrix
from nekmeshpy.hexmesh import Seam
from nekmeshpy.linemesh import Seam as PointSeam
from nekmeshpy.pointmesh import PointMesh
from nekmeshpy.quadmesh import Seam as EdgeSeam
from nekmeshpy.quadmesh.query import element_blocks

TJunction = namedtuple("TJunction", "core disc_minus disc_plus disc_branch")

#: default radial station fractions (a module-level singleton so it is not rebuilt --
#: or flagged -- as an argument default)
_RADIAL_DEFAULT = np.array([0.0, 0.6, 1.0])

#: Bounds on the bypass half-angle.  Below the floor the bypass quadrant spans so much
#: of the pipe that it folds; above the ceiling the two side quadrants do.  Both were
#: measured, not derived -- see ``auto_params``.
_PHI_W_FLOOR, _PHI_W_CEIL = np.deg2rad(60.0), np.deg2rad(170.0)

#: How much of the cob's square the branch bore may fill before
#: :func:`build_cob`'s collar folds -- see the check there.
_COB_FILL_MAX = 0.85



def footprint_angle(ratio):
    """Cylindrical half-angle subtended by the branch footprint, ``asin(r/sqrt(2))``.

    The footprint quadrant of the composite junction face spans exactly twice this, so
    it is the natural scale for the other three quadrants -- and the hard lower bound
    on ``PHI_W``, since below it the two side quadrants invert."""
    return float(np.arcsin(min(float(ratio), 1.0) / np.sqrt(2.0)))


def auto_params(R_MAIN, R_BRANCH):
    """``(PHI_W, CAP_TIP_BIAS, ORIGIN)`` chosen for the radius ratio.

    The quadrant construction has one shape that has to work across a 10:1 range of
    branch-to-main radius, and a single fixed set of these three -- which is what this
    file shipped before -- is only good near the middle of it.  Measured over the
    order-2 scaled Jacobian on 16 ratios from 0.10 to 0.99 (and validated on 7 more it
    was not fitted to), the fixed defaults leave the junction **inverted** above
    ratio 0.8 and fail to build at all near 1.0; these keep the worst element above
    0.10 across the whole range and within 0.08 of the best the parameters can do at
    any ratio.

    ``PHI_W = 5 * footprint_angle`` is the fit, and it has a plain reading: the side
    quadrant then spans ``4 * footprint_angle``, exactly **twice** the footprint
    quadrant's own ``2 * footprint_angle``.  The clamps are where that stops being
    achievable -- at small ratio it would leave the bypass spanning almost the whole
    pipe, at large ratio it would exceed 180 degrees.

    ``ORIGIN`` walks the junction hub from near the branch wall back to the axis as the
    branch grows: a small branch sits in a shallow dimple far off-axis, and putting the
    hub out there is what keeps the crotch caps from being stretched across the pipe.

    **The three ports are unaffected.**  ``disc_minus`` / ``disc_plus`` /
    ``disc_branch`` come out bit-identical whatever these are set to -- only the
    junction's own interior changes -- so re-tuning them can never disturb a seam
    something downstream is already bonded to.

    Caveat worth knowing: below ratio ~0.3 the quality gain is paid for in element
    *size* uniformity.  At ratio 0.15 the worst element improves 8x (0.022 -> 0.18) but
    the largest-to-smallest element volume ratio grows from ~100 to ~900.  That is
    inherent to a small branch on a large pipe, not an artefact of the tuning, but it
    is the reason the hub shift is capped rather than pushed as far as the Jacobian
    alone would like."""
    r = float(R_BRANCH) / float(R_MAIN)
    phi_w = float(np.clip(5.0 * footprint_angle(r), _PHI_W_FLOOR, _PHI_W_CEIL))
    origin = np.array([float(np.clip(0.80 - 0.6 * r, -0.4, 0.8)) * float(R_MAIN),
                       0.0, 0.0])
    return phi_w, 0.20, origin


def build_tjunction(R_MAIN, R_BRANCH, H_BRANCH, *, Z_NEAR=1.2, N_QUAD=2,
                     RADIAL=None, CENTER_SCALE=0.7, QUADRANT_SCALE=0.7,
                     order=2, PHI_W=None, CAP_TIP_BIAS=None,
                     N_TRANS=5, N_BRANCH=4, ORIGIN=None, element_tag="",
                     branch_tag="", port_tags=None):
    RADIAL = _RADIAL_DEFAULT if RADIAL is None else RADIAL
    # each of the three defaults to the ratio-dependent choice; pass one explicitly to
    # override just that one
    _phi_w, _bias, _origin = auto_params(R_MAIN, R_BRANCH)
    PHI_W = _phi_w if PHI_W is None else PHI_W
    CAP_TIP_BIAS = _bias if CAP_TIP_BIAS is None else CAP_TIP_BIAS
    N = N_QUAD
    # The junction hub: the point every quadrant seam radiates from, and the base the
    # crotch caps' own centres are measured from.  Defaults to the main axis; moving it
    # toward the branch is one of the three knobs that trade quality between the crotch
    # caps and the transitions as the radius ratio changes.
    ORIGIN = _origin if ORIGIN is None else np.asarray(ORIGIN, dtype=float).reshape(3)
    TQ = np.deg2rad(45.0 - 90.0 * np.arange(5))

    def footprint(t):
        t = np.asarray(t, dtype=float)
        y = R_BRANCH * np.sin(t)
        z = R_BRANCH * np.cos(t)
        return np.stack([np.sqrt(R_MAIN**2 - y**2), y, z], axis=1)

    def opening(t):
        t = np.asarray(t, dtype=float)
        return np.stack([np.full(t.shape, H_BRANCH),
                         R_BRANCH * np.sin(t), R_BRANCH * np.cos(t)], axis=1)

    def cyl(phi, z):
        phi, z = np.asarray(phi, dtype=float), np.asarray(z, dtype=float)
        return np.stack([R_MAIN * np.cos(phi), R_MAIN * np.sin(phi),
                         np.broadcast_to(z, phi.shape)], axis=-1)

    def cyl_params(p):
        p = np.asarray(p, dtype=float)
        return np.stack([np.arctan2(p[..., 1], p[..., 0]), p[..., 2]], axis=-1)

    def cyl_pts(u):
        return cyl(u[:, 0], u[:, 1])

    def wall_mesh(w):
        return linemesh.on_surface(w, cyl_pts, order=order)

    def ruled_wall(pa, pb):
        return surfaces.ruled(pa, pb, 2 * N_QUAD)

    def foot_wall(fr):
        return surfaces.curve(lambda t: cyl_params(footprint(t)), fr)

    def plain_wall(w, phi0, phi1, z):
        return surfaces.reparam(w, (phi0, z), (phi1, z))

    def shift_wall(w, turns):
        return surfaces.shift(w, (2.0 * np.pi * turns, 0.0))

    def seam(target, fr, center=ORIGIN):
        return linemesh.line(center, target, fr, order=order)

    def quadrant(arc, seam1, seam2, wall_tag="", side_tags=None, element_tag=""):
        return quadmesh.quadrant_ogrid(arc, seam1, seam2, RADIAL,
                                       center_scale=CENTER_SCALE, wall_tag=wall_tag,
                                       element_tag=element_tag, side_tags=side_tags)

    def ring_sides(q, n):
        """Quadrant ``q`` of a ring of ``n``: its two seams named so the next quadrant
        round can be told which one it meets.  ``quadrant_ogrid`` takes them keyed
        ``seam1`` / ``seam2``, and ``seam2`` of one *is* ``seam1`` of the next."""
        return {"seam1": "s%d" % q, "seam2": "s%d" % ((q + 1) % n)}

    def ring(quads):
        """``n`` quadrants closed into a disc about their shared hub.  Every seam is
        stated, the hub included -- each seam line runs through it, so welding them all
        welds the hub with them."""
        n = len(quads)
        # lower block index first, always.  ``own="a"`` writes the a-side's coordinates
        # onto the b-side, while the surviving point id is the *lowest* of the welded
        # pair -- so stating the wrap-around seam as (n-1, 0) would keep block 0's id
        # carrying block n-1's coordinates, and the two differ in the last ulp.
        return quadmesh.attach(quads, [
            EdgeSeam(min(q, (q + 1) % n), "s%d" % ((q + 1) % n),
                     max(q, (q + 1) % n), "s%d" % ((q + 1) % n))
            for q in range(n)])

    def disc_at(arcs, center, names=("", "", "", "")):
        # quadrant_disc's own recipe, inlined: quadmesh.ogrid/spined_ogrid now cover the
        # single-boundary-loop case, but this junction's per-station arcs are built
        # independently per quadrant with an off-centroid hub, so they still need the
        # general form.
        #
        # ``names`` tags each quadrant as an *element*, which is how the disc, used as a
        # loft's bounding slice, hands one name per quadrant to that cap: a cap side is
        # the slice element, so ``first_tag`` picks the four names up on its own.
        fr = quadmesh.quadrant_seam_fractions(N_QUAD, RADIAL, QUADRANT_SCALE)
        seams = [seam(a.points[0], fr, center) for a in arcs]
        seams.append(seams[0])
        return ring([quadrant(arcs[q], seams[q], seams[q + 1], wall_tag="wall",
                              side_tags=ring_sides(q, 4), element_tag=names[q])
                     for q in range(4)])

    def plain_walls(walls, z, sign):
        ang = sign * np.deg2rad(-45.0 + 90.0 * np.arange(5))
        return [plain_wall(walls[q], ang[q], ang[q + 1], z) for q in range(4)]

    FR = quadmesh.quadrant_seam_fractions(N_QUAD, RADIAL, QUADRANT_SCALE)

    P = [footprint(TQ[q:q + 1])[0] for q in range(4)]
    WP, WM = cyl(PHI_W, 0.0), cyl(-PHI_W, 0.0)

    SP = [seam(p, FR) for p in P]
    SWP, SWM = seam(WP, FR), seam(WM, FR)

    FQ_FR = [linemesh.arclength_fractions(footprint, 2 * N_QUAD,
                                          t_range=(TQ[q], TQ[q + 1])) for q in range(4)]
    FQ = [linemesh.loft_fn(footprint, fr, order=order) for fr in FQ_FR]

    UP = [cyl_params(p) for p in P]
    UWP, UWM = np.array([PHI_W, 0.0]), np.array([-PHI_W, 0.0])

    TURN = np.array([2.0 * np.pi, 0.0])
    W_R = [foot_wall(FQ_FR[0][::-1]),
           ruled_wall(UP[0], UWP),
           ruled_wall(UWP, TURN - UWP),
           ruled_wall(TURN - UWP, UP[1] + TURN)]
    W_L = [foot_wall(FQ_FR[2][::-1]),
           ruled_wall(UP[2], UWM),
           ruled_wall(UWM, -TURN - UWM),
           ruled_wall(-TURN - UWM, UP[3] - TURN)]

    SIDE_RP, SIDE_RM = wall_mesh(W_R[1]), wall_mesh(W_R[3])
    SIDE_LM, SIDE_LP = wall_mesh(W_L[1]), wall_mesh(W_L[3])

    def arc_mids(walls):
        return [surfaces.node(w, N) for w in walls]

    def cap(sa, sb, sc, ab, bc, ca, tip_bias=CAP_TIP_BIAS, names=("", "", "")):
        (m_ab, w_ab), (m_bc, w_bc), (m_ca, w_ca) = ab, bc, ca
        mids = arc_mids((w_ab, w_bc, w_ca))
        wc_param = quadmesh.tri_patch_tip(*mids, tip_bias=tip_bias)
        wc = cyl_pts(wc_param[None, :])[0]
        # three of the four tetrahedron sides are seams against a transition, and
        # ``tetra`` carries each side's single element tag onto the faces it becomes
        return hexmesh.tetra([quadrant(m_ab, sa, sb, element_tag=names[0]),
                              quadrant(m_bc, sb, sc, element_tag=names[1]),
                              quadrant(m_ca, sc, sa, element_tag=names[2]),
                              quadmesh.tri_patch(cyl_pts, w_ab, w_bc, w_ca,
                                                 order=order, tip_bias=tip_bias,
                                                 mids=mids, element_tag="wall")],
                             center=ORIGIN + CENTER_SCALE* np.sqrt(1.5) * (wc - ORIGIN),
                             element_tag=element_tag)

    def unnamed(sec, names):
        """A port disc as the caller gets it: stripped of the interior seam names.

        Those name the core's own nine welds and are the core's business.  The disc is
        handed out as a template to loft a pipe off, and ``first_tag`` defaults to the
        bounding slice's own ``element_tags`` -- so left on, they would ride straight onto
        that pipe's cap and out into the export as boundary conditions."""
        drop = {n: "" for n in names if n}
        return quadmesh.retag_element(sec, drop) if drop else sec

    def leg(walls, sign, port_tag, names):
        z = sign * Z_NEAR
        w_plain = plain_walls(walls, z, sign)

        def station(s):
            return disc_at(
                [wall_mesh(surfaces.blend(walls[q], w_plain[q], s)) for q in range(4)],
                (1.0 - s) * ORIGIN + s * np.array([0.0, 0.0, z]), names)

        plain = station(1.0)
        # ``first_tag`` is left to its default, which is the near station's own
        # ``element_tags`` -- the four seam names.  ``last_tag`` can no longer be left to
        # its own: every station carries those names, so the *far* cap would inherit them
        # too and export four seam names as boundary conditions on the open port.  Hence
        # ``port_tag`` verbatim rather than ``port_tag or None`` -- "" is an explicit
        # override to untagged, None is "not asked for".
        transition = hexmesh.loft_fn(station, np.linspace(0.0, 1.0, N_TRANS + 1),
                                     order=order, element_tags=element_tag or None,
                                     last_tag=port_tag)
        return transition, unnamed(plain, names)

    def branch(names):
        open_arcs = [linemesh.loft_fn(opening, fr, order=order) for fr in FQ_FR]
        t = np.linspace(0.0, 1.0, N_BRANCH + 1)
        walls = [linemesh.blend(f, o, t) for f, o in zip(FQ, open_arcs)]
        c_open = np.array([H_BRANCH, 0.0, 0.0])
        sections = [disc_at([w[i] for w in walls],
                            (1.0 - t[i]) * ORIGIN + t[i] * c_open, names)
                   for i in range(t.size)]
        return (hexmesh.loft(sections, element_tags=element_tag or None,
                             last_tag=branch_tag),      # verbatim: see ``leg``
                unnamed(sections[-1], names))

    # ``port_tags`` names the two leg openings on the core so a caller can
    # ``hexmesh.attach`` its own legs to them by name rather than have ``merge``
    # rediscover the seam from coordinates.  Off by default: a *named* face that a
    # later ``merge`` buries is not inert -- the exporter writes one boundary row per
    # hex carrying one -- so only a caller that means to attach should ask for them.
    _pt_minus, _pt_plus = port_tags if port_tags else ("", "")

    # The core's nine interior seams, each named identically on the two blocks that meet
    # across it.  Every pair of the five blocks meets except the two crotch caps, so this
    # is a complete graph minus one edge -- and each name is carried by exactly one
    # quadrant of one transition's near disc and one patch of the block opposite.
    J_PM, J_PB, J_MB = "j_plus_minus", "j_plus_branch", "j_minus_branch"
    J_P_CP, J_P_CM = "j_plus_capP", "j_plus_capM"
    J_M_CP, J_M_CM = "j_minus_capP", "j_minus_capM"
    J_B_CP, J_B_CM = "j_branch_capP", "j_branch_capM"

    # each tuple is in its own disc's quadrant order: a leg's walls run
    # footprint / one crotch cap / bypass / the other cap, and the branch's run FQ[0..3]
    trans_plus, disc_plus = leg(W_R, 1, _pt_plus, (J_PB, J_P_CP, J_PM, J_P_CM))
    trans_minus, disc_minus = leg(W_L, -1, _pt_minus, (J_MB, J_M_CM, J_PM, J_M_CP))
    trans_branch, disc_branch = branch((J_PB, J_B_CM, J_MB, J_B_CP))

    blocks = [trans_plus, trans_minus, trans_branch,
              cap(SP[0], SP[3], SWP,
                  (linemesh.reverse(FQ[3]), foot_wall(FQ_FR[3][::-1])),
                  (linemesh.reverse(SIDE_LP), shift_wall(surfaces.reverse(W_L[3]), 1)),
                  (linemesh.reverse(SIDE_RP), surfaces.reverse(W_R[1])),
                  names=(J_B_CP, J_M_CP, J_P_CP)),
              cap(SP[2], SP[1], SWM,
                  (linemesh.reverse(FQ[1]), foot_wall(FQ_FR[1][::-1])),
                  (linemesh.reverse(SIDE_RM), shift_wall(surfaces.reverse(W_R[3]), -1)),
                  (linemesh.reverse(SIDE_LM), surfaces.reverse(W_L[1])),
                  names=(J_B_CM, J_P_CM, J_M_CM))]
    # ``attach``, not ``merge``: the junction is nine stated interfaces, not a proximity
    # search over five whole boundaries.  Lower block index first in every seam.
    core = hexmesh.attach(blocks, [
        Seam(0, J_PM, 1, J_PM), Seam(0, J_PB, 2, J_PB),
        Seam(0, J_P_CP, 3, J_P_CP), Seam(0, J_P_CM, 4, J_P_CM),
        Seam(1, J_MB, 2, J_MB),
        Seam(1, J_M_CP, 3, J_M_CP), Seam(1, J_M_CM, 4, J_M_CM),
        Seam(2, J_B_CP, 3, J_B_CP), Seam(2, J_B_CM, 4, J_B_CM)])
    return TJunction(core, disc_minus, disc_plus, disc_branch)


def build_eqtee(R, Z_NEAR, H_BRANCH, *, n_half=8, order=2, n_layers_main=5,
                n_layers_branch=5, radial=None, center_scale=0.7,
                quadrant_scale=0.7, element_tag="", branch_tag=""):
    """Equal-radius T-junction: main and branch pipes the *same* radius ``R``, meeting
    in a pair of planar elliptical collars (adapted from ``circular_pipe_tjunction.py``)
    rather than a quadrant footprint/crotch-cap construction -- which
    :func:`build_tjunction` needs, since its footprint curve degenerates as the radius
    ratio approaches 1. Same ``TJunction(core, disc_minus, disc_plus, disc_branch)``
    return and the same local frame as :func:`build_tjunction` (main axis Z, legs at
    ``z = -Z_NEAR``/``+Z_NEAR``; branch axis X, opening at ``x = H_BRANCH``), so the two
    are interchangeable to a caller. ``n_half`` must be a multiple of 4.
    ``center_scale``/``quadrant_scale`` are :func:`quadmesh.spined_ogrid
    <nekmeshpy.quadmesh.shape.spined_ogrid>`'s, which every station here is built
    from."""
    radial = _RADIAL_DEFAULT if radial is None else radial
    M = 2 * n_half

    # -- native frame here is (main=X, branch=Z), circular_pipe_tjunction's own --
    def arc_main_lower():
        return linemesh.arc(R, n_half, center=(0.0, 0.0, 0.0), normal=(-1.0, 0.0, 0.0),
                            start_theta=0.0, end_theta=np.pi,
                            first_tag="A1", last_tag="A2", order=order)

    def arc_collar(xside):
        def f(t):
            return np.column_stack(
                [xside * R * np.sin(t), R * np.cos(t), R * np.sin(t)])
        return linemesh.loft_fn(
            f, linemesh.arclength_fractions(f, n_half, t_range=(0.0, np.pi)),
            first_tag="A1", last_tag="A2", order=order)

    def join_arcs(p, q):
        # both shared ends are named, so the ring closes by two *stated* joins;
        # ``reverse`` carries a point's tag with it, so ``A1`` still names ``A1``
        # whichever way round the arc is stored
        return linemesh.attach([p, linemesh.reverse(q)],
                               [PointSeam(0, "A1", 1, "A1"),
                                PointSeam(0, "A2", 1, "A2")])

    def opening_main(x0):
        return linemesh.circle(R, M, center=(x0, 0.0, 0.0), normal=(-1.0, 0.0, 0.0),
                               order=order)

    def opening_branch(z0):
        return linemesh.circle(R, M, center=(0.0, 0.0, z0), normal=(0.0, 0.0, 1.0),
                               start_theta=np.pi / 2, order=order)

    def leg_slices(open_ring, seam_ring, n_layers):
        loops = linemesh.blend(open_ring, seam_ring, np.linspace(0.0, 1.0, n_layers + 1))
        return [quadmesh.spined_ogrid(loop, radial, center_scale=center_scale,
                                      quadrant_scale=quadrant_scale, wall_tag="wall")
               for loop in loops]

    a_lm = arc_main_lower()
    a_lb = arc_collar(-1.0)
    a_rb = arc_collar(+1.0)
    seam_left = join_arcs(a_lm, a_lb)
    seam_right = join_arcs(a_lm, a_rb)
    seam_branch = join_arcs(a_lb, a_rb)

    slices_minus = leg_slices(opening_main(-Z_NEAR), seam_left, n_layers_main)
    slices_plus = leg_slices(opening_main(Z_NEAR), seam_right, n_layers_main)
    slices_branch = leg_slices(opening_branch(H_BRANCH), seam_branch, n_layers_branch)

    def cap_tags(slice_, first, second):
        """Name a leg's seam cap by half-disc.  ``spined_ogrid`` welds the two halves in
        order, so the first half of the quads is ``join_arcs``' first arc -- and each
        half is shared with a *different* leg, so one name for the whole cap will not
        do.  ``last_tag`` takes an ``ElementTags`` over the slice's own elements."""
        half = slice_.n_quads // 2
        return ElementTags.from_dense(
            np.array([first] * half + [second] * (slice_.n_quads - half)))

    # the three legs meet pairwise on the three arcs: a_lm joins minus to plus, a_lb
    # joins minus to branch, a_rb joins plus to branch
    core = hexmesh.attach(
        [hexmesh.loft(slices_minus, element_tags=element_tag or None,
                      last_tag=cap_tags(slices_minus[-1], "attach1", "attach2")),
         hexmesh.loft(slices_plus, element_tags=element_tag or None,
                      last_tag=cap_tags(slices_plus[-1], "attach1", "attach3")),
         hexmesh.loft(slices_branch, element_tags=element_tag or None,
                      last_tag=cap_tags(slices_branch[-1], "attach2", "attach3"))],
        [Seam(0, "attach1", 1, "attach1"),
         Seam(0, "attach2", 2, "attach2"),
         Seam(1, "attach3", 2, "attach3")])
    disc_minus, disc_plus, disc_branch = (slices_minus[0], slices_plus[0],
                                          slices_branch[0])

    # rotate the native (main=X, branch=Z) frame into build_tjunction's own
    # (main=Z, branch=X): swapping two axes while leaving the third alone is a
    # reflection (det -1, inverts every element -- see the mirror note in
    # CLAUDE.md), so this also flips Y, which keeps it a proper rotation (det +1):
    # 180 degrees about the X=Z diagonal maps +X->+Z, +Z->+X, +Y->-Y.
    ang, axis = np.pi, (1.0, 0.0, 1.0)
    return TJunction(
        hexmesh.rotate(core, ang, axis=axis),
        quadmesh.rotate(disc_minus, ang, axis=axis),
        quadmesh.rotate(disc_plus, ang, axis=axis),
        quadmesh.rotate(disc_branch, ang, axis=axis))


def _map_section(qm, fn):
    """``qm``'s topology with **every** node table pushed through ``fn``.

    A section owns three of them -- the shared corners, the shared edges' interior nodes,
    and its own per-quad interior -- and at ``order > 1`` all three carry geometry.
    Mapping only ``points`` leaves the other two holding whatever the template had, which
    is why a later weld then rejects the block: two elements meeting on a shared edge
    disagree about where that edge's interior nodes are."""
    def m(a):
        return fn(a.reshape(-1, 3)).reshape(a.shape) if a.size else a

    lm = qm.line_mesh
    return QuadMesh(LineMesh(PointMesh(fn(qm.points), lm.point_tags), lm.lines,
                             m(lm.interior), lm.element_tags),
                    qm.quads, qm.orient, m(qm.interior), qm.element_tags)


def build_cob(R_MAIN, R_BRANCH, H_BRANCH, *, Z_NEAR=None, N_THETA_MAIN=32,
              RADIAL_MAIN=3, CENTER_SCALE_MAIN=0.8, N_THETA_BRANCH=16,
              RADIAL_BRANCH=None, CENTER_SCALE_BRANCH=0.8, N_BRANCH=10,
              N_Z_LEG=None, order=2, element_tag="", branch_tag="",
              wall_tag="wall", port_tags=None):
    """Unequal-radius T-junction by the **cob** construction -- the branch cut straight
    through the main pipe -- with :func:`build_tjunction`'s signature, frame and
    ``TJunction`` return, so the two are interchangeable to a caller.

    :func:`build_tjunction` radiates every seam from a hub, and at a small radius ratio
    its crotch cap degenerates into a pointy wedge deep inside the domain (at ratio 0.26
    it measures minSJ **-0.044 with 2 inverted elements**).  This one has no hub at all,
    so there is nothing to degenerate.  The price is the opposite bound: the branch's
    footprint has to fit inside the *cob*, the square block at the middle of the main
    pipe's O-grid, so the ratio cannot approach 1 -- which is exactly the range
    :func:`build_eqtee` covers.

    The construction, in the main pipe's own cross-section (see
    ``examples/cob_tjunction.py``, which this is the parametrized form of):

    * the **cob** is the middle ``N_THETA_BRANCH`` elements of the section -- a square
      block whose perimeter is exactly ``N_THETA_BRANCH`` edges, which is what lets the
      branch bore attach to it one-for-one;
    * the cob is **walked** wall-to-wall, element to element, leaving each quad by the
      side opposite the one entered.  That band is the branch's shadow through the pipe;
    * the band is removed and the rest of the section extruded in ``z``, leaving a slot;
    * the branch's cross-section is meshed **in the cylinder's own (arc, z) parameter
      space**, so every node of it lands exactly on the wall and the bore is exactly the
      analytic cylinder--cylinder intersection.  That top section is then mapped down the
      band's horizontal cuts and lofted into the slot.

    Because the slot's cuts are rows of the section itself, the junction's ``z`` faces
    come back **bit-identical** to the main pipe's cross-section -- so unlike
    :func:`build_tjunction`, whose legs need a ``N_TRANS`` morph to reach a plain disc,
    ``disc_minus`` / ``disc_plus`` here *are* that plain disc and a caller carries the
    pipe on with an ordinary extrude.

    ``N_THETA_BRANCH`` must be a multiple of 4 and is independent of ``N_THETA_MAIN``.
    ``Z_NEAR`` defaults to ``2 * L``, where ``L = (N_THETA_BRANCH / 4) * (2 pi R_MAIN /
    N_THETA_MAIN)`` is the slot's own square footprint; it must exceed ``L / 2``, since
    the leg is what carries the port face clear of the collar.

    **Radii are what gets meshed, not what gets asked for.**  Nothing here insets for a
    boundary layer: a caller that means to skin passes the *core* radii and grows the
    skin over the finished assembly itself (``examples/cob_tjunction.py`` and
    ``examples/chimera.py`` both do), because the skin has to cover the caller's whole
    wall -- legs, arms and bends included -- not one junction's share of it."""
    RADIAL_BRANCH = (np.array([0.0, 1.0]) if RADIAL_BRANCH is None
                     else np.asarray(RADIAL_BRANCH, dtype=float))
    if N_THETA_BRANCH % 4 != 0:
        raise ValueError("build_cob: N_THETA_BRANCH must be a multiple of 4, got %d"
                         % N_THETA_BRANCH)
    NSIDE = N_THETA_BRANCH // 4               # cells per side of the footprint square
    n_side = N_THETA_MAIN // 4                # cells per side of the O-grid's own core
    # The cob is the centred ``NSIDE x NSIDE`` block of that core, so its perimeter is
    # ``4 * NSIDE == N_THETA_BRANCH`` edges -- which is what lets the bore attach to it
    # one for one.  Centring it needs the two side counts to agree in parity.
    if NSIDE > n_side or (n_side - NSIDE) % 2 != 0:
        raise ValueError(
            "build_cob: the cob is the centred %d x %d block of the O-grid's %d x %d "
            "core, so N_THETA_BRANCH/4 must not exceed N_THETA_MAIN/4 and must match "
            "its parity (got %d, %d)"
            % (NSIDE, NSIDE, n_side, n_side, N_THETA_BRANCH, N_THETA_MAIN))
    # the slot is as long in z as the cob's top arc is wide, so it is square from above
    L = NSIDE * (2.0 * np.pi * R_MAIN / N_THETA_MAIN)
    # The bore is meshed inside that square, and the collar is what is left between the
    # two -- so the bore has to fit with room to spare, not merely fit.  Crowding does
    # not raise on its own: it comes back as a *folded collar*, and at 0.98 of the
    # half-width it measures 48 inverted elements.  The reference caller
    # (``examples/cob_tjunction.py``) sits at 0.57, and 0.85 is where this stops being
    # a warning and starts being a build that cannot come out clean.
    if R_BRANCH >= _COB_FILL_MAX * L / 2.0:
        raise ValueError(
            "build_cob: the bore (R_BRANCH = %g) fills %.2f of the cob's own square "
            "(half-width %g), past the %.2f the collar can absorb -- it would come back "
            "folded, not merely tight.  Raise N_THETA_BRANCH or lower N_THETA_MAIN: the "
            "square is (N_THETA_BRANCH/4) * (2 pi R_MAIN / N_THETA_MAIN) across."
            % (R_BRANCH, R_BRANCH / (L / 2.0), L / 2.0, _COB_FILL_MAX))
    Z_NEAR = 2.0 * L if Z_NEAR is None else float(Z_NEAR)
    if Z_NEAR <= L / 2.0:
        raise ValueError("build_cob: Z_NEAR must exceed L/2 = %g (the slot's own "
                         "half-length), got %g" % (L / 2.0, Z_NEAR))
    N_Z_LEG = (max(1, round((Z_NEAR - L / 2.0) / (L / NSIDE))) if N_Z_LEG is None
               else int(N_Z_LEG))
    # native frame here is (main = Z, branch = +Y); the return rotates it into
    # build_tjunction's own (branch = +X)
    BRANCH_AXIS = np.array([0.0, 1.0, 0.0])
    REGION = element_tag or None

    # -- the main pipe cross-section, and the cob's band through it ------------
    section = quadmesh.ogrid(linemesh.circle(R_MAIN, N_THETA_MAIN, order=order),
                             n_side, RADIAL_MAIN,
                             center_scale=CENTER_SCALE_MAIN, wall_tag=wall_tag)
    quads, pts, lines = section.quads, section.points, section.line_mesh.lines
    P = pts[section.corners]

    edge2q = defaultdict(list)
    for q in range(section.n_quads):
        for s in range(4):
            edge2q[int(quads[q, s])].append((q, s))

    def walk(e, from_q):
        """Straight run of elements: enter through edge ``e``, leave by the opposite
        side, repeat until the wall.  Returns ``(elements, the edges crossed)``."""
        el, ed = [], [e]
        while True:
            nxt = [(q, s) for (q, s) in edge2q[e] if q != from_q]
            if not nxt:
                return el, ed
            q, s = nxt[0]
            el.append(q)
            e, from_q = int(quads[q, (s + 2) % 4]), q
            ed.append(e)

    cob = np.argsort(np.linalg.norm(P.mean(axis=1), axis=1))[:NSIDE ** 2]
    own = {}
    for q in cob:
        for s in range(4):
            e = int(quads[q, s])
            own[e] = q if e not in own else -1
    mid = P[cob].mean(axis=(0, 1))
    rim = sorted((float((pts[lines[e]].mean(axis=0) - mid) @ BRANCH_AXIS), e, q)
                 for e, q in own.items() if q != -1)

    band = list(cob)
    for _, e, q in rim[-NSIDE:] + rim[:NSIDE]:
        band += walk(e, q)[0]
    band = sorted(set(band))
    band_set = set(band)

    # The slot's own boundary, named before the band is removed: an edge with one quad
    # in the band and one outside is where the collar will meet the pipe.
    # ``quadmesh.remove`` leaves the faces it exposes untagged, so naming them here is
    # the only way the lateral seam is addressable at all -- the tag rides the surviving
    # quad's edge through.
    #
    # It goes on a *copy*, not on ``section``: the legs extrude the same section and keep
    # the band, so there the very same edges are interior.  Tagging them in place would
    # name faces that no seam ever consumes, and the exporter would write a boundary row
    # for each -- boundary conditions in the middle of the pipe.
    _slot_rows = np.array([[q, s + 1] for e, lst in edge2q.items() if len(lst) == 2
                           for (q, s) in lst
                           if (lst[0][0] in band_set) != (lst[1][0] in band_set)
                           and q not in band_set], dtype=np.int64)
    slot_section = quadmesh.tag_edges(section, _slot_rows, "att_slot")

    # the band's four columns, each walked from the wall it starts on right through
    wall_edges = [e for e, lst in edge2q.items() if len(lst) == 1
                  and lst[0][0] in band_set]
    foot = sorted(wall_edges, key=lambda e: float(pts[lines[e]].mean(axis=0)
                                                  @ BRANCH_AXIS))
    cols = [walk(e, -1) for e in foot[:NSIDE]]
    nrow = len(cols[0][0])

    def chain_edges(ids):
        """One horizontal cut as ``[(edge id, walked backwards), ...]``, end to end."""
        segs = [(e, int(lines[e, 0]), int(lines[e, 1])) for e in ids]
        cnt = defaultdict(int)
        for _, a, b in segs:
            cnt[a] += 1
            cnt[b] += 1
        cur = [p for p, c in cnt.items() if c == 1][0]
        out, left = [], list(segs)
        while left:
            for seg in left:
                e, a, b = seg
                if a == cur:
                    out.append((e, False))
                    cur = b
                elif b == cur:
                    out.append((e, True))
                    cur = a
                else:
                    continue
                left.remove(seg)
                break
        return out

    # The band as one node lattice, ``(nrow*order+1, NSIDE*order+1, 3)``.
    #
    # Cuts have to be available *between* element boundaries, not only on them.
    # ``loft`` straight-subdivides along its sweep, so at order > 1 handing it only the
    # boundary cuts drops every mid-node onto a chord of the band's own curved rows --
    # corner-clean, and inverted the moment the curved block is read.  Reading the whole
    # lattice once makes a cut at any GLL level just a row of it, which is what
    # ``sweep_nodes`` wants.
    BLK = element_blocks(section).reshape(section.n_quads, order + 1, order + 1, 3)

    def band_block(q, e_in):
        """Element ``q``'s nodes as ``[v, a]`` -- ``v`` up the band away from edge
        ``e_in``, ``a`` across it."""
        s = int(np.flatnonzero(quads[q] == e_in)[0])
        b = BLK[q]
        if s == 0:                                     # side 1 is j=0
            return b
        if s == 2:                                     # side 3 is j=n
            return b[::-1]
        if s == 3:                                     # side 4 is i=0
            return b.transpose(1, 0, 2)
        return b.transpose(1, 0, 2)[::-1]              # side 2 is i=n

    LAT = np.empty((nrow * order + 1, NSIDE * order + 1, 3))
    below = None
    for k in range(nrow):
        row = np.empty((order + 1, NSIDE * order + 1, 3))
        where = {cols[c][1][k]: c for c in range(NSIDE)}
        for slot, (e, rev) in enumerate(
                chain_edges([cols[c][1][k] for c in range(NSIDE)])):
            B = band_block(cols[where[e]][0][k], e)
            # ``chain_edges`` says which way the cut walks this edge; the block's own
            # ``a`` need not agree, and a tail-matching heuristic cannot orient the
            # *first* one.
            head = pts[lines[e, 1]] if rev else pts[lines[e, 0]]
            if not np.allclose(B[0, 0], head, atol=1e-12):
                B = B[:, ::-1, :]
            row[:, slot * order:(slot + 1) * order + 1] = B
        if below is not None and not np.allclose(row[0], below, atol=1e-12):
            row = row[:, ::-1, :]                      # a whole row can walk the other way
        LAT[k * order:(k + 1) * order + 1] = row
        below = row[-1]

    # run the lattice the way the top section does (increasing arc == decreasing x)
    if LAT[0, -1, 0] > LAT[0, 0, 0]:
        LAT = LAT[:, ::-1, :]

    def cut_at(level):
        """``(NSIDE, order+1, 3)`` node blocks of the cut at sweep ``level``."""
        row = LAT[level]
        return np.stack([row[i * order:(i + 1) * order + 1] for i in range(NSIDE)])

    GLL = gll_nodes(order)

    def on_cut(blk, a):
        """Evaluate a cut's piecewise Lagrange curve at ``a`` in ``[-1, 1]``.

        The section's own nodes sit at the GLL positions of a matching element run, so
        this lands on the cut's stored nodes exactly rather than interpolating near
        them."""
        t = (a + 1.0) / 2.0 * NSIDE
        i = np.clip(t.astype(int), 0, NSIDE - 1)
        M = lagrange_matrix(GLL, t - i)              # (P, order+1)
        return np.einsum("pk,pkc->pc", M, blk[i])

    # -- the branch cross-section, meshed on the wall itself -------------------
    def foot_param(t):
        """The exact cylinder--cylinder intersection, in ``(arc s, z)`` parameter
        coords."""
        x = R_BRANCH * np.sin(t)
        z = R_BRANCH * np.cos(t)
        y = np.sqrt(R_MAIN ** 2 - x ** 2)
        return np.stack([R_MAIN * (np.arctan2(y, x) - np.pi / 2),
                         np.zeros_like(t), z], axis=1)

    def to_cyl(p):
        """``(arc s, z)`` -> the cylinder.  ``s`` is arc length, so the parameter domain
        is a true ``L x L`` square and the O-grid built in it is not distorted by the
        wrap."""
        phi = np.pi / 2 + p[:, 0] / R_MAIN
        return np.stack([R_MAIN * np.cos(phi), R_MAIN * np.sin(phi), p[:, 2]], axis=1)

    # the collar's lateral surface is this loop swept: two of its sides face the pipe
    # across the slot, the other two are the collar's own ends where the legs butt
    # against it
    square = linemesh.rectangle(L, L, N_THETA_BRANCH, normal=BRANCH_AXIS, order=order,
                                side_tags={"left": "att_slot", "right": "att_slot",
                                           "bottom": "att_endA", "top": "att_endB"})
    # Pair the footprint with the square *angularly* -- one bore node per square node on
    # the same ray from the centre.  Spacing the bore by arc length instead leaves the
    # two loops out of phase and the annulus comes back folded.
    td = np.linspace(0.0, 2.0 * np.pi, 4001)
    fang = np.unwrap(np.arctan2(foot_param(td)[:, 2], foot_param(td)[:, 0]))
    aim = np.mod(np.arctan2(square.points[:, 2], square.points[:, 0]) - fang[0],
                 2.0 * np.pi) + fang[0]
    # ``loft`` would straight-subdivide between the corners and drop every high-order
    # node onto a chord; ``loft_fn`` evaluates the intersection itself at the whole node
    # lattice.
    t_bore = np.unwrap(np.interp(aim, fang, td), period=2.0 * np.pi)
    # A closed ``loft_fn`` wants the trailing wrap value too, so the seam element's own
    # nodes get evaluated rather than closed with a chord -- but the run has to be
    # *monotonic* first, and matching the square angularly hands it back descending.
    # Wrapping the wrong way makes the seam element span nearly the whole bore, and
    # ``loft_fn`` dutifully places its interior node there.  Order 1 never sees it: only
    # corners are placed, and those are right either way.
    wrap = 2.0 * np.pi if t_bore[-1] > t_bore[0] else -2.0 * np.pi
    bore_loop = linemesh.loft_fn(foot_param, np.append(t_bore, t_bore[0] + wrap),
                                 loop=True, order=order)

    # the bore disc's wall *is* the collar's inner loop -- the one seam between them, so
    # name it on both sides and state the join
    bore_p = quadmesh.spined_ogrid(bore_loop, RADIAL_BRANCH,
                                   center_scale=CENTER_SCALE_BRANCH,
                                   wall_tag="attach1")
    # ``annulus`` winds the opposite way round from ``spined_ogrid`` on the same loop, so
    # reverse both of its loops to make the joined section consistently wound.
    collar_p = quadmesh.annulus(linemesh.reverse(bore_loop), linemesh.reverse(square),
                                RADIAL_BRANCH, inner_tag="attach1")
    top_p = quadmesh.attach([bore_p, collar_p], [EdgeSeam(0, "attach1", 1, "attach1")])
    TOP = _map_section(top_p, to_cyl)

    def to_cut(blk):
        def fn(p):
            out = on_cut(blk, p[:, 0] / (L / 2.0))
            out[:, 2] = p[:, 2]
            return out
        return fn

    slices = [_map_section(top_p, to_cut(cut_at(k * order))) for k in range(nrow)]
    slices.append(TOP)                          # the top one sits exactly on the wall
    # one intermediate profile per interior GLL level of every layer, so the sweep
    # follows the band's rows instead of chording across them
    inner = [[_map_section(top_p, to_cut(cut_at(k * order + m))) for m in range(1, order)]
             for k in range(nrow)]
    # TOP is the collar's far cap: its bore half is the branch's root, its rim half is
    # the pipe wall the bore was cut out of
    _top_tags = ElementTags.from_dense(
        np.array(["att_bore"] * bore_p.n_quads + [wall_tag] * collar_p.n_quads))
    # ``first_tag`` is the collar's *far* cap -- the band runs wall to wall, so the slot's
    # bottom is the pipe wall opposite the branch and the bore is blind.  It has to be
    # stated: the default is the bounding slice's own ``element_tags``, and ``top_p``
    # carries none, so the far wall would come back untagged and the skin would grow
    # round a hole in itself.
    collar = hexmesh.loft(slices, sweep_nodes=inner if order > 1 else None,
                          first_tag=wall_tag, last_tag=_top_tags, element_tags=REGION)

    # -- the pipe around the slot, and the two legs out to +-Z_NEAR ------------
    # a leg's cap meets *two* blocks: the pipe over the section, the collar over the band
    # it removed.  One name for the whole cap will not do, and ``first_tag`` / ``last_tag``
    # take an ``ElementTags`` over the slice's own elements for exactly that.
    _leg_cap = ElementTags.from_dense(
        np.where(np.isin(np.arange(section.n_quads), band), "att_band", "att_pipe"))
    _pt_minus, _pt_plus = port_tags if port_tags else ("", "")

    mid_pipe = hexmesh.extrude(quadmesh.remove(slot_section, band), length=L, layers=NSIDE,
                               axis=(0.0, 0.0, 1.0), origin=(0.0, 0.0, -L / 2),
                               element_tags=REGION,
                               first_tag="att_pipe_lo", last_tag="att_pipe_hi")
    run = Z_NEAR - L / 2.0
    leg_plus = hexmesh.extrude(section, length=run, layers=N_Z_LEG, axis=(0.0, 0.0, 1.0),
                               origin=(0.0, 0.0, L / 2), element_tags=REGION,
                               first_tag=_leg_cap, last_tag=_pt_plus)
    leg_minus = hexmesh.extrude(section, length=run, layers=N_Z_LEG, axis=(0.0, 0.0, 1.0),
                                origin=(0.0, 0.0, -Z_NEAR), element_tags=REGION,
                                first_tag=_pt_minus, last_tag=_leg_cap)

    # -- the branch: the bore disc off the wall, straight out to the opening ---
    # the disc's own (x, z) are already the bore circle, so holding them and carrying y
    # up to H_BRANCH sweeps an exact cylinder with a curved root and a flat cap.
    # ``bore_p``'s wall was named for the section join above; the branch sweeps that same
    # rim into its *outer* faces, which become the branch's own wall -- a seam name that
    # outlives its seam is a name for something that no longer exists.
    bore_wall = _map_section(quadmesh.retag_edge(bore_p, {"attach1": wall_tag}), to_cyl)
    flat = _map_section(bore_wall, lambda p: np.stack(
        [p[:, 0], np.full(p.shape[0], H_BRANCH), p[:, 2]], axis=1))
    stations = quadmesh.blend(bore_wall, flat, np.linspace(0.0, 1.0, N_BRANCH + 1))
    branch = hexmesh.loft(stations, first_tag="att_bore", last_tag=branch_tag,
                          element_tags=REGION)

    # six seams, every one named on both sides: the collar against the slot it fills, its
    # two ends against the legs, its bore cap against the branch, and the pipe against
    # each leg.  Lower block index first in every seam.
    core = hexmesh.attach(
        [collar, mid_pipe, leg_plus, leg_minus, branch],
        [Seam(0, "att_slot", 1, "att_slot"),
         Seam(0, "att_endA", 2, "att_band"),
         Seam(0, "att_endB", 3, "att_band"),
         Seam(0, "att_bore", 4, "att_bore"),
         Seam(1, "att_pipe_hi", 2, "att_pipe"),
         Seam(1, "att_pipe_lo", 3, "att_pipe")])

    # the two ports *are* the plain section, bit for bit -- that is the whole point of
    # the cob construction, and what lets a caller carry the pipe on with a plain extrude
    disc_plus = quadmesh.translate(section, (0.0, 0.0, Z_NEAR))
    disc_minus = quadmesh.translate(section, (0.0, 0.0, -Z_NEAR))
    disc_branch = flat

    # rotate the native (branch = +Y) frame into build_tjunction's own (branch = +X):
    # -90 degrees about +Z, a proper rotation (det +1), so no element is inverted.
    ang, axis = -np.pi / 2.0, (0.0, 0.0, 1.0)
    return TJunction(hexmesh.rotate(core, ang, axis=axis),
                     quadmesh.rotate(disc_minus, ang, axis=axis),
                     quadmesh.rotate(disc_plus, ang, axis=axis),
                     quadmesh.rotate(disc_branch, ang, axis=axis))


def skin_wall(core, offsets, *, wall_tags="wall", element_tag="", inner_prefix="_skin_"):
    """``core`` with a boundary-layer shell grown outward over its ``wall_tags`` faces.

    The offsets are **cumulative distances from the wall**, starting at ``0.0`` -- so
    ``[0.0, 0.02, 0.05]`` is two layers, the finished surface sitting ``0.05`` outside the
    wall the core was meshed at.  A caller that wants the finished wall on a nominal
    radius therefore meshes the core *inset* by ``offsets[-1]``: an offset is a uniform
    thickness along the surface normal, not a scaling, so the inset is the same number on
    every radius in the model, and every radius comes out right at once.

    **Which faces get skinned is a tagging decision, and it is the caller's.**  Only the
    named groups are offset; an opening -- an inlet, an outlet, a dead-end cap -- must
    carry some other name or it is skinned too, its cap pushed out bodily and its rim left
    to argue with the wall's.  Several names may be skinned together (a wall split into a
    plain part and a named conjugate interface, say): each group's own name comes back on
    the shell's *outer* cap, so a seam a caller means to bond to afterwards survives the
    skinning instead of being flattened into one group.

    The rims are not left bare either: an edge where a skinned face meets an unskinned one
    is tagged here with **that** face's name, so the shell's lateral faces come out
    carrying the opening's own boundary condition rather than untagged.  That is why this
    starts from the *untagged* :func:`hexmesh.boundary_mesh
    <nekmeshpy.hexmesh.lower.boundary_mesh>`, which names each extracted quad after the
    parent face it came from, instead of asking for one group directly.

    The join is ``attach``, not ``merge``: the interface is the core's own wall groups
    against the shell's inner cap, so it is stated rather than rediscovered from
    coordinates -- and attaching clears the buried faces, which a *named* interior face
    otherwise is not (the exporter writes one boundary row per hex carrying one)."""
    offsets = np.asarray(offsets, dtype=float).reshape(-1)
    if offsets.size < 2 or offsets[0] != 0.0:
        raise ValueError("skin_wall: offsets are cumulative distances from the wall and "
                         "must start at 0.0 with at least one layer after it, got %s"
                         % (offsets.tolist(),))
    tags = [wall_tags] if isinstance(wall_tags, str) else list(wall_tags)

    surf = hexmesh.boundary_mesh(core)          # every quad named after its parent face
    names = np.asarray(surf.element_tags.dense(surf.n_quads))
    skinned = np.isin(names, tags)
    if not skinned.any():
        raise ValueError("skin_wall: no boundary face carries any of %s; this core has %s"
                         % (tags, sorted(set(names.tolist())) or "nothing tagged"))

    # A rim edge has a skinned face on one side and an unskinned one on the other; scatter
    # both sides into per-edge tables (last write wins, a free choice among equals) and
    # name the rim after the face that is *not* being skinned.
    eids = np.asarray(surf.quads, dtype=np.int64)                  # (Q,4) edge ids
    n_edges = surf.line_mesh.n_lines
    flat_e = eids.ravel()
    flat_q = np.repeat(np.arange(surf.n_quads), 4)
    flat_s = np.tile(np.arange(4), surf.n_quads)
    hit = skinned[flat_q]
    n_skin = np.bincount(flat_e, weights=hit.astype(float), minlength=n_edges)
    n_all = np.bincount(flat_e, minlength=n_edges)
    other = np.full(n_edges, "", dtype=names.dtype)
    other[flat_e[~hit]] = names[flat_q[~hit]]
    w_q = np.full(n_edges, -1, dtype=np.int64)
    w_s = np.full(n_edges, -1, dtype=np.int64)
    w_q[flat_e[hit]] = flat_q[hit]
    w_s[flat_e[hit]] = flat_s[hit]
    rim = np.flatnonzero((n_skin > 0) & (n_all - n_skin > 0) & (other != ""))
    surf = quadmesh.tag_edges(surf, np.stack([w_q[rim], w_s[rim] + 1], axis=1),
                              other[rim])

    wall = quadmesh.select(surf, skinned)
    skins = [wall] + [quadmesh.offset(wall, float(d)) for d in offsets[1:]]
    # ``first_tag`` must be stated: it defaults to the slice's own ``element_tags``, which
    # is the wall group's own name -- and the shell's two caps would then be two copies of
    # it, which is not an interface.  Prefixing keeps the correspondence one group to one
    # group, so the seam below is stated per group and not by proximity.  ``last_tag`` is
    # left to that default, which is exactly right: the shell's outer cap *is* the
    # finished wall, and carries each group's own name onward.
    inner = ElementTags.from_dense(
        np.char.add(inner_prefix, np.asarray(wall.element_tags.dense(wall.n_quads))))
    shell = hexmesh.loft(skins, first_tag=inner, element_tags=element_tag or None)
    return hexmesh.attach([core, shell],
                          [Seam(0, t, 1, inner_prefix + t) for t in tags])
