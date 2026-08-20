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

from collections import namedtuple

import numpy as np

from nekmeshpy import ElementTags, hexmesh, linemesh, quadmesh
from nekmeshpy.core import surfaces
from nekmeshpy.hexmesh import Seam
from nekmeshpy.linemesh import Seam as PointSeam
from nekmeshpy.quadmesh import Seam as EdgeSeam

TJunction = namedtuple("TJunction", "core disc_minus disc_plus disc_branch")

#: default radial station fractions (a module-level singleton so it is not rebuilt --
#: or flagged -- as an argument default)
_RADIAL_DEFAULT = np.array([0.0, 0.6, 1.0])

#: Bounds on the bypass half-angle.  Below the floor the bypass quadrant spans so much
#: of the pipe that it folds; above the ceiling the two side quadrants do.  Both were
#: measured, not derived -- see ``auto_params``.
_PHI_W_FLOOR, _PHI_W_CEIL = np.deg2rad(60.0), np.deg2rad(170.0)



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

    def quadrant(arc, seam1, seam2, wall_tag="", side_tags=None):
        return quadmesh.quadrant_ogrid(arc, seam1, seam2, RADIAL,
                                       center_scale=CENTER_SCALE, wall_tag=wall_tag,
                                       side_tags=side_tags)

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

    def disc(pieces):
        n = len(pieces)
        return ring([quadrant(arc, s1, s2, wall_tag="wall", side_tags=ring_sides(q, n))
                     for q, (arc, s1, s2) in enumerate(pieces)])

    def disc_at(arcs, center):
        # quadrant_disc's own recipe, inlined: quadmesh.ogrid/spined_ogrid now cover the
        # single-boundary-loop case, but this junction's per-station arcs are built
        # independently per quadrant with an off-centroid hub, so they still need the
        # general form.
        fr = quadmesh.quadrant_seam_fractions(N_QUAD, RADIAL, QUADRANT_SCALE)
        seams = [seam(a.points[0], fr, center) for a in arcs]
        seams.append(seams[0])
        return ring([quadrant(arcs[q], seams[q], seams[q + 1], wall_tag="wall",
                              side_tags=ring_sides(q, 4)) for q in range(4)])

    def plain_walls(composite, z, sign):
        ang = sign * np.deg2rad(-45.0 + 90.0 * np.arange(5))
        return [plain_wall(composite[q], ang[q], ang[q + 1], z) for q in range(4)]

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
    BYPASS = wall_mesh(W_R[2])

    COMPOSITE_R = disc([(linemesh.reverse(FQ[0]), SP[1], SP[0]),
                        (SIDE_RP, SP[0], SWP),
                        (BYPASS, SWP, SWM),
                        (SIDE_RM, SWM, SP[1])])
    COMPOSITE_L = disc([(linemesh.reverse(FQ[2]), SP[3], SP[2]),
                        (SIDE_LM, SP[2], SWM),
                        (linemesh.reverse(BYPASS), SWM, SWP),
                        (SIDE_LP, SWP, SP[3])])

    def arc_mids(walls):
        return [surfaces.node(w, N) for w in walls]

    def cap(sa, sb, sc, ab, bc, ca, tip_bias=CAP_TIP_BIAS):
        (m_ab, w_ab), (m_bc, w_bc), (m_ca, w_ca) = ab, bc, ca
        mids = arc_mids((w_ab, w_bc, w_ca))
        wc_param = quadmesh.tri_patch_tip(*mids, tip_bias=tip_bias)
        wc = cyl_pts(wc_param[None, :])[0]
        return hexmesh.tetra([quadrant(m_ab, sa, sb), quadrant(m_bc, sb, sc),
                              quadrant(m_ca, sc, sa),
                              quadmesh.tri_patch(cyl_pts, w_ab, w_bc, w_ca,
                                                 order=order, tip_bias=tip_bias,
                                                 mids=mids, element_tag="wall")],
                             center=ORIGIN + CENTER_SCALE* np.sqrt(1.5) * (wc - ORIGIN),
                             element_tag=element_tag)

    def leg(composite, walls, sign, port_tag):
        z = sign * Z_NEAR
        w_plain = plain_walls(walls, z, sign)

        def station(s):
            return disc_at(
                [wall_mesh(surfaces.blend(walls[q], w_plain[q], s)) for q in range(4)],
                (1.0 - s) * ORIGIN + s * np.array([0.0, 0.0, z]))

        plain = station(1.0)
        transition = hexmesh.loft_fn(station, np.linspace(0.0, 1.0, N_TRANS + 1),
                                     order=order, element_tags=element_tag or None,
                                     last_tag=port_tag or None)
        return transition, plain

    def branch():
        open_arcs = [linemesh.loft_fn(opening, fr, order=order) for fr in FQ_FR]
        t = np.linspace(0.0, 1.0, N_BRANCH + 1)
        walls = [linemesh.blend(f, o, t) for f, o in zip(FQ, open_arcs)]
        c_open = np.array([H_BRANCH, 0.0, 0.0])
        sections = [disc_at([w[i] for w in walls],
                            (1.0 - t[i]) * ORIGIN + t[i] * c_open)
                   for i in range(t.size)]
        return (hexmesh.loft(sections, element_tags=element_tag or None,
                             last_tag=branch_tag or None),
                sections[-1])

    # ``port_tags`` names the two leg openings on the core so a caller can
    # ``hexmesh.attach`` its own legs to them by name rather than have ``merge``
    # rediscover the seam from coordinates.  Off by default: a *named* face that a
    # later ``merge`` buries is not inert -- the exporter writes one boundary row per
    # hex carrying one -- so only a caller that means to attach should ask for them.
    _pt_minus, _pt_plus = port_tags if port_tags else ("", "")
    trans_plus, disc_plus = leg(COMPOSITE_R, W_R, 1, _pt_plus)
    trans_minus, disc_minus = leg(COMPOSITE_L, W_L, -1, _pt_minus)
    trans_branch, disc_branch = branch()

    blocks = [trans_plus, trans_minus, trans_branch,
              cap(SP[0], SP[3], SWP,
                  (linemesh.reverse(FQ[3]), foot_wall(FQ_FR[3][::-1])),
                  (linemesh.reverse(SIDE_LP), shift_wall(surfaces.reverse(W_L[3]), 1)),
                  (linemesh.reverse(SIDE_RP), surfaces.reverse(W_R[1]))),
              cap(SP[2], SP[1], SWM,
                  (linemesh.reverse(FQ[1]), foot_wall(FQ_FR[1][::-1])),
                  (linemesh.reverse(SIDE_RM), shift_wall(surfaces.reverse(W_R[3]), -1)),
                  (linemesh.reverse(SIDE_LM), surfaces.reverse(W_L[1])))]
    core = hexmesh.merge(blocks)
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
