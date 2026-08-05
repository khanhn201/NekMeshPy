"""Standalone, parametrized quadrant T-junction builder -- adapted from
examples/quadrant_pipe_tjunction.py, generalized into a function so it can be
called multiple times at different radii/positions.  Local frame: main pipe axis
= Z (legs at z = -Z_NEAR "minus" and z = +Z_NEAR "plus"), branch axis = X
(opening at x = H_BRANCH), exactly quadrant_pipe_tjunction.py's own convention.

Returns a TJunction(core, disc_minus, disc_plus, disc_branch): ``core`` is the
merged crotch/transition hex block (untagged), and the three discs are the
*plain* QuadMesh cross-sections at each leg's outward end -- for a caller to
continue building from (extrude/loft/sweep) rather than re-deriving the
junction's own geometry.
"""
from __future__ import annotations

from collections import namedtuple

import numpy as np

from nekmeshpy import hexmesh, linemesh, quadmesh
from nekmeshpy.model import surfaces

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
                     RADIAL=None, CENTER_SCALE=0.7,
                     order=2, PHI_W=None, CAP_TIP_BIAS=None,
                     N_TRANS=5, N_BRANCH=4, ORIGIN=None):
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

    def quadrant(arc, seam1, seam2, wall_tag=""):
        return quadmesh.quadrant_ogrid(arc, seam1, seam2, RADIAL,
                                       center_scale=CENTER_SCALE, wall_tag=wall_tag)

    def disc(pieces):
        return quadmesh.merge([quadrant(arc, s1, s2, wall_tag="wall")
                               for arc, s1, s2 in pieces])

    def plain_walls(composite, z, sign):
        ang = sign * np.deg2rad(-45.0 + 90.0 * np.arange(5))
        return [plain_wall(composite[q], ang[q], ang[q + 1], z) for q in range(4)]

    FR = quadmesh.quadrant_seam_fractions(N_QUAD, RADIAL, CENTER_SCALE)

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
                             center=ORIGIN + CENTER_SCALE * np.sqrt(1.5) * (wc - ORIGIN))

    def leg(composite, walls, sign):
        z = sign * Z_NEAR
        w_plain = plain_walls(walls, z, sign)

        def station(s):
            return quadmesh.quadrant_disc(
                [wall_mesh(surfaces.blend(walls[q], w_plain[q], s)) for q in range(4)],
                (1.0 - s) * ORIGIN + s * np.array([0.0, 0.0, z]),
                RADIAL, center_scale=CENTER_SCALE,
                wall_tag="wall")

        plain = station(1.0)
        transition = hexmesh.loft_fn(station, np.linspace(0.0, 1.0, N_TRANS + 1),
                                     order=order)
        return transition, plain

    def branch():
        open_arcs = [linemesh.loft_fn(opening, fr, order=order) for fr in FQ_FR]
        t = np.linspace(0.0, 1.0, N_BRANCH + 1)
        walls = [linemesh.blend(f, o, t) for f, o in zip(FQ, open_arcs)]
        c_open = np.array([H_BRANCH, 0.0, 0.0])
        sections = [quadmesh.quadrant_disc([w[i] for w in walls],
                                           (1.0 - t[i]) * ORIGIN + t[i] * c_open, RADIAL,
                                           center_scale=CENTER_SCALE, wall_tag="wall")
                   for i in range(t.size)]
        return hexmesh.loft(sections), sections[-1]

    trans_plus, disc_plus = leg(COMPOSITE_R, W_R, 1)
    trans_minus, disc_minus = leg(COMPOSITE_L, W_L, -1)
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
