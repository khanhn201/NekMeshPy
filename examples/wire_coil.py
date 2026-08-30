"""A helically coiled wire inside a round pipe, meshed for conjugate heat transfer.

The domain has three regions.  The **wire** is a continuous helix: its round
cross-section is an o-grid ``_disc`` (an inward-facing quadrant given a wide arc,
``INNER_ARC_DEG``), swept one turn per pitch by ``cross_map`` so consecutive turns
abut and weld (``build_turn``).  The **solid pipe wall** is an annular shell,
``R_TUBE .. R_TUBE + WALL_THICK``.  Everything else -- the film between the wire
and the wall, the wedge between consecutive turns, and the core down the axis --
is **fluid**.

The fluid between the turns is the interesting part.  An ``inner_sheet`` -- a
staircase quad band on the cylinder ``r = R_INNER`` -- is the surface the inter-turn
fill lofts against; a ``staircase`` line traces the template's top edge onto it and
a ``spiral`` line traces the same edge back onto the wire's inward wall.  ``between``
lofts staircase -> spiral and out through ``OG_NSIDE`` more radial layers to the
wire's top / bottom quadrant faces.  ``gap_hi`` / ``gap_lo`` fill the layer-0 wedge
from the collar rows; ``topq_gap`` / ``botq_gap`` fill the outer layers.  ``wall_hex``
lofts the coil's outward wrap out to ``R_TUBE`` and ``wall_solid`` continues it
through the pipe wall.  ``core_hex`` o-grids the cylinder inside the inner sheet.

Every block is welded with ``merge`` where its faces coincide node-for-node and
with ``attach`` where only the corners do (the straight-subdivided guide lines sit
~1e-3 off the curved template edges).  The final ``mesh`` is tagged: ``interface``
for every fluid/solid face (conjugate -- a wall on the fluid side, nothing on the
solid), ``inlet`` / ``outlet`` at the fluid ends, ``outer`` on the pipe skin, and
``wall`` / ``cut`` on the remaining exposed fluid / solid faces.

    PYTHONPATH=. python examples/wire_coil.py

Produces ``wire_coil.re2`` and ``wire_coil.vtu``.
"""
from collections import defaultdict

import numpy as np
from scipy.spatial import cKDTree as _KD

from nekmeshpy import hexmesh, linemesh, quadmesh, writer
from nekmeshpy.core.tags import ElementTags
from nekmeshpy.hexmesh.assemble import boundary_face_ids as _bfid
from nekmeshpy.linemesh import LineMesh
from nekmeshpy.pointmesh import PointMesh
from nekmeshpy.quadmesh import QuadMesh
from nekmeshpy.quadmesh.query import element_blocks

R_HELIX = 0.375
R_TUBE  = 0.5                  # tube inner radius (fluid <-> solid wall boundary)
WALL_THICK = 0.10             # radial thickness of the solid outer wall
N_WALL     = 3               # hex layers across the solid wall
RW_NOM  = 0.125                # nominal wire radius, e/2 = D/8
CLEAR   = 0.004                # fluid film left between wire and tube wall
RW      = RW_NOM - CLEAR       # meshed wire radius -> outer reach 0.496

PITCH     = (1.0 / 6.0) / 0.375      # axial rise per turn
SHEET_GAP = 0.10                     # radial clearance wire surface -> inner sheet

LAYERS_WIRE, TURNS_WIRE = 30, 2      # blend layers over the coil's turns
NU = LAYERS_WIRE // TURNS_WIRE       # horizontal blend elements per turn (may be ODD:
                                    # the two strips need not have equal counts)
NV = 2                              # vertical elements / diamond resolution
SHEET_TURNS = 2                     # pitches of coil to build
INNER_ARC_DEG = 120.0             # tube o-grid: angular span of the inward-facing
                                   # quadrant (90 = square core; -> 180 max)
INNER_DZ =  0.0             # extra z shift of the inner sheet off centre
INNER_PHASE_DEG = 10.0             # extra rotation of the inner sheet about z
SPIRAL_DZ_FRAC = 0.18              # spiral z lift above the staircase, in pitches
                                   # (the spiral is the staircase's own shape,
                                   # scaled out to the inward-wall radius)
SPIRAL_ROT_FRAC = 0.75            # extra CW rotation of the spiral, in units of one
                                   # element's angular pitch (2pi / elems-per-turn)
GRADE = 0.87                       # strip column grading: >1 stretches the columns
                                   # nearest the diamond wide and compresses them
                                   # toward the outer sides (the template's top and
                                   # bottom edges then go non-uniform -- fine)
ORDER = 2
IFACE = "interface"                # the conjugate fluid/solid surface
CUT = "cut"                        # the saw cuts at the two ends of the helix
OUTER = "outer"                    # the pipe's outer skin
_RIM_LO, _RIM_HI = "_rim_lo", "_rim_hi"     # wall_wrap's two boundary loops
_END_LO, _END_HI = "_end_lo", "_end_hi"   # the template's two vertical sides, kept
                                   # apart only so a turn can drop the one that is
                                   # an interior seam rather than a domain end
SKIN = "skin"                      # the OUTWARD half of it -- a separate name only
                                   # long enough to pick the wall film off it, then
                                   # renamed to IFACE like the rest of the coil wall
assert NV % 2 == 0, (NU, NV)

nd = NV // 2
nhR = NU // 2                         # right-strip columns
nhL = NU - nhR                        # left-strip columns  (nhL >= nhR)

# --- key points in (u, v): u in [0,1] -> theta;  v in [-0.5,1] -> z offset -----
# D1 sits one diamond NW of D2; they share D1's lower-right edge (B1->R1), which
# is also D2's top-left edge (T2->L2).  hh = diamond half-height (v), hw = half-
# width (u).  Strips span v in [0,2hh]; D2's bottom apex B2 hangs at v = -hh.
# The two strips carry nhL / nhR columns and the diamond one more (width 2*hw).
# Pinning every column and the diamond to the SAME width h and packing them into
# u in [0,1] gives h = 1/(NU+2), hw = h, and the diamond centre shifted to
# _DC = (nhL+1)*h so the left/right column widths come out equal despite nhL != nhR.
hh = 0.5
_h = 1.0 / (NU + 2)
hw = _h
_DC = (nhL + 1) * _h                  # diamond centre u (0.5 only when nhL == nhR)
B1 = (_DC,          0.0,        0.0)   # D1 bottom vertex  == D2 left vertex L2
T1 = (_DC,          2.0 * hh,   0.0)   # D1 top apex
L1 = (_DC - hw,     hh,         0.0)
R1 = (_DC + hw,     hh,         0.0)   # D1 right vertex
Lg = (_DC - 2 * hw, 2.0 * hh,   0.0)   # ghost top: L1 + (removed D2's edge vector)
# the sheet's left/right edges run PARALLEL to the diamond edges (slope 2hh:-2hw
# over the full height), not vertical -- so the wrap seam segments come out
# uniform.  This shears the strips into parallelograms; u then runs [-hw, 1+hw]
# and the seam u=-hw*v-line == (u+1)-line closes cleanly.
TL, TR = (-hw, 2.0 * hh, 0.0), (1.0 - hw, 2.0 * hh, 0.0)
ML, MR = (0.0, hh, 0.0),       (1.0,      hh,       0.0)   # sheared edge at v = hh


def seg(a, b, n):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return a + np.linspace(0.0, 1.0, n + 1)[:, None] * (b - a)


L1_U = _DC - hw                   # u of D1's LEFT tip -- the morph anchor
_UMIN, _UMAX = -hw, 1.0           # flat's u extent


def _morph_u(u, anchor):
    """redistribute u about ``anchor``: GRADE < 1 spreads the columns nearest the
    anchor wide, compressing them away from it.  PERIOD-1 EXACT -- mu(u+1) ==
    mu(u)+1 -- so it never resizes the domain and the tube's wrap seam (u and
    u+1) still coincides.  Applied to every node, diamond vertices included."""
    u = np.asarray(u, float)
    d = u - anchor
    k = np.round(d)                                    # whole turns
    r = d - k                                          # in [-0.5, 0.5]
    rs = np.sign(r) * np.abs(2.0 * r) ** GRADE / 2.0   # stretch, stays in [-.5,.5]
    return anchor + k + rs


def mu(u):
    """the template morph -- anchored at D1's left tip."""
    return _morph_u(u, L1_U)


def chain(*legs):
    """One LineMesh through the concatenated straight legs (order ORDER)."""
    pts = np.vstack([legs[0]] + [leg[1:] for leg in legs[1:]])
    return linemesh.loft(pts, order=ORDER)


def map_nodes(qm, fn):
    """``qm`` with ``fn`` applied to every stored node -- shared corners, shared
    edge interiors and private face interiors alike."""
    lm = qm.line_mesh
    pm = PointMesh(fn(lm.points), lm.point_mesh.element_tags)
    li = lm.interior
    lm2 = LineMesh(pm, lm.lines,
                   fn(li.reshape(-1, 3)).reshape(li.shape) if li.size else li,
                   lm.element_tags)
    qi = qm.interior
    return QuadMesh(lm2, qm.quads, qm.orient,
                    fn(qi.reshape(-1, 3)).reshape(qi.shape) if qi.size else qi,
                    qm.element_tags)


# --- the one diamond D1 (D2 removed).  (NV/2)**2 quads.  Its lower-right and
# lower-left edges (B1->R1, B1->L1) are now free -- D1 hangs below the strips.
D1 = quadmesh.structured({                           # corners B1, R1, T1, L1
    "bottom": chain(seg(B1, R1, nd)),               # lower-right -> free (torn)
    "right":  chain(seg(R1, T1, nd)),               # upper-right -> right strip
    "top":    chain(seg(T1, L1, nd)),               # upper-left  -> free (torn)
    "left":   chain(seg(L1, B1, nd)),               # lower-left  -> free (torn)
})

# --- left strip: HALF height (v in [hh, 2hh]); its lower part is removed with
# D2, so it blends only onto the ghost L1->Lg.
left = quadmesh.structured({
    "bottom": chain(seg(ML, L1, nhL)),
    "right":  chain(seg(L1, Lg, nd)),               # ghost only
    "top":    chain(seg(Lg, TL, nhL)),
    "left":   chain(seg(TL, ML, nd)),
}, side_tags={"left": _END_HI})                     # u = -hw -> the turn's HIGH-z end

# --- right strip: HALF height; blends onto D1's upper-right edge only.
right = quadmesh.structured({
    "bottom": chain(seg(R1, MR, nhR)),
    "right":  chain(seg(MR, TR, nd)),
    "top":    chain(seg(TR, T1, nhR)),
    "left":   chain(seg(T1, R1, nd)),               # D1.T1->R1
}, side_tags={"right": _END_LO})                    # u = 1 -> the turn's LOW-z end

flat = quadmesh.merge([left, D1, right], tol=1e-9)

# --- torus-only collars: a uniform line above the strip tops (v = 2hh) and below
# the strip bottoms (v = hh), each joined to the notched sheet edge by one quad
# row.  ``flat`` (the inner_sheet's template) stays untouched -- collars feed the
# o-grid crossings only, and the inner_sheet loft skips them.
# the template is centred at V_CENTER = D1's centroid (v = hh).  The uniform
# collars are placed symmetrically about it (half-span COLLAR_HALF), so the
# mapped z = z_mid lands there in both the torus (via V_LO/V_HI) and the inner
# sheet (via loop_map); D1's tips (v = 0, v = 2hh) sit symmetric about it too.
V_CENTER = hh
COLLAR_HALF = 2.0 * hh
COLLAR_TOP = V_CENTER + COLLAR_HALF
COLLAR_BOT = V_CENTER - COLLAR_HALF
_P = flat.points


def _chain(v_edge, notch):
    row = _P[np.abs(_P[:, 1] - v_edge) < 1e-7]
    alln = np.vstack([row, np.asarray(notch, float).reshape(1, 3)])
    return alln[np.argsort(alln[:, 0])]


def _uniform(v, ref):
    u = np.linspace(ref[:, 0].min(), ref[:, 0].max(), ref.shape[0])
    return np.column_stack([u, np.full(u.size, v), np.zeros(u.size)])


_tc, _bc = _chain(2.0 * hh, L1), _chain(hh, B1)
_ENDS = {"x_min": _END_HI, "x_max": _END_LO}        # the same two sides, one row on
collar_t = quadmesh.from_grid(np.stack([_tc, _uniform(COLLAR_TOP, _tc)], axis=1),
                              order=ORDER, side_tags=_ENDS)
collar_b = quadmesh.from_grid(np.stack([_uniform(COLLAR_BOT, _bc), _bc], axis=1),
                              order=ORDER, side_tags=_ENDS)
flat_t = quadmesh.merge([flat, collar_t, collar_b], tol=1e-9)


def _mu_pts(P):
    """morph a node array's u toward D1's right tip (the whole-template remap),
    node by node -- so it stays self-consistent through order-2 midsides."""
    P = np.asarray(P, float).reshape(-1, 3)
    return np.column_stack([mu(P[:, 0]), P[:, 1], P[:, 2]])


flat_m = map_nodes(flat, _mu_pts)                  # morphed -- feeds the layers
flat_tm = map_nodes(flat_t, _mu_pts)              # morphed -- feeds the coil


V_LO, V_HI = float(flat_t.points[:, 1].min()), float(flat_t.points[:, 1].max())

# --- the wire tube cross-section as an O-GRID disc (x = R - R_HELIX, y = z).
# flat_t maps onto each VERTICAL CROSSING of it (bottom -> top), and the crossings
# loft in order of increasing x -- i.e. increasing big radius R.  Crossing 0 is
# the inner wall = the inward-facing quadrant; the cob construction
# (tjunction_lib.build_cob) with rows and columns swapped.
OG_NSIDE  = 2                          # o-grid core cells per side (even)
OG_RADIAL = 1                          # o-grid ring layers

def _og_boundary(radius, inner_deg):
    """closed 8-pt loop on a circle whose o-grid quarter split points sit at +x,
    +y, -x, -y, but with the -x-facing pair of edges (and the +x pair) widened to
    span ``inner_deg`` each, the two side pairs sharing the rest."""
    he = inner_deg / 2.0
    side = 90.0 - he
    ang = np.deg2rad(np.cumsum([0.0, he, side, side, he, he, side, side]))
    pts = np.column_stack([radius * np.cos(ang), radius * np.sin(ang),
                           np.zeros(ang.size)])
    ac = np.r_[ang, ang[0] + 2.0 * np.pi]
    mid_a = 0.5 * (ac[:-1] + ac[1:])
    mid = np.column_stack([radius * np.cos(mid_a), radius * np.sin(mid_a),
                           np.zeros(mid_a.size)])
    return linemesh.loft(pts, loop=True, interior=mid[:, None, :], order=ORDER)


_disc = quadmesh.ogrid(_og_boundary(RW, INNER_ARC_DEG), OG_NSIDE, OG_RADIAL)
_dq = np.asarray(_disc.quads).reshape(-1, 4)
_dl = np.asarray(_disc.line_mesh.lines).reshape(-1, 2)
_dp = _disc.points
_e2q = defaultdict(list)
for _q in range(_disc.n_quads):
    for _s in range(4):
        _e2q[int(_dq[_q, _s])].append((_q, _s))


def _walk(e, from_q):
    el, ed = [], [e]
    while True:
        nxt = [(q, s) for (q, s) in _e2q[e] if q != from_q]
        if not nxt:
            return el, ed
        q, s = nxt[0]
        el.append(q)
        e, from_q = int(_dq[q, (s + 2) % 4]), q
        ed.append(e)


def _mid(e):
    return _dp[_dl[e]].mean(axis=0)


# left-QUARTER wall edges (arc facing -x), bottom -> top; walk each rightward
_left = sorted((e for e, lst in _e2q.items()
               if len(lst) == 1 and _mid(e)[0] < 0
               and abs(_mid(e)[1]) <= abs(_mid(e)[0]) + 1e-9),
              key=lambda e: _mid(e)[1])
_cols = [_walk(e, -1) for e in _left]                    # OG_NSIDE horizontal walks
_ncol = len(_cols[0][0])                                 # 2*OG_RADIAL + OG_NSIDE
_BLK = element_blocks(_disc).reshape(_disc.n_quads, ORDER + 1, ORDER + 1, 3)

# LAT: axis 0 = y (across the walk, up the disc), axis 1 = x (along, -x -> +x)
_LAT = np.empty((OG_NSIDE * ORDER + 1, _ncol * ORDER + 1, 3))
for _r, (_els, _eds) in enumerate(_cols):
    strip = np.empty((ORDER + 1, _ncol * ORDER + 1, 3))
    for _c, (_q, _ein) in enumerate(zip(_els, _eds)):
        blk = _BLK[_q]
        if float(np.ptp(blk[:, :, 1].mean(axis=1))) < float(np.ptp(blk[:, :, 1].mean(axis=0))):
            blk = blk.transpose(1, 0, 2)                 # axis 0 -> larger y spread
        if blk[-1, :, 1].mean() < blk[0, :, 1].mean():
            blk = blk[::-1]                              # axis 0 -> +y
        if blk[:, -1, 0].mean() < blk[:, 0, 0].mean():
            blk = blk[:, ::-1]                           # axis 1 -> +x
        strip[:, _c * ORDER:(_c + 1) * ORDER + 1] = blk
    _LAT[_r * ORDER:(_r + 1) * ORDER + 1] = strip
_NX = _LAT.shape[1]                                      # number of vertical crossings


RISE = PITCH / (2.0 * np.pi)                             # axial rise per radian


def cross_map(c, turn):
    """(u, v, 0) -> flat_t on vertical crossing ``c`` of the tube o-grid at helix
    ``turn``: u -> theta (global u = turn + 1 - u), v (V_LO..V_HI) -> up the
    crossing, z = RISE * global-theta."""
    col = _LAT[:, c, :]                                  # (ny, 3): (x = R-R_HELIX, z, 0)

    def fn(P):
        # quadratic (order-2) interpolation PER o-grid element down the crossing,
        # so the mapped nodes land on the o-grid's curved arc, not on its chords
        # -- a plain linear read here facets the inner wall.  Hardcoded for ORDER 2.
        P = np.asarray(P, dtype=float).reshape(-1, 3)
        frac = np.clip((P[:, 1] - V_LO) / (V_HI - V_LO), 0.0, 1.0)
        s = frac * OG_NSIDE
        e = np.clip(s.astype(int), 0, OG_NSIDE - 1)
        xi = (s - e)[:, None]
        a = e * ORDER
        off = (2.0 * (xi - 0.5) * (xi - 1.0) * col[a]
               - 4.0 * xi * (xi - 1.0) * col[a + 1]
               + 2.0 * xi * (xi - 0.5) * col[a + 2])      # (N,3): (R-R_HELIX, z, 0)
        gu = (turn + 1.0) - P[:, 0]
        th = gu * 2.0 * np.pi
        z = RISE * gu * 2.0 * np.pi + off[:, 1]
        rr = R_HELIX + off[:, 0]
        return np.column_stack([rr * np.cos(th), rr * np.sin(th), z])
    return fn


R_INNER = R_HELIX - RW - SHEET_GAP

# --- the o-grid's TOP and BOTTOM quadrants: a quad band from the central band's
# top/bottom edge (_LAT[-1] / _LAT[0], _ncol elements wide -- same resolution) out
# to the disc wall, swept round theta on the collar stations so it welds to the
# central band's top/bottom face column for column.
# uniform stations -- the morph is applied inside _place (node-wise), so a
# midside station is mu(mean) not mean(mu), matching the morphed tube.
_QSTA_U = np.linspace(_tc[:, 0].min(), _tc[:, 0].max(), _tc.shape[0])[:-1]


def _place(uk, turn):
    """place a quadrant-section node (x = R - R_HELIX, y = z-offset) at station
    ``uk`` of helix ``turn``, morphed the same way the tube is."""
    uk = float(mu(np.array([uk]))[0])
    gu = (turn + 1.0) - uk
    th = gu * 2.0 * np.pi
    zc = RISE * gu * 2.0 * np.pi

    def fn(P):
        P = np.asarray(P, dtype=float).reshape(-1, 3)
        rr = R_HELIX + P[:, 0]
        return np.column_stack([rr * np.cos(th), rr * np.sin(th), zc + P[:, 1]])
    return fn


def _quadrant_sec(lat_row):
    """one quad layer: ``lat_row`` (the butterfly core's top/bottom edge, kept
    exactly) lofted straight out along its own radius to the disc surface."""
    rad = RW / np.linalg.norm(lat_row[:, :2], axis=1)
    wall = np.column_stack([lat_row[:, 0] * rad, lat_row[:, 1] * rad,
                            np.zeros(lat_row.shape[0])])
    inner = linemesh.loft(lat_row[::ORDER], order=ORDER,
                          interior=lat_row[1::ORDER][:, None, :])
    outer = linemesh.loft(wall[::ORDER], order=ORDER,
                          interior=wall[1::ORDER][:, None, :])
    # the sweep runs core-edge -> disc wall, so its last cap IS the coil's wall
    # over the top / bottom of the tube.  Swept, that cap becomes lateral faces.
    return quadmesh.loft([inner, outer], last_tag=IFACE)


def _sweep_quadrant(lat_row, turn, shift=0):
    """sweep the quadrant section round one helix ``turn``.  ``shift`` slides the
    station window that many elements EARLIER (>0) or LATER (<0) at both ends, so
    the band leads / trails the tube by one element."""
    sec = _quadrant_sec(lat_row)
    u = _QSTA_U
    n = len(u)
    uu = np.r_[u - 1.0, u, u + 1.0]
    uo = uu[n - shift:2 * n - shift + 1]
    umid = 0.5 * (uo[:-1] + uo[1:])
    return hexmesh.loft([map_nodes(sec, _place(uk, turn=turn)) for uk in uo],
                        element_tags="solid",
                        first_tag=CUT if turn == SHEET_TURNS - 1 else None,
                        last_tag=CUT if turn == 0 else None,
                        sweep_nodes=[[map_nodes(sec, _place(um, turn=turn))]
                                    for um in umid])


_CORE = slice(OG_RADIAL * ORDER, OG_RADIAL * ORDER + OG_NSIDE * ORDER + 1)


def build_turn(turn):
    """one turn of the CONTINUOUS helix: the diamond-template o-grid tube (flat_t
    across the crossings) + top/bottom quadrants, at global u = local u + turn,
    z = RISE * global-theta.  Turns abut and weld at integer u."""
    # the tube sweeps radially, so its two helix ends are lateral faces and come
    # from the section's own vertical sides -- named ``cut`` only on the outermost
    # turns, since on any other turn that side is the seam the next turn welds to.
    sec = quadmesh.retag_edge(flat_tm, {
        _END_LO: CUT if turn == 0 else "",
        _END_HI: CUT if turn == SHEET_TURNS - 1 else ""})

    def _cx(c):
        return quadmesh.merge([map_nodes(sec, cross_map(c, turn=turn))], tol=1e-9)
    # the crossings loft in order of increasing radius, so the sweep's FIRST slice
    # is the o-grid's inward arc -- the wall the inner sheet, the layers and the
    # inter-turn gap are all built onto -- and its LAST is the outward arc the
    # wall film is built onto.  Both are the conjugate surface; they differ only
    # in which block meets them, which is why the outward one keeps its own name
    # until ``wall_wrap`` has been picked off it.
    tube = hexmesh.loft([_cx(c) for c in range(0, _NX, ORDER)], element_tags="solid",
                        first_tag=IFACE, last_tag=SKIN,
                        sweep_nodes=[[_cx(c)] for c in range(1, _NX, ORDER)])
    topq = _sweep_quadrant(_LAT[-1, _CORE], turn=turn)
    botq = _sweep_quadrant(_LAT[0, _CORE][::-1], turn=turn, shift=-1)
    return hexmesh.merge([tube, topq, botq], tol=1e-9)


# === INNER SHEET: a staircase quad band on the cylinder r = R_INNER, 3*PITCH
# tall.  Flattened it is 3 element rows x (K1+K2) columns; the BOTTOM row keeps
# only its right K1 columns and the TOP row only its left K2 columns (K1/K2 =
# the template's element counts either side of the diamond), so stacked turns
# interlock top-left into bottom-right. ======================================
K1 = right.n_quads // nd + 1          # right strip + the diamond column
K2 = left.n_quads // nd               # left strip
NSHEET = K1 + K2                      # the middle (2nd) row is always this wide
INNER_LAYERS = 3


def _bandu(uu, v0, v1):
    vv = np.arange(v0, v1 + 1, dtype=float)
    U, V = np.meshgrid(np.asarray(uu, float), vv, indexing="ij")
    return quadmesh.from_grid(np.stack([U, V, np.zeros_like(U)], axis=-1),
                              order=ORDER)


# === CONTINUOUS HELIX: the diamond-template o-grid tube, one turn per pitch,
# turns abutting and welding -- one continuous coil. ==========================

wire_coil = hexmesh.merge([build_turn(k) for k in range(SHEET_TURNS)], tol=1e-9)
b = hexmesh.bounds(wire_coil)

# --- place the inner sheet so its centre lands on the coil's z centre.
INNER_ZC = 0.5 * (float(b.min[2]) + float(b.max[2]))


def _cyl(P):
    # same theta(u) law as the coil's turn 0 (cross_map: th = (1 - u) * 2pi),
    # so the sheet's bottom row overlays the coil's D1 + right-strip band.
    P = np.asarray(P, float).reshape(-1, 3)
    vc = P[:, 1] - 0.5 * INNER_LAYERS
    th = (1.0 - P[:, 0]) * 2.0 * np.pi + np.deg2rad(INNER_PHASE_DEG)
    z = vc * PITCH + INNER_ZC + INNER_DZ
    return np.column_stack([R_INNER * np.cos(th), R_INNER * np.sin(th), z])


# the same morph in the inner sheet's own u in [0,1], anchored at the sheet-u of
# D1's right tip (right edge of the D1 column) -- so the sheet grades exactly as
# the template does, column for column.
_uR = (NSHEET - K1) / NSHEET
_uL = K2 / NSHEET
def mu_s(u):
    return _morph_u(u, _uR)                       # sheet-u of D1's left edge


def _mus_pts(P):
    P = np.asarray(P, float).reshape(-1, 3)
    return np.column_stack([mu_s(P[:, 0]), P[:, 1], P[:, 2]])


_U_FULL = np.linspace(0.0, 1.0, NSHEET + 1)
_u_sheet = np.linspace(_uR, 1.0, K1 + 1)
_u2 = np.linspace(_uL, 0.0, nhL + 1)

inner_sheet = map_nodes(quadmesh.merge([
    _bandu(_U_FULL[NSHEET - K1:], 0, 1),   # bottom row -- right K1 columns
    _bandu(_U_FULL, 1, 2),                 # middle row -- full K1+K2 width
    _bandu(_U_FULL[:K2 + 1], 2, 3),        # top row -- left K2 columns
], tol=1e-9), _mus_pts)
inner_sheet = map_nodes(inner_sheet, _cyl)

# === CORE: fill the cylinder inside the inner sheet, conforming to its NSHEET
# columns exactly.  NSHEET is now 4*n_side, so an o-grid butterfly takes the ring
# directly (n_side = NSHEET/4, CORE_RADIAL ring layers).  Extruded CORE_LAYERS up.
CORE_LAYERS = 2
CORE_RADIAL = 1
_core_zs = [(-0.5 + i) * PITCH + INNER_ZC + INNER_DZ    # v = 1, 2, 3 -- ``_cyl``'s
            for i in range(CORE_LAYERS + 1)]           # own z law, INNER_DZ included


def _sheet_ring(zt):
    m = np.abs(inner_sheet.points[:, 2] - zt) < 0.02
    pts = inner_sheet.points[m]
    pts = pts[np.argsort(np.arctan2(pts[:, 1], pts[:, 0]))]
    keep = [0]
    for i in range(1, len(pts)):
        if np.linalg.norm(pts[i] - pts[keep[-1]]) > 1e-7:
            keep.append(i)
    return pts[keep]


_ring0 = _sheet_ring(_core_zs[1])                         # the full NSHEET ring
_NR = len(_ring0)
assert _NR % 4 == 0, "core o-grid needs NSHEET divisible by 4 (got %d)" % _NR
_CORE_R = float(np.hypot(_ring0[:, 0], _ring0[:, 1]).mean())


def _core_disc(z, wall=""):
    ang = np.arctan2(_ring0[:, 1], _ring0[:, 0])
    mid_ang = ang + 0.5 * np.diff(np.r_[ang, ang[0] + 2.0 * np.pi])
    corn = np.column_stack([_CORE_R * np.cos(ang), _CORE_R * np.sin(ang),
                            np.full(_NR, z)])
    mids = np.column_stack([_CORE_R * np.cos(mid_ang), _CORE_R * np.sin(mid_ang),
                            np.full(_NR, z)])
    loop = linemesh.loft(corn, loop=True, interior=mids[:, None, :], order=ORDER)
    return quadmesh.ogrid(loop, _NR // 4, CORE_RADIAL, wall_tag=wall)


# two lofts: the lower CORE_LAYERS-1 layers weld to the inner sheet, the top layer
# does not (the sheet's top row is only K2 columns wide), so its ring is exposed to
# the outlet.  ``wall_tag`` on the top layer's FIRST slice rides onto that lateral
# hex-face family; a single 3-slice loft would tag the whole column instead.
_core_lo = hexmesh.loft([_core_disc(z) for z in _core_zs[:-1]],
                        element_tags="fluid", first_tag=_END_LO)
core_hex = hexmesh.merge([_core_lo, hexmesh.loft(
    [_core_disc(_core_zs[-2], wall=_END_HI), _core_disc(_core_zs[-1])],
    element_tags="fluid", last_tag=_END_HI)], tol=1e-9)
_cq = hexmesh.quality_summary(core_hex)

# the two transition-layer template grids (also used by the hex layers below):
# column 1 of each is the template's TOP edge -- the very edge that the staircase
# is on the inner-sheet side and the spiral is on the inward-facing-wall side.
_sb, _st = seg(R1, MR, nhR), seg(T1, TR, nhR)
_G_coil = np.stack(
    [np.vstack([B1, _sb[:K1]]), np.vstack([L1, _st[:K1]])], axis=1)   # (K1+1, 2, 3)
_G_coil2 = np.stack([seg(L1, ML, nhL), seg(Lg, TL, nhL)], axis=1)     # (nhL+1, 2, 3)


# --- the STAIRCASE line = the trace of the template's TOP edge onto the inner
# sheet, ONE continuous line, 40 segments:
#   from (u=1, v=1) go LEFT K1=10 cols along row v=1  (bottom-row top edge)
#   UP 1 to v=2 at the corner u=_uR
#   LEFT a full loop, K1+K2=19 cols, along row v=2  (middle-row top edge)
#   UP 1 to v=3
#   LEFT K2=9 cols along row v=3 to u=0  (top-row top edge)
def _sheet_xyz(uv):
    return _cyl(_mus_pts(np.asarray(uv, float)))


def _seg(u_arr, v):
    return [(float(u), float(v), 0.0) for u in u_arr]


_loop_u = np.r_[_U_FULL[NSHEET - K1::-1],                   # _uR .. 0
                _U_FULL[NSHEET - 1:NSHEET - K1 - 1:-1]]     # (wrap) 1- .. _uR
# ``between`` spans _NBTW pitches (turns _ST .. _ST + _NBTW - 1) so BOTH gap hex
# lofts -- gap_hi over turns 0..1, gap_lo over turns -1..0 -- land on it.
_NBTW = 3
_stair_uv = _seg(_u_sheet[::-1], 1)             # v=1, u 1 -> _uR      (10 segs)
for _i in range(1, _NBTW):                      # full-loop middle rows
    _stair_uv += [(_uL, _i + 1.0, 0.0)]         # riser                (1 seg)
    _stair_uv += _seg(_loop_u[1:], _i + 1)      # v=_i+1, full loop    (19 segs)
_stair_uv += [(_uL, _NBTW + 1.0, 0.0)]          # riser                (1 seg)
_stair_uv += _seg(_u2[1:], _NBTW + 1)           # top row, u _uR -> 0  (9 segs)
_stair_pts = _sheet_xyz(_stair_uv)
_stair_pts[:, 2] -= PITCH                         # match the spiral's turn-_ST start
staircase = linemesh.loft(_stair_pts, order=ORDER)


# --- the SPIRAL line = the TOP edge of gap_hi_target (the coil's inward-wall
# COLLAR_TOP row, v -> V_HI: column 1 of _G_coil / _G_coil2 with its v overridden
# to V_HI, mapped by cross_map at crossing 0), chained turn by turn exactly as the
# staircase chains its inner-sheet trace, then shifted SPIRAL_DZ_FRAC of a pitch
# so it lands out in the gap between the wire turns.
def _atv(a, v):
    a = np.asarray(a, float).copy()
    a[:, 1] = v
    return a


_Gc = _mu_pts(_atv(_G_coil[:, 1, :], V_HI))        # D1 + right-strip COLLAR_TOP
_Gc2 = _mu_pts(_atv(_G_coil2[:, 1, :], V_HI))      # left-strip COLLAR_TOP
# start one turn earlier (turns _ST .. _ST + _NBTW - 1) so ``between`` reaches a
# full pitch lower -- gap_lo then has a surface to mesh down onto.
_ST = -1
_sp_rows = []
for _j in range(_NBTW):
    _k = _ST + _j
    _right = _Gc[::-1] if _j == 0 else _Gc[::-1][1:]   # dedup wrap-seam node
    _sp_rows += [cross_map(0, turn=_k)(_right),         # TR -> L1
                 cross_map(0, turn=_k)(_Gc2[:1]),       # riser: Lg
                 cross_map(0, turn=_k)(_Gc2[1:])]       # loop left
spiral_pts = np.vstack(_sp_rows)
spiral_pts[:, 2] += SPIRAL_DZ_FRAC * PITCH
# nudge clockwise by a fraction of one element's angular pitch
_rot = -SPIRAL_ROT_FRAC * 2.0 * np.pi / len(_QSTA_U)   # one tube-sweep element
_c, _s = np.cos(_rot), np.sin(_rot)
spiral_pts[:, :2] = spiral_pts[:, :2] @ np.array([[_c, _s], [-_s, _c]])
spiral = linemesh.loft(spiral_pts, order=ORDER)

# --- ``between`` is now a RADIAL STACK of OG_NSIDE + 1 quad layers.  Layer 0 is
# staircase -> spiral (already lofted by gap_lo / gap_hi).  The other OG_NSIDE
# layers are the spiral scaled out to successive radii -- one per o-grid core cell
# radially -- so they weld to the wire's top / bottom quadrant faces.  The spiral
# is a circular helix at a single radius, so a uniform xy scale is exact.
assert _stair_pts.shape[0] == spiral_pts.shape[0], (_stair_pts.shape, spiral_pts.shape)
R_SPIRAL = float(np.hypot(spiral_pts[:, 0], spiral_pts[:, 1]).mean())
# the OG_NSIDE extra layers end on the radial CORNERS of the wire's top/bottom
# quadrant section -- _quadrant_sec(_LAT[-1,_CORE]): the core's top-edge corners
# AND their wall projection, so the span is the full topq/botq radial extent
# (R_SPIRAL .. 2*R_HELIX-R_SPIRAL for the symmetric OG_NSIDE=2 case), not just the
# narrower core edge.  ``_place`` maps x -> R - R_HELIX.
_lr = _LAT[-1, _CORE][::ORDER]
_wall_r = _lr * (RW / np.hypot(_lr[:, 0], _lr[:, 1]))[:, None]
_sec_r = np.sort(R_HELIX + np.r_[_lr[:, 0], _wall_r[:, 0]])
_BTW_R = [R_SPIRAL, R_HELIX, float(_sec_r.max())][:OG_NSIDE + 1]


def _spiral_at(R):
    p = spiral_pts.copy()
    p[:, :2] *= R / R_SPIRAL
    return p


_btw_lines = [staircase] + [linemesh.loft(_spiral_at(R), order=ORDER) for R in _BTW_R]
between = quadmesh.loft(_btw_lines)
_NPT = NSHEET + 1                                     # corner steps per pitch (riser + loop)


def _turn_tags(qm, end_tag, at_start):
    """``qm`` (a one-row staircase->spiral strip) with the one full turn at its
    helix end -- the first ``_NPT`` quads if ``at_start`` else the last -- given
    ``element_tags``.  A ``loft`` carries a section's element_tags onto the hex
    face family it becomes, so the gap block's whole end cross-section (not just
    its outermost edge) comes out named."""
    _d = np.full(qm.n_quads, "", dtype="<U16")
    if at_start:
        _d[:_NPT] = end_tag
    else:
        _d[-_NPT:] = end_tag
    return QuadMesh(qm.line_mesh, qm.quads, qm.orient, qm.interior,
                    ElementTags.from_dense(_d))


# _stair_pts / spiral_pts chain low z -> high z, so their first turn is the
# domain's LO opening and their last turn its HI one.
# ``tag_edges`` on the strip's helix-end cap edge (quad 0 side 4 at the LO end,
# last quad side 2 at the HI end) -- ``hexmesh.loft`` sweeps it onto the gap
# block's first / last element's exposed lateral side.
_bl = quadmesh.loft([linemesh.loft(_stair_pts[:2 * _NPT + 1], order=ORDER),
                     linemesh.loft(spiral_pts[:2 * _NPT + 1], order=ORDER)])
between_lo = _turn_tags(
    quadmesh.tag_edges(_bl, [(0, 4), (_bl.n_quads - 1, 2)], [_END_LO, _END_HI]),
    _END_LO, True)
_bh = quadmesh.loft([linemesh.loft(_stair_pts[_NPT:], order=ORDER),
                     linemesh.loft(spiral_pts[_NPT:], order=ORDER)])
between_hi = _turn_tags(
    quadmesh.tag_edges(_bh, [(0, 4), (_bh.n_quads - 1, 2)], [_END_LO, _END_HI]),
    _END_HI, False)


# --- fill the inter-turn wedge.  ``between`` sits SPIRAL_DZ_FRAC of a pitch above
# the template-top-edge trace, i.e. it now lies below the NEXT turn's bottom.  So
# loft hexes toward it from (a) THIS turn's inward-wall TOP element and (b) the
# NEXT turn's inward-wall BOTTOM element.  _iw_chain re-runs the spiral chain with
# the template's v swept to a wall node row and the turns shifted by ``dt``.
def _iw_row(vval=None, dt=0, col=1):
    # the transition layers' coil-side edge, EXACTLY: the _G_coil / _G_coil2 row
    # ``col`` (1 = v=2hh top, 0 = v=hh bottom), its own v kept when vval is None
    # else overridden (COLLAR_TOP / COLLAR_BOT), chained in the spiral order with
    # order-2 midsides from cross_map of the RAW (u,v) midpoints -- matching the
    # layers' from_grid, so the shared edge coincides node-for-node.
    g0 = _G_coil[:, col, :].copy()
    h0 = _G_coil2[:, col, :].copy()
    if vval is not None:
        g0[:, 1] = vval
        h0[:, 1] = vval

    def piece(raw, k):                                    # raw (u,v,0) corner list
        c = cross_map(0, turn=k)(_mu_pts(raw))
        m = cross_map(0, turn=k)(_mu_pts(0.5 * (raw[:-1] + raw[1:])))
        return c, m

    cs, ms = [], []
    for k in range(dt, dt + SHEET_TURNS):
        a = g0[::-1]                                      # TR .. T1 .. L1
        ca, ma = piece(a, k)
        cg = cross_map(0, turn=k)(_mu_pts(h0[:1]))        # Lg (riser end)
        mg = cross_map(0, turn=k)(_mu_pts(0.5 * (a[-1:] + h0[:1])))  # L1->Lg mid
        cb, mb = piece(h0, k)                             # Lg .. TL
        turn_c = np.vstack([ca, cg, cb[1:]])
        turn_m = np.vstack([ma, mg, mb])
        cs.append(turn_c if k == dt else turn_c[1:])   # dedup the wrap-seam node
        ms.append(turn_m)                               # ...but keep its midside
    return linemesh.loft(np.vstack(cs), order=ORDER,
                         interior=np.vstack(ms)[:, None, :])



def _rvl(L):
    return linemesh.loft(L.points[::-1], order=ORDER)


def _gap_hex2(bot_line, top_line, target):
    """loft a coil collar face (bot_line -> top_line) toward ``target``, trying
    both line reversals and both loft orders until one lofts without a fold."""
    for a, b in [(bot_line, top_line), (top_line, bot_line),
                 (_rvl(bot_line), _rvl(top_line)), (_rvl(top_line), _rvl(bot_line))]:
        face = quadmesh.loft([a, b])
        # ``target`` first, so its helix-end edge tag rides onto the hex face
        for order in ([target, face], [face, target]):
            try:
                h = hexmesh.loft(order, element_tags="fluid")
            except ValueError:
                continue
            if hexmesh.quality_summary(h).n_inverted == 0:
                return h
    return None


# gap_hi: fills the wedge above turns 0..1 from the collar-top row down to
# ``between``'s upper quads.
gap_hi = _gap_hex2(_iw_row(dt=0), _iw_row(V_HI, dt=0), between_hi)

# gap_lo: same coil-face turns, meshed DOWN one pitch onto ``between``'s lower
# quads -- the inter-turn gap that was being skipped.
gap_lo = _gap_hex2(_iw_row(V_LO, dt=0, col=0), _iw_row(dt=0, col=0), between_lo)

# --- topq / botq inter-turn fill: ``between``'s OG_NSIDE OUTER layers lofted to
# the wire's top / bottom quadrant faces, exactly as gap_hi / gap_lo do it for
# layer 0.  The coil-side face is the quadrant section's WALL edge (``_LAT`` row
# projected to radius RW), swept by ``_place`` over the same stations the quadrant
# band itself uses -- so it coincides with topq / botq node-for-node.
def _wall_face(lat_row, dt=0, shift=0, nturns=SHEET_TURNS):
    # ``lat_row`` projected to the tube wall (radius RW), swept by ``_place`` over
    # the quadrant band's own stations, built the SAME way as _btw_outer_slice
    # (one sweep line per radial corner, then quadmesh.loft) so the two pair up.
    lat_row = np.asarray(lat_row, float)
    wall = lat_row * (RW / np.hypot(lat_row[:, 0], lat_row[:, 1]))[:, None]
    wc = wall[::ORDER]
    u = _QSTA_U
    n = len(u)
    uu = np.r_[u - 1.0, u, u + 1.0]
    uo = uu[n - shift:2 * n - shift + 1][::-1]         # sweep dir to match _btw_outer
    um = 0.5 * (uo[:-1] + uo[1:])
    lines = []
    for r in range(wc.shape[0]):
        cc, mm = [], []
        for k in range(dt, dt + nturns):
            ck = [_place(x, turn=k)(wc[r:r + 1])[0] for x in uo]
            cc += ck if k == dt else ck[1:]
            mm += [_place(x, turn=k)(wc[r:r + 1])[0] for x in um]
        lines.append(linemesh.loft(np.array(cc), order=ORDER,
                                   interior=np.array(mm)[:, None, :]))
    return quadmesh.loft(lines, last_tag=SKIN)      # outermost radial corner


def _btw_outer_slice(a, b):
    # a == 0 -> reaches the helix bottom (-> botq_gap); b is None -> the top
    # (-> topq_gap).  Tag ONE full turn at that end -- the ``_NPT`` quads at the
    # helix end of EACH radial layer -- so the gap block's whole end cross-section
    # comes out named.  ``quadmesh.loft`` of N lines is radial-major: layer i is
    # quads [i*S : (i+1)*S] with S the per-layer sweep count.
    _lo = a == 0
    _q = quadmesh.loft([linemesh.loft(_spiral_at(R)[a:b], order=ORDER)
                        for R in _BTW_R], last_tag=SKIN)
    _S = _q.n_quads // OG_NSIDE
    _d = np.full(_q.n_quads, "", dtype="<U16")
    # ``_spiral_at`` runs low z -> high z, so the a-end of every radial layer is a
    # LO helix cut and the b-end a HI one; ``tag_edges`` sweeps each onto the gap
    # block's first / last element's exposed lateral side.  The element_tags name
    # the one full turn at whichever end reaches a domain opening.
    _re, _rt = [], []
    for _i in range(OG_NSIDE):
        if _lo:
            _d[_i * _S:_i * _S + _NPT] = _END_LO
        else:
            _d[(_i + 1) * _S - _NPT:(_i + 1) * _S] = _END_HI
        _re += [(_i * _S, 4), ((_i + 1) * _S - 1, 2)]
        _rt += [_END_LO, _END_HI]
    _q = quadmesh.tag_edges(_q, _re, _rt)
    return QuadMesh(_q.line_mesh, _q.quads, _q.orient, _q.interior,
                    ElementTags.from_dense(_d))


def _quad_hex(face, target):
    """loft the coil quadrant face -> a ``between`` outer slice, no fold.
    ``target`` first so its helix-end edge tag becomes a hex face tag."""
    for order in ([target, face], [face, target]):
        try:
            h = hexmesh.loft(order, element_tags="fluid")
        except ValueError:
            continue
        if hexmesh.quality_summary(h).n_inverted == 0:
            return h
    return None


_tq_face = _wall_face(_LAT[-1, _CORE], dt=0, shift=0)
_bq_face = _wall_face(_LAT[0, _CORE], dt=0, shift=-1)
_tq_tgt = _btw_outer_slice(_NPT, None)                 # turns 0..1, like gap_hi
_bq_tgt = _btw_outer_slice(0, 2 * _NPT + 1)            # turns -1..0, like gap_lo
topq_gap = _quad_hex(_tq_face, _tq_tgt)
botq_gap = _quad_hex(_bq_face, _bq_tgt)
_gaps = [g for g in (gap_hi, gap_lo, topq_gap, botq_gap) if g is not None]
assert len(_gaps) == 4, "an inter-turn fill lost its non-folding ordering"
gap = hexmesh.merge(_gaps, tol=1e-9)


# === wall_wrap: the surface bounding the OUTER fluid region (coil <-> the solid
# cylinder) on the coil side.  It is not searched for -- every block named its own
# outward face where it was built (``last_tag=SKIN`` on the tube's radial sweep and
# on the two gap blocks' outward radial corner), so this just collects them.  A
# geometric test here would not survive INNER_ARC_DEG: the o-grid's outward arc
# spans that angle, so a fixed normal cone silently drops a strip of it -- at 162
# deg it lost 30 of the coil's 100 outward faces, and the fluid film and pipe wall
# around them (120 elements) went with it.
def _skin_of(block):
    # off the BOUNDARY surface, not the shared-face table: the latter stores each
    # face in whatever orientation its owner gave it, and a mixed-winding section
    # folds the loft below.  boundary_mesh winds them all outward and carries the
    # face tags through as its own element_tags.
    bm = hexmesh.boundary_mesh(block)
    return quadmesh.select(bm, np.flatnonzero(
        np.asarray(bm.element_tags.dense(bm.n_quads)) == SKIN))


_ww_wire, _ww_tq, _ww_bq = (_skin_of(b) for b in (wire_coil, topq_gap, botq_gap))
wall_wrap = quadmesh.merge([_ww_wire, _ww_tq, _ww_bq], tol=1e-9)
# from here it is a section to sweep, not a name: left in place it would ride up
# as the film's and the shell's cap names (``loft`` defaults first_tag/last_tag to
# the bounding slice's own element_tags).
wall_wrap = quadmesh.retag_element(wall_wrap, {SKIN: ""})
# the ribbon's rim rides up as a lateral face wherever it is swept.  It bounds the
# FLUID film first (an open end) and the SOLID shell after that (a saw cut).  The
# rim is TWO separable loops -- the wrap spirals a turn, so its boundary never
# closes on itself -- one wholly below the other in z; name them apart here so the
# film's LO loop is the inlet and its HI loop the outlet, no flood fill needed.
_be = quadmesh.boundary_edges(wall_wrap)
_zc = wall_wrap.points[np.asarray(wall_wrap.corners)][:, :, 2]        # per-quad, 4
_z_edge = 0.5 * (_zc[_be[:, 0], (_be[:, 1] - 1)]
                 + _zc[_be[:, 0], _be[:, 1] % 4])                     # each edge's z
_lo = _z_edge < _z_edge.mean()
wall_wrap = quadmesh.tag_edges(wall_wrap, _be[_lo], _RIM_LO)
wall_wrap = quadmesh.tag_edges(wall_wrap, _be[~_lo], _RIM_HI)

# the film has been picked off, so the coil's outward wall is just conjugate
# surface like the rest of it.  The two gap blocks are fluid on both sides of
# theirs -- it meets the film, not the coil -- so that name goes away entirely.
wire_coil = hexmesh.retag_face(wire_coil, {SKIN: IFACE})
topq_gap = hexmesh.retag_face(topq_gap, {SKIN: ""})
botq_gap = hexmesh.retag_face(botq_gap, {SKIN: ""})
gap = hexmesh.retag_face(gap, {SKIN: ""})


# --- OUTER fluid: loft wall_wrap straight out to the cylinder inner wall
# (R = R_TUBE) by projecting every node onto that radius from the z-axis.
def _to_R(P, R):
    P = np.asarray(P, float).reshape(-1, 3)
    s = R / np.hypot(P[:, 0], P[:, 1])
    return np.column_stack([P[:, 0] * s, P[:, 1] * s, P[:, 2]])


wall_wrap_out = map_nodes(wall_wrap, lambda P: _to_R(P, R_TUBE))
wall_hex = hexmesh.loft([wall_wrap, wall_wrap_out], element_tags="fluid")
wall_hex = hexmesh.retag_face(wall_hex, {_RIM_LO: "inlet", _RIM_HI: "outlet"})
_wq = hexmesh.quality_summary(wall_hex)

# --- SOLID outer wall: keep projecting wall_wrap_out radially outward, N_WALL
# hex layers from R_TUBE to R_TUBE + WALL_THICK.  Tagged solid.
wall_solid = hexmesh.loft(
    [map_nodes(wall_wrap_out, lambda P, R=r: _to_R(P, R))
     for r in np.linspace(R_TUBE, R_TUBE + WALL_THICK, N_WALL + 1)],
    element_tags="solid", last_tag=OUTER)      # the sweep ends ON the pipe skin
wall_solid = hexmesh.retag_face(wall_solid, {_RIM_LO: CUT, _RIM_HI: CUT})
_sq = hexmesh.quality_summary(wall_solid)


# === FIRST HEX LAYER: the coil template's diamond + the whole right strip
# lofted straight in to the sheet's bottom row, element by element.
# a in 0..K1 : a=[0,1] is D1, a=[1..K1] the right-strip elements;
# a=[0,1] (D1) lands on the sheet bottom row's LEFT-most element. ============
_G_sheet = np.stack(
    [np.column_stack([_u_sheet, np.zeros(K1 + 1), np.zeros(K1 + 1)]),
     np.column_stack([_u_sheet, np.ones(K1 + 1), np.zeros(K1 + 1)])], axis=1)

_qc = map_nodes(map_nodes(quadmesh.from_grid(_G_coil, order=ORDER), _mu_pts), cross_map(0, turn=0))
# the sheet's first (bottom) row -- a bottom opening; ``element_tag`` rides onto
# layer1's sheet-side face family and comes out ``inlet``.
_qs = map_nodes(map_nodes(quadmesh.from_grid(
    _G_sheet, order=ORDER, element_tag=_END_LO,
    side_tags={"x_min": _END_LO, "x_max": _END_LO}), _mus_pts), _cyl)
layer1 = hexmesh.loft([_qs, _qc], element_tags="fluid")
lq = hexmesh.quality_summary(layer1)


# === SECOND HEX LAYER: the template's LEFT strip (left half of the diamond
# template) lofted straight in to the left part of the sheet's 2nd row (middle
# row), element by element, starting from the staircase corner.
# a in 0..(K2-1) : a=0 is the strip edge next to D1 (the corner), a increases
# leftward out to ML/TL. ==================================================
_G_sheet2 = np.stack(
    [np.column_stack([_u2, np.full(nhL + 1, 1.0), np.zeros(nhL + 1)]),
     np.column_stack([_u2, np.full(nhL + 1, 2.0), np.zeros(nhL + 1)])], axis=1)

_qc2 = map_nodes(map_nodes(quadmesh.from_grid(_G_coil2, order=ORDER), _mu_pts), cross_map(0, turn=0))
_qs2 = map_nodes(map_nodes(quadmesh.from_grid(_G_sheet2, order=ORDER), _mus_pts), _cyl)
layer2 = hexmesh.loft([_qs2, _qc2], element_tags="fluid")
l2q = hexmesh.quality_summary(layer2)


# === THIRD HEX LAYER: the SECOND turn's diamond + right strip lofted in to the
# right part of the sheet's 2nd (middle) row -- so the centre layer is turn 0's
# left strip (layer2, left K2) + turn 1's D1 + right strip (this, right K1),
# K1 + K2 = NSHEET wide, the two turns interlocking at the centre. ===========
_u3 = _u_sheet
_G_sheet3 = np.stack(
    [np.column_stack([_u3, np.full(K1 + 1, 1.0), np.zeros(K1 + 1)]),
     np.column_stack([_u3, np.full(K1 + 1, 2.0), np.zeros(K1 + 1)])], axis=1)

_qc3 = map_nodes(map_nodes(quadmesh.from_grid(_G_coil, order=ORDER), _mu_pts), cross_map(0, turn=1))
_qs3 = map_nodes(map_nodes(quadmesh.from_grid(_G_sheet3, order=ORDER), _mus_pts), _cyl)
layer3 = hexmesh.loft([_qs3, _qc3], element_tags="fluid")
l3q = hexmesh.quality_summary(layer3)


# === FOURTH HEX LAYER: the SECOND turn's LEFT strip lofted in to the left part
# of the sheet's TOP row -- turn 1's template thus splits the same way turn 0's
# did (layer1/layer2): D1 + right strip -> right K1, left strip -> left K2. ===
_G_sheet4 = np.stack(
    [np.column_stack([_u2, np.full(nhL + 1, 2.0), np.zeros(nhL + 1)]),
     np.column_stack([_u2, np.full(nhL + 1, 3.0), np.zeros(nhL + 1)])], axis=1)

_qc4 = map_nodes(map_nodes(quadmesh.from_grid(_G_coil2, order=ORDER), _mu_pts), cross_map(0, turn=1))
_qs4 = map_nodes(map_nodes(quadmesh.from_grid(
    _G_sheet4, order=ORDER,
    side_tags={"x_min": _END_HI, "x_max": _END_HI}), _mus_pts), _cyl)
layer4 = hexmesh.loft([_qs4, _qc4], element_tags="fluid")
l4q = hexmesh.quality_summary(layer4)


# === assemble.  merge the coil + the four transition layers (they share curved
# faces node-for-node); ATTACH the rest, whose straight-subdivided guide edges sit
# ~1e-3 off the curved template edges -- too far for merge's shared-edge check,
# but attach reads only the corners (which coincide exactly).
def _bqids(m):
    ids = np.flatnonzero(_bfid(m))
    cen = m.points[np.asarray(m.quad_mesh.corners)[ids]].mean(axis=1)
    return ids, cen


def _face_region(mesh):
    """the region of the hex behind each face -- a boundary face has exactly one."""
    reg = mesh.element_tags.dense(mesh.n_hexes)
    out = np.full(mesh.quad_mesh.n_quads, "", dtype=object)
    for e, row in enumerate(np.asarray(mesh.hexes)):
        for q in row:
            if not out[q]:
                out[q] = reg[e]
    return out


def _weld(base, block, own):
    """attach ``block`` onto ``base`` along every boundary face they share (the
    blocks were built so those faces coincide to ~1e-16).

    A seam is stated in TWO halves, because ``attach_tag`` is one name for a whole
    seam and burying a face clears its name otherwise: the pairs whose two sides
    sit in different regions keep ``interface``, the rest are named nothing.  That
    is the conjugate surface carried across the weld rather than re-derived after
    it."""
    bi, bc = _bqids(base)
    ki, kc = _bqids(block)
    dist, near = _KD(bc).query(kc)
    sel = dist < 1e-6
    b_ids, k_ids = bi[near[sel]], ki[sel]
    assert len(np.unique(b_ids)) == len(b_ids), "weld seam is not one-to-one"
    conj = _face_region(base)[b_ids] != _face_region(block)[k_ids]
    seams = [hexmesh.Seam(0, b_ids[m], 1, k_ids[m], own=own, attach_tag=t)
             for m, t in ((conj, IFACE), (~conj, None)) if m.any()]
    out = hexmesh.attach([base, block], seams)
    # attach clears a buried face's name; the blocks' own boundary names (the
    # ends the core and film carry) must ride through untouched.  Concatenation
    # is base-then-block, so re-stamp them by matching the block's tagged
    # boundary faces onto the result.
    keep = np.asarray(block.face_tags.dense(block.quad_mesh.n_quads))
    kb = np.flatnonzero(_bfid(block) & (keep != ""))
    if kb.size:
        oi, oc = _bqids(out)
        kc = block.points[np.asarray(block.quad_mesh.corners)[kb]].mean(axis=1)
        d, near = _KD(oc).query(kc)
        hit = d < 1e-6
        out = hexmesh.tag_faces(out, oi[near[hit]], list(keep[kb[hit]]))
    return out


assembly = hexmesh.merge([wire_coil, layer1, layer2, layer3, layer4], tol=1e-9)
assembly = _weld(assembly, gap, own="a")
assembly = _weld(assembly, wall_hex, own="a")       # outer fluid film
assembly = _weld(assembly, wall_solid, own="a")     # solid pipe wall
assembly = _weld(assembly, core_hex, own="b")       # axial core (own="b": keep
#                                its o-grid edges; the sheet's curved midsides
#                                would fold the thin axis elements)


# === finish naming.  Every tag on the finished mesh is set by construction; the
# only work here is to turn the two placeholders into their solver names and to
# assert that what construction claims matches the assembled topology.
#
#   interface  the conjugate fluid/solid wall -- the coil named its own wall (the
#              tube's inward radial cap ``first_tag=IFACE``, the quadrant section's
#              outer cap ``last_tag=IFACE``) and ``_weld`` carried it across every
#              seam instead of burying it.
#   outer      the pipe skin -- ``last_tag=OUTER`` on ``wall_solid``'s radial sweep.
#   cut        the solid saw cut -- the coil's two helix ends (``retag_edge`` on
#              the template's two vertical sides, outermost turns only) and
#              ``wall_wrap``'s two rim loops swept into the shell (``_RIM_*``).
#   inlet / outlet  the fluid openings -- ``core_hex``'s two z caps
#              (``first/last_tag=_END_*``), the four inter-turn fills' helix ends
#              (the ``between`` slices' edge tags riding onto the loft) and
#              ``wall_wrap``'s two rim loops swept into the film (``_RIM_*``).
#
# Whatever fluid boundary is left bare is a real gap in the model -- the film and
# the inter-turn fills are helical ribbons, not closed annuli, so their lateral
# sides are exposed -- and it is left bare rather than guessed at.
GROUPS = {
    "interface": {"fluid": "W  ", "solid": None},
    "inlet":  "v  ",
    "outlet": "O  ",
    "outer":  "I  ",
    "cut":    "I  ",
}

mesh = hexmesh.retag_face(assembly, {_END_LO: "inlet", _END_HI: "outlet"})

_reg = mesh.element_tags.dense(mesh.n_hexes)
_inc = np.asarray(mesh.hexes)
_nqf = mesh.quad_mesh.n_quads
_owner = np.full((_nqf, 2), -1, np.int64)
_slot = np.zeros(_nqf, np.int64)
for _e in range(mesh.n_hexes):
    for _q in _inc[_e]:
        _owner[_q, _slot[_q]] = _e
        _slot[_q] += 1
_bnd = _bfid(mesh)
_built = np.asarray(mesh.face_tags.dense(_nqf))
_solid_face = _reg[_owner[:, 0]] == "solid"

_conjugate = ((~_bnd) & (_owner[:, 1] >= 0)
              & (_reg[_owner[:, 0]] != _reg[_owner[:, 1]]))
assert set(np.flatnonzero(_conjugate)) == set(hexmesh.tagged_faces(mesh, IFACE)), (
    "the conjugate surface named at construction is not the assembled topology's")
assert not (_bnd & _solid_face & (_built == "")).any(), (
    "a solid boundary face is unnamed -- every one is the pipe skin or a saw cut, "
    "both set by construction")

_bare = int((_bnd & (_built == "")).sum())
if _bare:
    print("note: %d fluid boundary faces left bare (helical ribbons are not closed "
          "laterally)" % _bare)

# === report + export -----------------------------------------------------------
_q = hexmesh.quality_summary(mesh)
print("wire coil: %d hex elements, %d points" % (mesh.n_hexes, mesh.n_points))
print("scaled Jacobian: min=%.4f mean=%.4f  inverted=%d"
      % (_q.min, _q.mean, _q.n_inverted))
print("watertight:", hexmesh.is_watertight(mesh),
      " conforming:", hexmesh.is_conforming(mesh))
print("regions:", ", ".join(sorted(mesh.element_tags.group_tags)))
print("faces:  ", ", ".join(sorted(mesh.face_group_tags)))

writer.to_re2(mesh, "wire_coil.re2", groups=GROUPS)
writer.to_vtu(mesh, "wire_coil.vtu", groups=GROUPS)
