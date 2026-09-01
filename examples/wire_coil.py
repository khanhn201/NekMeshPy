"""A helically coiled wire inside a round pipe, meshed for conjugate heat transfer.

The domain has three regions.  The **wire** is a continuous helix: a skewed
rectangular ``template`` wraps the axial core ``N_LAYERS`` rows per pitch, its
cross-section ``coil_template`` is blended through the wire disc by ``coil_arc_map``
(``L = 1`` the ``INNER_ARC_DEG`` inward arc, ``L = -1`` the ``OUTER_ARC_DEG`` outward
one, in between the arc angle interpolates *and* the radius pulls off the wall by
``COIL_SHRINK`` -- an o-grid butterfly without calling ``ogrid``), and each turn is
lofted and welded to the next.  The **solid pipe wall** is an annular shell,
``R_TUBE .. R_TUBE + WALL_THICK``.  Everything else -- the film between the wire and
the wall, the wedge between consecutive turns (``lower`` / ``higher`` / ``cap_lo`` /
``cap_hi`` lofted against a staircase ``mi_sheet``), the ``branch`` from the core out
to the wire's inward arc, and the axial ``core`` -- is **fluid**.

Every block is a ``transform_fn`` warp of a flat template plus a ``loft``, welded with
``merge`` where its faces coincide node-for-node.  Faces are named at construction and
carried across the welds: ``coil`` for the wire's own conjugate fluid/solid surface,
``wall`` for the pipe wall's -- two distinct physical seams -- ``outer`` on the pipe
skin, and four groups on the two axial ends: ``inlet`` / ``outlet`` for the fluid,
``cut_lo`` / ``cut_hi`` for the solid saw cut.

Two of those groups are named on a *whole* surface and then trimmed by the weld:
``core``'s entire outer cylinder is ``inlet`` and the last turn's ``branch`` cap is
``outlet``, and ``merge(..., clear_seam_tags=["inlet", "outlet"])`` drops those names
off every face the helical band buries -- what stays named is exactly the wedge each
band fails to cover.

The cell is ``TURNS_WIRE`` **whole** pitches of the helix, which makes those two ends a
periodic pair under a pure axial translation by ``LEAD`` -- the screw symmetry
``(theta + 2*pi*n, z + n*PITCH)`` has no rotation left in it at integer ``n``.  So all
four end groups export as Nek ``P`` and ``periodic=`` states the two pairings; nothing
here is an *opening*.  A periodic channel has no driver, so a run off this mesh needs a
body force (or a fixed flow rate) in the ``.usr``.

The export names ``fluid=`` too: Nek's conjugate heat transfer needs the fluid
(velocity-mesh) elements listed first and separately counted from the solid
(temperature-only) ones (``nelgv`` vs ``nelgt`` in the ``.re2`` header), which is a
statement about the *element order* in the file, not just their tags -- so ``to_re2``
reorders the written bytes to put every ``"fluid"`` element first.  ``nelgv < nelgt``
also means Nek's reader expects a **second** boundary block -- one per solved field --
so ``GROUPS`` (velocity, fluid-only) and ``THERMAL`` (temperature, every element) are
two separate tables rather than one.

    PYTHONPATH=. python examples/wire_coil.py

Produces ``wire_coil.re2``, ``wire_coil.vtu`` and ``wire_coil.f00000``.
"""
import numpy as np

from nekmeshpy import hexmesh, linemesh, quadmesh, writer
from nekmeshpy.core import affine
from nekmeshpy.core.tags import ElementTags

R_HELIX = 0.375
RW_NOM  = 0.125
CLEAR   = 0.004
RW      = RW_NOM - CLEAR
PITCH     = (1.0 / 6.0) / 0.375

R_TUBE  = 0.5
WALL_THICK, N_WALL = 0.10, 3

R_CORE = R_HELIX - RW - 0.13
CORE_ZSHIFT = 0.1       # manual axial shift of the core relative to the coil

TURNS_WIRE = 2
LEAD = PITCH * TURNS_WIRE   # periodic cell = whole pitches, so the helix screw
                           # (theta + 2*pi*n, z + n*PITCH) is a pure axial shift
N_THETA = 24
N_LAYERS = 3            # template v-rows per pitch == core z-layers/pitch

COIL_TEMPLATE_SKEW = 0.1
COIL_TEMPLATE_FRAC = 0.8


INNER_ARC_DEG = 130.0
OUTER_ARC_DEG = 110.0

Ls = [-1, -0.7, 0.0, 0.7, 1]   # Latitude to mesh the coil


ORDER = 2


assert N_THETA % N_LAYERS == 0

# the whole outer cylinder is named "inlet"/"outlet" (caps + wall); the helical band
# buries all of it except the wedge the rising staircase leaves bare at each end, and
# merge(clear_seam_tags=...) drops the buried names.
core = linemesh.circle(R_CORE, N_THETA, order=ORDER)
core = quadmesh.ogrid(core, N_THETA//4, 1, wall_tag="inlet")
core = hexmesh.extrude(core, PITCH*TURNS_WIRE, N_LAYERS*TURNS_WIRE,
                       first_tag="inlet", last_tag="outlet")   # z=0 bottom / z=top
core = hexmesh.translate(core, (0.0, 0.0, -CORE_ZSHIFT))

# Template
# x: [0,1] map to [0,2pi*r]
# y: [0,1] map to [0,PITCH]
template = []
_nx = N_THETA // N_LAYERS
for i in range(N_LAYERS):
    rect = quadmesh.rectangle(
        [(i/N_LAYERS, i/N_LAYERS, 0.0),
         ((i+1)/N_LAYERS, i/N_LAYERS, 0.0),
         ((i+1)/N_LAYERS, i/N_LAYERS+1, 0.0),
         (i/N_LAYERS, i/N_LAYERS+1, 0.0)],
        _nx, N_LAYERS, order=ORDER,
        side_tags={"left": "u0"} if i == 0 else None)
    if i == N_LAYERS - 1:
        # u1 = last rect's right column, minus the lowest element (periodic seam)
        rect = quadmesh.tag_edges(
            rect, np.array([[(_nx - 1)*N_LAYERS + j, 2] for j in range(1, N_LAYERS)]),
            "u1")
    template.append(rect)
template = quadmesh.merge(template)


def staircase_line(v0):
    """The template's bottom staircase lifted to ``v0`` -- ``v0=0`` traces the W
    (``coil_template``'s low edge), ``v0=1`` the M -- as one LineMesh, its
    N_THETA//N_LAYERS-per-run horizontals and single-element risers matching the
    template edges so the same transform lands it on ``coil_template``'s boundary."""
    nx = N_THETA // N_LAYERS
    pts = [(0.0, v0, 0.0)]
    for i in range(N_LAYERS):
        u0, u1, v = i/N_LAYERS, (i+1)/N_LAYERS, v0 + i/N_LAYERS
        pts += [(u0 + (u1 - u0)*j/nx, v, 0.0) for j in range(1, nx + 1)]
        pts.append((u1, v + 1.0/N_LAYERS, 0.0))
    return linemesh.loft(np.array(pts), order=ORDER)


def wrap(turn):
    def fn(P):
        P = np.asarray(P, float).reshape(-1, 3)
        u, v = P[:, 0], P[:, 1]
        th = 2.0 * np.pi * u
        r = R_CORE
        z = (v + turn) * PITCH - CORE_ZSHIFT
        return np.column_stack([r * np.cos(th), r * np.sin(th), z])
    return fn

core_side = [quadmesh.transform_fn(template, wrap(k)) for k in range(TURNS_WIRE)]



layer_height = 1.0/N_LAYERS
def coil_template_fn(P):
    P = np.asarray(P, float).reshape(-1, 3)
    u, v, w = P[:, 0], P[:, 1], P[:, 2]
    v += -1.0*u                            # Line up the elements on both side, shifted by one layer.
    v += -0.5 + layer_height*0.5           # Recenter on v=0
    v  = v/(0.5+layer_height*0.5)          # Rescale to v=[-1,1]
    u += v*COIL_TEMPLATE_SKEW              # Skew in u
    v  = v*COIL_TEMPLATE_FRAC              # Rescale to v=[-1,1]*COIL_TEMPLATE_FRAC
    return np.column_stack([u,v,w])
def coil_template_bound(y): # Project to top or bottom horizontal line
    def fn(P):
        P = np.asarray(P, float).reshape(-1, 3)
        u, v, w = P[:, 0], P[:, 1], P[:, 2]
        v = np.full(u.shape, y)
        return np.column_stack([u,v,w])
    return fn
coil_template_inner = quadmesh.transform_fn(template, coil_template_fn)
lo_line = linemesh.transform_fn(staircase_line(0.0), coil_template_fn)
hi_line = linemesh.transform_fn(staircase_line(1.0), coil_template_fn)
mi_line = linemesh.transform_fn(staircase_line(0.5), coil_template_fn)
lo_bound = linemesh.transform_fn(lo_line, coil_template_bound(-1.0))
hi_bound = linemesh.transform_fn(hi_line, coil_template_bound(1.0))

def _u_tag(qm, bottom_last=False):
    """tag a 1-layer u-swept strip's ends: quad 0 side 4 -> u0, last quad side 2 -> u1.
    ``bottom_last`` also tags the last quad's side 1 (its inner/hi_line face) u1 -- the
    top band's u1 boundary runs along that skewed profile edge, not just the end."""
    n = qm.n_quads - 1
    rows, tags = [[0, 4], [n, 2]], ["u0", "u1"]
    if bottom_last:
        rows.append([n, 1])
        tags.append("u1")
    return quadmesh.tag_edges(qm, np.array(rows), tags)


# coil_template's u=0 / u=1 boundary: inner arc-blend's left/right column (last rect's
# lowest element dropped) + each band's first (u0) / last (u1) edge, plus the top band's
# last-element inner face.  renamed per turn where used (coil's L-loop; cleared on
# branch's core_side/coil_inner).
coil_template = quadmesh.merge([coil_template_inner,
                                _u_tag(quadmesh.loft([lo_bound, lo_line])),
                                _u_tag(quadmesh.loft([hi_line, hi_bound]), bottom_last=True)])


# --- map coil_template onto the wire's cross-section: v -> along an arc, L -> through
# the disc.  L=1 is the INNER_ARC_DEG inward arc, L=-1 the OUTER_ARC_DEG outward arc,
# in between the arc angle blends AND the radius pulls off the RW wall toward the
# centre (COIL_SHRINK), an o-grid butterfly without calling ogrid.
RISE = PITCH / (2.0 * np.pi)
INNER_HALF = np.deg2rad(INNER_ARC_DEG / 2.0)
OUTER_HALF = np.deg2rad(OUTER_ARC_DEG / 2.0)
COIL_SHRINK = 0.30
COIL_KNEE = 0.7        # |L| at which the shrink is fully on; flat between the knees


def coil_arc_map(turn, L=1.0):
    def fn(P):
        P = np.asarray(P, float).reshape(-1, 3)
        u, v = P[:, 0], P[:, 1]
        gu = turn + u
        th = gu * 2.0 * np.pi
        b_in = np.pi - v * INNER_HALF                    # inner arc point (about -R)
        b_out = v * OUTER_HALF                           # outer arc point (about +R)
        s = 0.5 * (1.0 + L)                              # 1 -> inner wall, 0 -> outer
        x = s * RW * np.cos(b_in) + (1.0 - s) * RW * np.cos(b_out)
        y = s * RW * np.sin(b_in) + (1.0 - s) * RW * np.sin(b_out)
        # trapezoid: 0 shrink at the walls, ramps up over |L| in [KNEE,1], flat within
        t = np.clip((1.0 - abs(L)) / (1.0 - COIL_KNEE), 0.0, 1.0)
        bow = 1.0 - COIL_SHRINK * t
        rr = R_HELIX + bow * x
        z = RISE * th + bow * y
        return np.column_stack([rr * np.cos(th), rr * np.sin(th), z])
    return fn


def _coil_sec(k):
    # coil_template's u0/u1 -> cut_lo on turn 0, cut_hi on the last turn, else dropped
    return quadmesh.retag_edge(coil_template, {"u0": "cut_lo" if k == 0 else "",
                                               "u1": "cut_hi" if k == TURNS_WIRE - 1 else ""})


coil = hexmesh.merge([hexmesh.loft(
    [quadmesh.transform_fn(_coil_sec(k), coil_arc_map(k, L)) for L in Ls],
    element_tags="solid")
    for k in range(TURNS_WIRE)], tol=1e-9)


# -- top / bottom caps: one hex layer from the coil's shrunk-core v=+-1 face (rims at
# Ls[1], Ls[-2]) out to the RW wall over the wedge.  inner reuses coil_arc_map so it
# welds to coil's own middle-layer face; outer rides the RW wall arc (beta swept
# OUTER_HALF -> pi-INNER_HALF), curved via an order-2 midside so the cap follows the
# coil surface instead of chording flat.  (Assumes Ls = [-1, -k, k, 1].)
def _rim(vline, k, L):
    return linemesh.transform_fn(vline, coil_arc_map(k, L))


def _cap_wall(vline, k, sgn, tau):
    b0, b1 = sgn * OUTER_HALF, sgn * (np.pi - INNER_HALF)
    beta = 0.5 * (1.0 - tau) * b0 + 0.5 * (1.0 + tau) * b1

    def fn(P):
        th = (k + np.asarray(P, float).reshape(-1, 3)[:, 0]) * 2.0 * np.pi
        rr = R_HELIX + RW * np.cos(beta)
        return np.column_stack([rr * np.cos(th), rr * np.sin(th),
                                RISE * th + RW * np.sin(beta)])
    return linemesh.transform_fn(vline, fn)


_NL_CAP = len(Ls) - 3   # radial layers in a _cap_outer strip


def _u_tag_layers(qm, nl, k, lo_tag, hi_tag):
    """layer-major strip: tag EVERY layer's u=0 edge (turn 0) / u=1 edge (last turn)."""
    nu = qm.n_quads // nl
    rows, tags = [], []
    if k == 0:
        rows += [[i * nu, 4] for i in range(nl)]
        tags += [lo_tag] * nl
    if k == TURNS_WIRE - 1:
        rows += [[i * nu + nu - 1, 2] for i in range(nl)]
        tags += [hi_tag] * nl
    return quadmesh.tag_edges(qm, np.array(rows).reshape(-1, 2), tags) if rows else qm


def _cap_outer(vline, k, sgn):
    # one wall point per interior latitude, tau stretched so Ls[1] -> -1, Ls[-2] -> +1:
    # the outer edge spans the WHOLE wedge (beta b0..b1) with its ends on coil's L=+-1
    # wall layers.  sweep_nodes puts each segment's order-2 midside on the RW arc too,
    # so the outer face is projected to the coil surface, not chorded flat.
    _tau = [l / Ls[-2] for l in Ls[1:-1]]
    _mid = [0.5 * (a + b) for a, b in zip(_tau[:-1], _tau[1:])]
    return quadmesh.loft([_cap_wall(vline, k, sgn, t) for t in _tau],
                         sweep_nodes=[[_cap_wall(vline, k, sgn, m)] for m in _mid])


def _cap(vline, k, sgn):
    inner = _u_tag_layers(quadmesh.loft([_rim(vline, k, l) for l in Ls[1:-1]]),
                          _NL_CAP, k, "cut_lo", "cut_hi")
    outer = _u_tag_layers(_cap_outer(vline, k, sgn), _NL_CAP, k, "cut_lo", "cut_hi")
    return hexmesh.loft([inner, outer], element_tags="solid")


coil = hexmesh.merge(
    [coil]
    + [_cap(hi_bound, k, 1.0) for k in range(TURNS_WIRE)]
    + [_cap(lo_bound, k, -1.0) for k in range(TURNS_WIRE)], tol=1e-9)

coil_inner = [quadmesh.transform_fn(coil_template_inner, coil_arc_map(k))
                       for k in range(TURNS_WIRE)]

# -- Mesh branch from core to inner arc
_clr_u = {"u0": "", "u1": ""}   # face_wrap doesn't use the wrap-only helix tags


def _branch_sec(qm, k):
    # the inner's u0 (turn 0) / u1 (last turn) edges promote to branch's inlet / outlet
    return quadmesh.retag_edge(qm, {"u0": "inlet" if k == 0 else "",
                                    "u1": "outlet" if k == TURNS_WIRE - 1 else ""})


# the last turn's band rises a pitch past the core top; name its whole core-side cap
# "outlet" and let the final merge bury the part that welds onto the core, leaving
# only the overhang tagged.
branch = hexmesh.merge([
    hexmesh.loft([_branch_sec(core_side[k], k), _branch_sec(coil_inner[k], k)],
                 first_tag="outlet" if k == TURNS_WIRE - 1 else None,
                 last_tag="coil")
    for k in range(TURNS_WIRE)])




# -- MID SHEET --
mi_arc = [linemesh.translate(linemesh.transform_fn(mi_line, coil_arc_map(k, Ls[-2])),
                             (0.0, 0.0, PITCH / 2.0))
          for k in range(-1,TURNS_WIRE)]
mi_core = [linemesh.transform_fn(staircase_line(1.0), wrap(k)) for k in range(-1,TURNS_WIRE)]


def _R_at(turn, L):
    """mean radius from the z axis of the coil surface along mi_line at latitude L."""
    p = linemesh.transform_fn(mi_line, coil_arc_map(turn, L)).points
    return float(np.hypot(p[:, 0], p[:, 1]).mean())


def _xy_scale(line, f):
    return linemesh.transform_fn(
        line, lambda P: np.column_stack([P[:, 0] * f, P[:, 1] * f, P[:, 2]]))


# core -> mi_arc (matches Ls[-1] = 1), then one more layer per strictly-interior
# latitude (Ls[-3:0:-1]): mi_arc xy-scaled to the coil surface radius at that L.
_MI_LS = Ls[-3:0:-1]
mi_sheet = quadmesh.merge([quadmesh.loft(
    [mi_core[k], mi_arc[k]]
    + [_xy_scale(mi_arc[k], _R_at(k - 1, L) / _R_at(k - 1, Ls[-2])) for L in _MI_LS])
    for k in range(TURNS_WIRE + 1)], tol=1e-9)


# --- lower fill, step 1: mi_sheet's first layer (mi_core -> mi_arc) lofted to the
# lo band (quadmesh.loft([lo_bound, lo_line])) mapped onto the coil surface at L=1.
_lo_band = _u_tag(quadmesh.loft([lo_bound, lo_line]))
_hi_band = _u_tag(quadmesh.loft([hi_line, hi_bound]), bottom_last=True)
lower = hexmesh.merge([
    hexmesh.loft([_branch_sec(quadmesh.transform_fn(_lo_band, coil_arc_map(k, 1.0)), k),
                  quadmesh.loft([mi_arc[k], mi_core[k]])],
                 first_tag="coil", last_tag="inlet" if k == 0 else None)
    for k in range(TURNS_WIRE)], tol=1e-9)
higher = hexmesh.merge([
    hexmesh.loft([_branch_sec(quadmesh.transform_fn(_hi_band, coil_arc_map(k, 1.0)), k),
                  quadmesh.loft([mi_core[k+1], mi_arc[k+1]])],
                 first_tag="coil", last_tag="outlet" if k == TURNS_WIRE - 1 else None)
    for k in range(TURNS_WIRE)], tol=1e-9)


# -- bottom cap of the coil lofted to mi_sheet's OUTER layers (mi_arc -> scaled@_MI_LS,
# same section count as the cap's outer face), like lower/higher.
def _mi_scaled(k, L):
    return _xy_scale(mi_arc[k], _R_at(k - 1, L) / _R_at(k - 1, Ls[-2]))


def _mi_outer(k):
    return quadmesh.loft(list(reversed(
        [mi_arc[k]] + [_mi_scaled(k, L) for L in _MI_LS])))


cap_lo = hexmesh.merge([
    hexmesh.loft([_u_tag_layers(_cap_outer(lo_bound, k, -1.0), _NL_CAP, k, "inlet", "outlet"),
                  _mi_outer(k)],
                 first_tag="coil", last_tag="inlet" if k == 0 else None)
    for k in range(TURNS_WIRE)], tol=1e-9)
cap_hi = hexmesh.merge([
    hexmesh.loft([_u_tag_layers(_cap_outer(hi_bound, k, 1.0), _NL_CAP, k, "inlet", "outlet"),
                  _mi_outer(k + 1)],
                 first_tag="coil", last_tag="outlet" if k == TURNS_WIRE - 1 else None)
    for k in range(TURNS_WIRE)], tol=1e-9)






# face_wrap: (1) the coil's outward OUTER_ARC_DEG surface (L = Ls[0]) + (2) the
# OUTWARD lateral side of cap_lo / cap_hi -- the loft's tau = _TAU[0] (beta = b0,
# the outer-arc-end) face, from _cap_wall out to _mi_outer's outermost profile.
_TAU = [l / Ls[-2] for l in Ls[1:-1]]
face_wrap = quadmesh.merge(
    [quadmesh.transform_fn(quadmesh.retag_edge(coil_template, _clr_u), coil_arc_map(k, Ls[0]))
     for k in range(TURNS_WIRE)]
    + [quadmesh.loft([_mi_scaled(k, _MI_LS[-1]), _cap_wall(lo_bound, k, -1.0, _TAU[0])])
       for k in range(TURNS_WIRE)]
    + [quadmesh.loft([_cap_wall(hi_bound, k, 1.0, _TAU[0]), _mi_scaled(k + 1, _MI_LS[-1])])
       for k in range(TURNS_WIRE)], tol=1e-9)


# face_wrap's boundary is two helix-cut loops; tag them so film/pipe promote them
# to lateral faces.  higher-z loop -> outlet, lower -> inlet (fluid film ends); pipe
# renames its copies to cut_hi / cut_lo below.
def _tag_wrap_loops(qm):
    be = quadmesh.boundary_edges(qm)                             # (N,2) quad, side
    lid = np.asarray(qm.quads)[be[:, 0], be[:, 1] - 1]
    ends = np.asarray(qm.line_mesh.lines)[lid]                   # (N,2) node ids
    parent = np.arange(qm.line_mesh.n_lines)

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for a, b in ends:
        parent[find(a)] = find(b)
    root = np.array([find(a) for a in ends[:, 0]])
    zmean = {r: qm.points[ends[root == r].ravel(), 2].mean() for r in np.unique(root)}
    hi_root = max(zmean, key=zmean.get)
    tags = np.where(root == hi_root, "outlet", "inlet")
    return quadmesh.tag_edges(qm, be, list(tags))


face_wrap = _tag_wrap_loops(face_wrap)


# --- fluid film: face_wrap -> its radial projection onto the pipe wall (R_TUBE),
# then the solid pipe wall (R_TUBE -> R_TUBE + WALL_THICK, N_WALL layers).
def _to_cyl(R):
    def fn(P):
        P = np.asarray(P, float).reshape(-1, 3)
        s = R / np.hypot(P[:, 0], P[:, 1])
        return np.column_stack([P[:, 0] * s, P[:, 1] * s, P[:, 2]])
    return fn


_wrap_tube = quadmesh.transform_fn(face_wrap, _to_cyl(R_TUBE))
# face_wrap's first TURNS_WIRE blocks are the coil's OUTER_ARC surface (order-preserved
# by merge); the film's face on that part is the coil conjugate interface.
_wrap_coil = np.full(face_wrap.n_quads, "", dtype="<U8")
_wrap_coil[:TURNS_WIRE * coil_template.n_quads] = "coil"
film = hexmesh.loft([face_wrap, _wrap_tube],
                    first_tag=ElementTags.from_dense(_wrap_coil),
                    last_tag="wall")   # R_TUBE: fluid film <-> pipe conjugate face
pipe = hexmesh.retag_face(hexmesh.loft(
    [quadmesh.transform_fn(face_wrap, _to_cyl(R_TUBE + WALL_THICK * i / N_WALL))
     for i in range(N_WALL + 1)], element_tags="solid", last_tag="outer"),
    {"inlet": "cut_lo", "outlet": "cut_hi"})

mesh = hexmesh.merge([coil, branch, core, lower, higher, cap_lo, cap_hi, film, pipe],
                     tol=1e-9, clear_seam_tags=["inlet", "outlet"])

# everything not tagged solid at construction is fluid
_reg = np.asarray(mesh.element_tags.dense(mesh.n_hexes), dtype="<U8")
_reg[_reg == ""] = "fluid"
mesh = hexmesh.HexMesh(mesh.quad_mesh, mesh.hexes, mesh.orient, mesh.interior,
                       ElementTags.from_dense(_reg))

# --- conjugate-surface check: the coil/wall faces named at construction must be
# exactly the assembled topology's fluid<->solid interfaces, nothing more or less.
_inc = np.asarray(mesh.hexes)
_owner = np.full((mesh.quad_mesh.n_quads, 2), -1, np.int64)
_slot = np.zeros(mesh.quad_mesh.n_quads, np.int64)
for _e in range(mesh.n_hexes):
    for _q in _inc[_e]:
        _owner[_q, _slot[_q]] = _e
        _slot[_q] += 1
_reg2 = np.asarray(mesh.element_tags.dense(mesh.n_hexes))
_conj = ((_owner[:, 1] >= 0) & (_reg2[_owner[:, 0]] != _reg2[_owner[:, 1]]))
_named = set(hexmesh.tagged_faces(mesh, "coil")) | set(hexmesh.tagged_faces(mesh, "wall"))
assert set(np.flatnonzero(_conj)) == _named, \
    "coil/wall named at construction != the assembly's fluid/solid interfaces"

# --- Nek export.  The two axial ends are a periodic pair under a pure z-shift by
# LEAD (whole pitches), so inlet/outlet and the solid saw cuts cut_lo/cut_hi all
# carry 'P  ' rather than an opening.  GROUPS is the velocity field (fluid only):
# coil/wall are walls on the fluid side and write nothing on the solid side; the
# fluid ends are periodic.  THERMAL is temperature (every element): the conjugate
# surfaces stay conformal ('E  ', omitted), the periodic ends carry heat around on
# both the fluid pair and the solid cut, and only the pipe's outer skin is a real
# (insulated) boundary.
GROUPS = {
    "coil": {"fluid": "v  ", "solid": None},
    "wall": {"fluid": "W  ", "solid": None},
    "inlet":  "P  ",
    "outlet": "P  ",
}
PERIODIC = [
    hexmesh.Periodic("inlet", "outlet", affine.translation([0.0, 0.0, LEAD])),
    hexmesh.Periodic("cut_lo", "cut_hi", affine.translation([0.0, 0.0, LEAD])),
]
THERMAL = {
    "inlet":  "P  ",
    "outlet": "P  ",
    "cut_lo": "P  ",
    "cut_hi": "P  ",
    "outer":  "f  ",
}

print(hexmesh.report(mesh))
print("regions:", dict(zip(*np.unique(mesh.element_tags.dense(mesh.n_hexes),
                                      return_counts=True))))
print("faces  :", ", ".join(sorted(mesh.face_tags.group_tags)))

writer.to_re2(mesh, "wire_coil.re2", groups=GROUPS, periodic=PERIODIC,
              fluid="fluid", thermal=THERMAL)
writer.to_vtu(mesh, "wire_coil.vtu", groups={**THERMAL, **GROUPS})
writer.to_fld(mesh, "wire_coil.f00000", fluid="fluid")
