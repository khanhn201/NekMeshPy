"""Unequal-radius T-junction by the **cob** construction: the branch cut straight
through the main pipe.

The quadrant construction (``quadrant_pipe_tjunction.py``) radiates seams from a hub, and
at a small radius ratio its crotch cap degenerates into a pointy wedge deep inside the
domain -- at ratio 0.26 it measures minSJ **-0.044 with 2 inverted elements**. This one
has no hub at all, so there is nothing to degenerate.

The construction, in the main pipe's own cross-section:

* the **cob** is the middle ``N_THETA_BRANCH`` elements of the section -- a square block
  whose perimeter is exactly ``N_THETA_BRANCH`` edges, which is what lets the branch bore
  attach to it one-for-one;
* the cob is **walked** wall-to-wall, element to element, leaving each quad by the side
  opposite the one entered.  That band is the branch's shadow through the pipe.  Walking
  rather than selecting on a coordinate threshold is what keeps it exactly the cob's width
  the whole way, including through the curved O-grid rings at the wall;
* the band is removed and the rest of the section extruded in ``z``, leaving a slot;
* the branch's cross-section is meshed **in the cylinder's own (arc, z) parameter space**,
  so every node of it lands exactly on the wall and the bore is exactly the analytic
  cylinder-cylinder intersection.  That top section is then mapped down the band's
  horizontal cuts -- which shrink, bend, and open out again -- and lofted into the slot.

Because the slot's cuts are rows of the section itself, the block's ``z`` faces come back
**bit-identical** to the main pipe's cross-section, so the pipe carries on from either end
with a plain extrude and no transition.

    PYTHONPATH=. python examples/cob_tjunction.py

Produces ``cob_tjunction.re2`` and ``.vtu``.
"""

import logging
from collections import defaultdict

import numpy as np

from nekmeshpy import (
    ElementTags,
    LineMesh,
    QuadMesh,
    hexmesh,
    linemesh,
    quadmesh,
    writer,
)
from nekmeshpy.core.fields import gll_nodes, lagrange_matrix
from nekmeshpy.hexmesh import Seam
from nekmeshpy.pointmesh import PointMesh
from nekmeshpy.quadmesh import Seam as EdgeSeam
from nekmeshpy.quadmesh.query import element_blocks

logging.basicConfig(level=logging.INFO, format="%(message)s")

# -- parameters ---------------------------------------------------------------
R_MAIN = 0.5                  # main pipe radius (axis Z)
N_THETA_MAIN = 32             # main pipe azimuthal cells -- sets the wall cell size
RADIAL_MAIN = 3               # main pipe O-grid radial layers
CENTER_SCALE_MAIN = 0.8       # main pipe O-grid hub placement
R_BRANCH = 0.1322             # branch bore radius (axis +Y); ratio 0.264
N_THETA_BRANCH = 16           # branch azimuthal cells; MULTIPLE OF 4, independent of main
CENTER_SCALE_BRANCH = 0.8     # branch O-grid hub placement
#: Branch O-grid radial stations.  Two layers, so the bore is not thrown straight at the
#: square in one step: the extra ring interpolates between the cob's centre and the slot
#: it has to land in, which keeps the wedges that reach the square corners from spanning
#: the whole transition on their own.
RADIAL_BRANCH = np.array([0.0, 1.0])
Z_DOMAIN = 3.0                # main pipe runs z in [-Z_DOMAIN, +Z_DOMAIN]
H_BRANCH = 1.5                # branch tip at y = H_BRANCH
N_Z_LEG = 14                  # hex layers per main-pipe leg
N_BRANCH = 10                 # hex layers along the branch
#: Boundary-layer stations as a fraction of ``R_MAIN``.  The core is meshed at the first
#: one and :func:`quadmesh.offset <nekmeshpy.quadmesh.morph.offset>` skins it out to the
#: rest, so the wall lands on ``R_MAIN`` exactly.
#:
#: **The first station is bounded by the geometry, not by taste.**  An offset can only
#: move a surface by less than its own local feature size, and the smallest feature on
#: this wall is not on the pipe -- it is the bore imprint the cut leaves on the *far*
#: wall, whose cells run about ``RC_BRANCH / 4``.  ``0.8`` asks for a skin of ``0.1``
#: against cells of ``0.009`` and folds four of them inside out; ``0.9`` (a skin of
#: ``0.05``, ~4x the smallest cell) now folds four elements at the branch root, and
#: ``0.95`` is the thickest of this family that comes back clean.
R_SKIN = np.array([0.95, 0.975, 1.0])
ORDER = 2
OUT_NAME = "cob_tjunction"

GROUPS = {"wall": "W  ", "inlet": "v  ", "outlet": "O  ", "branch": "O  "}

BRANCH_AXIS = np.array([0.0, 1.0, 0.0])
NSIDE = N_THETA_BRANCH // 4               # cells per side of the footprint square
# The skin is a *uniform thickness* offset along the wall normal, not a scaling, so the
# core is the finished geometry inset by that thickness -- both radii, since the branch
# wall is skinned too.  That is what puts the final wall on R_MAIN / R_BRANCH exactly.
T_BL = (1.0 - R_SKIN[0]) * R_MAIN
RC_MAIN = R_MAIN - T_BL
RC_BRANCH = R_BRANCH - T_BL
# the slot is as long in z as the cob's top arc is wide, so it is square seen from above
L = NSIDE * (2.0 * np.pi * RC_MAIN / N_THETA_MAIN)


def map_section(qm, fn):
    """``qm``'s topology with **every** node table pushed through ``fn``.

    A section owns three of them -- the shared corners, the shared edges' interior nodes,
    and its own per-quad interior -- and at ``order > 1`` all three carry geometry.
    Mapping only ``points`` leaves the other two holding whatever the template had, which
    is why ``merge`` then rejects the block: two elements meeting on a shared edge
    disagree about where that edge's interior nodes are.
    """
    def m(a):
        return fn(a.reshape(-1, 3)).reshape(a.shape) if a.size else a

    lm = qm.line_mesh
    return QuadMesh(LineMesh(PointMesh(fn(qm.points), lm.point_tags), lm.lines,
                             m(lm.interior), lm.element_tags),
                    qm.quads, qm.orient, m(qm.interior), qm.element_tags)


# -- the main pipe cross-section, and the cob's band through it ----------------
section = quadmesh.ogrid(linemesh.circle(RC_MAIN, N_THETA_MAIN, order=ORDER),
                         N_THETA_MAIN // 4, RADIAL_MAIN,
                         center_scale=CENTER_SCALE_MAIN)
quads, pts, lines = section.quads, section.points, section.line_mesh.lines
P = pts[section.corners]

edge2q = defaultdict(list)
for q in range(section.n_quads):
    for s in range(4):
        edge2q[int(quads[q, s])].append((q, s))


def walk(e, from_q):
    """Straight run of elements: enter through edge ``e``, leave by the opposite side,
    repeat until the wall.  Returns ``(elements, the edges crossed)``."""
    el, ed = [], [e]
    while True:
        nxt = [(q, s) for (q, s) in edge2q[e] if q != from_q]
        if not nxt:
            return el, ed
        q, s = nxt[0]
        el.append(q)
        e, from_q = int(quads[q, (s + 2) % 4]), q
        ed.append(e)


cob = np.argsort(np.linalg.norm(P.mean(axis=1), axis=1))[:N_THETA_BRANCH]
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

# The slot's own boundary, named before the band is removed: an edge with one quad in the
# band and one outside is where the collar will meet the pipe.  ``quadmesh.remove`` leaves
# the faces it exposes untagged, so naming them here is the only way the lateral seam is
# addressable at all -- the tag rides the surviving quad's edge through.
#
# It goes on a *copy*, not on ``section``: the legs are extruded from the same section and
# keep the band, so there the very same edges are interior.  Tagging them in place would
# name 784 faces that no seam ever consumes, and the exporter would write a boundary row
# for each -- boundary conditions in the middle of the pipe.
_slot_rows = np.array([[q, s + 1] for e, lst in edge2q.items() if len(lst) == 2
                       for (q, s) in lst
                       if (lst[0][0] in band_set) != (lst[1][0] in band_set)
                       and q not in band_set], dtype=np.int64)
slot_section = quadmesh.tag_edges(section, _slot_rows, "att_slot")

# the band's four columns, each walked from the wall it starts on right through
wall_edges = [e for e, lst in edge2q.items() if len(lst) == 1 and lst[0][0] in band_set]
foot = sorted(wall_edges, key=lambda e: float(pts[lines[e]].mean(axis=0) @ BRANCH_AXIS))
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


# The band as one node lattice, ``(nrow*ORDER+1, NSIDE*ORDER+1, 3)``.
#
# Cuts have to be available *between* element boundaries, not only on them.  ``loft``
# straight-subdivides along its sweep, so at ORDER > 1 handing it only the boundary cuts
# drops every mid-node onto a chord of the band's own curved rows -- corner-clean, and
# inverted the moment the curved block is read.  Reading the whole lattice once makes a
# cut at any GLL level just a row of it, which is what ``sweep_nodes`` wants.
BLK = element_blocks(section).reshape(section.n_quads, ORDER + 1, ORDER + 1, 3)


def band_block(q, e_in):
    """Element ``q``'s nodes as ``[v, a]`` -- ``v`` up the band away from edge ``e_in``,
    ``a`` across it."""
    s = int(np.flatnonzero(quads[q] == e_in)[0])
    b = BLK[q]
    if s == 0:                                     # side 1 is j=0
        return b
    if s == 2:                                     # side 3 is j=n
        return b[::-1]
    if s == 3:                                     # side 4 is i=0
        return b.transpose(1, 0, 2)
    return b.transpose(1, 0, 2)[::-1]              # side 2 is i=n


LAT = np.empty((nrow * ORDER + 1, NSIDE * ORDER + 1, 3))
below = None
for k in range(nrow):
    row = np.empty((ORDER + 1, NSIDE * ORDER + 1, 3))
    where = {cols[c][1][k]: c for c in range(NSIDE)}
    for slot, (e, rev) in enumerate(chain_edges([cols[c][1][k] for c in range(NSIDE)])):
        B = band_block(cols[where[e]][0][k], e)
        # ``chain_edges`` says which way the cut walks this edge; the block's own ``a``
        # need not agree, and a tail-matching heuristic cannot orient the *first* one.
        head = pts[lines[e, 1]] if rev else pts[lines[e, 0]]
        if not np.allclose(B[0, 0], head, atol=1e-12):
            B = B[:, ::-1, :]
        row[:, slot * ORDER:(slot + 1) * ORDER + 1] = B
    if below is not None and not np.allclose(row[0], below, atol=1e-12):
        row = row[:, ::-1, :]                      # a whole row can walk the other way
    LAT[k * ORDER:(k + 1) * ORDER + 1] = row
    below = row[-1]


# run the lattice the way the top section does (increasing arc == decreasing x)
if LAT[0, -1, 0] > LAT[0, 0, 0]:
    LAT = LAT[:, ::-1, :]


def cut_at(level):
    """``(NSIDE, ORDER+1, 3)`` node blocks of the cut at sweep ``level``."""
    row = LAT[level]
    return np.stack([row[i * ORDER:(i + 1) * ORDER + 1] for i in range(NSIDE)])


GLL = gll_nodes(ORDER)


def on_cut(blk, a):
    """Evaluate a cut's piecewise Lagrange curve at ``a`` in ``[-1, 1]``.

    The section's own nodes sit at the GLL positions of a matching element run, so this
    lands on the cut's stored nodes exactly rather than interpolating near them."""
    t = (a + 1.0) / 2.0 * NSIDE
    i = np.clip(t.astype(int), 0, NSIDE - 1)
    M = lagrange_matrix(GLL, t - i)              # (P, order+1)
    return np.einsum("pk,pkc->pc", M, blk[i])


# -- the branch cross-section, meshed on the wall itself -----------------------
def foot_param(t):
    """The exact cylinder-cylinder intersection, in ``(arc s, z)`` parameter coords."""
    x = RC_BRANCH * np.sin(t)
    z = RC_BRANCH * np.cos(t)
    y = np.sqrt(RC_MAIN ** 2 - x ** 2)
    return np.stack([RC_MAIN * (np.arctan2(y, x) - np.pi / 2), np.zeros_like(t), z], axis=1)


def to_cyl(p):
    """``(arc s, z)`` -> the cylinder.  ``s`` is arc length, so the parameter domain is a
    true ``L x L`` square and the O-grid built in it is not distorted by the wrap."""
    phi = np.pi / 2 + p[:, 0] / RC_MAIN
    return np.stack([RC_MAIN * np.cos(phi), RC_MAIN * np.sin(phi), p[:, 2]], axis=1)


# the collar's lateral surface is this loop swept: two of its sides face the pipe across
# the slot, the other two are the collar's own ends where the legs butt against it
square = linemesh.rectangle(L, L, N_THETA_BRANCH, normal=BRANCH_AXIS, order=ORDER,
                            side_tags={"left": "att_slot", "right": "att_slot",
                                       "bottom": "att_endA", "top": "att_endB"})
# Pair the footprint with the square *angularly* -- one bore node per square node on the
# same ray from the centre.  Spacing the bore by arc length instead leaves the two loops
# out of phase and the annulus comes back folded.
td = np.linspace(0.0, 2.0 * np.pi, 4001)
fang = np.unwrap(np.arctan2(foot_param(td)[:, 2], foot_param(td)[:, 0]))
aim = np.mod(np.arctan2(square.points[:, 2], square.points[:, 0]) - fang[0],
             2.0 * np.pi) + fang[0]
# ``loft`` would straight-subdivide between the 16 corners and drop every high-order node
# onto a chord; ``loft_fn`` evaluates the intersection itself at the whole node lattice.
t_bore = np.unwrap(np.interp(aim, fang, td), period=2.0 * np.pi)
# A closed ``loft_fn`` wants the trailing wrap value too, so the seam element's own nodes
# get evaluated rather than closed with a chord -- but the run has to be *monotonic*
# first, and matching the square angularly hands it back descending.  Wrapping the wrong
# way makes the seam element span 5.96 rad instead of 0.32, and ``loft_fn`` dutifully
# places its interior node almost all the way round the bore.  Order 1 never sees it:
# only corners are placed, and those are right either way.
wrap = 2.0 * np.pi if t_bore[-1] > t_bore[0] else -2.0 * np.pi
bore_loop = linemesh.loft_fn(foot_param, np.append(t_bore, t_bore[0] + wrap),
                             loop=True, order=ORDER)

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
TOP = map_section(top_p, to_cyl)


def to_cut(blk):
    def fn(p):
        out = on_cut(blk, p[:, 0] / (L / 2.0))
        out[:, 2] = p[:, 2]
        return out
    return fn


slices = [map_section(top_p, to_cut(cut_at(k * ORDER))) for k in range(nrow)]
slices.append(TOP)                             # the top one sits exactly on the wall
# one intermediate profile per interior GLL level of every layer, so the sweep follows
# the band's rows instead of chording across them
inner = [[map_section(top_p, to_cut(cut_at(k * ORDER + m))) for m in range(1, ORDER)]
         for k in range(nrow)]
# TOP is the collar's far cap: its bore half is the branch's root, its rim half is wall
_top_tags = ElementTags.from_dense(
    np.array(["att_bore"] * bore_p.n_quads + [""] * collar_p.n_quads))
collar = hexmesh.loft(slices, sweep_nodes=inner if ORDER > 1 else None,
                      last_tag=_top_tags)

# -- the pipe around the slot, and the legs out to the domain ------------------
# a leg's cap meets *two* blocks: the pipe over the section, the collar over the band it
# removed.  One name for the whole cap will not do, and ``first_tag`` / ``last_tag`` take
# an ``ElementTags`` over the slice's own elements for exactly that.
_leg_cap = ElementTags.from_dense(
    np.where(np.isin(np.arange(section.n_quads), band), "att_band", "att_pipe"))

mid_pipe = hexmesh.extrude(quadmesh.remove(slot_section, band), length=L, layers=NSIDE,
                           axis=(0.0, 0.0, 1.0), origin=(0.0, 0.0, -L / 2),
                           first_tag="att_pipe_lo", last_tag="att_pipe_hi")
leg = Z_DOMAIN - L / 2
downstream = hexmesh.extrude(section, length=leg, layers=N_Z_LEG, axis=(0.0, 0.0, 1.0),
                             origin=(0.0, 0.0, L / 2), first_tag=_leg_cap,
                             last_tag="outlet")
upstream = hexmesh.extrude(section, length=leg, layers=N_Z_LEG, axis=(0.0, 0.0, 1.0),
                           origin=(0.0, 0.0, -Z_DOMAIN), first_tag="inlet",
                           last_tag=_leg_cap)

# -- the branch: the bore disc off the wall, straight out to the tip -----------
# the disc's own (x, z) are already the bore circle, so holding them and carrying y up to
# H_BRANCH sweeps an exact cylinder with a curved root and a flat cap.
# ``bore_p``'s wall was named for the section join above; the branch sweeps that same
# rim into its *outer* faces, which must stay untagged so the free-face sweep below
# names them ``wall`` and the skin grows over them.  A seam name that outlives its seam
# is a name for something that no longer exists.
bore_wall = map_section(quadmesh.retag_edge(bore_p, {"attach1": ""}), to_cyl)
flat = map_section(bore_wall, lambda p: np.stack(
    [p[:, 0], np.full(p.shape[0], H_BRANCH), p[:, 2]], axis=1))
stations = quadmesh.blend(bore_wall, flat, np.linspace(0.0, 1.0, N_BRANCH + 1))
branch = hexmesh.loft(stations, first_tag="att_bore", last_tag="branch")

# six seams, every one named on both sides: the collar against the slot it fills, its two
# ends against the legs, its bore cap against the branch, and the pipe against each leg.
core = hexmesh.attach(
    [collar, mid_pipe, downstream, upstream, branch],
    [Seam(0, "att_slot", 1, "att_slot"),
     Seam(0, "att_endA", 2, "att_band"),
     Seam(0, "att_endB", 3, "att_band"),
     Seam(0, "att_bore", 4, "att_bore"),
     Seam(1, "att_pipe_hi", 2, "att_pipe"),
     Seam(1, "att_pipe_lo", 3, "att_pipe")])

# -- name the core's wall, so the skin knows what to grow from -----------------
named = core.face_tags.dense(core.quad_mesh.n_quads)
# ``boundary_face_ids`` hands back a mask over the faces, not the ids themselves
free = np.flatnonzero(hexmesh.boundary_face_ids(core))
core = hexmesh.tag_faces(core, free[named[free] == ""], "wall")

# -- the boundary layer: skin the wall outward ---------------------------------
# ``boundary_mesh(core, "wall")`` is the whole point of naming first -- it hands back the
# pipe wall *and* the branch wall as one surface and leaves the inlet, outlet and branch
# cap out of it, so the openings are never skinned.  Their rims still ride outward with
# it, because a free edge of the surface offsets along its own averaged normal, which is
# radial there and so moves no node in z (or, on the branch, in y).
wall = hexmesh.boundary_mesh(core, "wall")
skins = [wall] + [quadmesh.offset(wall, (r - R_SKIN[0]) * R_MAIN) for r in R_SKIN[1:]]
# the inner cap is named apart from the outer one so ``attach`` can be told which face
# group meets the core -- both would otherwise inherit ``wall`` from the skin they were
# lofted through, and a group twice the size is not the interface
shell = hexmesh.loft(skins, first_tag="inner")

# ``attach``, not ``merge``: the interface is the core's whole ``wall`` group and the
# shell's own ``inner``, so say so rather than have the weld rediscover it from
# coordinates.  It also clears the buried faces, which is what the hand-rolled rebuild
# through raw constructors used to be for -- a *named* interior face is not inert, the
# exporter writes one boundary row per hex carrying one.
mesh = hexmesh.attach([core, shell], [Seam(0, "wall", 1, "inner")])

free = np.flatnonzero(hexmesh.boundary_face_ids(mesh))
mid = mesh.points[mesh.quad_mesh.corners[free]].mean(axis=1)
tags = np.full(free.shape[0], "wall", dtype="<U8")
tags[np.abs(mid[:, 2] + Z_DOMAIN) < 1e-9] = "inlet"
tags[np.abs(mid[:, 2] - Z_DOMAIN) < 1e-9] = "outlet"
tags[np.abs(mid[:, 1] - H_BRANCH) < 1e-9] = "branch"
mesh = hexmesh.tag_faces(mesh, free, tags)

print("core wall %d quads -> %d skin layers -> %d shell hexes"
      % (wall.n_quads, len(R_SKIN) - 1, shell.n_hexes))
print(hexmesh.report(mesh))
writer.to_re2(mesh, OUT_NAME + ".re2", groups=GROUPS)
writer.to_vtu(mesh, OUT_NAME + ".vtu", groups=GROUPS)
