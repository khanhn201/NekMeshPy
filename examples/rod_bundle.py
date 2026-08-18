"""61-rod hexagonal fuel bundle, 2-D cross-section (three regions, conjugate).

A wire-wrap-free fast-reactor pin bundle: 61 solid rods on a triangular lattice of
``PITCH``, the coolant between them, and the hexagonal duct wall that closes the
assembly.  Three regions come out named -- ``rod`` / ``fluid`` / ``duct`` -- so the
section is ready for a conjugate heat-transfer solve.

The decomposition is the lattice's own Voronoi tiling.  Each rod owns the hexagonal
cell of points nearer to it than to any other rod: a regular hexagon of circumradius
``PITCH/sqrt(3)``, vertices at 30 + 60k degrees, and those cells tile the plane
exactly.  So one rod's mesh is two pieces and nothing else:

    ogrid(rod circle)                 -> the solid rod
    annulus(rod circle, cell hexagon) -> its share of the coolant

Both loops carry ``6 * PER_SIDE`` points and the circle starts at 30 degrees, so a
circle point sits within a few degrees of the hexagon point it pairs with in
``annulus`` and the radial spokes come out nearly radial.  ``ogrid`` wants
``4 * n_side`` boundary points with ``n_side`` even, so the count must be a multiple
of 8 as well as of 6: ``PER_SIDE = 4`` gives 24, the smallest that works.

The 61 cells do **not** union into a hexagon -- the outer edge of the tiling is the
zigzag every honeycomb patch has -- so the duct cannot simply be lofted off them side
by side.  Instead the zigzag is collected as one closed loop (a cell side is on it
exactly when the lattice neighbour across that side is absent from the bundle) and
``annulus`` fills from it straight out to the duct's inner hexagon.  That pairing is
what fixes the duct's own discretization: each zigzag point is assigned to the duct
side whose 60-degree sector it falls in, and each duct side is then sampled uniformly
with the count its sector received.  With ``PER_SIDE`` even a zigzag point lands on
every 60-degree ray, so the duct's six corners are nodes rather than cut corners.

The section is then swept ``SPAN`` along the pin axis. Its per-quad region split
rides up as ``element_tags``; the caps are named explicitly, because left alone they
default to the bounding slice's own region names and the bundle would export an
opening called ``fluid``. Only the coolant has an inlet and an outlet -- the rod and
duct ends are solid metal, ``cut`` at both. The two interfaces are **conjugate**: one
face, one name, but a different condition each side, so ``GROUPS`` gives them a
per-region code and the metal side writes no row at all.

    PYTHONPATH=. python examples/rod_bundle.py

Produces ``rod_bundle.re2`` and ``rod_bundle.vtu``.
"""

import logging

import numpy as np

from nekmeshpy import hexmesh, linemesh, quadmesh, writer
from nekmeshpy.core.fields import geometric_spacing, gll_nodes
from nekmeshpy.core.tags import ElementTags
from nekmeshpy.quadmesh import QuadMesh

logging.basicConfig(level=logging.INFO, format="%(message)s")

# -- parameters --------------------------------------------------------------
PITCH_MIN = 1.0              # rod centre-to-centre spacing, if the wire allows it
R_ROD = 0.40                 # rod radius
RINGS = 4                    # lattice rings about the centre rod -> 1+3*4*5 = 61
WALL_GAP = 0.35              # coolant left between the tiling's zigzag and the duct
DUCT_T = 0.5                 # duct wall thickness

PER_SIDE = 4                 # points per hexagonal-cell side -> 24 round a rod
N_CORE = 3                   # radial layers inside the rod's O-grid ring
N_COOL = 3                   # radial layers from rod wall out to the cell hexagon
COOL_GRADING = 1.3           # >1 clusters coolant layers onto the rod wall
N_GAP = 4                    # radial layers across the outer coolant gap
N_DUCT = 3                   # radial layers through the duct wall

# Wire wrap.  A real spacer wire spirals along the rod, so on any one slice it sits at
# one angle -- here ``WIRE_THETA`` on every rod, which is the slice a helix of pitch
# equal to the span cuts at z = 0.  It is not meshed as a body of its own: the rod wall
# is a single closed profile, and the wire is a local outward bulge in it, so the
# topology (O-grid core + ring out to the cell hexagon) is untouched and the wire's
# metal simply joins the rod's own solid region.
WIRE_H = 0.16                # radial height of the bulge above the rod wall
WIRE_HALF = np.deg2rad(25.0)  # angular half-width of the bulge
WIRE_CLEAR = 0.04            # coolant left between the wire's crest and the cell wall
WIRE_LEAD = 40.0 * (2.0 * R_ROD)   # axial rise per complete turn of the wire (20 D)
SPAN = WIRE_LEAD             # one full turn, so the block is axially periodic
N_SPAN = 40                  # hex layers across the span
ORDER = 2                    # polynomial order; >1 bows the rod walls onto the circle

OUT_NAME = "rod_bundle"

# boundary name -> Nek BC code, applied only at export.  The two interfaces are
# **conjugate**: one face, one name, but a different condition on each side of it, so
# they take a per-region mapping instead of a single code -- the coolant sees a wall
# and the metal writes no row at all.
GROUPS = {"inlet": "v  ", "outlet": "O  ", "cut": "I  ", "outer": "I  ",
          "rod_surface": {"fluid": "W  ", "rod": None},
          "duct_surface": {"fluid": "W  ", "duct": None}}

# The wire sets the pitch, the way it does in a real bundle -- two conditions, both
# of which the assertions below used to raise on:
#   * the gap between two rods, ``PITCH - 2*R_ROD``, holds the wire plus a clearance
#     either side (every rod's wire points the same way, so the wall a wire advances is
#     the wall squeezed toward the rod behind it);
#   * the crest clears the cell corner at ``CELL_R = PITCH/sqrt(3)``, which the push
#     pins in place because three walls meet there.
# ``PITCH_MIN`` wins when the wire is small enough not to care.
PITCH = max(PITCH_MIN,
            2.0 * R_ROD + WIRE_H + 2.0 * WIRE_CLEAR,
            np.sqrt(3.0) * (R_ROD + WIRE_H + WIRE_CLEAR))

N_CIRC = 6 * PER_SIDE                        # points on a rod circle == on its cell
CELL_R = PITCH / np.sqrt(3.0)                # cell hexagon circumradius
FLAT = RINGS * PITCH * np.sqrt(3.0) / 2.0    # centre -> outermost rod row
DUCT_IN = FLAT + CELL_R + WALL_GAP           # centre -> duct inner face (apothem)

# The wire is taller than the coolant gap it sits in (``PITCH/2 - R_ROD``), so it would
# poke straight through the cell wall.  The wall has to give way by the overshoot plus
# the clearance -- see ``cell_boundary`` for why that is a shift of the *whole* lattice
# of cell walls along the wire, and not a bulge in the one wall the wire faces.
WIRE_PUSH = max(0.0, WIRE_H + WIRE_CLEAR - (PITCH / 2.0 - R_ROD))

# Feasibility, checked here so an infeasible wire says so instead of surfacing 200
# lines later as slices that will not weld.
#
# The binding limit is not the rod's own cell -- it is the coolant gap *between two
# rods*, ``PITCH - 2*R_ROD`` wide, which has to hold the wire with coolant on both
# sides.  Every rod carries a wire pointing the same way, so the wall a wire advances
# is the very wall being squeezed toward the rod behind it; the slide has to be at
# least ``WIRE_H - (PITCH/2 - R_ROD) + WIRE_CLEAR`` to clear the crest and at most
# ``(PITCH/2 - R_ROD) - WIRE_CLEAR`` before it cuts into the rod behind, and those two
# meet at the cap below.  A wire that fills the gap is one that touches its neighbour,
# which is what a real spacer does and what no body-fitted mesh can carry.
# ``PITCH`` is derived to satisfy both of these, so they are self-checks -- hence the
# slack, which is there for the float equality at the boundary and nothing else.
SLACK = 1e-9
WIRE_H_MAX = (PITCH - 2.0 * R_ROD) - 2.0 * WIRE_CLEAR
assert WIRE_H <= WIRE_H_MAX + SLACK, (
    "WIRE_H = %.3f does not fit: the gap between two rods is PITCH - 2*R_ROD = %.3f "
    "and needs WIRE_CLEAR = %.3f of coolant either side, leaving %.3f. Raise PITCH to "
    "%.3f (the real parametrization -- the wire is what sets the pitch), or lower "
    "WIRE_H." % (WIRE_H, PITCH - 2.0 * R_ROD, WIRE_CLEAR, WIRE_H_MAX,
                 2.0 * R_ROD + WIRE_H + 2.0 * WIRE_CLEAR))

# and the crest must still fit inside the cell, whose corners the push pins in place
assert R_ROD + WIRE_H + WIRE_CLEAR <= CELL_R + SLACK, (
    "WIRE_H = %.3f puts the crest at %.3f, past the cell corner at CELL_R = %.3f. The "
    "push tapers to zero at the corners (three walls meet there), so no push can make "
    "room: lower WIRE_H below %.3f or raise PITCH."
    % (WIRE_H, R_ROD + WIRE_H, CELL_R, CELL_R - R_ROD - WIRE_CLEAR))

A1 = np.array([PITCH, 0.0, 0.0])
A2 = np.array([PITCH / 2.0, PITCH * np.sqrt(3.0) / 2.0, 0.0])

# cell side s runs from vertex (30 + 60s) to vertex (90 + 60s), so its outward normal
# points at 60 + 60s -- which is the lattice step to the neighbour across it
NEIGHBOUR = [(0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1), (1, 0)]


def hexagon(centre, circumradius, angle0, per_side):
    """A regular hexagon's ``(6,3)`` vertices and its ``(6*per_side,3)`` CCW loop
    points, first vertex at ``angle0`` degrees, each side sampled endpoint-exclusive."""
    ang = np.deg2rad(angle0) + np.arange(6) * (np.pi / 3.0)
    V = np.asarray(centre, dtype=float) + circumradius * np.stack(
        [np.cos(ang), np.sin(ang), np.zeros(6)], axis=1)
    f = (np.arange(per_side) / per_side)[:, None]
    pts = np.concatenate([V[k] + f * (V[(k + 1) % 6] - V[k]) for k in range(6)])
    return V, pts


def wire_bump(theta, wire_theta):
    """The wire's radial footprint on the rod wall: a raised cosine, ``WIRE_H`` tall at
    ``wire_theta`` and falling to zero *with zero slope* at +/- ``WIRE_HALF``, so the
    profile leaves the plain circle smoothly instead of kinking off it.

    ``WIRE_HALF`` is deliberately wider than a tangent wire's true footprint
    (``asin(r_wire / (R_ROD + r_wire))``, about 11 degrees for a wire that fills the
    gap): at ``PER_SIDE = 4`` the wall carries a node every 15 degrees, so a true
    footprint would land on one or two of them. Resolving one needs a finer wall --
    ``PER_SIDE = 8`` halves the spacing and keeps ``N_CIRC`` a multiple of 8, which is
    what ``ogrid`` requires."""
    d = (theta - wire_theta + np.pi) % (2.0 * np.pi) - np.pi
    return np.where(np.abs(d) < WIRE_HALF,
                    WIRE_H * 0.5 * (1.0 + np.cos(np.pi * d / WIRE_HALF)), 0.0)


def rod_wall(centre, wire_theta):
    """The rod wall as a closed ``LineMesh``: ``linemesh.circle`` with the wire bulge
    added, same ``N_CIRC`` points from the same start angle, so the cell hexagon still
    pairs with it index-for-index in ``annulus``.

    ``circle`` places its high-order nodes on the true arc; a plain ``loft`` over the
    corner points would straight-subdivide between them and throw that away (the trap
    in ``CLAUDE.md``), so the interior nodes are evaluated on the profile too -- at the
    GLL parameters inside each segment, which is where the element actually reads
    them."""
    theta = np.linspace(0.0, 2.0 * np.pi, N_CIRC, endpoint=False) + np.pi / 6.0

    def at(a):
        r = R_ROD + wire_bump(a, wire_theta)
        return centre + np.stack(
            [r * np.cos(a), r * np.sin(a), np.zeros_like(a)], axis=-1)

    if ORDER == 1:
        return linemesh.loft(at(theta), loop=True)
    g = gll_nodes(ORDER)[1:ORDER]
    interior = at(theta[:, None] + g[None, :] * (2.0 * np.pi / N_CIRC))
    return linemesh.loft(at(theta), loop=True, interior=interior, order=ORDER)


def cell_boundary(V, per_side, order, wire_dir):
    """A cell's boundary as a closed ``LineMesh``, with the wire push in it.

    A cell wall is **shared**: it is one stored curve that both cells either side of it
    reference, so a push cannot be authored per cell -- the two would have to agree to
    the last bit, and "bulge my own wall outward" and "let my neighbour's wall in" are
    not the same expression.  What is authored instead is a property of the *side*:
    every wall slides along ``wire_dir`` by ``WIRE_PUSH * (n.w)^2 * sin^2(pi t)``, read
    off the side's own two vertices and its own parameter.  The neighbour walks the
    same side backwards, so it sees ``1 - t`` and the opposite normal -- and both the
    squared dot and ``sin^2`` are symmetric under exactly those two flips, so it
    computes the same curve to round-off.

    That the taper vanishes at ``t = 0`` and ``t = 1`` is what pins the cell *vertices*,
    where three walls meet carrying three different normals and no single displacement
    could satisfy all of them.

    The result is not a local dent: every wall in the lattice slides the same way, so
    the wall a wire pushes on advances and the wall behind the rod is squeezed in
    by its neighbour's wire by the same amount -- which is what the geometry actually
    does, every rod carrying a wire."""
    def side_at(k, t):
        a, b = V[k], V[(k + 1) % 6]
        e = b - a
        n = np.array([-e[1], e[0], 0.0]) / np.linalg.norm(e)
        amp = WIRE_PUSH * float(n @ wire_dir) ** 2
        return (a + t[:, None] * e
                + amp * (np.sin(np.pi * t) ** 2)[:, None] * wire_dir)

    t0 = np.arange(per_side) / per_side
    pts = np.concatenate([side_at(k, t0) for k in range(6)])
    if order == 1:
        return linemesh.loft(pts, loop=True), None
    g = gll_nodes(order)[1:order]
    interior = np.concatenate([
        np.stack([side_at(k, (i + g) / per_side) for i in range(per_side)])
        for k in range(6)])
    return linemesh.loft(pts, loop=True, interior=interior, order=order), interior


def region(section, tag):
    """``section`` with every quad tagged ``tag``.  The quad rung's factories take no
    region argument -- ``element_tags`` enters at the sweep that lifts a section into a
    volume -- so a 2-D region is named by rebuilding the container over its own parts."""
    return QuadMesh(section.line_mesh, section.quads, section.orient,
                    section.interior, ElementTags.uniform(section.n_quads, tag))


def key(vertex):
    """A hashable identity for a cell vertex, tight enough that only coincident
    vertices collide (cell vertices are ``PITCH/sqrt(3)`` apart at the closest)."""
    return tuple(np.round(vertex / PITCH, 9))


# -- the lattice -------------------------------------------------------------
LATTICE = [(i, j)
           for i in range(-RINGS, RINGS + 1)
           for j in range(-RINGS, RINGS + 1)
           if abs(i + j) <= RINGS]
assert len(LATTICE) == 61, len(LATTICE)
PRESENT = set(LATTICE)

def build_section(wire_theta):
    """The whole 2-D section with the wire at ``wire_theta`` on every rod.

    Rebuilt per slice rather than morphed, and the slices ``hexmesh.loft`` welds
    together have to be *conformal* -- same point count, same numbering, only the
    coordinates moving.  Nothing here reads a coordinate to decide an index: the
    lattice is enumerated in a fixed order, the zigzag chains through vertices that the
    push pins in place, and the duct's sectors are counted off the **unpushed** loop.
    So the connectivity is a function of the constants alone and comes back identical
    at every angle, which is asserted once below rather than assumed."""
    wire_dir = np.array([np.cos(wire_theta), np.sin(wire_theta), 0.0])

    # -- one Voronoi cell per rod: solid O-grid core + its share of the coolant ---
    blocks = []
    zigzag = {}                                  # start vertex -> (end vertex, 4 points)

    for (i, j) in LATTICE:
        centre = i * A1 + j * A2
        V, cell_ref = hexagon(centre, CELL_R, 30.0, PER_SIDE)
        cell, cell_int = cell_boundary(V, PER_SIDE, ORDER, wire_dir)
        cell_pts = cell.points
        wall = rod_wall(centre, wire_theta)

        # ``annulus`` winds its quads (loop tangent) x (radial), so a CCW loop hands back
        # a section wound onto -z where ``ogrid`` hands back +z.  ``hexmesh.loft`` flips
        # its own template for a uniformly left-handed section but rejects a *mixed* one,
        # so the two families have to agree: reversing **both** of an annulus's loops
        # leaves the index pairing -- and so the geometry -- untouched and flips the
        # winding to match the O-grid.
        blocks.append(region(
            quadmesh.ogrid(wall, N_CIRC // 4, N_CORE, wall_tag="rod_surface"), "rod"))
        blocks.append(region(
            quadmesh.annulus(linemesh.reverse(wall), linemesh.reverse(cell),
                             geometric_spacing(N_COOL, COOL_GRADING),
                             inner_tag="rod_surface"), "fluid"))

        for s in range(6):
            di, dj = NEIGHBOUR[s]
            if (i + di, j + dj) not in PRESENT:
                sl = slice(s * PER_SIDE, (s + 1) * PER_SIDE)
                zigzag[key(V[s])] = (key(V[(s + 1) % 6]), cell_ref[sl], cell_pts[sl],
                                     None if cell_int is None else cell_int[sl])

    # -- chain the boundary sides into the tiling's outer loop -------------------
    # every zigzag vertex is the start of exactly one boundary side, so the chain closes
    start = next(iter(zigzag))
    node, refs, chain, ints = start, [], [], []
    while True:
        end, ref, pts, block = zigzag[node]
        refs.append(ref)
        chain.append(pts)
        ints.append(block)
        node = end
        if node == start:
            break
    assert len(chain) == len(zigzag), (len(chain), len(zigzag))
    ref_pts = np.concatenate(refs)
    inner_pts = np.concatenate(chain)
    inner_int = None if ints[0] is None else np.concatenate(ints)

    # roll the loop so index 0 is its point on the +x ray -- the duct's first corner, so
    # the two loops pair corner-for-corner and every sector gets the same count
    # read the sectors off the **unpushed** zigzag: the push is along one direction, so
    # it breaks the six-fold symmetry the corner hunt relies on, while leaving the index
    # bookkeeping it produces perfectly valid
    theta = np.rad2deg(np.arctan2(ref_pts[:, 1], ref_pts[:, 0]))
    corners = np.flatnonzero(np.abs((theta + 30.0) % 60.0 - 30.0) < 1e-9)
    assert corners.shape[0] == 6, corners
    first = corners[np.argmin(np.abs(theta[corners]))]
    inner_pts = np.roll(inner_pts, -first, axis=0)
    if inner_int is not None:
        inner_int = np.roll(inner_int, -first, axis=0)
    n_side = inner_pts.shape[0] // 6
    assert np.array_equal(np.sort((corners - first) % inner_pts.shape[0]),
                          n_side * np.arange(6)), corners

    # -- the duct: its inner face inherits the zigzag's per-sector point counts --
    _, duct_in_pts = hexagon(np.zeros(3), 2.0 * DUCT_IN / np.sqrt(3.0), 0.0, n_side)
    _, duct_out_pts = hexagon(np.zeros(3), 2.0 * (DUCT_IN + DUCT_T) / np.sqrt(3.0),
                              0.0, n_side)

    inner = linemesh.loft(inner_pts, loop=True, interior=inner_int, order=ORDER)
    duct_in = linemesh.loft(duct_in_pts, loop=True, order=ORDER)
    duct_out = linemesh.loft(duct_out_pts, loop=True, order=ORDER)

    blocks.append(region(quadmesh.annulus(linemesh.reverse(inner),
                                          linemesh.reverse(duct_in), N_GAP,
                                          outer_tag="duct_surface"), "fluid"))
    blocks.append(region(quadmesh.annulus(linemesh.reverse(duct_in),
                                          linemesh.reverse(duct_out), N_DUCT,
                                          inner_tag="duct_surface",
                                          outer_tag="outer"), "duct"))

    return quadmesh.merge(blocks)



# -- the helical stack: one section per z station -----------------------------
# The wire is a helix, so its angle is a function of z: one complete turn per
# ``WIRE_LEAD``.  There is no sweep primitive for that -- ``extrude`` and ``sweep`` both
# carry *one* section rigidly, and this section changes shape station to station -- so
# the slices are built individually and ``hexmesh.loft`` welds the stack.
#
# ``loft`` straight-subdivides along its sweep direction (the trap in ``CLAUDE.md``),
# which here is exactly right: a station's geometry is only meaningful at its own z, and
# the discretization *is* the helix's resolution. 40 layers over one turn is a station
# every 9 degrees of wire rotation.
fractions = np.linspace(0.0, 1.0, N_SPAN + 1)
slices = [quadmesh.translate(build_section(2.0 * np.pi * f), (0.0, 0.0, f * SPAN))
          for f in fractions]
section = slices[0]

# the conformality ``loft`` requires, checked rather than trusted: a rebuild that
# renumbered under the rotation would weld a station to the wrong neighbour
for other in slices[1:]:
    assert other.n_points == section.n_points
    assert np.array_equal(other.corners, section.corners)
    assert np.array_equal(other.line_mesh.lines, section.line_mesh.lines)

# ``element_tags`` takes an ElementTags over *one slice's* elements, which tags each
# swept column by the quad it came from -- so the section's own rod / fluid / duct
# split rides up into the volume unchanged.
#
# The caps must be named explicitly.  Left alone they default to the bounding slice's
# own ``element_tags``, and a hex's region name is never a boundary name: the bundle
# would export its ends as a boundary condition called "fluid".  Only the coolant has
# an inlet and an outlet; the rod and duct ends are solid metal, named once for both
# ends because nothing flows through them.
region_of = section.element_tags.dense(section.n_quads)
inlet = ElementTags.from_dense(np.where(region_of == "fluid", "inlet", "cut"))
outlet = ElementTags.from_dense(np.where(region_of == "fluid", "outlet", "cut"))

mesh = hexmesh.loft(slices, element_tags=section.element_tags,
                    first_tag=inlet, last_tag=outlet)

# -- report + export ---------------------------------------------------------
q = hexmesh.quality_summary(mesh)
print("rods %d   hexes %d   points %d   order %d"
      % (len(LATTICE), mesh.n_hexes, mesh.n_points, mesh.order))
# the .vtu carries the regions as a per-cell ``element_tag`` integer; the mapping is
# a function of the mesh alone, so print it rather than shipping a legend
_, region_names = writer.element_tag_ids(mesh.element_tags, mesh.n_hexes)
print("regions:", ", ".join("%s=%d" % (n, i + 1)
                            for i, n in enumerate(region_names)))
print("boundaries:", ", ".join(mesh.face_group_tags))
print("wire lead %.2f (%.0f D), span %.2f, %d layers -> %.1f deg/layer"
      % (WIRE_LEAD, WIRE_LEAD / (2.0 * R_ROD), SPAN, N_SPAN, 360.0 / N_SPAN))
print("section area %.6f   volume %.6f" % (quadmesh.area(section),
                                           hexmesh.volume(mesh)))
print("minSJ %.4f   inverted %d   watertight %s"
      % (q.min, q.n_inverted, hexmesh.is_watertight(mesh)))
writer.to_re2(mesh, OUT_NAME + ".re2", groups=GROUPS)
writer.to_vtu(mesh, OUT_NAME + ".vtu", groups=GROUPS)
