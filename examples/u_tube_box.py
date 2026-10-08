"""Solid block with a U-shaped bore: the outer solid of a U tube in a box.

A structured box is meshed first, ignoring the tube.  Every hex whose centre lies
inside a square duct around the U's centreline is removed, leaving a stair-stepped cavity.  The
cavity's exposed faces are lifted out as a quad surface (``boundary_mesh``), every one
of its nodes is projected radially onto the tube wall (``transform_fn``), and one layer
of hexes is lofted from the stair-step to that wall.  Welding that shell to the box
(``attach``) closes the solid, and its inner surface is the tube wall, ready for a
fluid block to meet at ``interface``.

The U lies in the plane ``z = 0``: two legs along +x at ``y = +-U_HALF``, both opening
on the ``x = 0`` face, joined by a semicircle of radius ``U_HALF``.

    PYTHONPATH=. python examples/u_tube_box.py

Produces ``u_tube_box.re2`` and ``u_tube_box.vtu``.
"""

import logging

import numpy as np

from nekmeshpy import BoundaryConditions, HexMesh, hexmesh, quadmesh, writer
from nekmeshpy.tags import Tags

logging.basicConfig(level=logging.INFO, format="%(message)s")

# -- parameters --------------------------------------------------------------
BOX_X = 33.0                 # box extent in x; the tube opens on the x = 0 face
BOX_HALF_Y = 19.0
BOX_HALF_Z = 9.0
H = 1.0                      # box cell size
R_TUBE = 3.0                 # tube radius
U_HALF = 10.0                # half the leg spacing == radius of the U's bend
U_LEG = 14.0                 # straight leg length, x = 0 to the start of the bend
HALF_W = 4.0                 # half-width of the square duct cut around the axis; whole
                             # cells, so the leg's cross-section is an exact 8x8 block
ORDER = 2
CORE_FILL = 0.75             # the shrunk core reaches this fraction of the tube radius
OUT_NAME = "u_tube_box"


# -- the U's centreline: nearest axis point to each node ----------------------
def axis_point(p):
    """Closest point on the U's centreline to each row of ``p``."""
    p = np.asarray(p, dtype=float)
    c = np.zeros_like(p)
    straight = p[:, 0] <= U_LEG
    c[straight, 0] = p[straight, 0]
    c[straight, 1] = np.where(p[straight, 1] >= 0.0, U_HALF, -U_HALF)
    bend = ~straight
    d = p[bend, :2] - np.array([U_LEG, 0.0])
    c[bend, :2] = np.array([U_LEG, 0.0]) + U_HALF * d / np.linalg.norm(d, axis=1)[:, None]
    return c


def project_to_wall(p):
    """Each node moved along its radial to the tube surface."""
    v = p - axis_point(p)
    return axis_point(p) + R_TUBE * v / np.linalg.norm(v, axis=1)[:, None]


# -- 1. structured box, tube ignored ------------------------------------------
nx, ny, nz = round(BOX_X / H), round(2 * BOX_HALF_Y / H), round(2 * BOX_HALF_Z / H)
grid = np.stack(np.meshgrid(np.linspace(0.0, BOX_X, nx + 1),
                            np.linspace(-BOX_HALF_Y, BOX_HALF_Y, ny + 1),
                            np.linspace(-BOX_HALF_Z, BOX_HALF_Z, nz + 1),
                            indexing="ij"), axis=-1)
box = hexmesh.from_grid(grid, order=ORDER,
                        side_tags={s: "outer" for s in
                                   ("x_min", "x_max", "y_min", "y_max", "z_min", "z_max")})
# ``from_grid``'s element_tag never reaches the hexes, so name the region here
box = HexMesh(box.quad_mesh, box.hexes, box.orient, box.interior,
              Tags.full(box.n_hexes, "solid"))

# -- 2. remove the hexes around the U -----------------------------------------
centres = box.points[box.corners].mean(axis=1)
off = centres - axis_point(centres)
# square, not round: the offset is measured across the axis (in-plane, then z), so a
# leg's cavity is a block of whole cells and its stair-step faces are flat squares
near = np.maximum(np.linalg.norm(off[:, :2], axis=1), np.abs(off[:, 2])) < HALF_W


cells = box                      # the full box; the fluid is cut out of it below
box = hexmesh.remove(box, near)

# -- 3. the cavity's exposed faces, then that surface on the tube wall ---------
exposed = hexmesh.boundary_face_ids(box)
exposed[hexmesh.tagged_faces(box, "outer")] = False
box = hexmesh.tag_faces(box, np.flatnonzero(exposed), "seam")

stair = hexmesh.boundary_mesh(box, "seam")
# the cavity's open rim lies in the x = 0 face, so it is part of the outer surface
stair = quadmesh.tag_edges(stair, quadmesh.boundary_edges(stair), "outer")
wall = quadmesh.transform_fn(stair, project_to_wall)

# -- 4. one layer of hexes from the stair-step to the wall --------------------
shell = hexmesh.loft([stair, wall], element_tags="solid", last_tag="interface")

# -- 5. the fluid: the removed voxels, shrunk into the tube ----------------------
# The removed voxels are the stair-stepped tube the solid shell was lofted off, so their
# outer surface *is* ``stair``.  Squeezing them straight onto the wall would flatten every
# voxel that touches it, so they shrink about the axis instead (``shrink``) to a Cartesian
# core that fits well inside the tube, and a second shell -- the mirror of the solid's --
# is lofted from the wall in to that core.  Both pass through the same ``shrink`` on the
# same nodes, so the core's surface and the shell's inner cap agree exactly.
DMAX = np.linalg.norm(stair.points - axis_point(stair.points), axis=1).max()
SHRINK = CORE_FILL * R_TUBE / DMAX


def shrink(p):
    c = axis_point(p)
    return c + (p - c) * SHRINK


core = hexmesh.retag_element(hexmesh.select(cells, near), {"solid": "fluid"})
ends = hexmesh.tagged_faces(core, "outer")           # the tube's two openings, in x = 0
cq = core.quad_mesh
assert np.allclose(cq.points[cq.corners[ends]][..., 0], 0.0)
edge = hexmesh.boundary_face_ids(core)
edge[ends] = False
core = hexmesh.tag_faces(core, np.flatnonzero(edge), "seam")
side = cq.points[cq.corners[ends]].mean(axis=1)[:, 1]
core = hexmesh.tag_faces(core, ends, np.where(side < 0.0, "inlet", "outlet"))
core = hexmesh.transform_fn(core, shrink)

inner = quadmesh.transform_fn(stair, shrink)
# the shell's rim, on the same x = 0 plane, is the tube's opening: inlet on the -y leg
rim = quadmesh.boundary_edges(wall)
rim_y = wall.points[wall.corners[rim[:, 0]]].mean(axis=1)[:, 1]
wall_f = quadmesh.tag_edges(wall, rim, np.where(rim_y < 0.0, "inlet", "outlet"))
fshell = hexmesh.loft([wall_f, inner], element_tags="fluid", first_tag="interface",
                      last_tag="seam")

# -- 6. weld: box | solid shell | fluid shell | core ----------------------------
mesh = hexmesh.attach(
    [box, shell, fshell, core],
    [hexmesh.Seam(0, "seam", 1, "seam"),
     hexmesh.Seam(1, "interface", 2, "interface", attach_tag="interface"),
     hexmesh.Seam(2, "seam", 3, "seam")])

# -- checks + export -----------------------------------------------------------
assert hexmesh.is_watertight(mesh) and hexmesh.is_conforming(mesh)
print("faces:", ", ".join(sorted(mesh.face_group_tags)))
stats = hexmesh.quality_summary(mesh)
print("u-tube box: %d hexes (%d fluid), order %d, scaled Jacobian min=%.4f mean=%.4f"
      % (mesh.n_hexes, core.n_hexes + fshell.n_hexes, mesh.order, stats.min, stats.mean))

print(hexmesh.report(mesh))
# the solid has no velocity unknown, so the wall is written from the fluid side only
VEL_BC = {"interface": ("W  ", 1), "inlet": ("v  ", 2), "outlet": ("O  ", 3)}
TEMP_BC = {"outer": ("f  ", 1)}
BC = BoundaryConditions(velocity=VEL_BC, temperature=TEMP_BC)
writer.to_re2(mesh, OUT_NAME + ".re2", bc=BC, fluid="fluid")
writer.to_vtu(mesh, OUT_NAME + ".vtu", bc=BC)
