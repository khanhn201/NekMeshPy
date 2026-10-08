import numpy as np

from nekmeshpy import BoundaryConditions, hexmesh, linemesh, quadmesh, writer
from nekmeshpy.core import affine
from nekmeshpy.core.fields import geometric_spacing
from nekmeshpy.quadmesh import QuadMesh
from nekmeshpy.tags import Tags

# -- parameters --------------------------------------------------------------
R_IN = 0.5                   # pipe inner radius
THICKNESS = 0.1              # wall thickness
LENGTH = 5.0

N_AXIAL = 20
N_SIDE = 6
N_RADIAL = 6
N_WALL = 3                   # radial layers through the wall
RADIAL_GRADING = 0.5         # <1 clusters cells toward the wall

ORDER = 2                    # polynomial order; .re2 stays linear either way
OUT_NAME = "periodic_cht_pipe"

R_OUT = R_IN + THICKNESS

# boundary name -> (Nek BC code, boundaryID), applied only at export
VEL_BC =  {"interface": ("W  ", 1),
           "inlet"    : ("P  ", 2),
           "outlet"   : ("P  ", 2)}
TEMP_BC = {"inlet"    : ("P  ", 1),
           "outlet"   : ("P  ", 1),
           "solid_in" : ("P  ", 2),
           "solid_out": ("P  ", 2),
           "outer"    : ("I  ", 3)}
BC = BoundaryConditions(velocity=VEL_BC, temperature=TEMP_BC)
PERIODIC = [
    hexmesh.Periodic("inlet", "outlet", affine.translation([0.0, 0.0, LENGTH])),
    hexmesh.Periodic("solid_in", "solid_out", affine.translation([0.0, 0.0, LENGTH])),
]


def region(section, tag):
    """``section`` with every quad tagged ``tag``: the quad rung's factories take no region
    argument, so a 2-D region is named by rebuilding the container over its own parts."""
    return QuadMesh(section.line_mesh, section.quads, section.orient,
                    section.interior, Tags.full(section.n_quads, tag))


# -- the 2-D cross-section, meshed in full ----------------------------------
inner = linemesh.circle(R_IN, 4 * N_SIDE, order=ORDER)
outer = linemesh.circle(R_OUT, 4 * N_SIDE, order=ORDER)

fluid = region(quadmesh.ogrid(inner, N_SIDE, N_RADIAL, wall_tag="interface"), "fluid")
solid = region(quadmesh.annulus(
    linemesh.reverse(inner), linemesh.reverse(outer),
    geometric_spacing(N_WALL, 1.0), inner_tag="interface", outer_tag="outer"), "solid")

# the interface stays named: it is the face a wall condition goes on
section = quadmesh.attach(
    [fluid, solid],
    [quadmesh.Seam(0, "interface", 1, "interface", attach_tag="interface")])

# -- extrude: regions and side names ride up with it ------------------------
# the caps default to the slice's own region name, and a region is never a boundary
# name, so they are named per element -- fluid ends flow, solid ends only conduct
region_of = section.element_tags.to_dense()
first = Tags.from_dense(np.where(region_of == "fluid", "inlet", "solid_in"))
last = Tags.from_dense(np.where(region_of == "fluid", "outlet", "solid_out"))

mesh = hexmesh.extrude(section, LENGTH, N_AXIAL, axis=(0.0, 0.0, 1.0),
                       element_tags=section.element_tags,
                       first_tag=first, last_tag=last)

# -- report + export ---------------------------------------------------------
assert hexmesh.is_watertight(mesh) and hexmesh.is_conforming(mesh)
assert set(mesh.element_tags.unique()) == {"fluid", "solid"}
assert set(mesh.face_group_tags) == {"interface", "outer", "inlet", "outlet",
                                     "solid_in", "solid_out"}

stats = hexmesh.quality_summary(mesh)
assert stats.min > 0.0, "inverted element: min scaled Jacobian %g" % stats.min
print("periodic cht pipe: %d hexes (%d fluid), %d points, order %d"
      % (mesh.n_hexes, int((mesh.element_tags.to_dense() == "fluid").sum()),
         mesh.n_points, mesh.order))
print("scaled Jacobian: min=%.4f mean=%.4f" % (stats.min, stats.mean))
print("groups:", ", ".join(mesh.face_group_tags))

writer.to_re2(mesh, OUT_NAME + ".re2", bc=BC, periodic=PERIODIC, fluid="fluid")
writer.to_vtu(mesh, OUT_NAME + ".vtu", bc=BC)

