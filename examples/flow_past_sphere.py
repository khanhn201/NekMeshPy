"""Flow past a sphere (external flow, cubed-sphere shell).

The domain fills the region between a spherical body (radius ``R``) and a cubic
far-field box (half-side ``S``), built with :meth:`HexMesh.annulus` (the 3-D
sibling of :meth:`QuadMesh.annulus`): fill the shell between two **closed quad
surfaces**, an inner sphere and an outer cube.

The outer cube is built with :meth:`QuadMesh.box`, each face carrying the far-field
side it forms (``inlet`` / ``outlet`` / ...) as per-quad ``element_tags``. The inner
sphere (:meth:`QuadMesh.sphere`) reuses the cube's connectivity -- so the two pair by
index. ``annulus`` blends the radial shells (clustered toward the sphere) and turns
each surface's per-quad tags into the inner / outer wall faces (no free boundary
edges, so wall groups come from element tags).

    PYTHONPATH=. python examples/flow_past_sphere.py

Produces ``flow_past_sphere.re2`` / ``.rea`` and ``flow_past_sphere.vtk``.
"""

import logging

from nekmeshpy import HexMesh, QuadMesh, export
from nekmeshpy.model.fields import geometric_spacing

logging.basicConfig(level=logging.INFO, format="%(message)s")

# -- parameters --------------------------------------------------------------
R = 0.5                      # sphere radius
S = 4.0                      # far-field cube half-side (domain [-S, S]^3)
N_FACE = 12                  # cells per direction on each cube face
N_RADIAL = 12                # radial cells from sphere out to the cube
RADIAL_GRADING = 1.15        # >1 clusters radial layers toward the sphere
OUT_NAME = "flow_past_sphere"

# boundary name -> Nek BC code, applied only at export
GROUPS = {"inlet": "v  ", "outlet": "O  ", "sphere": "W  ",
          "top": "SYM", "bottom": "SYM", "front": "SYM", "back": "SYM"}

# -- two closed quad surfaces: outer cube (tagged per face) and inner sphere -
# QuadMesh.box tags each face with the far-field side it forms; QuadMesh.sphere
# reuses the same (N_FACE) connectivity, so the two pair by index for annulus.
cube = QuadMesh.box(S, N_FACE, face_tags={
    "x_max": "outlet", "x_min": "inlet",
    "y_max": "top", "y_min": "bottom",
    "z_max": "back", "z_min": "front"})
sphere = QuadMesh.sphere(R, N_FACE)

# fill the shell sphere -> cube, radial clustered toward the sphere; inner cap
# tagged `sphere`, outer cap per cube-face element tag
mesh = HexMesh.annulus(sphere, cube,
                       radial=geometric_spacing(N_RADIAL, RADIAL_GRADING))

# -- report + export ---------------------------------------------------------
print(mesh.report())
export.to_re2(mesh, OUT_NAME, groups=GROUPS)
export.to_vtk(mesh, OUT_NAME + ".vtk", groups=GROUPS)
print("groups:", ", ".join(mesh.boundary_group_tags))
