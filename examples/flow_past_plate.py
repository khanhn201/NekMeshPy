"""Flow past a thin plate (external flow, all-hex O-grid).

Same construction as ``flow_past_cylinder.py`` -- a ring O-grid
(:meth:`QuadMesh.annulus`) between the body and a square far-field box, swept along
the span -- but the body is a **thin ellipse** (half-length ``A``, half-thickness
``B << A``) standing in for a flat plate. A true zero-thickness plate would need a
C-grid; the thin ellipse keeps a non-degenerate O-grid as ``B`` -> 0.

Boundaries are named on the source line geometry (see ``flow_past_cylinder.py``):
body ``plate`` on the inner loop, far-field sides tagged by
:meth:`LineMesh.far_field_box` (built index-aligned to the inner), carried
through ``annulus`` -> :meth:`HexMesh.extrude`; the sweep names the caps
``front`` / ``back``.

    PYTHONPATH=. python examples/flow_past_plate.py

Produces ``flow_past_plate.re2`` / ``.rea`` and ``flow_past_plate.vtk``.
"""

import logging

import numpy as np

from nekmeshpy import HexMesh, LineMesh, QuadMesh, export
from nekmeshpy.model.fields import geometric_spacing

logging.basicConfig(level=logging.INFO, format="%(message)s")

# -- parameters --------------------------------------------------------------
A = 1.0                      # plate half-length (streamwise)
B = 0.08                     # plate half-thickness (B << A)
HALF_BOX = 6.0               # far-field box half-width (domain [-HB, HB]^2 in xy)
SPAN = 2.0                   # spanwise (z) extent
N_THETA = 80                 # azimuthal cells around the plate
N_RADIAL = 24                # radial cells from plate out to the box
RADIAL_GRADING = 1.12        # >1 clusters radial layers toward the plate
N_SPAN = 4                   # hex layers across the span
# bilinear (algebraic) ring: a winslow solve tangles the high-aspect cells at the
# sharp ends, so keep the algebraic fill
SMOOTHING_METHOD = "bilinear"
OUT_NAME = "flow_past_plate"

# boundary name -> Nek BC code, applied only at export
GROUPS = {"inlet": "v  ", "outlet": "O  ", "plate": "W  ",
          "top": "SYM", "bottom": "SYM", "front": "SYM", "back": "SYM"}

# -- build the ring section: thin-ellipse body -> named square far field ------
# ellipse sampled at N_THETA uniform direction angles (ray from origin meets it)
theta = np.linspace(0.0, 2.0 * np.pi, N_THETA, endpoint=False)
r = 1.0 / np.sqrt((np.cos(theta) / A) ** 2 + (np.sin(theta) / B) ** 2)
inner = LineMesh.loop(np.column_stack([r * np.cos(theta), r * np.sin(theta),
                                       np.zeros(N_THETA)]),
                      element_tags=["plate"] * N_THETA)

# far-field box built index-aligned to the inner (one box-perimeter point per
# inner-point ray); sides named by the direction they face -- bottom / outlet /
# top / inlet (see flow_past_cylinder.py)
outer = LineMesh.far_field_box(inner, HALF_BOX,
                               side_tags=["bottom", "outlet", "top", "inlet"])

section = QuadMesh.annulus(inner, outer, geometric_spacing(N_RADIAL, RADIAL_GRADING),
                           smoothing_method=SMOOTHING_METHOD)

# -- sweep along the span, naming the end caps front/back --------------------
mesh = HexMesh.extrude(section, axis=(0.0, 0.0, 1.0), length=SPAN,
                       layers=np.linspace(0.0, 1.0, N_SPAN + 1),
                       first_tag="front", last_tag="back")

# -- report + export ---------------------------------------------------------
print(mesh.report())
export.to_re2(mesh, OUT_NAME, groups=GROUPS)
export.to_vtk(mesh, OUT_NAME + ".vtk", groups=GROUPS)
print("groups:", ", ".join(mesh.boundary_group_tags))
