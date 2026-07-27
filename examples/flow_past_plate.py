"""Mesh the flow past a thin plate (external flow, all-hex O-grid).

A flat, gmsh-style script: edit the constants below and re-run.  Same idea as
``flow_past_cylinder.py`` -- a ring O-grid (:meth:`nekmeshpy.QuadMesh.annulus`)
from the body out to a square far-field box, swept along the span, with the
boundaries named **as the mesh is built** (inner ring ``plate``; outer square split
into ``inlet`` / ``outlet`` / ``top`` / ``bottom``; span caps ``front`` / ``back``).
The only change from the cylinder is the body: a **thin ellipse** (half-length
``A``, half-thickness ``B`` with ``B << A``) standing in for a flat plate.

*Modeling choice:* a true zero-thickness plate would need a C-grid the toolkit does
not provide; a thin ellipse keeps a smooth, everywhere-non-degenerate all-hex O-grid
around the body while approximating the plate as ``B`` -> 0.

Run with::

    PYTHONPATH=. python examples/flow_past_plate.py

Produces ``flow_past_plate.re2`` / ``.rea`` and ``flow_past_plate.vtk``.
"""

import logging

import numpy as np

from nekmeshpy import CurveLoop, HexMesh, QuadMesh, export
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
# raw algebraic (radially-graded) ring: an elliptic (winslow) solve tangles the
# very high aspect-ratio cells near the sharp ends, so keep the algebraic fill.
INTERIOR_METHOD = None
OUT_NAME = "flow_past_plate"

# boundary name -> Nek BC code, applied only at export
GROUPS = {"inlet": "v  ", "outlet": "O  ", "plate": "W  ",
          "top": "SYM", "bottom": "SYM", "front": "SYM", "back": "SYM"}

# -- build the ring section: thin-ellipse body -> square far field ------------
# sample the ellipse at even *geometric* angles (not even parameter theta) so the
# ring is radially aligned with the square just like the circle in
# flow_past_cylinder -- a node lands on each box corner (no face straddles a
# corner) and the azimuthal spacing does not collapse at the sharp ends.
phi = np.linspace(0.0, 2.0 * np.pi, N_THETA, endpoint=False)
r = 1.0 / np.sqrt((np.cos(phi) / A) ** 2 + (np.sin(phi) / B) ** 2)
inner = CurveLoop(np.column_stack([r * np.cos(phi), r * np.sin(phi), np.zeros(N_THETA)]))
# annulus pairs inner/outer by index (no resampling): project the 4-corner box loop
# onto the inner's azimuthal angles so the two rings match point-for-point.
outer = CurveLoop([(-HALF_BOX, -HALF_BOX, 0.0), (HALF_BOX, -HALF_BOX, 0.0),
                   (HALF_BOX, HALF_BOX, 0.0), (-HALF_BOX, HALF_BOX, 0.0)]
                  ).radial_match(inner)
section = QuadMesh.annulus(inner, outer, geometric_spacing(N_RADIAL, RADIAL_GRADING),
                           interior_method=INTERIOR_METHOD, inner_name="plate",
                           outer_name={"x_min": "inlet", "x_max": "outlet",
                                       "y_min": "bottom", "y_max": "top"})

# -- sweep along the span, naming the end caps front/back --------------------
zs = np.linspace(0.0, SPAN, N_SPAN + 1)
slices = [QuadMesh(section.points + np.array([0.0, 0.0, z]), section.quads,
                   boundaries=section.boundaries, boundary_names=section.boundary_names)
          for z in zs]
mesh = HexMesh.loft(slices, first_cap="front", last_cap="back")

# -- report + export ---------------------------------------------------------
print(mesh.report())
export.to_re2(mesh, OUT_NAME, groups=GROUPS)
export.to_vtk(mesh, OUT_NAME + ".vtk", groups=GROUPS)
print("groups:", ", ".join(mesh.boundary_group_names))
