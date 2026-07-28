"""Straight rectangular duct (structured hex), exported for Nek5000.

:meth:`QuadMesh.structured` builds a Coons patch from the four rectangle edges in
the yz plane at x=0; :meth:`HexMesh.extrude` sweeps it along ``+x`` (inlet at
``x=0``, outlet at ``x=LENGTH``).

``structured`` does not resample -- it uses each edge's own node distribution. To
cluster cells toward the walls we sample the edges at non-uniform fractions
(symmetric two-sided geometric clustering) and the Coons patch carries the grading
through. ``WALL_GRADING`` sets the strength (``1.0`` = uniform).

    PYTHONPATH=. python examples/rectangular_pipe.py

Produces ``rectangular_pipe.re2`` / ``.rea`` and ``rectangular_pipe.vtk``.
"""

import logging

from nekmeshpy import HexMesh, QuadMesh, export
from nekmeshpy.model.fields import geometric_spacing, symmetric_spacing

logging.basicConfig(level=logging.INFO, format="%(message)s")

# -- parameters --------------------------------------------------------------
WIDTH = 2.0                  # cross-section extent along y
HEIGHT = 1.0                 # cross-section extent along z
LENGTH = 6.0                 # duct length along the axis (+x)
NX = 16                      # cells across the width (even: symmetric clustering)
NY = 8                       # cells across the height (even: symmetric clustering)
N_AXIAL = 48                 # hex layers along the axis
AXIAL_GRADING = 0.97         # <1 clusters cells toward the inlet
WALL_GRADING = 1.15          # >1 thins cross-section cells toward the walls
AXIS = (1.0, 0.0, 0.0)       # sweep direction: down the duct (+x)
CENTER = (0.0, 0.0, 0.0)
SMOOTHING_METHOD = "bilinear"     # no-op: keep the exact (graded) Coons section
OUT_NAME = "rectangular_pipe"

# boundary name -> Nek BC code, applied only at export
GROUPS = {"wall": "W  ", "inlet": "v  ", "outlet": "O  "}


# -- build the structured cross-section, then extrude it along the axis -------
# four rectangle corners (CCW) in the yz plane at x=0; the symmetric wall-clustered
# node fractions grade the section toward the walls.  All four sides are the wall.
corners = [(0.0, -WIDTH / 2, -HEIGHT / 2), (0.0, WIDTH / 2, -HEIGHT / 2),
           (0.0, WIDTH / 2, HEIGHT / 2), (0.0, -WIDTH / 2, HEIGHT / 2)]
section = QuadMesh.rectangle(
    corners, NX, NY,
    x_frac=symmetric_spacing(NX, WALL_GRADING),   # width-direction node fractions
    y_frac=symmetric_spacing(NY, WALL_GRADING),   # height-direction node fractions
    side_tags={s: "wall" for s in ("bottom", "right", "top", "left")},
    smoothing_method=SMOOTHING_METHOD)

mesh = HexMesh.extrude(
    section, axis=AXIS, length=LENGTH,
    layers=geometric_spacing(N_AXIAL, AXIAL_GRADING),
    origin=CENTER, first_tag="inlet", last_tag="outlet")

# -- report + export ---------------------------------------------------------
stats = mesh.quality_summary()
print("rectangular duct: %d hex elements, %d points" % (mesh.n_hexes, mesh.n_points))
print("scaled Jacobian: min=%.4f mean=%.4f" % (stats["min"], stats["mean"]))

export.to_re2(mesh, OUT_NAME, groups=GROUPS)
export.to_vtk(mesh, OUT_NAME + ".vtk", groups=GROUPS)
print("groups:", ", ".join(mesh.boundary_group_tags))
