"""Mesh the flow past a circular cylinder (external flow, all-hex O-grid).

A flat, gmsh-style script: edit the constants below and re-run.  The 2-D section is
a ring O-grid built by :meth:`nekmeshpy.QuadMesh.annulus` between the circular body
(inner :class:`~nekmeshpy.CurveLoop`) and a square far-field box (outer loop) --
"circle extruded to a square at the boundary" -- with the radial layers clustered
toward the cylinder for a boundary layer (``RADIAL_GRADING`` > 1).  The boundaries
are named **as the mesh is built**: ``annulus`` tags the inner ring ``cylinder``
and splits the outer square into ``inlet`` / ``outlet`` / ``top`` / ``bottom``, and
the span sweep (:meth:`nekmeshpy.HexMesh.extrude`) names the two end caps
``front`` / ``back`` -- no post-hoc boundary detection.

Run with::

    PYTHONPATH=. python examples/flow_past_cylinder.py

Produces ``flow_past_cylinder.re2`` / ``.rea`` and ``flow_past_cylinder.vtk``.
"""

import logging

import numpy as np

from nekmeshpy import CurveLoop, HexMesh, QuadMesh, export
from nekmeshpy.model.fields import geometric_spacing

logging.basicConfig(level=logging.INFO, format="%(message)s")

# -- parameters --------------------------------------------------------------
R = 0.5                      # cylinder radius
HALF_BOX = 6.0               # far-field box half-width (domain [-HB, HB]^2 in xy)
SPAN = 2.0                   # spanwise (z) extent
N_THETA = 64                 # azimuthal cells around the cylinder
N_RADIAL = 24                # radial cells from cylinder out to the box
RADIAL_GRADING = 1.12        # >1 clusters radial layers toward the cylinder
N_SPAN = 4                   # hex layers across the span
INTERIOR_METHOD = "winslow"  # relax the ring interior (rings held fixed)
OUT_NAME = "flow_past_cylinder"

# boundary name -> Nek BC code, applied only at export
GROUPS = {"inlet": "v  ", "outlet": "O  ", "cylinder": "W  ",
          "top": "SYM", "bottom": "SYM", "front": "SYM", "back": "SYM"}

# -- build the ring section: circle body -> square far field ------------------
inner = CurveLoop.circle(R, N_THETA)
# annulus pairs inner/outer by index (no resampling): project the 4-corner box loop
# onto the inner's azimuthal angles so the two rings match point-for-point.
outer = CurveLoop([(-HALF_BOX, -HALF_BOX, 0.0), (HALF_BOX, -HALF_BOX, 0.0),
                   (HALF_BOX, HALF_BOX, 0.0), (-HALF_BOX, HALF_BOX, 0.0)]
                  ).radial_match(inner)
section = QuadMesh.annulus(inner, outer, geometric_spacing(N_RADIAL, RADIAL_GRADING),
                           interior_method=INTERIOR_METHOD, inner_name="cylinder",
                           outer_name={"x_min": "inlet", "x_max": "outlet",
                                       "y_min": "bottom", "y_max": "top"})

# -- sweep along the span, naming the end caps front/back --------------------
# extrude rigidly translates the section along +z, so it stays in the world xy
# plane and the section's edge names ride onto the swept side faces.
mesh = HexMesh.extrude(section, axis=(0.0, 0.0, 1.0), length=SPAN,
                       layers=np.linspace(0.0, 1.0, N_SPAN + 1),
                       first_cap="front", last_cap="back")

# -- report + export ---------------------------------------------------------
print(mesh.report())
export.to_re2(mesh, OUT_NAME, groups=GROUPS)
export.to_vtk(mesh, OUT_NAME + ".vtk", groups=GROUPS)
print("groups:", ", ".join(mesh.boundary_group_names))
