"""Mesh a straight circular pipe as an all-hex O-grid and export for Nek5000.

A flat, gmsh-style script: edit the constants below and re-run.  The cross-section
is a "butterfly" O-grid built by :meth:`nekmeshpy.QuadMesh.ogrid` from the circle
boundary, then extruded along the axis by :meth:`nekmeshpy.HexMesh.extrude`.

Run with::

    PYTHONPATH=. python examples/circular_pipe.py

Produces ``circular_pipe.re2`` / ``.rea`` and ``circular_pipe.vtk``.
"""

import logging

from nekmeshpy import CurveLoop, HexMesh, QuadMesh, export
from nekmeshpy.model.fields import geometric_spacing

logging.basicConfig(level=logging.INFO, format="%(message)s")

# -- parameters --------------------------------------------------------------
RADIUS = 0.5
LENGTH = 5.0
N_AXIAL = 40                 # hex layers along the axis
N_SIDE = 6                   # central square block cells per side
N_RADIAL = 4                 # O-ring layers out to the wall
CENTER_SCALE = 0.55
RADIAL_GRADING = 1.15        # >1 clusters cells toward the wall
AXIAL_GRADING = 1.0
AXIS = (0.0, 0.0, 1.0)
CENTER = (0.0, 0.0, 0.0)
INTERIOR_METHOD = "conduction"   # per-section interior; raises the O-grid quality
OUT_NAME = "circular_pipe"

# boundary name -> Nek BC code, applied only at export
GROUPS = {"wall": "W  ", "inlet": "v  ", "outlet": "O  "}

# -- build the O-grid cross-section, then extrude it along the axis -----------
# the section's interior is filled + repositioned here (wall held fixed); the
# straight extrude then just copies it along the axis
section = QuadMesh.ogrid(
    CurveLoop.circle(RADIUS, 4 * N_SIDE), N_SIDE,
    geometric_spacing(N_RADIAL, RADIAL_GRADING),
    center_scale=CENTER_SCALE, wall_name="wall", interior_method=INTERIOR_METHOD)

mesh = HexMesh.extrude(
    section, axis=AXIS, length=LENGTH,
    layers=geometric_spacing(N_AXIAL, AXIAL_GRADING),
    origin=CENTER, first_cap="inlet", last_cap="outlet")

# -- report + export ---------------------------------------------------------
stats = mesh.quality_summary()
print("circular pipe: %d hex elements, %d points" % (mesh.n_hexes, mesh.n_points))
print("scaled Jacobian: min=%.4f mean=%.4f" % (stats["min"], stats["mean"]))

export.to_re2(mesh, OUT_NAME, groups=GROUPS)
export.to_vtk(mesh, OUT_NAME + ".vtk", groups=GROUPS)
print("groups:", ", ".join(mesh.boundary_group_names))
