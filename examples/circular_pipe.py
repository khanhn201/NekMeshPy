"""Mesh a straight circular pipe as an all-hex O-grid and export for Nek5000.

Run with::

    python examples/circular_pipe.py

Produces ``circular_pipe.re2`` / ``.rea`` (native Nek) and ``circular_pipe.vtk``.
The cross-section is a "butterfly" O-grid, so there is no degenerate element at
the centre and every hex has a positive scaled Jacobian.
"""

import logging

from nekmeshpy import CircularPipe, export, quality

logging.basicConfig(level=logging.INFO, format="%(message)s")

# radius 0.5, length 5; 40 elements along the axis; O-grid resolution n_side x
# n_radial; cluster cells toward the wall with radial_grading > 1.
pipe = CircularPipe(
    radius=0.5,
    length=5.0,
    n_axial=40,
    n_side=6,          # central square block cells per side
    n_radial=4,        # O-ring layers out to the wall
    center_scale=0.55, # square corner at 55% of the radius
    radial_grading=1.15,
)

mesh = pipe.run()
stats = quality.summary(*mesh.weld()[:2])
print("circular pipe: %d hex elements, %d nodes"
      % (mesh.n_elements, mesh.weld()[2]))
print("scaled Jacobian: min=%.4f mean=%.4f" % (stats["min"], stats["mean"]))

export.to_re2(mesh, "circular_pipe")        # circular_pipe.re2 + circular_pipe.rea
export.to_vtk(mesh, "circular_pipe.vtk")
# any meshio format also works, e.g. export.write(mesh, "circular_pipe.vtu")
print("groups:", ", ".join(g.name for g in mesh.physical_groups))
