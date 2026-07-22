"""Mesh a straight rectangular duct (structured hex) and export for Nek5000.

Run with::

    python examples/rectangular_pipe.py

Produces ``rectangular_pipe.re2`` / ``.rea`` and ``rectangular_pipe.vtk``.
The side faces are tagged ``wall``, the first cap ``inlet`` and the last
``outlet``.
"""

import logging

from nekmeshpy import RectangularPipe, export, quality

logging.basicConfig(level=logging.INFO, format="%(message)s")

# a 2 x 1 duct, length 6, swept along +z; grade cells toward the inlet
duct = RectangularPipe(
    width=2.0,
    height=1.0,
    length=6.0,
    nx=16,
    ny=8,
    n_axial=48,
    axial_grading=0.97,   # <1 clusters cells toward the inlet
)

mesh = duct.run()
stats = quality.summary(*mesh.weld()[:2])
print("rectangular duct: %d hex elements, %d nodes"
      % (mesh.n_elements, mesh.weld()[2]))
print("scaled Jacobian: min=%.4f mean=%.4f" % (stats["min"], stats["mean"]))

export.to_re2(mesh, "rectangular_pipe")
export.to_vtk(mesh, "rectangular_pipe.vtk")
print("groups:", ", ".join(g.name for g in mesh.physical_groups))
