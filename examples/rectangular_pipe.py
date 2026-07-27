"""Mesh a straight rectangular duct (structured hex) and export for Nek5000.

A flat, gmsh-style script: edit the constants below and re-run.  The cross-section
is a structured quad grid built by :meth:`nekmeshpy.QuadMesh.structured` from the
four rectangle edges (transfinite/Coons interpolation) **in the yz plane at
x=0**, then swept along the duct axis ``+x`` by :meth:`nekmeshpy.HexMesh.extrude`
(a rigid translation of the section in its real 3-D placement).  The duct
therefore runs from the ``inlet`` at ``x=0`` to the ``outlet`` at ``x=LENGTH``.

:meth:`nekmeshpy.QuadMesh.structured` does **not** resample: it uses each edge
curve's own node distribution.  So to cluster cells toward the walls (a
boundary-layer-style grid) we simply sample the four edges at non-uniform
fractions -- here symmetric two-sided geometric clustering via
:meth:`~nekmeshpy.geometry.curve.Curve.resample` -- and the Coons patch
carries that grading into the section exactly.  ``WALL_GRADING`` controls how
strongly the cells thin toward the walls (``1.0`` = uniform).

Run with::

    PYTHONPATH=. python examples/rectangular_pipe.py

Produces ``rectangular_pipe.re2`` / ``.rea`` and ``rectangular_pipe.vtk``.
"""

import logging

import numpy as np

from nekmeshpy import Curve, HexMesh, QuadMesh, export
from nekmeshpy.model.fields import geometric_spacing

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
INTERIOR_METHOD = "bilinear"     # no-op: keep the exact (graded) Coons section
OUT_NAME = "rectangular_pipe"

# boundary name -> Nek BC code, applied only at export
GROUPS = {"wall": "W  ", "inlet": "v  ", "outlet": "O  "}


def wall_clustered(n: int, ratio: float) -> np.ndarray:
    """``n+1`` symmetric fractions in ``[0,1]`` clustered toward *both* ends (a
    two-sided near-wall distribution): geometric spacing on each half, mirrored.
    ``n`` should be even so the two halves match exactly."""
    m = n // 2
    g = geometric_spacing(m, ratio)                  # [0,1], cells grow away from 0
    first = 0.5 * g                                  # [0, 0.5], clustered near 0
    second = 1.0 - 0.5 * g[::-1]                      # [0.5, 1], clustered near 1
    return np.concatenate([first[:-1], second])      # 2*m+1 = n+1 fractions


# -- build the structured cross-section, then extrude it along the axis -------
# four rectangle corners (CCW) in the yz plane at x=0; sample each edge at the
# wall-clustered fractions so structured (which uses the edges' own nodes, no
# resampling) grades the section.  The section lives in 3-D and extrude sweeps it
# rigidly along +x, so no reorientation of the section is needed.
c0 = (0.0, -WIDTH / 2, -HEIGHT / 2)
c1 = (0.0, WIDTH / 2, -HEIGHT / 2)
c2 = (0.0, WIDTH / 2, HEIGHT / 2)
c3 = (0.0, -WIDTH / 2, HEIGHT / 2)
xf = wall_clustered(NX, WALL_GRADING)        # width-direction node fractions
yf = wall_clustered(NY, WALL_GRADING)        # height-direction node fractions
edges = [Curve([c0, c1]).resample(xf), Curve([c1, c2]).resample(yf),
         Curve([c2, c3]).resample(xf), Curve([c3, c0]).resample(yf)]
section = QuadMesh.structured(
    edges, interior_method=INTERIOR_METHOD,
    boundary_names={"bottom": "wall", "right": "wall", "top": "wall", "left": "wall"})

mesh = HexMesh.extrude(
    section, axis=AXIS, length=LENGTH,
    layers=geometric_spacing(N_AXIAL, AXIAL_GRADING),
    origin=CENTER, first_cap="inlet", last_cap="outlet")

# -- report + export ---------------------------------------------------------
stats = mesh.quality_summary()
print("rectangular duct: %d hex elements, %d points" % (mesh.n_hexes, mesh.n_points))
print("scaled Jacobian: min=%.4f mean=%.4f" % (stats["min"], stats["mean"]))

export.to_re2(mesh, OUT_NAME, groups=GROUPS)
export.to_vtk(mesh, OUT_NAME + ".vtk", groups=GROUPS)
print("groups:", ", ".join(mesh.boundary_group_names))
