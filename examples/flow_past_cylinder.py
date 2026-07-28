"""Mesh the flow past a circular cylinder (external flow, all-hex O-grid).

A flat, gmsh-style script: edit the constants below and re-run.  The 2-D section is
a ring O-grid built by :meth:`nekmeshpy.QuadMesh.annulus` between the circular body
(inner :class:`~nekmeshpy.LineMesh` loop) and a square far-field box (outer loop) --
"circle blended out to a square at the boundary" -- with the radial layers clustered
toward the cylinder for a boundary layer (``RADIAL_GRADING`` > 1).

The boundaries are named **at the lowest level -- the source line geometry** -- and
carried right through the build: the inner body is ``cylinder`` (one tag per line
element on the inner ``LineMesh.circle``); the four far-field sides are named **on the
outer** :class:`~nekmeshpy.LineMesh` loop itself -- one tag per corner-to-corner line
element (``element_tags=[...]``).  Those element tags survive
:meth:`~nekmeshpy.LineMesh.radial_match`, ``annulus`` copies them onto the inner / outer
ring's edges, and the span sweep (:meth:`nekmeshpy.HexMesh.extrude`) carries them onto
the swept side faces and names the two end caps ``front`` / ``back`` at the hex level --
no post-hoc boundary detection, no by-hand patch assembly.

Run with::

    PYTHONPATH=. python examples/flow_past_cylinder.py

Produces ``flow_past_cylinder.re2`` / ``.rea`` and ``flow_past_cylinder.vtk``.
"""

import logging

import numpy as np

from nekmeshpy import HexMesh, LineMesh, QuadMesh, export
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
SMOOTHING_METHOD = "bilinear"  # per-section interior repositioning
OUT_NAME = "flow_past_cylinder"

# boundary name -> Nek BC code, applied only at export
GROUPS = {"inlet": "v  ", "outlet": "O  ", "cylinder": "W  ",
          "top": "SYM", "bottom": "SYM", "front": "SYM", "back": "SYM"}

# -- build the ring section: circle body -> named square far field ------------
inner = LineMesh.circle(R, N_THETA, element_tags=["cylinder"] * N_THETA)
# The far-field box is a 4-corner loop; each corner-to-corner line element is one box
# side, named here at the source.  Element m runs corner m -> corner m+1, so with
# the corners listed CCW from the lower-left the sides are: bottom (y=-HB), outlet
# (x=+HB, flow exits), top (y=+HB), inlet (x=-HB, flow enters).
outer = LineMesh.loop(
    [(-HALF_BOX, -HALF_BOX, 0.0), (HALF_BOX, -HALF_BOX, 0.0),
     (HALF_BOX, HALF_BOX, 0.0), (-HALF_BOX, HALF_BOX, 0.0)],
    element_tags=["bottom", "outlet", "top", "inlet"],
).radial_match(inner)        # match to the inner's angles; the side names ride along

section = QuadMesh.annulus(inner, outer, geometric_spacing(N_RADIAL, RADIAL_GRADING),
                           smoothing_method=SMOOTHING_METHOD)

# -- sweep along the span, naming the end caps front/back --------------------
# extrude rigidly translates the section along +z, so it stays in the world xy
# plane and the section's edge names ride onto the swept side faces.
mesh = HexMesh.extrude(section, axis=(0.0, 0.0, 1.0), length=SPAN,
                       layers=np.linspace(0.0, 1.0, N_SPAN + 1),
                       first_tag="front", last_tag="back")

# -- report + export ---------------------------------------------------------
print(mesh.report())
export.to_re2(mesh, OUT_NAME, groups=GROUPS)
export.to_vtk(mesh, OUT_NAME + ".vtk", groups=GROUPS)
print("groups:", ", ".join(mesh.boundary_group_tags))
