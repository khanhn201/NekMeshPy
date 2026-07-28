"""Mesh the flow past a thin plate (external flow, all-hex O-grid).

A flat, gmsh-style script: edit the constants below and re-run.  Same construction
as ``flow_past_cylinder.py`` -- a ring O-grid (:meth:`nekmeshpy.QuadMesh.annulus`)
between the body and a square far-field box, swept along the span -- the only change
being the body: a **thin ellipse** (half-length ``A``, half-thickness ``B`` with
``B << A``) standing in for a flat plate.

*Modeling choice:* a true zero-thickness plate would need a C-grid the toolkit does
not provide; a thin ellipse keeps a smooth, everywhere-non-degenerate all-hex O-grid
around the body while approximating the plate as ``B`` -> 0.

The boundaries are named **at the lowest level -- the source line geometry** (see
``flow_past_cylinder.py``): the body is ``plate`` (one tag per line element on the inner
loop) and the four far-field sides are named on the outer :class:`~nekmeshpy.LineMesh`
loop's line elements, carried through :meth:`~nekmeshpy.LineMesh.radial_match` ->
``annulus`` -> :meth:`nekmeshpy.HexMesh.extrude` onto the swept side faces; the sweep
names the span caps ``front`` / ``back`` at the hex level.

Run with::

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
# bilinear (algebraic, radially-graded) ring: an elliptic (winslow) solve tangles
# the very high aspect-ratio cells near the sharp ends, so keep the algebraic fill.
SMOOTHING_METHOD = "bilinear"
OUT_NAME = "flow_past_plate"

# boundary name -> Nek BC code, applied only at export
GROUPS = {"inlet": "v  ", "outlet": "O  ", "plate": "W  ",
          "top": "SYM", "bottom": "SYM", "front": "SYM", "back": "SYM"}

# -- build the ring section: thin-ellipse body -> named square far field ------
# the ellipse sampled at N_THETA uniform direction angles (the ray from the origin
# at angle theta meets the ellipse), so radial_match below lands equal cells per side.
theta = np.linspace(0.0, 2.0 * np.pi, N_THETA, endpoint=False)
r = 1.0 / np.sqrt((np.cos(theta) / A) ** 2 + (np.sin(theta) / B) ** 2)
inner = LineMesh.loop(np.column_stack([r * np.cos(theta), r * np.sin(theta),
                                       np.zeros(N_THETA)]),
                      element_tags=["plate"] * N_THETA)

# named far-field box: line element m (corner m -> corner m+1) is one box side (see
# flow_past_cylinder.py) -- bottom / outlet / top / inlet, CCW from the lower-left.
outer = LineMesh.loop(
    [(-HALF_BOX, -HALF_BOX, 0.0), (HALF_BOX, -HALF_BOX, 0.0),
     (HALF_BOX, HALF_BOX, 0.0), (-HALF_BOX, HALF_BOX, 0.0)],
    element_tags=["bottom", "outlet", "top", "inlet"],
).radial_match(inner)

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
