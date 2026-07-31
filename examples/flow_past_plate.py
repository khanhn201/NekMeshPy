"""Flow past a thin plate (external flow, all-hex O-grid).

Same construction as ``flow_past_cylinder.py`` -- a ring O-grid
(:meth:`QuadMesh.annulus`) between the body and a square far-field box, swept along
the span -- but the body is a **thin ellipse** (half-length ``A``, half-thickness
``B << A``) standing in for a flat plate. A true zero-thickness plate would need a
C-grid; the thin ellipse keeps a non-degenerate O-grid as ``B`` -> 0.

Boundaries are named on the source line geometry (see ``flow_past_cylinder.py``):
body ``plate`` on the inner loop, far-field sides tagged per side by
:meth:`LineMesh.rectangle` (the ellipse angles are offset so index 0 meets the
box's lower-left corner, pairing the loops index-for-index), carried through
``annulus`` -> :meth:`HexMesh.extrude`; the sweep names the caps ``front`` /
``back``.

    PYTHONPATH=. python examples/flow_past_plate.py

Produces ``flow_past_plate.re2`` and ``flow_past_plate.vtu``.
"""

import logging

import numpy as np

from nekmeshpy import HexMesh, LineMesh, QuadMesh, export
from nekmeshpy.model.fields import geometric_spacing, gll_nodes

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
ORDER = 2                    # polynomial order; 1 = linear. "bilinear" is a no-op
                             # fill, so it stays legal above order 1 (a
                             # repositioning smoother would be rejected)
OUT_NAME = "flow_past_plate"

# boundary name -> Nek BC code, applied only at export
GROUPS = {"inlet": "v  ", "outlet": "O  ", "plate": "W  ",
          "top": "SYM", "bottom": "SYM", "front": "SYM", "back": "SYM"}

# -- build the ring section: thin-ellipse body -> named square far field ------
# ellipse sampled at N_THETA uniform direction angles, offset so index 0 meets the
# box's lower-left corner (so the loops pair index-for-index in annulus)
CORNER = np.arctan2(-HALF_BOX, -HALF_BOX)
theta = np.linspace(0.0, 2.0 * np.pi, N_THETA, endpoint=False) + CORNER


def ellipse(th):
    """Points on the **exact** ellipse at direction angles ``th`` (any shape); the
    trailing axis of the result is xyz."""
    rad = 1.0 / np.sqrt((np.cos(th) / A) ** 2 + (np.sin(th) / B) ** 2)
    return np.stack([rad * np.cos(th), rad * np.sin(th), np.zeros_like(th)], axis=-1)


# There is no analytic ``LineMesh`` factory for an ellipse (``circle`` is the only
# one), so place the high-order nodes here the way ``circle`` does: element k spans
# theta[k] .. theta[k] + dtheta, and its interior GLL nodes go on the exact ellipse
# rather than on ``LineMesh.loft``'s default straight chord.  Without this the wall
# would be high-order in storage and linear in geometry.
interior = (None if ORDER == 1 else
            ellipse(theta[:, None]
                    + gll_nodes(ORDER)[1:ORDER][None, :] * (2.0 * np.pi / N_THETA)))
inner = LineMesh.loft(ellipse(theta), element_tags=["plate"] * N_THETA,
                      order=ORDER, interior=interior, loop=True)

# square far field discretized into N_THETA line elements (N_THETA/4 per side),
# sides named by the direction they face -- bottom / outlet / top / inlet
outer = LineMesh.rectangle(2 * HALF_BOX, 2 * HALF_BOX, N_THETA, order=ORDER,
                           side_tags=["bottom", "outlet", "top", "inlet"])

section = QuadMesh.annulus(inner, outer, geometric_spacing(N_RADIAL, RADIAL_GRADING),
                           smoothing_method=SMOOTHING_METHOD)

# -- sweep along the span, naming the end caps front/back --------------------
mesh = HexMesh.extrude(section, axis=(0.0, 0.0, 1.0), length=SPAN,
                       layers=np.linspace(0.0, 1.0, N_SPAN + 1),
                       first_tag="front", last_tag="back")

# -- report + export ---------------------------------------------------------
print(mesh.report())
export.to_re2(mesh, OUT_NAME + ".re2", groups=GROUPS)
export.to_vtu(mesh, OUT_NAME + ".vtu", groups=GROUPS)
print("groups:", ", ".join(mesh.boundary_group_tags))
