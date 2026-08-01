"""Flow past a circular cylinder (external flow, all-hex O-grid).

:meth:`QuadMesh.annulus` builds a ring O-grid between the circular body (inner
:class:`LineMesh` loop) and a square far-field box (outer loop), radial layers
clustered toward the cylinder (``RADIAL_GRADING`` > 1).

Boundaries are named on the source line geometry: the body is ``cylinder`` (per
line element on the inner loop); the four far-field sides are tagged per side by
:meth:`LineMesh.rectangle`. The circle is rotated (``start_theta``) so its index 0
meets the box's lower-left corner, so the two loops carry the same point count and
pair index-for-index in ``annulus`` (the radial spokes are not straight, but the
mesh conforms). ``annulus`` copies the tags onto the ring edges, and the span sweep
(:meth:`HexMesh.extrude`) carries them onto the side faces and names the caps
``front`` / ``back``.

    PYTHONPATH=. python examples/flow_past_cylinder.py

Produces ``flow_past_cylinder.re2`` and ``flow_past_cylinder.vtu``.
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
N_THETA = 24                 # azimuthal cells around the cylinder
N_RADIAL = 24                # radial cells from cylinder out to the box
RADIAL_GRADING = 1.2        # >1 clusters radial layers toward the cylinder
N_SPAN = 4                   # hex layers across the span
ORDER = 2                    # polynomial order; 1 = linear. High order bows the
                             # ring's inner wall onto the true circle (curved
                             # .vtu; .re2 stays linear either way)
SMOOTHING_METHOD = None  # per-section interior repositioning
OUT_NAME = "flow_past_cylinder"

# boundary name -> Nek BC code, applied only at export
GROUPS = {"inlet": "v  ", "outlet": "O  ", "cylinder": "W  ",
          "top": "SYM", "bottom": "SYM", "front": "SYM", "back": "SYM"}

# -- build the ring section: circle body -> named square far field ------------
# rotate the circle so its index 0 meets the box's lower-left corner, so the two
# loops pair index-for-index in annulus (the radial spokes are not straight)
CORNER = np.arctan2(-HALF_BOX, -HALF_BOX)
inner = LineMesh.circle(R, N_THETA, start_theta=CORNER,
                        element_tags=["cylinder"] * N_THETA, order=ORDER)
# square far field discretized into N_THETA line elements (N_THETA/4 per side),
# sides named bottom (y=-HB), outlet (x=+HB), top (y=+HB), inlet (x=-HB)
outer = LineMesh.rectangle(2 * HALF_BOX, 2 * HALF_BOX, N_THETA, order=ORDER,
                           side_tags={"bottom": "bottom", "right": "outlet", "top": "top", "left": "inlet"})

section = QuadMesh.annulus(inner, outer, geometric_spacing(N_RADIAL, RADIAL_GRADING),
                           smoothing_method=SMOOTHING_METHOD)

# -- sweep along the span, naming the end caps front/back --------------------
# extrude translates the section along +z; edge names ride onto the side faces
mesh = HexMesh.extrude(section, axis=(0.0, 0.0, 1.0), length=SPAN,
                       layers=np.linspace(0.0, 1.0, N_SPAN + 1),
                       first_tag="front", last_tag="back")

# -- report + export ---------------------------------------------------------
print(mesh.report())
export.to_re2(mesh, OUT_NAME + ".re2", groups=GROUPS)
export.to_vtu(mesh, OUT_NAME + ".vtu", groups=GROUPS)
print("groups:", ", ".join(mesh.boundary_group_tags))
