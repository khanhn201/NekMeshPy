"""Straight circular pipe as an all-hex O-grid, exported for Nek5000.

:meth:`QuadMesh.ogrid` fills the circle boundary; :meth:`HexMesh.extrude` sweeps
it along the axis.

    PYTHONPATH=. python examples/circular_pipe.py

Produces ``circular_pipe.re2`` and ``circular_pipe.vtu``.
"""

import logging

from nekmeshpy import hexmesh, linemesh, quadmesh, writer
from nekmeshpy.core.fields import geometric_spacing

logging.basicConfig(level=logging.INFO, format="%(message)s")

# -- parameters --------------------------------------------------------------
RADIUS = 0.5
LENGTH = 5.0
N_AXIAL = 40                 # hex layers along the axis
N_SIDE = 6                   # central square block cells per side
N_RADIAL = 8                 # O-ring layers out to the wall
CENTER_SCALE = 0.7
RADIAL_GRADING = 0.5        # <1 clusters cells toward the wall
AXIAL_GRADING = 1.0
ORDER = 2                    # polynomial order; 1 = linear. High order bows the
                            # O-ring wall onto the true circle (curved .vtu;
                            # .re2 stays linear either way)
AXIS = (0.0, 0.0, 1.0)
CENTER = (0.0, 0.0, 0.0)
OUT_NAME = "circular_pipe"

# boundary name -> Nek BC code, applied only at export
GROUPS = {"wall": "W  ", "inlet": "v  ", "outlet": "O  "}

# -- build the O-grid cross-section, then extrude it along the axis -----------
# interior filled + repositioned (wall fixed); extrude copies it along the axis
wall = linemesh.circle(RADIUS, 4 * N_SIDE,
                       element_tag="wall", order=ORDER)
radial = geometric_spacing(N_RADIAL, RADIAL_GRADING)

section = quadmesh.ogrid(wall, N_SIDE, radial,
                         center_scale=CENTER_SCALE, quadrant_scale=CENTER_SCALE)

mesh = hexmesh.extrude(
    section, axis=AXIS, length=LENGTH,
    layers=geometric_spacing(N_AXIAL, AXIAL_GRADING),
    origin=CENTER, first_tag="inlet", last_tag="outlet")

# -- report + export ---------------------------------------------------------
stats = hexmesh.quality_summary(mesh)
print("circular pipe: %d hex elements, %d points" % (mesh.n_hexes, mesh.n_points))
print("scaled Jacobian: min=%.4f mean=%.4f" % (stats.min, stats.mean))

writer.to_re2(mesh, OUT_NAME + ".re2", groups=GROUPS)
writer.to_vtu(mesh, OUT_NAME + ".vtu", groups=GROUPS)  # XML: renders curved cells
print("groups:", ", ".join(mesh.face_group_tags))
