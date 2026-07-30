"""Flow past a half-cylinder on the ground (external flow).

A half-cylinder of radius ``R`` rests on the floor; flow fills the channel up to a
flat ceiling at ``H`` -- the classic "bump in a channel".

The 2-D section is a **single** transfinite block (:meth:`QuadMesh.structured`)
whose bottom edge is one composite curve: flat ground ``[-W,-R]``, semicircular
bump ``-R..R``, flat ground ``[R,W]``. Keeping the semicircle mid-edge (vs a
three-block split) avoids the degenerate corner where bump meets ground at
``x = +/-R``; ``smoothing_method="bilinear"`` fills the grid over the bump.

Each edge line tags itself: bottom (ground + bump) ``wall``, ends ``inlet`` /
``outlet``, ceiling ``top``. The span sweep (:meth:`HexMesh.extrude`) names the caps
``front`` / ``back``.

    PYTHONPATH=. python examples/flow_past_half_cylinder.py

Produces ``flow_past_half_cylinder.re2`` / ``.rea`` and ``.vtu``.
"""

import logging

import numpy as np

from nekmeshpy import HexMesh, LineMesh, QuadMesh, export, trimesh

logging.basicConfig(level=logging.INFO, format="%(message)s")

# -- parameters --------------------------------------------------------------
R = 0.5                      # half-cylinder radius (sits on the floor y=0)
W = 5.0                      # channel half-length (inlet at -W, outlet at +W)
H = 3.0                      # channel height (flat ceiling)
SPAN = 2.0                   # spanwise (z) extent
NX = 120                     # cells along the channel (bottom edge)
NY = 40                      # cells from floor/bump to ceiling
N_SPAN = 4                   # hex layers across the span
N_ARC = 160                  # sample points on the semicircular bump
OUT_NAME = "flow_past_half_cylinder"

# boundary name -> Nek BC code, applied only at export
GROUPS = {"inlet": "v  ", "outlet": "O  ", "wall": "W  ",
          "top": "SYM", "front": "SYM", "back": "SYM"}

# -- composite bottom edge: ground -> semicircular bump -> ground -------------
theta = np.linspace(np.pi, 0.0, N_ARC)                     # -R..R over the top
z = np.zeros(N_ARC)
bump = np.column_stack([R * np.cos(theta), R * np.sin(theta), z])
left_ground = np.column_stack([np.linspace(-W, -R, 40), np.zeros(40), np.zeros(40)])
right_ground = np.column_stack([np.linspace(R, W, 40), np.zeros(40), np.zeros(40)])
bottom_pts = np.vstack([left_ground[:-1], bump, right_ground[1:]])   # drop shared ends
# the composite bottom (ground -> bump -> ground) has no closed-form arc length, so
# sample it by arc length to the exact NX+1 edge points; tag the whole edge "wall"
# at the line level so structured names the bottom side from it.
pts = trimesh.ops.resample_polyline(bottom_pts, np.linspace(0.0, 1.0, NX + 1))   # (-W,0)->(W,0)
bottom = LineMesh.open(pts, element_tags=["wall"] * NX)
right = LineMesh.line((W, 0.0, 0.0), (W, H, 0.0),
                      np.linspace(0.0, 1.0, NY + 1), element_tag="outlet")
top = LineMesh.line((W, H, 0.0), (-W, H, 0.0),
                    np.linspace(0.0, 1.0, NX + 1), element_tag="top")
left = LineMesh.line((-W, H, 0.0), (-W, 0.0, 0.0),
                     np.linspace(0.0, 1.0, NY + 1), element_tag="inlet")
section = QuadMesh.structured([bottom, right, top, left], smoothing_method="bilinear")

# -- sweep along the span, naming the end caps front/back --------------------
# extrude translates the section along +z; edge names ride onto the side faces
mesh = HexMesh.extrude(section, axis=(0.0, 0.0, 1.0), length=SPAN,
                       layers=np.linspace(0.0, 1.0, N_SPAN + 1),
                       first_tag="front", last_tag="back")

# -- report + export ---------------------------------------------------------
print(mesh.report())
export.to_re2(mesh, OUT_NAME, groups=GROUPS)
export.to_vtu(mesh, OUT_NAME + ".vtu", groups=GROUPS)
print("groups:", ", ".join(mesh.boundary_group_tags))
