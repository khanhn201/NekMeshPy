"""Mesh the flow past a half-cylinder sitting on the ground (external flow).

A flat, gmsh-style script: edit the constants below and re-run.  A half-cylinder of
radius ``R`` rests on the floor and the flow fills the channel above, up to a flat
ceiling at ``H`` -- the classic "bump in a channel".

The 2-D section is a **single** transfinite block
(:meth:`nekmeshpy.QuadMesh.structured`) whose bottom edge is one composite curve:
flat ground ``[-W,-R]``, then the semicircular bump up and over ``-R..R``, then flat
ground ``[R,W]``.  Putting the semicircle in the *middle* of one edge (rather than
splitting the domain into three blocks) avoids the degenerate vertical-tangent
corner a block split would create where the bump meets the ground at ``x = +/-R``;
``smoothing_method="bilinear"`` then fills the grid over the bump.  Boundaries
are named **at the lowest level -- each edge line tags itself**: the bottom edge
(ground + bump) is ``wall``, ``inlet`` / ``outlet`` on the ends, ``top`` on the
ceiling; ``structured`` reads each side's uniform edge tag, and the span sweep
(:meth:`nekmeshpy.HexMesh.loft`) names the two end caps ``front`` / ``back`` at the
hex level.

Run with::

    PYTHONPATH=. python examples/flow_past_half_cylinder.py

Produces ``flow_past_half_cylinder.re2`` / ``.rea`` and ``.vtk``.
"""

import logging

import numpy as np

from nekmeshpy import HexMesh, LineMesh, QuadMesh, export

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
# structured uses the edges' own nodes (no resampling): resample each edge to the
# matching division count -- bottom/top to NX+1, left/right to NY+1.  Each edge tags
# itself at the line level (the tag rides through resample), so structured names each
# side from its edge's uniform tag -- no boundary_tags override.
bottom = LineMesh.open(bottom_pts, element_tags=["wall"] * (len(bottom_pts) - 1)
                      ).resample(np.linspace(0.0, 1.0, NX + 1))              # (-W,0)->(W,0)
right = LineMesh.open([(W, 0.0, 0.0), (W, H, 0.0)], element_tags=["outlet"]
                     ).resample(np.linspace(0.0, 1.0, NY + 1))
top = LineMesh.open([(W, H, 0.0), (-W, H, 0.0)], element_tags=["top"]
                   ).resample(np.linspace(0.0, 1.0, NX + 1))
left = LineMesh.open([(-W, H, 0.0), (-W, 0.0, 0.0)], element_tags=["inlet"]
                    ).resample(np.linspace(0.0, 1.0, NY + 1))
section = QuadMesh.structured([bottom, right, top, left], smoothing_method="bilinear")

# -- sweep along the span, naming the end caps front/back --------------------
zs = np.linspace(0.0, SPAN, N_SPAN + 1)
slices = [QuadMesh(section.points + np.array([0.0, 0.0, z]), section.quads,
                   boundaries=section.boundaries, boundary_tags=section.boundary_tags)
          for z in zs]
mesh = HexMesh.loft(slices, first_tag="front", last_tag="back")

# -- report + export ---------------------------------------------------------
print(mesh.report())
export.to_re2(mesh, OUT_NAME, groups=GROUPS)
export.to_vtk(mesh, OUT_NAME + ".vtk", groups=GROUPS)
print("groups:", ", ".join(mesh.boundary_group_tags))
