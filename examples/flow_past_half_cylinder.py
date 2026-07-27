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
``interior_method="winslow"`` then regularizes the grid over the bump.  Boundaries
are named **as the mesh is built**: the section tags its own sides (bottom edge --
ground + bump -- ``wall``, ``inlet`` / ``outlet`` on the ends, ``top`` on the
ceiling) and the span sweep (:meth:`nekmeshpy.HexMesh.loft`) names the two end caps
``front`` / ``back``.

Run with::

    PYTHONPATH=. python examples/flow_past_half_cylinder.py

Produces ``flow_past_half_cylinder.re2`` / ``.rea`` and ``.vtk``.
"""

import logging

import numpy as np

from nekmeshpy import Curve, HexMesh, QuadMesh, export

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
# matching division count -- bottom/top to NX+1, left/right to NY+1.
bottom = Curve(bottom_pts).resample(np.linspace(0.0, 1.0, NX + 1))   # (-W,0) -> (W,0)
right = Curve([(W, 0.0, 0.0), (W, H, 0.0)]).resample(np.linspace(0.0, 1.0, NY + 1))
top = Curve([(W, H, 0.0), (-W, H, 0.0)]).resample(np.linspace(0.0, 1.0, NX + 1))
left = Curve([(-W, H, 0.0), (-W, 0.0, 0.0)]).resample(np.linspace(0.0, 1.0, NY + 1))
section = QuadMesh.structured(
    [bottom, right, top, left], interior_method="winslow",
    boundary_names={"bottom": "wall", "right": "outlet", "top": "top", "left": "inlet"})

# -- sweep along the span, naming the end caps front/back --------------------
zs = np.linspace(0.0, SPAN, N_SPAN + 1)
slices = [QuadMesh(section.points + np.array([0.0, 0.0, z]), section.quads,
                   boundaries=section.boundaries, boundary_names=section.boundary_names)
          for z in zs]
mesh = HexMesh.loft(slices, first_cap="front", last_cap="back")

# -- report + export ---------------------------------------------------------
print(mesh.report())
export.to_re2(mesh, OUT_NAME, groups=GROUPS)
export.to_vtk(mesh, OUT_NAME + ".vtk", groups=GROUPS)
print("groups:", ", ".join(mesh.boundary_group_names))
