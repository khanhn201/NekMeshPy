"""Classic backward-facing step (expansion flow).

The 2-D section is three structured rectangles -- inlet channel
``[-L_UP,0] x [0,H]``, downstream channel ``[0,L_DOWN] x [0,H]``, recirculation
region ``[0,L_DOWN] x [-STEP,0]`` -- welded with :meth:`QuadMesh.merge` and swept
along the span by :meth:`HexMesh.extrude`.

Each rectangle tags its own outer edge lines (``inlet`` / ``outlet`` / ``wall``);
shared internal edges stay untagged so ``merge`` welds them away. The span sweep
names the end caps ``front`` / ``back``.

    PYTHONPATH=. python examples/backward_facing_step.py

Produces ``backward_facing_step.re2`` / ``.rea`` and ``backward_facing_step.vtk``.
"""

import logging

import numpy as np

from nekmeshpy import HexMesh, QuadMesh, export

logging.basicConfig(level=logging.INFO, format="%(message)s")

# -- parameters --------------------------------------------------------------
L_UP = 4.0                   # inlet channel length upstream of the step
L_DOWN = 16.0                # channel length downstream of the step
H = 1.0                      # inlet channel height (above y=0)
STEP = 1.0                   # step height (expansion depth, below y=0)
SPAN = 2.0                   # spanwise (z) extent
NX_UP = 16                   # cells along the upstream channel
NX_DOWN = 48                 # cells along the downstream channel
NY_CH = 12                   # cells across the channel height (0..H)
NY_STEP = 12                 # cells across the step depth (-STEP..0)
N_SPAN = 8                   # hex layers across the span
OUT_NAME = "backward_facing_step"

# boundary name -> Nek BC code, applied only at export (span faces = symmetry)
GROUPS = {"inlet": "v  ", "outlet": "O  ", "wall": "W  ",
          "front": "SYM", "back": "SYM"}


# -- build the L-shaped section from three structured rectangles --------------
# QuadMesh.rectangle tags only the named outer sides; shared edges stay untagged
# so merge welds them into interior faces.
def rect(x0: float, x1: float, y0: float, y1: float, nx: int, ny: int,
         side_tags: dict[str, str]) -> QuadMesh:
    """Structured quad grid over ``[x0,x1]x[y0,y1]`` with the named outer sides
    (bottom/right/top/left) tagged; an absent side stays untagged."""
    return QuadMesh.rectangle(
        [(x0, y0, 0.0), (x1, y0, 0.0), (x1, y1, 0.0), (x0, y1, 0.0)],
        nx, ny, side_tags=side_tags)


section = QuadMesh.merge([
    rect(-L_UP, 0.0, 0.0, H, NX_UP, NY_CH,             # inlet channel
         {"left": "inlet", "bottom": "wall", "top": "wall"}),
    rect(0.0, L_DOWN, 0.0, H, NX_DOWN, NY_CH,          # downstream upper channel
         {"right": "outlet", "top": "wall"}),
    rect(0.0, L_DOWN, -STEP, 0.0, NX_DOWN, NY_STEP,    # recirculation region
         {"right": "outlet", "bottom": "wall", "left": "wall"}),  # left = step face
])

# -- sweep along the span, naming the end caps front/back --------------------
# extrude translates the xy section along +z; edge names ride onto the side faces
mesh = HexMesh.extrude(section, axis=(0.0, 0.0, 1.0), length=SPAN,
                       layers=np.linspace(0.0, 1.0, N_SPAN + 1),
                       first_tag="front", last_tag="back")

# -- report + export ---------------------------------------------------------
print(mesh.report())
export.to_re2(mesh, OUT_NAME, groups=GROUPS)
export.to_vtk(mesh, OUT_NAME + ".vtk", groups=GROUPS)
print("groups:", ", ".join(mesh.boundary_group_tags))
