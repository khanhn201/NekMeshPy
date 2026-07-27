"""Mesh the classic backward-facing step (external/internal expansion flow).

A flat, gmsh-style script: edit the constants below and re-run.  The 2-D section is
three structured rectangles -- the inlet channel ``[-L_UP,0] x [0,H]``, the
downstream upper channel ``[0,L_DOWN] x [0,H]`` and the recirculation region below
the step ``[0,L_DOWN] x [-STEP,0]`` -- welded with :meth:`nekmeshpy.QuadMesh.merge`
(matching divisions on the shared edges) and swept along the span by
:meth:`nekmeshpy.HexMesh.extrude`.

Boundaries are named **as the mesh is built**: each rectangle names its own outer
sides (``inlet`` on the upstream inlet, ``outlet`` on the downstream ends, ``wall``
on the channel walls / step floor / vertical step face) while the shared internal
edges are left unnamed so :meth:`nekmeshpy.QuadMesh.merge` welds them away, and the
span sweep names the two end caps ``front`` / ``back``.

Run with::

    PYTHONPATH=. python examples/backward_facing_step.py

Produces ``backward_facing_step.re2`` / ``.rea`` and ``backward_facing_step.vtk``.
"""

import logging

import numpy as np

from nekmeshpy import Curve, HexMesh, QuadMesh, export

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
# each rectangle names only its true outer sides; shared edges are left unnamed so
# the merge welds them into interior faces (edges = [bottom, right, top, left]).
def rect(x0: float, x1: float, y0: float, y1: float, nx: int, ny: int,
         boundary_names: dict[str, str]) -> QuadMesh:
    """A structured quad grid over the axis-aligned rectangle ``[x0,x1]x[y0,y1]``,
    with the named outer sides tagged at build time."""
    c0, c1, c2, c3 = ((x0, y0, 0.0), (x1, y0, 0.0),
                      (x1, y1, 0.0), (x0, y1, 0.0))
    # structured uses the edges' own nodes (no resampling): sample each straight
    # edge to the matching division count (uniform here).
    edges = [Curve([c0, c1]).resample(np.linspace(0.0, 1.0, nx + 1)),
             Curve([c1, c2]).resample(np.linspace(0.0, 1.0, ny + 1)),
             Curve([c2, c3]).resample(np.linspace(0.0, 1.0, nx + 1)),
             Curve([c3, c0]).resample(np.linspace(0.0, 1.0, ny + 1))]
    return QuadMesh.structured(edges, boundary_names=boundary_names)


section = QuadMesh.merge([
    rect(-L_UP, 0.0, 0.0, H, NX_UP, NY_CH,             # inlet channel
         {"left": "inlet", "bottom": "wall", "top": "wall"}),
    rect(0.0, L_DOWN, 0.0, H, NX_DOWN, NY_CH,          # downstream upper channel
         {"right": "outlet", "top": "wall"}),
    rect(0.0, L_DOWN, -STEP, 0.0, NX_DOWN, NY_STEP,    # recirculation region
         {"right": "outlet", "bottom": "wall", "left": "wall"}),  # left = step face
])

# -- sweep along the span, naming the end caps front/back --------------------
# extrude rigidly translates the xy section along +z; the section's edge names
# ride onto the swept side faces.
mesh = HexMesh.extrude(section, axis=(0.0, 0.0, 1.0), length=SPAN,
                       layers=np.linspace(0.0, 1.0, N_SPAN + 1),
                       first_cap="front", last_cap="back")

# -- report + export ---------------------------------------------------------
print(mesh.report())
export.to_re2(mesh, OUT_NAME, groups=GROUPS)
export.to_vtk(mesh, OUT_NAME + ".vtk", groups=GROUPS)
print("groups:", ", ".join(mesh.boundary_group_names))
