"""Unequal-radius T-junction by the **cob** construction, with a boundary layer on the
wall -- ``tjunction_lib.build_cob``'s reference caller, the way
``quadrant_pipe_tjunction.py`` is ``build_tjunction``'s.

The construction lives in :func:`tjunction_lib.build_cob`; see there for what a cob is
and why the branch is bored straight through the main pipe rather than radiated from a
hub.  What this file adds is everything that is a *caller's* choice: how far to run the
legs, what to call the openings, and how thick a boundary layer to grow.

Local frame is the library's own: main pipe axis ``z``, branch axis ``+x`` with its
opening at ``x = H_BRANCH``.

The junction is meshed at the **core** radii -- the finished ones inset by the layer
thickness -- and :func:`tjunction_lib.skin_wall` grows the layer back out over the
``wall`` group, landing the wall on ``R_MAIN`` / ``R_BRANCH`` exactly.  An offset is a
uniform thickness along the surface normal rather than a scaling, which is why one inset
serves both radii.

    PYTHONPATH=. python examples/cob_tjunction.py

Produces ``cob_tjunction.re2`` and ``.vtu``.
"""

import logging
import os
import sys

import numpy as np

from nekmeshpy import hexmesh, writer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tjunction_lib import build_cob, skin_wall  # noqa: E402  (needs the path above)

logging.basicConfig(level=logging.INFO, format="%(message)s")

# -- parameters ---------------------------------------------------------------
R_MAIN = 0.5                  # main pipe radius (axis z), finished
N_THETA_MAIN = 32             # main pipe azimuthal cells -- sets the wall cell size
RADIAL_MAIN = 3               # main pipe O-grid radial layers
CENTER_SCALE_MAIN = 0.8       # main pipe O-grid hub placement
R_BRANCH = 0.1322             # branch bore radius (axis +x), finished; ratio 0.264
N_THETA_BRANCH = 16           # branch azimuthal cells; MULTIPLE OF 4
CENTER_SCALE_BRANCH = 0.8     # branch O-grid hub placement
#: Branch O-grid radial stations.  Two layers, so the bore is not thrown straight at the
#: cob's square in one step: the extra ring interpolates between the bore and the slot it
#: has to land in, which keeps the wedges that reach the square's corners from spanning
#: the whole transition on their own.
RADIAL_BRANCH = np.array([0.0, 1.0])
Z_DOMAIN = 3.0                # main pipe runs z in [-Z_DOMAIN, +Z_DOMAIN]
H_BRANCH = 1.5                # branch tip at x = H_BRANCH
N_Z_LEG = 14                  # hex layers per main-pipe leg
N_BRANCH = 10                 # hex layers along the branch
#: Boundary-layer stations, as distances out from the wall the core is meshed at.
#:
#: **The thickness is bounded by the geometry, not by taste.**  An offset can only move a
#: surface by less than its own local feature size, and the smallest feature on this wall
#: is not on the pipe -- it is the bore's imprint on the *far* wall, whose cells run about
#: ``R_BRANCH / 4``.  A skin of ``0.2 * R_MAIN`` folds four of them inside out; ``0.1``
#: still folds four at the branch root; ``0.05`` is the thickest of this family that comes
#: back clean.
BL = 0.05 * R_MAIN * np.array([0.0, 0.5, 1.0])
ORDER = 2
OUT_NAME = "cob_tjunction"

GROUPS = {"wall": "W  ", "inlet": "v  ", "outlet": "O  ", "branch": "O  "}

T_BL = float(BL[-1])

# -- build --------------------------------------------------------------------
# ``port_tags`` names the two leg caps on the junction itself, so the openings are named
# by *construction* rather than picked out of the finished mesh by coordinate -- and
# ``skin_wall`` then carries each name onto the shell's own lateral ring at that opening,
# which is the ring the layer leaves round a cap it deliberately does not skin.
core = build_cob(R_MAIN - T_BL, R_BRANCH - T_BL, H_BRANCH,
                 Z_NEAR=Z_DOMAIN, N_Z_LEG=N_Z_LEG,
                 N_THETA_MAIN=N_THETA_MAIN, RADIAL_MAIN=RADIAL_MAIN,
                 CENTER_SCALE_MAIN=CENTER_SCALE_MAIN,
                 N_THETA_BRANCH=N_THETA_BRANCH, RADIAL_BRANCH=RADIAL_BRANCH,
                 CENTER_SCALE_BRANCH=CENTER_SCALE_BRANCH, N_BRANCH=N_BRANCH,
                 order=ORDER, branch_tag="branch",
                 port_tags=("inlet", "outlet")).core

mesh = skin_wall(core, BL)

print("core %d hexes -> %d skin layers -> %d in all"
      % (core.n_hexes, BL.size - 1, mesh.n_hexes))
print(hexmesh.report(mesh))
writer.to_re2(mesh, OUT_NAME + ".re2", groups=GROUPS)
writer.to_vtu(mesh, OUT_NAME + ".vtu", groups=GROUPS)
