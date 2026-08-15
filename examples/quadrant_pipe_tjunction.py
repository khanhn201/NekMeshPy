"""T-junction of a smaller branch pipe on a main pipe, meshed as a **single welded
component** with the junction topology of the reference vascular mesher.

Run with::

    PYTHONPATH=. python examples/quadrant_pipe_tjunction.py

Produces ``quadrant_pipe_tjunction.re2`` and ``quadrant_pipe_tjunction.vtu``.

**The construction below is built by** ``tjunction_lib.build_tjunction`` --
this file is its reference caller, kept in full because the construction is easiest
to read as one straight-line story rather than split across a general function's
keyword arguments.  What follows documents the algorithm ``build_tjunction`` actually
runs; this script itself is now just that call plus the two leg extrusions to the end
planes (``build_tjunction`` stops at the plain per-leg disc so a caller can continue
however it needs to -- extrude, loft, sweep -- which is exactly what
``chimera_full.py``'s T1/T2 junctions do instead of extruding straight).

The decomposition
-----------------

Every cross-section is a **full disc of four** ``quadrant_ogrid`` **blocks**, and the
junction is built so that *one quadrant of the main pipe is a quadrant of the branch*.
That is what welds the branch to the main pipe instead of merely parking it against
the wall.

Four solid regions meet at the junction centre ``O``, the point where the two axes
cross, and every interface between them is a quadrant face radiating from ``O``:

=====================  =======================================================
region                 its near-junction face
=====================  =======================================================
branch stub            the **footprint disc** -- four quadrants ``A/D/C/B``
                       whose wall is the exact intersection curve
``+z`` (outlet) leg    ``[A, sideRP, bypass, sideRM]``
``-z`` (inlet) leg     ``[C, sideLM, bypass reversed, sideLP]``
``+y`` crotch cap      ``[B, sideLP, sideRP]``
``-y`` crotch cap      ``[D, sideRM, sideLM]``
=====================  =======================================================

so the nine internal faces are

* ``bypass`` -- between the two legs: the part of the cross-section far from the
  branch, spanning ``phi`` from ``+PHI_W`` round through ``pi`` to ``-PHI_W``;
* ``A`` / ``C`` -- between the branch and the ``+z`` / ``-z`` leg.  **These are the
  welded quadrant-to-quadrant interfaces**: quadrant ``A`` of the branch's footprint
  disc *is* the branch-facing quadrant of the outlet leg's face;
* ``B`` / ``D`` -- between the branch and the two caps;
* the four ``side*`` quadrants -- between a leg and a cap.  Their wall arcs are the
  ruled ``(phi, z)`` curves that carry a footprint corner round the main cylinder to
  a bypass edge corner at ``z = 0``.

Each leg then morphs from its composite face to a plain 45-degree-rotated
four-quadrant disc and extrudes straight to its end plane; the branch morphs from the
footprint disc to a plain circular disc at ``x = H_BRANCH``.  Adjacent quadrants are
handed the *same* seam ``LineMesh`` object -- one of the six radii ``O -> P1..P4``,
``O -> W+``, ``O -> W-`` -- so they weld bit-exactly rather than to a tolerance.

**Both morphs are done in a surface parameter, not on the points.**  A leg's
transition interpolates its wall in the cylinder's own ``(phi, z)``; the branch
interpolates in the branch angle, where ``footprint`` and ``opening`` differ only in
``x``.  Lerping the *points* instead draws a chord between two points of a cylinder,
which dips inside it, so every intermediate station would sit proud of the wall.

The two crotch caps
-------------------

A crotch is the corner region bounded by three quadrant faces meeting at ``O`` plus a
curved triangle of main-pipe wall.  That is exactly an **octant of a 3-D O-grid**:
index the corner by ``(i, j, k)`` along its three seams and it is an ``n x n x n``
core hex block -- whose three inner faces are the three quadrant *cores* -- plus three
``n x n x Nradial`` slabs carrying the core's three outer faces out to a third of the
wall triangle each.  That gives ``n**2 + 2*n*Nradial`` quads on each shared face,
which is exactly what a ``quadrant_ogrid`` has, and the slabs reuse that factory's own
ring formula ``(1-tau)*perimeter + tau*wall``, so the two halves of every shared ring
band agree and the cap welds.

The wall triangle's tip -- the point its three patches converge to, and the direction
the octant core's far corner is placed along -- defaults to the plain centroid of the
three arcs' midpoints. ``CAP_TIP_BIAS`` reweights that toward the branch-facing arc
(always the crotch's ``ab`` argument), sliding the tip -- and the visible fan of mesh
lines around it -- closer to the branch.

Order
-----

Exact at any ``ORDER``, which took some care: a junction like this is where the
straight-subdivision traps all bite at once.

* Every wall curve is carried as its **surface parametrization** (:class:`Wall`), not
  as sampled points, and meshed with ``LineMesh.loft_fn`` -- so the footprint, the
  side transitions and the bypass are all evaluated on the true curve at every node.
  Sampling them into arrays and calling ``LineMesh.loft`` chords the wall from
  ``order > 1`` on.
* A leg's transition is a ``HexMesh.loft_fn`` over the blend parameter, not a
  ``loft`` of a section stack.  ``loft`` is straight **along the sweep**, so its wall
  nodes would be chords between stations at different ``phi`` -- measured here at
  order 3, that alone put them 7.2e-4 off the cylinder.
* The cap blocks are ``evaluated_block``s -- nested ``loft_fn`` over an explicit
  map on the unit cube -- rather than ``HexMesh.from_grid``, which takes corners and
  blends straight.  A ``from_grid`` cap is fine at order 1 and at ``order > 1`` both
  leaves the wall and disagrees with ``quadrant_ogrid``'s bowed ring bands along the
  faces they share, which ``merge`` rightly rejects.

Order 1 remains a strict no-op: the corner coordinates are bit-identical at ``ORDER``
1, 2, 3 and 4, and every wall node lies on the main or branch cylinder to 2.2e-16 at
all of them.
"""

import logging
import os
import sys

import numpy as np

from nekmeshpy import export, hexmesh

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tjunction_lib import build_tjunction  # noqa: E402  (needs the path above)

logging.basicConfig(level=logging.INFO, format="%(message)s")

# -- parameters ---------------------------------------------------------------
R_MAIN = 1.0                 # main pipe radius (axis = z)
R_BRANCH = 0.5               # branch pipe radius (axis = x), deliberately < R_MAIN
L_MAIN = 3.0                 # main pipe runs z = -L_MAIN .. +L_MAIN
Z_NEAR = 1.2                 # where a leg has finished morphing to a plain disc
H_BRANCH = 2.5               # branch opening plane, x = H_BRANCH

N_QUAD = 3                   # cells per quadrant half-arc; a quadrant spans 2*N_QUAD
RADIAL = np.array([0.0, 0.45, 0.8, 1.0])   # O-ring positions, core perimeter -> wall
CENTER_SCALE = 0.6           # hub corner at CENTER_SCALE * R along the arc midpoint
QUADRANT_SCALE = 0.5         # seam corner at QUADRANT_SCALE * R along the arc's own
                             # radius; kept below CENTER_SCALE so the seam corner does
                             # not bow out past the hub's own chord (min scaled
                             # Jacobian collapses toward 0 as the two converge)

N_TRANS = 5                  # layers from a leg's composite face to its plain disc
N_LEG = 6                    # layers from the plain disc to the end plane
N_BRANCH = 8                 # layers in the branch, footprint -> opening

ORDER = 3                    # exact at any order; see the module docstring

OUT_NAME = "quadrant_pipe_tjunction"
GROUPS = {"wall": "W  ", "inlet": "v  ", "outlet": "O  ", "branch": "O  "}

# -- the junction itself --------------------------------------------------------
# PHI_W / CAP_TIP_BIAS / ORIGIN (the bypass edge, the crotch wall-triangle tip bias,
# and the junction hub -- see the module docstring) default inside build_tjunction to
# tjunction_lib.auto_params(R_MAIN, R_BRANCH), the radius-ratio-dependent choice; pass
# any of them explicitly here to override just that one.
core, disc_minus, disc_plus, disc_branch = build_tjunction(
    R_MAIN, R_BRANCH, H_BRANCH, Z_NEAR=Z_NEAR, N_QUAD=N_QUAD, RADIAL=RADIAL,
    CENTER_SCALE=CENTER_SCALE, QUADRANT_SCALE=QUADRANT_SCALE, order=ORDER,
    N_TRANS=N_TRANS, N_BRANCH=N_BRANCH, branch_tag="branch")

# -- the two legs: build_tjunction stops at the plain per-leg disc so a caller can
# continue however it needs to (extrude, loft, sweep); here that is a straight
# extrude to each end plane, exactly as the reference decomposes it.
leg_plus = hexmesh.extrude(disc_plus, L_MAIN - Z_NEAR, N_LEG,
                           axis=(0.0, 0.0, 1.0), last_tag="outlet")
leg_minus = hexmesh.extrude(disc_minus, L_MAIN - Z_NEAR, N_LEG,
                            axis=(0.0, 0.0, -1.0), last_tag="inlet")

mesh = hexmesh.merge([core, leg_plus, leg_minus])

print(hexmesh.report(mesh))
print(hexmesh.topology_report(mesh))

export.to_re2(mesh, OUT_NAME + ".re2", groups=GROUPS)
export.to_vtu(mesh, OUT_NAME + ".vtu", groups=GROUPS)
