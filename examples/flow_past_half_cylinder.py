"""Flow past a half-cylinder on the ground (external flow).

A half-cylinder of radius ``R`` rests on the floor; flow fills the channel up to a
flat ceiling at ``H`` -- the classic "bump in a channel".

The 2-D section is a **single** transfinite block (:meth:`QuadMesh.structured`)
whose bottom edge is one composite curve: flat ground ``[-W,-R]``, semicircular
bump ``-R..R``, flat ground ``[R,W]``. Keeping the semicircle mid-edge (vs a
three-block split) avoids the degenerate corner where bump meets ground at
``x = +/-R``.

The three pieces are built **analytically** -- two :meth:`LineMesh.line` runs and
one :meth:`LineMesh.arc` -- and welded end to end with :meth:`LineMesh.merge`, so
no resampling is involved: the nodes land exactly where they are asked for, the
corners land exactly on ``x = +/-R``, and at ``ORDER > 1`` the bump's GLL nodes sit
on the **exact** circle (``structured`` samples its transfinite map at the GLL-refined
lattice against each edge's own nodes, so the bump is exact and its curvature bows the
interior above it too). Sampling a polyline and calling ``LineMesh.loft`` instead
would give high-order storage with straight-subdivided -- i.e. linear -- geometry.

Each edge line tags itself: bottom (ground + bump) ``wall``, ends ``inlet`` /
``outlet``, ceiling ``top``. The span sweep (:meth:`HexMesh.extrude`) names the caps
``front`` / ``back``.

    PYTHONPATH=. python examples/flow_past_half_cylinder.py

Produces ``flow_past_half_cylinder.re2`` / ``.rea`` and ``.vtu``.
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
N_GROUND = 45                # cells on each flat ground run ([-W,-R] and [R,W])
N_BUMP = 24                  # cells over the semicircular bump (-R..R); see the
                             # note below -- the bump wants to be *finer* than the
                             # ground, not coarser
NX = 2 * N_GROUND + N_BUMP   # cells along the channel (bottom edge) = 114
NY = 40                      # cells from floor/bump to ceiling
N_SPAN = 4                   # hex layers across the span
ORDER = 2                    # polynomial order; 1 = linear.  The bottom edge is
                             # analytic, so its GLL nodes lie on the exact ground
                             # lines / bump circle, and the transfinite interior is
                             # evaluated at the refined lattice, so the bump's
                             # curvature carries up into the channel
OUT_NAME = "flow_past_half_cylinder"

# boundary name -> Nek BC code, applied only at export
GROUPS = {"inlet": "v  ", "outlet": "O  ", "wall": "W  ",
          "top": "SYM", "front": "SYM", "back": "SYM"}

# -- composite bottom edge: ground -> semicircular bump -> ground -------------
# Each piece is placed exactly where it belongs -- no resampling.  The flat runs are
# (W-R)/N_GROUND = 0.1 long; the bump is deliberately finer (pi*R/N_BUMP = 0.052).
#
# Why the bump is refined: at x = +/-R the circle leaves the ground *vertically*, so
# the first bump cell's bottom edge stands at 90 - (pi/N_BUMP)/2 degrees while the
# transfinite grid lines above it are near-vertical too -- a sliver.  The skew is
# governed by how far that cell's top node (uniform in x along the flat ceiling)
# leans away from its bottom node, so refining the bump *shortens* the lean and
# improves the cell.  Sweeping N_BUMP at fixed NX = 120 gives min scaled Jacobian
# 0.107 at N_BUMP = 10, 0.147 at 18, 0.222 at 24 and 0.272 at 30: monotone above
# ~10, so N_BUMP = 24 is the coarsest bump that still clears the 0.2 floor the tests
# assert.  (That floor is the corner metric; scaled_jacobian(high_order=True) on the
# same mesh reads 0.181 -- the curved wall genuinely costs quality at the GLL nodes.)
left_ground = LineMesh.line((-W, 0.0, 0.0), (-R, 0.0, 0.0),
                            np.linspace(0.0, 1.0, N_GROUND + 1),
                            element_tag="wall", order=ORDER)
# theta pi -> 0 walks the bump left to right over the top, so the three runs chain
# in order; at ORDER > 1 the interior GLL nodes sit on the exact circle
bump = LineMesh.arc(R, N_BUMP, center=(0.0, 0.0, 0.0), normal=(0.0, 0.0, 1.0),
                    start_theta=np.pi, end_theta=0.0,
                    element_tags=["wall"] * N_BUMP, order=ORDER)
right_ground = LineMesh.line((R, 0.0, 0.0), (W, 0.0, 0.0),
                             np.linspace(0.0, 1.0, N_GROUND + 1),
                             element_tag="wall", order=ORDER)
# weld the shared ends at x = -R and x = +R into one (-W,0) -> (W,0) chain; the whole
# edge is tagged "wall" at the line level, so structured names the bottom side from it
bottom = LineMesh.merge([left_ground, bump, right_ground])
right = LineMesh.line((W, 0.0, 0.0), (W, H, 0.0),
                      np.linspace(0.0, 1.0, NY + 1), element_tag="outlet",
                      order=ORDER)
top = LineMesh.line((W, H, 0.0), (-W, H, 0.0),
                    np.linspace(0.0, 1.0, NX + 1), element_tag="top", order=ORDER)
left = LineMesh.line((-W, H, 0.0), (-W, 0.0, 0.0),
                     np.linspace(0.0, 1.0, NY + 1), element_tag="inlet",
                     order=ORDER)
section = QuadMesh.structured([bottom, right, top, left])

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
