"""Mesh the flow past a circular cylinder (external flow, all-hex O-grid).

A flat, gmsh-style script: edit the constants below and re-run.  The domain is the
region between the circular body (radius ``R``) and a square far-field box
(half-width ``HALF_BOX``), swept along the span ``+z``.  It is the 2-D analogue of
the cubed-sphere shell in ``flow_past_sphere.py``: **four structured wedge patches,
one per box side**, each blending a quarter-arc of the cylinder out to one side of
the square (radial layers clustered toward the cylinder for a boundary layer,
``RADIAL_GRADING`` > 1).  The wedge seams meet at the four box corners.

Each patch is a structured ``(radial, theta, span)`` grid built with
:meth:`nekmeshpy.HexMesh.from_grid` and tagged **as it is built**: its radial-in
face is ``cylinder`` and its radial-out face is the box side it forms
(``inlet`` / ``outlet`` / ``top`` / ``bottom``), the span caps are ``front`` /
``back``, and its two azimuthal (seam) faces are left untagged so
:meth:`nekmeshpy.HexMesh.merge` welds the four wedges into one block along the
shared corner seams with no stale interior tags -- no post-hoc boundary detection.

Tagging distinct outer sides is done by this merge, not by ``QuadMesh.annulus``
(which tags the outer ring only as a whole): the split into named sides lives in the
example, not the primitive.

Run with::

    PYTHONPATH=. python examples/flow_past_cylinder.py

Produces ``flow_past_cylinder.re2`` / ``.rea`` and ``flow_past_cylinder.vtk``.
"""

import logging

import numpy as np

from nekmeshpy import HexMesh, export
from nekmeshpy.model.fields import geometric_spacing

logging.basicConfig(level=logging.INFO, format="%(message)s")

# -- parameters --------------------------------------------------------------
R = 0.5                      # cylinder radius
HALF_BOX = 6.0               # far-field box half-width (domain [-HB, HB]^2 in xy)
SPAN = 2.0                   # spanwise (z) extent
N_THETA = 64                 # azimuthal cells around the cylinder (multiple of 4)
N_RADIAL = 24                # radial cells from cylinder out to the box
RADIAL_GRADING = 1.12        # >1 clusters radial layers toward the cylinder
N_SPAN = 4                   # hex layers across the span
OUT_NAME = "flow_past_cylinder"

# boundary name -> Nek BC code, applied only at export
GROUPS = {"inlet": "v  ", "outlet": "O  ", "cylinder": "W  ",
          "top": "SYM", "bottom": "SYM", "front": "SYM", "back": "SYM"}

# the four box sides: name and the arc centre angle (degrees) facing that side; each
# wedge spans centre +/- 45 deg, so the seams land exactly on the box corners.
SIDES = [("outlet", 0.0), ("top", 90.0), ("inlet", 180.0), ("bottom", 270.0)]

if N_THETA % 4 != 0:
    raise ValueError("N_THETA must be a multiple of 4 (one wedge per box side)")


def square_projection(direction: np.ndarray) -> np.ndarray:
    """The point where the ray from the origin along ``direction`` (unit, in xy)
    meets the square box |x|,|y| = HALF_BOX -- an L-infinity projection, so the
    radial line stays straight from the cylinder out to the box side."""
    return HALF_BOX * direction / np.max(np.abs(direction))


# -- build one structured wedge per box side, then weld them ------------------
nq = N_THETA // 4                                    # azimuthal cells per wedge
t = geometric_spacing(N_RADIAL, RADIAL_GRADING)      # 0 (cylinder) .. 1 (box)
zs = np.linspace(0.0, SPAN, N_SPAN + 1)

patches = []
for name, centre in SIDES:
    theta = np.deg2rad(np.linspace(centre - 45.0, centre + 45.0, nq + 1))
    body = np.column_stack([R * np.cos(theta), R * np.sin(theta), np.zeros(nq + 1)])
    outer = np.array([square_projection(np.array([np.cos(a), np.sin(a), 0.0]))
                      for a in theta])               # box-side point per azimuth
    # P[radial, theta, span]: (radial-out, theta, span) is right-handed -> +hexes.
    P = np.zeros((N_RADIAL + 1, nq + 1, N_SPAN + 1, 3))
    for it in range(nq + 1):
        ring = (1.0 - t)[:, None] * body[it] + t[:, None] * outer[it]   # (Nr+1, 3)
        P[:, it, :, :] = ring[:, None, :] + np.array([0.0, 0.0, 1.0]) * zs[None, :, None]
    # x = radial (in=cylinder / out=this side); z = span caps; y (theta) seams stay
    # untagged so merge welds the wedges along the shared box-corner seams.
    patches.append(HexMesh.from_grid(P, face_tags={
        "x_min": "cylinder", "x_max": name, "z_min": "front", "z_max": "back"}))

mesh = HexMesh.merge(patches)                         # weld the four corner seams

# -- report + export ---------------------------------------------------------
print(mesh.report())
export.to_re2(mesh, OUT_NAME, groups=GROUPS)
export.to_vtk(mesh, OUT_NAME + ".vtk", groups=GROUPS)
print("groups:", ", ".join(mesh.boundary_group_names))
