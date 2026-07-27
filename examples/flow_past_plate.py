"""Mesh the flow past a thin plate (external flow, all-hex O-grid).

A flat, gmsh-style script: edit the constants below and re-run.  Same construction
as ``flow_past_cylinder.py`` -- **four structured wedge patches, one per box side**,
each blending a quarter-arc of the body out to one side of the square far field and
welded with :meth:`nekmeshpy.HexMesh.merge` (the 2-D analogue of the cubed-sphere
shell in ``flow_past_sphere.py``).  The boundaries are named as the mesh is built
(radial-in ``plate``; radial-out the box side ``inlet`` / ``outlet`` / ``top`` /
``bottom``; span caps ``front`` / ``back``).  The only change from the cylinder is
the body: a **thin ellipse** (half-length ``A``, half-thickness ``B`` with
``B << A``) standing in for a flat plate.

*Modeling choice:* a true zero-thickness plate would need a C-grid the toolkit does
not provide; a thin ellipse keeps a smooth, everywhere-non-degenerate all-hex O-grid
around the body while approximating the plate as ``B`` -> 0.  The body points are
sampled by **direction angle** (the ray from the origin at angle ``theta`` meets the
ellipse), so each wedge's arc and the box side it maps to stay radially aligned right
through the sharp ends.

Run with::

    PYTHONPATH=. python examples/flow_past_plate.py

Produces ``flow_past_plate.re2`` / ``.rea`` and ``flow_past_plate.vtk``.
"""

import logging

import numpy as np

from nekmeshpy import HexMesh, export
from nekmeshpy.model.fields import geometric_spacing

logging.basicConfig(level=logging.INFO, format="%(message)s")

# -- parameters --------------------------------------------------------------
A = 1.0                      # plate half-length (streamwise)
B = 0.08                     # plate half-thickness (B << A)
HALF_BOX = 6.0               # far-field box half-width (domain [-HB, HB]^2 in xy)
SPAN = 2.0                   # spanwise (z) extent
N_THETA = 80                 # azimuthal cells around the plate (multiple of 4)
N_RADIAL = 24                # radial cells from plate out to the box
RADIAL_GRADING = 1.12        # >1 clusters radial layers toward the plate
N_SPAN = 4                   # hex layers across the span
OUT_NAME = "flow_past_plate"

# boundary name -> Nek BC code, applied only at export
GROUPS = {"inlet": "v  ", "outlet": "O  ", "plate": "W  ",
          "top": "SYM", "bottom": "SYM", "front": "SYM", "back": "SYM"}

# the four box sides: name and the arc centre angle (degrees) facing that side; each
# wedge spans centre +/- 45 deg, so the seams land exactly on the box corners.
SIDES = [("outlet", 0.0), ("top", 90.0), ("inlet", 180.0), ("bottom", 270.0)]

if N_THETA % 4 != 0:
    raise ValueError("N_THETA must be a multiple of 4 (one wedge per box side)")


def ellipse_point(theta: float) -> np.ndarray:
    """The point where the ray from the origin at angle ``theta`` meets the ellipse
    (half-length ``A`` along x, half-thickness ``B`` along y)."""
    r = 1.0 / np.sqrt((np.cos(theta) / A) ** 2 + (np.sin(theta) / B) ** 2)
    return np.array([r * np.cos(theta), r * np.sin(theta), 0.0])


def square_projection(direction: np.ndarray) -> np.ndarray:
    """The point where the ray from the origin along ``direction`` (unit, in xy)
    meets the square box |x|,|y| = HALF_BOX -- an L-infinity projection, so the
    radial line stays straight from the plate out to the box side."""
    return HALF_BOX * direction / np.max(np.abs(direction))


# -- build one structured wedge per box side, then weld them ------------------
nq = N_THETA // 4                                    # azimuthal cells per wedge
t = geometric_spacing(N_RADIAL, RADIAL_GRADING)      # 0 (plate) .. 1 (box)
zs = np.linspace(0.0, SPAN, N_SPAN + 1)

patches = []
for name, centre in SIDES:
    theta = np.deg2rad(np.linspace(centre - 45.0, centre + 45.0, nq + 1))
    body = np.array([ellipse_point(a) for a in theta])
    outer = np.array([square_projection(np.array([np.cos(a), np.sin(a), 0.0]))
                      for a in theta])               # box-side point per azimuth
    # P[radial, theta, span]: (radial-out, theta, span) is right-handed -> +hexes.
    P = np.zeros((N_RADIAL + 1, nq + 1, N_SPAN + 1, 3))
    for it in range(nq + 1):
        ring = (1.0 - t)[:, None] * body[it] + t[:, None] * outer[it]   # (Nr+1, 3)
        P[:, it, :, :] = ring[:, None, :] + np.array([0.0, 0.0, 1.0]) * zs[None, :, None]
    # x = radial (in=plate / out=this side); z = span caps; y (theta) seams stay
    # untagged so merge welds the wedges along the shared box-corner seams.
    patches.append(HexMesh.from_grid(P, face_tags={
        "x_min": "plate", "x_max": name, "z_min": "front", "z_max": "back"}))

mesh = HexMesh.merge(patches)                         # weld the four corner seams

# -- report + export ---------------------------------------------------------
print(mesh.report())
export.to_re2(mesh, OUT_NAME, groups=GROUPS)
export.to_vtk(mesh, OUT_NAME + ".vtk", groups=GROUPS)
print("groups:", ", ".join(mesh.boundary_group_names))
