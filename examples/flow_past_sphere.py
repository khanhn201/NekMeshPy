"""Mesh the flow past a sphere (external flow, cubed-sphere shell).

A flat, gmsh-style script: edit the constants below and re-run.  The domain is the
region between a spherical body (radius ``R``) and a cubic far-field box
(half-side ``S``) -- "sphere surface extruded to a cube at the boundary".  It is
built with :meth:`nekmeshpy.HexMesh.annulus`, the 3-D sibling of
:meth:`nekmeshpy.QuadMesh.annulus`: fill the shell between two **closed quad
surfaces** (an inner sphere surface and an outer cube surface), exactly as the 2-D
flow-past-cylinder fills the ring between an inner and an outer loop.

The outer cube surface is six flat gnomonic patches (one per cube face) welded into
one closed surface with :meth:`nekmeshpy.QuadMesh.merge`; each patch carries the
far-field side it forms (``inlet`` / ``outlet`` / ``top`` / ``bottom`` / ``front`` /
``back``) as its per-quad ``element_tags``.  The inner sphere surface reuses the cube
surface's connectivity with points ``R * normalize(cube point)`` -- so the two
surfaces pair by index automatically -- and tags every quad ``sphere``.  ``annulus``
blends the radial shells (clustered toward the sphere for a boundary layer) and turns
each surface's per-quad ``element_tags`` into the inner / outer wall faces; there are
no free boundary edges to tag, so the wall groups come from the element tags, not the
boundary tags.

Run with::

    PYTHONPATH=. python examples/flow_past_sphere.py

Produces ``flow_past_sphere.re2`` / ``.rea`` and ``flow_past_sphere.vtk``.
"""

import logging

import numpy as np

from nekmeshpy import HexMesh, QuadMesh, export
from nekmeshpy.model.fields import geometric_spacing

logging.basicConfig(level=logging.INFO, format="%(message)s")

# -- parameters --------------------------------------------------------------
R = 0.5                      # sphere radius
S = 4.0                      # far-field cube half-side (domain [-S, S]^3)
N_FACE = 12                  # cells per direction on each cube face
N_RADIAL = 12                # radial cells from sphere out to the cube
RADIAL_GRADING = 1.15        # >1 clusters radial layers toward the sphere
OUT_NAME = "flow_past_sphere"

# boundary name -> Nek BC code, applied only at export
GROUPS = {"inlet": "v  ", "outlet": "O  ", "sphere": "W  ",
          "top": "SYM", "bottom": "SYM", "front": "SYM", "back": "SYM"}

# the six cube faces: outward normal n with right-handed tangents (u x v = n)
FACES = [
    ((1, 0, 0), (0, 1, 0), (0, 0, 1)),
    ((-1, 0, 0), (0, 0, 1), (0, 1, 0)),
    ((0, 1, 0), (0, 0, 1), (1, 0, 0)),
    ((0, -1, 0), (1, 0, 0), (0, 0, 1)),
    ((0, 0, 1), (1, 0, 0), (0, 1, 0)),
    ((0, 0, -1), (0, 1, 0), (1, 0, 0)),
]

# each cube face's outward normal -> the flow-box side name it forms
WORLD_SIDE = {(1, 0, 0): "outlet", (-1, 0, 0): "inlet",
              (0, 1, 0): "top", (0, -1, 0): "bottom",
              (0, 0, 1): "back", (0, 0, -1): "front"}

# -- two closed quad surfaces: outer cube (tagged per face) and inner sphere -
ab = np.linspace(-1.0, 1.0, N_FACE + 1)
A, B = np.meshgrid(ab, ab, indexing="ij")                  # (Nf+1, Nf+1)

patches = []
for nrm, u, v in FACES:
    n = np.asarray(nrm, float)
    u = np.asarray(u, float)
    v = np.asarray(v, float)
    cube_face = S * (n + A[..., None] * u + B[..., None] * v)   # (Nf+1, Nf+1, 3)
    # tag the whole patch with the far-field side it forms; its lateral edges are
    # shared with neighbours -> left untagged so merge welds them cleanly.
    patches.append(QuadMesh.from_grid(cube_face, element_tag=WORLD_SIDE[nrm]))

cube = QuadMesh.merge(patches)                             # closed cube surface
# inner sphere surface: same connectivity, points R * normalize(cube point), so the
# two surfaces pair by index; every quad is the sphere wall.
sphere = QuadMesh(
    R * cube.points / np.linalg.norm(cube.points, axis=1, keepdims=True),
    cube.quads, element_tags=np.full(cube.n_quads, "sphere"))

# fill the shell sphere -> cube; radial clustered toward the sphere (0 = sphere wall,
# 1 = cube).  Inner cap tagged `sphere`, outer cap tagged per cube-face element tag.
mesh = HexMesh.annulus(sphere, cube,
                       radial=geometric_spacing(N_RADIAL, RADIAL_GRADING))

# -- report + export ---------------------------------------------------------
print(mesh.report())
export.to_re2(mesh, OUT_NAME, groups=GROUPS)
export.to_vtk(mesh, OUT_NAME + ".vtk", groups=GROUPS)
print("groups:", ", ".join(mesh.boundary_group_tags))
