"""Mesh the flow past a sphere (external flow, cubed-sphere shell).

A flat, gmsh-style script: edit the constants below and re-run.  The domain is the
region between a spherical body (radius ``R``) and a cubic far-field box
(half-side ``S``) -- "sphere surface extruded to a cube at the boundary".  It is
built as a **cubed-sphere shell**: six structured hex patches, one per cube face.

For a cube face with outward normal ``n`` and right-handed tangents ``u, v``
(``u x v = n``), gnomonic coordinates ``a, b in [-1,1]`` give a cube point
``S*(n + a*u + b*v)`` and the sphere point ``R * normalize(cube_pt)``; the radial
index blends sphere -> cube (clustered toward the sphere for a boundary layer), and
the ordering ``(a, b, radial-out)`` is right-handed so
:meth:`nekmeshpy.HexMesh.from_grid` yields positive hexes.  Each patch is tagged
**as it is built**: its inner (radial-in) face is ``sphere`` and its outer face is
the cube side it belongs to (``inlet`` / ``outlet`` / ``top`` / ``bottom`` /
``front`` / ``back``), while its four lateral faces are left untagged so
:meth:`nekmeshpy.HexMesh.merge` welds the six patches into one shell along the
shared gnomonic edges with no stale interior tags.

Run with::

    PYTHONPATH=. python examples/flow_past_sphere.py

Produces ``flow_past_sphere.re2`` / ``.rea`` and ``flow_past_sphere.vtk``.
"""

import logging

import numpy as np

from nekmeshpy import HexMesh, export
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

# -- build one cubed-sphere patch per cube face ------------------------------
ab = np.linspace(-1.0, 1.0, N_FACE + 1)
t = geometric_spacing(N_RADIAL, RADIAL_GRADING)             # 0 (sphere) .. 1 (cube)
A, B = np.meshgrid(ab, ab, indexing="ij")                  # (Nf+1, Nf+1)

patches = []
for nrm, u, v in FACES:
    n = np.asarray(nrm, float)
    u = np.asarray(u, float)
    v = np.asarray(v, float)
    cube = S * (n + A[..., None] * u + B[..., None] * v)   # (Nf+1, Nf+1, 3)
    sphere = R * cube / np.linalg.norm(cube, axis=-1, keepdims=True)
    # P[i,j,k] blends sphere(k=0) -> cube(k=last); (a, b, radial-out) is right-handed
    P = ((1.0 - t)[None, None, :, None] * sphere[:, :, None, :]
         + t[None, None, :, None] * cube[:, :, None, :])   # (Nf+1, Nf+1, Nr+1, 3)
    # k-axis (z_min/z_max) is radial: inner = sphere, outer = this cube side.  The
    # i/j lateral faces are shared with the neighbouring patches -> left untagged.
    patches.append(HexMesh.from_grid(
        P, face_tags={"z_min": "sphere", "z_max": WORLD_SIDE[nrm]}))

mesh = HexMesh.merge(patches)                              # weld shared gnomonic edges

# -- report + export ---------------------------------------------------------
print(mesh.report())
export.to_re2(mesh, OUT_NAME, groups=GROUPS)
export.to_vtk(mesh, OUT_NAME + ".vtk", groups=GROUPS)
print("groups:", ", ".join(mesh.boundary_group_names))
