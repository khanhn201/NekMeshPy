"""Flow past a hemisphere on the ground (external flow).

Like ``flow_past_sphere.py`` -- a cubed-sphere shell between body and box -- but
only the **upper half**: a hemisphere of radius ``R`` on the floor ``z=0``, domain
filling the half-box ``[-S,S] x [-S,S] x [0,S]``.

Five structured hex patches: four vertical side patches (cube faces ``+/-x`` /
``+/-y`` with tangent ``v=+z`` restricted to ``b in [0,1]``, so their bottom edge
sits on ``z=0``) plus the top patch ``+z``. Dropping the ``-z`` patch opens the
hemisphere at the bottom; the side patches' ``z=0`` faces form the ground annulus.

Each patch is tagged at build: inner face ``hemisphere``, outer face the cube side
it forms, each side patch's ``z=0`` face ``ground``, shared lateral faces untagged
so :meth:`HexMesh.merge` welds them. The ceiling sits at ``z=S``.

    PYTHONPATH=. python examples/flow_past_hemisphere.py

Produces ``flow_past_hemisphere.re2`` / ``.rea`` and ``.vtu``.
"""

import logging

import numpy as np

from nekmeshpy import HexMesh, export
from nekmeshpy.model.fields import geometric_spacing

logging.basicConfig(level=logging.INFO, format="%(message)s")

# -- parameters --------------------------------------------------------------
R = 0.5                      # hemisphere radius (rests on the floor z=0)
S = 4.0                      # far-field half-box: [-S,S] x [-S,S] x [0,S]
N_FACE = 6                  # horizontal cells per direction on each cube face
N_HALF = 6                   # vertical cells over z in [0,S] on the side patches
N_RADIAL = 12                # radial cells from hemisphere out to the box
RADIAL_GRADING = 1.15        # >1 clusters radial layers toward the hemisphere
ORDER = 4
OUT_NAME = "flow_past_hemisphere"

# boundary name -> Nek BC code, applied only at export
GROUPS = {"inlet": "v  ", "outlet": "O  ", "hemisphere": "W  ", "ground": "W  ",
          "top": "SYM", "front": "SYM", "back": "SYM"}

# four vertical side faces: outward normal n, horizontal tangent u, vertical v=+z
# (u x v = n), so restricting the v-coordinate b to [0,1] keeps z >= 0
SIDES = [
    ((1, 0, 0), (0, 1, 0)),      # +x
    ((-1, 0, 0), (0, -1, 0)),    # -x
    ((0, 1, 0), (-1, 0, 0)),     # +y
    ((0, -1, 0), (1, 0, 0)),     # -y
]
VZ = np.array([0.0, 0.0, 1.0])

# each side patch's outward cube normal -> the flow-box side name it forms
WORLD_SIDE = {(1, 0, 0): "outlet", (-1, 0, 0): "inlet",
              (0, 1, 0): "back", (0, -1, 0): "front"}

# -- build the shell: four side patches (upper half) + the top patch ----------
t = geometric_spacing(N_RADIAL, RADIAL_GRADING)            # 0 (sphere) .. 1 (cube)


def patch(cube: np.ndarray, face_tags: dict[str, str]) -> HexMesh:
    """Radial blend of a gnomonic cube-face grid ``cube`` (Ni+1, Nj+1, 3) from the
    sphere (R*normalize) out to the cube, with build-time ``face_tags``. The k-axis
    is radial: z_min = sphere surface, z_max = the cube (flow-box) side."""
    sphere = R * cube / np.linalg.norm(cube, axis=-1, keepdims=True)
    P = ((1.0 - t)[None, None, :, None] * sphere[:, :, None, :]
         + t[None, None, :, None] * cube[:, :, None, :])   # (Ni+1, Nj+1, Nr+1, 3)
    return HexMesh.from_grid(P, face_tags=face_tags)


patches = []
a_side = np.linspace(-1.0, 1.0, N_FACE + 1)
b_side = np.linspace(0.0, 1.0, N_HALF + 1)                  # upper half only
A_s, B_s = np.meshgrid(a_side, b_side, indexing="ij")
for nrm, u in SIDES:
    n = np.asarray(nrm, float)
    u = np.asarray(u, float)
    cube = S * (n + A_s[..., None] * u + B_s[..., None] * VZ)
    # b=0 edge (y_min) -> the z=0 ground annulus; z_min = hemisphere, z_max = this
    # cube side. a-axis and b=1 (shared with top) stay untagged.
    patches.append(patch(cube, {"z_min": "hemisphere", "z_max": WORLD_SIDE[nrm],
                                "y_min": "ground"}))

# top patch (+z): full cube face at z=S
ab = np.linspace(-1.0, 1.0, N_FACE + 1)
A_t, B_t = np.meshgrid(ab, ab, indexing="ij")
cube_top = S * (np.array([0.0, 0.0, 1.0])
                + A_t[..., None] * np.array([1.0, 0.0, 0.0])
                + B_t[..., None] * np.array([0.0, 1.0, 0.0]))
patches.append(patch(cube_top, {"z_min": "hemisphere", "z_max": "top"}))

mesh = HexMesh.merge(patches)                              # weld shared gnomonic edges

# -- report + export ---------------------------------------------------------
print(mesh.report())
export.to_re2(mesh, OUT_NAME, groups=GROUPS)
export.to_vtu(mesh, OUT_NAME + ".vtu", groups=GROUPS)
print("groups:", ", ".join(mesh.boundary_group_tags))
