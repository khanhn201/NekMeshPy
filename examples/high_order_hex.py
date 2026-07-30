"""High-order (order-N) volume: a spectral-element spherical shell whose curved
inner wall is the true sphere.

The 3-D motivating high-order case. A cubed-:meth:`QuadMesh.sphere` and a
:meth:`QuadMesh.box`, both at ``order=N`` (so they pair by index), bound a shell
filled with :meth:`HexMesh.annulus`. Because ``annulus`` blends the two surfaces'
curved blocks radially, **every** ``(N+1)**3`` node of each hex is placed in 3-D:
the inner-wall nodes lie on the exact sphere, not on the faceted cube-projection.
The corner connectivity stays linear -- ``re2`` export is unaffected -- while ``vtu``
export emits ``VTK_LAGRANGE_HEXAHEDRON`` cells that render the curved shell.

    PYTHONPATH=. python examples/high_order_hex.py

Produces ``high_order_hex.re2`` / ``.rea`` (linear corners) and ``high_order_hex.vtu``
(high-order Lagrange hexahedra).
"""

import logging

import numpy as np

from nekmeshpy import HexMesh, QuadMesh, export

logging.basicConfig(level=logging.INFO, format="%(message)s")
_log = logging.getLogger("nekmeshpy")

# -- parameters --------------------------------------------------------------
R = 1.0                      # inner sphere radius
S = 3.0                      # outer cube half-side
N_FACE = 2                   # cells per cube-face axis (6 * N_FACE**2 surface quads)
N_RADIAL = 2                 # radial shell layers sphere -> cube
ORDER = 4                    # polynomial order: (ORDER+1)**3 GLL nodes per hex
OUT_NAME = "high_order_hex"

# boundary name -> Nek BC code, applied only at export
GROUPS = {"inlet": "v  ", "outlet": "O  ", "sphere": "W  ",
          "top": "SYM", "bottom": "SYM", "front": "SYM", "back": "SYM"}

# -- two closed order-N quad surfaces, paired by index -----------------------
cube = QuadMesh.box(S, N_FACE, order=ORDER, face_tags={
    "x_max": "outlet", "x_min": "inlet",
    "y_max": "top", "y_min": "bottom",
    "z_max": "back", "z_min": "front"})
sphere = QuadMesh.sphere(R, N_FACE, order=ORDER)

# -- fill the shell sphere -> cube -------------------------------------------
mesh = HexMesh.annulus(sphere, cube, radial=np.linspace(0.0, 1.0, N_RADIAL + 1))

# inner-wall nodes sit on the true sphere, to machine precision.  The wall is a
# set of *shared faces* in the hex B-rep, so read it straight off the entities:
# ``mesh.hex`` maps each hex's 6 local faces to face ids in ``mesh.quads`` (the
# QuadMesh of unique faces), whose own B-rep holds the corner points, the shared
# edge nodes, and each face's private interior nodes.
rc = np.linalg.norm(mesh.points[mesh.hexes], axis=2)
inner = np.all(np.isclose(rc[:, HexMesh.FACE_POINTS[4]], R, atol=1e-9), axis=1)
faces = mesh.quads                                   # shared-face B-rep
fid = mesh.hex[inner, 4]                             # face 5 = inner (sphere) wall
radii = np.linalg.norm(np.vstack([
    faces.points[faces.quads[fid]].reshape(-1, 3),          # face corners
    faces.lines.interior[faces.quad[fid]].reshape(-1, 3),   # shared edge nodes
    faces.interior[fid].reshape(-1, 3)]), axis=1)           # private face nodes
_log.info("order-%d shell: %d hexes, %d nodes/hex",
          mesh.order, mesh.n_hexes, (ORDER + 1) ** 3)
_log.info("inner-wall node radius: min=%.15f max=%.15f (target %.1f)",
          radii.min(), radii.max(), R)

# -- export ------------------------------------------------------------------
export.to_re2(mesh, OUT_NAME, groups=GROUPS)          # linear corners (Nek)
export.to_vtu(mesh, OUT_NAME + ".vtu", groups=GROUPS)  # high-order Lagrange hexes (XML)
_log.info("wrote %s.re2 / .rea / .vtu", OUT_NAME)
