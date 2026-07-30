"""Flow past a hemisphere on the ground (external flow).

Like ``flow_past_sphere.py`` -- a cubed-sphere shell between body and far field,
built with :meth:`HexMesh.annulus` -- but only the **upper half**: a hemisphere of
radius ``R`` resting on the floor ``z=0``, the domain filling the half-box
``[-S,S] x [-S,S] x [0,S]``.

:meth:`QuadMesh.half_box` is ``box`` with the ``-z`` patch dropped and its four
upright sides restricted to ``z >= 0``, so it is open at the ground rim;
:meth:`QuadMesh.hemisphere` is that same surface projected onto the sphere, so the
two carry identical connectivity and pair point-for-point. ``annulus`` blends the
radial shells (clustered toward the body) and turns each surface's per-quad tags
into the inner / outer wall faces; the **rim** edges tagged ``ground`` on the inner
surface sweep into the shell's side faces -- the flat ground annulus at ``z=0``.

At ``ORDER > 1`` every node of the hemisphere patches -- corners, shared edge
interiors and private quad interiors -- lies on the exact sphere, so the wall is
genuinely curved rather than straight-subdivided.

    PYTHONPATH=. python examples/flow_past_hemisphere.py

Produces ``flow_past_hemisphere.re2`` / ``.rea`` and ``.vtu``.
"""

import logging

from nekmeshpy import HexMesh, QuadMesh, export
from nekmeshpy.model.fields import geometric_spacing

logging.basicConfig(level=logging.INFO, format="%(message)s")

# -- parameters --------------------------------------------------------------
R = 0.5                      # hemisphere radius (rests on the floor z=0)
S = 4.0                      # far-field half-box: [-S,S] x [-S,S] x [0,S]
N_FACE = 6                   # horizontal cells per direction on each patch
N_HALF = 6                   # vertical cells over z in [0,S] on the side patches
N_RADIAL = 12                # radial cells from hemisphere out to the box
RADIAL_GRADING = 1.15        # >1 clusters radial layers toward the hemisphere
ORDER = 2                    # polynomial order; 1 = linear.  Both surfaces are
                             # built at ORDER (annulus rejects a mismatch), so the
                             # inner-wall nodes bow onto the true hemisphere
                             # (curved .vtu; .re2 stays linear either way)
OUT_NAME = "flow_past_hemisphere"

# boundary name -> Nek BC code, applied only at export
GROUPS = {"inlet": "v  ", "outlet": "O  ", "hemisphere": "W  ", "ground": "W  ",
          "top": "SYM", "front": "SYM", "back": "SYM"}

# -- two paired surfaces: outer half box (tagged per patch) and inner hemisphere
# half_box tags each patch with the far-field side it forms; hemisphere reuses the
# same (N_FACE, N_HALF) connectivity, so the two pair by index for annulus.  The
# ground rim rides on the inner surface, whose boundaries the shells inherit.
outer = QuadMesh.half_box(S, N_FACE, n_vertical=N_HALF, order=ORDER, face_tags={
    "x_max": "outlet", "x_min": "inlet",
    "y_max": "back", "y_min": "front", "z_max": "top"})
inner = QuadMesh.hemisphere(R, N_FACE, n_vertical=N_HALF, order=ORDER,
                            rim_tag="ground")

# fill the shell hemisphere -> half box, radial clustered toward the body; inner cap
# tagged `hemisphere`, outer cap per patch tag, rim swept into the `ground` annulus
mesh = HexMesh.annulus(inner, outer,
                       radial=geometric_spacing(N_RADIAL, RADIAL_GRADING))

# -- report + export ---------------------------------------------------------
print(mesh.report())
export.to_re2(mesh, OUT_NAME, groups=GROUPS)
export.to_vtu(mesh, OUT_NAME + ".vtu", groups=GROUPS)
print("groups:", ", ".join(mesh.boundary_group_tags))
