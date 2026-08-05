"""Corner-defined structured hex block, exported for Nek5000.

Trilinearly interpolates eight corners into an ``nx x ny x nz`` grid and hands it
to :meth:`HexMesh.from_grid`. Per-axis grading is a geometric ratio, or drive
spacing from a :mod:`nekmeshpy.model.fields` size field (``SIZE_FIELD``).

    PYTHONPATH=. python examples/transfinite_block.py
"""

import logging

import numpy as np

from nekmeshpy import export, hexmesh
from nekmeshpy.model import fields

logging.basicConfig(level=logging.INFO, format="%(message)s")

# -- parameters --------------------------------------------------------------
# eight corners in Nek order (bottom face CCW, then top face)
CORNERS = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
                    [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]], dtype=float)
DIVISIONS = (4, 4, 4)             # nx, ny, nz
GRADING = (1.0, 1.0, 1.0)         # per-axis geometric ratio (1 = uniform)
SIZE_FIELD = None                 # e.g. fields.ConstantField(0.1) to drive spacing
ORDER = 2                         # polynomial order; 1 = linear.  from_grid places
                                  # straight-sided (trilinear) GLL nodes; the block
                                  # is flat, so they land on the same faces
OUT_NAME = "block"

# boundary name -> Nek BC code, applied only at export (tags follow this order)
GROUPS = {"x_min": "W  ", "x_max": "W  ", "y_min": "W  ", "y_max": "W  ",
          "z_min": "v  ", "z_max": "O  "}
SIDES = ("x_min", "x_max", "y_min", "y_max", "z_min", "z_max")

# Nek corner parameter coordinates (u,v,w) in the unit cube
_CORNER_UVW = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
                        [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]], dtype=float)


# -- helpers -----------------------------------------------------------------
def trilinear(corners, u, v, w):
    """Map parameter ``(u,v,w)`` in the unit cube to physical space."""
    cu, cv, cw = _CORNER_UVW[:, 0], _CORNER_UVW[:, 1], _CORNER_UVW[:, 2]
    wgt = (np.where(cu == 0, 1 - u, u)
           * np.where(cv == 0, 1 - v, v)
           * np.where(cw == 0, 1 - w, w))
    return wgt @ corners


def axis_positions(n, ratio, i0, i1):
    """Normalized positions (0..1) along one axis: from the size field if set,
    else geometric grading."""
    if SIZE_FIELD is not None:
        return fields.distribution_from_field(
            SIZE_FIELD, trilinear(CORNERS, *i0), trilinear(CORNERS, *i1))
    return fields.geometric_spacing(n, ratio)


# -- build -------------------------------------------------------------------
nx, ny, nz = (int(d) for d in DIVISIONS)
us = axis_positions(nx, GRADING[0], (0, .5, .5), (1, .5, .5))
vs = axis_positions(ny, GRADING[1], (.5, 0, .5), (.5, 1, .5))
ws = axis_positions(nz, GRADING[2], (.5, .5, 0), (.5, .5, 1))

P = np.empty((len(us), len(vs), len(ws), 3))
for i, u in enumerate(us):
    for j, v in enumerate(vs):
        for k, w in enumerate(ws):
            P[i, j, k] = trilinear(CORNERS, u, v, w)

mesh = hexmesh.from_grid(P, side_tags={s: s for s in SIDES}, order=ORDER)

# -- report + export ---------------------------------------------------------
stats = hexmesh.quality_summary(mesh)
print("block: %d hex elements, %d points" % (mesh.n_hexes, mesh.n_points))
print("scaled Jacobian: min=%.4f mean=%.4f" % (stats.min, stats.mean))

export.to_re2(mesh, OUT_NAME + ".re2", groups=GROUPS)
export.to_vtu(mesh, OUT_NAME + ".vtu", groups=GROUPS)
print("groups:", ", ".join(mesh.face_group_tags))
