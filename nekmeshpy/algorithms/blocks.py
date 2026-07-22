"""Structured hex primitives -- generic building blocks independent of the
bifurcation pipeline.

:class:`TransfiniteBlock` trilinearly interpolates eight corner points into an
``nx x ny x nz`` grid of hexes, tags the six boundary faces as named physical
groups, and returns a :class:`~nekmeshpy.geometry.hexmesh.HexMesh`.  Per-axis grading can
be a geometric ratio or driven by a :mod:`nekmeshpy.model.fields` size field.

It is registered as the ``transfinite_block`` algorithm and exercises exactly
the same assembly / weld / quality / export machinery as the bifurcation
template -- demonstrating that :class:`HexMesh` is a generic hex container.
"""

import numpy as np

from ..geometry.hexmesh import HexMesh
from ..model import fields as _fields
from ..model.physical import PhysicalGroup, PhysicalGroups
from .registry import register_algorithm

# Nek corner parameter coordinates (u,v,w) in the unit cube
_CORNER_UVW = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
                        [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]], dtype=float)

# default face groups for the six block sides (xi-/xi+/eta-/eta+/zeta-/zeta+)
_BLOCK_GROUPS = PhysicalGroups([
    PhysicalGroup("x_min", 1, 2, "W  "),
    PhysicalGroup("x_max", 2, 2, "W  "),
    PhysicalGroup("y_min", 3, 2, "W  "),
    PhysicalGroup("y_max", 4, 2, "W  "),
    PhysicalGroup("z_min", 5, 2, "v  "),
    PhysicalGroup("z_max", 6, 2, "O  "),
])


@register_algorithm("transfinite_block")
class TransfiniteBlock:
    def __init__(self, corners, divisions=(4, 4, 4), grading=(1.0, 1.0, 1.0),
                 size_field=None, groups=None):
        self.corners = np.asarray(corners, dtype=float).reshape(8, 3)
        self.nx, self.ny, self.nz = (int(d) for d in divisions)
        self.grading = tuple(grading)
        self.size_field = size_field
        self.groups = groups or _BLOCK_GROUPS

    # -- corner map ------------------------------------------------------
    def _trilinear(self, u, v, w):
        """Map parameter (u,v,w) in the unit cube to physical space."""
        cu, cv, cw = _CORNER_UVW[:, 0], _CORNER_UVW[:, 1], _CORNER_UVW[:, 2]
        wgt = (np.where(cu == 0, 1 - u, u)
               * np.where(cv == 0, 1 - v, v)
               * np.where(cw == 0, 1 - w, w))
        return wgt @ self.corners

    def _axis_positions(self, n, ratio, i0, i1):
        """Normalized node positions along one axis (0..1)."""
        if self.size_field is not None:
            p0 = self._trilinear(*i0)
            p1 = self._trilinear(*i1)
            return _fields.distribution_from_field(self.size_field, p0, p1)
        return _fields.geometric_spacing(n, ratio)

    # -- build -----------------------------------------------------------
    def run(self):
        us = self._axis_positions(self.nx, self.grading[0], (0, .5, .5), (1, .5, .5))
        vs = self._axis_positions(self.ny, self.grading[1], (.5, 0, .5), (.5, 1, .5))
        ws = self._axis_positions(self.nz, self.grading[2], (.5, .5, 0), (.5, .5, 1))
        nx, ny, nz = len(us) - 1, len(vs) - 1, len(ws) - 1

        # node grid P[i,j,k]
        P = np.empty((len(us), len(vs), len(ws), 3))
        for i, u in enumerate(us):
            for j, v in enumerate(vs):
                for k, w in enumerate(ws):
                    P[i, j, k] = self._trilinear(u, v, w)

        mesh = HexMesh(groups=self.groups)
        gtag = {name: self.groups.tag_for(name) for name in
                ("x_min", "x_max", "y_min", "y_max", "z_min", "z_max")}
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    nodes = np.array([
                        P[i, j, k], P[i + 1, j, k], P[i + 1, j + 1, k], P[i, j + 1, k],
                        P[i, j, k + 1], P[i + 1, j, k + 1], P[i + 1, j + 1, k + 1],
                        P[i, j + 1, k + 1]])
                    eid = mesh.add_hex(nodes)
                    if i == 0:
                        mesh._tag(eid, 4, gtag["x_min"])
                    if i == nx - 1:
                        mesh._tag(eid, 2, gtag["x_max"])
                    if j == 0:
                        mesh._tag(eid, 1, gtag["y_min"])
                    if j == ny - 1:
                        mesh._tag(eid, 3, gtag["y_max"])
                    if k == 0:
                        mesh._tag(eid, 5, gtag["z_min"])
                    if k == nz - 1:
                        mesh._tag(eid, 6, gtag["z_max"])
        return mesh.finalize()
