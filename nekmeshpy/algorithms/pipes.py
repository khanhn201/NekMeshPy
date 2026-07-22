"""Straight-pipe / duct hex-meshing algorithms.

Two generic primitives built on the same :class:`~nekmeshpy.geometry.hexmesh.HexMesh`
assembly / weld / quality / export machinery as the bifurcation template:

* :class:`CircularPipe` -- an all-hex O-grid ("butterfly") disc swept along an
  axis, with no degenerate cells at the centre.
* :class:`RectangularPipe` -- a structured rectangular duct.

Both tag three named physical groups -- ``wall`` (side), ``inlet`` (first cap)
and ``outlet`` (last cap) -- with sensible Nek BC codes, and support an
arbitrary sweep axis and geometric axial/radial grading.

    from nekmeshpy import CircularPipe, export
    mesh = CircularPipe(radius=0.5, length=4.0, n_axial=40).run()
    export.to_re2(mesh, "pipe")
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from ..geometry.hexmesh import HexMesh
from ..geometry.quadmesh import QuadMesh
from ..model.fields import geometric_spacing
from ..model.physical import PhysicalGroup, PhysicalGroups
from .registry import register_algorithm

Vec = Sequence[float]

# wall / inlet / outlet, matching the Nek default codes used elsewhere
_PIPE_GROUPS = PhysicalGroups([
    PhysicalGroup("wall", 1, 2, "W  "),
    PhysicalGroup("inlet", 2, 2, "v  "),
    PhysicalGroup("outlet", 3, 2, "O  "),
])


def _axis_frame(axis: Vec) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Right-handed orthonormal frame ``(e1, e2, ez)`` with ``ez`` along
    ``axis``.  ``e1``/``e2`` span the cross-section plane."""
    ez: np.ndarray = np.asarray(axis, dtype=float)
    n = np.linalg.norm(ez)
    if n == 0:
        raise ValueError("pipe axis must be non-zero")
    ez = ez / n
    ref = np.array([1.0, 0.0, 0.0]) if abs(ez[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = np.cross(ref, ez)
    e1 = e1 / np.linalg.norm(e1)
    e2 = np.cross(ez, e1)
    return e1, e2, ez


def rectangular_section(width: float, height: float, nx: int, ny: int
                        ) -> tuple[np.ndarray, np.ndarray, set]:
    """Structured quad grid over ``[-w/2,w/2] x [-h/2,h/2]``.

    Returns ``(nodes (K,2), quads (M,4), wall_edges)`` where ``wall_edges`` is a
    set of ``frozenset({i,j})`` node-index pairs on the outer perimeter.
    """
    xs = np.linspace(-width / 2.0, width / 2.0, nx + 1)
    ys = np.linspace(-height / 2.0, height / 2.0, ny + 1)

    def nid(i: int, j: int) -> int:
        return i * (ny + 1) + j

    nodes = np.array([[xs[i], ys[j]] for i in range(nx + 1) for j in range(ny + 1)],
                     dtype=float)
    quads = np.array([[nid(i, j), nid(i + 1, j), nid(i + 1, j + 1), nid(i, j + 1)]
                      for i in range(nx) for j in range(ny)], dtype=np.int64)
    wall: set = set()
    for j in range(ny):
        wall.add(frozenset((nid(0, j), nid(0, j + 1))))
        wall.add(frozenset((nid(nx, j), nid(nx, j + 1))))
    for i in range(nx):
        wall.add(frozenset((nid(i, 0), nid(i + 1, 0))))
        wall.add(frozenset((nid(i, ny), nid(i + 1, ny))))
    return nodes, quads, wall


def circular_section(radius: float, n_side: int, n_radial: int,
                     center_scale: float = 0.5, radial_grading: float = 1.0
                     ) -> tuple[np.ndarray, np.ndarray, set]:
    """All-hex "butterfly" O-grid over a disc of the given ``radius``.

    A central ``n_side x n_side`` square block is surrounded by ``n_radial``
    O-ring layers blending its perimeter out to the circle, so there is no
    collapsed cell at the centre.  ``center_scale`` sets the square corner
    distance as a fraction of ``radius``; ``radial_grading`` geometrically
    grades the O-ring layers (``>1`` clusters toward the wall).

    Returns ``(nodes (K,2), quads (M,4), wall_edges)``.
    """
    if n_side < 1 or n_radial < 1:
        raise ValueError("circular_section needs n_side >= 1 and n_radial >= 1")
    a = center_scale * radius / np.sqrt(2.0)

    # central square grid
    coords = np.linspace(-a, a, n_side + 1)

    def cid(i: int, j: int) -> int:
        return i * (n_side + 1) + j

    nodes = [[coords[i], coords[j]] for i in range(n_side + 1) for j in range(n_side + 1)]
    quads = [[cid(i, j), cid(i + 1, j), cid(i + 1, j + 1), cid(i, j + 1)]
             for i in range(n_side) for j in range(n_side)]

    # ordered CCW perimeter of the central square (4*n_side nodes)
    peri_ids = ([cid(i, 0) for i in range(n_side + 1)]        # bottom  j=0
                + [cid(n_side, j) for j in range(1, n_side + 1)]   # right i=n_side
                + [cid(i, n_side) for i in range(n_side - 1, -1, -1)]  # top j=n_side
                + [cid(0, j) for j in range(n_side - 1, 0, -1)])       # left i=0
    P = len(peri_ids)                                        # == 4*n_side
    nodes_arr = np.asarray(nodes, dtype=float)
    peri_pos = nodes_arr[peri_ids, :]
    circle_pos = radius * peri_pos / np.linalg.norm(peri_pos, axis=1, keepdims=True)

    # radial layer fractions in (0,1], layer 0 being the square perimeter itself
    fracs = geometric_spacing(n_radial, radial_grading)[1:]

    ring = [list(peri_ids)]
    for t in fracs:
        layer = (1.0 - t) * peri_pos + t * circle_pos
        base = len(nodes)
        nodes.extend(layer.tolist())
        ring.append(list(range(base, base + P)))

    for r in range(n_radial):
        a_ids, b_ids = ring[r], ring[r + 1]
        for k in range(P):
            kn = (k + 1) % P
            quads.append([a_ids[k], a_ids[kn], b_ids[kn], b_ids[k]])

    wall: set = set()
    outer = ring[n_radial]
    for k in range(P):
        wall.add(frozenset((outer[k], outer[(k + 1) % P])))

    return np.asarray(nodes, dtype=float), np.asarray(quads, dtype=np.int64), wall


class _SweptPipe:
    """Shared sweep/extrude driver: a 2-D section swept along an axis into a
    :class:`HexMesh` with wall / inlet / outlet groups."""

    length: float
    n_axial: int
    axis: Vec
    center: Vec
    axial_grading: float
    groups: PhysicalGroups

    def _section(self) -> tuple[np.ndarray, np.ndarray, set]:
        raise NotImplementedError

    def run(self) -> HexMesh:
        nodes2d, quads, wall_edges = self._section()
        e1, e2, ez = _axis_frame(self.axis)
        origin: np.ndarray = np.asarray(self.center, dtype=float)
        s = geometric_spacing(self.n_axial, self.axial_grading) * float(self.length)

        plane = nodes2d[:, 0:1] * e1[None, :] + nodes2d[:, 1:2] * e2[None, :]
        slices = [QuadMesh(origin[None, :] + plane + sk * ez[None, :], quads,
                           wall_edges=wall_edges) for sk in s]

        mesh = HexMesh(groups=self.groups)
        mesh.add_extruded_section(
            slices,
            first_cap_tag=self.groups.tag_for("inlet"),
            last_cap_tag=self.groups.tag_for("outlet"),
            wall_tag=self.groups.tag_for("wall"))
        return mesh.finalize()


@register_algorithm("circular_pipe")
class CircularPipe(_SweptPipe):
    """All-hex circular pipe via a swept O-grid disc."""

    def __init__(self, radius: float = 0.5, length: float = 1.0,
                 n_axial: int = 10, n_side: int = 4, n_radial: int = 3,
                 center_scale: float = 0.5, radial_grading: float = 1.0,
                 axial_grading: float = 1.0, axis: Vec = (0.0, 0.0, 1.0),
                 center: Vec = (0.0, 0.0, 0.0),
                 groups: PhysicalGroups | None = None) -> None:
        self.radius = float(radius)
        self.length = float(length)
        self.n_axial = int(n_axial)
        self.n_side = int(n_side)
        self.n_radial = int(n_radial)
        self.center_scale = float(center_scale)
        self.radial_grading = float(radial_grading)
        self.axial_grading = float(axial_grading)
        self.axis = axis
        self.center = center
        self.groups = groups or _PIPE_GROUPS

    def _section(self) -> tuple[np.ndarray, np.ndarray, set]:
        return circular_section(self.radius, self.n_side, self.n_radial,
                                self.center_scale, self.radial_grading)


@register_algorithm("rectangular_pipe")
class RectangularPipe(_SweptPipe):
    """Structured rectangular duct."""

    def __init__(self, width: float = 1.0, height: float = 1.0, length: float = 1.0,
                 nx: int = 8, ny: int = 8, n_axial: int = 10,
                 axial_grading: float = 1.0, axis: Vec = (0.0, 0.0, 1.0),
                 center: Vec = (0.0, 0.0, 0.0),
                 groups: PhysicalGroups | None = None) -> None:
        self.width = float(width)
        self.height = float(height)
        self.length = float(length)
        self.nx = int(nx)
        self.ny = int(ny)
        self.n_axial = int(n_axial)
        self.axial_grading = float(axial_grading)
        self.axis = axis
        self.center = center
        self.groups = groups or _PIPE_GROUPS

    def _section(self) -> tuple[np.ndarray, np.ndarray, set]:
        return rectangular_section(self.width, self.height, self.nx, self.ny)
