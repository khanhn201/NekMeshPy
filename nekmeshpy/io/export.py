"""Export / generic-view free functions for a ``HexMesh``.

Exports to a shared-point ``Mesh``, meshio, or native Nek ``.re2``/``.rea`` and
``.vtk``. The ``groups`` parameter maps each boundary name to a Nek BC code and tag.
"""

from __future__ import annotations

import logging
import os
import struct
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Union

import numpy as np

from .._typing import FloatArray, IntArray
from ..model import topology
from ..model.mesh import Mesh
from ..model.physical import PhysicalGroup, PhysicalGroups

if TYPE_CHECKING:
    from ..hexmesh import HexMesh

# .rea header/footer templates
_TEMPLATES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")
_log = logging.getLogger("nekmeshpy")

# accepted types for the ``groups`` export parameter
GroupsArg = Union[PhysicalGroups, Mapping[str, str], None]


def _as_groups(mesh: HexMesh, groups: GroupsArg) -> PhysicalGroups:
    """Normalise the ``groups`` argument to a ``PhysicalGroups``.

    A ``PhysicalGroups`` passes through; a ``{name: nek_code}`` mapping becomes a
    registry with 1-based tags in insertion order; ``None`` auto-numbers the mesh's
    distinct boundary names.
    """
    if isinstance(groups, PhysicalGroups):
        return groups
    if groups is None:
        names = mesh.boundary_group_tags
        return PhysicalGroups(
            PhysicalGroup(name, i + 1) for i, name in enumerate(names))
    return PhysicalGroups(
        PhysicalGroup(name, i + 1, 2, code)
        for i, (name, code) in enumerate(groups.items()))


# -- generic mesh view --------------------------------------------------
def to_mesh(mesh: HexMesh, groups: GroupsArg = None) -> Mesh:
    """Return a shared-point ``Mesh``: welded points, ``hexahedron`` cells, and one
    ``quad`` boundary cell per tagged face grouped into named ``cell_sets``."""
    X, HC, _ = mesh.weld()
    g = _as_groups(mesh, groups)
    b = mesh.boundaries
    bnames = mesh.boundary_tags

    conn_rows = []           # welded point ids of each boundary face
    name_rows = []           # name of each boundary face
    for r in range(b.shape[0]):
        elem, face = int(b[r, 0]), int(b[r, 1])
        conn_rows.append(HC[elem, mesh.FACE_POINTS[face - 1, :]])
        name_rows.append(str(bnames[r]))
    quad_conn = (np.array(conn_rows, dtype=np.int64) if conn_rows
                 else np.zeros((0, 4), np.int64))
    quad_name = np.array(name_rows, dtype=np.str_)

    cells = {"hexahedron": HC}
    if quad_conn.shape[0]:
        cells["quad"] = quad_conn

    cell_sets: dict[str, dict[str, IntArray]] = {}
    point_sets: dict[str, IntArray] = {}
    field_data: dict[str, IntArray] = {}
    for grp in g:
        sel = np.flatnonzero(quad_name == grp.name)
        if sel.size == 0:
            continue
        cell_sets[grp.name] = {"quad": sel}
        point_sets[grp.name] = np.unique(quad_conn[sel].ravel())
        field_data[grp.name] = np.array([grp.tag, grp.dim], dtype=np.int64)

    return Mesh(points=X, cells=cells, point_sets=point_sets,
                cell_sets=cell_sets, field_data=field_data)


def to_meshio(mesh: HexMesh, groups: GroupsArg = None) -> Any:
    """Return a meshio mesh view (requires ``meshio``)."""
    return to_mesh(mesh, groups).to_meshio()


def write(mesh: HexMesh, path: str, file_format: str | None = None,
          *, groups: GroupsArg = None) -> str:
    """Write through meshio to any supported format; for native Nek use ``to_re2``."""
    return to_mesh(mesh, groups).write(path, file_format=file_format)


# -- native Nek export --------------------------------------------------
def _str_to_double(s: str) -> float:
    b = bytearray(8)
    for i, ch in enumerate(s):
        b[i] = ord(ch)
    return struct.unpack("<d", bytes(b))[0]


def to_re2(mesh: HexMesh, filename: str, *, groups: GroupsArg = None) -> HexMesh:
    """Write ``<filename>.re2`` (binary) and ``<filename>.rea`` (ASCII)."""
    g = _as_groups(mesh, groups)
    elements = mesh.points[mesh.hexes]            # (N,8,3) per-element coords
    boundaries = mesh.boundaries
    bnames = mesh.boundary_tags
    num_elem = elements.shape[0]
    with open(filename + ".rea", "w") as fh:
        with open(os.path.join(_TEMPLATES, "header.rea"), "r") as hf:
            fh.write(hf.read())
        fh.write("**MESH DATA** 6 lines are X,Y,Z;X,Y,Z. Columns corners 1-4;5-8\n")
        fh.write("      %8i   %8i   %8i NELT,NDIM,NELV\n" % (-num_elem, 3, num_elem))
        with open(os.path.join(_TEMPLATES, "footer.rea"), "r") as ff:
            fh.write(ff.read())
        fh.write("\n")
    with open(filename + ".re2", "wb") as fid:
        header = "#v004%16d%3d%16d%4d hdr" % (num_elem, 3, num_elem, 1)
        fid.write(header.ljust(80).encode("ascii"))
        fid.write(struct.pack("<f", np.float32(6.54321)))
        for i in range(num_elem):
            fid.write(struct.pack("<d", 0.0))
            fid.write(elements[i, :, 0].astype("<f8").tobytes())
            fid.write(elements[i, :, 1].astype("<f8").tobytes())
            fid.write(elements[i, :, 2].astype("<f8").tobytes())
        fid.write(struct.pack("<d", 0.0))
        fid.write(struct.pack("<d", float(boundaries.shape[0])))
        for b in range(boundaries.shape[0]):
            elem = int(boundaries[b, 0]) + 1
            face = int(boundaries[b, 1])
            name = str(bnames[b])
            buf2: FloatArray = np.zeros(8, dtype="<f8")
            buf2[0] = float(elem)
            buf2[1] = float(face)
            grp = g.get(name)
            if grp is not None:
                buf2[7] = _str_to_double(grp.code)
            else:
                _log.warning("unknown boundary name: %s", name)
            fid.write(buf2.tobytes())
    return mesh


def to_vtk(mesh: HexMesh, fname: str, *, groups: GroupsArg = None) -> HexMesh:
    """Write an ASCII VTK unstructured grid with per-point ``bc_id`` tags."""
    g = _as_groups(mesh, groups)
    elements = mesh.points[mesh.hexes]            # (N,8,3) per-element coords
    boundaries = mesh.boundaries
    bnames = mesh.boundary_tags
    N = elements.shape[0]
    X = elements.reshape(N * 8, 3)
    nX = X.shape[0]
    with open(fname, "w") as fid:
        fid.write("# vtk DataFile Version 2.0\n")
        fid.write("Hexes\n")
        fid.write("ASCII\n")
        fid.write("DATASET UNSTRUCTURED_GRID\n")
        fid.write("\n")
        fid.write("POINTS %d float\n" % nX)
        for r in range(nX):
            fid.write("%f %f %f \n" % (X[r, 0], X[r, 1], X[r, 2]))
        fid.write("\n")
        fid.write("CELLS %d %d\n" % (N, N * 9))
        for e in range(N):
            base = 8 * e
            fid.write("8 %s\n" % " ".join(str(base + k) for k in range(8)))
        fid.write("\n")
        fid.write("CELL_TYPES %d\n" % N)
        for _ in range(N):
            fid.write("12\n")
        fid.write("\n")
        iftoiv = mesh.FACE_POINTS + 1
        tmp = np.zeros((8, N), dtype=np.int64)
        for i in range(boundaries.shape[0]):
            face = int(boundaries[i, 1])
            elem = int(boundaries[i, 0])
            name = str(bnames[i])
            grp = g.get(name)
            if grp is None:
                _log.warning("unknown boundary name: %s", name)
                continue
            for j in iftoiv[face - 1, :]:
                tmp[j - 1, elem] = grp.tag
        fid.write("POINT_DATA %d\n" % (N * 8))
        fid.write("SCALARS bc_id int 1\n")
        fid.write("LOOKUP_TABLE default\n")
        for val in tmp.flatten(order="F"):
            fid.write("%d\n" % val)
        fid.write("\n")
    return mesh


# -- reporting ----------------------------------------------------------
def summary(mesh: HexMesh) -> None:
    """Log element/boundary counts, per-name face totals, and the topology report."""
    _log.info("mesh: %d hex elements, %d boundary faces",
              mesh.hexes.shape[0], mesh.boundaries.shape[0])
    for name in mesh.boundary_group_tags:
        _log.info("  %-14s: %d faces",
                  name, int(np.sum(mesh.boundary_tags == name)))
    rep = topology.hex_report(*mesh.weld()[:2])
    _log.info("  watertight=%s  conformal=%s  components=%d  "
              "non-manifold faces=%d  hanging points=%d",
              rep["watertight"], rep["conformal"], rep["n_components"],
              rep["n_nonmanifold_faces"], rep["n_hanging_points"])
