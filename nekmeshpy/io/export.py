"""Export / generic-view free functions for a :class:`HexMesh`.

:class:`HexMesh` is a pure hex container (``elements`` (N,8,3) + ``boundaries``);
turning it into a shared-node :class:`~nekmeshpy.model.mesh.Mesh`, a meshio mesh, an
arbitrary meshio file, or the native Nek ``.re2``/``.rea`` and ASCII ``.vtk``
lives here.  The byte layout is ported verbatim, so the exported files stay
byte-identical to the reference.
"""

import logging
import os
import struct

import numpy as np

from ..model.mesh import Mesh

# package root (one level up from this io/ subpackage), holding templates/
_PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_log = logging.getLogger("nekmeshpy")


# -- generic mesh view --------------------------------------------------
def to_mesh(mesh):
    """Return a shared-node :class:`~nekmeshpy.model.mesh.Mesh`: welded points,
    ``hexahedron`` volume cells, plus one ``quad`` boundary cell per tagged
    face grouped into named ``cell_sets`` (and ``point_sets``) taken from
    ``mesh.physical_groups``."""
    mesh.finalize()
    X, HC, _ = mesh.weld()
    groups = mesh.physical_groups
    b = mesh.boundaries

    quad_conn = []           # welded node ids of each boundary face
    quad_tag = []            # tag of each boundary face
    for r in range(b.shape[0]):
        elem, face, tag = int(b[r, 0]), int(b[r, 1]), int(b[r, 2])
        quad_conn.append(HC[elem, mesh.FACE_NODES[face - 1, :]])
        quad_tag.append(tag)
    quad_conn = (np.array(quad_conn, dtype=np.int64) if quad_conn
                 else np.zeros((0, 4), np.int64))
    quad_tag = np.array(quad_tag, dtype=np.int64)

    cells = {"hexahedron": HC}
    if quad_conn.shape[0]:
        cells["quad"] = quad_conn

    cell_sets, point_sets, field_data = {}, {}, {}
    for g in groups:
        sel = np.flatnonzero(quad_tag == g.tag)
        if sel.size == 0:
            continue
        cell_sets[g.name] = {"quad": sel}
        point_sets[g.name] = np.unique(quad_conn[sel].ravel())
        field_data[g.name] = np.array([g.tag, g.dim], dtype=np.int64)

    return Mesh(points=X, cells=cells, point_sets=point_sets,
                cell_sets=cell_sets, field_data=field_data)


def to_meshio(mesh):
    """Return a :class:`meshio.Mesh` view (requires ``meshio``)."""
    return to_mesh(mesh).to_meshio()


def write(mesh, path, file_format=None):
    """Write through :mod:`meshio` to any supported format
    (``.vtu``, ``.msh``, ``.xdmf``, ``.exo`` ...).  For the native Nek
    formats use :func:`to_re2`."""
    return to_mesh(mesh).write(path, file_format=file_format)


# -- native Nek export --------------------------------------------------
def _str_to_double(s):
    b = bytearray(8)
    for i, ch in enumerate(s):
        b[i] = ord(ch)
    return struct.unpack("<d", bytes(b))[0]


def to_re2(mesh, filename):
    """Write ``<filename>.re2`` (binary) and ``<filename>.rea`` (ASCII)."""
    mesh.finalize()
    elements = mesh.elements
    boundaries = mesh.boundaries
    num_elem = elements.shape[0]
    with open(filename + ".rea", "w") as fh:
        with open(os.path.join(_PKG, "templates", "header.rea"), "r") as hf:
            fh.write(hf.read())
        fh.write("**MESH DATA** 6 lines are X,Y,Z;X,Y,Z. Columns corners 1-4;5-8\n")
        fh.write("      %8i   %8i   %8i NELT,NDIM,NELV\n" % (-num_elem, 3, num_elem))
        with open(os.path.join(_PKG, "templates", "footer.rea"), "r") as ff:
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
        groups = mesh.physical_groups
        for b in range(boundaries.shape[0]):
            elem = int(boundaries[b, 0]) + 1
            face = int(boundaries[b, 1])
            tag = int(boundaries[b, 2])
            buf2 = np.zeros(8, dtype="<f8")
            buf2[0] = float(elem)
            buf2[1] = float(face)
            code = groups.code_for(tag)
            if code is not None:
                buf2[7] = _str_to_double(code)
            else:
                _log.warning("unknown tag: %s", tag)
            fid.write(buf2.tobytes())
    return mesh


def to_vtk(mesh, fname):
    """Write an ASCII VTK unstructured grid with per-node ``bc_id``."""
    mesh.finalize()
    elements = mesh.elements
    boundaries = mesh.boundaries
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
        iftoiv = mesh.FACE_NODES + 1
        tmp = np.zeros((8, N), dtype=np.int64)
        for i in range(boundaries.shape[0]):
            face = int(boundaries[i, 1])
            elem = int(boundaries[i, 0])
            tag = int(boundaries[i, 2])
            for j in iftoiv[face - 1, :]:
                tmp[j - 1, elem] = tag
        fid.write("POINT_DATA %d\n" % (N * 8))
        fid.write("SCALARS bc_id int 1\n")
        fid.write("LOOKUP_TABLE default\n")
        for val in tmp.flatten(order="F"):
            fid.write("%d\n" % val)
        fid.write("\n")
    return mesh


# -- reporting ----------------------------------------------------------
def summary(mesh, cfg):
    """Log the element/boundary counts and the per-tag face totals for the
    bifurcation physical groups defined in ``cfg``."""
    mesh.finalize()
    _log.info("mesh: %d hex elements, %d boundary faces",
              mesh.elements.shape[0], mesh.boundaries.shape[0])
    labels = ["wall", "trunk outlet", "top outlet 1", "top outlet 2"]
    tags = [cfg.tag_wall, cfg.tag_trunk, cfg.tag_top1, cfg.tag_top2]
    for t in range(4):
        _log.info("  tag %d (%-12s): %d faces",
                  tags[t], labels[t], int(np.sum(mesh.boundaries[:, 2] == tags[t])))
