"""Export / generic-view free functions for a ``HexMesh``.

Exports to a shared-point ``Mesh``, meshio, or native Nek ``.re2``/``.rea`` and
VTK XML (``.vtu``). The ``.vtu`` writer emits high-order VTK Lagrange cells at
``order > 1``; ``.re2`` always stays linear. The ``groups`` parameter maps each
boundary name to a Nek BC code and tag.
"""

from __future__ import annotations

import logging
import os
import struct
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Union

import numpy as np

from .._typing import FloatArray, IntArray
from ..model import conform, topology
from ..model.interp import hex_face_indices
from ..model.mesh import Mesh
from ..model.physical import PhysicalGroup, PhysicalGroups

if TYPE_CHECKING:
    from ..hexmesh import HexMesh
    from ..linemesh import LineMesh
    from ..quadmesh import QuadMesh

# VTK cell-type ids: linear + high-order (Lagrange) line / quad / hex.
_VTK_LINE = 3
_VTK_LAGRANGE_CURVE = 68
_VTK_QUAD = 9
_VTK_LAGRANGE_QUADRILATERAL = 70
_VTK_HEXAHEDRON = 12
_VTK_LAGRANGE_HEXAHEDRON = 72

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


def _hex_point_index(i: int, j: int, k: int, n: int) -> int:
    """VTK_LAGRANGE_HEXAHEDRON connectivity position of lattice node ``(i,j,k)`` at
    order ``n`` per axis -- VTK's ``PointIndexFromIJK`` recursion (corners, then the
    12 edges, then the 6 faces, then the interior)."""
    ibdy = i == 0 or i == n
    jbdy = j == 0 or j == n
    kbdy = k == 0 or k == n
    nbdy = int(ibdy) + int(jbdy) + int(kbdy)
    e = n - 1
    if nbdy == 3:                                        # corner
        return (2 if (i and j) else (1 if i else (3 if j else 0))) + (4 if k else 0)
    offset = 8
    if nbdy == 2:                                        # edge
        if not ibdy:
            return (i - 1) + (2 * e if j else 0) + (4 * e if k else 0) + offset
        if not jbdy:
            return (j - 1) + (e if i else 3 * e) + (4 * e if k else 0) + offset
        offset += 8 * e
        return (k - 1) + e * (3 if (i and j) else (1 if i else (2 if j else 0))) + offset
    offset += 12 * e
    if nbdy == 1:                                        # face
        if ibdy:
            return (j - 1) + e * (k - 1) + (e * e if i else 0) + offset
        offset += 2 * e * e
        if jbdy:
            return (i - 1) + e * (k - 1) + (e * e if j else 0) + offset
        offset += 2 * e * e
        return (i - 1) + e * (j - 1) + (e * e if k else 0) + offset
    offset += 6 * e * e                                  # interior
    return offset + (i - 1) + e * ((j - 1) + e * (k - 1))


def _lagrange_hex_perm(order: int) -> IntArray:
    """Map our lexicographic (``i`` fastest, ``i + row*j + row**2*k``) hex nodes to
    the VTK Lagrange-hexahedron node order (:func:`_hex_point_index`)."""
    n = order
    row = n + 1
    perm: IntArray = np.empty(row ** 3, dtype=np.int64)
    for k in range(row):
        for j in range(row):
            for i in range(row):
                perm[_hex_point_index(i, j, k, n)] = i + row * j + row * row * k
    return perm


def _lagrange_curve_perm(order: int) -> IntArray:
    """Map our lexicographic (ascending) curve nodes ``[0,1,...,N]`` to the VTK
    Lagrange-curve order ``[end0, end1, interior1, ..., interior(N-1)]``."""
    return np.array([0, order, *range(1, order)], dtype=np.int64)


def _lagrange_quad_perm(order: int) -> IntArray:
    """Map our lexicographic (``i`` fastest, index ``i + (order+1)*j``) quad nodes to
    the VTK Lagrange-quadrilateral order: 4 corners, then the four edges
    (bottom / right / top / left, each start->end), then the interior nodes
    (``i`` fastest)."""
    n = order
    row = n + 1

    def idx(i: int, j: int) -> int:
        return i + row * j
    corners = [idx(0, 0), idx(n, 0), idx(n, n), idx(0, n)]
    e0 = [idx(i, 0) for i in range(1, n)]
    e1 = [idx(n, j) for j in range(1, n)]
    e2 = [idx(i, n) for i in range(n - 1, 0, -1)]
    e3 = [idx(0, j) for j in range(n - 1, 0, -1)]
    interior = [idx(i, j) for j in range(1, n) for i in range(1, n)]
    return np.array(corners + e0 + e1 + e2 + e3 + interior, dtype=np.int64)


# -- node-array builders (for the .vtu writer) --------------------------
# Each returns the ``.vtu`` point array, the per-cell connectivity into it already in
# VTK node order, and the VTK cell-type id; the hex builder also returns the per-node
# ``bc_id``.  At ``order == 1`` the nodes stay **un-welded** (one block per element,
# connectivity = consecutive blocks) -- byte-for-byte the historical output.  At
# ``order > 1`` the conformal walk (:mod:`nekmeshpy.model.conform`) emits **shared**
# nodes: a node on an edge / face between two elements is written once.
def _unwelded(n_elem: int, m: int) -> IntArray:
    """Consecutive-block connectivity ``(n_elem, m)`` for un-welded node arrays."""
    return np.arange(n_elem * m, dtype=np.int64).reshape(n_elem, m)


def _hex_arrays(mesh: HexMesh,
                g: PhysicalGroups) -> tuple[FloatArray, IntArray, int, IntArray]:
    """Hex nodes + ``bc_id``: linear un-welded ``VTK_HEXAHEDRON`` at ``order == 1``, a
    conformal (shared-node) ``VTK_LAGRANGE_HEXAHEDRON`` (``(order+1)**3`` GLL nodes per
    cell) above it, whose face nodes inherit the boundary face's tag.

    ``bc_id`` precedence is the un-welded writer's rule, applied in the shared-node
    numbering: the boundary rows are scattered in ``mesh.boundaries`` order and the
    **last row to touch a node wins** (this is exactly how two boundary faces sharing an
    edge *within* one hex have always been resolved).  Welding widens the same rule
    across elements: an untagged element never writes, so a node shared by a tagged face
    and an untagged neighbour keeps its tag; where two *differently* tagged faces of
    different elements meet, the single shared node necessarily carries one of the two
    tags -- the later boundary row's."""
    if mesh.order > 1:
        order = mesh.order
        perm = _lagrange_hex_perm(order)
        nodes, conn_ho = conform.conformal_hex(
            mesh.points, mesh.hexes, mesh._elem_edges, mesh._edge_flip,
            mesh.quads.lines.interior, mesh.hex, mesh.face_orient,
            mesh.quads.interior, mesh.interior, order)
        bc: IntArray = np.zeros(nodes.shape[0], dtype=np.int64)
        face_idx = {f: hex_face_indices(f, order) for f in range(1, 7)}
        for i in range(mesh.boundaries.shape[0]):
            elem = int(mesh.boundaries[i, 0])
            face = int(mesh.boundaries[i, 1])
            grp = g.get(str(mesh.boundary_tags[i]))
            if grp is None:
                _log.warning("unknown boundary name: %s", str(mesh.boundary_tags[i]))
                continue
            bc[conn_ho[elem, face_idx[face]]] = grp.tag
        return nodes, conn_ho[:, perm], _VTK_LAGRANGE_HEXAHEDRON, bc
    elements = mesh.points[mesh.hexes]                   # (N,8,3) per-element coords
    N = elements.shape[0]
    X = elements.reshape(N * 8, 3)
    bc2: IntArray = np.zeros((N, 8), dtype=np.int64)
    for i in range(mesh.boundaries.shape[0]):
        elem = int(mesh.boundaries[i, 0])
        face = int(mesh.boundaries[i, 1])
        grp = g.get(str(mesh.boundary_tags[i]))
        if grp is None:
            _log.warning("unknown boundary name: %s", str(mesh.boundary_tags[i]))
            continue
        bc2[elem, mesh.FACE_POINTS[face - 1]] = grp.tag
    return X, _unwelded(N, 8), _VTK_HEXAHEDRON, bc2.reshape(N * 8)


def _line_arrays(mesh: LineMesh) -> tuple[FloatArray, IntArray, int]:
    """Line nodes: un-welded ``VTK_LINE`` (2 nodes) at ``order == 1``, a conformal
    (shared-node) ``VTK_LAGRANGE_CURVE`` (``order+1`` GLL nodes per cell) above it."""
    if mesh.order == 1:
        blocks = mesh.points[mesh.lines]                 # (L,2,3)
        L, m, _ = blocks.shape
        return blocks.reshape(L * m, 3), _unwelded(L, m), _VTK_LINE
    nodes, conn_ho = conform.conformal_line(
        mesh.points, mesh.lines, mesh.interior, mesh.order)
    perm = _lagrange_curve_perm(mesh.order)
    return nodes, conn_ho[:, perm], _VTK_LAGRANGE_CURVE


def _quad_arrays(mesh: QuadMesh) -> tuple[FloatArray, IntArray, int]:
    """Quad nodes: un-welded ``VTK_QUAD`` (4 CCW nodes) at ``order == 1``, a conformal
    (shared-node) ``VTK_LAGRANGE_QUADRILATERAL`` (``(order+1)**2`` GLL nodes per cell)
    above it."""
    if mesh.order == 1:
        blocks = mesh.points[mesh.quads]                 # (Q,4,3)
        Q, m, _ = blocks.shape
        return blocks.reshape(Q * m, 3), _unwelded(Q, m), _VTK_QUAD
    nodes, conn_ho = conform.conformal_quad(
        mesh.points, mesh.quads, mesh.quad, mesh.flip, mesh.lines.interior,
        mesh.interior, mesh.order)
    perm = _lagrange_quad_perm(mesh.order)
    return nodes, conn_ho[:, perm], _VTK_LAGRANGE_QUADRILATERAL


# -- the unstructured-grid writer ---------------------------------------
def _write_vtu(fname: str, X: FloatArray, conn: IntArray, cell_type: int,
               *, bc_out: IntArray | None = None) -> None:
    """XML VTK unstructured grid (``.vtu``): ``X`` is the ``(P,3)`` point array and
    ``conn`` the ``(N,m)`` per-cell connectivity into it, already in VTK node order
    (consecutive blocks when the nodes are un-welded, shared ids when conformal).
    ``bc_out``, if given, is one value per point, written as ``bc_id`` PointData.  The
    XML container renders VTK Lagrange cells reliably in ParaView / VisIt."""
    P = X.shape[0]
    N, m = conn.shape
    with open(fname, "w") as fid:
        fid.write('<?xml version="1.0"?>\n')
        fid.write('<VTKFile type="UnstructuredGrid" version="1.0" '
                  'byte_order="LittleEndian" header_type="UInt64">\n')
        fid.write("  <UnstructuredGrid>\n")
        fid.write('    <Piece NumberOfPoints="%d" NumberOfCells="%d">\n' % (P, N))
        fid.write("      <Points>\n")
        fid.write('        <DataArray type="Float64" NumberOfComponents="3" '
                  'format="ascii">\n')
        for r in range(P):
            fid.write("          %.17g %.17g %.17g\n" % (X[r, 0], X[r, 1], X[r, 2]))
        fid.write("        </DataArray>\n")
        fid.write("      </Points>\n")
        fid.write("      <Cells>\n")
        fid.write('        <DataArray type="Int64" Name="connectivity" '
                  'format="ascii">\n')
        for e in range(N):
            fid.write("          %s\n" % " ".join(str(int(c)) for c in conn[e]))
        fid.write("        </DataArray>\n")
        fid.write('        <DataArray type="Int64" Name="offsets" format="ascii">\n')
        for e in range(1, N + 1):
            fid.write("          %d\n" % (m * e))
        fid.write("        </DataArray>\n")
        fid.write('        <DataArray type="UInt8" Name="types" format="ascii">\n')
        for _ in range(N):
            fid.write("          %d\n" % cell_type)
        fid.write("        </DataArray>\n")
        fid.write("      </Cells>\n")
        if bc_out is not None:
            fid.write('      <PointData Scalars="bc_id">\n')
            fid.write('        <DataArray type="Int32" Name="bc_id" '
                      'format="ascii">\n')
            for val in bc_out:
                fid.write("          %d\n" % int(val))
            fid.write("        </DataArray>\n")
            fid.write("      </PointData>\n")
        fid.write("    </Piece>\n")
        fid.write("  </UnstructuredGrid>\n")
        fid.write("</VTKFile>\n")


# -- .vtu (XML; VTK Lagrange cells render reliably in ParaView / VisIt) --
def to_vtu(mesh: HexMesh, fname: str, *, groups: GroupsArg = None) -> HexMesh:
    """Write an XML VTK unstructured grid (``.vtu``) of a ``HexMesh`` with per-point
    ``bc_id`` tags.

    At ``order == 1`` each hex is a linear ``VTK_HEXAHEDRON``; at ``order > 1`` a
    ``VTK_LAGRANGE_HEXAHEDRON`` carrying the hex's ``(order+1)**3`` curved GLL nodes."""
    X, conn, cell_type, bc_out = _hex_arrays(mesh, _as_groups(mesh, groups))
    _write_vtu(fname, X, conn, cell_type, bc_out=bc_out)
    return mesh


def line_to_vtu(mesh: LineMesh, fname: str) -> LineMesh:
    """Write an XML VTK unstructured grid (``.vtu``) of a ``LineMesh``, un-welded (one
    node block per line element).  At ``order == 1`` each element is a ``VTK_LINE`` (2
    nodes); at ``order > 1`` a ``VTK_LAGRANGE_CURVE`` carrying the element's
    ``order+1`` GLL nodes -- so a high-order ``circle`` renders as its true arc."""
    X, conn, cell_type = _line_arrays(mesh)
    _write_vtu(fname, X, conn, cell_type)
    return mesh


def quad_to_vtu(mesh: QuadMesh, fname: str) -> QuadMesh:
    """Write an XML VTK unstructured grid (``.vtu``) of a ``QuadMesh``, un-welded (one
    node block per quad).  At ``order == 1`` each quad is a ``VTK_QUAD`` (4 CCW nodes);
    at ``order > 1`` a ``VTK_LAGRANGE_QUADRILATERAL`` carrying the element's
    ``(order+1)**2`` GLL nodes -- so a high-order ``sphere`` renders as its true
    surface."""
    X, conn, cell_type = _quad_arrays(mesh)
    _write_vtu(fname, X, conn, cell_type)
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
