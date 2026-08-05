"""Export / generic-view free functions for a ``HexMesh``.

Exports to a shared-point ``Mesh``, meshio, native Nek ``.re2`` / field file
(``<prefix>0.f00001``), or VTK XML (``.vtu``). Each writer takes the **full** output
filename, extension included. The ``.vtu`` writer emits high-order VTK Lagrange cells
at ``order > 1`` and the Nek field writer emits the full GLL node block; ``.re2``
always stays linear. The ``groups`` parameter maps each
boundary name to a Nek BC code and tag.
"""

from __future__ import annotations

import logging
import struct
from collections.abc import Mapping
from typing import Any, Union

import numpy as np

from .._typing import FloatArray, IntArray, PointArray
from ..hexmesh import HexMesh
from ..hexmesh.query import weld as hex_weld
from ..linemesh import LineMesh
from ..model import conform, topology
from ..model.fields import gll_nodes, lagrange_matrix, uniform_spacing
from ..model.interp import hex_face_indices
from ..model.mesh import Mesh
from ..model.physical import PhysicalGroup, PhysicalGroups
from ..quadmesh import QuadMesh

# VTK cell-type ids: linear + high-order (Lagrange) line / quad / hex.
_VTK_LINE = 3
_VTK_LAGRANGE_CURVE = 68
_VTK_QUAD = 9
_VTK_LAGRANGE_QUADRILATERAL = 70
_VTK_HEXAHEDRON = 12
_VTK_LAGRANGE_HEXAHEDRON = 72

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
        names = mesh.face_group_tags
        return PhysicalGroups(
            PhysicalGroup(name, i + 1) for i, name in enumerate(names))
    return PhysicalGroups(
        PhysicalGroup(name, i + 1, 2, code)
        for i, (name, code) in enumerate(groups.items()))


# -- generic mesh view --------------------------------------------------
def to_mesh(mesh: HexMesh, groups: GroupsArg = None) -> Mesh:
    """Return a shared-point ``Mesh``: welded points, ``hexahedron`` cells, and one
    ``quad`` boundary cell per tagged face grouped into named ``cell_sets``."""
    X, HC, _ = hex_weld(mesh)          # WeldResult unpacks as (points, hexes, n)
    g = _as_groups(mesh, groups)
    conn_rows = []           # welded point ids of each boundary face
    name_rows = []           # name of each boundary face
    for elem, face, name in mesh.face_tags:
        conn_rows.append(HC[elem, mesh.FACE_POINTS[face - 1, :]])
        name_rows.append(name)
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
    """Write the binary Nek ``.re2`` to ``filename`` (the **full** name, extension
    included -- nothing is appended).  The mesh is written **linear** at any order:
    Nek's re2 has no high-order format, so only the 8 corners of each hex are
    emitted."""
    g = _as_groups(mesh, groups)
    elements = mesh.points[mesh.hexes]            # (N,8,3) per-element coords
    face_tags = mesh.face_tags
    num_elem = elements.shape[0]
    with open(filename, "wb") as fid:
        header = "#v004%16d%3d%16d%4d hdr" % (num_elem, 3, num_elem, 1)
        fid.write(header.ljust(80).encode("ascii"))
        fid.write(struct.pack("<f", np.float32(6.54321)))
        for i in range(num_elem):
            fid.write(struct.pack("<d", 0.0))
            fid.write(elements[i, :, 0].astype("<f8").tobytes())
            fid.write(elements[i, :, 1].astype("<f8").tobytes())
            fid.write(elements[i, :, 2].astype("<f8").tobytes())
        fid.write(struct.pack("<d", 0.0))
        fid.write(struct.pack("<d", float(len(face_tags))))
        for elem0, face, name in face_tags:
            buf2: FloatArray = np.zeros(8, dtype="<f8")
            buf2[0] = float(elem0 + 1)
            buf2[1] = float(face)
            grp = g.get(name)
            if grp is not None:
                buf2[7] = _str_to_double(grp.code)
            else:
                _log.warning("unknown boundary name: %s", name)
            fid.write(buf2.tobytes())
    return mesh


_FLD_ETAG = 6.54321        # endian-identification float32, as in ``.re2``


def to_fld(mesh: HexMesh, filename: str, *,
           time: float = 0.0, istep: int = 0, wdsz: int = 8) -> HexMesh:
    """Write the binary Nek5000 field file (``<prefix>0.f00001``) to ``filename``
    (the **full** name -- nothing is appended).

    This is the **high-order** geometry export.  Unlike ``.re2``, which has no
    high-order format and so ships only the 8 corners of each hex, the field format
    stores a full ``lx1*ly1*lz1`` block of **GLL** nodes per element -- exactly what
    this toolkit's B-rep holds at ``order = N`` (``lx1 = N+1``).  The nodes are
    written in Nek's own per-element lexicographic order (``i`` fastest, then ``j``,
    then ``k``), which is the ordering the ``conform.conformal_*`` walk already
    produces, so the block is handed over without a permutation.

    Only ``fields = "X"`` is written: a mesh carries geometry and nothing else, so
    there is no solution data to emit.  The file is little-endian; ``wdsz`` selects
    single (``4``) or double (``8``) precision for the coordinates.  The trailing
    per-element min/max metadata block that Nek writes for 3-D files is emitted too,
    always in single precision.

    Format reference: Nek5000's ``#std`` header + endian tag + ``int32`` element map
    + field data + min/max metadata.
    """
    if wdsz not in (4, 8):
        raise ValueError("wdsz must be 4 (single) or 8 (double), got %r" % (wdsz,))
    order = mesh.order
    nodes, conn_ho = conform.conformal_hex(
        mesh.points, mesh.hexes, mesh._elem_edges, mesh._edge_flip,
        mesh.quads.lines.interior, mesh.hex, mesh.face_orient,
        mesh.quads.interior, mesh.interior, order)
    blocks = nodes[conn_ho]                       # (E, (order+1)**3, 3), i fastest
    nel = blocks.shape[0]
    lx1 = order + 1
    fields = "X"
    header = ("#std %1d %2d %2d %2d %10d %10d %20.13E %9d %6d %6d %s\n"
              % (wdsz, lx1, lx1, lx1, nel, nel, time, istep, 0, 1, fields))
    real = "<f8" if wdsz == 8 else "<f4"
    with open(filename, "wb") as fid:
        fid.write(header.ljust(132).encode("ascii"))
        fid.write(struct.pack("<f", np.float32(_FLD_ETAG)))
        fid.write(np.arange(1, nel + 1, dtype="<i4").tobytes())
        # per element, all x, then all y, then all z
        fid.write(np.ascontiguousarray(blocks.transpose(0, 2, 1)).astype(real).tobytes())
        # 3-D metadata: per element, per component, min then max -- always float32
        minmax: FloatArray = np.stack([blocks.min(axis=1), blocks.max(axis=1)], axis=-1)
        fid.write(minmax.astype("<f4").tobytes())
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
    (bottom / right / top / left), then the interior nodes (``i`` fastest).

    The corners run CCW but **the edges do not**: ``PointIndexFromIJK`` in VTK's
    ``vtkHigherOrderQuadrilateral`` numbers the top edge by ascending ``i`` and the
    left edge by ascending ``j``, i.e. both in the *axis* direction rather than in the
    CCW traversal direction.  Reversing them (which is what a CCW reading suggests) is
    a no-op at ``order == 2`` -- each edge run is a single node -- and only corrupts
    the cell from ``order == 3`` on."""
    n = order
    row = n + 1

    def idx(i: int, j: int) -> int:
        return i + row * j
    corners = [idx(0, 0), idx(n, 0), idx(n, n), idx(0, n)]
    e0 = [idx(i, 0) for i in range(1, n)]
    e1 = [idx(n, j) for j in range(1, n)]
    e2 = [idx(i, n) for i in range(1, n)]
    e3 = [idx(0, j) for j in range(1, n)]
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


def _to_equispaced(nodes: PointArray, conn_ho: IntArray,
                   order: int, dim: int) -> PointArray:
    """Re-place the conformal node array on **equispaced** parameters.

    VTK's Lagrange cells are *defined* on an equispaced node lattice -- there is no
    GLL cell type in VTK (``VTK_BEZIER_*`` takes control points, also not GLL).  The
    toolkit stores GLL nodes, so handing them over verbatim tells the reader the wrong
    parametrization and it reconstructs a different polynomial: measured on a unit cube
    at order 3, VTK renders the *identity* map with a 7.4e-2 excursion, one hump per
    element -- the visible crease at element joints.  At ``order == 2`` the two lattices
    coincide, which is why the artifact only appears from order 3 on.

    This is a change of nodal basis, not a resampling loss: each element's polynomial is
    one object, and it is re-read at ``order+1`` different parameters per axis, so the
    geometry is preserved exactly (to float round-off) and only the *labels* change.

    Shared entities stay consistent because the interpolation is a tensor product and
    the uniform/GLL lattices agree at ``0.0``/``1.0``: a node on a shared edge or face is
    evaluated at the boundary parameter in the transverse directions, which selects that
    entity's own nodes alone.  Both incident elements therefore compute the same value
    from the same data (differing only in float summation order, ~1e-16), so scattering
    into the shared array is well-defined whichever element writes last.
    """
    g = gll_nodes(order)
    u = uniform_spacing(order)
    if np.array_equal(g, u):                 # order <= 2: nothing to relabel
        return nodes
    A: FloatArray = lagrange_matrix(g, u)    # (order+1, order+1) basis change, per axis
    row = order + 1
    # (E, row**dim, 3) lexicographic (``i`` fastest) -> axis-per-direction, slowest first
    blocks = nodes[conn_ho].reshape((conn_ho.shape[0],) + (row,) * dim + (3,))
    for axis in range(1, dim + 1):
        blocks = np.moveaxis(np.tensordot(A, blocks, axes=([1], [axis])), 0, axis)
    out: PointArray = nodes.copy()
    out[conn_ho] = blocks.reshape(conn_ho.shape[0], row ** dim, 3)
    return out


def _hex_arrays(mesh: HexMesh,
                g: PhysicalGroups) -> tuple[PointArray, IntArray, int, IntArray]:
    """Hex nodes + ``bc_id``: linear un-welded ``VTK_HEXAHEDRON`` at ``order == 1``, a
    conformal (shared-node) ``VTK_LAGRANGE_HEXAHEDRON`` (``(order+1)**3`` GLL nodes per
    cell) above it, whose face nodes inherit the boundary face's tag.

    ``bc_id`` precedence is the un-welded writer's rule, applied in the shared-node
    numbering: the boundary rows are scattered in ``mesh.face_tags`` order and the
    **last row to touch a node wins** (this is exactly how two boundary faces sharing an
    edge *within* one hex have always been resolved).  Welding widens the same rule
    across elements: an untagged element never writes, so a node shared by a tagged face
    and an untagged neighbour keeps its tag; where two *differently* tagged faces of
    different elements meet, the single shared node necessarily carries one of the two
    tags -- the later boundary row's."""
    if mesh.order == 1:
        elements = mesh.points[mesh.hexes]               # (N,8,3) per-element coords
        N = elements.shape[0]
        X = elements.reshape(N * 8, 3)
        bc1: IntArray = np.zeros((N, 8), dtype=np.int64)
        for elem, face, name in mesh.face_tags:
            grp = g.get(name)
            if grp is None:
                _log.warning("unknown boundary name: %s", name)
                continue
            bc1[elem, mesh.FACE_POINTS[face - 1]] = grp.tag
        return X, _unwelded(N, 8), _VTK_HEXAHEDRON, bc1.reshape(N * 8)
    order = mesh.order
    perm = _lagrange_hex_perm(order)
    nodes, conn_ho = conform.conformal_hex(
        mesh.points, mesh.hexes, mesh._elem_edges, mesh._edge_flip,
        mesh.quads.lines.interior, mesh.hex, mesh.face_orient,
        mesh.quads.interior, mesh.interior, order)
    bc: IntArray = np.zeros(nodes.shape[0], dtype=np.int64)
    face_idx = {f: hex_face_indices(f, order) for f in range(1, 7)}
    for elem, face, name in mesh.face_tags:
        grp = g.get(name)
        if grp is None:
            _log.warning("unknown boundary name: %s", name)
            continue
        bc[conn_ho[elem, face_idx[face]]] = grp.tag
    nodes = _to_equispaced(nodes, conn_ho, order, 3)
    return nodes, conn_ho[:, perm], _VTK_LAGRANGE_HEXAHEDRON, bc


def _line_arrays(mesh: LineMesh) -> tuple[PointArray, IntArray, int]:
    """Line nodes: un-welded ``VTK_LINE`` (2 nodes) at ``order == 1``, a conformal
    (shared-node) ``VTK_LAGRANGE_CURVE`` (``order+1`` GLL nodes per cell) above it."""
    if mesh.order == 1:
        blocks = mesh.points[mesh.lines]                 # (L,2,3)
        L, m, _ = blocks.shape
        return blocks.reshape(L * m, 3), _unwelded(L, m), _VTK_LINE
    nodes, conn_ho = conform.conformal_line(
        mesh.points, mesh.lines, mesh.interior, mesh.order)
    perm = _lagrange_curve_perm(mesh.order)
    nodes = _to_equispaced(nodes, conn_ho, mesh.order, 1)
    return nodes, conn_ho[:, perm], _VTK_LAGRANGE_CURVE


def _quad_arrays(mesh: QuadMesh) -> tuple[PointArray, IntArray, int]:
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
    nodes = _to_equispaced(nodes, conn_ho, mesh.order, 2)
    return nodes, conn_ho[:, perm], _VTK_LAGRANGE_QUADRILATERAL


# -- the unstructured-grid writer ---------------------------------------
def _write_vtu(fname: str, X: PointArray, conn: IntArray, cell_type: int,
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
        # one formatted block per DataArray rather than a write() per row: the row
        # loops were ~17M write calls on a 490k-cell mesh.  ``tolist()`` converts to
        # Python scalars in C, so the remaining per-element cost is only the format.
        fid.write("".join("          %.17g %.17g %.17g\n" % (x, y, z)
                          for x, y, z in X.tolist()))
        fid.write("        </DataArray>\n")
        fid.write("      </Points>\n")
        fid.write("      <Cells>\n")
        fid.write('        <DataArray type="Int64" Name="connectivity" '
                  'format="ascii">\n')
        fid.write("".join("          %s\n" % " ".join(map(str, row))
                          for row in conn.tolist()))
        fid.write("        </DataArray>\n")
        fid.write('        <DataArray type="Int64" Name="offsets" format="ascii">\n')
        fid.write("".join("          %d\n" % off for off in range(m, m * N + 1, m)))
        fid.write("        </DataArray>\n")
        fid.write('        <DataArray type="UInt8" Name="types" format="ascii">\n')
        fid.write(("          %d\n" % cell_type) * N)
        fid.write("        </DataArray>\n")
        fid.write("      </Cells>\n")
        if bc_out is not None:
            fid.write('      <PointData Scalars="bc_id">\n')
            fid.write('        <DataArray type="Int32" Name="bc_id" '
                      'format="ascii">\n')
            fid.write("".join("          %d\n" % v for v in bc_out.tolist()))
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
              mesh.hexes.shape[0], len(mesh.face_tags))
    for name in mesh.face_group_tags:
        _log.info("  %-14s: %d faces", name, mesh.face_tags.count(name))
    w = hex_weld(mesh)
    rep = topology.hex_report(w.points, w.hexes)
    _log.info("  watertight=%s  conformal=%s  components=%d  "
              "non-manifold faces=%d  hanging points=%d",
              rep.watertight, rep.conformal, rep.n_components,
              rep.n_nonmanifold_faces, rep.n_hanging_points)
