"""Export / generic-view free functions for a ``HexMesh``."""

from __future__ import annotations

import base64
import json
import logging
import struct
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Union

import numpy as np

from .._typing import BoolArray, FloatArray, IntArray, PointArray
from ..core import conform
from ..core.boundary_condition import BCSpec, BoundaryConditions
from ..core.fields import gll_nodes, lagrange_matrix, uniform_spacing
from ..core.interp import hex_face_indices
from ..core.mesh import Mesh
from ..core.selection import Selection, mask_for_selection
from ..hexmesh import HexMesh
from ..hexmesh.lower import boundary_mesh
from ..hexmesh.periodic import Periodic, PeriodicPairs, periodic_pairs
from ..hexmesh.query import face_tag_rows
from ..linemesh import LineMesh
from ..quadmesh import QuadMesh
from ..quadmesh.assemble import select as quadmesh_select
from ..tags import Tags

# VTK cell-type ids: linear + high-order (Lagrange) line / quad / hex.
_VTK_LINE = 3
_VTK_LAGRANGE_CURVE = 68
_VTK_QUAD = 9
_VTK_LAGRANGE_QUADRILATERAL = 70
_VTK_HEXAHEDRON = 12
_VTK_LAGRANGE_HEXAHEDRON = 72

_log = logging.getLogger("nekmeshpy")

#: What ``bc=`` accepts: a ready ``BoundaryConditions``, a ``{name: code}`` table (each
#: value a ``BCSpec``) standing for its
#: velocity field alone, or ``None``.
BCArg = Union[BoundaryConditions, Mapping[str, BCSpec], None]
#: What ``to_re2``'s ``periodic=`` accepts: the specs, or a pairing already
#: resolved by :func:`hexmesh.periodic_pairs
#: <nekmeshpy.hexmesh.periodic.periodic_pairs>`.
PeriodicArg = Union[Sequence[Periodic], PeriodicPairs, None]
#: What ``to_re2``'s ``fluid=`` accepts: region name(s), a ready boolean mask, an array
#: of element ids, or ``None`` -- see :func:`core.selection.mask_for_selection
#: <nekmeshpy.core.selection.mask_for_selection>`, which resolves it.
FluidArg = Union[Selection, None]


def _export_rows(mesh: HexMesh, bc: BoundaryConditions,
                 partners: Mapping[tuple[int, int], tuple[int, int]] | None = None,
                 *, viewer: bool = False) -> list[tuple[int, int, str, str, int, int, int]]:
    """``(element, face, name, cbc, boundaryID, partner element, partner face)`` for every boundary
    row this mesh exports -- the one definition every writer here shares, so a face the
    ``.re2`` omits is not in the ``.vtu``'s cell sets either.

    This is where an asymmetric condition is resolved. A named face reconstructs to one
    row per hex carrying it, and each row's code is read against the **region** of the
    hex that owns it, so the two sides of a conjugate interface can differ even though
    the face has a single name. A region whose code is ``None`` contributes no row,
    which is how a face gets a condition from one side only.

    Over several fields a row exists if any field writes one, and carries the first
    such side (velocity before temperature) -- a view for the ``.vtu``; a ``.re2`` block
    is one field's own :func:`_field_rows`.

    ``viewer`` gives a table that states no ``boundaryID`` each name's position in that
    field's table, so a ``.vtu`` still tells boundaries apart.

    ``partners`` is :meth:`PeriodicPairs.partner_of
    <nekmeshpy.hexmesh.periodic.PeriodicPairs.partner_of>`; a row it does not name gets
    ``(-1, -1)``, which every writer but ``.re2`` drops on the floor."""
    rows, names = face_tag_rows(mesh)
    regions = mesh.element_tags.take_dense(rows[:, 0])
    out: list[tuple[int, int, str, str, int, int, int]] = []
    for (elem, face), name, region in zip(rows.tolist(), names.tolist(),
                                          regions.tolist()):
        pe, pf = (partners or {}).get((int(elem), int(face)), (-1, -1))
        if name not in bc:
            _log.warning("unknown boundary name: %s", name)
            out.append((int(elem), int(face), name, "   ", 0, pe, pf))
            continue
        side = next(((f, x) for f in bc.fields if bc.has(f, name)
                     if (x := bc.side(f, name, region)) is not None), None)
        if side is None:
            continue
        field, (cbc, bid) = side
        if viewer and not bc.numbered(field):
            bid = bc.tag_of(field, name)
        out.append((int(elem), int(face), name, cbc, bid, pe, pf))
    return out


def _as_bc(mesh: HexMesh, bc: BCArg) -> BoundaryConditions:
    """Normalise the ``bc`` argument to a ``BoundaryConditions``.

    A plain mapping is a velocity table. ``None`` enumerates the mesh's own tag
    vocabulary. That is enough for a *viewer* -- ``.vtu`` paints ``bc_id`` by id, and an
    id carries no physics -- but not for ``.re2``, which writes Nek BC **codes**, so
    :func:`to_re2` requires the mapping rather than inventing one."""
    if isinstance(bc, BoundaryConditions):
        return bc
    if bc is None:
        return BoundaryConditions({name: "E  " for name in mesh.face_group_tags})
    return BoundaryConditions(bc)


# -- generic mesh view --------------------------------------------------
def to_mesh(mesh: HexMesh, bc: BCArg = None) -> Mesh:
    """Return a shared-point ``Mesh``: welded points, ``hexahedron`` cells, and one
    ``quad`` boundary cell per tagged face grouped into named ``cell_sets``."""
    X, HC = mesh.points, mesh.corners
    g = _as_bc(mesh, bc)
    conn_rows = []           # welded point ids of each boundary face
    name_rows = []           # name of each boundary face
    first_id: dict[str, int] = {}   # boundaryID of each name's first row
    for elem, face, name, _cbc, bid, _pe, _pf in _export_rows(mesh, g, viewer=True):
        conn_rows.append(HC[elem, mesh.FACE_POINTS[face - 1, :]])
        name_rows.append(name)
        first_id.setdefault(name, bid)
    quad_conn = (np.array(conn_rows, dtype=np.int64) if conn_rows
                 else np.zeros((0, 4), np.int64))
    quad_name = np.array(name_rows, dtype=np.str_)

    cells = {"hexahedron": HC}
    if quad_conn.shape[0]:
        cells["quad"] = quad_conn

    cell_sets: dict[str, dict[str, IntArray]] = {}
    point_sets: dict[str, IntArray] = {}
    field_data: dict[str, IntArray] = {}
    for name in g:
        sel = np.flatnonzero(quad_name == name)
        if sel.size == 0:
            continue
        cell_sets[name] = {"quad": sel}
        point_sets[name] = np.unique(quad_conn[sel].ravel())
        field_data[name] = np.array([first_id[name], 2], dtype=np.int64)

    return Mesh(points=X, cells=cells, point_sets=point_sets,
                cell_sets=cell_sets, field_data=field_data)


def to_meshio(mesh: HexMesh, bc: BCArg = None) -> Any:
    """Return a meshio mesh view (requires ``meshio``)."""
    return to_mesh(mesh, bc).to_meshio()


def write(mesh: HexMesh, path: str, file_format: str | None = None,
          *, bc: BCArg = None) -> str:
    """Write through meshio to any supported format; for native Nek use ``to_re2``."""
    return to_mesh(mesh, bc).write(path, file_format=file_format)


# -- native Nek export --------------------------------------------------
def _str_to_double(s: str) -> float:
    b = bytearray(8)
    for i, ch in enumerate(s):
        b[i] = ord(ch)
    return struct.unpack("<d", bytes(b))[0]


#: The Nek BC code for a periodic face.  Unlike every other code, it is not a statement
#: the ``bc`` table can make on its own -- a ``'P'`` row also has to say *which*
#: element and face it is periodic with, which is what ``periodic=`` supplies.
PERIODIC_CODE = "P  "


def _fluid_first_order(mesh: HexMesh, fluid: FluidArg
                       ) -> tuple[IntArray, int, BoolArray | None]:
    """The **stable** permutation ``to_re2``'s ``fluid=`` writes elements under, the
    velocity-mesh element count Nek's ``.re2`` header calls ``nelgv``, and the fluid
    mask itself -- in the mesh's **own** numbering, which every later per-field check
    reads against rather than the written (fluid-first) one, since a mesh is free to
    store its solid block first.

    Nek5000 distinguishes a **velocity mesh** (the first ``nelgv`` elements) from the
    **total** mesh (``nelgt``): a conjugate run solves momentum only on the velocity
    mesh, so those elements must be listed first, contiguously -- not merely counted.
    ``fluid=None`` puts every element in it, unpermuted, which is today's behaviour and
    the right one for every single-domain mesh (``nelgv == nelgt``, no reorder) --
    and the ``None`` mask that comes back with it means exactly that: no region
    restricts any field, because there is only one."""
    n = mesh.n_hexes
    if fluid is None:
        return np.arange(n, dtype=np.int64), n, None
    mask: BoolArray = mask_for_selection(fluid, mesh.element_tags)
    order: IntArray = np.concatenate(
        [np.flatnonzero(mask), np.flatnonzero(~mask)]).astype(np.int64)
    return order, int(np.count_nonzero(mask)), mask


def _resolve_periodic(mesh: HexMesh, periodic: PeriodicArg
                      ) -> tuple[dict[tuple[int, int], tuple[int, int]], PeriodicPairs]:
    """The ``(element, face) -> partner`` lookup every field's boundary block reads
    ``bc(1)``/``bc(2)`` from, resolved **once**: the correspondence is a geometric fact
    about the mesh, not something that varies by which field later writes a row for it."""
    pairs = (periodic if isinstance(periodic, PeriodicPairs)
             else periodic_pairs(mesh, periodic or ()))
    if pairs.rows.size:
        _log.info("re2: %d periodic faces, worst pairing residual %.3e (tol %.3e)",
                  pairs.rows.shape[0], pairs.worst, pairs.tol)
    return pairs.partner_of(), pairs


def _check_periodic_names(mesh: HexMesh, bc: BoundaryConditions, field: str,
                          partners: Mapping[tuple[int, int], tuple[int, int]],
                          mask: BoolArray | None) -> None:
    """Raise unless a name coded ``'P  '`` in ``field``'s table and a periodic-paired name whose
    element lies in ``mask`` (the mesh's own numbering; ``None`` means every element)
    are the same set.

    ``mask`` is what makes this check per-**field** rather than per-mesh: a pair living
    entirely on the solid side of a conjugate mesh is invisible to the velocity field's
    own check (``mask`` = the fluid region) and expected there, but must still be coded
    ``'P  '`` in whichever field's table does cover it. A ``'P  '`` with no matching
    pair would write partner element 0, face 0, and a pair with no ``'P  '`` would
    export those faces as something else entirely -- neither shows up until the solver
    runs, so both are refused here."""
    coded = bc.names_with(field, PERIODIC_CODE)
    hexes: IntArray = np.asarray(mesh.hexes, dtype=np.int64)
    pairs = np.array([(elem, face) for elem, face in partners
                      if mask is None or mask[elem]], dtype=np.int64).reshape(-1, 2)
    paired = set(mesh.face_tags.take_dense(hexes[pairs[:, 0], pairs[:, 1] - 1]).tolist())
    if coded != paired:
        raise ValueError(
            "%s: %r is the periodic code, so a name carrying it and a name named by "
            "periodic= must be the same set. This table codes %s as periodic; "
            "periodic= pairs %s in this field's own region. A 'P' row with no pairing "
            "writes partner element 0, face 0, and a pairing with no 'P' exports as "
            "something else -- neither is visible until the solver reads the mesh."
            % (field, PERIODIC_CODE,
               ", ".join(repr(n) for n in sorted(coded)) or "nothing",
               ", ".join(repr(n) for n in sorted(paired)) or "nothing"))


def _field_rows(mesh: HexMesh, bc: BoundaryConditions, field: str,
                partners: Mapping[tuple[int, int], tuple[int, int]],
                mask: BoolArray | None) -> list[tuple[int, int, str, str, int, int, int]]:
    """One field's boundary rows: ``(element, face, name, code, partner element,
    partner face)``, restricted to elements in ``mask`` -- Nek's velocity field reads
    only the fluid region, a thermal one every element (``mask=None``).

    Unlike :func:`_export_rows`, a name **absent** from ``g`` produces no row and no
    warning here: once a mesh writes more than one field, a partial vocabulary is the
    point (a fluid-only velocity table naming nothing solid is correct, not a typo).

    A row on an element outside ``mask`` is not this field's to write, and is left out
    without being asked: a conjugate interface is one named face with a hex of each
    region on its sides, and the velocity block simply does not see the solid one. A
    name that is stated but has **no** row inside ``mask`` at all is unambiguously a
    mistake -- the wrong field's table -- and raises rather than writing nothing."""
    rows, names = face_tag_rows(mesh)
    regions = mesh.element_tags.take_dense(rows[:, 0])
    out: list[tuple[int, int, str, str, int, int, int]] = []
    inside: set[str] = set()
    outside: dict[str, int] = {}
    for (elem, face), name, region in zip(rows.tolist(), names.tolist(),
                                          regions.tolist()):
        if not bc.has(field, name):
            continue
        if mask is not None and not mask[elem]:
            outside.setdefault(name, elem)
            continue
        inside.add(name)
        side = bc.side(field, name, region)
        if side is None:
            continue
        pe, pf = partners.get((elem, face), (-1, -1))
        out.append((elem, face, name, side[0], side[1], pe, pf))
    for name, elem in outside.items():
        if name not in inside:
            raise ValueError(
                "%s: %r names element %d, outside this field's own region -- a "
                "velocity-field code on a non-fluid element, or vice versa."
                % (field, name, elem))
    return out


def to_re2(mesh: HexMesh, filename: str, *, bc: BCArg,
           periodic: PeriodicArg = None, fluid: FluidArg = None) -> HexMesh:
    """Write the binary Nek ``.re2`` to ``filename`` (the **full** name, extension
    included -- nothing is appended). The mesh is written **linear** at any order: Nek's
    re2 has no high-order format, so only the 8 corners of each hex are emitted.

    ``bc`` is **required**: this is the one writer that emits Nek BC codes, and a
    default would put a code the caller never chose in front of the solver. It is where
    a mesher states what its named surfaces physically *are*, so it belongs in the
    mesher, visible next to the tags it names. A :class:`BoundaryConditions
    <nekmeshpy.core.boundary_condition.BoundaryConditions>` carries one table per field;
    a plain ``{name: code}`` mapping is its velocity table alone.

    ``periodic`` is a sequence of :class:`Periodic
    <nekmeshpy.hexmesh.periodic.Periodic>` specs (or a ready :class:`PeriodicPairs
    <nekmeshpy.hexmesh.periodic.PeriodicPairs>`), and is the only thing that fills the
    boundary record's ``bc(1)`` / ``bc(2)`` fields -- the partner element and face a
    ``'P  '`` row is meaningless without.  It is stated separately from ``bc``
    because it is not a code but a *correspondence*, and the two must agree: every name
    coded ``'P  '`` is named by a spec and vice versa, or this raises::

        bc={"wall": "W  ", "inlet": "P  ", "outlet": "P  "},
        periodic=[hexmesh.Periodic("inlet", "outlet",
                                   affine.translation([0, 0, LENGTH]))]

    ``fluid`` names the region(s) that are the **velocity mesh** -- anything else
    (a different region, or untagged) is conduction-only.  Nek's header calls these
    counts ``nelgv`` (velocity) and ``nelgt`` (total); a conjugate run needs
    ``nelgv < nelgt`` *and* the velocity elements listed first, contiguously, which is
    a statement about **order**, not just a count -- so ``fluid=`` reorders the written
    bytes rather than taking ``nelv=`` on trust.  The mesh object this returns is
    unchanged; only the file's element numbering (and every row that names an element)
    is permuted.  ``fluid=None`` -- the default -- writes every element as velocity-mesh
    (``nelgv == nelgt``), which is today's behaviour and the right one for a
    single-domain mesh.

    Whenever ``fluid=`` carves out an actual solid region (``nelgv < nelgt``), Nek's
    ``.re2`` always carries **one boundary block per solved field** -- ``read_re2_data``
    reads at least two the moment ``nelgt > nelgv``, regardless of what the header
    declares, so a file with only one is truncated from the reader's point of view and
    ``ierr``s reading the (missing) second block. ``bc.temperature`` is that second field's own
    name -> code table, same shape as ``bc.velocity``, and is **required** exactly when
    ``fluid=`` makes the two counts differ. It may also be given on a mesh that is all
    fluid, where temperature is transported in the flow: the file then carries both
    blocks, each over every element. It covers **every** element, not just the
    fluid ones -- a name omitted from it is left conformal (``'E  '``, ordinary
    continuity), which is normally right for a genuinely conjugate interface: velocity
    needs an explicit wall there because the solid side carries no velocity unknowns to
    continue into, but temperature is solved on both sides, so nothing needs stating::

        writer.to_re2(mesh, "wire_coil.re2", periodic=PERIODIC, fluid="fluid",
                      bc=BoundaryConditions(velocity=VEL_BC, temperature=TEMP_BC))
    """
    if bc is None:
        raise ValueError(
            "to_re2 needs bc=: a name -> Nek BC code mapping for %s. The .re2 "
            "boundary block is boundary *conditions*, so there is no default that "
            "would not be a guess -- spell the mapping out where the tags are named, "
            'e.g. bc={"wall": "W  ", "inlet": "v  ", "outlet": "O  "}.'
            % (", ".join(repr(n) for n in mesh.face_group_tags) or "no named faces"))
    bc = _as_bc(mesh, bc)
    order, nelv, fluid_mask = _fluid_first_order(mesh, fluid)
    n_hexes = mesh.n_hexes
    multi_field = nelv < n_hexes
    has_temperature = "temperature" in bc.fields
    if multi_field and not has_temperature:
        raise ValueError(
            "to_re2: fluid=%r covers %d of this mesh's %d elements. Nek's .re2 format "
            "always carries a boundary block per field once nelgv < nelgt -- so "
            "bc needs a temperature table too: a name -> code mapping for the "
            "temperature field, covering every element (a name left out of it stays "
            "conformal, 'E  '). A file written with only the velocity block is one the "
            "reader expects to keep reading and cannot." % (fluid, nelv, n_hexes))
    partners, _pairs = _resolve_periodic(mesh, periodic)
    if not has_temperature:
        _check_periodic_names(mesh, bc, "velocity", partners, None)
        blocks = [_export_rows(mesh, bc, partners)]
    else:
        _check_periodic_names(mesh, bc, "velocity", partners, fluid_mask)
        _check_periodic_names(mesh, bc, "temperature", partners, None)
        blocks = [_field_rows(mesh, bc, "velocity", partners, fluid_mask),
                  _field_rows(mesh, bc, "temperature", partners, None)]

    # ``new_id_of[old]`` is where element ``old`` lands in the written, fluid-first
    # numbering -- the inverse of ``order``, and what every element id in a boundary
    # row (its own, and a periodic partner's) has to be read through before writing.
    new_id_of: IntArray = np.empty(n_hexes, dtype=np.int64)
    new_id_of[order] = np.arange(n_hexes, dtype=np.int64)
    elements = mesh.points[mesh.corners][order]      # (N,8,3), fluid-first
    with open(filename, "wb") as fid:
        # nBCre2 -- the number of boundary blocks that follow, one per field.  Nek's
        # own field count already falls back to 2 whenever nelgt>nelgv regardless of
        # this value, but a reader that trusts it outright (``re2torea``) deserves the
        # true count, not a name that happens not to matter to this particular check.
        header = "#v004%16d%3d%16d%4d hdr" % (elements.shape[0], 3, nelv, len(blocks))
        fid.write(header.ljust(80).encode("ascii"))
        fid.write(struct.pack("<f", np.float32(6.54321)))
        for i in range(elements.shape[0]):
            fid.write(struct.pack("<d", 0.0))
            fid.write(elements[i, :, 0].astype("<f8").tobytes())
            fid.write(elements[i, :, 1].astype("<f8").tobytes())
            fid.write(elements[i, :, 2].astype("<f8").tobytes())
        fid.write(struct.pack("<d", 0.0))
        for bnd in blocks:
            fid.write(struct.pack("<d", float(len(bnd))))
            for elem0, face, _name, code, bid, p_elem0, p_face in bnd:
                buf2: FloatArray = np.zeros(8, dtype="<f8")
                buf2[0] = float(new_id_of[elem0] + 1)
                buf2[1] = float(face)
                if p_elem0 >= 0:
                    # bc(1), bc(2): the element and face on the other side of the
                    # periodic boundary.  Element ids are 1-based on write, faces
                    # already 1-6, and read through the same fluid-first permutation
                    # as buf2[0].
                    buf2[2] = float(new_id_of[p_elem0] + 1)
                    buf2[3] = float(p_face)
                buf2[6] = float(bid)                    # bc(5): Nek's boundaryID
                buf2[7] = _str_to_double(code)
                fid.write(buf2.tobytes())
    return mesh


_FLD_ETAG = 6.54321        # endian-identification float32, as in ``.re2``


def to_fld(mesh: HexMesh, filename: str, *,
           time: float = 0.0, istep: int = 0, wdsz: int = 8,
           fluid: FluidArg = None) -> HexMesh:
    """Write the binary Nek5000 field file (``<prefix>0.f00001``) to ``filename`` (the
    **full** name -- nothing is appended).

    ``fluid`` must name the same region as the matching :func:`to_re2` call's own
    ``fluid=``, and for the same reason: Nek reads a restart/IC field file element by
    element against the ``.re2`` it already loaded, so the two files' element numbering
    has to agree.  ``to_re2``'s ``fluid=`` reorders its written bytes fluid-first
    whenever it carves out a solid region (``nelgv < nelgt``); a ``to_fld`` written in
    the mesh's own, unpermuted order would then describe element *i* of one file and
    element *i* of the other as two different pieces of geometry -- not merely stale,
    but silently misassigned, since both files are the same length and neither one
    knows it disagrees with the other. ``fluid=None`` writes every element unpermuted,
    which agrees with an unpermuted ``to_re2`` (``fluid=None`` there too)."""
    if wdsz not in (4, 8):
        raise ValueError("wdsz must be 4 (single) or 8 (double), got %r" % (wdsz,))
    fluid_order, _nelv, _ = _fluid_first_order(mesh, fluid)
    order = mesh.order
    nodes, conn_ho = conform.conformal_hex(
        mesh.points, mesh.corners, mesh._elem_edges, mesh._edge_flip,
        mesh.quad_mesh.line_mesh.interior, mesh.hexes, mesh.orient,
        mesh.quad_mesh.interior, mesh.interior, order)
    blocks = nodes[conn_ho][fluid_order]           # (E, (order+1)**3, 3), fluid-first
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
    the VTK Lagrange-quadrilateral order: 4 corners, then the four edges (bottom / right
    / top / left), then the interior nodes (``i`` fastest)."""
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
# ``order > 1`` the conformal walk (:mod:`nekmeshpy.core.conform`) emits **shared**
# nodes: a node on an edge / face between two elements is written once.
def element_tag_ids(tags: Tags, n_elements: int) -> tuple[IntArray, list[str]]:
    """``(per-element id (n_elements,), name per id)`` for a region table: 1-based
    positions in the **sorted** tag vocabulary, ``0`` where an element carries no tag.

    A ``.vtu`` cannot hold the names themselves -- VTK's string arrays are per-file
    field data, not something a cell scalar can point into -- so the array written is
    integer, exactly as ``bc_id`` already is.  What keeps it readable is that the
    mapping is a function of the mesh alone: ``sorted(mesh.element_tags.unique())``
    reproduces it anywhere, with no legend to carry alongside the file."""
    names, inverse = tags.unique(return_inverse=True)
    ids: IntArray = np.zeros(n_elements, dtype=np.int64)
    ids[tags.ids] = inverse + 1
    return ids, names


def _unwelded(n_elem: int, m: int) -> IntArray:
    """Consecutive-block connectivity ``(n_elem, m)`` for un-welded node arrays."""
    return np.arange(n_elem * m, dtype=np.int64).reshape(n_elem, m)


def _to_equispaced(nodes: PointArray, conn_ho: IntArray,
                   order: int, dim: int) -> PointArray:
    """Re-place the conformal node array on **equispaced** parameters."""
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
                g: BoundaryConditions) -> tuple[PointArray, IntArray, int, IntArray]:
    """Hex nodes + ``bc_id``: linear un-welded ``VTK_HEXAHEDRON`` at ``order == 1``, a
    conformal (shared-node) ``VTK_LAGRANGE_HEXAHEDRON`` (``(order+1)**3`` GLL nodes per
    cell) above it, whose face nodes inherit the boundary face's tag."""
    if mesh.order == 1:
        elements = mesh.points[mesh.corners]               # (N,8,3) per-element coords
        N = elements.shape[0]
        X = elements.reshape(N * 8, 3)
        bc1: IntArray = np.zeros((N, 8), dtype=np.int64)
        for elem, face, _name, _cbc, bid, _pe, _pf in _export_rows(mesh, g, viewer=True):
            bc1[elem, mesh.FACE_POINTS[face - 1]] = bid
        return X, _unwelded(N, 8), _VTK_HEXAHEDRON, bc1.reshape(N * 8)
    order = mesh.order
    perm = _lagrange_hex_perm(order)
    nodes, conn_ho = conform.conformal_hex(
        mesh.points, mesh.corners, mesh._elem_edges, mesh._edge_flip,
        mesh.quad_mesh.line_mesh.interior, mesh.hexes, mesh.orient,
        mesh.quad_mesh.interior, mesh.interior, order)
    bc: IntArray = np.zeros(nodes.shape[0], dtype=np.int64)
    face_idx = {f: hex_face_indices(f, order) for f in range(1, 7)}
    for elem, face, _name, _cbc, bid, _pe, _pf in _export_rows(mesh, g, viewer=True):
        bc[conn_ho[elem, face_idx[face]]] = bid
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
        blocks = mesh.points[mesh.corners]                 # (Q,4,3)
        Q, m, _ = blocks.shape
        return blocks.reshape(Q * m, 3), _unwelded(Q, m), _VTK_QUAD
    nodes, conn_ho = conform.conformal_quad(
        mesh.points, mesh.corners, mesh.quads, mesh.orient, mesh.line_mesh.interior,
        mesh.interior, mesh.order)
    perm = _lagrange_quad_perm(mesh.order)
    nodes = _to_equispaced(nodes, conn_ho, mesh.order, 2)
    return nodes, conn_ho[:, perm], _VTK_LAGRANGE_QUADRILATERAL


# -- the unstructured-grid writer ---------------------------------------
def _cell_tags(tags: Tags, n_elements: int) -> IntArray | None:
    """The ``element_tag`` cell array, or ``None`` for an untagged mesh -- which writes
    no ``CellData`` at all rather than a meaningless column of zeros."""
    if not tags:
        return None
    return element_tag_ids(tags, n_elements)[0]


def _b64(a: IntArray | PointArray) -> str:
    """One ``DataArray``'s payload for ``format="binary"``: the byte count followed by
    the bytes, base64-encoded **as a single blob**.

    That framing is why the file declares ``header_type``.  Encoding the header and the
    data separately is also seen in the wild, but it leaves base64 padding mid-string,
    which strict decoders reject -- so this writes the one form every reader takes."""
    raw = np.ascontiguousarray(a).tobytes()
    head = np.array([len(raw)], dtype="<u8").tobytes()
    return base64.b64encode(head + raw).decode("ascii")


def _write_vtu(fname: str, X: PointArray, conn: IntArray, cell_type: int,
               *, bc_out: IntArray | None = None, cell_out: IntArray | None = None,
               binary: bool = True) -> None:
    """XML VTK unstructured grid (``.vtu``): ``X`` is the ``(P,3)`` point array and
    ``conn`` the ``(N,m)`` per-cell connectivity into it, already in VTK node order
    (consecutive blocks when the nodes are un-welded, shared ids when conformal).

    ``binary`` writes each array as inline base64 rather than formatted text.  Ascii
    costs a Python-level format per row -- there is no vectorized float formatter -- so
    on a multi-million-node mesh the encoding, not the I/O, is the whole cost."""
    P = X.shape[0]
    N, m = conn.shape
    offsets: IntArray = np.arange(m, m * N + 1, m, dtype=np.int64)
    types: IntArray = np.full(N, cell_type, dtype=np.uint8)
    fmt = "binary" if binary else "ascii"

    def block(a: IntArray | PointArray, rows: Callable[[], str]) -> str:
        """A ``DataArray``'s body: one base64 payload, or the ascii rows -- which are
        built only if that is what is being written, hence the thunk."""
        return "          %s\n" % _b64(a) if binary else rows()

    with open(fname, "w") as fid:
        fid.write('<?xml version="1.0"?>\n')
        fid.write('<VTKFile type="UnstructuredGrid" version="1.0" '
                  'byte_order="LittleEndian" header_type="UInt64">\n')
        fid.write("  <UnstructuredGrid>\n")
        fid.write('    <Piece NumberOfPoints="%d" NumberOfCells="%d">\n' % (P, N))
        fid.write("      <Points>\n")
        fid.write('        <DataArray type="Float64" NumberOfComponents="3" '
                  'format="%s">\n' % fmt)
        # ascii: one formatted block per DataArray rather than a write() per row -- the
        # row loops were ~17M write calls on a 490k-cell mesh.  ``tolist()`` converts to
        # Python scalars in C, so the remaining per-element cost is only the format.
        fid.write(block(np.asarray(X, dtype=np.float64),
                        lambda: "".join("          %.17g %.17g %.17g\n" % (x, y, z)
                                        for x, y, z in X.tolist())))
        fid.write("        </DataArray>\n")
        fid.write("      </Points>\n")
        fid.write("      <Cells>\n")
        fid.write('        <DataArray type="Int64" Name="connectivity" '
                  'format="%s">\n' % fmt)
        fid.write(block(np.asarray(conn, dtype=np.int64),
                        lambda: "".join("          %s\n" % " ".join(map(str, row))
                                        for row in conn.tolist())))
        fid.write("        </DataArray>\n")
        fid.write('        <DataArray type="Int64" Name="offsets" format="%s">\n' % fmt)
        fid.write(block(offsets,
                        lambda: "".join("          %d\n" % o for o in offsets.tolist())))
        fid.write("        </DataArray>\n")
        fid.write('        <DataArray type="UInt8" Name="types" format="%s">\n' % fmt)
        fid.write(block(types, lambda: ("          %d\n" % cell_type) * N))
        fid.write("        </DataArray>\n")
        fid.write("      </Cells>\n")
        if bc_out is not None:
            fid.write('      <PointData Scalars="bc_id">\n')
            fid.write('        <DataArray type="Int32" Name="bc_id" '
                      'format="%s">\n' % fmt)
            fid.write(block(np.asarray(bc_out, dtype=np.int32),
                            lambda: "".join("          %d\n" % v
                                            for v in bc_out.tolist())))
            fid.write("        </DataArray>\n")
            fid.write("      </PointData>\n")
        if cell_out is not None:
            # per-**cell**, unlike bc_id: a region belongs to the element, and painting
            # it on nodes would make every interface node ambiguous between the two
            # regions that share it -- which is the whole point of a conjugate mesh.
            fid.write('      <CellData Scalars="element_tag">\n')
            fid.write('        <DataArray type="Int32" Name="element_tag" '
                      'format="%s">\n' % fmt)
            fid.write(block(np.asarray(cell_out, dtype=np.int32),
                            lambda: "".join("          %d\n" % v
                                            for v in cell_out.tolist())))
            fid.write("        </DataArray>\n")
            fid.write("      </CellData>\n")
        fid.write("    </Piece>\n")
        fid.write("  </UnstructuredGrid>\n")
        fid.write("</VTKFile>\n")


# -- .vtu (XML; VTK Lagrange cells render reliably in ParaView / VisIt) --
def to_vtu(mesh: HexMesh, fname: str, *, bc: BCArg = None,
           binary: bool = True) -> HexMesh:
    """Write an XML VTK unstructured grid (``.vtu``) of a ``HexMesh`` with per-point
    ``bc_id`` tags, and per-cell ``element_tag`` region ids where the mesh carries
    ``element_tags`` (see :func:`element_tag_ids` for the mapping).  ``binary=False``
    writes the arrays as text instead, which is readable and diffable but costs a
    Python format per row."""
    X, conn, cell_type, bc_out = _hex_arrays(mesh, _as_bc(mesh, bc))
    _write_vtu(fname, X, conn, cell_type, bc_out=bc_out,
               cell_out=_cell_tags(mesh.element_tags, mesh.n_hexes), binary=binary)
    return mesh


def line_to_vtu(mesh: LineMesh, fname: str, *, binary: bool = True) -> LineMesh:
    """Write an XML VTK unstructured grid (``.vtu``) of a ``LineMesh``, un-welded (one
    node block per line element), with per-cell ``element_tag`` region ids where the
    mesh carries ``element_tags``."""
    X, conn, cell_type = _line_arrays(mesh)
    _write_vtu(fname, X, conn, cell_type,
               cell_out=_cell_tags(mesh.element_tags, mesh.n_lines), binary=binary)
    return mesh


def quad_to_vtu(mesh: QuadMesh, fname: str, *, binary: bool = True) -> QuadMesh:
    """Write an XML VTK unstructured grid (``.vtu``) of a ``QuadMesh``, un-welded (one
    node block per quad), with per-cell ``element_tag`` region ids where the mesh
    carries ``element_tags``."""
    X, conn, cell_type = _quad_arrays(mesh)
    _write_vtu(fname, X, conn, cell_type,
               cell_out=_cell_tags(mesh.element_tags, mesh.n_quads), binary=binary)
    return mesh


# -- .vtp (lightweight surface, for e.g. an in-browser viewer) ----------
def _write_vtp(fname: str, X: PointArray, conn: IntArray, *,
               bc_out: IntArray | None = None, binary: bool = True) -> None:
    """XML VTK PolyData (``.vtp``): ``X`` is the ``(P,3)`` point array and ``conn`` the
    ``(N,4)`` per-quad **corner** connectivity into it (shared/welded ids), written as
    4-point polygons -- no triangulation needed, a VTK polygon takes any vertex count.
    ``bc_out``, when given, is a per-cell ``bc_id`` (one tag id per quad; a surface has
    no interior, so this is cell data rather than the ``.vtu`` writer's point data)."""
    P = X.shape[0]
    N, m = conn.shape
    offsets: IntArray = np.arange(m, m * N + 1, m, dtype=np.int64)
    fmt = "binary" if binary else "ascii"

    def block(a: IntArray | PointArray, rows: Callable[[], str]) -> str:
        return "          %s\n" % _b64(a) if binary else rows()

    with open(fname, "w") as fid:
        fid.write('<?xml version="1.0"?>\n')
        fid.write('<VTKFile type="PolyData" version="1.0" '
                  'byte_order="LittleEndian" header_type="UInt64">\n')
        fid.write("  <PolyData>\n")
        fid.write('    <Piece NumberOfPoints="%d" NumberOfVerts="0" NumberOfLines="0" '
                  'NumberOfStrips="0" NumberOfPolys="%d">\n' % (P, N))
        fid.write("      <Points>\n")
        fid.write('        <DataArray type="Float64" NumberOfComponents="3" '
                  'format="%s">\n' % fmt)
        fid.write(block(np.asarray(X, dtype=np.float64),
                        lambda: "".join("          %.17g %.17g %.17g\n" % (x, y, z)
                                        for x, y, z in X.tolist())))
        fid.write("        </DataArray>\n")
        fid.write("      </Points>\n")
        fid.write("      <Polys>\n")
        fid.write('        <DataArray type="Int64" Name="connectivity" '
                  'format="%s">\n' % fmt)
        fid.write(block(np.asarray(conn, dtype=np.int64),
                        lambda: "".join("          %s\n" % " ".join(map(str, row))
                                        for row in conn.tolist())))
        fid.write("        </DataArray>\n")
        fid.write('        <DataArray type="Int64" Name="offsets" format="%s">\n' % fmt)
        fid.write(block(offsets,
                        lambda: "".join("          %d\n" % o for o in offsets.tolist())))
        fid.write("        </DataArray>\n")
        fid.write("      </Polys>\n")
        if bc_out is not None:
            fid.write('      <CellData Scalars="bc_id">\n')
            fid.write('        <DataArray type="Int32" Name="bc_id" '
                      'format="%s">\n' % fmt)
            fid.write(block(np.asarray(bc_out, dtype=np.int32),
                            lambda: "".join("          %d\n" % v
                                            for v in bc_out.tolist())))
            fid.write("        </DataArray>\n")
            fid.write("      </CellData>\n")
        fid.write("    </Piece>\n")
        fid.write("  </PolyData>\n")
        fid.write("</VTKFile>\n")


def boundary_to_vtp(mesh: HexMesh, fname: str, *, tag: str | None = None,
                    binary: bool = True) -> QuadMesh:
    """Write ``mesh``'s named faces as a lightweight VTK PolyData (``.vtp``) and a
    per-poly ``bc_id``.

    With no ``tag``, every face ``mesh.face_tags`` names is exported (via
    :func:`quadmesh.select <nekmeshpy.quadmesh.assemble.select>` on ``mesh.quad_mesh``)
    -- **not** :func:`hexmesh.boundary_mesh <nekmeshpy.hexmesh.lower.boundary_mesh>`'s
    topological boundary, which drops interior planes on purpose (a T-junction's flux
    plane, say) even though they carry a name. A mesh with no named faces at all falls
    back to the topological boundary, untagged, so an unlabelled block still exports
    *something*. Passing an explicit ``tag`` still selects one named group by way of
    ``boundary_mesh`` (unchanged).

    Always written **corners only**, regardless of ``mesh``'s order -- ``.vtp``'s
    ``<Polys>`` has no curved-cell type (a VTK PolyData polygon is always flat), and a
    viewer only needs the outline, not curved-node fidelity. Interior *volume* nodes are
    dropped either way -- a viewer only shows the surface.

    When the surface carries names, a sidecar ``<fname minus .vtp>.groups.json``
    (``{bc_id: name}``) is written alongside it -- ``.vtp``'s own field-data support
    is a poor fit for strings, and a viewer needs the names to label a per-group
    visibility toggle (e.g. hiding a far-field box to see the body it encloses)."""
    if tag is not None:
        surf = boundary_mesh(mesh, tag)
    elif len(mesh.face_tags):
        surf = quadmesh_select(mesh.quad_mesh, mesh.face_tags.ids)
    else:
        surf = boundary_mesh(mesh)
    nodes, conn = surf.points, surf.corners
    names = surf.element_group_tags
    bc_out: IntArray | None = None
    if names:
        name_to_id = {name: i + 1 for i, name in enumerate(names)}
        bc_out, _ = element_tag_ids(surf.element_tags, surf.n_quads)
        groups_path = (fname[:-4] if fname.endswith(".vtp") else fname) + ".groups.json"
        with open(groups_path, "w") as fid:
            json.dump({str(i): name for name, i in name_to_id.items()}, fid)
    _write_vtp(fname, nodes, conn, bc_out=bc_out, binary=binary)
    return surf

