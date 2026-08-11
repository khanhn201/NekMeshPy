"""Shared fixtures for the NekMeshPy regression suite.

The concrete geometry meshers are flat scripts in ``examples/``; the tests run
them with :func:`runpy.run_path` and read the resulting ``mesh`` global.  The
``built_mesh`` fixture runs ``examples/carotid.py`` once per session (into a
temp dir), returning the assembled :class:`~nekmeshpy.hexmesh.HexMesh`
plus its written ``.re2``/``.vtu`` paths.  Golden reference outputs live
in ``tests/golden/`` (a frozen snapshot of the validated results).
"""

import os
import runpy
import struct
from collections import Counter

import matplotlib
import numpy as np
import pytest

# the suite runs headless (viz tests import matplotlib), so pin a non-interactive
# backend here -- no MPLBACKEND=Agg needed on the command line.
matplotlib.use("Agg")

_HERE = os.path.dirname(__file__)
_EXAMPLES = os.path.join(_HERE, "..", "examples")
GOLDEN = os.path.join(_HERE, "golden")

# bundled ``car`` surface used by the carotid example
CAR_VTX = os.path.join(_EXAMPLES, "data", "car.vtx")
CAR_TRI = os.path.join(_EXAMPLES, "data", "car.tri")


def run_example(name, tmp_path):
    """Execute the flat example script ``examples/<name>`` in ``tmp_path`` and
    return its module namespace (``ns["mesh"]`` is the built HexMesh)."""
    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        return runpy.run_path(os.path.join(_EXAMPLES, name), run_name="__main__")
    finally:
        os.chdir(cwd)


@pytest.fixture(scope="session")
def built_mesh(tmp_path_factory):
    out = tmp_path_factory.mktemp("mesh")
    ns = run_example("carotid.py", out)
    return {
        "mesh": ns["mesh"],
        "groups": ns["GROUPS"],          # the example's own name -> Nek code mapping
        "re2": os.path.join(out, "carotid.re2"),
        "vtu": os.path.join(out, "carotid.vtu"),
    }


def conformal(mesh):
    """``(nodes (M,3), conn_ho (E,(N+1)^d))`` conformal high-order view of any
    container, walked straight off its entity B-rep.

    This is the public replacement for the deleted ``mesh.to_conformal()`` facade:
    the tests below call it wherever they need the single global node numbering.
    """
    from nekmeshpy import HexMesh, LineMesh, QuadMesh
    from nekmeshpy.core import conform
    if isinstance(mesh, LineMesh):
        return conform.conformal_line(mesh.points, mesh.lines, mesh.interior,
                                      mesh.order)
    if isinstance(mesh, QuadMesh):
        return conform.conformal_quad(mesh.points, mesh.quads, mesh.quad, mesh.flip,
                                      mesh.lines.interior, mesh.interior, mesh.order)
    assert isinstance(mesh, HexMesh)
    return conform.conformal_hex(
        mesh.points, mesh.hexes, mesh._elem_edges, mesh._edge_flip,
        mesh.quads.lines.interior, mesh.hex, mesh.face_orient, mesh.quads.interior,
        mesh.interior, mesh.order)


def quad_from_entities(points, quads, edge_nodes=None, interior=None,
                       element_tags=None, *, order=1):
    """Local test scaffold: build a ``QuadMesh`` from corner ``points`` ``(P,3)`` +
    CCW ``quads`` ``(Q,4)`` plus already-decomposed high-order tables.

    The library used to expose this as ``QuadMesh._from_entities``; it was removed
    because every production caller either owns the edge ``LineMesh`` already
    (``loft`` / ``blend`` / the section factories) or re-derives the topology inline.
    The tests keep it as a scaffold for the corner -> B-rep round-trip checks.
    """
    from nekmeshpy import LineMesh, QuadMesh
    from nekmeshpy.core import conform
    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    conn = np.asarray(quads, dtype=np.int64).reshape(-1, 4)
    edges, elem_edges, flip = conform.unique_edges(conn, 2)
    lm = LineMesh(pts, edges, interior=edge_nodes)
    return QuadMesh(lm, elem_edges, flip, interior, element_tags)


def vtu_cell_types(path):
    """The distinct VTK cell type ids in a ``.vtu``, decoded from its binary ``types``
    array (base64 of a byte count followed by the raw values)."""
    import base64
    import xml.etree.ElementTree as ET
    root = ET.parse(path).getroot()
    ta = next(da for da in root.iter("DataArray") if da.get("Name") == "types")
    raw = base64.b64decode(ta.text.strip())
    n = int(np.frombuffer(raw[:8], "<u8")[0])
    return set(np.unique(np.frombuffer(raw[8:8 + n], "u1")).tolist())


def curved(mesh):
    """The per-element ``(E,(N+1)^d,3)`` node block, gathered transiently as
    ``nodes[conn_ho]`` from :func:`conformal` -- exactly what the deleted ``.curved``
    property used to return, but derived from the B-rep instead of stored."""
    nodes, conn_ho = conformal(mesh)
    return nodes[conn_ho]


def read_re2_coords(path):
    """Return (n_elem, coords[n_elem*3*8] float64, bnd_block bytes)."""
    with open(path, "rb") as f:
        hdr = f.read(80)
        f.read(4)  # test float32
        num_elem = int(hdr.split()[1])
        # each element: 1 group double + 8x + 8y + 8z doubles = 25 doubles
        elem_block = np.fromfile(f, dtype="<f8", count=num_elem * 25)
        rest = f.read()
    coords = elem_block.reshape(num_elem, 25)[:, 1:]  # drop the group double
    return num_elem, coords, rest


def read_re2_boundary(path):
    """The ``.re2`` boundary block decoded into a ``Counter`` of
    ``(element (1-based), face, code)`` -- the block's *content*, freed of its row order.

    The byte comparison in ``test_re2_boundary_block_identical`` is the stricter check
    and stays, but it also pins something that is not part of the contract: which order
    the rows happen to be written in. This is the part that must survive a refactor of
    how tags are stored, so it is asserted separately and can outlive a regenerated
    baseline."""
    with open(path, "rb") as f:
        hdr = f.read(80)
        f.read(4)
        num_elem = int(hdr.split()[1])
        np.fromfile(f, dtype="<f8", count=num_elem * 25)
        rest = f.read()
    n_bnd = int(np.frombuffer(rest[:16], dtype="<f8")[1])
    rows = np.frombuffer(rest[16:16 + n_bnd * 64], dtype="<f8").reshape(n_bnd, 8)
    # a code rides in the 8 bytes of one double; Nek's own field is 3 chars, so the
    # rest are the NUL padding ``_str_to_double`` left there
    return Counter(
        (int(r[0]), int(r[1]),
         struct.pack("<d", r[7]).decode("ascii", "replace").rstrip("\x00"))
        for r in rows)


def face_rows(mesh):
    """``[(element, face, tag), ...]`` for a ``HexMesh``, lexsorted by (element, face).

    A tag names a shared face now, so this is the reconstruction an exporter does --
    one entry per hex carrying a named face, which means two for a named *interior*
    face. Kept here because most of these tests were written against the old
    ``(element, side)`` storage and still read most naturally in those terms."""
    from nekmeshpy.hexmesh.query import face_tag_rows
    rows, names = face_tag_rows(mesh)
    return [(int(e), int(f), str(t))
            for (e, f), t in zip(rows.tolist(), names.tolist())]


def assert_same_side_tags(a, b):
    """The two side-tag tables carry the same rows in the same order.

    The tables set ``eq=False`` (the generated ``__eq__`` would compare ndarray
    fields and raise), so equality is spelt column by column -- and the columns differ
    by rung as the tags fold onto the shared entity: a table addressed by
    ``(element, side)`` still has both, one addressed by entity id has only ``ids``."""
    assert np.array_equal(a.tags, b.tags)
    if hasattr(a, "ids") or hasattr(b, "ids"):
        assert np.array_equal(a.ids, b.ids)
        return
    assert np.array_equal(a.elements, b.elements)
    assert np.array_equal(a.sides, b.sides)
