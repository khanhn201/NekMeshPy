"""Shared fixtures for the NekMeshPy regression suite.

The concrete geometry meshers are flat scripts in ``examples/``; the tests run
them with :func:`runpy.run_path` and read the resulting ``mesh`` global.  The
``built_mesh`` fixture runs ``examples/bifurcation.py`` once per session (into a
temp dir), returning the assembled :class:`~nekmeshpy.hexmesh.HexMesh`
plus its written ``.re2``/``.vtu`` paths.  Golden reference outputs live
in ``tests/golden/`` (a frozen snapshot of the validated results).
"""

import os
import runpy

import matplotlib
import numpy as np
import pytest

# the suite runs headless (viz tests import matplotlib), so pin a non-interactive
# backend here -- no MPLBACKEND=Agg needed on the command line.
matplotlib.use("Agg")

_HERE = os.path.dirname(__file__)
_EXAMPLES = os.path.join(_HERE, "..", "examples")
GOLDEN = os.path.join(_HERE, "golden")

# bundled ``car`` surface used by the bifurcation example
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
    ns = run_example("bifurcation.py", out)
    return {
        "mesh": ns["mesh"],
        "re2": os.path.join(out, "bifurcation.re2"),
        "vtu": os.path.join(out, "bifurcation.vtu"),
    }


def conformal(mesh):
    """``(nodes (M,3), conn_ho (E,(N+1)^d))`` conformal high-order view of any
    container, walked straight off its entity B-rep.

    This is the public replacement for the deleted ``mesh.to_conformal()`` facade:
    the tests below call it wherever they need the single global node numbering.
    """
    from nekmeshpy import HexMesh, LineMesh, QuadMesh
    from nekmeshpy.model import conform
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
                       edge_tags=None, element_tags=None, *, order=1):
    """Local test scaffold: build a ``QuadMesh`` from corner ``points`` ``(P,3)`` +
    CCW ``quads`` ``(Q,4)`` plus already-decomposed high-order tables.

    The library used to expose this as ``QuadMesh._from_entities``; it was removed
    because every production caller either owns the edge ``LineMesh`` already
    (``loft`` / ``blend`` / the section factories) or re-derives the topology inline.
    The tests keep it as a scaffold for the corner -> B-rep round-trip checks.
    """
    from nekmeshpy import LineMesh, QuadMesh
    from nekmeshpy.model import conform
    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    conn = np.asarray(quads, dtype=np.int64).reshape(-1, 4)
    edges, elem_edges, flip = conform.unique_edges(conn, 2)
    lm = LineMesh(pts, edges, interior=edge_nodes)
    return QuadMesh(lm, elem_edges, flip, interior, edge_tags,
                    element_tags)


def hex_from_entities(points, hexes, edge_nodes=None, face_nodes=None,
                      interior=None, face_tags=None, element_tags=None,
                      *, order=1):
    """Local test scaffold: build a ``HexMesh`` from corner ``points`` ``(P,3)`` +
    Nek-order ``hexes`` ``(E,8)`` plus already-decomposed high-order tables.

    The hex-level sibling of :func:`quad_from_entities`, and likewise the removed
    ``HexMesh._from_entities``.  ``conform.unique_edges(hexes, 3)`` and
    ``conform.unique_edges(canonical_conn, 2)`` are the same array, so an
    ``edge_nodes`` table scattered with the hex incidence indexes the shared-face
    ``QuadMesh`` consistently.
    """
    from nekmeshpy import HexMesh, LineMesh, QuadMesh
    from nekmeshpy.model import conform
    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    conn = np.asarray(hexes, dtype=np.int64).reshape(-1, 8)
    canonical_conn, elem_faces, face_orient = conform.canonical_faces(conn)
    q_edges, q_elem_edges, q_flip = conform.unique_edges(canonical_conn, 2)
    edge_lm = LineMesh(pts, q_edges, interior=edge_nodes)
    quads = QuadMesh(edge_lm, q_elem_edges, q_flip, face_nodes)
    return HexMesh(quads, elem_faces, face_orient, interior, face_tags,
                   element_tags)


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


def assert_same_side_tags(a, b):
    """The two side-tag tables carry the same rows in the same order.

    The tables set ``eq=False`` (the generated ``__eq__`` would compare ndarray
    fields and raise), so equality is spelt column by column."""
    assert np.array_equal(a.elements, b.elements)
    assert np.array_equal(a.sides, b.sides)
    assert np.array_equal(a.tags, b.tags)
