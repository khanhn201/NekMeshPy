"""Topology / watertightness checks for the volume and surface meshes."""

import numpy as np
from conftest import face_rows

from nekmeshpy import HexMesh, QuadMesh, TriMesh, hexmesh, linemesh, quadmesh, topology
from nekmeshpy.core import tags as tags_mod

# unit hex in Nek corner order
_UNIT_HEX = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
                      [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]], dtype=float)

# closed tetrahedron surface (every edge shared by exactly two triangles)
_TET_V = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
_TET_F = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]], dtype=np.int64)


def _stacked_hexes():
    """Two hexes sharing one full quad face (welded, watertight)."""
    pts = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
                    [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
                    [0, 0, 2], [1, 0, 2], [1, 1, 2], [0, 1, 2]], dtype=float)
    hexes = np.array([[0, 1, 2, 3, 4, 5, 6, 7],
                      [4, 5, 6, 7, 8, 9, 10, 11]], dtype=np.int64)
    return pts, hexes


# -- hex volume ---------------------------------------------------------
def test_single_hex_is_watertight():
    rep = topology.hex_report(_UNIT_HEX.reshape(8, 3), np.arange(8).reshape(1, 8))
    assert rep.watertight is True
    assert rep.conformal is True
    assert rep.n_hanging_points == 0
    assert rep.n_boundary_faces == 6
    assert rep.n_internal_faces == 0
    assert rep.n_nonmanifold_faces == 0
    assert rep.n_open_edges == 0
    assert rep.n_components == 1


def test_stacked_hexes_share_one_face():
    pts, hexes = _stacked_hexes()
    rep = topology.hex_report(pts, hexes)
    assert rep.watertight is True
    assert rep.n_internal_faces == 1
    assert rep.n_boundary_faces == 10
    assert rep.n_components == 1
    assert topology.is_watertight(pts, hexes) is True


def _box(x0, x1, y0, y1, z0, z1):
    """One hex in Nek corner order spanning the given axis-aligned box."""
    return np.array([[x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
                     [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]],
                    dtype=float)


def _hex(box):
    """A one-element HexMesh from an (8,3) Nek-ordered corner block."""
    return HexMesh.from_corners(box, np.arange(8).reshape(1, 8))


def test_t_junction_is_watertight_but_not_conformal():
    """A coarse hex abutting a 2x2 fan of fine hexes: the boundary still pairs
    into a closed manifold (watertight), so only the hanging-point test catches
    the non-conforming interface.  merge() welds the shared coarse/fine corners."""
    coarse = _hex(_box(0, 1, 0, 1, 0, 1))
    fine = [_hex(_box(1, 2, y0, y0 + 0.5, z0, z0 + 0.5))
            for y0 in (0.0, 0.5) for z0 in (0.0, 0.5)]
    mesh = hexmesh.merge([coarse, *fine])
    rep = hexmesh.topology_report(mesh)
    assert rep.n_open_edges == 0        # the defect leaves NO open edge...
    assert rep.watertight is True       # ...so it reads as watertight...
    assert rep.n_hanging_points >= 4     # ...but the hanging points expose it
    assert rep.conformal is False
    assert hexmesh.is_conforming(mesh) is False


def test_disjoint_hexes_are_two_components():
    pts = np.vstack([_UNIT_HEX, _UNIT_HEX + np.array([10.0, 0, 0])])
    hexes = np.array([np.arange(8), np.arange(8, 16)], dtype=np.int64)
    rep = topology.hex_report(pts, hexes)
    assert rep.n_components == 2
    # each piece has a closed boundary, but the mesh is not a single body
    assert rep.watertight is True
    assert rep.n_nonmanifold_faces == 0


def test_carotid_mesh_is_watertight(built_mesh):
    mesh = built_mesh["mesh"]
    rep = hexmesh.topology_report(mesh)
    assert rep.n_components == 1
    assert rep.n_nonmanifold_faces == 0
    assert rep.n_open_edges == 0
    assert rep.n_hanging_points == 0
    assert rep.watertight is True
    assert rep.conformal is True
    assert hexmesh.is_watertight(mesh) is True
    assert hexmesh.is_conforming(mesh) is True
    # the true (topological) boundary is exactly the wall + outlet faces; the
    # flux-measurement planes (flux_1/flux_2) are interior faces that also carry a name
    outer = ["wall", "trunk_outlet", "top_outlet_1", "top_outlet_2"]
    exterior = int(np.isin(mesh.face_tags.tags, outer).sum())
    assert rep.n_boundary_faces == exterior
    assert exterior < len(mesh.face_tags)      # flux planes are extra, interior


def test_hexmesh_report_matches_free_function(built_mesh):
    mesh = built_mesh["mesh"]
    X, HC, _ = hexmesh.weld(mesh)
    assert hexmesh.topology_report(mesh) == topology.hex_report(X, HC)


def test_tag_report_counts_both_disagreements(built_mesh):
    """The carotid names every boundary face and, on top of that, two interior flux
    planes -- one of each way ``face_tags`` and the boundary can differ."""
    mesh = built_mesh["mesh"]
    tags = hexmesh.tag_report(mesh)
    rep = hexmesh.topology_report(mesh)
    assert tags.n_rows == len(mesh.face_tags)
    assert tags.n_untagged_boundary == 0
    assert tags.n_tagged_interior == len(mesh.face_tags) - rep.n_boundary_faces > 0
    assert "untagged bdry  : 0 faces" in hexmesh.report(mesh)
    assert "interior tags  : %d rows" % tags.n_tagged_interior in hexmesh.report(mesh)


def test_tag_report_flags_an_untagged_boundary():
    """A lone hex with one of its six faces named: five boundary faces go uncovered."""
    mesh = HexMesh.from_corners(_UNIT_HEX, np.arange(8).reshape(1, 8))
    mesh = hexmesh.HexMesh(
        quadmesh.QuadMesh(mesh.quads.lines, mesh.quads.quad, mesh.quads.flip, None,
                          tags_mod.ElementTags([int(mesh.hex[0, 4])], ["bottom"])),
        mesh.hex, mesh.face_orient)
    assert hexmesh.tag_report(mesh) == (1, 5, 0)


def test_boundary_helpers_single_hex():
    mesh = HexMesh.from_corners(_UNIT_HEX, np.arange(8).reshape(1, 8))
    assert hexmesh.boundary_faces(mesh).shape == (6, 2)                 # all 6 faces
    assert set(hexmesh.boundary_faces(mesh)[:, 1]) == {1, 2, 3, 4, 5, 6}
    assert hexmesh.boundary_elements(mesh).tolist() == [0]
    assert hexmesh.boundary_points(mesh).tolist() == list(range(8))      # every point


def test_boundary_helpers_match_topology(built_mesh):
    mesh = built_mesh["mesh"]
    rep = hexmesh.topology_report(mesh)
    assert hexmesh.boundary_faces(mesh).shape[0] == rep.n_boundary_faces
    # boundary faces are the wall + outlet named faces (flux planes are interior)
    outer = ["wall", "trunk_outlet", "top_outlet_1", "top_outlet_2"]
    got = {(int(e), int(f)) for e, f in hexmesh.boundary_faces(mesh)}
    want = {(e, f) for e, f, t in face_rows(mesh) if t in outer}
    assert got == want
    # point ids on the domain boundary, consistent with the face points
    face_points = np.unique(
        mesh.hexes[hexmesh.boundary_faces(mesh)[:, 0][:, None],
                   mesh.FACE_POINTS[hexmesh.boundary_faces(mesh)[:, 1] - 1]])
    assert np.array_equal(hexmesh.boundary_points(mesh), face_points)
    assert set(hexmesh.boundary_elements(mesh)) <= set(range(mesh.n_hexes))


# -- triangle surface ---------------------------------------------------
def test_closed_tetrahedron():
    rep = topology.surface_report(_TET_V, _TET_F)
    assert rep["closed"] is True
    assert rep["n_boundary_edges"] == 0
    assert rep["n_boundary_loops"] == 0
    assert rep["n_nonmanifold_edges"] == 0
    assert rep["n_components"] == 1
    assert TriMesh(_TET_V, _TET_F).is_closed() is True


def test_trimesh_boundary_helpers():
    closed = TriMesh(_TET_V, _TET_F)
    assert closed.boundary_edges().shape[0] == 0          # closed -> no boundary
    assert closed.boundary_points().size == 0
    tri = TriMesh(_TET_V[:3], np.array([[0, 1, 2]]))
    assert tri.boundary_edges().shape == (3, 2)
    assert tri.boundary_elements().tolist() == [0]
    assert tri.boundary_points().tolist() == [0, 1, 2]


def test_chain_segments_builds_a_closed_loop():
    # Ordering an unordered segment soup into a ring is a *surface* op (it is how a
    # marched isocontour arrives), so it lives in trimesh.ops beside its only caller
    # and hands the ordered points to linemesh.loft(..., loop=True).
    from nekmeshpy.trimesh.ops import _chain_segments
    # four segments of a unit square, given unordered, chain into a closed loop
    segs = np.array([[0, 0, 0, 1, 0, 0], [1, 1, 0, 0, 1, 0],
                     [1, 0, 0, 1, 1, 0], [0, 1, 0, 0, 0, 0]], float)
    lm = _chain_segments(segs)
    # the degree-based walk closes the loop structurally: every point has degree 2
    assert lm is not None and linemesh.boundary_points(lm).size == 0
    # the four square corners are recovered (the walk repeats the start to close)
    assert len(np.unique(np.round(lm.points, 9), axis=0)) == 4
    assert _chain_segments(None) is None


def test_quadmesh_boundary_helpers():
    # two quads sharing edge (1,4); every point is on the perimeter
    points = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0],
                      [0, 1, 0], [1, 1, 0], [2, 1, 0]], dtype=float)
    quads = np.array([[0, 1, 4, 3], [1, 2, 5, 4]], dtype=np.int64)
    qm = QuadMesh.from_corners(points, quads)
    assert quadmesh.boundary_edges(qm).shape[0] == 6              # 8 edges - 2 shared
    assert quadmesh.boundary_elements(qm).tolist() == [0, 1]
    assert quadmesh.boundary_points(qm).tolist() == [0, 1, 2, 3, 4, 5]
    square = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=float)
    single = QuadMesh.from_corners(square, np.array([[0, 1, 2, 3]]))
    assert quadmesh.boundary_edges(single).shape[0] == 4
    assert quadmesh.boundary_points(single).tolist() == [0, 1, 2, 3]


def test_single_triangle_is_open():
    rep = topology.surface_report(_TET_V[:3], np.array([[0, 1, 2]]))
    assert rep["closed"] is False
    assert rep["n_boundary_edges"] == 3
    assert rep["n_boundary_loops"] == 1


def test_carotid_surface_has_three_openings():
    from conftest import CAR_TRI, CAR_VTX
    surf = TriMesh.from_files(CAR_VTX, CAR_TRI)
    rep = surf.topology_report()
    assert rep["n_components"] == 1
    assert rep["n_nonmanifold_edges"] == 0
    assert rep["n_boundary_loops"] == 3      # trunk + two branches
    assert rep["closed"] is False            # open at the three vessel ends


def test_format_report_roundtrips():
    hrep = topology.hex_report(_UNIT_HEX.reshape(8, 3), np.arange(8).reshape(1, 8))
    srep = topology.surface_report(_TET_V, _TET_F)
    assert "watertight" in topology.format_report(hrep)
    assert "closed" in topology.format_report(srep)


def test_hexmesh_from_arrays_is_watertight():
    """A HexMesh built from the array constructor reports watertight."""
    mesh = HexMesh.from_corners(_UNIT_HEX, np.arange(8).reshape(1, 8))
    assert hexmesh.is_watertight(mesh) is True
