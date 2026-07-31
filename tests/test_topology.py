"""Topology / watertightness checks for the volume and surface meshes."""

import numpy as np

from nekmeshpy import HexMesh, QuadMesh, TriMesh, topology

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
    assert rep["watertight"] is True
    assert rep["conformal"] is True
    assert rep["n_hanging_points"] == 0
    assert rep["n_boundary_faces"] == 6
    assert rep["n_internal_faces"] == 0
    assert rep["n_nonmanifold_faces"] == 0
    assert rep["n_open_edges"] == 0
    assert rep["n_components"] == 1


def test_stacked_hexes_share_one_face():
    pts, hexes = _stacked_hexes()
    rep = topology.hex_report(pts, hexes)
    assert rep["watertight"] is True
    assert rep["n_internal_faces"] == 1
    assert rep["n_boundary_faces"] == 10
    assert rep["n_components"] == 1
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
    mesh = HexMesh.merge([coarse, *fine])
    rep = mesh.topology_report()
    assert rep["n_open_edges"] == 0        # the defect leaves NO open edge...
    assert rep["watertight"] is True       # ...so it reads as watertight...
    assert rep["n_hanging_points"] >= 4     # ...but the hanging points expose it
    assert rep["conformal"] is False
    assert mesh.is_conforming() is False


def test_disjoint_hexes_are_two_components():
    pts = np.vstack([_UNIT_HEX, _UNIT_HEX + np.array([10.0, 0, 0])])
    hexes = np.array([np.arange(8), np.arange(8, 16)], dtype=np.int64)
    rep = topology.hex_report(pts, hexes)
    assert rep["n_components"] == 2
    # each piece has a closed boundary, but the mesh is not a single body
    assert rep["watertight"] is True
    assert rep["n_nonmanifold_faces"] == 0


def test_bifurcation_mesh_is_watertight(built_mesh):
    mesh = built_mesh["mesh"]
    rep = mesh.topology_report()
    assert rep["n_components"] == 1
    assert rep["n_nonmanifold_faces"] == 0
    assert rep["n_open_edges"] == 0
    assert rep["n_hanging_points"] == 0
    assert rep["watertight"] is True
    assert rep["conformal"] is True
    assert mesh.is_watertight() is True
    assert mesh.is_conforming() is True
    # the true (topological) boundary is exactly the wall + outlet faces; the
    # flux-measurement planes (flux_1/flux_2) are interior faces that also carry a name
    outer = ["wall", "trunk_outlet", "top_outlet_1", "top_outlet_2"]
    exterior = int(np.isin(mesh.boundary_tags, outer).sum())
    assert rep["n_boundary_faces"] == exterior
    assert exterior < mesh.boundaries.shape[0]      # flux planes are extra, interior


def test_hexmesh_report_matches_free_function(built_mesh):
    mesh = built_mesh["mesh"]
    X, HC, _ = mesh.weld()
    assert mesh.topology_report() == topology.hex_report(X, HC)


def test_boundary_helpers_single_hex():
    mesh = HexMesh.from_corners(_UNIT_HEX, np.arange(8).reshape(1, 8))
    assert mesh.boundary_faces().shape == (6, 2)                 # all 6 faces
    assert set(mesh.boundary_faces()[:, 1]) == {1, 2, 3, 4, 5, 6}
    assert mesh.boundary_elements().tolist() == [0]
    assert mesh.boundary_points().tolist() == list(range(8))      # every point


def test_boundary_helpers_match_topology(built_mesh):
    mesh = built_mesh["mesh"]
    rep = mesh.topology_report()
    assert mesh.boundary_faces().shape[0] == rep["n_boundary_faces"]
    # boundary faces are the wall + outlet named faces (flux planes are interior)
    outer = ["wall", "trunk_outlet", "top_outlet_1", "top_outlet_2"]
    exterior = mesh.boundaries[np.isin(mesh.boundary_tags, outer)]
    got = {(int(e), int(f)) for e, f in mesh.boundary_faces()}
    want = {(int(e), int(f)) for e, f in exterior}
    assert got == want
    # point ids on the domain boundary, consistent with the face points
    face_points = np.unique(
        mesh.hexes[mesh.boundary_faces()[:, 0][:, None],
                   mesh.FACE_POINTS[mesh.boundary_faces()[:, 1] - 1]])
    assert np.array_equal(mesh.boundary_points(), face_points)
    assert set(mesh.boundary_elements()) <= set(range(mesh.n_hexes))


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
    # and hands the ordered points to LineMesh.loft(..., loop=True).
    from nekmeshpy.trimesh.ops import _chain_segments
    # four segments of a unit square, given unordered, chain into a closed loop
    segs = np.array([[0, 0, 0, 1, 0, 0], [1, 1, 0, 0, 1, 0],
                     [1, 0, 0, 1, 1, 0], [0, 1, 0, 0, 0, 0]], float)
    lm = _chain_segments(segs)
    # the degree-based walk closes the loop structurally: every point has degree 2
    assert lm is not None and lm.boundary_points().size == 0
    # the four square corners are recovered (the walk repeats the start to close)
    assert len(np.unique(np.round(lm.points, 9), axis=0)) == 4
    assert _chain_segments(None) is None


def test_quadmesh_boundary_helpers():
    # two quads sharing edge (1,4); every point is on the perimeter
    points = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0],
                      [0, 1, 0], [1, 1, 0], [2, 1, 0]], dtype=float)
    quads = np.array([[0, 1, 4, 3], [1, 2, 5, 4]], dtype=np.int64)
    qm = QuadMesh.from_corners(points, quads)
    assert qm.boundary_edges().shape[0] == 6              # 8 edges - 2 shared
    assert qm.boundary_elements().tolist() == [0, 1]
    assert qm.boundary_points().tolist() == [0, 1, 2, 3, 4, 5]
    square = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=float)
    single = QuadMesh.from_corners(square, np.array([[0, 1, 2, 3]]))
    assert single.boundary_edges().shape[0] == 4
    assert single.boundary_points().tolist() == [0, 1, 2, 3]


def test_single_triangle_is_open():
    rep = topology.surface_report(_TET_V[:3], np.array([[0, 1, 2]]))
    assert rep["closed"] is False
    assert rep["n_boundary_edges"] == 3
    assert rep["n_boundary_loops"] == 1


def test_bifurcation_surface_has_three_openings():
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
    assert mesh.is_watertight() is True
