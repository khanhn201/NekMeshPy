"""``TetMesh`` container and ``tetmesh.ops`` (P1 conduction, boundary extraction,
capping).  gmsh-backed generation is a separate, skippable test -- everything else
here runs on small synthetic tet meshes."""

import itertools

import numpy as np
import pytest
from scipy.spatial import Delaunay

from nekmeshpy import TetMesh, TriMesh, tetmesh, trimesh


def _bipyramid():
    """Two tets glued on a shared triangular face: 5 points, all on the boundary."""
    points = np.array([
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0],   # the shared face
        [0.0, 0.0, 1.0],                                     # apex above
        [0.0, 0.0, -1.0],                                    # apex below
    ])
    tets = np.array([[0, 1, 2, 3], [0, 1, 2, 4]])
    return TetMesh(points, tets)


def _grid_mesh():
    """A 3x3x3 point grid, Delaunay-tetrahedralized: 27 points, exactly one interior
    point (originally (1,1,1)), the rest on the boundary of the cube.

    A perfectly regular grid gives Qhull degenerate (near-zero-volume) simplices, so
    the points are jittered slightly -- deterministically, and small enough that the
    one interior point is still identified by its original index, not by geometry."""
    pts = np.array(list(itertools.product([0.0, 1.0, 2.0], repeat=3)))
    interior = int(np.flatnonzero(np.all(pts == 1.0, axis=1))[0])
    rng = np.random.default_rng(0)
    pts = pts + rng.uniform(-1e-3, 1e-3, size=pts.shape)
    tets = Delaunay(pts).simplices
    return TetMesh(pts, tets), pts, interior


def test_bipyramid_boundary_faces():
    mesh = _bipyramid()
    assert mesh.n_points == 5
    assert mesh.n_tets == 2
    bf = mesh.boundary_faces()
    assert bf.shape == (6, 3)          # 8 face instances, 1 shared face cancels twice
    # the shared face (0,1,2) must not appear
    assert not np.any(np.all(np.sort(bf, axis=1) == [0, 1, 2], axis=1))


def test_boundary_mesh_is_a_trimesh():
    mesh = _bipyramid()
    surf = mesh.boundary_mesh()
    assert isinstance(surf, TriMesh)
    assert surf.n_tris == 6
    assert surf.is_closed()


def test_repr_survives_empty_and_normal():
    assert "5 points" in repr(_bipyramid())


def test_solve_dirichlet_reproduces_a_linear_field():
    # a linear field is exactly the P1 FEM solution to Laplace, so Dirichlet data
    # taken from it on the boundary must come back unchanged at the interior node too
    mesh, pts, interior = _grid_mesh()
    f = pts @ np.array([1.0, 2.0, 3.0])
    boundary = np.setdiff1d(np.arange(mesh.n_points), [interior])
    u = tetmesh.ops.solve_dirichlet(mesh, boundary, f[boundary])
    assert np.allclose(u, f, atol=1e-7)


def test_seam_fields_and_leg_label_shapes():
    mesh, pts, interior = _grid_mesh()
    boundary = np.setdiff1d(np.arange(mesh.n_points), [interior])
    # an arbitrary 3-way split of the boundary nodes into "caps", just to exercise
    # the machinery -- not a physically meaningful junction
    caps = np.array_split(boundary, 3)
    U = tetmesh.ops.seam_fields(mesh, caps)
    assert U.shape == (mesh.n_points, 3)
    assert np.all(np.isfinite(U))
    lab = tetmesh.ops.leg_label(mesh, U)
    assert lab.shape == (mesh.n_tets,)
    assert set(np.unique(lab)) <= {0, 1, 2, 3}


def test_cap_surface_closes_an_open_patch():
    # two triangles sharing a diagonal: a flat open patch whose outer 4 points form
    # a single boundary loop
    points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
                       [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
    patch = TriMesh(points, np.array([[0, 1, 2], [0, 2, 3]]))
    loops = patch.boundary_loops()
    assert len(loops) == 1
    ordered = trimesh.ops.order_boundary_loop(patch, loops[0])
    capped, sets = tetmesh.ops.cap_surface(patch, [ordered])
    assert len(capped.boundary_loops()) == 0        # closed: no open boundary left
    assert len(sets) == 1
    assert set(ordered.tolist()) <= set(sets[0].tolist())


def test_tet_mesh_generates_a_watertight_volume():
    pytest.importorskip("gmsh")
    # a small octahedron: 6 vertices, 8 outward-wound triangular faces
    V = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0],
                  [0, -1, 0], [0, 0, 1], [0, 0, -1]], dtype=float)
    F = np.array([
        [0, 2, 4], [2, 1, 4], [1, 3, 4], [3, 0, 4],
        [2, 0, 5], [1, 2, 5], [3, 1, 5], [0, 3, 5],
    ])
    surf = TriMesh(V, F)
    assert surf.is_closed()
    mesh = tetmesh.ops.tet_mesh(surf, near=0.6, far=0.6, ramp=1.0, centre=(0.0, 0.0, 0.0))
    assert mesh.n_tets > 0
    boundary_surf = mesh.boundary_mesh()
    assert boundary_surf.is_closed()
    # the wall's own points are kept, in order, at the front
    assert np.allclose(mesh.points[:6], V)
