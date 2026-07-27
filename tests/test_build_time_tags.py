"""Build-time boundary tagging: per-edge names on a QuadMesh section propagate to
the swept side faces (HexMesh.loft/extrude), NO_BOUNDARY suppresses a face, and a
NO_BOUNDARY seam lets HexMesh.merge stay a plain concatenate with no stale tag on
the welded-away interior face."""

from collections import Counter

import numpy as np

from nekmeshpy import NO_BOUNDARY, HexMesh, QuadMesh
from nekmeshpy.model.fields import uniform_spacing

# a unit square, one quad, CCW: edges (0,1) bottom, (1,2) right, (2,3) top, (3,0) left
_SQUARE_PTS = [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]]
_SQUARE_QUADS = [[0, 1, 2, 3]]
_SIDES = {frozenset((0, 1)): "bottom", frozenset((1, 2)): "right",
          frozenset((2, 3)): "top", frozenset((3, 0)): "left"}


def _face_centroids(mesh):
    """(K,3) centroid of every tagged boundary face."""
    out = []
    for e, f in mesh.boundaries:
        out.append(mesh.points[mesh.hexes[e, HexMesh.FACE_POINTS[f - 1]]].mean(axis=0))
    return np.array(out).reshape(-1, 3)


def test_boundary_names_derive_boundaries():
    qm = QuadMesh(_SQUARE_PTS, _SQUARE_QUADS, boundary_names=_SIDES)
    # only boundary_names given -> every named edge is held as a boundary
    assert qm.boundaries == set(_SIDES)
    assert qm.boundary_names[frozenset((0, 1))] == "bottom"


def test_loft_propagates_per_edge_boundary_names_to_side_faces():
    qm = QuadMesh(_SQUARE_PTS, _SQUARE_QUADS, boundary_names=_SIDES)
    blk = HexMesh.extrude(qm, length=1.0, layers=uniform_spacing(1),
                          first_cap="inlet", last_cap="outlet")
    counts = Counter(blk.boundary_names.tolist())
    # 4 named sides + 2 caps, each once (one hex)
    assert counts == {"bottom": 1, "right": 1, "top": 1, "left": 1,
                      "inlet": 1, "outlet": 1}
    assert blk.is_watertight()


def test_no_boundary_suppresses_a_swept_face():
    # name three edges "wall"; the right edge is declared NO_BOUNDARY
    qm = QuadMesh(_SQUARE_PTS, _SQUARE_QUADS,
                  boundary_names={frozenset((0, 1)): "wall", frozenset((2, 3)): "wall",
                              frozenset((3, 0)): "wall", frozenset((1, 2)): NO_BOUNDARY})
    blk = HexMesh.extrude(qm, length=1.0, layers=uniform_spacing(2))
    names = blk.boundary_names.tolist()
    assert "" not in names                       # NO_BOUNDARY never emitted
    # 3 walls x 2 layers = 6 wall faces; the 4th (right) edge is suppressed
    assert Counter(names)["wall"] == 6


def test_boundaries_without_names_tags_nothing():
    # boundaries alone no longer auto-tag side faces -- naming is boundary_names only
    qm = QuadMesh(_SQUARE_PTS, _SQUARE_QUADS, boundaries=list(_SIDES))
    blk = HexMesh.extrude(qm, length=1.0, layers=uniform_spacing(1),
                          first_cap="inlet", last_cap="outlet")
    # only the two caps are tagged; the four (unnamed) side faces are not
    assert Counter(blk.boundary_names.tolist()) == {"inlet": 1, "outlet": 1}


def test_quadmesh_merge_unions_and_remaps_boundary_names():
    a = QuadMesh([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], _SQUARE_QUADS,
                 boundary_names={frozenset((0, 1)): "a_bottom"})
    b = QuadMesh([[1, 0, 0], [2, 0, 0], [2, 1, 0], [1, 1, 0]], _SQUARE_QUADS,
                 boundary_names={frozenset((0, 1)): "b_bottom"})
    m = QuadMesh.merge([a, b])
    assert set(m.boundary_names.values()) == {"a_bottom", "b_bottom"}
    # both names still sit on the y=0 edges after the point remap
    for edge in m.boundary_names:
        assert np.allclose(m.points[list(edge), 1], 0.0)


def _box(lo, hi, tags):
    xs = np.linspace(lo[0], hi[0], 2)
    ys = np.linspace(lo[1], hi[1], 2)
    zs = np.linspace(lo[2], hi[2], 2)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    return HexMesh.from_grid(np.stack([X, Y, Z], axis=-1), face_tags=tags)


def test_no_boundary_seam_keeps_merge_a_plain_concatenate():
    """Two stacked boxes: tag every outer side, declare the touching faces
    NO_BOUNDARY, and merge -- no tag lands on the welded interior interface."""
    sides = {"x_min": "wall", "x_max": "wall", "y_min": "wall", "y_max": "wall"}
    lower = _box((0, 0, 0), (1, 1, 1), {**sides, "z_min": "bottom", "z_max": NO_BOUNDARY})
    upper = _box((0, 0, 1), (1, 1, 2), {**sides, "z_min": NO_BOUNDARY, "z_max": "top"})
    mesh = HexMesh.merge([lower, upper])

    assert mesh.is_watertight() and mesh.is_conforming()
    assert set(mesh.boundary_group_names) == {"wall", "bottom", "top"}
    # nothing tagged on the interior interface plane z=1
    z = _face_centroids(mesh)[:, 2]
    assert not np.any(np.isclose(z, 1.0))
    # the true outer caps are present and correctly placed
    assert np.all(np.isclose(z[mesh.boundary_names == "bottom"], 0.0))
    assert np.all(np.isclose(z[mesh.boundary_names == "top"], 2.0))
