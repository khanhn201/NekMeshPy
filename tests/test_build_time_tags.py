"""Build-time boundary tagging: per-edge names on a QuadMesh section propagate to
the swept side faces (HexMesh.loft/extrude), NO_BOUNDARY suppresses a face, and a
NO_BOUNDARY seam lets HexMesh.merge stay a plain concatenate with no stale tag on
the welded-away interior face."""

from collections import Counter

import numpy as np
import pytest

from nekmeshpy import NO_BOUNDARY, HexMesh, QuadMesh
from nekmeshpy.model.fields import uniform_spacing

# a unit square, one quad, CCW: side 1 (0,1) bottom, 2 (1,2) right, 3 (2,3) top,
# 4 (3,0) left -- boundaries are [quad id, side], parallel with boundary_tags.
_SQUARE_PTS = [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]]
_SQUARE_QUADS = [[0, 1, 2, 3]]
_SQUARE_BND = [[0, 1], [0, 2], [0, 3], [0, 4]]
_SQUARE_BND_TAGS = ["bottom", "right", "top", "left"]


def _face_centroids(mesh):
    """(K,3) centroid of every tagged boundary face."""
    out = []
    for e, f in mesh.boundaries:
        out.append(mesh.points[mesh.hexes[e, HexMesh.FACE_POINTS[f - 1]]].mean(axis=0))
    return np.array(out).reshape(-1, 3)


def _edge_points(mesh, row):
    """Point ids of tagged boundary edge ``row`` (via [quad, side])."""
    q, s = int(mesh.boundaries[row, 0]), int(mesh.boundaries[row, 1])
    return mesh.quads[q, QuadMesh.EDGE_POINTS[s - 1]]


def test_boundaries_stored_as_quad_side_parallel_with_names():
    qm = QuadMesh(_SQUARE_PTS, _SQUARE_QUADS,
                  boundaries=_SQUARE_BND, boundary_tags=_SQUARE_BND_TAGS)
    assert qm.boundaries.shape == (4, 2)
    assert qm.n_boundaries == 4
    assert qm.boundary_group_tags == ["bottom", "left", "right", "top"]
    # side 1 (pts 0-1) is named "bottom"
    row = next(r for r in range(4) if qm.boundaries[r].tolist() == [0, 1])
    assert qm.boundary_tags[row] == "bottom"
    assert np.allclose(qm.points[_edge_points(qm, row), 1], 0.0)   # bottom -> y=0


def test_mismatched_boundaries_and_names_raises():
    with pytest.raises(ValueError, match="must match"):
        QuadMesh(_SQUARE_PTS, _SQUARE_QUADS, boundaries=_SQUARE_BND,
                 boundary_tags=["only", "two"])


def test_loft_propagates_per_edge_boundary_tags_to_side_faces():
    qm = QuadMesh(_SQUARE_PTS, _SQUARE_QUADS,
                  boundaries=_SQUARE_BND, boundary_tags=_SQUARE_BND_TAGS)
    blk = HexMesh.extrude(qm, length=1.0, layers=uniform_spacing(1),
                          first_tag="inlet", last_tag="outlet")
    counts = Counter(blk.boundary_tags.tolist())
    # 4 named sides + 2 caps, each once (one hex)
    assert counts == {"bottom": 1, "right": 1, "top": 1, "left": 1,
                      "inlet": 1, "outlet": 1}
    assert blk.is_watertight()


def test_no_boundary_suppresses_a_swept_face():
    # name three edges "wall"; the right edge (side 2) is declared NO_BOUNDARY
    qm = QuadMesh(_SQUARE_PTS, _SQUARE_QUADS,
                  boundaries=[[0, 1], [0, 3], [0, 4], [0, 2]],
                  boundary_tags=["wall", "wall", "wall", NO_BOUNDARY])
    blk = HexMesh.extrude(qm, length=1.0, layers=uniform_spacing(2))
    names = blk.boundary_tags.tolist()
    assert "" not in names                       # NO_BOUNDARY never emitted
    # 3 walls x 2 layers = 6 wall faces; the 4th (right) edge is suppressed
    assert Counter(names)["wall"] == 6


def test_untagged_section_tags_only_caps():
    # a section with no tagged boundaries -> only the caps are named
    qm = QuadMesh(_SQUARE_PTS, _SQUARE_QUADS)
    blk = HexMesh.extrude(qm, length=1.0, layers=uniform_spacing(1),
                          first_tag="inlet", last_tag="outlet")
    assert Counter(blk.boundary_tags.tolist()) == {"inlet": 1, "outlet": 1}


def test_quadmesh_merge_concats_and_offsets_boundary_tags():
    a = QuadMesh([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], _SQUARE_QUADS,
                 boundaries=[[0, 1]], boundary_tags=["a_bottom"])
    b = QuadMesh([[1, 0, 0], [2, 0, 0], [2, 1, 0], [1, 1, 0]], _SQUARE_QUADS,
                 boundaries=[[0, 1]], boundary_tags=["b_bottom"])
    m = QuadMesh.merge([a, b])
    assert set(m.boundary_tags.tolist()) == {"a_bottom", "b_bottom"}
    # each name still sits on a y=0 edge after the point weld + quad-id offset
    for r in range(m.n_boundaries):
        assert np.allclose(m.points[_edge_points(m, r), 1], 0.0)


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
    assert set(mesh.boundary_group_tags) == {"wall", "bottom", "top"}
    # nothing tagged on the interior interface plane z=1
    z = _face_centroids(mesh)[:, 2]
    assert not np.any(np.isclose(z, 1.0))
    # the true outer caps are present and correctly placed
    assert np.all(np.isclose(z[mesh.boundary_tags == "bottom"], 0.0))
    assert np.all(np.isclose(z[mesh.boundary_tags == "top"], 2.0))
