"""Build-time boundary tagging: per-edge names on a QuadMesh section propagate to
the swept side faces (HexMesh.loft/extrude), NO_TAG suppresses a face, and a
NO_TAG seam lets HexMesh.merge stay a plain concatenate with no stale tag on
the welded-away interior face."""

from collections import Counter

import numpy as np
import pytest

from nekmeshpy import NO_TAG, EdgeTags, HexMesh, QuadMesh, hexmesh, quadmesh
from nekmeshpy.model.fields import uniform_spacing

# a unit square, one quad, CCW: side 1 (0,1) bottom, 2 (1,2) right, 3 (2,3) top,
# 4 (3,0) left -- edge tags are (quad id, side, tag) rows.
_SQUARE_PTS = [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]]
_SQUARE_QUADS = [[0, 1, 2, 3]]
_SQUARE_BND = [[0, 1], [0, 2], [0, 3], [0, 4]]
_SQUARE_BND_TAGS = ["bottom", "right", "top", "left"]


def _face_centroids(mesh):
    """(K,3) centroid of every tagged face."""
    out = []
    for e, f, _tag in mesh.face_tags:
        out.append(mesh.points[mesh.hexes[e, HexMesh.FACE_POINTS[f - 1]]].mean(axis=0))
    return np.array(out).reshape(-1, 3)


def _edge_points(mesh, row):
    """Point ids of tagged edge ``row`` (via [quad, side])."""
    q, s = int(mesh.edge_tags.elements[row]), int(mesh.edge_tags.sides[row])
    return mesh.quads[q, QuadMesh.EDGE_POINTS[s - 1]]


def test_edge_tags_stored_as_quad_side_rows_with_names():
    qm = QuadMesh.from_corners(_SQUARE_PTS, _SQUARE_QUADS,
                  edge_tags=EdgeTags.from_pairs(_SQUARE_BND, _SQUARE_BND_TAGS))
    assert qm.edge_tags.rows.shape == (4, 2)
    assert qm.n_edge_tags == 4
    assert qm.edge_group_tags == ["bottom", "left", "right", "top"]
    # side 1 (pts 0-1) is named "bottom"
    row = next(r for r in range(4) if list(qm.edge_tags.rows[r]) == [0, 1])
    assert qm.edge_tags.tags[row] == "bottom"
    assert np.allclose(qm.points[_edge_points(qm, row), 1], 0.0)   # bottom -> y=0


def test_mismatched_rows_and_names_raises():
    """Desynchronized rows and names are rejected by the table, not the container."""
    with pytest.raises(ValueError, match="same length"):
        EdgeTags.from_pairs(_SQUARE_BND, ["only", "two"])


def test_loft_propagates_per_edge_tags_to_side_faces():
    qm = QuadMesh.from_corners(_SQUARE_PTS, _SQUARE_QUADS,
                  edge_tags=EdgeTags.from_pairs(_SQUARE_BND, _SQUARE_BND_TAGS))
    blk = hexmesh.extrude(qm, length=1.0, layers=uniform_spacing(1),
                          first_tag="inlet", last_tag="outlet")
    counts = Counter(blk.face_tags.tags.tolist())
    # 4 named sides + 2 caps, each once (one hex)
    assert counts == {"bottom": 1, "right": 1, "top": 1, "left": 1,
                      "inlet": 1, "outlet": 1}
    assert hexmesh.is_watertight(blk)


def test_no_boundary_suppresses_a_swept_face():
    # name three edges "wall"; the right edge (side 2) is declared NO_TAG
    qm = QuadMesh.from_corners(_SQUARE_PTS, _SQUARE_QUADS,
                  edge_tags=EdgeTags.from_pairs([[0, 1], [0, 3], [0, 4], [0, 2]], ["wall", "wall", "wall", NO_TAG]))
    blk = hexmesh.extrude(qm, length=1.0, layers=uniform_spacing(2))
    names = blk.face_tags.tags.tolist()
    assert "" not in names                       # NO_TAG never emitted
    # 3 walls x 2 layers = 6 wall faces; the 4th (right) edge is suppressed
    assert Counter(names)["wall"] == 6


def test_untagged_section_tags_only_caps():
    # a section with no tagged edges -> only the caps are named
    qm = QuadMesh.from_corners(_SQUARE_PTS, _SQUARE_QUADS)
    blk = hexmesh.extrude(qm, length=1.0, layers=uniform_spacing(1),
                          first_tag="inlet", last_tag="outlet")
    assert Counter(blk.face_tags.tags.tolist()) == {"inlet": 1, "outlet": 1}


def test_quadmesh_merge_concats_and_offsets_edge_tags():
    a = QuadMesh.from_corners([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], _SQUARE_QUADS,
                 edge_tags=EdgeTags.from_pairs([[0, 1]], ["a_bottom"]))
    b = QuadMesh.from_corners([[1, 0, 0], [2, 0, 0], [2, 1, 0], [1, 1, 0]], _SQUARE_QUADS,
                 edge_tags=EdgeTags.from_pairs([[0, 1]], ["b_bottom"]))
    m = quadmesh.merge([a, b])
    assert set(m.edge_tags.tags.tolist()) == {"a_bottom", "b_bottom"}
    # each name still sits on a y=0 edge after the point weld + quad-id offset
    for r in range(m.n_edge_tags):
        assert np.allclose(m.points[_edge_points(m, r), 1], 0.0)


def _box(lo, hi, tags):
    xs = np.linspace(lo[0], hi[0], 2)
    ys = np.linspace(lo[1], hi[1], 2)
    zs = np.linspace(lo[2], hi[2], 2)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    return hexmesh.from_grid(np.stack([X, Y, Z], axis=-1), side_tags=tags)


def test_no_boundary_seam_keeps_merge_a_plain_concatenate():
    """Two stacked boxes: tag every outer side, declare the touching faces
    NO_TAG, and merge -- no tag lands on the welded interior interface."""
    sides = {"x_min": "wall", "x_max": "wall", "y_min": "wall", "y_max": "wall"}
    lower = _box((0, 0, 0), (1, 1, 1), {**sides, "z_min": "bottom", "z_max": NO_TAG})
    upper = _box((0, 0, 1), (1, 1, 2), {**sides, "z_min": NO_TAG, "z_max": "top"})
    mesh = hexmesh.merge([lower, upper])

    assert hexmesh.is_watertight(mesh) and hexmesh.is_conforming(mesh)
    assert set(mesh.face_group_tags) == {"wall", "bottom", "top"}
    # nothing tagged on the interior interface plane z=1
    z = _face_centroids(mesh)[:, 2]
    assert not np.any(np.isclose(z, 1.0))
    # the true outer caps are present and correctly placed
    assert np.all(np.isclose(z[mesh.face_tags.tags == "bottom"], 0.0))
    assert np.all(np.isclose(z[mesh.face_tags.tags == "top"], 2.0))
