"""Build-time boundary tagging: per-edge names on a QuadMesh section propagate to
the swept side faces (HexMesh.loft/extrude), NO_TAG suppresses a face, and a
NO_TAG seam lets HexMesh.merge stay a plain concatenate with no stale tag on
the welded-away interior face."""

from collections import Counter

import numpy as np
import pytest
from conftest import face_rows

from nekmeshpy import NO_TAG, ElementTags, HexMesh, QuadMesh, hexmesh, quadmesh
from nekmeshpy.core.fields import uniform_spacing
from nekmeshpy.quadmesh.tag import tag_edges

# a unit square, one quad, CCW: side 1 (0,1) bottom, 2 (1,2) right, 3 (2,3) top,
# 4 (3,0) left -- edge tags are (quad id, side, tag) rows.
_SQUARE_PTS = [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]]
_SQUARE_QUADS = [[0, 1, 2, 3]]
_SQUARE_BND = [[0, 1], [0, 2], [0, 3], [0, 4]]
_SQUARE_BND_TAGS = ["bottom", "right", "top", "left"]


def _face_centroids(mesh):
    """(K,3) centroid of every tagged face."""
    out = []
    for e, f, _tag in face_rows(mesh):
        out.append(mesh.points[mesh.hexes[e, HexMesh.FACE_POINTS[f - 1]]].mean(axis=0))
    return np.array(out).reshape(-1, 3)


def _edge_points(mesh, row):
    """Point ids of tagged edge ``row`` (via [quad, side])."""
    q, s = int(mesh.edge_tags.elements[row]), int(mesh.edge_tags.sides[row])
    return mesh.quads[q, QuadMesh.EDGE_POINTS[s - 1]]


def _tagged_square():
    """The unit square with its four boundary edges named, authored the way a factory
    does -- in ``(quad, side)`` rows, which ``tag_edges`` resolves onto the shared
    edges those rows point at."""
    return tag_edges(QuadMesh.from_corners(_SQUARE_PTS, _SQUARE_QUADS),
                     _SQUARE_BND, _SQUARE_BND_TAGS)


def test_edge_tags_are_stored_on_the_shared_edges_they_name():
    qm = _tagged_square()
    assert qm.n_edge_tags == 4
    assert qm.edge_group_tags == ["bottom", "left", "right", "top"]
    # the tag rides an edge id, so the geometry is read straight off ``lines``
    named = dict(qm.edge_tags)
    eid = next(e for e, t in named.items() if t == "bottom")
    assert np.allclose(qm.points[qm.line_mesh.lines[eid], 1], 0.0)     # bottom -> y=0


def test_mismatched_ids_and_names_raises():
    """Desynchronized ids and names are rejected by the table, not the container."""
    with pytest.raises(ValueError, match="same length"):
        ElementTags([0, 1], ["only"])


def test_loft_propagates_per_edge_tags_to_side_faces():
    qm = _tagged_square()
    blk = hexmesh.extrude(qm, length=1.0, layers=uniform_spacing(1),
                          first_tag="inlet", last_tag="outlet")
    counts = Counter(blk.face_tags.tags.tolist())
    # 4 named sides + 2 caps, each once (one hex)
    assert counts == {"bottom": 1, "right": 1, "top": 1, "left": 1,
                      "inlet": 1, "outlet": 1}
    assert hexmesh.is_watertight(blk)


def test_no_boundary_suppresses_a_swept_face():
    # name three edges "wall"; the right edge (side 2) is declared NO_TAG
    qm = tag_edges(QuadMesh.from_corners(_SQUARE_PTS, _SQUARE_QUADS),
                   [[0, 1], [0, 3], [0, 4], [0, 2]],
                   ["wall", "wall", "wall", NO_TAG])
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


def test_quadmesh_merge_carries_edge_tags_onto_the_merged_edges():
    a = tag_edges(QuadMesh.from_corners(
        [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], _SQUARE_QUADS),
        [[0, 1]], ["a_bottom"])
    b = tag_edges(QuadMesh.from_corners(
        [[1, 0, 0], [2, 0, 0], [2, 1, 0], [1, 1, 0]], _SQUARE_QUADS),
        [[0, 1]], ["b_bottom"])
    m = quadmesh.merge([a, b])
    assert set(m.edge_tags.tags.tolist()) == {"a_bottom", "b_bottom"}
    # each name still sits on a y=0 edge after the point weld rebuilt the edge table
    for eid, _ in m.edge_tags:
        assert np.allclose(m.points[m.line_mesh.lines[eid], 1], 0.0)


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
    names = np.array([t for _, _, t in face_rows(mesh)])
    assert np.all(np.isclose(z[names == "bottom"], 0.0))
    assert np.all(np.isclose(z[names == "top"], 2.0))
