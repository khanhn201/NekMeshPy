"""Unit tests for the 3-D shell factory :meth:`HexMesh.annulus` (the sibling of
:meth:`QuadMesh.annulus` one dimension up), the supporting
:meth:`QuadMesh.from_grid`, and the per-quad cap generalization of
:meth:`HexMesh.loft`.  ``annulus`` fills the region between two index-paired closed
quad surfaces and tags the inner / outer wall faces from the surfaces' per-quad
``element_tags`` (a closed surface has no free boundary edges)."""

import numpy as np
import pytest

from nekmeshpy import ElementTags, HexMesh, QuadMesh, hexmesh, quadmesh
from nekmeshpy.core.fields import uniform_spacing

# the six cube faces: outward normal n with right-handed tangents (u x v = n)
_FACES = [
    ((1, 0, 0), (0, 1, 0), (0, 0, 1)),
    ((-1, 0, 0), (0, 0, 1), (0, 1, 0)),
    ((0, 1, 0), (0, 0, 1), (1, 0, 0)),
    ((0, -1, 0), (1, 0, 0), (0, 0, 1)),
    ((0, 0, 1), (1, 0, 0), (0, 1, 0)),
    ((0, 0, -1), (0, 1, 0), (1, 0, 0)),
]
_SIDE = {(1, 0, 0): "xp", (-1, 0, 0): "xm", (0, 1, 0): "yp",
         (0, -1, 0): "ym", (0, 0, 1): "zp", (0, 0, -1): "zm"}


def _cube_surface(half, nf, *, tag_faces=True):
    """A closed axis-aligned cube surface (Chebyshev radius ``half``) as a merged
    quad mesh; each face optionally element-tagged with its side name."""
    ab = np.linspace(-1.0, 1.0, nf + 1)
    a, b = np.meshgrid(ab, ab, indexing="ij")
    patches = []
    for nrm, u, v in _FACES:
        n, u, v = (np.asarray(x, float) for x in (nrm, u, v))
        face = half * (n + a[..., None] * u + b[..., None] * v)
        patches.append(quadmesh.from_grid(
            face, element_tag=_SIDE[nrm] if tag_faces else ""))
    return quadmesh.merge(patches)


def _face_pts(mesh, e, f):
    return mesh.points[mesh.hexes[e, HexMesh.FACE_POINTS[f - 1]]]


# -- QuadMesh.from_grid ------------------------------------------------------

def test_quad_from_grid_connectivity_winding_and_element_tag():
    xs, ys = np.linspace(0.0, 2.0, 3), np.linspace(0.0, 1.0, 2)
    X, Y = np.meshgrid(xs, ys, indexing="ij")               # (3,2)
    P = np.stack([X, Y, np.zeros_like(X)], axis=-1)         # ni=2, nj=1 -> 2 quads
    qm = quadmesh.from_grid(P, element_tag="patch")
    assert qm.n_quads == 2
    assert qm.element_group_tags == ["patch"]               # not clipped to "p"
    # every quad is wound CCW (outward normal +z)
    for q in qm.quads:
        p = qm.points[q]
        nrm = np.cross(p[1] - p[0], p[3] - p[0])
        assert nrm[2] > 0


def test_quad_from_grid_edge_tags_land_on_correct_sides():
    xs = ys = np.linspace(0.0, 1.0, 3)
    X, Y = np.meshgrid(xs, ys, indexing="ij")               # (3,3) -> 4 quads
    P = np.stack([X, Y, np.zeros_like(X)], axis=-1)
    qm = quadmesh.from_grid(P, side_tags={"x_min": "west", "y_max": "north"})
    assert qm.edge_group_tags == ["north", "west"]
    for eid, name in qm.edge_tags:
        mid = qm.points[qm.lines.lines[eid]].mean(axis=0)
        if name == "west":
            assert np.isclose(mid[0], 0.0)                  # x_min edge
        else:
            assert np.isclose(mid[1], 1.0)                  # y_max edge


# -- HexMesh.annulus ---------------------------------------------------------

def test_annulus_shell_watertight_and_tagged_from_element_tags():
    outer = _cube_surface(2.0, 2)                           # faces tagged xp/xm/...
    inner = QuadMesh.from_corners(0.5 * outer.points, outer.quads,       # inner cube, half=1.0
                     element_tags=ElementTags.uniform(outer.n_quads, "body"))
    mesh = hexmesh.annulus(inner, outer, uniform_spacing(3))

    assert hexmesh.is_watertight(mesh) and hexmesh.is_conforming(mesh)
    assert float(hexmesh.scaled_jacobian(mesh).min()) > 0.0
    assert set(mesh.face_group_tags) == {"body", "xp", "xm", "yp", "ym",
                                             "zp", "zm"}
    assert mesh.element_group_tags == []                    # hexes stay untagged
    # inner element_tags -> inner wall (Chebyshev radius 1.0); outer -> radius 2.0
    for r in range(mesh.n_face_tags):
        e, f = int(mesh.face_tags.elements[r]), int(mesh.face_tags.sides[r])
        cheb = float(np.max(np.abs(_face_pts(mesh, e, f))))
        expected = 1.0 if str(mesh.face_tags.tags[r]) == "body" else 2.0
        assert cheb == pytest.approx(expected)


def test_annulus_scalar_wall_tags_fallback():
    outer = _cube_surface(2.0, 1, tag_faces=False)          # untagged surfaces
    inner = QuadMesh.from_corners(0.5 * outer.points, outer.quads)
    mesh = hexmesh.annulus(inner, outer, uniform_spacing(2),
                           inner_tag="body", outer_tag="far")
    assert set(mesh.face_group_tags) == {"body", "far"}


def test_annulus_scalar_tag_overrides_surface_element_tags():
    # a non-empty scalar inner_tag / outer_tag OVERRIDES the surface's per-quad
    # element_tags for the whole wall (upper overrides lower)
    outer = _cube_surface(2.0, 2)                           # faces tagged xp/xm/...
    inner = QuadMesh.from_corners(0.5 * outer.points, outer.quads,
                     element_tags=ElementTags.uniform(outer.n_quads, "body"))
    mesh = hexmesh.annulus(inner, outer, uniform_spacing(3),
                           inner_tag="cylinder", outer_tag="far")
    # the per-quad surface tags are gone; each wall is a single overridden group
    assert set(mesh.face_group_tags) == {"cylinder", "far"}


def test_annulus_rejects_mismatched_point_counts():
    a, b = _cube_surface(1.0, 1), _cube_surface(2.0, 2)
    with pytest.raises(ValueError, match="equal point counts"):
        hexmesh.annulus(a, b, uniform_spacing(2))


def test_annulus_rejects_mismatched_connectivity():
    outer = _cube_surface(2.0, 2)
    inner = QuadMesh.from_corners(0.5 * outer.points, outer.quads[::-1])  # same points, diff quads
    with pytest.raises(ValueError, match="identical quad connectivity"):
        hexmesh.annulus(inner, outer, uniform_spacing(2))


def test_annulus_rejects_touching_surfaces():
    outer = _cube_surface(2.0, 2)
    inner = QuadMesh.from_corners(outer.points.copy(), outer.quads)      # coincident with outer
    with pytest.raises(ValueError, match="touch or cross"):
        hexmesh.annulus(inner, outer, uniform_spacing(2))


# -- HexMesh.loft per-quad caps ----------------------------------------------

def _two_quad_slices():
    s0 = QuadMesh.from_corners([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [2, 0, 0], [2, 1, 0]],
                  [[0, 1, 2, 3], [1, 4, 5, 2]])
    s1 = QuadMesh.from_corners(s0.points + np.array([0.0, 0.0, 1.0]), s0.quads)
    return s0, s1


def test_loft_per_quad_first_tag_and_scalar_last_tag():
    s0, s1 = _two_quad_slices()
    block = hexmesh.loft([s0, s1], last_tag="top",
                         first_tag=ElementTags.from_dense(["capA", "capB"]))
    tag_at = {(e, f): t for e, f, t in block.face_tags}
    assert tag_at[(0, 5)] == "capA"        # per-quad bottom caps
    assert tag_at[(1, 5)] == "capB"
    assert tag_at[(0, 6)] == "top"         # scalar top cap on both columns
    assert tag_at[(1, 6)] == "top"


def test_loft_cap_tags_must_name_quads_the_section_has():
    s0, s1 = _two_quad_slices()
    with pytest.raises(ValueError, match="only 2 elements"):
        hexmesh.loft([s0, s1], first_tag=ElementTags([2], ["off_the_end"]))
    with pytest.raises(TypeError, match="cap tag must be"):
        hexmesh.loft([s0, s1], first_tag=["a", "b"])
