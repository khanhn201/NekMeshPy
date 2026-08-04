"""``HexMesh.tetra`` -- the curvilinear tetrahedron fill."""

import numpy as np
import pytest

from nekmeshpy import HexMesh, LineMesh, QuadMesh

V = np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
RADIAL = np.array([0.0, 0.45, 0.8, 1.0])
NR = RADIAL.size - 1
CS = 0.55


def _triangle(a, b, c, n=2, order=1, tag=""):
    """A flat triangle as three structured patches meeting at its centroid."""
    ctr = (a + b + c) / 3.0
    mab, mbc, mca = (a + b) / 2, (b + c) / 2, (c + a) / 2
    u = np.linspace(0.0, 1.0, n + 1)
    pats = []
    for v, e1, e2 in ((a, mab, mca), (b, mbc, mab), (c, mca, mbc)):
        g = ((1 - u)[:, None, None] * ((1 - u)[None, :, None] * v
                                       + u[None, :, None] * e2)
             + u[:, None, None] * ((1 - u)[None, :, None] * e1
                                   + u[None, :, None] * ctr))
        pats.append(QuadMesh.from_grid(g, element_tag=tag, order=order))
    return QuadMesh.merge(pats)


def _unit_tet(n=2, order=1, tags=("", "", "", ""), **kw):
    faces = [_triangle(V[0], V[1], V[2], n, order, tags[0]),
             _triangle(V[0], V[1], V[3], n, order, tags[1]),
             _triangle(V[0], V[2], V[3], n, order, tags[2]),
             _triangle(V[1], V[2], V[3], n, order, tags[3])]
    return HexMesh.tetra(faces, **kw)


@pytest.mark.parametrize("n", [1, 2, 3])
def test_counts_and_topology(n):
    t = _unit_tet(n)
    # four corner blocks, each n x n x n
    assert t.hexes.shape[0] == 4 * n ** 3
    rep = t.topology_report()
    assert rep.watertight and rep.conformal and rep.n_components == 1
    assert rep.n_open_edges == 0 and rep.n_hanging_points == 0
    assert float(np.min(t.scaled_jacobian())) > 0.0


def test_face_order_does_not_matter():
    """The faces are matched by geometry, so any permutation gives the same mesh."""
    a = _unit_tet(2)
    faces = [_triangle(V[0], V[1], V[2], 2), _triangle(V[0], V[1], V[3], 2),
             _triangle(V[0], V[2], V[3], 2), _triangle(V[1], V[2], V[3], 2)]
    b = HexMesh.tetra([faces[2], faces[0], faces[3], faces[1]])
    assert b.hexes.shape[0] == a.hexes.shape[0]
    assert np.allclose(np.sort(b.points, axis=0), np.sort(a.points, axis=0),
                       atol=1e-12)


def _boundary_nodes(mesh):
    """Corner ids of every topological boundary face."""
    hexes = mesh.hexes
    seen: dict[tuple[int, ...], list[tuple[int, int]]] = {}
    for e in range(hexes.shape[0]):
        for s in range(6):
            k = tuple(sorted(int(v) for v in hexes[e][HexMesh.FACE_POINTS[s]]))
            seen.setdefault(k, []).append((e, s))
    return np.unique(np.concatenate(
        [np.array(k) for k, v in seen.items() if len(v) == 1]))


def test_boundary_is_exactly_the_four_faces():
    """Every boundary node lies on one of the tetrahedron's four planes, and every
    interior node strictly inside -- the fill neither leaks out nor leaves a gap."""
    t = _unit_tet(3)
    p = t.points
    bnd = _boundary_nodes(t)
    on = np.zeros(p.shape[0], dtype=bool)
    for k in range(3):
        on |= np.abs(p[:, k]) < 1e-12
    on |= np.abs(p.sum(axis=1) - 1.0) < 1e-12
    assert np.all(on[bnd])
    interior = np.setdiff1d(np.arange(p.shape[0]), bnd)
    assert interior.size > 0
    assert not np.any(on[interior])


@pytest.mark.parametrize("order", [1, 2, 3])
def test_flat_tet_stays_flat_at_order_n(order):
    """A straight-sided tetrahedron has every node -- high order included -- on one of
    its four planes or strictly inside, never bulging through a face."""
    t = _unit_tet(2, order=order)
    assert t.order == order
    p = t.points
    assert p[:, 0].min() > -1e-12 and p[:, 1].min() > -1e-12
    assert p[:, 2].min() > -1e-12 and p.sum(axis=1).max() < 1.0 + 1e-12
    assert float(np.min(t.scaled_jacobian(high_order=True))) > 0.0


def test_tags_follow_their_face():
    """Each face's ``element_tags`` name the boundary faces it becomes -- and follow
    the face itself, not the order it was handed in."""
    t = _unit_tet(2, tags=("bottom", "front", "left", "slant"))
    names = set(t.face_group_tags)
    assert names == {"bottom", "front", "left", "slant"}
    counts = {n: t.face_tags.count(n) for n in names}
    assert set(counts.values()) == {3 * 2 * 2}          # 3 patches of n x n each
    # a tagged face's boundary rows must be genuine boundary faces
    hexes = t.hexes
    key: dict[tuple[int, ...], int] = {}
    for e in range(hexes.shape[0]):
        for s in range(6):
            k = tuple(sorted(int(v) for v in hexes[e][HexMesh.FACE_POINTS[s]]))
            key[k] = key.get(k, 0) + 1
    for e, s, _tag in t.face_tags:
        k = tuple(sorted(int(v) for v in hexes[e][HexMesh.FACE_POINTS[s - 1]]))
        assert key[k] == 1


def test_center_override_moves_only_the_interior():
    """``center`` repositions the shared far corner; the four faces are untouched."""
    a = _unit_tet(2)
    b = _unit_tet(2, center=(0.3, 0.3, 0.3))
    assert a.hexes.shape == b.hexes.shape
    face = np.abs(a.points.sum(axis=1) - 1.0) < 1e-12
    assert np.allclose(np.sort(a.points[face], axis=0),
                       np.sort(b.points[face], axis=0), atol=1e-12)
    assert not np.allclose(a.points, b.points, atol=1e-9)


def test_rejects_a_face_that_is_not_a_three_patch_triangle():
    grid = np.zeros((3, 3, 3))
    grid[..., 0] = np.arange(3)[:, None]
    grid[..., 1] = np.arange(3)[None, :]
    square = QuadMesh.from_grid(grid)
    faces = [square] + [_triangle(V[0], V[1], V[2], 2)] * 3
    with pytest.raises(ValueError, match="three structured patches"):
        HexMesh.tetra(faces)


def test_rejects_wrong_face_count_and_mixed_order():
    with pytest.raises(ValueError, match="exactly 4 faces"):
        HexMesh.tetra([_triangle(V[0], V[1], V[2], 2)] * 3)
    faces = [_triangle(V[0], V[1], V[2], 2), _triangle(V[0], V[1], V[3], 2),
             _triangle(V[0], V[2], V[3], 2), _triangle(V[1], V[2], V[3], 2, order=2)]
    with pytest.raises(ValueError, match="share an order"):
        HexMesh.tetra(faces)


def test_rejects_faces_that_do_not_close():
    """Four triangles that do not share six edges are not a tetrahedron."""
    far = V[3] + np.array([0.0, 0.0, 5.0])
    faces = [_triangle(V[0], V[1], V[2], 2), _triangle(V[0], V[1], V[3], 2),
             _triangle(V[0], V[2], V[3], 2), _triangle(V[1], V[2], far, 2)]
    with pytest.raises(ValueError, match="4 corners|share their six edges"):
        HexMesh.tetra(faces)


def test_quadrant_faces_make_an_octant():
    """The junction case: three ``quadrant_ogrid`` faces plus a wall patch.

    Each quadrant face is already a three-patch triangle -- core plus the two halves
    of its ring band -- so the octant of a 3-D O-grid falls out of ``tetra`` with the
    block split the faces already carry: an ``n**3`` core and three ``n x n x Nradial``
    slabs."""
    n = 2
    fr = QuadMesh.quadrant_seam_fractions(n, RADIAL, CS)
    e = np.eye(3)
    seams = [LineMesh.line(np.zeros(3), d, fr) for d in e]

    def great(a, b):
        """The quarter great-circle from ``a`` to ``b``, stated in the caller's own
        basis so its two ends are the seams' ends."""
        return LineMesh.loft_curve(
            lambda t: (np.cos(t)[:, None] * a + np.sin(t)[:, None] * b),
            np.linspace(0.0, np.pi / 2, 2 * n + 1))

    arcs = [great(e[0], e[1]), great(e[1], e[2]), great(e[2], e[0])]
    quads = [QuadMesh.quadrant_ogrid(arcs[0], seams[0], seams[1], RADIAL,
                                     center_scale=CS),
             QuadMesh.quadrant_ogrid(arcs[1], seams[1], seams[2], RADIAL,
                                     center_scale=CS),
             QuadMesh.quadrant_ogrid(arcs[2], seams[2], seams[0], RADIAL,
                                     center_scale=CS)]
    # the spherical side, as three patches about the octant's centre direction
    ctr = np.ones(3) / np.sqrt(3.0)
    mids = [a.points[n] for a in arcs]
    pats = []
    u = np.linspace(0.0, 1.0, n + 1)
    for v, m1, m2 in ((arcs[0].points[0], mids[0], mids[2]),
                      (arcs[1].points[0], mids[1], mids[0]),
                      (arcs[2].points[0], mids[2], mids[1])):
        g = ((1 - u)[:, None, None] * ((1 - u)[None, :, None] * v
                                       + u[None, :, None] * m2)
             + u[:, None, None] * ((1 - u)[None, :, None] * m1
                                   + u[None, :, None] * ctr))
        g /= np.linalg.norm(g, axis=-1, keepdims=True)          # onto the sphere
        pats.append(QuadMesh.from_grid(g, element_tag="sphere"))
    t = HexMesh.tetra(quads + [QuadMesh.merge(pats)])
    assert t.hexes.shape[0] == n ** 3 + 3 * (n * n * NR)
    rep = t.topology_report()
    assert rep.watertight and rep.conformal and rep.n_components == 1
    assert float(np.min(t.scaled_jacobian())) > 0.0
    assert set(t.face_group_tags) == {"sphere"}
    # the tagged side really is on the sphere; the three quadrant sides are the flat
    # cuts through the ball, so they are not (and must not be).
    ids = np.unique(np.concatenate(
        [t.hexes[e][list(HexMesh.FACE_POINTS[s - 1])]
         for e, s, tag in t.face_tags if tag == "sphere"]))
    assert np.allclose(np.linalg.norm(t.points[ids], axis=1), 1.0, atol=1e-12)
    flat = np.setdiff1d(_boundary_nodes(t), ids)
    assert flat.size > 0
    assert np.all(np.min(np.abs(t.points[flat]), axis=1) < 1e-12)
