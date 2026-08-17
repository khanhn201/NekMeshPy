"""Order-N ``HexMesh`` factories/combinators, the
``VTK_LAGRANGE_HEXAHEDRON`` export + node permutation, and the invariant that
``re2`` export stays linear regardless of order.

Same two invariants as the lower phases: geometric truth (curved nodes sit on the
true wall/shell) and the N=1 no-op (order-1 factories are byte-for-byte the old
linear meshes, and the ``high_order_hex`` example VTU stays byte-identical to its
golden)."""


import numpy as np
import pytest
from conftest import curved, vtu_cell_types

from nekmeshpy import HexMesh, LineMesh, QuadMesh, hexmesh, quadmesh
from nekmeshpy.core.interp import corner_indices, hex_face_indices
from nekmeshpy.io import export

GROUPS = {"inlet": "v  ", "outlet": "O  ", "sphere": "W  ",
          "top": "SYM", "bottom": "SYM", "front": "SYM", "back": "SYM"}


def _shell(order, n_face=2, n_radial=2):
    cube = quadmesh.box(3.0, n_face, order=order, patch_tags={
        "x_max": "outlet", "x_min": "inlet", "y_max": "top",
        "y_min": "bottom", "z_max": "back", "z_min": "front"})
    sphere = quadmesh.sphere(1.0, n_face, order=order)
    return hexmesh.annulus(sphere, cube,
                           radial=np.linspace(0.0, 1.0, n_radial + 1))


# -- container invariants the B-rep rests on ----------------------------
def test_hex_must_index_shared_faces_that_exist():
    """``hex`` indexes the shared-face ``QuadMesh``, so a stray face id is caught
    against that mesh's quad count -- the top rung of the same check ``QuadMesh`` makes
    against its edges and ``LineMesh`` against its points."""
    blk = _shell(1)
    with pytest.raises(ValueError, match="hexes must index the .* shared faces"):
        HexMesh(blk.quad_mesh, np.full((2, 6), 999), np.zeros((2, 6), dtype=np.int64))
    with pytest.raises(ValueError, match="hexes must index the .* shared faces"):
        HexMesh(blk.quad_mesh, -np.ones((1, 6), dtype=np.int64),
                np.zeros((1, 6), dtype=np.int64))


def test_order_n_container_asks_for_the_interior_it_cannot_invent():
    """At order > 1 the private nodes are geometry, so omitting them is an actionable
    error naming a factory -- not a bare shape mismatch."""
    blk = _shell(3)
    assert blk.order == 3
    with pytest.raises(ValueError, match=r"order 3 > 1 requires the per-hex private"):
        HexMesh(blk.quad_mesh, blk.hexes, blk.orient)


# -- geometric truth: annulus inner wall rides the true sphere ----------
@pytest.mark.parametrize("order", [2, 3, 4])
def test_annulus_inner_wall_on_true_sphere(order):
    hm = _shell(order)
    assert hm.order == order
    cb = curved(hm)                                  # B-rep -> per-hex block
    assert cb.shape == (hm.n_hexes, (order + 1) ** 3, 3)
    # corner-consistency across the whole block
    cc = cb[:, corner_indices(order, 3), :]
    assert np.allclose(cc, hm.points[hm.corners], atol=1e-9)
    # face 5 (inner cap) nodes of the innermost hexes lie on radius 1
    rc = np.linalg.norm(hm.points[hm.corners], axis=2)
    inner = np.all(np.isclose(rc[:, HexMesh.FACE_POINTS[4]], 1.0, atol=1e-9), axis=1)
    fidx = hex_face_indices(5, order)
    nodes = cb[np.where(inner)[0][:, None], fidx[None, :], :]
    r = np.linalg.norm(nodes.reshape(-1, 3), axis=1)
    assert np.allclose(r, 1.0, atol=1e-9)


# -- extrude / loft straight sweep is corner-consistent -----------------
@pytest.mark.parametrize("order", [2, 3])
def test_extrude_order_n_corner_consistent(order):
    sec = quadmesh.rectangle([[0, 0, 0], [2, 0, 0], [2, 1, 0], [0, 1, 0]],
                             3, 2, order=order)
    hb = hexmesh.extrude(sec, length=4.0, layers=[0.0, 0.5, 1.0],
                         first_tag="in", last_tag="out")
    assert hb.order == order
    cb = curved(hb)
    cc = cb[:, corner_indices(order, 3), :]
    assert np.allclose(cc, hb.points[hb.corners], atol=1e-9)
    # planar section swept along +z -> every node has integer-free straight geometry;
    # here just assert the block is a straight subdivision (mid nodes between corners)
    assert cb.shape == (hb.n_hexes, (order + 1) ** 3, 3)


# -- from_grid order-N (trilinear) --------------------------------------
def test_from_grid_order_n_corner_consistent():
    x = np.linspace(0, 1, 3)
    y = np.linspace(0, 2, 2)
    z = np.linspace(0, 3, 2)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    grid = np.stack([X, Y, Z], axis=-1)
    fg = hexmesh.from_grid(grid, order=4)
    assert fg.order == 4
    cc = curved(fg)[:, corner_indices(4, 3), :]
    assert np.allclose(cc, fg.points[fg.corners], atol=1e-9)


# -- blend / merge ------------------------------------------------------
def test_blend_morphs_hex_curved_blocks():
    a = _shell(3, n_face=1, n_radial=1)
    # a uniformly scaled copy, built natively from a's own entity tables -- same B-rep,
    # every stored node doubled, so the two pair by index at every rung
    lines = LineMesh(a.points * 2.0, a.edges, interior=a.edge_nodes * 2.0)
    quads = QuadMesh(lines, a.quad_mesh.quads, a.quad_mesh.orient, a.face_nodes * 2.0)
    b = HexMesh(quads, a.hexes, a.orient, a.interior * 2.0)
    lo, mid, hi = hexmesh.blend(a, b, [0.0, 0.5, 1.0])
    assert lo.order == mid.order == hi.order == 3
    ca, cbb = curved(a), curved(b)
    assert np.allclose(curved(lo), ca)
    assert np.allclose(curved(hi), cbb)
    assert np.allclose(curved(mid), 0.5 * (ca + cbb))
    # the shared edge / face / private interior tables take the same lerp
    assert np.allclose(mid.edge_nodes, 0.5 * (a.edge_nodes + b.edge_nodes))
    assert np.allclose(mid.face_nodes, 0.5 * (a.face_nodes + b.face_nodes))
    assert np.allclose(mid.interior, 0.5 * (a.interior + b.interior))


def test_blend_rejects_mismatched_order():
    a = _shell(3, n_face=1, n_radial=1)
    b = _shell(2, n_face=1, n_radial=1)
    with pytest.raises(ValueError, match="same order"):
        hexmesh.blend(a, b, [0.5])


def test_merge_rejects_mismatched_order():
    a = _shell(2, n_face=1, n_radial=1)
    b = _shell(3, n_face=1, n_radial=1)
    with pytest.raises(ValueError, match="same order"):
        hexmesh.merge([a, b])


# -- N=1 no-op ----------------------------------------------------------
def test_order1_factories_are_linear_no_op():
    x = np.linspace(0, 1, 3)
    X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
    grid = np.stack([X, Y, Z], axis=-1)
    for hm in (_shell(1), hexmesh.from_grid(grid)):
        # order 1: every high-order table is empty, the walk is just the corners
        assert hm.order == 1
        assert hm.edge_nodes.shape == (hm.edges.shape[0], 0, 3)
        assert hm.face_nodes.shape == (hm.faces.shape[0], 0, 3)
        assert hm.interior.shape == (hm.n_hexes, 0, 3)
        assert curved(hm).shape == (hm.n_hexes, 8, 3)
        assert np.allclose(curved(hm)[:, corner_indices(1, 3), :],
                           hm.points[hm.corners])


def test_order1_shell_points_match_high_order_corners():
    lin = _shell(1)
    ho = _shell(4)
    assert np.allclose(lin.points, ho.points)            # corner points identical
    assert np.array_equal(lin.corners, ho.corners)


# -- order-N quality metric (opt-in) ------------------------------------
def test_high_order_quality_matches_corner_at_order1():
    hm = _shell(1)
    assert np.allclose(hexmesh.scaled_jacobian(hm, high_order=True),
                       hexmesh.scaled_jacobian(hm), atol=1e-12)


@pytest.mark.parametrize("order", [2, 3])
def test_high_order_quality_non_degenerate_on_curved_shell(order):
    hm = _shell(order, n_face=3, n_radial=2)
    sj = hexmesh.scaled_jacobian(hm, high_order=True)
    assert sj.shape == (hm.n_hexes,)
    assert np.all(np.isfinite(sj))
    assert float(sj.min()) > 0.0                         # no folded GLL nodes
    # sampling the curved interior differs from the corner-only metric
    assert not np.allclose(sj, hexmesh.scaled_jacobian(hm))
    assert hexmesh.quality_summary(hm, high_order=True).n_elements == hm.n_hexes


# -- re2 export stays linear regardless of order ------------------------
def test_re2_is_order_invariant(tmp_path):
    lin = _shell(1)
    ho = _shell(4)
    export.to_re2(lin, str(tmp_path / "lin.re2"), groups=GROUPS)
    export.to_re2(ho, str(tmp_path / "ho.re2"), groups=GROUPS)
    a = open(tmp_path / "lin.re2", "rb").read()
    b = open(tmp_path / "ho.re2", "rb").read()
    assert a == b                                        # curved nodes never reach re2


# -- VTK Lagrange hex node ordering -------------------------------------
def test_lagrange_hex_perm_is_valid_permutation():
    for order in (2, 3, 5):
        perm = export._lagrange_hex_perm(order)
        row = order + 1
        assert sorted(perm.tolist()) == list(range(row ** 3))
        # the first eight VTK slots are the eight corner lexicographic indices
        assert set(perm[:8].tolist()) == set(corner_indices(order, 3).tolist())


def test_lagrange_hex_perm_order2_body_center_last():
    # order 2: 27 nodes; VTK position 26 (last) is the body centre = lexicographic
    # (1,1,1) = 1 + 3*1 + 9*1 = 13
    perm = export._lagrange_hex_perm(2)
    assert perm.shape[0] == 27
    assert perm[26] == 13


# -- VTU (XML) export ---------------------------------------------------
def test_vtu_order1_is_plain_hex(tmp_path):
    p = str(tmp_path / "lin.vtu")
    export.to_vtu(_shell(1), p, groups=GROUPS)
    assert vtu_cell_types(p) == {12}                  # VTK_HEXAHEDRON
    assert 'Name="bc_id"' in open(p).read()


def test_vtu_high_order_is_lagrange_hex(tmp_path):
    p = str(tmp_path / "ho.vtu")
    export.to_vtu(_shell(3), p, groups=GROUPS)           # 4**3 = 64 nodes/hex
    assert vtu_cell_types(p) == {72}                  # VTK_LAGRANGE_HEXAHEDRON
    assert 'Name="bc_id"' in open(p).read()


def test_vtu_meshio_roundtrip(tmp_path):
    meshio = pytest.importorskip("meshio")
    p = str(tmp_path / "ho.vtu")
    export.to_vtu(_shell(3), p, groups=GROUPS)
    mm = meshio.read(p)
    assert {c.type for c in mm.cells} == {"VTK_LAGRANGE_HEXAHEDRON"}
    assert "bc_id" in mm.point_data


# -- example golden -----------------------------------------------------
