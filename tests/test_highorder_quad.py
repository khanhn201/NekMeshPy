"""Order-N ``QuadMesh`` factories/combinators and the
``VTK_LAGRANGE_QUADRILATERAL`` export.

Same two invariants as the line phase: geometric truth (curved nodes sit on the
true surface / wall) and the N=1 no-op (order-1 factories are byte-for-byte the old
linear meshes, and the ``high_order_quad`` example VTU stays byte-identical to its
golden)."""

import os

import numpy as np
import pytest
from conftest import GOLDEN, curved, run_example

from nekmeshpy import LineMesh, QuadMesh
from nekmeshpy.io import export
from nekmeshpy.model.interp import corner_indices, quad_edge_indices


# -- geometric truth: sphere nodes lie on the true sphere --------------
@pytest.mark.parametrize("order", [2, 3, 5])
def test_sphere_nodes_lie_on_the_true_sphere(order):
    r = 3.0
    s = QuadMesh.sphere(r, 3, order=order)
    assert s.order == order
    cb = curved(s)                                       # B-rep -> per-quad block
    assert cb.shape == (s.n_quads, (order + 1) ** 2, 3)
    radii = np.linalg.norm(cb.reshape(-1, 3), axis=1)
    assert np.allclose(radii, r, atol=1e-12)             # every node on the sphere
    # shared edge nodes and private interiors are on the sphere too
    assert np.allclose(np.linalg.norm(s.edge_nodes.reshape(-1, 3), axis=1), r,
                       atol=1e-12)
    assert np.allclose(np.linalg.norm(s.interior.reshape(-1, 3), axis=1), r,
                       atol=1e-12)
    # corner sub-slice coincides with the linear corner points
    corners = corner_indices(order, 2)
    assert np.allclose(cb[:, corners, :], s.points[s.quads])


@pytest.mark.parametrize("order", [2, 4])
def test_box_nodes_on_flat_faces(order):
    b = QuadMesh.box(1.0, 2, order=order)
    assert b.order == order
    cb = curved(b)
    assert cb.shape == (b.n_quads, (order + 1) ** 2, 3)
    # every node of a box lies on the |coord| == 1 cube surface (one face-normal
    # component is +-1 to machine precision)
    c = np.abs(cb)
    assert np.all(np.isclose(np.max(c, axis=2), 1.0, atol=1e-12))
    corners = corner_indices(order, 2)
    assert np.allclose(cb[:, corners, :], b.points[b.quads])


# -- region walls: ogrid / annulus curved nodes ride the true loop ------
@pytest.mark.parametrize("order", [2, 3])
def test_ogrid_wall_nodes_on_the_circle(order):
    r = 2.0
    loop = LineMesh.circle(r, 16, order=order)          # 16 pts = 4 per side
    # no smoothing: the wall overlay stamps the true arc regardless (a repositioning
    # smoother is rejected at order > 1 -- see test_high_order_smoothing_rejected).
    qm = QuadMesh.ogrid(loop, 4, [0.0, 0.5, 1.0])
    assert qm.order == order
    cb = curved(qm)
    # the outermost ring of quads carries wall nodes on side 1; every node there
    # must sit on the true circle
    idx = quad_edge_indices(1, order)
    # collect all nodes that lie on the wall arc: pick quads whose side-1 corners
    # are both at radius r
    corner = qm.points[qm.quads]
    rc = np.linalg.norm(corner[:, :, :2], axis=2)
    v0, v1 = QuadMesh.EDGE_POINTS[0]
    on_wall = np.isclose(rc[:, v0], r, atol=1e-9) & np.isclose(rc[:, v1], r, atol=1e-9)
    wall_nodes = cb[np.where(on_wall)[0][:, None], idx[None, :], :]
    wr = np.linalg.norm(wall_nodes.reshape(-1, 3)[:, :2], axis=1)
    assert np.allclose(wr, r, atol=1e-9)


@pytest.mark.parametrize("order", [2, 3])
def test_annulus_interior_rings_are_high_order_curved(order):
    # the radial blend is a genuine high-order blend: an *interior* ring's
    # tangential edges carry curved nodes (blend of the two loops' arcs), not a
    # straight subdivision -- so interior nodes sit off the straight chord.
    r_in, r_out = 1.0, 3.0
    L = 16
    inner = LineMesh.circle(r_in, L, order=order)
    outer = LineMesh.circle(r_out, L, order=order)
    qm = QuadMesh.annulus(inner, outer, radial=[0.0, 0.5, 1.0])   # 2 layers
    idx = quad_edge_indices(1, order)            # side-1 (tangential) node indices
    # first quad of layer 1: its side-1 (bottom) edge is the middle ring (t=0.5),
    # the average of the inner and outer arcs -> still curved.
    edge = curved(qm)[L, idx, :]                   # (order+1, 3) nodes along the edge
    a, b = edge[0], edge[-1]
    d = b - a
    dist = np.linalg.norm(np.cross(edge[1:-1] - a, d), axis=1) / np.linalg.norm(d)
    assert np.any(dist > 1e-6)                    # off the chord: genuinely curved


@pytest.mark.parametrize("order", [2, 3])
def test_annulus_wall_nodes_on_both_rings(order):
    r_in, r_out = 1.0, 3.0
    inner = LineMesh.circle(r_in, 16, order=order)
    outer = LineMesh.circle(r_out, 16, order=order)
    qm = QuadMesh.annulus(inner, outer, radial=[0.0, 0.5, 1.0])
    assert qm.order == order
    cb = curved(qm)
    corner = qm.points[qm.quads]
    rc = np.linalg.norm(corner[:, :, :2], axis=2)
    # inner wall on side 1, outer wall on side 3
    for side, radius in ((1, r_in), (3, r_out)):
        idx = quad_edge_indices(side, order)
        v0, v1 = QuadMesh.EDGE_POINTS[side - 1]
        on = np.isclose(rc[:, v0], radius, atol=1e-9) & \
            np.isclose(rc[:, v1], radius, atol=1e-9)
        nodes = cb[np.where(on)[0][:, None], idx[None, :], :]
        rr = np.linalg.norm(nodes.reshape(-1, 3)[:, :2], axis=1)
        assert np.allclose(rr, radius, atol=1e-9)


# -- high-order smoothing is rejected (corner-only relaxers) -----------
@pytest.mark.parametrize("method", ["conduction", "winslow"])
def test_high_order_smoothing_rejected(method):
    # a repositioning smoother moves only corner nodes, so it cannot smooth an
    # order-N section; the factory must raise rather than silently degrade.
    loop = LineMesh.circle(2.0, 16, order=3)
    with pytest.raises(NotImplementedError, match="order-3"):
        QuadMesh.ogrid(loop, 4, [0.0, 0.5, 1.0], smoothing_method=method)
    inner = LineMesh.circle(1.0, 16, order=3)
    outer = LineMesh.circle(3.0, 16, order=3)
    with pytest.raises(NotImplementedError, match="order-3"):
        QuadMesh.annulus(inner, outer, radial=[0.0, 0.5, 1.0],
                         smoothing_method=method)


def test_high_order_noop_smoothing_allowed():
    # the no-op strategies (bilinear/tfi/none) leave every node in place, so they
    # stay allowed at any order (e.g. circular_pipe.py runs order 5 + "bilinear").
    loop = LineMesh.circle(2.0, 16, order=3)
    for method in ("bilinear", "tfi", "none"):
        qm = QuadMesh.ogrid(loop, 4, [0.0, 0.5, 1.0], smoothing_method=method)
        assert qm.order == 3


# -- structured / rectangle straight-sided elevation --------------------
@pytest.mark.parametrize("order", [2, 3])
def test_rectangle_nodes_corner_consistent(order):
    qm = QuadMesh.rectangle([[0, 0, 0], [4, 0, 0], [4, 2, 0], [0, 2, 0]],
                            3, 2, order=order)
    assert qm.order == order
    corners = corner_indices(order, 2)
    cb = curved(qm)
    assert np.allclose(cb[:, corners, :], qm.points[qm.quads])
    # planar: every node stays at z=0
    assert np.allclose(cb[..., 2], 0.0)


# -- combinators: blend / loft / merge / from_grid ---------------------
def test_from_grid_order_n_corner_consistent():
    x = np.linspace(0, 1, 4)
    y = np.linspace(0, 1, 3)
    X, Y = np.meshgrid(x, y, indexing="ij")
    grid = np.stack([X, Y, np.zeros_like(X)], axis=-1)
    qm = QuadMesh.from_grid(grid, order=3)
    assert qm.order == 3
    corners = corner_indices(3, 2)
    assert np.allclose(curved(qm)[:, corners, :], qm.points[qm.quads])


def test_blend_morphs_quad_curved_blocks():
    a = QuadMesh.sphere(1.0, 2, order=3)
    b = QuadMesh.sphere(3.0, 2, order=3)
    lo, mid, hi = QuadMesh.blend(a, b, [0.0, 0.5, 1.0])
    assert lo.order == mid.order == hi.order == 3
    ca, cbb = curved(a), curved(b)
    assert np.allclose(curved(lo), ca)
    assert np.allclose(curved(hi), cbb)
    assert np.allclose(curved(mid), 0.5 * (ca + cbb))
    # the shared edge / private interior tables take the same lerp
    assert np.allclose(mid.edge_nodes, 0.5 * (a.edge_nodes + b.edge_nodes))
    assert np.allclose(mid.interior, 0.5 * (a.interior + b.interior))


def test_blend_rejects_mismatched_order():
    a = QuadMesh.sphere(1.0, 2, order=3)
    b = QuadMesh.sphere(1.0, 2, order=2)
    with pytest.raises(ValueError, match="same order"):
        QuadMesh.blend(a, b, [0.5])


def test_merge_concatenates_curved_blocks():
    a = QuadMesh.box(1.0, 1, order=2)
    # merging box with itself is degenerate; instead merge the six face patches
    # implicitly via box already; here check merge rejects mixed order
    b = QuadMesh.box(1.0, 1, order=3)
    with pytest.raises(ValueError, match="same order"):
        QuadMesh.merge([a, b])


# -- N=1 no-op ----------------------------------------------------------
def test_order1_factories_are_linear_no_op():
    for qm in (QuadMesh.sphere(1.0, 2),
               QuadMesh.box(1.0, 2),
               QuadMesh.annulus(LineMesh.circle(1.0, 12),
                                LineMesh.circle(2.0, 12),
                                radial=[0.0, 1.0])):
        # order 1: every high-order table is empty, the walk is just the corners
        assert qm.order == 1
        assert qm.edge_nodes.shape == (qm.edges.shape[0], 0, 3)
        assert qm.interior.shape == (qm.n_quads, 0, 3)
        assert curved(qm).shape == (qm.n_quads, 4, 3)
        assert np.allclose(curved(qm)[:, corner_indices(1, 2), :],
                           qm.points[qm.quads])


def test_order1_sphere_points_match_high_order_corners():
    lin = QuadMesh.sphere(1.7, 3)
    ho = QuadMesh.sphere(1.7, 3, order=4)
    assert np.allclose(lin.points, ho.points)            # corner points identical
    assert np.array_equal(lin.quads, ho.quads)


# -- order-N quality metric (opt-in) ------------------------------------
def test_high_order_quality_matches_corner_at_order1():
    qm = QuadMesh.sphere(1.7, 3)
    assert np.allclose(qm.scaled_jacobian(high_order=True),
                       qm.scaled_jacobian(), atol=1e-12)


@pytest.mark.parametrize("order", [2, 3])
def test_high_order_quality_non_degenerate_on_curved_sphere(order):
    qm = QuadMesh.sphere(1.7, 3, order=order)
    sj = qm.scaled_jacobian(high_order=True)
    assert sj.shape == (qm.n_quads,)
    assert np.all(np.isfinite(sj))
    assert float(sj.min()) > 0.0                         # no folded surface nodes
    assert qm.quality_summary(high_order=True)["n_elements"] == qm.n_quads


# -- VTK Lagrange quad node ordering ------------------------------------
def test_lagrange_quad_perm_order2():
    # order 2: 3x3 lexicographic (i fastest); VTK wants
    # corners [0,2,8,6], edges [1, 5, 7, 3], interior [4]
    perm = export._lagrange_quad_perm(2)
    assert np.array_equal(perm, [0, 2, 8, 6, 1, 5, 7, 3, 4])


# -- VTU (XML) export ---------------------------------------------------
def _vtu_cell_types(path):
    import xml.dom.minidom as m
    d = m.parse(path)
    ta = [da for da in d.getElementsByTagName("DataArray")
          if da.getAttribute("Name") == "types"][0]
    return set(ta.firstChild.data.split())


def _vtu_num_points(path):
    import xml.dom.minidom as m
    d = m.parse(path)
    return int(d.getElementsByTagName("Piece")[0].getAttribute("NumberOfPoints"))


def test_vtu_order1_is_plain_quad(tmp_path):
    p = str(tmp_path / "lin.vtu")
    export.quad_to_vtu(QuadMesh.box(1.0, 1), p)
    assert _vtu_cell_types(p) == {"9"}                   # VTK_QUAD
    assert _vtu_num_points(p) == 24                      # 6 faces x 4 nodes


def test_vtu_high_order_is_lagrange_quad(tmp_path):
    p = str(tmp_path / "ho.vtu")
    export.quad_to_vtu(QuadMesh.box(1.0, 1, order=3), p)
    assert _vtu_cell_types(p) == {"70"}                  # VTK_LAGRANGE_QUADRILATERAL
    # conformal (welded) numbering: shared corners and shared edge-interior nodes
    # are written once.  A closed cube surface at order 3 has 8 corners
    # + 12 edges x (3-1) edge nodes + 6 faces x (3-1)^2 private interiors = 56.
    assert _vtu_num_points(p) == 56


def test_vtu_meshio_roundtrip(tmp_path):
    meshio = pytest.importorskip("meshio")
    p = str(tmp_path / "ho.vtu")
    export.quad_to_vtu(QuadMesh.sphere(1.0, 2, order=3), p)
    mm = meshio.read(p)
    assert {c.type for c in mm.cells} == {"VTK_LAGRANGE_QUADRILATERAL"}


# -- example golden -----------------------------------------------------
def test_high_order_quad_example_matches_golden(tmp_path):
    ns = run_example("high_order_quad.py", tmp_path)
    mesh = ns["mesh"]
    assert isinstance(mesh, QuadMesh) and mesh.order == 5
    with open(os.path.join(tmp_path, "high_order_quad.vtu"), "rb") as f:
        got = f.read()
    with open(os.path.join(GOLDEN, "high_order_quad.vtu"), "rb") as f:
        ref = f.read()
    assert got == ref
