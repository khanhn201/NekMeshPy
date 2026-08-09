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

from nekmeshpy import LineMesh, QuadMesh, linemesh, quadmesh
from nekmeshpy.core.fields import uniform_spacing
from nekmeshpy.core.interp import corner_indices, quad_edge_indices
from nekmeshpy.io import export


# -- geometric truth: sphere nodes lie on the true sphere --------------
@pytest.mark.parametrize("order", [2, 3, 5])
def test_sphere_nodes_lie_on_the_true_sphere(order):
    r = 3.0
    s = quadmesh.sphere(r, 3, order=order)
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
    b = quadmesh.box(1.0, 2, order=order)
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
    loop = linemesh.circle(r, 16, order=order)          # 16 pts = 4 per side
    # no smoothing: the wall overlay stamps the true arc regardless (a repositioning
    # smoother is rejected at order > 1 -- see test_high_order_smoothing_rejected).
    qm = quadmesh.ogrid(loop, 4, [0.0, 0.5, 1.0])
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


@pytest.mark.parametrize("order", [2, 3, 4])
def test_structured_stamps_its_edges_high_order_nodes(order):
    # ``structured`` must honour the high-order nodes of the edges it is handed:
    # a curved bottom edge (LineMesh.arc) has to land on the block's bottom side,
    # not be replaced by the transfinite blend's straight subdivision (which would
    # miss the arc by the chord sagitta, ~2e-2 here).
    r, n = 1.0, 8
    bottom = linemesh.arc(r, n, start_theta=np.pi, end_theta=0.0, order=order)
    right = linemesh.line((r, 0.0, 0.0), (r, 2.0, 0.0), uniform_spacing(3),
                          order=order)
    top = linemesh.line((r, 2.0, 0.0), (-r, 2.0, 0.0), uniform_spacing(n),
                        order=order)
    left = linemesh.line((-r, 2.0, 0.0), (-r, 0.0, 0.0), uniform_spacing(3),
                         order=order)
    qm = quadmesh.structured([bottom, right, top, left])
    assert qm.order == order
    cb = curved(qm)
    idx = quad_edge_indices(1, order)                  # the bottom side's nodes
    corner = qm.points[qm.quads]
    rc = np.linalg.norm(corner[:, :, :2], axis=2)
    v0, v1 = QuadMesh.EDGE_POINTS[0]
    on_wall = np.isclose(rc[:, v0], r, atol=1e-9) & np.isclose(rc[:, v1], r, atol=1e-9)
    assert int(np.count_nonzero(on_wall)) == n
    wall = cb[np.where(on_wall)[0][:, None], idx[None, :], :].reshape(-1, 3)
    assert np.max(np.abs(np.linalg.norm(wall[:, :2], axis=1) - r)) < 1e-13


# -- region *interiors* are curved too, not a straight fill inside a curved wall --
def _edge_chord_deviation(qm):
    """Per shared edge, the max distance of its interior nodes from the straight
    chord between its two corners -- 0 for a straight-subdivided edge."""
    lm = qm.lines
    a = lm.points[lm.lines[:, 0]]
    b = lm.points[lm.lines[:, 1]]
    d = b - a
    dev = np.linalg.norm(np.cross(lm.interior - a[:, None, :], d[:, None, :]),
                         axis=2) / np.linalg.norm(d, axis=1)[:, None]
    return dev.max(axis=1)                               # (n_edges,)


def _on_radius(qm, r):
    """Mask of shared edges whose *both* corners sit at radius ``r`` (the wall)."""
    lm = qm.lines
    rc = np.linalg.norm(lm.points[:, :2], axis=1)
    return np.isclose(rc, r, atol=1e-9)[lm.lines].all(axis=1)


@pytest.mark.parametrize("order", [2, 3, 4])
def test_ogrid_interior_edges_are_curved(order):
    # the wall's curvature must propagate inward: each O-ring is a *blend* of the
    # block perimeter and the wall loop, so an interior ring's tangential edges bow
    # too.  A straight-subdivided interior (the old behaviour) leaves every non-wall
    # edge dead on its chord at ~1e-17.
    r = 2.0
    loop = linemesh.circle(r, 16, order=order)
    qm = quadmesh.ogrid(loop, 4, [0.0, 0.5, 1.0])        # one interior ring at t=0.5
    dev = _edge_chord_deviation(qm)
    wall = _on_radius(qm, r)
    assert dev[wall].max() > 1e-2                        # the wall itself is curved
    # the t=0.5 ring inherits ~half the wall's bow -- orders of magnitude above the
    # 1e-17 a straight fill would give
    assert dev[~wall].max() > 0.3 * dev[wall].max()


@pytest.mark.parametrize("order", [2, 3, 4])
def test_half_ogrid_interior_edges_are_curved(order):
    # half_ogrid had no order>1 coverage at all.  Reached through spined_ogrid (the
    # bifurcation path), which runs two half_ogrids and merges them.
    r = 1.5
    loop = linemesh.circle(r, 32, order=order)
    qm = quadmesh.spined_ogrid(loop, [0.0, 0.4, 1.0])
    assert qm.order == order
    dev = _edge_chord_deviation(qm)
    wall = _on_radius(qm, r)
    assert dev[wall].max() > 1e-3
    assert dev[~wall].max() > 0.3 * dev[wall].max()
    # and the wall nodes still land on the exact circle
    lm = qm.lines
    wr = np.linalg.norm(lm.interior[wall].reshape(-1, 3)[:, :2], axis=1)
    assert np.allclose(wr, r, atol=1e-9)


@pytest.mark.parametrize("order", [2, 3, 4])
def test_structured_interior_matches_the_analytic_coons_blend(order):
    # ``structured`` owns an exact transfinite map, so at order N it is evaluated at
    # the GLL-refined lattice rather than subdivided from corners.  An annular sector
    # (two concentric arcs + two straight radial edges) makes the exact answer
    # analytic: the straight radial sides are their own linear blend, so the patch
    # collapses to a radial lerp of the arcs -- every node in a block column shares
    # the bottom arc's angle, and every node in a block row shares one radius.
    r_in, r_out, n, nr = 1.0, 2.0, 6, 3
    inner = linemesh.arc(r_in, n, start_theta=0.0, end_theta=np.pi / 2, order=order)
    outer = linemesh.arc(r_out, n, start_theta=np.pi / 2, end_theta=0.0, order=order)
    ip, op = inner.points, outer.points
    right = linemesh.line(ip[-1], op[0], uniform_spacing(nr + 1), order=order)
    left = linemesh.line(op[-1], ip[0], uniform_spacing(nr + 1), order=order)
    qm = quadmesh.structured([inner, right, outer, left])
    assert qm.order == order
    row = order + 1
    blk = curved(qm).reshape(-1, row, row, 3)            # (Q, j, i, 3), i fastest
    ang = np.arctan2(blk[..., 1], blk[..., 0])
    rad = np.linalg.norm(blk[..., :2], axis=-1)
    assert np.max(np.abs(ang - ang[:, 0:1, :])) < 1e-13  # angle depends on i only
    assert np.max(np.abs(rad - rad[:, :, 0:1])) < 1e-13  # radius depends on j only
    assert r_in - 1e-12 <= rad.min() and rad.max() <= r_out + 1e-12
    # interior edges genuinely bow (the whole point): non-wall edges are off-chord
    dev = _edge_chord_deviation(qm)
    assert dev[~(_on_radius(qm, r_in) | _on_radius(qm, r_out))].max() > 1e-3


def test_structured_rejects_a_non_chain_edge():
    # the overlay maps line k onto the k-th interval of the side, so an edge whose
    # lines are not the consecutive chain would be stamped in the wrong place
    edges = [linemesh.line((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), uniform_spacing(3),
                           order=2),
             linemesh.line((1.0, 0.0, 0.0), (1.0, 1.0, 0.0), uniform_spacing(2),
                           order=2),
             linemesh.line((1.0, 1.0, 0.0), (0.0, 1.0, 0.0), uniform_spacing(3),
                           order=2),
             linemesh.line((0.0, 1.0, 0.0), (0.0, 0.0, 0.0), uniform_spacing(2),
                           order=2)]
    scrambled = LineMesh(edges[0].points, [[0, 2], [2, 1], [1, 3]],
                         interior=np.zeros((3, 1, 3)))
    with pytest.raises(ValueError, match="consecutive chain"):
        quadmesh.structured([scrambled, edges[1], edges[2], edges[3]])


@pytest.mark.parametrize("order", [2, 3])
def test_annulus_interior_rings_are_high_order_curved(order):
    # the radial blend is a genuine high-order blend: an *interior* ring's
    # tangential edges carry curved nodes (blend of the two loops' arcs), not a
    # straight subdivision -- so interior nodes sit off the straight chord.
    r_in, r_out = 1.0, 3.0
    L = 16
    inner = linemesh.circle(r_in, L, order=order)
    outer = linemesh.circle(r_out, L, order=order)
    qm = quadmesh.annulus(inner, outer, radial=[0.0, 0.5, 1.0])   # 2 layers
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
    inner = linemesh.circle(r_in, 16, order=order)
    outer = linemesh.circle(r_out, 16, order=order)
    qm = quadmesh.annulus(inner, outer, radial=[0.0, 0.5, 1.0])
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
    loop = linemesh.circle(2.0, 16, order=3)
    with pytest.raises(NotImplementedError, match="order-3"):
        quadmesh.ogrid(loop, 4, [0.0, 0.5, 1.0], smoothing_method=method)
    inner = linemesh.circle(1.0, 16, order=3)
    outer = linemesh.circle(3.0, 16, order=3)
    with pytest.raises(NotImplementedError, match="order-3"):
        quadmesh.annulus(inner, outer, radial=[0.0, 0.5, 1.0],
                         smoothing_method=method)


def test_high_order_noop_smoothing_allowed():
    # the no-op strategies (bilinear/tfi/none) leave every node in place, so they
    # stay allowed at any order (e.g. circular_pipe.py runs order 2 + "bilinear").
    loop = linemesh.circle(2.0, 16, order=3)
    for method in ("bilinear", "tfi", "none"):
        qm = quadmesh.ogrid(loop, 4, [0.0, 0.5, 1.0], smoothing_method=method)
        assert qm.order == 3


# -- structured / rectangle straight-sided elevation --------------------
@pytest.mark.parametrize("order", [2, 3])
def test_rectangle_nodes_corner_consistent(order):
    qm = quadmesh.rectangle([[0, 0, 0], [4, 0, 0], [4, 2, 0], [0, 2, 0]],
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
    qm = quadmesh.from_grid(grid, order=3)
    assert qm.order == 3
    corners = corner_indices(3, 2)
    assert np.allclose(curved(qm)[:, corners, :], qm.points[qm.quads])


def test_blend_morphs_quad_curved_blocks():
    a = quadmesh.sphere(1.0, 2, order=3)
    b = quadmesh.sphere(3.0, 2, order=3)
    lo, mid, hi = quadmesh.blend(a, b, [0.0, 0.5, 1.0])
    assert lo.order == mid.order == hi.order == 3
    ca, cbb = curved(a), curved(b)
    assert np.allclose(curved(lo), ca)
    assert np.allclose(curved(hi), cbb)
    assert np.allclose(curved(mid), 0.5 * (ca + cbb))
    # the shared edge / private interior tables take the same lerp
    assert np.allclose(mid.edge_nodes, 0.5 * (a.edge_nodes + b.edge_nodes))
    assert np.allclose(mid.interior, 0.5 * (a.interior + b.interior))


def test_blend_rejects_mismatched_order():
    a = quadmesh.sphere(1.0, 2, order=3)
    b = quadmesh.sphere(1.0, 2, order=2)
    with pytest.raises(ValueError, match="same order"):
        quadmesh.blend(a, b, [0.5])


def test_merge_concatenates_curved_blocks():
    a = quadmesh.box(1.0, 1, order=2)
    # merging box with itself is degenerate; instead merge the six face patches
    # implicitly via box already; here check merge rejects mixed order
    b = quadmesh.box(1.0, 1, order=3)
    with pytest.raises(ValueError, match="same order"):
        quadmesh.merge([a, b])


# -- N=1 no-op ----------------------------------------------------------
def test_order1_factories_are_linear_no_op():
    for qm in (quadmesh.sphere(1.0, 2),
               quadmesh.box(1.0, 2),
               quadmesh.annulus(linemesh.circle(1.0, 12),
                                linemesh.circle(2.0, 12),
                                radial=[0.0, 1.0])):
        # order 1: every high-order table is empty, the walk is just the corners
        assert qm.order == 1
        assert qm.edge_nodes.shape == (qm.edges.shape[0], 0, 3)
        assert qm.interior.shape == (qm.n_quads, 0, 3)
        assert curved(qm).shape == (qm.n_quads, 4, 3)
        assert np.allclose(curved(qm)[:, corner_indices(1, 2), :],
                           qm.points[qm.quads])


def test_order1_sphere_points_match_high_order_corners():
    lin = quadmesh.sphere(1.7, 3)
    ho = quadmesh.sphere(1.7, 3, order=4)
    assert np.allclose(lin.points, ho.points)            # corner points identical
    assert np.array_equal(lin.quads, ho.quads)


# -- order-N quality metric (opt-in) ------------------------------------
def test_high_order_quality_matches_corner_at_order1():
    qm = quadmesh.sphere(1.7, 3)
    assert np.allclose(quadmesh.scaled_jacobian(qm, high_order=True),
                       quadmesh.scaled_jacobian(qm), atol=1e-12)


@pytest.mark.parametrize("order", [2, 3])
def test_high_order_quality_non_degenerate_on_curved_sphere(order):
    qm = quadmesh.sphere(1.7, 3, order=order)
    sj = quadmesh.scaled_jacobian(qm, high_order=True)
    assert sj.shape == (qm.n_quads,)
    assert np.all(np.isfinite(sj))
    assert float(sj.min()) > 0.0                         # no folded surface nodes
    assert quadmesh.quality_summary(qm, high_order=True).n_elements == qm.n_quads


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
    export.quad_to_vtu(quadmesh.box(1.0, 1), p)
    assert _vtu_cell_types(p) == {"9"}                   # VTK_QUAD
    assert _vtu_num_points(p) == 24                      # 6 faces x 4 nodes


def test_vtu_high_order_is_lagrange_quad(tmp_path):
    p = str(tmp_path / "ho.vtu")
    export.quad_to_vtu(quadmesh.box(1.0, 1, order=3), p)
    assert _vtu_cell_types(p) == {"70"}                  # VTK_LAGRANGE_QUADRILATERAL
    # conformal (welded) numbering: shared corners and shared edge-interior nodes
    # are written once.  A closed cube surface at order 3 has 8 corners
    # + 12 edges x (3-1) edge nodes + 6 faces x (3-1)^2 private interiors = 56.
    assert _vtu_num_points(p) == 56


def test_vtu_meshio_roundtrip(tmp_path):
    meshio = pytest.importorskip("meshio")
    p = str(tmp_path / "ho.vtu")
    export.quad_to_vtu(quadmesh.sphere(1.0, 2, order=3), p)
    mm = meshio.read(p)
    assert {c.type for c in mm.cells} == {"VTK_LAGRANGE_QUADRILATERAL"}


def test_lagrange_quad_perm_matches_vtk_order3():
    """Pin the order-3 permutation against VTK's own spec, transcribed by hand.

    ``vtkHigherOrderQuadrilateral::PointIndexFromIJK`` numbers the four edge runs
    bottom / right / top / left, and -- unlike the corners -- **not** CCW: the top
    edge ascends in ``i`` and the left edge ascends in ``j``.  Order 3 is the lowest
    order that can see this (at order 2 each run is one node, so a reversed run is
    indistinguishable), which is exactly how the reversal shipped unnoticed.
    """
    n, row = 3, 4

    def ij(i, j):
        return i + row * j
    expected = [ij(0, 0), ij(n, 0), ij(n, n), ij(0, n),      # corners, CCW
                ij(1, 0), ij(2, 0),                          # bottom, ascending i
                ij(n, 1), ij(n, 2),                          # right,  ascending j
                ij(1, n), ij(2, n),                          # top,    ascending i
                ij(0, 1), ij(0, 2),                          # left,   ascending j
                ij(1, 1), ij(2, 1), ij(1, 2), ij(2, 2)]      # interior, i fastest
    assert list(export._lagrange_quad_perm(3)) == expected


def test_vtu_quad_nodes_land_on_the_bilinear_lattice(tmp_path):
    """Every emitted node of an affine quad must sit on that quad's **equispaced**
    lattice -- the parametrization a VTK Lagrange cell is defined on, and therefore the
    one the writer relabels to (``export._to_equispaced``).  The toolkit *stores* GLL,
    so this is also the check that the relabel actually happened: at orders 3 and 4 the
    two lattices differ, and a writer that shipped the GLL nodes verbatim would land
    them here at the wrong slots.

    An affine (planar, uniformly spaced) mesh makes each node's true position an
    exact affine function of its ``(i, j)`` lattice slot, so a permutation that
    scrambles a run shows up as a node sitting at another slot's coordinates.  Runs
    at orders 3 and 4, where the edge runs carry more than one node.
    """
    meshio = pytest.importorskip("meshio")
    for order in (2, 3, 4):
        p = str(tmp_path / ("aff%d.vtu" % order))
        corners = np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0],
                            [3.0, 2.0, 0.0], [0.0, 2.0, 0.0]])
        export.quad_to_vtu(quadmesh.rectangle(corners, 2, 2, order=order), p)
        mm = meshio.read(p)
        conn = np.vstack([c.data for c in mm.cells])
        pts = np.asarray(mm.points, dtype=float)
        inv = np.argsort(export._lagrange_quad_perm(order))
        g = uniform_spacing(order)          # VTK's Lagrange node parameters
        for cell in conn:
            block = pts[cell][inv]          # back to lexicographic (i fastest)
            c = block[corner_indices(order, 2)]      # CCW: (0,0) (n,0) (n,n) (0,n)
            uu, vv = np.meshgrid(g, g, indexing="xy")
            want = (c[0] * ((1 - uu) * (1 - vv))[..., None]
                    + c[1] * (uu * (1 - vv))[..., None]
                    + c[3] * ((1 - uu) * vv)[..., None]
                    + c[2] * (uu * vv)[..., None]).reshape(-1, 3)
            assert np.max(np.abs(block - want)) < 1e-12


# -- example golden -----------------------------------------------------
def test_high_order_quad_example_matches_golden(tmp_path):
    ns = run_example("high_order_quad.py", tmp_path)
    mesh = ns["mesh"]
    # the example's own ORDER constant is the contract -- read it back rather
    # than pinning a literal here, so the two can never drift apart.
    assert isinstance(mesh, QuadMesh) and mesh.order == ns["ORDER"] > 1
    with open(os.path.join(tmp_path, "high_order_quad.vtu"), "rb") as f:
        got = f.read()
    with open(os.path.join(GOLDEN, "high_order_quad.vtu"), "rb") as f:
        ref = f.read()
    assert got == ref
