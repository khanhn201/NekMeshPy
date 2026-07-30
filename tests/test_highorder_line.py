"""Phase 2 tests: order-N ``LineMesh`` factories (``line`` / ``circle`` /
``rectangle``), high-order ``blend``, and the ``VTK_LAGRANGE_CURVE`` export.

Two invariants ride through: geometric truth (curved nodes sit on the true shape)
and the N=1 no-op (order-1 factories are byte-for-byte the old linear meshes, and
the ``high_order_curve`` example VTU stays byte-identical to its golden)."""

import os

import numpy as np
import pytest
from conftest import GOLDEN, run_example

from nekmeshpy import LineMesh
from nekmeshpy.io import export


# -- geometric truth: curved nodes lie on the true shape ---------------
@pytest.mark.parametrize("order", [2, 3, 5, 7])
def test_circle_nodes_lie_on_the_true_arc(order):
    r = 2.5
    c = LineMesh.circle(r, 8, order=order)
    assert c.order == order
    assert c.curved is not None
    assert c.curved.shape == (8, order + 1, 3)
    radii = np.linalg.norm(c.curved.reshape(-1, 3), axis=1)
    assert np.allclose(radii, r, atol=1e-12)             # every node on the circle
    # corner nodes coincide with the linear loop points
    assert np.allclose(c.curved[:, [0, order], :], c.points[c.lines])


def test_circle_curved_off_the_chord():
    # the interior HO nodes must bulge off the straight chord onto the arc
    c = LineMesh.circle(1.0, 4, order=3)
    chord_mid = 0.5 * (c.curved[:, 0, :] + c.curved[:, -1, :])
    interior = c.curved[:, 1:-1, :]
    # interior nodes are strictly farther from the origin than the chord midpoint
    assert np.all(np.linalg.norm(interior, axis=2)
                  > np.linalg.norm(chord_mid, axis=1)[:, None])


@pytest.mark.parametrize("order", [2, 4])
def test_line_nodes_on_straight_segment(order):
    lm = LineMesh.line([0, 0, 0], [3, 0, 0], [0.0, 0.5, 1.0], order=order)
    assert lm.order == order and lm.curved.shape == (2, order + 1, 3)
    # all nodes collinear on the x-axis (y=z=0), corner-consistent
    assert np.allclose(lm.curved[..., 1:], 0.0)
    assert np.allclose(lm.curved[:, [0, order], :], lm.points[lm.lines])


@pytest.mark.parametrize("order", [2, 3])
def test_rectangle_nodes_on_straight_sides(order):
    rc = LineMesh.rectangle(4.0, 2.0, 8, order=order)
    assert rc.order == order and rc.curved.shape == (8, order + 1, 3)
    assert np.allclose(rc.curved[:, [0, order], :], rc.points[rc.lines])
    # every node lies on the segment joining its element's two corners
    a = rc.curved[:, 0, :]
    b = rc.curved[:, order, :]
    for j in range(1, order):
        t = (rc.curved[:, j, :] - a) / (b - a + 1e-30)
        # collinear: the same parameter along every nonzero component
        span = np.ptp(np.where(np.abs(b - a) > 1e-9, t, np.nan), axis=1)
        assert np.all(np.nan_to_num(span) < 1e-9)


# -- N=1 no-op: order-1 factories reproduce the old linear meshes -------
def test_order1_factories_are_linear_no_op():
    for lm in (LineMesh.circle(1.3, 12),
               LineMesh.line([0, 0, 0], [1, 2, 3], np.linspace(0, 1, 5)),
               LineMesh.rectangle(3.0, 1.0, 8)):
        # order 1 still materializes the 2-endpoint block, corner-consistent
        assert lm.order == 1
        assert lm.curved.shape == (lm.lines.shape[0], 2, 3)
        assert np.allclose(lm.curved, lm.points[lm.lines])


def test_order1_circle_points_match_high_order_corners():
    lin = LineMesh.circle(1.7, 10)
    ho = LineMesh.circle(1.7, 10, order=4)
    assert np.allclose(lin.points, ho.points)            # corner points identical
    assert np.array_equal(lin.lines, ho.lines)


# -- high-order blend ---------------------------------------------------
def test_blend_morphs_curved_blocks():
    a = LineMesh.circle(1.0, 6, order=3)
    b = LineMesh.circle(3.0, 6, order=3)
    lo, mid, hi = LineMesh.blend(a, b, [0.0, 0.5, 1.0])
    assert lo.order == mid.order == hi.order == 3
    assert np.allclose(lo.curved, a.curved)
    assert np.allclose(hi.curved, b.curved)
    assert np.allclose(mid.curved, 0.5 * (a.curved + b.curved))
    # blended block stays corner-consistent with the blended points
    assert np.allclose(mid.curved[:, [0, 3], :], mid.points[mid.lines])


def test_blend_rejects_mismatched_order():
    a = LineMesh.circle(1.0, 6, order=3)
    b = LineMesh.circle(1.0, 6, order=2)
    with pytest.raises(ValueError, match="same order"):
        LineMesh.blend(a, b, [0.5])


# -- VTK Lagrange curve node ordering -----------------------------------
def test_vtk_curve_perm_puts_endpoints_first():
    # our block is ascending [p0..pN]; VTK curve wants [p0, pN, interior...]
    perm = export._lagrange_curve_perm(5)
    assert np.array_equal(perm, [0, 5, 1, 2, 3, 4])


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


def test_vtu_order1_is_plain_line(tmp_path):
    p = str(tmp_path / "lin.vtu")
    export.line_to_vtu(LineMesh.circle(1.0, 5), p)
    assert _vtu_cell_types(p) == {"3"}                   # VTK_LINE
    assert _vtu_num_points(p) == 10                      # 5 elems x 2 nodes


def test_vtu_high_order_is_lagrange_curve(tmp_path):
    p = str(tmp_path / "ho.vtu")
    export.line_to_vtu(LineMesh.circle(1.0, 4, order=5), p)
    assert _vtu_cell_types(p) == {"68"}                  # VTK_LAGRANGE_CURVE
    assert _vtu_num_points(p) == 24                      # 4 elems x 6 nodes


# -- example golden -----------------------------------------------------
def test_high_order_curve_example_matches_golden(tmp_path):
    ns = run_example("high_order_curve.py", tmp_path)
    mesh = ns["mesh"]
    assert isinstance(mesh, LineMesh) and mesh.order == 5
    with open(os.path.join(tmp_path, "high_order_curve.vtu"), "rb") as f:
        got = f.read()
    with open(os.path.join(GOLDEN, "high_order_curve.vtu"), "rb") as f:
        ref = f.read()
    assert got == ref
