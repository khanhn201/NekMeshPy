"""Order-N ``LineMesh`` factories (``line`` / ``circle`` /
``rectangle``), high-order ``blend``, and the ``VTK_LAGRANGE_CURVE`` export.

Two invariants ride through: geometric truth (curved nodes sit on the true shape)
and the N=1 no-op (order-1 factories are byte-for-byte the old linear meshes, and
the ``high_order_curve`` example VTU stays byte-identical to its golden)."""


import numpy as np
import pytest
from conftest import curved, vtu_cell_types

from nekmeshpy import linemesh
from nekmeshpy.io import export


# -- geometric truth: curved nodes lie on the true shape ---------------
@pytest.mark.parametrize("order", [2, 3, 5, 7])
def test_circle_nodes_lie_on_the_true_arc(order):
    r = 2.5
    c = linemesh.circle(r, 8, order=order)
    assert c.order == order
    cb = curved(c)                                       # B-rep -> per-line block
    assert cb.shape == (8, order + 1, 3)
    radii = np.linalg.norm(cb.reshape(-1, 3), axis=1)
    assert np.allclose(radii, r, atol=1e-12)             # every node on the circle
    # corner nodes coincide with the linear loop points
    assert np.allclose(cb[:, [0, order], :], c.points[c.lines])
    # the private interior nodes are the whole high-order state of a LineMesh
    assert np.allclose(cb[:, 1:order, :], c.interior)


def test_circle_curved_off_the_chord():
    # the interior HO nodes must bulge off the straight chord onto the arc
    c = linemesh.circle(1.0, 4, order=3)
    cb = curved(c)
    chord_mid = 0.5 * (cb[:, 0, :] + cb[:, -1, :])
    interior = cb[:, 1:-1, :]
    # interior nodes are strictly farther from the origin than the chord midpoint
    assert np.all(np.linalg.norm(interior, axis=2)
                  > np.linalg.norm(chord_mid, axis=1)[:, None])


@pytest.mark.parametrize("order", [2, 4])
def test_line_nodes_on_straight_segment(order):
    lm = linemesh.line([0, 0, 0], [3, 0, 0], [0.0, 0.5, 1.0], order=order)
    cb = curved(lm)
    assert lm.order == order and cb.shape == (2, order + 1, 3)
    # all nodes collinear on the x-axis (y=z=0), corner-consistent
    assert np.allclose(cb[..., 1:], 0.0)
    assert np.allclose(cb[:, [0, order], :], lm.points[lm.lines])


@pytest.mark.parametrize("order", [2, 3])
def test_rectangle_nodes_on_straight_sides(order):
    rc = linemesh.rectangle(4.0, 2.0, 8, order=order)
    cb = curved(rc)
    assert rc.order == order and cb.shape == (8, order + 1, 3)
    assert np.allclose(cb[:, [0, order], :], rc.points[rc.lines])
    # every node lies on the segment joining its element's two corners
    a = cb[:, 0, :]
    b = cb[:, order, :]
    for j in range(1, order):
        t = (cb[:, j, :] - a) / (b - a + 1e-30)
        # collinear: the same parameter along every nonzero component
        span = np.ptp(np.where(np.abs(b - a) > 1e-9, t, np.nan), axis=1)
        assert np.all(np.nan_to_num(span) < 1e-9)


# -- N=1 no-op: order-1 factories reproduce the old linear meshes -------
def test_order1_factories_are_linear_no_op():
    for lm in (linemesh.circle(1.3, 12),
               linemesh.line([0, 0, 0], [1, 2, 3], np.linspace(0, 1, 5)),
               linemesh.rectangle(3.0, 1.0, 8)):
        # order 1: the private interior is empty and the walk is just the corners
        assert lm.order == 1
        assert lm.interior.shape == (lm.lines.shape[0], 0, 3)
        assert curved(lm).shape == (lm.lines.shape[0], 2, 3)
        assert np.allclose(curved(lm), lm.points[lm.lines])


def test_order1_circle_points_match_high_order_corners():
    lin = linemesh.circle(1.7, 10)
    ho = linemesh.circle(1.7, 10, order=4)
    assert np.allclose(lin.points, ho.points)            # corner points identical
    assert np.array_equal(lin.lines, ho.lines)


# -- high-order blend ---------------------------------------------------
def test_blend_morphs_curved_blocks():
    a = linemesh.circle(1.0, 6, order=3)
    b = linemesh.circle(3.0, 6, order=3)
    lo, mid, hi = linemesh.blend(a, b, [0.0, 0.5, 1.0])
    assert lo.order == mid.order == hi.order == 3
    ca, cb_, cmid = curved(a), curved(b), curved(mid)
    assert np.allclose(curved(lo), ca)
    assert np.allclose(curved(hi), cb_)
    assert np.allclose(cmid, 0.5 * (ca + cb_))
    # the blended private interior takes the same lerp as the corners
    assert np.allclose(mid.interior, 0.5 * (a.interior + b.interior))
    # blended block stays corner-consistent with the blended points
    assert np.allclose(cmid[:, [0, 3], :], mid.points[mid.lines])


def test_blend_rejects_mismatched_order():
    a = linemesh.circle(1.0, 6, order=3)
    b = linemesh.circle(1.0, 6, order=2)
    with pytest.raises(ValueError, match="same order"):
        linemesh.blend(a, b, [0.5])


# -- VTK Lagrange curve node ordering -----------------------------------
def test_vtk_curve_perm_puts_endpoints_first():
    # our block is ascending [p0..pN]; VTK curve wants [p0, pN, interior...]
    perm = export._lagrange_curve_perm(5)
    assert np.array_equal(perm, [0, 5, 1, 2, 3, 4])


# -- VTU (XML) export ---------------------------------------------------
def _vtu_num_points(path):
    import xml.dom.minidom as m
    d = m.parse(path)
    return int(d.getElementsByTagName("Piece")[0].getAttribute("NumberOfPoints"))


def test_vtu_order1_is_plain_line(tmp_path):
    p = str(tmp_path / "lin.vtu")
    export.line_to_vtu(linemesh.circle(1.0, 5), p)
    assert vtu_cell_types(p) == {3}                   # VTK_LINE
    assert _vtu_num_points(p) == 10                      # 5 elems x 2 nodes


def test_vtu_high_order_is_lagrange_curve(tmp_path):
    p = str(tmp_path / "ho.vtu")
    export.line_to_vtu(linemesh.circle(1.0, 4, order=5), p)
    assert vtu_cell_types(p) == {68}                  # VTK_LAGRANGE_CURVE
    # conformal (welded) numbering: shared corners are written once.  A closed
    # 4-element loop has 4 corners + 4 x (6-2) private interior nodes = 20.
    assert _vtu_num_points(p) == 20


# -- example golden -----------------------------------------------------
