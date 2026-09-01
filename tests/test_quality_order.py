"""Sampling the scaled Jacobian above a mesh's own order.

``scaled_jacobian`` *samples*: it is exact at the ``(order+1)**dim`` GLL nodes of an
element and says nothing between them.  An element's map is degree ``order`` per
direction, but its Jacobian determinant is a polynomial of much higher degree, so a
positive reading at the nodes is not a certificate.  A solver that runs the same mesh
at ``lx1 = 8`` evaluates the very same map on far more points and can find a fold the
mesh's own order never looked at -- which is not a disagreement about the geometry,
only about where it was measured.

These tests pin both halves: that reading a block on a finer lattice is a change of
nodal basis and invents nothing, and that the metric plumbed on top of it actually
catches an element its own order clears.
"""

import numpy as np
import pytest

from nekmeshpy import hexmesh, linemesh, quadmesh
from nekmeshpy.core.interp import (
    corner_indices,
    resample_block,
    sampled_scaled_jacobian,
    scaled_jacobian,
    tensor_nodes,
)
from nekmeshpy.core.quality import SCAN_ORDER
from nekmeshpy.hexmesh.quality import corner_scaled_jacobian
from nekmeshpy.quadmesh import QuadMesh
from nekmeshpy.quadmesh._helpers import entities_from_blocks

#: An order-2 quad that reads a healthy **+0.7297** on its own 9 nodes and is inverted
#: on *every* finer lattice from 3 to 12. Deliberately robust that way, so these tests
#: do not quietly stop testing anything when :data:`SCAN_ORDER` changes.
FOLDED = np.array([
    [0.157093, 0.284329, 0.0], [0.044413, 0.216317, 0.0],
    [0.616656, -0.177226, 0.0], [0.158494, 0.282983, 0.0],
    [0.613362, 0.553789, 0.0], [0.888034, 0.642626, 0.0],
    [-0.080654, 0.823541, 0.0], [0.274297, 0.972976, 0.0],
    [1.030513, 1.172745, 0.0]])[None, :, :]

#: The same idea, but one whose sign *alternates* with the sampling order: clean at 2,
#: folded at 3, clean again at 4, clean at 7, folded at 8. It exists to pin the fact
#: that "check one order higher" is not a rule -- and it is not hypothetical, it is
#: what stopped an earlier version of these tests from catching anything at all when
#: the default order moved from 11 to 7.
ERRATIC = np.array([
    [-0.457590, -0.518394, 0.0], [0.725492, -0.322374, 0.0],
    [1.332396, -0.261499, 0.0], [-0.441890, 0.445449, 0.0],
    [0.261058, 0.585512, 0.0], [0.627117, 0.781947, 0.0],
    [-0.192898, 1.297756, 0.0], [0.984728, 1.295968, 0.0],
    [0.808285, 0.865612, 0.0]])[None, :, :]


def _folded_quad():
    """``FOLDED`` as a one-element order-2 ``QuadMesh``."""
    pts = FOLDED[0][corner_indices(2, 2)]
    quads = np.array([[0, 1, 2, 3]], dtype=np.int64)
    lm, elem_edges, flip, interior = entities_from_blocks(
        FOLDED, quads, pts, 2, "test_quality_order")
    return QuadMesh(lm, elem_edges, flip, interior)


# -- resample_block: a change of basis, not a refinement ---------------------

def test_resampling_to_the_same_order_is_the_identity():
    assert resample_block(FOLDED, 2, 2, 2) is FOLDED


@pytest.mark.parametrize("dim", [2, 3])
def test_resampling_reproduces_the_map_it_was_given(dim):
    """A degree-2 map read on any finer lattice returns that map's own values.

    The map here is quadratic, so the order-2 nodes determine it exactly and every
    resampled node must land on the analytic surface -- if resampling were adding
    geometry rather than re-reading it, this is where that would show."""
    def f(p):
        x = p[..., 0]
        y = p[..., 1] if dim > 1 else np.zeros_like(x)
        z = p[..., 2] if dim > 2 else np.zeros_like(x)
        return np.stack([x + 0.3 * y * y, y - 0.2 * x * z, z + 0.1 * x * y], axis=-1)

    ref = 2.0 * tensor_nodes(2, dim) - 1.0                    # (M,dim) on [-1,1]
    pad = np.zeros((ref.shape[0], 3))
    pad[:, :dim] = ref
    block = f(pad)[None, :, :]

    for n in (3, 5, 8):
        got = resample_block(block, 2, n, dim)
        fine = 2.0 * tensor_nodes(n, dim) - 1.0
        pad_n = np.zeros((fine.shape[0], 3))
        pad_n[:, :dim] = fine
        assert got.shape == (1, (n + 1) ** dim, 3)
        assert np.allclose(got[0], f(pad_n), atol=1e-12)


def test_resampling_below_the_stored_order_is_refused():
    """Downward would drop the nodes that make the element curved and report a
    *better* number for a mesh that is not the one stored."""
    with pytest.raises(ValueError, match="below the stored order"):
        resample_block(FOLDED, 2, 1, 2)


# -- the metric on top of it -------------------------------------------------

def test_the_nodes_of_an_element_do_not_certify_it():
    """Exact at the 9 nodes, wrong about the element: the same map folds on 81."""
    assert scaled_jacobian(FOLDED, 2, dim=2)[0] == pytest.approx(0.7297, abs=5e-3)
    at8 = scaled_jacobian(resample_block(FOLDED, 2, 8, 2), 8, dim=2)[0]
    assert at8 < 0.0


def test_quality_summary_order_catches_what_the_mesh_order_clears():
    mesh = _folded_quad()
    assert mesh.order == 2

    own = quadmesh.quality_summary(mesh)
    assert own.n_inverted == 0 and own.min > 0.7          # looks healthy, and is not

    at8 = quadmesh.quality_summary(mesh, order=8)
    assert at8.n_inverted == 1
    assert at8.min < 0.0


@pytest.mark.parametrize("rung", ["quad", "hex"])
def test_the_default_is_the_mesh_own_order(rung):
    """``order=None`` must stay exactly what it was before the argument existed."""
    if rung == "quad":
        mesh, ns = _folded_quad(), quadmesh
    else:
        section = quadmesh.ogrid(linemesh.circle(1.0, 8, order=2), 2, 2)
        mesh, ns = hexmesh.extrude(section, 1.0, 2), hexmesh
    assert ns.quality_summary(mesh) == ns.quality_summary(mesh, order=mesh.order)
    assert np.array_equal(ns.scaled_jacobian(mesh),
                          ns.scaled_jacobian(mesh, order=mesh.order))


def test_a_sound_mesh_stays_sound_when_read_finer():
    """The check is not simply pessimistic: an element that is genuinely fine reads
    fine at every order, so a positive answer here still means something."""
    section = quadmesh.ogrid(linemesh.circle(1.0, 8, order=2), 2, 2)
    mesh = hexmesh.extrude(section, 1.0, 2)
    base = hexmesh.quality_summary(mesh).min
    for n in (3, 5, 8):
        assert hexmesh.quality_summary(mesh, order=n).min == pytest.approx(base,
                                                                          abs=1e-9)


def test_sampling_order_is_not_monotone():
    """A coarse-but-finer lattice can still miss the fold -- which is why the order to
    ask for is the solver's, not merely "more than the mesh's".

    ``ERRATIC`` goes negative at 3, positive again at 4, is *clean at 7* -- the current
    default -- and negative at 8. A check one order above the mesh would clear it; so
    would a check at the default. Only the order actually being run is informative."""
    got = {n: (scaled_jacobian(ERRATIC, 2, dim=2)[0] if n == 2 else
               scaled_jacobian(resample_block(ERRATIC, 2, n, 2), n, dim=2)[0])
           for n in (2, 3, 4, 7, 8)}
    assert got[2] > 0.0
    assert got[3] < 0.0
    assert got[4] > 0.0
    assert got[7] > 0.0          # clean at the default, folded either side of it
    assert got[8] < 0.0


# -- the scan, and the warning it exists to raise ----------------------------

def test_scan_defaults_to_the_solver_order():
    """One order -- the one the solver runs. Intermediate orders cost real time and
    certify nothing, since the reading is not monotone in the sampling order."""
    mesh = _folded_quad()
    scan = quadmesh.order_scan(mesh)
    assert scan.orders == (SCAN_ORDER,)
    assert scan.skipped == ()


def test_nothing_to_check_when_the_mesh_is_already_at_the_solver_order():
    """No lattice above its own to read, so the default asks for none rather than
    re-reading what ``quality_summary`` already reported."""
    section = quadmesh.ogrid(linemesh.circle(1.0, 8, order=2), 2, 2)
    mesh = hexmesh.extrude(section, 1.0, 2)
    scan = hexmesh.order_scan(mesh, orders=None, budget=10 ** 12)
    assert scan.orders == (SCAN_ORDER,)          # order 2 mesh, so there is one
    assert quadmesh.order_scan(_folded_quad(), orders=[2]).orders == (2,)


def test_scan_catches_the_fold_and_reports_it_unclean():
    scan = quadmesh.order_scan(_folded_quad())
    assert not scan.clean
    order, worst = scan.worst
    assert worst < 0.0
    assert scan.n_inverted[scan.orders.index(order)] == 1


def test_scan_of_a_sound_mesh_is_clean():
    section = quadmesh.ogrid(linemesh.circle(1.0, 8, order=2), 2, 2)
    scan = hexmesh.order_scan(hexmesh.extrude(section, 1.0, 2))
    assert scan.clean
    assert all(m > 0.0 for m in scan.min_sj)


def test_budget_drops_the_highest_orders_and_says_which():
    """A dropped order is *unchecked*, not clean -- so it has to come back named."""
    mesh = _folded_quad()
    scan = quadmesh.order_scan(mesh, orders=[3, 4, 5], budget=1)
    assert scan.orders == ()            # at SCAN_ORDER a big mesh runs to minutes,
    assert scan.skipped == (3, 4, 5)    # so the budget may decline the lot
    assert not scan.clean               # ...and declined is unchecked, not clean


def test_scan_refuses_to_sample_below_the_mesh_order():
    with pytest.raises(ValueError, match="below the mesh's own order"):
        quadmesh.order_scan(_folded_quad(), orders=[1, 4])


def test_explicit_orders_are_honoured():
    scan = quadmesh.order_scan(_folded_quad(), orders=[8], budget=10 ** 12)
    assert scan.orders == (8,)
    assert scan.n_inverted == (1,)


def test_the_budget_declines_a_mesh_too_big_to_check_and_says_so():
    """A report must not quietly spend minutes -- but nor may it call the result
    clean."""
    section = quadmesh.ogrid(linemesh.circle(1.0, 16, order=2), 4, 3)
    mesh = hexmesh.extrude(section, 1.0, 8)
    scan = hexmesh.order_scan(mesh, budget=1000)
    assert scan.orders == ()
    assert scan.skipped == (SCAN_ORDER,)
    assert not scan.clean
    line = hexmesh.quality.format_scan(scan, mesh.order)
    assert "not checked" in line and "scan budget" in line
    assert str(SCAN_ORDER) in line          # say which order went unread
    assert "WARNING" not in line


def test_report_warns_when_a_finer_lattice_inverts(caplog):
    """The whole point: a mesh clean at its own order must not report as healthy."""
    section = quadmesh.ogrid(linemesh.circle(1.0, 8, order=2), 2, 2)
    good = hexmesh.extrude(section, 1.0, 2)
    with caplog.at_level("WARNING"):
        text = hexmesh.report(good)
    assert "sampling" in text
    assert "WARNING" not in text
    assert not caplog.records

    scan = quadmesh.order_scan(_folded_quad())
    line = quadmesh.quality.format_scan(scan, 2)
    assert "** WARNING **" in line
    assert "folded between its own nodes" in line


def test_scan_line_marks_only_the_inverting_orders():
    scan = quadmesh.order_scan(_folded_quad(), orders=[3, 4, 5], budget=10 ** 12)
    line = quadmesh.quality.format_scan(scan, 2).splitlines()[0]
    assert "N=3" in line and "N=4" in line and "N=5" in line
    assert line.count("inv") == sum(1 for i in scan.n_inverted if i)


def test_a_scan_that_read_nothing_is_not_clean():
    """Unchecked is not clean -- the same reason ``skipped`` is reported at all."""
    mesh = _folded_quad()
    empty = quadmesh.order_scan(mesh, orders=[])
    assert empty.orders == ()
    assert not empty.clean
    assert "not checked" in quadmesh.quality.format_scan(empty, mesh.order)
    with pytest.raises(ValueError, match="no order was sampled"):
        _ = empty.worst


def test_the_budget_constant_is_read_at_call_time():
    """Assigning ``core.quality.SCAN_BUDGET`` must take effect.

    Captured as a default argument it would bind at import and silently ignore the
    assignment -- which it did, and which invalidated a whole round of my own
    measurements before it was noticed."""
    import nekmeshpy.core.quality as core_quality
    mesh = _folded_quad()
    was = core_quality.SCAN_BUDGET
    try:
        core_quality.SCAN_BUDGET = 0
        assert quadmesh.order_scan(mesh).orders == ()
        core_quality.SCAN_BUDGET = 10 ** 12
        assert quadmesh.order_scan(mesh).orders == (SCAN_ORDER,)
    finally:
        core_quality.SCAN_BUDGET = was


def test_the_solver_order_constant_is_read_at_call_time():
    """Same for ``SCAN_ORDER`` -- and it matters more, since matching it to your own
    solver is the single thing a user is most likely to change."""
    import nekmeshpy.core.quality as core_quality
    mesh = _folded_quad()
    was = core_quality.SCAN_ORDER
    try:
        core_quality.SCAN_ORDER = 9        # not the default, so this can only
        assert quadmesh.order_scan(mesh).orders == (9,)   # pass if it is read live
    finally:
        core_quality.SCAN_ORDER = was
    assert quadmesh.order_scan(mesh).orders == (SCAN_ORDER,)


# -- the linear (.re2) reading: a different map, not a coarser sampling ---------
#
# ``.re2`` has no curved format at any stored order -- every element it writes is the
# straight-sided hex through its 8 corners alone.  That is not "the mesh's own order
# sampled coarser": it is a genuinely different map, one that discards the interior
# nodes rather than reading fewer of them.  So an element can be clean at its own
# curved order and still invert once flattened for export, when it is valid only
# *because of* the curvature that map throws away.
#
# ``CURVED_VALID_LINEAR_INVERTED`` is a real element out of an earlier
# ``examples/wire_coil.py`` (one of four it then wrote with a negative corner reading;
# the current mesher no longer does), frozen here rather than rebuilt from the example
# so this test does not depend on wire_coil's geometry at all. Order 2,
# 27 nodes in the usual tensor lattice order (:func:`corner_indices` picks out the 8
# corners). It sits on the wire's own ``coil`` surface, where a tight O-grid transition
# is valid only because its curved interior bows around a fold the straight corners
# alone cannot avoid.
CURVED_VALID_LINEAR_INVERTED = np.array([
    [-0.36187459, -0.28843946, 0.26982383],
    [-0.29109897, -0.35172468, 0.29940227],
    [-0.21346698, -0.39655513, 0.32834877],
    [-0.41631322, -0.20207384, 0.25418694],
    [-0.35702470, -0.28457376, 0.28479682],
    [-0.28714464, -0.34694680, 0.31437525],
    [-0.45488458, -0.08503267, 0.23529412],
    [-0.41073375, -0.19936563, 0.26915993],
    [-0.35217482, -0.28070807, 0.29976981],
    [-0.37486961, -0.29879740, 0.26982383],
    [-0.30288164, -0.36596127, 0.30393254],
    [-0.22210060, -0.41259372, 0.33652813],
    [-0.43126315, -0.20933037, 0.25418694],
    [-0.37147582, -0.29609232, 0.28932709],
    [-0.29875814, -0.36097899, 0.32255462],
    [-0.47121962, -0.08808621, 0.23529412],
    [-0.42735882, -0.20743525, 0.27369020],
    [-0.36641846, -0.29206124, 0.30794917],
    [-0.38786463, -0.30915535, 0.26982383],
    [-0.31466431, -0.38019786, 0.30846281],
    [-0.23073423, -0.42863231, 0.34470750],
    [-0.44621308, -0.21658690, 0.25418694],
    [-0.38592693, -0.30761087, 0.29385737],
    [-0.31037165, -0.37501118, 0.33073398],
    [-0.48755466, -0.09113976, 0.23529412],
    [-0.44398388, -0.21550488, 0.27822048],
    [-0.38066209, -0.30341442, 0.31612853],
])[None, :, :]


def _corner_hex():
    """``CURVED_VALID_LINEAR_INVERTED``'s 8 corners and their flat ``(1,8)`` hex
    connectivity into its own 27-point block -- everything :func:`corner_scaled_jacobian`
    needs, with no HexMesh container built at all (a factory would only ever
    straight-subdivide between given points, so there is no way to hand one this
    element's genuine curvature without the same low-level block this test already
    has)."""
    pts = CURVED_VALID_LINEAR_INVERTED[0]
    hexes = corner_indices(2, 3)[None, :]
    return pts, hexes


def test_curved_clean_element_reads_inverted_at_its_corners_alone():
    """The defect ``quality_summary`` now watches for: an element the curved order
    reads as sound, whose 8 corners alone -- the geometry ``.re2`` actually exports --
    do not agree."""
    pts, hexes = _corner_hex()
    corner_sj = corner_scaled_jacobian(pts, hexes)
    assert corner_sj[0] == pytest.approx(-0.03535304, abs=1e-6)

    curved_sj = sampled_scaled_jacobian(CURVED_VALID_LINEAR_INVERTED, 2, 2, 3)
    assert curved_sj[0] == pytest.approx(0.10566487, abs=1e-6)

    assert corner_sj[0] < 0.0 < curved_sj[0]


def test_a_sound_curved_mesh_has_no_corner_linear_disagreement():
    """The ordinary case: a mesh that is not depending on its own curvature to stay
    valid reads clean both ways, and ``quality_summary`` says nothing about it."""
    section = quadmesh.ogrid(linemesh.circle(1.0, 8, order=2), 2, 2)
    mesh = hexmesh.extrude(section, 1.0, 2)
    linear = corner_scaled_jacobian(mesh.points, mesh.corners)
    assert np.all(linear > 0.0)


def test_quality_summary_only_checks_corners_above_order_1():
    """At order 1 the curved and linear readings are the same map -- there is nothing
    a second check could catch that the first did not already, so it is skipped."""
    section = quadmesh.ogrid(linemesh.circle(1.0, 8, order=1), 2, 2)
    mesh = hexmesh.extrude(section, 1.0, 2)
    assert mesh.order == 1
    own = hexmesh.quality_summary(mesh)
    assert np.array_equal(corner_scaled_jacobian(mesh.points, mesh.corners),
                          hexmesh.scaled_jacobian(mesh))
    assert own.n_inverted == 0


def test_quality_summary_warns_only_when_linear_disagrees(caplog):
    """A clean curved mesh with a clean linear reading too must stay quiet -- the
    warning is for the disagreement, not for the check having run at all."""
    section = quadmesh.ogrid(linemesh.circle(1.0, 8, order=2), 2, 2)
    mesh = hexmesh.extrude(section, 1.0, 2)
    with caplog.at_level("WARNING"):
        hexmesh.quality_summary(mesh)
    assert "linear corners" not in caplog.text


def test_corner_summary_matches_corner_scaled_jacobian():
    section = quadmesh.ogrid(linemesh.circle(1.0, 8, order=2), 2, 2)
    mesh = hexmesh.extrude(section, 1.0, 2)
    stats = hexmesh.corner_summary(mesh)
    sj = hexmesh.corner_scaled_jacobian(mesh)
    assert stats.min == pytest.approx(float(np.min(sj)))
    assert stats.n_elements == mesh.n_hexes
    assert stats.n_inverted == int(np.sum(sj <= 0))


def test_format_linear_reports_the_warning_line():
    stats = hexmesh.quality.corner_summary(*_corner_hex())
    text = hexmesh.quality.format_linear(stats, mesh_order=2)
    assert "WARNING" in text
    assert "invert once flattened" in text
    assert "1 inverted" in text


def test_format_linear_is_quiet_when_clean():
    section = quadmesh.ogrid(linemesh.circle(1.0, 8, order=2), 2, 2)
    mesh = hexmesh.extrude(section, 1.0, 2)
    stats = hexmesh.corner_summary(mesh)
    text = hexmesh.quality.format_linear(stats, mesh_order=mesh.order)
    assert "WARNING" not in text


def test_report_includes_the_linear_line_above_order_1():
    section2 = quadmesh.ogrid(linemesh.circle(1.0, 8, order=2), 2, 2)
    mesh2 = hexmesh.extrude(section2, 1.0, 2)
    assert "linear (.re2)" in hexmesh.report(mesh2)

    section1 = quadmesh.ogrid(linemesh.circle(1.0, 8, order=1), 2, 2)
    mesh1 = hexmesh.extrude(section1, 1.0, 2)
    assert "linear (.re2)" not in hexmesh.report(mesh1)


# -- the deeper trap: valid corners, folded trilinear interior -----------------
#
# ``corner_scaled_jacobian`` reads only the 8 vertices, but a trilinear hex's
# Jacobian is a polynomial in each direction -- it can still fold *between* the
# corners while every corner reads positive.  That gap is not hypothetical: a real
# solver's own geometry generation (built from exactly the corners ``.re2``
# exports, resampled at its own working polynomial order) failed on an element
# nekmeshpy's own corner check called clean.
#
# ``CORNER_CLEAN_TRILINEAR_FOLDED`` is that element, frozen the same way as
# ``CURVED_VALID_LINEAR_INVERTED`` above (a real element from an earlier
# ``examples/wire_coil.py``, order 2, 27-node lattice) -- but this one is a
# *different* fold: clean at every
# one of its 8 corners (+0.0053), and still negative once the trilinear map through
# those same 8 corners is resampled at order 7 (-0.0151), the polynomial order
# ``kgj.par`` (the real case this was found against) actually runs at.
CORNER_CLEAN_TRILINEAR_FOLDED = np.array([
    [-0.11098995, 0.10675781, -0.20915033],
    [-0.11098995, 0.10675781, 0.01307190],
    [-0.11098995, 0.10675781, 0.23529412],
    [-0.08737652, 0.12681224, -0.20915033],
    [-0.08737652, 0.12681224, 0.01307190],
    [-0.08770648, 0.12658426, 0.23529412],
    [-0.06096442, 0.14141902, -0.20915033],
    [-0.06096442, 0.14141902, 0.01307190],
    [-0.06096442, 0.14141902, 0.23529412],
    [-0.11482056, 0.16567083, -0.03182936],
    [-0.13601594, 0.15454366, 0.10200601],
    [-0.15818433, 0.14117630, 0.23398978],
    [-0.08168454, 0.18458897, -0.03827359],
    [-0.10408723, 0.17772977, 0.09541993],
    [-0.12782595, 0.16911485, 0.22722894],
    [-0.04631531, 0.19671869, -0.04459880],
    [-0.06916597, 0.19408496, 0.08897571],
    [-0.09359410, 0.19016819, 0.22064286],
    [-0.11865116, 0.22458384, 0.14549160],
    [-0.16104194, 0.20232950, 0.19094013],
    [-0.20537871, 0.17559479, 0.23268543],
    [-0.07599256, 0.24236570, 0.13260316],
    [-0.12079795, 0.22864730, 0.17776797],
    [-0.16827539, 0.21141745, 0.21916376],
    [-0.03166619, 0.25201836, 0.11995272],
    [-0.07736751, 0.24675089, 0.16487952],
    [-0.12622378, 0.23891735, 0.20599160],
])[None, :, :]


def test_linear_at_order_1_is_exactly_the_corner_reading():
    """``linear_scaled_jacobian(mesh, order=1)`` must be ``corner_scaled_jacobian``,
    bit for bit: it is the same map (the trilinear hex through the 8 corners), read
    at the same 8 points."""
    section = quadmesh.ogrid(linemesh.circle(1.0, 8, order=2), 2, 2)
    mesh = hexmesh.extrude(section, 1.0, 2)
    assert np.array_equal(hexmesh.linear_scaled_jacobian(mesh, order=1),
                          hexmesh.corner_scaled_jacobian(mesh))


def test_clean_corners_can_still_fold_between_them():
    """The gap ``corner_scaled_jacobian`` cannot see: an element every one of whose
    8 vertices reads positive, whose trilinear interior still folds once resampled
    at the order a real solver actually runs."""
    pts, hexes = CORNER_CLEAN_TRILINEAR_FOLDED[0], corner_indices(2, 3)[None, :]
    assert corner_scaled_jacobian(pts, hexes)[0] > 0.0

    corners_nek_order = pts[corner_indices(2, 3)]      # (8,3), the 8 actual corners
    tensor_block = np.empty((1, 8, 3))
    tensor_block[0, corner_indices(1, 3), :] = corners_nek_order
    sj7 = sampled_scaled_jacobian(tensor_block, 1, 7, 3)
    assert sj7[0] == pytest.approx(-0.01509415, abs=1e-6)


def test_linear_order_scan_needs_no_mesh_order_floor():
    """Unlike ``order_scan``, the trilinear map is not stored at any order, so
    ``order=1`` is always a legitimate reading of it -- even on a curved mesh."""
    section = quadmesh.ogrid(linemesh.circle(1.0, 8, order=2), 2, 2)
    mesh = hexmesh.extrude(section, 1.0, 2)
    scan = hexmesh.linear_order_scan(mesh, orders=[1, 3], budget=10 ** 12)
    assert scan.orders == (1, 3)


def test_linear_order_scan_refuses_order_below_one():
    section = quadmesh.ogrid(linemesh.circle(1.0, 8, order=1), 2, 2)
    mesh = hexmesh.extrude(section, 1.0, 2)
    with pytest.raises(ValueError, match="order must be >= 1"):
        hexmesh.linear_order_scan(mesh, orders=[0])


def test_linear_order_scan_defaults_to_the_solver_order():
    section = quadmesh.ogrid(linemesh.circle(1.0, 8, order=2), 2, 2)
    mesh = hexmesh.extrude(section, 1.0, 2)
    assert hexmesh.linear_order_scan(mesh, budget=10 ** 12).orders == (SCAN_ORDER,)


def test_linear_order_scan_respects_the_budget():
    section = quadmesh.ogrid(linemesh.circle(1.0, 8, order=2), 2, 2)
    mesh = hexmesh.extrude(section, 1.0, 8)
    scan = hexmesh.linear_order_scan(mesh, budget=1)
    assert scan.orders == ()
    assert scan.skipped == (SCAN_ORDER,)
    assert not scan.clean


def test_report_includes_linear_sampling_above_order_1():
    section2 = quadmesh.ogrid(linemesh.circle(1.0, 8, order=2), 2, 2)
    mesh2 = hexmesh.extrude(section2, 1.0, 2)
    assert "linear sampling" in hexmesh.report(mesh2)

    section1 = quadmesh.ogrid(linemesh.circle(1.0, 8, order=1), 2, 2)
    mesh1 = hexmesh.extrude(section1, 1.0, 2)
    assert "linear sampling" not in hexmesh.report(mesh1)


def test_report_warns_on_a_linear_sampling_fold(caplog):
    """A sound mesh must stay quiet -- this only pins that the *sound* case does not
    spuriously warn; the real fold is covered directly above."""
    section = quadmesh.ogrid(linemesh.circle(1.0, 8, order=2), 2, 2)
    mesh = hexmesh.extrude(section, 1.0, 2)
    with caplog.at_level("WARNING"):
        hexmesh.report(mesh)
    assert "trilinear geometry is resampled" not in caplog.text
