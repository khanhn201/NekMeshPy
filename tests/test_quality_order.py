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
    scaled_jacobian,
    tensor_nodes,
)
from nekmeshpy.core.quality import SCAN_ORDER
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
