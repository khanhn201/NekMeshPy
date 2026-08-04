"""Unit tests for ``LineMesh.loft_curve`` -- ``loft`` with the profiles **evaluated**
from an analytic parametrization instead of handed in -- and its grading helper
``LineMesh.arclength_fractions``.

The contract ``loft_curve`` exists to enforce: **every** node of the chain (the ``n+1``
corners *and* each element's ``order-1`` private ``interior`` GLL nodes) is placed by
evaluating the parametrization, so nothing ever lands on a chord.  That is exactly
what ``LineMesh.loft`` of the same sampled points cannot do -- the direct inverse of
the defect this factory closes.  ``fractions`` are the **parameter values themselves**
-- they reach ``f`` verbatim, with no normalization and no remapping, so they both
state the domain and grade the nodes within it; ``arclength_fractions`` is the
separate, explicit inversion that turns "evenly spaced by arc length" into such a
grading, and it perturbs only *where along* the curve the nodes sit.

``loop`` is the same flag ``loft`` carries, taken by the **trailing wrap value**:
``n+1`` fractions whose last maps back onto the first point, so the seam element
has a far parameter of its own and is meshed on the curve like every other one.
"""

import numpy as np
import pytest

from nekmeshpy import LineMesh

R = 0.5


def _collar(t):
    """The T-junction collar: the intersection of two equal-radius cylinders, i.e.
    ``x^2 + y^2 = R^2`` *and* ``y^2 + z^2 = R^2``.  Not a circle in any plane, and
    not constant speed (``|p'| = R*sqrt(1 + cos^2 t)``)."""
    t = np.asarray(t, dtype=float)
    return np.column_stack([-R * np.sin(t), R * np.cos(t), R * np.sin(t)])


def _circle_f(radius=1.25):
    def f(t):
        t = np.asarray(t, dtype=float)
        return np.column_stack([radius * np.cos(t), radius * np.sin(t),
                                np.zeros_like(t)])
    return f


def _uniform(n, t0=0.0, t1=np.pi):
    """The plain ungraded ``fractions`` for ``n`` elements over ``[t0, t1]`` -- the
    parameter values themselves, defaulting to the collar's own ``t: 0 -> pi``."""
    return np.linspace(t0, t1, n + 1)


def _all_nodes(lm):
    """Every node the chain stores: the corners plus each element's interior."""
    blocks = [np.asarray(lm.points)]
    if lm.interior.size:
        blocks.append(lm.interior.reshape(-1, 3))
    return np.vstack(blocks)


def _collar_residual(P):
    """Max violation of the two cylinder equations over the node array."""
    P = np.asarray(P, dtype=float)
    return max(np.max(np.abs(P[:, 1] ** 2 + P[:, 2] ** 2 - R ** 2)),
               np.max(np.abs(P[:, 0] ** 2 + P[:, 1] ** 2 - R ** 2)))


def _collar_t(P):
    """The collar parameter ``t in [0, pi]`` a node array sits at (``y = R cos t``)."""
    P = np.asarray(P, dtype=float)
    return np.arccos(np.clip(P[:, 1] / R, -1.0, 1.0))


# -- geometric truth: every node sits on the true curve ----------------------

@pytest.mark.parametrize("order", [1, 2, 3, 4])
def test_every_node_lies_on_the_true_collar_curve(order):
    lm = LineMesh.loft_curve(_collar, _uniform(6), order=order)
    assert lm.order == order
    # corners and private interior nodes are checked separately: both must be exact
    assert _collar_residual(lm.points) < 1e-13
    if order > 1:
        assert lm.interior.size > 0
        assert _collar_residual(lm.interior.reshape(-1, 3)) < 1e-13
    assert _collar_residual(_all_nodes(lm)) < 1e-13


@pytest.mark.parametrize("order", [2, 3, 4])
def test_loft_of_sampled_points_loses_the_curve(order):
    # the regression this factory closes: sampling the curve into an array and
    # calling ``loft`` straight-subdivides between the samples, so the interior
    # nodes fall off the collar by a *visible* amount -- while ``loft_curve`` stays exact.
    n = 6
    exact = LineMesh.loft_curve(_collar, _uniform(n), order=order)
    lofted = LineMesh.loft(_collar(np.linspace(0.0, np.pi, n + 1)), order=order)

    assert np.allclose(lofted.points, exact.points, atol=1e-14)   # corners agree
    assert _collar_residual(lofted.interior.reshape(-1, 3)) > 1e-3
    assert _collar_residual(_all_nodes(exact)) < 1e-12


# -- the parameter values reach ``f`` verbatim -------------------------------

@pytest.mark.parametrize("order", [1, 2, 3])
def test_fractions_are_passed_to_the_callable_unremapped(order):
    # the whole contract of the argument: ``fractions`` *are* the parameters, so a
    # corner is bit-exactly ``f(fractions[k])`` -- no normalization, no rescaling
    # onto some other interval, and ``f`` called once with the whole node lattice.
    seen = []

    def record(t):
        seen.append(np.array(t, dtype=float))
        return _collar(t)

    ts = np.array([0.3, 1.1, 1.9, 2.7])          # arbitrary, non-uniform, not in [0,1]
    lm = LineMesh.loft_curve(record, ts, order=order)

    assert len(seen) == 1                        # one call, the whole lattice
    assert seen[0].shape == (3 * order + 1,)     # n*order + 1 nodes
    assert np.array_equal(seen[0][::order], ts)  # the corners are the values given
    assert np.array_equal(np.asarray(lm.points), _collar(ts))     # bit-exact


# -- shapes / counts ---------------------------------------------------------

@pytest.mark.parametrize("order", [1, 2, 3, 5])
@pytest.mark.parametrize("n", [1, 4, 7])
def test_shapes(n, order):
    lm = LineMesh.loft_curve(_collar, _uniform(n), order=order)
    assert np.asarray(lm.points).shape == (n + 1, 3)
    assert lm.interior.shape == (n, order - 1, 3)
    assert lm.n_lines == n
    assert lm.lines.tolist() == [[i, i + 1] for i in range(n)]
    assert lm.boundary_points().size == 2            # open chain: two degree-1 ends


def test_order_one_interior_is_empty():
    lm = LineMesh.loft_curve(_collar, _uniform(5))
    assert lm.order == 1
    assert lm.interior.shape == (5, 0, 3)
    assert lm.interior.size == 0


# -- agreement with the analytic special case ``arc`` ------------------------

@pytest.mark.parametrize("order", [1, 2, 4])
def test_curve_reproduces_arc_for_a_circular_parametrization(order):
    # ``arc`` is the circular special case; ``loft_curve`` must reproduce it to machine
    # precision (not bitwise -- ``arc`` places its nodes without the generic path).
    radius, n, t0, t1 = 1.25, 5, 0.2, 1.7
    a = LineMesh.arc(radius, n, start_theta=t0, end_theta=t1, order=order)
    c = LineMesh.loft_curve(_circle_f(radius), _uniform(n, t0, t1), order=order)
    assert np.allclose(c.points, a.points, rtol=0.0, atol=1e-15)
    assert c.interior.shape == a.interior.shape
    if order > 1:
        assert np.allclose(c.interior, a.interior, rtol=0.0, atol=1e-15)


# -- grading: ``fractions`` --------------------------------------------------

@pytest.mark.parametrize("order", [2, 3, 4])
def test_graded_fractions_place_every_node_in_its_own_element_span(order):
    # the capability the old ``n``-only signature could not express: a non-uniform
    # grading, honored *per element* -- element i's private interior nodes ride the
    # GLL nodes of its own ``fractions[i]..fractions[i+1]`` span, on the true curve.
    n = 5
    fr = np.pi * np.linspace(0.0, 1.0, n + 1) ** 2   # clustered at the t = 0 end
    lm = LineMesh.loft_curve(_collar, fr, order=order)

    assert lm.n_lines == n
    assert _collar_residual(_all_nodes(lm)) < 1e-13

    t_corner = _collar_t(lm.points)
    t_interior = _collar_t(lm.interior.reshape(-1, 3)).reshape(n, order - 1)
    for i in range(n):
        lo, hi = t_corner[i], t_corner[i + 1]
        assert lo < hi
        # *strictly* inside its own element -- never in a neighbour's span
        assert np.all(t_interior[i] > lo + 1e-12)
        assert np.all(t_interior[i] < hi - 1e-12)
        assert np.all(np.diff(t_interior[i]) > 0.0)


def test_graded_fractions_set_the_corner_spacing():
    # the corners land exactly where the grading asks: t_k = fractions[k]
    n = 6
    fr = np.pi * np.linspace(0.0, 1.0, n + 1) ** 2
    lm = LineMesh.loft_curve(_collar, fr, order=3)
    assert np.allclose(_collar_t(lm.points), fr, atol=1e-12)
    # and the grading really is non-uniform: the last span dwarfs the first
    d = np.diff(_collar_t(lm.points))
    assert d[-1] / d[0] > 5.0


def test_graded_and_uniform_fractions_agree_when_the_grading_is_uniform():
    a = LineMesh.loft_curve(_collar, _uniform(4), order=3)
    b = LineMesh.loft_curve(_collar, np.linspace(0.0, np.pi, 5), order=3)
    assert np.allclose(a.points, b.points, rtol=0.0, atol=0.0)
    assert np.allclose(a.interior, b.interior, rtol=0.0, atol=0.0)


# -- spacing: ``arclength_fractions`` ----------------------------------------

def test_arclength_fractions_shape_and_ends():
    fr = LineMesh.arclength_fractions(_collar, 7, t_range=(0.0, np.pi))
    assert fr.shape == (8,)
    # the returned values are the parameters themselves, spanning ``t_range``
    assert fr[0] == pytest.approx(0.0, abs=1e-15)
    assert fr[-1] == pytest.approx(np.pi, abs=1e-15)
    assert np.all(np.diff(fr) > 0.0)             # strictly ascending


def test_arclength_fractions_descend_for_a_descending_t_range():
    # the values run from ``t_range[0]`` to ``t_range[1]`` whichever way the range
    # goes, so the helper needs no special handling and its output stays a valid
    # ``fractions`` array -- a descending one, which meshes the curve backwards
    fr = LineMesh.arclength_fractions(_collar, 7, t_range=(np.pi, 0.0))
    assert np.all(np.diff(fr) < 0.0)
    assert fr[0] == pytest.approx(np.pi, abs=1e-15)
    assert fr[-1] == pytest.approx(0.0, abs=1e-15)
    # and it does drive ``loft_curve`` backwards, ends first
    lm = LineMesh.loft_curve(_collar, fr)
    assert np.allclose(lm.points[0], _collar(np.array([np.pi]))[0], atol=1e-14)
    assert np.allclose(lm.points[-1], _collar(np.array([0.0]))[0], atol=1e-14)


def test_uniform_and_arclength_differ_for_a_non_constant_speed_curve():
    tr = (0.0, np.pi)
    u = LineMesh.loft_curve(_collar, _uniform(8))
    a = LineMesh.loft_curve(_collar, LineMesh.arclength_fractions(_collar, 8, t_range=tr))
    assert np.max(np.abs(u.points - a.points)) > 1e-3
    # both still lie exactly on the curve -- spacing moves nodes *along* it only
    assert _collar_residual(_all_nodes(u)) < 1e-13
    assert _collar_residual(_all_nodes(a)) < 1e-13
    # the two ends are pinned regardless of spacing
    assert np.allclose(u.points[[0, -1]], a.points[[0, -1]], atol=1e-12)


@pytest.mark.parametrize("order", [1, 3])
def test_uniform_and_arclength_agree_for_a_constant_speed_curve(order):
    # a circle parametrized by angle has constant speed, so arc length and parameter
    # are proportional and the two gradings must place the same nodes
    f = _circle_f(1.25)
    tr = (0.0, 1.9)
    u = LineMesh.loft_curve(f, _uniform(7, *tr), order=order)
    a = LineMesh.loft_curve(f, LineMesh.arclength_fractions(f, 7, t_range=tr,
                                                       samples=20001),
                       order=order)
    assert np.allclose(u.points, a.points, atol=1e-12)
    if order > 1:
        assert np.allclose(u.interior, a.interior, atol=1e-12)


def test_arclength_fractions_are_even_in_arc_length():
    # consecutive corner-to-corner chords are near-equal (loose: the helper inverts a
    # chord-length table, so it is only as even as that discretization)
    tr = (0.0, np.pi)
    lm = LineMesh.loft_curve(_collar,
                        LineMesh.arclength_fractions(_collar, 10, t_range=tr,
                                                     samples=20001))
    d = np.linalg.norm(np.diff(np.asarray(lm.points), axis=0), axis=1)
    spread = np.max(np.abs(d - d.mean())) / d.mean()
    assert spread < 2e-2      # chords, not arcs: curvature keeps this from being 0
    # and the uniform-parameter chain is an order of magnitude less even, so this
    # pins the grading rule rather than the curve
    du = np.linalg.norm(np.diff(np.asarray(
        LineMesh.loft_curve(_collar, _uniform(10)).points), axis=0), axis=1)
    assert np.max(np.abs(du - du.mean())) / du.mean() > 10 * spread


def test_arclength_graded_nodes_still_lie_on_the_curve_at_high_order():
    # the point of splitting the inversion out: the table decides *where along* the
    # curve the nodes sit, never whether they are on it
    lm = LineMesh.loft_curve(_collar,
                        LineMesh.arclength_fractions(_collar, 6,
                                                     t_range=(0.0, np.pi)),
                        order=4)
    assert _collar_residual(lm.points) < 1e-13
    assert _collar_residual(lm.interior.reshape(-1, 3)) < 1e-13


# -- traversal direction and partial spans -----------------------------------

def test_descending_fractions_run_the_curve_backwards():
    # descending ``fractions`` are *the* way to reverse a parametric chain: there is
    # no direction flag, the values simply run the other way.
    fwd = LineMesh.loft_curve(_collar, _uniform(6), order=3)
    bwd = LineMesh.loft_curve(_collar, _uniform(6)[::-1], order=3)
    assert np.allclose(bwd.points, fwd.points[::-1], atol=1e-13)
    # the interior nodes reverse on both axes, exactly as ``reverse`` would carry them
    assert np.allclose(bwd.interior, fwd.interior[::-1, ::-1, :], atol=1e-13)
    assert _collar_residual(_all_nodes(bwd)) < 1e-13
    # ... which is to say: it agrees with ``LineMesh.reverse`` of the forward chain
    rev = fwd.reverse()
    assert np.allclose(bwd.points, rev.points, atol=1e-13)
    assert np.allclose(bwd.interior, rev.interior, atol=1e-13)


def test_partial_span_of_fractions_meshes_only_that_span():
    lm = LineMesh.loft_curve(_collar, _uniform(4, 0.5, 1.1))
    assert np.allclose(lm.points[0], _collar(np.array([0.5]))[0], atol=1e-14)
    assert np.allclose(lm.points[-1], _collar(np.array([1.1]))[0], atol=1e-14)
    # and nothing outside it: every node stays inside the requested parameter span
    t = _collar_t(_all_nodes(lm))
    assert np.all(t > 0.5 - 1e-12)
    assert np.all(t < 1.1 + 1e-12)


# -- loop=True: the closed sweep ---------------------------------------------
#
# ``loft_curve`` takes the same ``loop`` flag as ``loft``, and takes it by the
# **trailing wrap value**: the caller passes ``n+1`` fractions whose last maps back
# onto the first point, so the seam element has a far parameter of its own to be
# evaluated at.  That is the whole reason the period is not inferred from ``n``
# fractions -- without ``fr[n]`` the seam element would be the one element of the ring
# that got straight-subdivided, which is the exact defect this factory exists to close.

def _ring_fractions(n):
    """The wrap-terminated fractions for an ``n``-element ring of a ``2*pi``-periodic
    parametrization: ``n+1`` values, the last one back at the first point."""
    return np.linspace(0.0, 2.0 * np.pi, n + 1)


@pytest.mark.parametrize("order", [1, 2, 3, 4])
def test_loop_gives_a_ring_with_no_free_end(order):
    n = 8
    ring = LineMesh.loft_curve(_circle_f(), _ring_fractions(n), loop=True, order=order)
    # n points, not n+1: the wrap value is the seam's far end, not a point of its own
    assert ring.points.shape == (n, 3)
    assert ring.n_lines == n
    assert ring.lines[-1].tolist() == [n - 1, 0]     # the seam rung, appended once
    assert ring.boundary_points().size == 0          # no degree-1 end anywhere


@pytest.mark.parametrize("order", [2, 3, 4])
def test_loop_seam_element_rides_the_true_curve(order):
    # the property the trailing wrap value buys: the seam element's private interior
    # is evaluated on ``fr[n-1]..fr[n]`` like every other element's, so it is on the
    # circle rather than on the chord that closes it.
    n, radius = 8, 1.25
    ring = LineMesh.loft_curve(_circle_f(radius), _ring_fractions(n),
                               loop=True, order=order)
    r = np.linalg.norm(_all_nodes(ring)[:, :2], axis=1)
    assert np.max(np.abs(r - radius)) < 1e-13

    seam = ring.interior[-1]                          # (order-1, 3)
    assert np.max(np.abs(np.linalg.norm(seam[:, :2], axis=1) - radius)) < 1e-13
    # and it genuinely spans the last interval: its nodes sit past the last corner
    assert np.all(np.arctan2(seam[:, 1], seam[:, 0]) % (2.0 * np.pi)
                  > 2.0 * np.pi * (n - 1) / n)
    # the straight closing chord would be visibly inside the circle
    chord = 0.5 * (ring.points[-1] + ring.points[0])
    assert radius - np.linalg.norm(chord[:2]) > 1e-2


@pytest.mark.parametrize("order", [1, 2, 3])
def test_loop_is_the_open_chain_minus_its_duplicate_end(order):
    n = 6
    fr = _ring_fractions(n)
    ring = LineMesh.loft_curve(_circle_f(), fr, loop=True, order=order)
    chain = LineMesh.loft_curve(_circle_f(), fr, order=order)
    assert chain.points.shape == (n + 1, 3)
    assert chain.boundary_points().size == 2
    # bit-exact: closing the sweep drops the duplicate point, it does not move any node
    assert np.array_equal(ring.points, chain.points[:-1])
    assert np.array_equal(ring.interior, chain.interior)


def test_loop_at_order_one_is_loft_of_the_corners():
    n = 7
    fr = _ring_fractions(n)
    ring = LineMesh.loft_curve(_circle_f(), fr, loop=True)
    direct = LineMesh.loft(_circle_f()(fr[:-1]), loop=True)
    assert np.array_equal(ring.points, direct.points)
    assert np.array_equal(ring.lines, direct.lines)


@pytest.mark.parametrize("order", [1, 2, 3, 4])
def test_loop_reproduces_the_circle_factory(order):
    # the circular special case of a closed parametrization: same nodes as the
    # analytic ``circle`` factory, which places them without evaluating a callable
    n, radius = 12, 1.25
    ring = LineMesh.loft_curve(_circle_f(radius), _ring_fractions(n),
                               loop=True, order=order)
    ref = LineMesh.circle(radius, n, order=order)
    assert np.array_equal(ring.lines, ref.lines)
    assert np.allclose(ring.points, ref.points, atol=1e-15)
    assert np.allclose(ring.interior, ref.interior, atol=1e-15)


def test_loop_tags_one_element_per_line_including_the_seam():
    n = 5
    tags = ["a", "a", "b", "b", "seam"]
    ring = LineMesh.loft_curve(_circle_f(), _ring_fractions(n), loop=True,
                               order=2, element_tags=tags)
    assert ring.element_tags.dense(ring.n_lines).tolist() == tags        # n tags, not n+1
    with pytest.raises(ValueError, match="element_tags length"):
        LineMesh.loft_curve(_circle_f(), _ring_fractions(n), loop=True,
                            element_tags=tags + ["extra"])


def test_loop_rejects_a_parametrization_that_does_not_close():
    # a half turn: f(fr[-1]) is nowhere near f(fr[0]), so welding it would invent
    # geometry.  Loud rather than silent -- the caller forgot the wrap value.
    with pytest.raises(ValueError, match="last fraction to map back"):
        LineMesh.loft_curve(_circle_f(), np.linspace(0.0, np.pi, 7), loop=True)


def test_loop_rejects_fewer_than_three_fractions():
    # two fractions is one element, and its far end is the wrap -- there is no ring
    with pytest.raises(ValueError, match="at least 3 fractions"):
        LineMesh.loft_curve(_circle_f(), np.array([0.0, 2.0 * np.pi]), loop=True)


# -- tags --------------------------------------------------------------------

def test_element_tags_land_on_the_elements():
    tags = ["a", "a", "b", "b", "b"]
    lm = LineMesh.loft_curve(_collar, _uniform(5), order=2, element_tags=tags)
    assert lm.element_tags.dense(lm.n_lines).tolist() == tags
    assert lm.element_group_tags == ["a", "b"]


def test_untagged_curve_stays_untagged():
    assert LineMesh.loft_curve(_collar, _uniform(3)).element_group_tags == []


def test_element_tags_length_validated():
    with pytest.raises(ValueError, match="element_tags length"):
        LineMesh.loft_curve(_collar, _uniform(4), element_tags=["a", "b"])


# -- rejections --------------------------------------------------------------

@pytest.mark.parametrize("fractions", [0.0, [0.25], []])
def test_rejects_fewer_than_two_fractions(fractions):
    with pytest.raises(ValueError, match="at least 2 fractions"):
        LineMesh.loft_curve(_collar, np.asarray(fractions, dtype=float))


def test_rejects_callable_returning_wrong_shape():
    def bad(t):
        return np.column_stack([t, t])           # (K,2), not (K,3)
    with pytest.raises(ValueError, match=r"must return \(len\(t\), 3\) points"):
        LineMesh.loft_curve(bad, _uniform(4, 0.0, 1.0))


@pytest.mark.parametrize("n", [0, -3])
def test_arclength_fractions_rejects_non_positive_n(n):
    with pytest.raises(ValueError, match="n >= 1"):
        LineMesh.arclength_fractions(_collar, n, t_range=(0.0, np.pi))


def test_arclength_fractions_rejects_too_few_samples():
    with pytest.raises(ValueError, match="samples >= 2"):
        LineMesh.arclength_fractions(_collar, 4, t_range=(0.0, np.pi), samples=1)


def test_arclength_fractions_rejects_degenerate_t_range():
    with pytest.raises(ValueError, match="endpoints to differ"):
        LineMesh.arclength_fractions(_collar, 4, t_range=(0.7, 0.7))


def test_arclength_fractions_rejects_a_zero_length_curve():
    def point(t):
        t = np.asarray(t, dtype=float)
        return np.column_stack([np.ones_like(t), np.zeros_like(t), np.zeros_like(t)])
    with pytest.raises(ValueError, match="zero length"):
        LineMesh.arclength_fractions(point, 4, t_range=(0.0, 1.0))
    # the same degenerate callable is *accepted* by ``loft_curve`` itself -- the rejection
    # is about the arc-length inversion, not about the curve
    assert LineMesh.loft_curve(point, _uniform(4, 0.0, 1.0)).n_lines == 4
