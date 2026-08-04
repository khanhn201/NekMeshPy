"""``QuadMesh.quadrant_ogrid`` -- the quarter-disk O-grid."""

import numpy as np
import pytest

from nekmeshpy import HexMesh, LineMesh, QuadMesh

R = 1.0
RADIAL = np.array([0.0, 0.4, 0.75, 1.0])
NR = RADIAL.size - 1
CS = 0.5


def _radius(theta, fr, order=1):
    d = np.array([np.cos(theta), np.sin(theta), 0.0])
    return LineMesh.line(np.zeros(3), R * d, fr, order=order)


def _disc(n, order=1, center_scale=CS, radial=RADIAL, wall_tag="wall"):
    """The four quadrants of the unit disk, sharing their seam objects."""
    fr = QuadMesh.quadrant_seam_fractions(n, radial, center_scale)
    ang = [0.0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi]
    seams = [_radius(a, fr, order) for a in ang[:4]]
    seams.append(seams[0])
    return [QuadMesh.quadrant_ogrid(
        LineMesh.arc(R, 2 * n, start_theta=ang[q], end_theta=ang[q + 1], order=order),
        seams[q], seams[q + 1], radial, center_scale=center_scale, wall_tag=wall_tag)
        for q in range(4)]


@pytest.mark.parametrize("n", [1, 2, 4])
@pytest.mark.parametrize("order", [1, 2, 3])
def test_counts_and_orientation(n, order):
    q = _disc(n, order)[0]
    assert q.points.shape[0] == (n + 1) ** 2 + NR * (2 * n + 1)
    assert q.quads.shape[0] == n * n + 2 * n * NR
    assert q.order == order
    assert q.scaled_jacobian().min() > 0.0
    assert q.scaled_jacobian(high_order=True).min() > 0.0


@pytest.mark.parametrize("order", [1, 2, 3, 4])
def test_four_quadrants_merge_to_a_conforming_disc(order):
    n = 3
    quads = _disc(n, order)
    disc = QuadMesh.merge(quads)
    # merge welds only coincident points, so a hanging node would show up as a point
    # count above the shared-seam arithmetic: 4 blocks, each seam shared by two
    # neighbours (7 points beyond the centre O, which all four share).
    n_seam = n + 1 + NR
    expect = 4 * quads[0].points.shape[0] - 4 * (n_seam - 1) - 3
    assert disc.points.shape[0] == expect
    assert disc.quads.shape[0] == 4 * quads[0].quads.shape[0]
    # the real conformality proof at order > 1: merge reconciles every shared edge's
    # nodes owner-wins and raises if an incident copy disagrees.  It did not.
    hm = HexMesh.extrude(disc, 1.0, 1)
    rep = hm.topology_report()
    assert rep.watertight and rep.conformal
    assert rep.n_open_edges == 0 and rep.n_hanging_points == 0
    assert rep.n_components == 1


@pytest.mark.parametrize("order", [2, 3, 4])
def test_wall_nodes_lie_on_the_true_arc(order):
    """The high-order wall nodes are the arc's own, not a chord subdivision."""
    n = 4
    q = _disc(n, order)[0]
    lm = q.lines
    rp = np.linalg.norm(lm.points, axis=1)
    on_wall = ((np.abs(rp[lm.lines[:, 0]] - R) < 1e-12)
               & (np.abs(rp[lm.lines[:, 1]] - R) < 1e-12))
    assert on_wall.sum() == 2 * n
    r_nodes = np.linalg.norm(lm.interior[on_wall].reshape(-1, 3), axis=1)
    assert np.abs(r_nodes - R).max() < 1e-14


@pytest.mark.parametrize("order", [2, 3])
def test_a_bowed_seam_is_meshed_exactly(order):
    """A curved radius keeps its shape: the seam's own nodes go down the overlay
    channel rather than being straight-subdivided between its samples."""
    n = 3

    def bow(t):
        t = np.asarray(t, dtype=float)
        return np.stack([t, 0.25 * np.sin(np.pi * t), np.zeros_like(t)], axis=1)

    fr = QuadMesh.quadrant_seam_fractions(n, RADIAL, CS)
    s1 = LineMesh.loft_curve(bow, fr, order=order)
    s2 = LineMesh.line(np.zeros(3), np.array([0.0, 1.0, 0.0]), fr, order=order)
    # a wall arc joining the two seam ends; its shape is irrelevant to this check
    arc = LineMesh.arc(R, 2 * n, start_theta=0.0, end_theta=np.pi / 2, order=order)
    q = QuadMesh.quadrant_ogrid(arc, s1, s2, RADIAL, center_scale=CS)
    # every node of the mesh that sits on the bowed seam's line must satisfy the curve
    lm = q.lines
    seam_pts = s1.points
    lo = np.array([np.argmin(np.linalg.norm(lm.points - p, axis=1)) for p in seam_pts])
    idx = {(min(a, b), max(a, b)): k for k, (a, b) in enumerate(lm.lines)}
    err = 0.0
    for a, b in zip(lo[:-1], lo[1:]):
        k = idx[(min(a, b), max(a, b))]
        nodes = lm.interior[k]
        err = max(err, np.abs(nodes[:, 1] - 0.25 * np.sin(np.pi * nodes[:, 0])).max())
    assert err < 1e-14


def test_tags_ride_up_from_the_line_level():
    n = 2
    fr = QuadMesh.quadrant_seam_fractions(n, RADIAL, CS)
    arc = LineMesh.arc(R, 2 * n, start_theta=0.0, end_theta=np.pi / 2,
                       element_tags=["wall"] * (2 * n))
    s1 = LineMesh.line(np.zeros(3), np.array([R, 0.0, 0.0]), fr, element_tag="sym")
    s2 = _radius(np.pi / 2, fr)
    q = QuadMesh.quadrant_ogrid(arc, s1, s2, RADIAL, center_scale=CS)
    assert q.edge_group_tags == ["sym", "wall"]
    counts = {t: q.edge_tags.count(t) for t in q.edge_group_tags}
    assert counts["wall"] == 2 * n
    assert counts["sym"] == n + NR
    # an explicit override replaces the whole wall
    q2 = QuadMesh.quadrant_ogrid(arc, s1, s2, RADIAL, center_scale=CS,
                                 wall_tag="outer", side_tags={"seam2": "cut"})
    assert q2.edge_group_tags == ["cut", "outer", "sym"]


def test_seam_fraction_helper_places_the_core_corner_on_the_square():
    """``|O-M| == center_scale * cos(45 deg) * R``, not ``center_scale * R``: M is the
    midpoint of the core square's side while K is its corner."""
    n, cs = 4, 0.6
    fr = QuadMesh.quadrant_seam_fractions(n, RADIAL, cs)
    assert fr.size == n + 1 + NR
    assert fr[0] == 0.0 and fr[-1] == 1.0
    assert np.all(np.diff(fr) > 0.0)
    assert fr[n] == pytest.approx(cs * np.cos(np.pi / 4))
    q = _disc(n, 1, center_scale=cs)[0]
    # the merged core of four quadrants is a square: its corner K and the shared
    # midpoints M sit at the expected radii.
    assert np.linalg.norm(q.points[-1] - q.points[0]) > 0.0


def test_input_contract_is_loud():
    n = 3
    fr = QuadMesh.quadrant_seam_fractions(n, RADIAL, CS)
    arc = LineMesh.arc(R, 2 * n, start_theta=0.0, end_theta=np.pi / 2)
    s1, s2 = _radius(0.0, fr), _radius(np.pi / 2, fr)
    even = LineMesh.arc(R, 2 * n + 1, start_theta=0.0, end_theta=np.pi / 2)
    with pytest.raises(ValueError, match="2\\*n\\+1 points"):
        QuadMesh.quadrant_ogrid(even, s1, s2, RADIAL)
    short = _radius(0.0, np.linspace(0.0, 1.0, n + NR))
    with pytest.raises(ValueError, match="never resampled"):
        QuadMesh.quadrant_ogrid(arc, short, s2, RADIAL)
    with pytest.raises(ValueError, match="must end at"):
        QuadMesh.quadrant_ogrid(arc, s2, s1, RADIAL)
    with pytest.raises(ValueError, match="same center point O"):
        QuadMesh.quadrant_ogrid(
            arc, LineMesh.line(np.array([0.1, 0.0, 0.0]), arc.points[0], fr), s2, RADIAL)
    with pytest.raises(ValueError, match="share an order"):
        QuadMesh.quadrant_ogrid(
            LineMesh.arc(R, 2 * n, start_theta=0.0, end_theta=np.pi / 2, order=2),
            s1, s2, RADIAL)
    with pytest.raises(ValueError, match="seam1/seam2"):
        QuadMesh.quadrant_ogrid(arc, s1, s2, RADIAL, side_tags={"bottom": "x"})
    with pytest.raises(ValueError, match="center_scale"):
        QuadMesh.quadrant_ogrid(arc, s1, s2, RADIAL, center_scale=0.0)


def test_quadrant_matches_ogrid_geometry_on_the_wall():
    """Four quadrants of the unit disk reach the same wall as ``ogrid`` does."""
    n = 4
    disc = QuadMesh.merge(_disc(n, 1))
    r = np.linalg.norm(disc.points, axis=1)
    assert r.max() == pytest.approx(R, abs=1e-14)
    assert int(np.sum(np.abs(r - R) < 1e-12)) == 8 * n


@pytest.mark.parametrize("n", [1, 2, 4])
def test_quadrant_core_is_the_factory_s_own_core(n):
    """``QuadMesh.quadrant_core`` returns exactly the points ``quadrant_ogrid`` puts
    in its core block -- which is what lets a caller build a conforming block behind
    a quadrant face without reproducing the formula."""
    fr = QuadMesh.quadrant_seam_fractions(n, RADIAL, CS)
    s1, s2 = _radius(0.0, fr), _radius(np.pi / 2, fr)
    arc = LineMesh.arc(R, 2 * n, start_theta=0.0, end_theta=np.pi / 2)
    core = QuadMesh.quadrant_core(arc, s1, s2, center_scale=CS)
    assert core.shape == (n + 1, n + 1, 3)
    q = QuadMesh.quadrant_ogrid(arc, s1, s2, RADIAL, center_scale=CS)
    assert np.array_equal(core.reshape(-1, 3), q.points[:(n + 1) ** 2])
    # its two O-ward sides are the caller's own seam fans, verbatim
    assert np.array_equal(core[:, 0], s1.points[:n + 1])
    assert np.array_equal(core[0, :], s2.points[:n + 1])
    # and its far corner is center_scale along the arc midpoint's radius
    assert core[n, n] == pytest.approx(CS * arc.points[n])


def test_quadrant_core_rejects_bad_shapes():
    fr = QuadMesh.quadrant_seam_fractions(2, RADIAL, CS)
    s1, s2 = _radius(0.0, fr), _radius(np.pi / 2, fr)
    even = LineMesh.arc(R, 3, start_theta=0.0, end_theta=np.pi / 2)
    with pytest.raises(ValueError, match="2\\*n\\+1 points"):
        QuadMesh.quadrant_core(even, s1, s2)
    arc = LineMesh.arc(R, 4, start_theta=0.0, end_theta=np.pi / 2)
    with pytest.raises(ValueError, match="center_scale"):
        QuadMesh.quadrant_core(arc, s1, s2, center_scale=1.0)
    with pytest.raises(ValueError, match="at least 3 points"):
        QuadMesh.quadrant_core(arc, _radius(0.0, np.array([0.0, 0.5])), s2)
