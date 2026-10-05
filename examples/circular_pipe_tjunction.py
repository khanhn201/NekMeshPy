"""Circular-pipe T-junction as a conformal all-hex block.

Built from **analytic geometry** (not a scanned surface like ``carotid.py``):
a main pipe (axis X, radius ``R``) and a branch (axis +Z, radius ``R``) tee off at
the origin. Same three-leg construction as the carotid:

* the two saddle points are ``A1 = (0, +R, 0)`` and ``A2 = (0, -R, 0)``; the
  shared **spine** is the ``A1 - A2`` segment through the origin;
* three arcs run ``A1 -> A2`` -- ``aLM`` (main lower wall, plane ``x = 0``) and
  ``aLB`` / ``aRB`` (the two halves of the intersection collar). Each leg's seam
  ring is the pair of arcs it shares with its neighbours :meth:`LineMesh.merge`-d
  at ``A1``/``A2`` into a closed loop, so reusing the arcs welds adjacent legs
  conformally;
* each leg blends a circular opening to its seam ring (:meth:`LineMesh.blend`); each
  station is :meth:`QuadMesh.spined_ogrid`-ed over its A1..A2 chord (split into two
  half-discs and merged), the stack is :meth:`HexMesh.loft`-ed, and the three blocks
  :meth:`HexMesh.merge`-d.

    PYTHONPATH=. python examples/circular_pipe_tjunction.py

Produces ``circular_pipe_tjunction.re2`` and ``.vtu``.
"""

import logging

import numpy as np

from nekmeshpy import (
    ElementTags,
    hexmesh,
    linemesh,
    quadmesh,
    writer,
)
from nekmeshpy.hexmesh import Seam
from nekmeshpy.linemesh import Seam as PointSeam

logging.basicConfig(level=logging.INFO, format="%(message)s")

# -- parameters --------------------------------------------------------------
R = 0.5                       # pipe radius (main and branch are equal)
L = 2.0                       # main pipe half-length (opening at x = -L and x = +L)
H = 1.5                       # branch height (opening at z = H)
N_HALF = 8                    # half-ring resolution; MULTIPLE OF 4
N_SLICES_MAIN = 12            # cross-sections per main leg (hex layers = this - 1)
N_SLICES_BRANCH = 10          # cross-sections in the branch
CENTER_SCALE = 0.5            # inner square-core size (fraction of the diameter)
RADIAL = np.array([0.0, 0.4, 0.8, 1.0])   # O-ring layer positions (first 0, last 1.0)
ORDER = 4                     # polynomial order; 1 = linear.  The seam arcs and
                              # openings are meshed at ORDER, so .vtu renders the
                              # curved walls (.re2 stays linear either way)
OUT_NAME = "circular_pipe_tjunction"

# boundary name -> Nek BC code, applied only at export
GROUPS = {"wall": "W  ", "inlet": "v  ", "outlet": "O  ", "branch": "O  "}

M = 2 * N_HALF                # points per full cross-section ring


# -- analytic seam arcs (each A1 -> A2, N_HALF+1 arc-length-even points) ------
def arc_main_lower():
    """Main lower wall in plane ``x = 0`` (seam of the two main legs): the
    ``z <= 0`` semicircle of ``y^2 + z^2 = R^2``.  Constant-speed in angle, so the
    arc-length-even samples are exactly the angle-even points -- i.e. exactly what
    :meth:`LineMesh.arc` places.  ``normal = -x`` gives the in-plane frame
    ``e1 = +y``, ``e2 = -z``, so sweeping ``theta`` from ``0`` to ``pi`` walks
    ``+y -> -z -> -y``; at ``ORDER > 1`` every GLL node lands on the exact circle
    (an explicit point array would only be straight-subdivided between samples)."""
    return linemesh.arc(R, N_HALF, center=(0.0, 0.0, 0.0), normal=(-1.0, 0.0, 0.0),
                        start_theta=0.0, end_theta=np.pi,
                        first_tag="A1", last_tag="A2", order=ORDER)


def arc_collar(xside):
    """One half of the intersection collar (seam between a main leg and the branch):
    the ``z >= 0`` branch of ``y^2+z^2 = x^2+y^2 = R^2``.  Two equal-radius cylinders
    meet in a pair of **planar ellipses**, so this curve has the closed form
    ``p(t) = (xside*R sin t, R cos t, R sin t)``, ``t: 0 -> pi`` -- it lies in the
    plane ``x = xside*z`` with semi-axes ``R*sqrt(2)`` and ``R``.

    :meth:`LineMesh.loft_fn` evaluates that form at every node, corners *and* the
    interior GLL nodes, so the collar is exact to machine precision at any ``ORDER``.
    Its arc length has no closed form, so the even-by-arc-length grading is an
    explicit first step: :meth:`LineMesh.arclength_fractions` inverts a chord table
    over ``t: 0 -> pi`` into the ``t`` values themselves, and ``curve`` evaluates the
    ellipse at exactly those parameters -- it neither normalizes nor remaps them, so
    the helper's output is handed straight on and the domain is stated once.  Only
    *where along* the ellipse the nodes sit inherits that table's discretization
    error -- every node still lands on the true ellipse to machine precision, because
    it is placed by evaluating the closed form and never by interpolating the table.
    (Sampling this into an array and calling ``LineMesh.loft`` instead would
    straight-subdivide it and put the interior nodes ~2.6% of ``R`` off the curve.)"""
    def f(t):
        return np.column_stack(
            [xside * R * np.sin(t), R * np.cos(t), R * np.sin(t)])

    return linemesh.loft_fn(
        f, linemesh.arclength_fractions(f, N_HALF, t_range=(0.0, np.pi)),
        first_tag="A1", last_tag="A2", order=ORDER)


def join_arcs(p, q):
    """Two shared-endpoint ``A1 -> A2`` arcs into a closed ring of ``M`` points
    (index 0 at ``A1``, index ``N_HALF`` at ``A2``); ``q`` is reversed so the loop
    traverses without crossing.

    Both ends are named, so the ring is closed by two *stated* joins rather than by a
    coordinate weld -- ``reverse`` carries a point's tag with it, so ``A1`` still names
    ``A1`` whichever way round the arc is stored."""
    return linemesh.attach([p, linemesh.reverse(q)],
                           [PointSeam(0, "A1", 1, "A1"), PointSeam(0, "A2", 1, "A2")])


# -- circular openings (M points, matching the seam's point order) ------------
def opening_main(x0):
    """Circle of radius ``R`` in plane ``x = x0``, traversed +y (index 0) -> -z ->
    -y (index ``N_HALF``) -> +z, so its lower half matches ``arc_main_lower``
    index-for-index and its upper half the collar arc. ``normal = -x`` gives the
    in-plane frame ``e1 = +y``, ``e2 = -z``, so point ``k`` lands on
    ``(0, R cos, -R sin)`` -- the required clockwise traversal from ``+y``.
    At ``ORDER > 1`` every GLL node lands on the exact circle."""
    return linemesh.circle(R, M, center=(x0, 0.0, 0.0), normal=(-1.0, 0.0, 0.0),
                           order=ORDER)


def opening_branch(z0):
    """Circle of radius ``R`` in plane ``z = z0``, traversed +y (index 0) -> -x ->
    -y -> +x, so its ``x <= 0`` half matches ``aLB`` and its ``x >= 0`` half
    ``aRB``. ``normal = +z`` gives ``e1 = +x``, ``e2 = +y``; the ``start_theta =
    pi/2`` phase turns ``(cos, sin)`` into ``(-sin, cos)`` so index 0 is ``+y``.
    At ``ORDER > 1`` every GLL node lands on the exact circle."""
    return linemesh.circle(R, M, center=(0.0, 0.0, z0), normal=(0.0, 0.0, 1.0),
                           start_theta=np.pi / 2, order=ORDER)


# -- leg builder: blend opening -> seam, O-grid each station across the spine --
def leg_slices(open_ring, seam_ring, n_slices):
    """Blend ``open_ring -> seam_ring`` over ``n_slices`` stations; each station is a
    disc :meth:`QuadMesh.spined_ogrid`-ed over its natural straight A1..A2 diameter
    (two half-discs welded along it)."""
    loops = linemesh.blend(open_ring, seam_ring, np.linspace(0.0, 1.0, n_slices))
    # spined_ogrid splits the loop along its A1..A2 chord (default spine) and merges
    # the two half-discs; wall_tag names every wall edge (see flow_past_cylinder.py
    # for the tag-flow-down convention).
    return [quadmesh.spined_ogrid(
        loop, RADIAL, center_scale=CENTER_SCALE, quadrant_scale=CENTER_SCALE,
        wall_tag="wall") for loop in loops]


# -- pipeline (flat driver) --------------------------------------------------
a_lm = arc_main_lower()
a_lb = arc_collar(-1.0)
a_rb = arc_collar(+1.0)

seam_left = join_arcs(a_lm, a_lb)     # main-left  : lower main wall + x<=0 collar
seam_right = join_arcs(a_lm, a_rb)    # main-right : lower main wall + x>=0 collar
seam_branch = join_arcs(a_lb, a_rb)   # branch     : the full collar

def cap_tags(slice_, first, second):
    """Name a leg's seam cap by half-disc: ``spined_ogrid`` welds the two halves in
    order, so the first half of the quads is the first arc's side and the second half
    the other's.  Each half is shared with a *different* leg, which is why one name for
    the whole cap will not do -- ``last_tag`` takes an ``ElementTags`` over the slice's
    own elements for exactly this.

    Mis-assigning the halves cannot pass silently: ``attach`` pairs a named group
    against a named group and refuses anything that is not one-to-one."""
    half = slice_.n_quads // 2
    return ElementTags.from_dense(
        np.array([first] * half + [second] * (slice_.n_quads - half)))


sl_left = leg_slices(opening_main(-L), seam_left, N_SLICES_MAIN)
sl_right = leg_slices(opening_main(+L), seam_right, N_SLICES_MAIN)
sl_branch = leg_slices(opening_branch(H), seam_branch, N_SLICES_BRANCH)

# the three legs meet pairwise on the three arcs: a_lm joins left to right, a_lb joins
# left to branch, a_rb joins right to branch.  Each is one half of two legs' seam caps.
blocks = [
    hexmesh.loft(sl_left, first_tag="inlet",
                 last_tag=cap_tags(sl_left[-1], "attach1", "attach2")),
    hexmesh.loft(sl_right, first_tag="outlet",
                 last_tag=cap_tags(sl_right[-1], "attach1", "attach3")),
    hexmesh.loft(sl_branch, first_tag="branch",
                 last_tag=cap_tags(sl_branch[-1], "attach2", "attach3")),
]

mesh = hexmesh.attach(blocks, [Seam(0, "attach1", 1, "attach1"),
                               Seam(0, "attach2", 2, "attach2"),
                               Seam(1, "attach3", 2, "attach3")])

# -- report + export ---------------------------------------------------------
print(hexmesh.report(mesh))

writer.to_re2(mesh, OUT_NAME + ".re2", groups=GROUPS)
writer.to_vtu(mesh, OUT_NAME + ".vtu", groups=GROUPS)
