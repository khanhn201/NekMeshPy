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
    TriMesh,
    hexmesh,
    linemesh,
    quadmesh,
    smoothing,
    writer,
)

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
# post-assembly untangle/polish sweeps (0 = off).  hexmesh.smoothing.smooth is
# order-1 only (it moves corner nodes and would leave the curved nodes behind), so
# it is switched off at ORDER > 1; the call below is kept so a reader can set
# ORDER = 1 + SMOOTH_ITERS = 8 to exercise the STL-constrained wall polish.
SMOOTH_ITERS = 0
SMOOTH_LAMBDA = 0.5
N_SURF = 48                   # tris per ring on the analytic wall (smoothing target)
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
                        start_theta=0.0, end_theta=np.pi, order=ORDER)


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
        order=ORDER)


def join_arcs(p, q):
    """Two shared-endpoint ``A1 -> A2`` arcs into a closed ring of ``M`` points
    (index 0 at ``A1``, index ``N_HALF`` at ``A2``), welded at ``A1``/``A2`` by
    :meth:`LineMesh.merge`; ``q`` is reversed so the loop traverses without
    crossing."""
    return linemesh.merge([p, linemesh.reverse(q)])


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


# -- analytic wall surface (smoothing projection target) ---------------------
def wall_surface():
    """Triangulated walls of the three tubes, trimmed at the collar, as one
    :class:`TriMesh` projection target for the smoother.  The main tube spans
    ``x in [-L, L]`` minus the branch opening (``z > 0``, ``|x| < sqrt(R^2 -
    y^2)``); the branch tube spans ``z in [0, H]``."""
    th = np.linspace(0.0, 2.0 * np.pi, N_SURF, endpoint=False)
    tris, pts = [], []

    def add_quad(p00, p01, p11, p10):
        b = len(pts)
        pts.extend([p00, p01, p11, p10])
        tris.extend([[b, b + 1, b + 2], [b, b + 2, b + 3]])

    # main tube wall (y = R cos, z = R sin), skipping the branch cut-out
    nx = 2 * (N_SLICES_MAIN - 1)
    xs = np.linspace(-L, L, nx + 1)
    for i in range(nx):
        for j in range(N_SURF):
            y0, z0 = R * np.cos(th[j]), R * np.sin(th[j])
            y1, z1 = R * np.cos(th[(j + 1) % N_SURF]), R * np.sin(th[(j + 1) % N_SURF])
            xa, xb = xs[i], xs[i + 1]
            # skip panels inside the collar (branch opening: z>0 and |x|<sqrt(R^2-y^2))
            cut = R * R - max(y0, y1) ** 2
            if z0 > 1e-9 and z1 > 1e-9 and cut > 0 and abs(0.5 * (xa + xb)) < np.sqrt(cut):
                continue
            add_quad([xa, y0, z0], [xb, y0, z0], [xb, y1, z1], [xa, y1, z1])

    # branch tube wall (x = R cos, y = R sin), from the collar up to z = H
    nz = N_SLICES_BRANCH - 1
    for j in range(N_SURF):
        x0, y0 = R * np.cos(th[j]), R * np.sin(th[j])
        x1, y1 = R * np.cos(th[(j + 1) % N_SURF]), R * np.sin(th[(j + 1) % N_SURF])
        zc0 = np.sqrt(max(R * R - y0 * y0, 0.0))   # collar height at this station
        zc1 = np.sqrt(max(R * R - y1 * y1, 0.0))
        for i in range(nz):
            za0 = zc0 + (H - zc0) * i / nz
            zb0 = zc0 + (H - zc0) * (i + 1) / nz
            za1 = zc1 + (H - zc1) * i / nz
            zb1 = zc1 + (H - zc1) * (i + 1) / nz
            add_quad([x0, y0, za0], [x0, y0, zb0], [x1, y1, zb1], [x1, y1, za1])

    return TriMesh(np.array(pts, dtype=float), np.array(tris, dtype=np.int64))


# -- pipeline (flat driver) --------------------------------------------------
a_lm = arc_main_lower()
a_lb = arc_collar(-1.0)
a_rb = arc_collar(+1.0)

seam_left = join_arcs(a_lm, a_lb)     # main-left  : lower main wall + x<=0 collar
seam_right = join_arcs(a_lm, a_rb)    # main-right : lower main wall + x>=0 collar
seam_branch = join_arcs(a_lb, a_rb)   # branch     : the full collar

blocks = [
    hexmesh.loft(leg_slices(opening_main(-L), seam_left, N_SLICES_MAIN),
                 first_tag="inlet"),
    hexmesh.loft(leg_slices(opening_main(+L), seam_right, N_SLICES_MAIN),
                 first_tag="outlet"),
    hexmesh.loft(leg_slices(opening_branch(H), seam_branch, N_SLICES_BRANCH),
                 first_tag="branch"),
]

mesh = hexmesh.merge(blocks)

if SMOOTH_ITERS > 0:
    # takes the result: the smoother builds a new mesh rather than writing through
    # this one's live points
    mesh = smoothing.smooth(mesh, wall_surface(), smooth_iters=SMOOTH_ITERS,
                     smooth_lambda=SMOOTH_LAMBDA, wall="wall",
                     project_to_stl=True)

# -- report + export ---------------------------------------------------------
print(hexmesh.report(mesh))

writer.to_re2(mesh, OUT_NAME + ".re2", groups=GROUPS)
writer.to_vtu(mesh, OUT_NAME + ".vtu", groups=GROUPS)
