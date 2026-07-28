"""Circular-pipe T-junction as a conformal all-hex block.

Built from **analytic geometry** (not a scanned surface like ``bifurcation.py``):
a main pipe (axis X, radius ``R``) and a branch (axis +Z, radius ``R``) tee off at
the origin. Same three-leg construction as the bifurcation:

* the two saddle points are ``A1 = (0, +R, 0)`` and ``A2 = (0, -R, 0)``; the
  shared **spine** is the ``A1 - A2`` segment through the origin;
* three arcs run ``A1 -> A2`` -- ``aLM`` (main lower wall, plane ``x = 0``) and
  ``aLB`` / ``aRB`` (the two halves of the intersection collar). Each leg's seam
  ring is the pair of arcs it shares with its neighbours, so reusing the arcs
  welds adjacent legs conformally;
* each leg blends a circular opening to its seam ring; each station is split along
  the spine into two :meth:`QuadMesh.half_ogrid` half-discs and merged, the stack
  is :meth:`HexMesh.loft`-ed, and the three blocks :meth:`HexMesh.merge`-d.

    PYTHONPATH=. python examples/circular_pipe_tjunction.py

Produces ``circular_pipe_tjunction.re2`` / ``.rea`` and ``.vtk``.
"""

import logging

import numpy as np

from nekmeshpy import (
    HexMesh,
    LineMesh,
    QuadMesh,
    TriMesh,
    export,
    smoothing,
    trimesh,
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
SMOOTHING_METHOD = "bilinear"    # per-section interior repositioning
SMOOTH_ITERS = 8              # post-assembly untangle/polish sweeps (0 = off)
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
    arc-length-even samples are exactly the angle-even points (built analytically)."""
    a = np.linspace(0.0, -np.pi, N_HALF + 1)
    return np.column_stack([np.zeros_like(a), R * np.cos(a), R * np.sin(a)])


def arc_collar(xside):
    """One half of the intersection collar (seam between a main leg and the
    branch): ``y^2+z^2 = x^2+y^2 = R^2``, ``z >= 0`` gives
    ``(xside*sqrt(R^2-y^2), y, sqrt(R^2-y^2))`` for ``y`` from ``+R -> -R``.  This
    Viviani curve has no closed-form arc length, so sample it densely and resample
    by arc length to the exact ``N_HALF+1`` seam points."""
    y = np.linspace(R, -R, 400)
    r = np.sqrt(np.maximum(R * R - y * y, 0.0))
    p = np.column_stack([xside * r, y, r])
    return trimesh.ops.resample_polyline(p, np.linspace(0.0, 1.0, N_HALF + 1))


def join_arcs(p, q):
    """Two ``A1 -> A2`` arcs into a closed ring of ``M`` points: index 0 at
    ``A1``, index ``N_HALF`` at ``A2``."""
    return np.vstack([p[0:N_HALF], q[::-1][0:N_HALF]])


# -- analytic circular openings (M points, matching the seam's point order) ---
def opening_main(x0):
    """Circle of radius ``R`` in plane ``x = x0``, traversed +y (index 0) -> -z ->
    -y (index ``N_HALF``) -> +z, so its lower half matches ``arc_main_lower``
    index-for-index and its upper half the collar arc."""
    a = -np.pi * np.arange(M) / N_HALF
    return np.column_stack([np.full(M, x0), R * np.cos(a), R * np.sin(a)])


def opening_branch(z0):
    """Circle of radius ``R`` in plane ``z = z0``, traversed +y (index 0) -> -x ->
    -y -> +x, so its ``x <= 0`` half matches ``aLB`` and its ``x >= 0`` half
    ``aRB``."""
    g = np.pi * np.arange(M) / N_HALF
    return np.column_stack([-R * np.sin(g), R * np.cos(g), np.full(M, z0)])


# -- leg builder: blend opening -> seam, split each station along the spine ---
def leg_slices(open_ring, seam_ring, n_slices):
    """Blend ``open_ring -> seam_ring`` over ``n_slices`` stations; each station is
    two spine-split :meth:`QuadMesh.half_ogrid` half-discs merged into a disc. End
    stations keep the raw algebraic fill; interior ones use ``SMOOTHING_METHOD``."""
    slices = []
    for s in range(n_slices):
        w = s / (n_slices - 1)                     # 0 at opening, 1 at seam
        ring = (1.0 - w) * open_ring + w * seam_ring
        e1, e2 = ring[0, :], ring[N_HALF, :]
        arc1 = ring[0:N_HALF + 1, :]
        arc2 = np.vstack([ring[N_HALF:M, :], ring[0:1, :]])
        arc1_lm = LineMesh.open(arc1, element_tags=["wall"] * N_HALF)
        arc2_lm = LineMesh.open(arc2, element_tags=["wall"] * N_HALF)
        # the spine is the straight A1..A2 diameter; sample it at the exact canonical
        # fractions half_ogrid indexes (h2's spine runs the opposite way, e2..e1)
        fr = QuadMesh.half_ogrid_spine_fractions(arc1_lm, CENTER_SCALE, RADIAL)
        spn1 = e1 + fr[:, None] * (e2 - e1)
        spn2 = e2 + fr[:, None] * (e1 - e2)
        m = SMOOTHING_METHOD if 0 < s < n_slices - 1 else None
        # tag the arc "wall" at the line level so half_ogrid rides it onto the
        # wall edges (see flow_past_cylinder.py)
        h1 = QuadMesh.half_ogrid(
            arc1_lm, LineMesh.open(spn1),
            RADIAL, center_scale=CENTER_SCALE, smoothing_method=m)
        h2 = QuadMesh.half_ogrid(
            arc2_lm, LineMesh.open(spn2),
            RADIAL, center_scale=CENTER_SCALE, smoothing_method=m)
        slices.append(QuadMesh.merge([h1, h2]))
    return slices


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
    HexMesh.loft(leg_slices(opening_main(-L), seam_left, N_SLICES_MAIN),
                 first_tag="inlet"),
    HexMesh.loft(leg_slices(opening_main(+L), seam_right, N_SLICES_MAIN),
                 first_tag="outlet"),
    HexMesh.loft(leg_slices(opening_branch(H), seam_branch, N_SLICES_BRANCH),
                 first_tag="branch"),
]

mesh = HexMesh.merge(blocks)

if SMOOTH_ITERS > 0:
    smoothing.smooth(mesh, wall_surface(), smooth_iters=SMOOTH_ITERS,
                     smooth_lambda=SMOOTH_LAMBDA, wall="wall",
                     project_to_stl=True)

# -- report + export ---------------------------------------------------------
print(mesh.report())

export.to_re2(mesh, OUT_NAME, groups=GROUPS)
export.to_vtk(mesh, OUT_NAME + ".vtk", groups=GROUPS)
