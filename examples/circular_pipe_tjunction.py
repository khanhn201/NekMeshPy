"""Mesh a circular-pipe T-junction as a conformal all-hex block and export it.

A flat, gmsh-style script (edit the constants, re-run).  Unlike ``bifurcation.py``
-- which reads a triangulated vessel and *finds* its seams -- this junction is
built from **analytic geometry**: a straight main pipe (axis X, radius ``R``) and
a branch pipe (axis +Z, radius ``R``) tee off at the origin.  The whole junction
is the same three-leg construction as the bifurcation, so the pieces come from the
same toolkit primitives:

* the two saddle points where the three tube walls meet are ``A1 = (0, +R, 0)`` and
  ``A2 = (0, -R, 0)``; the shared **spine** is the straight ``A1 - A2`` segment
  through the origin;
* three arcs run ``A1 -> A2`` over the junction surface -- ``aLM`` (the main pipe's
  lower wall, in the plane ``x = 0``), ``aLB`` / ``aRB`` (the two halves of the
  cylinder-cylinder intersection collar, ``x <= 0`` and ``x >= 0``).  Each leg's
  **seam ring** is the pair of arcs it shares with its two neighbours, so adjacent
  legs are welded conformally when the arc arrays are reused;
* each leg blends from a clean circular opening to its seam ring; every station is
  split along the spine into two :meth:`QuadMesh.half_ogrid` half-discs and merged,
  then the stack is :meth:`HexMesh.loft`-ed into a block; the three blocks are
  :meth:`HexMesh.merge`-d (welding the coincident seam quads) and polished.

Run with::

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
    """The main pipe's lower wall in the plane ``x = 0`` (the seam shared by the
    two main legs): the ``z <= 0`` semicircle of ``y^2 + z^2 = R^2``."""
    a = np.linspace(0.0, -np.pi, 400)
    p = np.column_stack([np.zeros_like(a), R * np.cos(a), R * np.sin(a)])
    return LineMesh.open(p).resample(np.linspace(0.0, 1.0, N_HALF + 1)).points


def arc_collar(xside):
    """One half of the cylinder-cylinder intersection collar (the seam between a
    main leg and the branch): ``y^2+z^2 = x^2+y^2 = R^2`` with ``z >= 0`` gives
    ``(xside*sqrt(R^2-y^2), y, sqrt(R^2-y^2))`` for ``y`` running ``+R -> -R``."""
    y = np.linspace(R, -R, 400)
    r = np.sqrt(np.maximum(R * R - y * y, 0.0))
    p = np.column_stack([xside * r, y, r])
    return LineMesh.open(p).resample(np.linspace(0.0, 1.0, N_HALF + 1)).points


def join_arcs(p, q):
    """Two arcs (each ``A1 -> A2``) into a closed ring of ``M`` points, index 0 at
    ``A1`` and index ``N_HALF`` at ``A2`` (ported from the bifurcation join)."""
    return np.vstack([p[0:N_HALF], q[::-1][0:N_HALF]])


# -- analytic circular openings (M points, matching the seam's point order) ---
def opening_main(x0):
    """A circle of radius ``R`` in the plane ``x = x0``, traversed +y (index 0) ->
    down through -z -> -y (index ``N_HALF``) -> up through +z, so its lower half
    corresponds index-for-index to ``arc_main_lower`` and its upper half to the
    collar arc."""
    a = -np.pi * np.arange(M) / N_HALF
    return np.column_stack([np.full(M, x0), R * np.cos(a), R * np.sin(a)])


def opening_branch(z0):
    """A circle of radius ``R`` in the plane ``z = z0``, traversed +y (index 0) ->
    through -x -> -y -> through +x, so its ``x <= 0`` half matches ``aLB`` and its
    ``x >= 0`` half matches ``aRB``."""
    g = np.pi * np.arange(M) / N_HALF
    return np.column_stack([-R * np.sin(g), R * np.cos(g), np.full(M, z0)])


# -- leg builder: blend opening -> seam, split each station along the spine ---
def leg_slices(open_ring, seam_ring, n_slices):
    """Blend ``open_ring -> seam_ring`` over ``n_slices`` stations; build each
    station as two spine-split :meth:`QuadMesh.half_ogrid` half-discs merged into a
    full disc.  The end stations (opening cap, welded seam) keep the raw algebraic
    fill; interior stations are repositioned by ``SMOOTHING_METHOD``."""
    idx = np.arange(N_HALF + 1)[:, None] / N_HALF
    slices = []
    for s in range(n_slices):
        w = s / (n_slices - 1)                     # 0 at opening, 1 at seam
        ring = (1.0 - w) * open_ring + w * seam_ring
        e1, e2 = ring[0, :], ring[N_HALF, :]
        spn = e1 + idx * (e2 - e1)                 # straight A1..A2 diameter
        arc1 = ring[0:N_HALF + 1, :]
        arc2 = np.vstack([ring[N_HALF:M, :], ring[0:1, :]])
        m = SMOOTHING_METHOD if 0 < s < n_slices - 1 else None
        # the arc IS the wall: tag it at the line level (one tag per arc segment),
        # so half_ogrid rides it onto the wall edges (see flow_past_cylinder.py).
        h1 = QuadMesh.half_ogrid(
            LineMesh.open(arc1, element_tags=["wall"] * N_HALF), LineMesh.open(spn),
            RADIAL, center_scale=CENTER_SCALE, smoothing_method=m)
        h2 = QuadMesh.half_ogrid(
            LineMesh.open(arc2, element_tags=["wall"] * N_HALF), LineMesh.open(spn[::-1, :]),
            RADIAL, center_scale=CENTER_SCALE, smoothing_method=m)
        slices.append(QuadMesh.merge([h1, h2]))
    return slices


# -- analytic wall surface (smoothing projection target) ---------------------
def wall_surface():
    """Triangulated walls of the three tubes, trimmed at the collar, as a single
    :class:`TriMesh` for the smoother to keep wall nodes on.  The main tube spans
    ``x in [-L, L]`` but its ``z > 0`` part is removed where the branch opens
    (``|x| < sqrt(R^2 - y^2)``); the branch tube spans ``z in [0, H]``."""
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
