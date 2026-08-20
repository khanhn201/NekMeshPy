"""
Carotid vessel mesher: load a triangulated surface, solve three Laplacian
seam fields, cut into legs A/B/C, build conformal seam rings + a central spine,
refit each station's scanned wall ring as a truncated Fourier series so the
high-order nodes sit on a genuine curve, extrude each leg's O-grid sections into
hexes, weld, smooth, export::

    PYTHONPATH=. python examples/carotid.py

Writes ``carotid.re2`` (Nek5000/NekRS) and ``carotid.vtu``.
"""

import logging
import os

import numpy as np

from nekmeshpy import (
    ElementTags,
    TriMesh,
    hexmesh,
    linemesh,
    quadmesh,
    smoothing,
    trimesh,
    viz,
    writer,
)
from nekmeshpy.hexmesh import Seam

logging.basicConfig(level=logging.INFO, format="%(message)s")

# -- parameters (defaults reproduce the bundled ``car`` case) ----------------
_DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
VTX = os.path.join(_DATA, "car.vtx")     # vertices  (N x 3)
TRI = os.path.join(_DATA, "car.tri")     # triangles (M x 3, 1-based)

N_HALF = 8                    # half-ring resolution; MULTIPLE OF 4
N_SLICES = 30                 # cross-sections per leg (hex layers = n_slices)
MIN_LOOP_PTS = 6              # ignore isocontour loops smaller than this
CENTER_SCALE = 0.5            # inner square-core size (fraction of diameter)
QUADRANT_SCALE = 0.45         # half_ogrid's own hub is built from center_scale; this
                             # is quadrant_seam_fractions' independent scale for the
                             # apex seam spined_ogrid synthesizes between the two
                             # halves -- 0.45 is the min-scaled-Jacobian optimum measured
                             # against 0.5's center_scale (search stepped by 0.05)
RADIAL = np.array([0.0, 0.4, 0.7, 0.9, 1.0])   # O-ring layer positions (first 0, last 1.0)
PROJECT_TO_STL = True
# polynomial order.  The wall is genuinely curved at ORDER > 1: each interior
# station's scanned ring is refit as a truncated Fourier series
# (``trimesh.ops.fourier_ring``) and meshed with ``LineMesh.loft_fn``, so the wall's
# high-order nodes sit on that analytic loop rather than on chords between the scanned
# samples.  The spine stays linear (``LineMesh.loft`` straight-subdivides it), which is
# all a flat half-disc seam needs.
ORDER = 3
FOURIER_KEEP = 0.5            # fraction of the rFFT modes kept in the wall refit
SMOOTH_ITERS = 0             # post-assembly untangle/polish sweeps (0 = off)
SMOOTH_LAMBDA = 0.5
FLUX_OFFSET = 2              # hex layers in from the outlet cap (0 = off)
OUT_NAME = "carotid"
EXPORT_RE2 = True
EXPORT_VTK = True
EXPORT_FLD = True            # Nek field file: the GLL nodes .re2 cannot carry
PLOT = False


# -- seam / opening solvers --------------------------------------------------
# Everything past identifying the three openings -- conduction fields, the cut into
# legs, the conformal seam rings, the Fourier wall refit -- is generic trifurcation
# machinery shared with ``examples/femoral.py`` and lives in ``trimesh.ops``.
def order_openings(surf):
    """Order the three boundary loops: A = trunk (lowest mean Z), B/C by mean X.
    Returns 3 vertex-index arrays."""
    loops = surf.boundary_loops()
    assert len(loops) == 3, "expected exactly 3 boundary loops, got %d" % len(loops)
    Z = surf.points[:, 2]
    X = surf.points[:, 0]
    meanZ = np.array([Z[c].mean() for c in loops])
    iA = int(np.argmin(meanZ))
    rest = [i for i in range(3) if i != iA]
    meanX = np.array([X[loops[i]].mean() for i in rest])
    order = np.argsort(meanX, kind="stable")
    return [loops[iA], loops[rest[order[0]]], loops[rest[order[1]]]]


# -- O-grid leg builder ------------------------------------------------------
def ogrid_leg(fine_rings, seam_ring, spine, surface, frlev, *,
              radial, center_scale, quadrant_scale, project_to_stl):
    """Turn a stack of fine interior rings (opening -> seam) into ``nr`` full-disk
    :class:`QuadMesh` slices: each station's two half-O-grids are repositioned
    then merged along the shared spine."""
    seam_pts = seam_ring.points
    spine_pts = spine.points
    frlev = np.asarray(frlev, dtype=float)
    M = seam_pts.shape[0]
    nh = M // 2

    # conformalize the scanned interior rings onto the seam ring's topology (the
    # one intrinsic interpolation that can't be pushed to the caller)
    rings = trimesh.ops.conform_ring_stack(fine_rings, seam_pts, frlev, nh)
    ni = len(rings)
    nr = ni + 1

    RS = np.zeros((nr, M, 3))
    for k in range(ni):
        RS[k, :, :] = rings[k]
    RS[nr - 1, :, :] = seam_pts

    if project_to_stl:
        sub = RS[0:nr - 1, :, :].reshape((nr - 1) * M, 3)
        sub = trimesh.ops.project_points(surface, sub)
        RS[0:nr - 1, :, :] = sub.reshape(nr - 1, M, 3)
    RS[nr - 1, :, :] = seam_pts

    A1 = spine_pts[0, :]
    A2 = spine_pts[-1, :]
    dev = spine_pts - (A1 + (np.arange(nh + 1)[:, None] / nh) * (A2 - A1))
    ringlev = np.arange(nr) / (nr - 1)

    slices = []
    for k in range(nr):
        R = RS[k, :, :]
        # The seam ring (k = nr-1) is *shared* with the two other legs, so it cannot be
        # refit here -- a per-leg refit would mix the two seam arcs this leg happens to
        # see and the blocks would no longer weld.  It arrives already curved instead:
        # ``seam_rings`` fits each of the three shared arcs once, globally
        # (``trimesh.ops.arc_curve``), so use that ring verbatim.  Re-lofting its
        # *points* would straight-subdivide it and throw the curve away at ORDER > 1
        # -- which is exactly what used to leave this one station with 63 degrees of
        # corner at its element joints while every other station sat within 0.2 degrees.
        # Every interior station is private, so refit it into an analytic curve.
        if k == nr - 1:
            wall = seam_ring
        else:
            wall = trimesh.ops.fourier_wall(R, ORDER, "wall", keep=FOURIER_KEEP)
        wpts = wall.points
        e1 = wpts[0, :]
        e2 = wpts[nh, :]
        # this station's spine is the (nh+1)-point deviating diameter A1..A2;
        # spined_ogrid splits the wall loop along it into two half-O-grids and merges.
        spn = (e1 + (np.arange(nh + 1)[:, None] / nh) * (e2 - e1)) + ringlev[k] * dev
        # spined_ogrid meshes the spine exactly at the points given, so resample this
        # scanned diameter to the sampling it consumes here rather than hand it a
        # curve at the wrong count (there is no analytic form to evaluate: the
        # deviation comes off the STL, so the chord is the honest interpolant).
        spn = trimesh.ops.resample_polyline(
            spn, quadmesh.spine_fractions(nh // 4, radial, quadrant_scale))
        # wall tagged on the loop (see flow_past_cylinder.py) so spined_ogrid rides
        # it onto the wall edges.
        slice_ = quadmesh.spined_ogrid(
            wall, radial, spine=linemesh.loft(spn, order=ORDER),
            center_scale=center_scale, quadrant_scale=quadrant_scale)
        slices.append(slice_)
    return slices


def flux_name_for(outlet_name):
    """Flux-plane boundary name for a leg's outlet name, or "" if it has none."""
    if FLUX_OFFSET <= 0:
        return ""
    if outlet_name == "top_outlet_2":
        return "flux_1"
    if outlet_name == "top_outlet_1":
        return "flux_2"
    return ""


# -- pipeline (flat driver) --------------------------------------------------
surf = TriMesh.from_files(VTX, TRI)

gloops = order_openings(surf)
U = trimesh.ops.seam_fields(surf, gloops)

V, faces = trimesh.ops.cut_surface_from_fields(surf, U)
rings, A1, A2, spine = trimesh.ops.seam_rings(
    V, faces, gloops, N_HALF, order=ORDER, keep=FOURIER_KEEP, element_tag="wall")

outlet_name = ["trunk_outlet", "top_outlet_1", "top_outlet_2"]
levels = np.linspace(0, 1, N_SLICES + 2)[1:-1]

#: The two regions a flux plane separates. They exist only to give the plane a
#: direction: a measurement surface is one shared face with one name, and which of its
#: two sides the exported row is written from is read from the region of the element
#: that owns it.
FLUX_UPSTREAM, FLUX_DOWNSTREAM = "flux_upstream", "flux_downstream"

#: name -> Nek BC code, applied only at export (byte-exact reference).
#:
#: The flux planes are **interior**, so each of their faces has a hex on either side
#: and reconstructs to a row for each. Flux through a surface has a direction, so only
#: the upstream side is written -- the downstream row would be the same measurement
#: counted backwards. Anything else is one code from every side.
GROUPS = {
    "wall": "W  ",
    "trunk_outlet": "v  ",
    "top_outlet_1": "int",
    "top_outlet_2": "O  ",
    "flux_1": {FLUX_UPSTREAM: "f1 ", FLUX_DOWNSTREAM: None},
    "flux_2": {FLUX_UPSTREAM: "f2 ", FLUX_DOWNSTREAM: None},
}

def cap_tags(slice_, first, second):
    """Name a leg's seam cap by half-disc.  ``spined_ogrid`` welds the two halves in
    order, so the first half of the quads is the first arc's side; each half is shared
    with a *different* leg, which is why one name for the whole cap will not do.
    ``last_tag`` takes an ``ElementTags`` over the slice's own elements for this."""
    half = slice_.n_quads // 2
    return ElementTags.from_dense(
        np.array([first] * half + [second] * (slice_.n_quads - half)))


#: Which shared arc each leg's two seam half-discs sit on.  ``trimesh.ops.seam_rings``
#: builds the three rings as ``(ab, ac)``, ``(ab, bc)``, ``(ac, bc)``, so leg 0 meets
#: leg 1 on ``ab`` and leg 2 on ``ac``, and so on round the trifurcation.
LEG_CAP = {0: ("attach1", "attach2"), 1: ("attach1", "attach3"),
           2: ("attach2", "attach3")}

blocks: list = []
seam_block: dict = {}
flux_seams: list = []
for leg in range(3):
    sub, us, _, _, _ = trimesh.ops.leg_field(V, faces, leg, gloops)
    fr, frlev = trimesh.ops.extract_rings(sub, us, levels, MIN_LOOP_PTS)
    slices = ogrid_leg(fr, rings[leg], spine, surf, frlev,
                       radial=RADIAL, center_scale=CENTER_SCALE,
                       quadrant_scale=QUADRANT_SCALE,
                       project_to_stl=PROJECT_TO_STL)
    flux_name = flux_name_for(outlet_name[leg])
    off = FLUX_OFFSET
    cap = cap_tags(slices[-1], *LEG_CAP[leg])
    # opening cap = leg outlet; seam end is interior.  With a flux plane, split
    # the leg there (a cap of the downstream segment); attach re-joins them.
    if flux_name and 0 < off < len(slices) - 1:
        # The flux plane is the seam between the two, so it is an *interior* face with
        # a hex on each side. Flux has a direction, so the two segments are named as
        # regions and the plane is written from the upstream one only -- the other row
        # would be the same measurement counted backwards. Slices run outlet -> seam,
        # so ``slices[off:]`` is the upstream side.  Both sides carry the plane's name
        # so ``attach`` can be told which group meets which; ``attach_tag`` puts it back
        # on the fused face, which is what makes it the tagged interior plane.
        blocks.append(hexmesh.loft(slices[:off + 1], first_tag=outlet_name[leg],
                                   last_tag=flux_name, element_tags=FLUX_DOWNSTREAM))
        blocks.append(hexmesh.loft(slices[off:], first_tag=flux_name, last_tag=cap,
                                   element_tags=FLUX_UPSTREAM))
        flux_seams.append(Seam(len(blocks) - 2, flux_name, len(blocks) - 1, flux_name,
                               attach_tag=flux_name))
    else:
        blocks.append(hexmesh.loft(slices, first_tag=outlet_name[leg], last_tag=cap))
    seam_block[leg] = len(blocks) - 1

# every seam is named on both sides now -- the two flux planes within their legs, and
# the three half-discs where the legs meet about the shared spine
mesh = hexmesh.attach(blocks, flux_seams + [
    Seam(seam_block[0], "attach1", seam_block[1], "attach1"),
    Seam(seam_block[0], "attach2", seam_block[2], "attach2"),
    Seam(seam_block[1], "attach3", seam_block[2], "attach3"),
])

if SMOOTH_ITERS > 0:
    # takes the result: the smoother builds a new mesh rather than writing through
    # this one's live points
    mesh = smoothing.smooth(mesh, surf, smooth_iters=SMOOTH_ITERS, smooth_lambda=SMOOTH_LAMBDA,
                     wall="wall", project_to_stl=PROJECT_TO_STL)

print(hexmesh.report(mesh))
if EXPORT_VTK:
    writer.to_vtu(mesh, OUT_NAME + ".vtu", groups=GROUPS)
if EXPORT_RE2:
    writer.to_re2(mesh, OUT_NAME + ".re2", groups=GROUPS)
if EXPORT_FLD:
    # .re2 is corner-only at any order; the field file carries the full GLL block,
    # so this is the export that actually preserves the ORDER = 3 geometry.
    writer.to_fld(mesh, OUT_NAME + "0.f00001")
if PLOT:
    viz.plot(mesh, ["wall", "trunk_outlet", "top_outlet_1", "top_outlet_2"], OUT_NAME)
print("carotid: %d hex elements, %d points" % (mesh.n_hexes, mesh.n_points))
