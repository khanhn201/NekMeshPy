"""Case study: the carotid HEX mesh rebuilt on the smoothed wall (examples/carotid.py recipe, order 1).

``cs_wall_surface.py`` smooths the carotid wall as a free-standing surface and says plainly that the hexes do not
follow.  This script closes that gap the way the construction guide allows: **not** by dragging interior nodes,
but by *regenerating the hexes through the same recipe* with the wall rings replaced.

**The recipe, and what is regenerated vs reused.**  Everything up to the wall rings is ``examples/carotid.py``'s
own code (copied here, flat, not ``runpy``'d -- the build needs a different ``RADIAL`` and wall rings, and the
pipeline's module-level ``ORDER`` / ``RADIAL`` cannot be overridden from outside): the Laplacian seam fields, the cut
into legs A/B/C, the three shared seam rings + spine, each leg's ring stack conformed onto the seam topology and
projected onto the tessellation.  Reused unchanged: the O-grid section builder (``quadmesh.spined_ogrid``: radial
fractions, ``CENTER_SCALE``, ``QUADRANT_SCALE``, the spine), ``hexmesh.loft`` per leg (split at the flux planes),
the ``Seam`` welds, the tags and the boundary names.  Regenerated: each station's **wall ring** (and the shared seam
rings).  The originals are the Fourier-refit rings; the new ones are the smoothed wall's own vertices, read back
per (leg, station, ring index) -- the same ring topology and ordering, so the section builder cannot tell the
difference except that the ring is no longer planar or equally spaced.  A shared seam ring moves identically in
the three legs because it is one set of wall vertices (the smoother treats the welded wall as one surface), so
``attach`` still welds with no tolerance.  Every interior layer then follows from the wall ring alone, which is
what makes the boundary layer a property of the **construction** (``RADIAL`` measured from the wall inward), not
something the surface smoother could undo.

**Layers.**  The rebuild keeps the original layering: ``RADIAL = [0, .4, .7, .9, 1]`` has four layers between the
core square and the wall, the wall layer being ``0.1`` of that gap.  ``RADIAL`` is a constant at the top; assign
another array to change it.

**Smoother and the trifurcation.**  The smoother may move the shared seam rings freely (``FREEZE_NEAR_SEAM = 0``) and
then the last layer of one leg inverts: that region of the as-built mesh is already its weakest (corner scaled J
0.096) and a ring sliding along the axis folds it.  ``FREEZE_NEAR_SEAM`` holds that many end rings of every leg
as fixed as the open rim; the script prints what each choice costs.

**Order 1 only.**  The smoother works on the linear shadow and the rebuild is linear (the original is order 3 with
Fourier-refit curved wall nodes; this is its corner geometry -- identical corners, none of the curvature).  There
is no lift back to a curved wall.  All numbers below are order-1 meshes; ``SAMPLE_ORDER`` is the finer lattice the
trilinear map is re-read on.

    PYTHONPATH=. python examples/smoother/quad/cs_carotid_hex.py

Writes (current directory) ``cs_carotid_hex_{before,after}.vtu`` (original / rebuilt on the smoothed wall),
``_free_after`` (smoother free at the seam: the failing run), ``_perturbed`` / ``_perturbed_smoothed``; each with ``sj``
(scaled Jacobian, ``SAMPLE_ORDER`` lattice) and ``bl`` (1 on a wall-layer hex) per hex.  ``cs_carotid_hex_section_{before,after}.vtp`` are the two plane cuts (``sj``, ``plane`` per
polygon) and ``cs_carotid_hex_section.png`` the figure.  Also ``cs_carotid_hex.vtu`` (the final mesh with boundary names).
``mesh`` is the final ``HexMesh`` (rebuilt on the smoothed wall, same layers).

**Timing.**  With ``TIMING = True`` the script ends with a table of where the time went, from reading the tessellation to
the exported rebuilt mesh: one run, serial, wall clock (``time.perf_counter``), a rough estimate that varies by machine and
run.  The stages are marked by one-line ``lap(...)`` calls after the steps they time; the stressed variants and the other
smoothing variants are charged to a separate ``extras`` line, so they do not enter the table's percentages.
"""

import copy
import logging
import os
import time

import numpy as np
from scipy.spatial import cKDTree

from nekmeshpy import Tags, TriMesh, hexmesh, linemesh, quadmesh, trimesh, writer
from nekmeshpy.core import conform
from nekmeshpy.hexmesh import Seam
from nekmeshpy.quadmesh.smoother import constrain as cn
from nekmeshpy.quadmesh.smoother import smooth

logging.basicConfig(level=logging.INFO, format="%(message)s")

# -- parameters --------------------------------------------------------------
_DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir, "data")
VTX = os.path.join(_DATA, "car.vtx")
TRI = os.path.join(_DATA, "car.tri")

N_HALF = 8                    # carotid.py's resolution (half ring, multiple of 4)
N_SLICES = 30
MIN_LOOP_PTS = 6
CENTER_SCALE = 0.5
QUADRANT_SCALE = 0.45
PROJECT_TO_STL = True
FOURIER_KEEP = 0.5
FLUX_OFFSET = 2
RADIAL = np.array([0.0, 0.4, 0.7, 0.9, 1.0])      # carotid.py's own layers: four, the wall layer 0.1 of the gap

FREEZE_NEAR_SEAM = 1          # smoother also holds the last this-many wall rings of every leg fixed (1 = the shared seam ring)
ALT_FREEZE = 2                # compared against: hold one ring more
SCHEDULE = smooth.user_schedule(outer=2, n_opt=10)          # as cs_wall_surface.py
NOISE = 0.2                   # stressed variant: tangential perturbation, fraction of the local edge
SEED = 13
TIMING = True                 # print the stage timing table at the end (extras are timed apart, see lap())
STRESS = True                 # also rebuild hexes on a perturbed wall and on its smoothed repair
SAMPLE_ORDER = 7              # the solver's order: finer lattice the trilinear map is re-read on
OUT = "cs_carotid_hex"
FLUX_UPSTREAM, FLUX_DOWNSTREAM = "flux_upstream", "flux_downstream"
VEL_BC = {"wall": "W  ", "trunk_outlet": "v  ", "top_outlet_1": "int", "top_outlet_2": "O  ",
          "flux_1": {FLUX_UPSTREAM: "f1 ", FLUX_DOWNSTREAM: None},
          "flux_2": {FLUX_UPSTREAM: "f2 ", FLUX_DOWNSTREAM: None}}
LEG_CAP = {0: ("attach1", "attach2"), 1: ("attach1", "attach3"), 2: ("attach2", "attach3")}
OUTLET_NAME = ["trunk_outlet", "top_outlet_1", "top_outlet_2"]

# -- timing: lap(stage) charges the time since the previous lap to ``stage`` -------------------------------------
S1 = "1  read the tessellation (TriMesh)"
S2 = "2  seam fields, cut into legs"
S3 = "3  seam rings, leg rings, spine"
S4 = "4  O-grid sections, extrusion, welds, tags (original hexes)"
S5 = "5  extract the wall quads"
S6 = "6  smoother set-up (gate, overlaps, surface, normals)"
S7 = {"laplace": "7a smoothing: laplace", "optimize": "7b smoothing: optimize", "laplace_local": "7c smoothing: local lifts"}
S8 = "8  rebuild the hexes on the smoothed wall"
S9 = "9  quality readings (original and rebuilt)"
S10 = "10 export (original and rebuilt, .vtu)"
EXTRAS = "extras"                # variants beyond original -> smoothed (held) -> rebuilt, and their readings
LAPS = {}
_last = time.perf_counter()


def lap(stage):
    global _last
    now = time.perf_counter()
    LAPS[stage] = LAPS.get(stage, 0.0) + now - _last
    _last = now


# -- carotid.py's pipeline (copied) -------------------------------------------
def order_openings(surf):
    loops = surf.boundary_loops()
    assert len(loops) == 3, "expected exactly 3 boundary loops, got %d" % len(loops)
    Z = surf.points[:, 2]
    X = surf.points[:, 0]
    iA = int(np.argmin(np.array([Z[c].mean() for c in loops])))
    rest = [i for i in range(3) if i != iA]
    order = np.argsort(np.array([X[loops[i]].mean() for i in rest]), kind="stable")
    return [loops[iA], loops[rest[order[0]]], loops[rest[order[1]]]]


def loop_of(points):
    """The closed wall ring through ``points`` (order 1), tagged ``wall`` -- what ``fourier_wall`` returns."""
    return linemesh.loft(points, loop=True, element_tags="wall")


def ogrid_leg(fine_rings, seam_ring, spine, surface, frlev, radial, move):
    """carotid.py's ``ogrid_leg`` at order 1.  ``move`` maps a wall ring ``(M,3)`` to its replacement (``None``:
    the original Fourier-refit ring).  Returns the slices and each station's wall ring."""
    seam_pts = seam_ring.points
    spine_pts = spine.points
    M = seam_pts.shape[0]
    nh = M // 2
    rings = trimesh.ops.conform_ring_stack(fine_rings, seam_pts, np.asarray(frlev, dtype=float), nh)
    nr = len(rings) + 1
    RS = np.zeros((nr, M, 3))
    for k in range(nr - 1):
        RS[k] = rings[k]
    RS[nr - 1] = seam_pts
    if PROJECT_TO_STL:
        sub = trimesh.ops.project_points(surface, RS[0:nr - 1].reshape((nr - 1) * M, 3))
        RS[0:nr - 1] = sub.reshape(nr - 1, M, 3)
    RS[nr - 1] = seam_pts
    A1, A2 = spine_pts[0], spine_pts[-1]
    dev = spine_pts - (A1 + (np.arange(nh + 1)[:, None] / nh) * (A2 - A1))
    ringlev = np.arange(nr) / (nr - 1)
    slices, walls = [], []
    for k in range(nr):
        if k == nr - 1:
            # shared with the other two legs, so moved once as a ring, never refit per leg.  The ring stack
            # above was conformed onto the *original* seam: the conform is part of the construction, the move
            # is a replacement of the finished wall ring.
            wall = seam_ring if move is None else loop_of(move(seam_ring.points))
        else:
            wall = trimesh.ops.fourier_wall(RS[k], 1, "wall", keep=FOURIER_KEEP)
            if move is not None:
                wall = loop_of(move(wall.points))
        wpts = wall.points
        e1, e2 = wpts[0], wpts[nh]
        spn = (e1 + (np.arange(nh + 1)[:, None] / nh) * (e2 - e1)) + ringlev[k] * dev
        spn = trimesh.ops.resample_polyline(spn, quadmesh.spine_fractions(nh // 4, radial, QUADRANT_SCALE))
        slices.append(quadmesh.spined_ogrid(wall, radial, spine=linemesh.loft(spn, order=1),
                                            center_scale=CENTER_SCALE, quadrant_scale=QUADRANT_SCALE))
        walls.append(wpts.copy())
    return slices, walls


def cap_tags(slice_, first, second):
    half = slice_.n_quads // 2
    return Tags.from_dense(np.array([first] * half + [second] * (slice_.n_quads - half)))


def flux_name_for(outlet_name):
    return {"top_outlet_2": "flux_1", "top_outlet_1": "flux_2"}.get(outlet_name, "")


t_geo = _t_start = time.perf_counter()
lap(EXTRAS)                                                   # (the function definitions above)
surf = TriMesh.from_files(VTX, TRI)
gloops = order_openings(surf)
lap(S1)
U = trimesh.ops.seam_fields(surf, gloops)
V, faces = trimesh.ops.cut_surface_from_fields(surf, U)
lap(S2)
seam_rings0, _, _, spine0 = trimesh.ops.seam_rings(V, faces, gloops, N_HALF, order=1, keep=FOURIER_KEEP,
                                                   element_tag="wall")
lap(S3)
levels = np.linspace(0, 1, N_SLICES + 2)[1:-1]
leg_rings = []
for leg in range(3):
    sub, us, _, _, _ = trimesh.ops.leg_field(V, faces, leg, gloops)
    lap(S2)
    leg_rings.append(trimesh.ops.extract_rings(sub, us, levels, MIN_LOOP_PTS))
    lap(S3)
geo_time = time.perf_counter() - t_geo


def build(radial, move=None):
    """Hex mesh through carotid.py's recipe with layers ``radial``; ``move`` replaces the wall rings.
    Returns ``(mesh, wall_rings)`` with ``wall_rings`` a list of ``(M,3)`` station rings (every leg, every
    station, in the final positions)."""
    blocks, flux_seams, seam_block, wall_rings = [], [], {}, []
    for leg in range(3):
        fr, frlev = copy.deepcopy(leg_rings[leg])      # the ring stack is consumed in place
        slices, walls = ogrid_leg(fr, seam_rings0[leg], spine0, surf, frlev, radial, move)
        wall_rings += walls
        flux_name = flux_name_for(OUTLET_NAME[leg])
        cap = cap_tags(slices[-1], *LEG_CAP[leg])
        if flux_name and 0 < FLUX_OFFSET < len(slices) - 1:
            off = FLUX_OFFSET
            blocks.append(hexmesh.loft(slices[:off + 1], first_tag=OUTLET_NAME[leg], last_tag=flux_name,
                                       element_tags=FLUX_DOWNSTREAM))
            blocks.append(hexmesh.loft(slices[off:], first_tag=flux_name, last_tag=cap,
                                       element_tags=FLUX_UPSTREAM))
            flux_seams.append(Seam(len(blocks) - 2, flux_name, len(blocks) - 1, flux_name, attach_tag=flux_name))
        else:
            blocks.append(hexmesh.loft(slices, first_tag=OUTLET_NAME[leg], last_tag=cap))
        seam_block[leg] = len(blocks) - 1
    m = hexmesh.attach(blocks, flux_seams + [
        Seam(seam_block[0], "attach1", seam_block[1], "attach1"),
        Seam(seam_block[0], "attach2", seam_block[2], "attach2"),
        Seam(seam_block[1], "attach3", seam_block[2], "attach3")])
    return m, wall_rings


def mover(src_points, dst_points):
    """``move(P)``: each row of ``P`` is a vertex of the wall ``src_points``; return the same vertex in
    ``dst_points`` (a bit-exact lookup, so a vertex shared by three legs lands identically in all three)."""
    tree = cKDTree(src_points)
    scale = float(np.ptp(src_points, axis=0).max())

    def move(P):
        d, i = tree.query(P)
        if d.max() > 1e-9 * scale:
            raise AssertionError("a ring point is not a wall vertex (%.2e)" % d.max())
        return dst_points[i]
    return move


# -- measurements -------------------------------------------------------------
_SJ = {}


def fine_sj(m):
    """Per-hex scaled Jacobian of the order-1 map re-read on the (SAMPLE_ORDER+1)^3 lattice (cached per mesh)."""
    if id(m) not in _SJ:
        _SJ[id(m)] = (m, hexmesh.scaled_jacobian(m, order=SAMPLE_ORDER))
    return _SJ[id(m)][1]


def stats(sj):
    return dict(min=float(sj.min()), mean=float(sj.mean()), q05=float(np.quantile(sj, 0.05)),
                n_inv=int((sj <= 0).sum()))


def hex_reading(m):
    names, counts = m.face_tags.unique(return_counts=True)
    return dict(corner=stats(hexmesh.corner_scaled_jacobian(m)), fine=stats(fine_sj(m)),
                volume=hexmesh.volume(m), n=m.n_hexes,
                ok=bool(hexmesh.is_watertight(m) and hexmesh.is_conforming(m)),
                faces={str(k): int(v) for k, v in zip(names, counts)})


def wall_reading(w):
    nrm, _ = cn.surface_normals(w.points, w.corners, cn.claimed_surfaces(res))
    q = smooth.quality(w.points, w.corners, nrm)
    s = smooth.shape_stats(w.points, w.corners)
    return dict(q=q, size_max=s.size_ratio_max, size_mean=s.size_ratio_mean,
                ov=int(quadmesh.overlapping_quads(w).size))


def layer_reading(m, rings):
    """The wall layer of ``m``: for every wall hex, each wall corner's edge into the hex (the first layer's
    thickness) against the local radius (mean distance of that station's wall ring to its centroid), the layer's
    aspect ratio (mean wall-face edge / thickness) and the scaled Jacobian of the hexes in it."""
    ids = hexmesh.tagged_faces(m, "wall")
    rows, _ = hexmesh.face_rows(m, ids)
    he, hf = rows[:, 0], rows[:, 1] - 1
    C, F = m.corners[he], m.FACE_POINTS[hf]
    edges = conform._LOCAL_EDGES[3]
    ring_pts = np.vstack(rings)
    ring_rad = np.concatenate([np.full(len(r), np.linalg.norm(r - r.mean(axis=0), axis=1).mean()) for r in rings])
    tree = cKDTree(ring_pts)
    th, rel, ar = [], [], []
    for r in range(len(he)):
        on = set(F[r].tolist())
        wxyz = m.points[C[r, F[r]]]
        tang = np.mean([np.linalg.norm(wxyz[i] - wxyz[(i + 1) % 4]) for i in range(4)])
        for a, b in edges:
            if (a in on) != (b in on):
                w, n = (a, b) if a in on else (b, a)
                t = np.linalg.norm(m.points[C[r, n]] - m.points[C[r, w]])
                th.append(t)
                rel.append(t / ring_rad[tree.query(m.points[C[r, w]])[1]])
                ar.append(tang / t)
    wall_hex = np.unique(he)
    return dict(rel=np.array(rel), th=np.array(th), ar=np.array(ar), n=wall_hex.size, wall_hex=wall_hex,
                sj=stats(fine_sj(m)[wall_hex]), sj_corner=stats(hexmesh.corner_scaled_jacobian(m)[wall_hex]))


def hex_line(label, r):
    c, f = r["corner"], r["fine"]
    print("  %-34s %6d hexes | corner min %7.4f mean %.4f q05 %.4f inverted %d | order-%d min %7.4f mean %.4f "
          "q05 %.4f inverted %d | vol %.3f | conformal+watertight %s"
          % (label, r["n"], c["min"], c["mean"], c["q05"], c["n_inv"], SAMPLE_ORDER, f["min"], f["mean"], f["q05"],
             f["n_inv"], r["volume"], r["ok"]))


def wall_line(label, n, r):
    print("  %-34s %d quads | inverted %d overlapping %d | scaled J min %.4f mean %.4f q05 %.4f | size ratio "
          "max %.3f mean %.3f" % (label, n, r["q"].n_inverted, r["ov"], r["q"].sj_min, r["q"].sj_mean,
                                  r["q"].sj_q05, r["size_max"], r["size_mean"]))


def worst_hex(m, rings):
    """Where the worst corner-reading hex sits: ``(hex id, leg, station k, ring index)``, nearest station ring."""
    j = int(np.argmin(hexmesh.corner_scaled_jacobian(m)))
    d, i = cKDTree(np.vstack(rings)).query(m.points[m.corners[j]].mean(axis=0))
    nr, M = len(rings) // 3, rings[0].shape[0]
    st, ri = i // M, i % M
    return j, st // nr, st % nr, ri


# -- 1. the original wall, and its smoothed versions --------------------------
print("carotid hex mesh on the smoothed wall: order 1, RADIAL %s (the original's four layers)"
      % np.round(RADIAL, 3).tolist())
print("  geometry stage (seam fields, cuts, ring stacks): %.1f s" % geo_time)
T = {}
t0 = time.perf_counter()
orig, rings_orig = build(RADIAL)
T["orig"] = time.perf_counter() - t0
lap(S4)
wall0 = hexmesh.boundary_mesh(orig, "wall")          # the original is already order 1: nothing to linearize
lap(S5)
tess = TriMesh.from_files(VTX, TRI)
surface = cn.surface_from_trimesh(tess.points, tess.tris, name="carotid tessellation")
rim = quadmesh.boundary_points(wall0)
tree0 = cKDTree(wall0.points)
nr_leg = len(rings_orig) // 3
ell = cn.vertex_length(wall0.points, wall0.corners)
gate, n_overlap = quadmesh.check_connectivity(wall0), int(quadmesh.overlapping_quads(wall0).size)
lap(S6)
print("  as-built order-1 hexes: %d; wall %d quads, %d vertices, %d on the open rim; each leg has %d stations "
      "(last = shared seam ring)" % (orig.n_hexes, wall0.n_quads, wall0.n_points, rim.size, nr_leg))
print("  smoother's input checks on the wall: connectivity gate %s, %d overlapping quads"
      % ("passed" if gate.passed else "FAILED", n_overlap))
lap(EXTRAS)


def frozen_set(n_last):
    """Open rim plus the wall vertices of the last ``n_last`` station rings of every leg."""
    near = [tree0.query(rings_orig[leg * nr_leg + k])[1] for leg in range(3) for k in range(nr_leg - n_last, nr_leg)]
    return np.union1d(rim, np.concatenate(near)) if near else rim


def resolve_for(frozen):
    free = np.setdiff1d(np.arange(wall0.n_points), frozen)
    return cn.resolve(wall0, [cn.Frozen(frozen), cn.OnSurface(free, surface)])


def smooth_wall(w0, frozen, main=False):
    """Smooth ``w0``; ``main`` charges the set-up and each stage to the table (else to the extras)."""
    r = resolve_for(frozen)
    nrm, _ = cn.surface_normals(w0.points, w0.corners, cn.claimed_surfaces(r))
    lap(S6 if main else EXTRAS)
    t = time.perf_counter()
    out = w0
    for stage in SCHEDULE:       # one stage at a time with the same normals: identical to one run_schedule call
        out, _ = smooth.run_schedule(out, r, [stage], normals=nrm)      # (points compared bit for bit, scratch check)
        lap(S7[stage[0]] if main else EXTRAS)
    return out, time.perf_counter() - t


FROZEN = frozen_set(FREEZE_NEAR_SEAM)
lap(S6)
res = resolve_for(rim)                       # the tessellation constraint used to SCORE every wall (rim only)
walls = {}
for key, frozen in (("free", rim), ("held", FROZEN), ("alt", frozen_set(ALT_FREEZE))):
    walls[key], T["smooth_" + key] = smooth_wall(wall0, frozen, main=(key == "held"))
    mv = np.linalg.norm(walls[key].points - wall0.points, axis=1) / ell
    print("  wall smoothed, %s: %d vertices held, %.1f s, max move %.3f local edges (mean %.4f), held vertices "
          "moved %.1e" % ("only the open rim fixed" if key == "free" else "last %d ring(s) of every leg also fixed"
                          % (FREEZE_NEAR_SEAM if key == "held" else ALT_FREEZE), frozen.size, T["smooth_" + key], mv.max(), mv.mean(),
                          np.abs(walls[key].points[frozen] - wall0.points[frozen]).max()))
    lap(EXTRAS)

variants = {}                                  # label -> (mesh, rings, build seconds)


def add(label, radial, move=None, main=False):
    t = time.perf_counter()
    m, r = build(radial, move)
    variants[label] = (m, r, time.perf_counter() - t)
    lap(S8 if main else EXTRAS)


variants["original"] = (orig, rings_orig, T["orig"])
add("smoothed (free)", RADIAL, mover(wall0.points, walls["free"].points))
add("smoothed (held)", RADIAL, mover(wall0.points, walls["held"].points), main=True)
add("smoothed (held more)", RADIAL, mover(wall0.points, walls["alt"].points))
if STRESS:
    rng = np.random.default_rng(SEED)
    movable = np.flatnonzero(res.kind != cn.FROZEN)
    P = np.array(wall0.points, dtype=float)
    P[movable] += rng.uniform(-NOISE, NOISE, (movable.size, 3)) * ell[movable, None]
    wall_k = smooth.with_points(wall0, cn.snap(P, res))
    wall_ks, T["smooth_perturbed"] = smooth_wall(wall_k, FROZEN)
    add("perturbed wall", RADIAL, mover(wall0.points, wall_k.points))
    add("perturbed+smoothed", RADIAL, mover(wall0.points, wall_ks.points))

# -- 2. the readings ----------------------------------------------------------
print("\nHEX QUALITY (corner = the 8-corner trilinear hex, which is the geometry .re2 exports; order-%d = the same "
      "map re-read on a %d^3 lattice; q05 = 5th percentile)" % (SAMPLE_ORDER, SAMPLE_ORDER + 1))
readings = {}
for label, (m, _rings, _secs) in variants.items():
    readings[label] = hex_reading(m)
    lap(S9 if label in ("original", "smoothed (held)") else EXTRAS)
    hex_line(label, readings[label])
names0 = readings["original"]["faces"]
for label, r in readings.items():
    assert r["faces"].keys() == names0.keys(), "boundary names changed in %s" % label
print("  boundary names, faces per name: original %s | final %s"
      % (names0, readings["smoothed (held)"]["faces"]))
print("  build time per hex mesh (carotid.py recipe, order 1): %s s"
      % ", ".join("%.1f" % v[2] for v in variants.values()))
FINAL = "smoothed (held)"
mesh = variants[FINAL][0]
print("  final mesh: watertight %s, conforming %s, %d hexes, %d points (hexmesh.is_overlap_free is a ~14 s "
      "search, not run here)" % (hexmesh.is_watertight(mesh), hexmesh.is_conforming(mesh), mesh.n_hexes,
                                 mesh.n_points))
print("  volume change, smoothed (held) vs original: %+.4f %%"
      % (100 * (readings[FINAL]["volume"] / readings["original"]["volume"] - 1)))
for label in ("original", "smoothed (free)", FINAL):
    j, leg, k, ri = worst_hex(variants[label][0], variants[label][1])
    print("  worst hex of %-30s: hex %d, leg %d, station %d of %d, ring index %d (of %d), corner SJ %.4f"
          % (label, j, leg, k, nr_leg - 1, ri, rings_orig[0].shape[0],
             hexmesh.corner_scaled_jacobian(variants[label][0])[j]))

print("\nWALL (the hexes' own wall faces, scored against the tessellation's normals; 'overlapping' = "
      "quadmesh.overlapping_quads)")
wall_of = {}
for label in ("original", "smoothed (free)", FINAL, "smoothed (held more)"):
    w = hexmesh.boundary_mesh(variants[label][0], "wall")
    wall_of[label] = wall_reading(w)
    wall_line(label, w.n_quads, wall_of[label])
    if "smoothed" in label:
        key = "free" if "free" in label else "alt" if "more" in label else "held"
        dev = np.abs(w.points[cKDTree(w.points).query(walls[key].points)[1]] - walls[key].points).max()
        print("  %-34s hexes' wall vs the smoothed wall: max difference %.1e" % ("", dev))
if STRESS:
    for label, w in (("perturbed wall (input)", wall_k), ("perturbed+smoothed (held)", wall_ks)):
        wall_of[label] = wall_reading(w)
        wall_line(label, w.n_quads, wall_of[label])

print("\nBOUNDARY LAYER (first layer = the wall hexes' edges into the volume; local radius = mean distance of "
      "the station's wall ring to its centroid)")
layers = {}
for label, (m, rings, _) in variants.items():
    layers[label] = L = layer_reading(m, rings)
    rel = L["rel"]
    print("  %-34s thickness/radius min %.4f median %.4f max %.4f | aspect (wall edge / thickness) median %.1f max "
          "%.1f | wall-hex SJ order-%d min %7.4f mean %.4f q05 %.4f inverted %d (%d wall hexes)"
          % (label, rel.min(), np.median(rel), rel.max(), np.median(L["ar"]), L["ar"].max(), SAMPLE_ORDER,
             L["sj"]["min"], L["sj"]["mean"], L["sj"]["q05"], L["sj"]["n_inv"], L["n"]))
a, b = "original", FINAL
da, db = layers[a]["rel"], layers[b]["rel"]
print("  smoothing's effect on the first layer (same layers): thickness/radius min %.4f -> %.4f, median %.4f -> "
      "%.4f, max %.4f -> %.4f" % (da.min(), db.min(), np.median(da), np.median(db), da.max(), db.max()))

sj_a, sj_b = fine_sj(variants["original"][0]), fine_sj(variants[FINAL][0])
print("\nper hex, smoothed (held) minus original wall, order-%d scaled J: worse in %d / %d hexes (largest drop %.3f, "
      "largest gain %.3f)" % (SAMPLE_ORDER, (sj_b < sj_a - 1e-9).sum(), sj_b.size, (sj_a - sj_b).max(), (sj_b - sj_a).max()))


lap(EXTRAS)


# -- 3. cross-sections: scaled J of the hexes a plane cuts ---------------------
def plane_cut(m, sj, origin, normal):
    """Polygons where the plane cuts the hexes (each hex's cut = the plane's crossings of its 12 edges, ordered
    around their centroid), with the hex's scaled Jacobian."""
    X = m.points[m.corners]
    d = (X - origin) @ normal
    e12 = conform._LOCAL_EDGES[3]
    u = np.cross(normal, [0.0, 0.0, 1.0])
    u /= np.linalg.norm(u)
    v = np.cross(normal, u)
    polys, vals = [], []
    for e in np.flatnonzero((d.min(axis=1) < 0) & (d.max(axis=1) > 0)):
        da, db = d[e, e12[:, 0]], d[e, e12[:, 1]]
        x = da * db < 0
        t = (da[x] / (da[x] - db[x]))[:, None]
        pts = X[e, e12[x, 0]] + t * (X[e, e12[x, 1]] - X[e, e12[x, 0]])
        if len(pts) < 3:
            continue
        c = pts.mean(axis=0)
        ang = np.arctan2((pts - c) @ v, (pts - c) @ u)
        polys.append(pts[np.argsort(ang)])
        vals.append(sj[e])
    return polys, np.array(vals)


def planes_for(rings):
    """The two cuts: perpendicular to the trunk's axis mid-leg, and the longitudinal plane through the trifurcation
    containing the trunk axis and the direction between the two branches."""
    R = np.array(rings).reshape(3, nr_leg, -1, 3).mean(axis=2)             # station centroids
    mid = nr_leg // 2
    ax = R[0, mid + 1] - R[0, mid - 1]
    p1 = (R[0, mid], ax / np.linalg.norm(ax))
    trunk = R[0, nr_leg - 3] - R[0, nr_leg - 12]
    trunk /= np.linalg.norm(trunk)
    bc = R[1, nr_leg - 12] - R[2, nr_leg - 12]
    n2 = np.cross(trunk, bc)
    return {"leg A cross-section": p1, "trifurcation, along the trunk": (spine0.points.mean(axis=0),
                                                                          n2 / np.linalg.norm(n2))}


def write_vtp(name, cuts):
    """ASCII PolyData of the cut polygons (``sj`` and ``plane`` per cell), a few hundred kB."""
    pts, conn, off, sj, pl = [], [], [], [], []
    for pid, (polys, vals) in enumerate(cuts):
        for pg, v in zip(polys, vals):
            conn += list(range(len(pts), len(pts) + len(pg)))
            pts += pg.tolist()
            off.append(len(conn))
            sj.append(v)
            pl.append(pid)
    with open(name, "w") as f:
        f.write('<?xml version="1.0"?>\n<VTKFile type="PolyData" version="1.0" byte_order="LittleEndian">\n<PolyData>\n'
                '<Piece NumberOfPoints="%d" NumberOfVerts="0" NumberOfLines="0" NumberOfStrips="0" '
                'NumberOfPolys="%d">\n<Points><DataArray type="Float32" NumberOfComponents="3" format="ascii">\n'
                % (len(pts), len(off)))
        f.write("\n".join("%.5g %.5g %.5g" % tuple(p) for p in pts))
        f.write('\n</DataArray></Points>\n<Polys><DataArray type="Int32" Name="connectivity" format="ascii">\n')
        f.write(" ".join(map(str, conn)))
        f.write('\n</DataArray><DataArray type="Int32" Name="offsets" format="ascii">\n')
        f.write(" ".join(map(str, off)))
        f.write('\n</DataArray></Polys>\n<CellData Scalars="sj"><DataArray type="Float32" Name="sj" format="ascii">\n')
        f.write(" ".join("%.4f" % v for v in sj))
        f.write('\n</DataArray><DataArray type="Int32" Name="plane" format="ascii">\n')
        f.write(" ".join(map(str, pl)))
        f.write("\n</DataArray></CellData>\n</Piece>\n</PolyData>\n</VTKFile>\n")


def section_figure(pairs, name):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_under("red")
    planes = planes_for(rings_orig)
    fig, axs = plt.subplots(len(pairs), len(planes), figsize=(10.5, 4.6 * len(pairs)), squeeze=False,
                            gridspec_kw={"width_ratios": [1.0, 0.62]})
    cuts_out = {}
    for r, (title, label) in enumerate(pairs):
        m = variants[label][0]
        for c, (pname, (o, n)) in enumerate(planes.items()):
            polys, vals = plane_cut(m, fine_sj(m), o, n)
            cuts_out.setdefault(title, []).append((polys, vals))
            u = np.cross(n, [0.0, 0.0, 1.0])
            u /= np.linalg.norm(u)
            v = np.cross(n, u)
            xy = [np.column_stack([(p - o) @ u, (p - o) @ v]) for p in polys]
            pc = PolyCollection(xy, array=vals, cmap=cmap, edgecolors="0.25", linewidths=0.25, clim=(0.0, 1.0))
            ax = axs[r, c]
            ax.add_collection(pc)
            allp = np.vstack(xy)
            ax.set_xlim(allp[:, 0].min() - 0.05, allp[:, 0].max() + 0.05)
            ax.set_ylim(allp[:, 1].min() - 0.05, allp[:, 1].max() + 0.05)
            ax.set_aspect("equal")
            ax.set_title("%s\n%s: %d hexes cut, min SJ %.3f" % (title, pname, len(polys), vals.min()), fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
    fig.subplots_adjust(left=0.02, right=0.89, top=0.92, bottom=0.03, hspace=0.18, wspace=0.08)
    cax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(pc, cax=cax, extend="min").set_label("scaled Jacobian (order-%d sampling; red = inverted)" % SAMPLE_ORDER,
                                                      fontsize=9)
    fig.savefig(name, dpi=150)
    plt.close(fig)
    return cuts_out


def write_hex(label, name):
    """``.vtu`` of an order-1 hex mesh with ``sj`` (order-SAMPLE_ORDER scaled J) and ``bl`` (1 = wall-layer hex)."""
    m, rings, _ = variants[label]
    bl = np.zeros(m.n_hexes)
    bl[layers[label]["wall_hex"]] = 1.0
    X, conn, ctype, bc = writer._hex_arrays(m, writer._as_bc(m, VEL_BC))
    writer._write_vtu(name, X, conn, ctype, bc_out=bc, cell_out=writer._cell_tags(m.element_tags, m.n_hexes),
                      cell_data={"sj": fine_sj(m), "bl": bl})


pairs = [("original (carotid.py recipe, corners)", "original"),
         ("rebuilt on the smoothed wall (this script)", FINAL)]
cuts = section_figure(pairs, OUT + "_section.png")
for (title, _label), tag in zip(pairs, ("before", "after")):
    write_vtp("%s_section_%s.vtp" % (OUT, tag), cuts[title])
lap(EXTRAS)
for label, tag in (("original", "before"), (FINAL, "after")):
    write_hex(label, "%s_%s.vtu" % (OUT, tag))
lap(S10)
write_hex("smoothed (free)", OUT + "_free_after.vtu")
if STRESS:
    write_hex("perturbed wall", OUT + "_perturbed.vtu")
    write_hex("perturbed+smoothed", OUT + "_perturbed_smoothed.vtu")
print("\nsection sizes: %s" % ", ".join("%s %d kB" % (f, os.path.getsize(f) // 1000) for f in sorted(os.listdir("."))
                                       if f.startswith(OUT + "_section")))
print("total %.1f s (seam/leg geometry %.1f, wall smoothing %.1f, final rebuild %.1f)"
      % (time.perf_counter() - t_geo, geo_time, T["smooth_held"], variants[FINAL][2]))
lap(EXTRAS)

# -- timing table (one run; a rough estimate) ------------------------------------------------------------------------
STAGES = [S1, S2, S3, S4, S5, S6, *S7.values(), S8, S9, S10]
TIMING_ROWS = [(name, LAPS.get(name, 0.0)) for name in STAGES]
TIMING_TOTAL = sum(sec for _, sec in TIMING_ROWS)
TIMING_EXTRAS = LAPS.get(EXTRAS, 0.0)
T_SMOOTH = sum(LAPS[name] for name in S7.values())
if TIMING:
    print("\nTIMING (one serial run, wall clock; a rough estimate -- it varies by machine and run)")
    for name, sec in TIMING_ROWS:
        print("  %-56s %8.2f s %6.1f %%" % (name, sec, 100 * sec / TIMING_TOTAL))
    print("  %-56s %8.2f s %6.1f %%" % ("total of the stages above", TIMING_TOTAL, 100.0))
    print("  %-56s %8.2f s   (not in the table: other smoothing variants, the perturbed input, their readings, figures)"
          % ("extras", TIMING_EXTRAS))
    print("  smoothing (7) vs building the original hexes (4) and rebuilding them (8): %.2f s vs %.2f s and %.2f s"
          % (T_SMOOTH, LAPS[S4], LAPS[S8]))
