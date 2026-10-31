"""Constraints: what a vertex may do while it is smoothed.

Three small studies, each reported by numbers computed here:

(a) A disk.  An O-grid of the unit circle whose boundary vertices were unevenly spaced along the
    circle and whose interior is jittered.  With the boundary **frozen** the uneven spacing is
    permanent; with the wall claimed as ``OnCurve(circle)`` the vertices **slide** along the circle
    (and are snapped back every step), so the spacing evens out and the vertices never leave it.

(b) A box surface (six faces, closed, no boundary).  Face vertices are ``OnSurface`` planes, the 12
    creases are ``OnCurve`` lines (they slide along their own edge), and the 8 corners are
    ``Frozen``.  After smoothing: corners moved by exactly 0, creases still on their lines, faces on
    their planes.

(c) A thin boundary layer.  A rectangle whose bottom rows are a graded layer a few hundredths thick.
    The shape metric wants squares, so unprotected the layer inflates; ``KeepLayer`` freezes the
    layer's rings exactly as meshed.  The same run is done both ways.

    PYTHONPATH=. python examples/smoother/03_constraints.py

Produces ``03_<a|b|c>_*.vtu`` (before / after, with ``sj_min`` and the displacement) and
``03_*_frames.json`` for the sliding disk and the kept layer.  ``mesh`` is the sliding disk.
"""

import time

import numpy as np

from nekmeshpy import linemesh, quadmesh
from nekmeshpy.io.writer import quad_to_vtu
from nekmeshpy.quadmesh.smoother import constrain as cn
from nekmeshpy.quadmesh.smoother import frames, smooth

# -- parameters --------------------------------------------------------------
DISK_RADIUS = 1.0
DISK_WALL_N = 24             # wall segments
DISK_JITTER = 0.2            # interior jitter, fraction of the wall spacing
DISK_SLIP = 0.3              # boundary vertex angular slip, fraction of the wall spacing
BOX_HALF = 1.0
BOX_N = 5                    # cells per box edge
BOX_JITTER = 0.3             # fraction of a cell
LAYER_NX = 12                # strip: cells along x
LAYER_ROWS = (0.0, 0.004, 0.0096, 0.017, 0.030)   # y of the thin layer's rows (4 layers)
STRIP_CELLS = 8              # uniform cells above the layer, up to y = 1
KEEP_RINGS = 4               # rings KeepLayer protects = the layer's 4 layers
SEED = 3
SCHEDULE = smooth.user_schedule(outer=5)


def per_quad_sj(m, nrm):
    return smooth.corner_metrics(m.points, m.corners, nrm).sj.min(axis=1)


def line(label, q, extra=""):
    print("  %-9s inverted %d | sj min %.4f mean %.4f | worst 5%% %.4f | F/quad %.4f %s"
          % (label, q.n_inverted, q.sj_min, q.sj_mean, q.sj_q05, q.objective_mean, extra))


# ============================================================================
# (a) disk: frozen boundary versus a boundary that slides on the circle
# ============================================================================
rng = np.random.default_rng(SEED)
t_start = time.perf_counter()
disk0 = quadmesh.ogrid(linemesh.circle(DISK_RADIUS, DISK_WALL_N, element_tag="wall"), 6, 4, wall_tag="wall")
P = np.array(disk0.points, dtype=float)
on_wall = np.isclose(np.hypot(P[:, 0], P[:, 1]), DISK_RADIUS, atol=1e-9)
wall_ids = np.flatnonzero(on_wall)
h_wall = 2 * np.pi * DISK_RADIUS / DISK_WALL_N
inner = np.flatnonzero(~on_wall)
P[inner, :2] += rng.uniform(-DISK_JITTER, DISK_JITTER, (inner.size, 2)) * h_wall
theta = np.arctan2(P[wall_ids, 1], P[wall_ids, 0]) + rng.uniform(-DISK_SLIP, DISK_SLIP, wall_ids.size) * h_wall
P[wall_ids, 0], P[wall_ids, 1] = DISK_RADIUS * np.cos(theta), DISK_RADIUS * np.sin(theta)
disk = smooth.with_points(disk0, P)
nrm_disk = smooth.reference_normals(disk0.points, disk0.corners)
circle = cn.circle([0, 0, 0], DISK_RADIUS)


def wall_gaps(pts):
    ang = np.sort(np.arctan2(pts[wall_ids, 1], pts[wall_ids, 0]))
    gap = np.diff(np.r_[ang, ang[0] + 2 * np.pi])
    return gap.min() / gap.max()               # 1 = evenly spaced


print("(a) disk, %d quads, wall of %d vertices uneven by up to %.2f segment, interior jitter %.2f"
      % (disk.n_quads, wall_ids.size, DISK_SLIP, DISK_JITTER))
q_in = smooth.quality(disk.points, disk.corners, nrm_disk)
line("input", q_in, "| wall spacing min/max %.3f" % wall_gaps(disk.points))
disk_out = {}
for label, cons in (("frozen", []), ("sliding", [cn.OnCurve("wall", circle)])):
    res = cn.resolve(disk, cons)
    out, logs = smooth.run_schedule(disk, res, SCHEDULE, normals=nrm_disk, record=(label == "sliding"))
    disk_out[label] = (out, logs)
    radial = np.abs(np.hypot(out.points[wall_ids, 0], out.points[wall_ids, 1]) - DISK_RADIUS).max()
    line(label, logs[-1].after, "| wall spacing min/max %.3f | max radial error %.1e" % (wall_gaps(out.points), radial))
quad_to_vtu(disk, "03_a_disk_before.vtu", cell_data={"sj_min": per_quad_sj(disk, nrm_disk)})
for label, (out, _) in disk_out.items():
    quad_to_vtu(out, "03_a_disk_%s.vtu" % label, point_data={"displacement": out.points - disk.points},
                cell_data={"sj_min": per_quad_sj(out, nrm_disk)})
frames.write_frames("03_a_disk_frames.json", disk, disk_out["sliding"][1], nrm_disk,
                    title="03a disk: boundary sliding on the circle", decimals=5)
mesh = disk_out["sliding"][0]
print("  (%.1f s)" % (time.perf_counter() - t_start))

# ============================================================================
# (b) box surface: planes for the faces, lines for the creases, frozen corners
# ============================================================================
t_start = time.perf_counter()
box0 = quadmesh.box(BOX_HALF, BOX_N)
B = np.array(box0.points, dtype=float)
nrm_box = smooth.reference_normals(box0.points, box0.corners)
on_face = np.isclose(np.abs(B), BOX_HALF)                 # (P, 3): which axes sit on a face plane
n_face = on_face.sum(axis=1)
corner_ids, crease_ids, face_ids = (np.flatnonzero(n_face == k) for k in (3, 2, 1))
cell = 2 * BOX_HALF / BOX_N

rng = np.random.default_rng(SEED)
B[face_ids] += rng.uniform(-BOX_JITTER, BOX_JITTER, (face_ids.size, 3)) * cell * ~on_face[face_ids]
for v in crease_ids:                                      # slip along the crease only
    B[v] += rng.uniform(-BOX_JITTER, BOX_JITTER) * cell * ~on_face[v]
box = smooth.with_points(box0, B)

constraints = [cn.Frozen(corner_ids)]
for axis in range(3):
    for sign in (-1.0, 1.0):                              # six planes
        ids = face_ids[(on_face[face_ids, axis]) & (np.sign(B[face_ids, axis]) == sign)]
        normal = np.eye(3)[axis]
        constraints.append(cn.OnSurface(ids, cn.plane(sign * BOX_HALF * normal, normal)))
creases = {}                                              # twelve lines
for v in crease_ids:
    a1, a2 = np.flatnonzero(on_face[v])
    creases.setdefault((a1, a2, np.sign(B[v, a1]), np.sign(B[v, a2])), []).append(v)
for (a1, a2, s1, s2), ids in creases.items():
    origin = np.zeros(3)
    origin[a1], origin[a2] = s1 * BOX_HALF, s2 * BOX_HALF
    free_axis = 3 - a1 - a2
    constraints.append(cn.OnCurve(np.array(ids), cn.line(origin, np.eye(3)[free_axis])))
res = cn.resolve(box, constraints)

print("(b) box surface, %d quads: %d corners (frozen), %d crease points on %d lines, %d face points on 6 planes"
      % (box.n_quads, corner_ids.size, crease_ids.size, len(creases), face_ids.size))
q_in = smooth.quality(box.points, box.corners, nrm_box)
line("input", q_in)
box_out, box_logs = smooth.run_schedule(box, res, SCHEDULE, normals=nrm_box)
line("smoothed", box_logs[-1].after)
Q = box_out.points
corner_moved = np.abs(Q[corner_ids] - box.points[corner_ids]).max()
face_dist = np.abs(np.abs(Q[face_ids][on_face[face_ids]].reshape(-1)) - BOX_HALF).max()
crease_dist = max(np.abs(np.abs(Q[v][on_face[v]]) - BOX_HALF).max() for v in crease_ids)
moved_crease = np.abs(Q[crease_ids] - box.points[crease_ids]).max()
print("  corners moved: %.1e (max)   crease points off their lines: %.1e (max), slid by up to %.3f"
      % (corner_moved, crease_dist, moved_crease))
print("  face points off their planes: %.1e (max)   on_target_report passed: %s"
      % (face_dist, cn.on_target_report(Q, box.corners, res).passed))
quad_to_vtu(box, "03_b_box_before.vtu", cell_data={"sj_min": per_quad_sj(box, nrm_box)})
quad_to_vtu(box_out, "03_b_box_after.vtu", point_data={"displacement": Q - box.points},
            cell_data={"sj_min": per_quad_sj(box_out, nrm_box)})
print("  (%.1f s)" % (time.perf_counter() - t_start))

# ============================================================================
# (c) a thin layer: KeepLayer versus none
# ============================================================================
t_start = time.perf_counter()
rng = np.random.default_rng(SEED)
ys = np.r_[np.asarray(LAYER_ROWS), np.linspace(LAYER_ROWS[-1], 1.0, STRIP_CELLS + 1)[1:]]
xs = np.linspace(0.0, 1.0, LAYER_NX + 1)
XX, YY = np.meshgrid(xs, ys, indexing="ij")
S = np.stack([XX.ravel(), YY.ravel(), np.zeros(XX.size)], axis=1)
sidx = np.arange(S.shape[0]).reshape(len(xs), len(ys))
squads = np.stack([sidx[:-1, :-1], sidx[1:, :-1], sidx[1:, 1:], sidx[:-1, 1:]], axis=-1).reshape(-1, 4)
nrm_strip = smooth.reference_normals(S, squads)
above = np.zeros(S.shape[0], bool)
above[sidx[1:-1, len(LAYER_ROWS):-1]] = True               # free vertices above the layer
S[above, :2] += rng.uniform(-0.3, 0.3, (above.sum(), 2)) * [1 / LAYER_NX, 1 / STRIP_CELLS]
strip = quadmesh.QuadMesh.from_corners(S, squads)
wall_row = sidx[:, 0]                                      # seeds: the wall the layer grows from


def layer_top(pts):
    return pts[sidx[1:-1, KEEP_RINGS], 1]                  # y of the layer's outer ring, interior columns


thick0 = layer_top(strip.points)
print("(c) strip of %d quads; layer of %d rings, outer ring at y = %.4f (aspect ratio of its cells ~%.0f:1)"
      % (strip.n_quads, KEEP_RINGS, LAYER_ROWS[-1], (1 / LAYER_NX) / (LAYER_ROWS[1] - LAYER_ROWS[0])))
q_in = smooth.quality(strip.points, strip.corners, nrm_strip)
line("input", q_in)
strip_out = {}
for label, cons in (("no KeepLayer", []), ("KeepLayer", [cn.KeepLayer(wall_row, rings=KEEP_RINGS)])):
    res = cn.resolve(strip, cons)
    out, logs = smooth.run_schedule(strip, res, SCHEDULE, normals=nrm_strip, record=(label == "KeepLayer"))
    strip_out[label] = (out, logs)
    change = np.abs(layer_top(out.points) - thick0) / thick0
    line(label, logs[-1].after, "| layer thickness changed by max %.1f %%" % (100 * change.max()))
quad_to_vtu(strip, "03_c_layer_before.vtu", cell_data={"sj_min": per_quad_sj(strip, nrm_strip)})
for label, (out, _) in strip_out.items():
    quad_to_vtu(out, "03_c_layer_%s.vtu" % label.replace(" ", "_"), point_data={"displacement": out.points - strip.points},
                cell_data={"sj_min": per_quad_sj(out, nrm_strip)})
frames.write_frames("03_c_layer_frames.json", strip, strip_out["KeepLayer"][1], nrm_strip,
                    title="03c layer: KeepLayer on a thin boundary layer", decimals=5)
print("  (%.1f s)" % (time.perf_counter() - t_start))
