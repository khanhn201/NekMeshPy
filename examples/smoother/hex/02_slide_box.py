"""Hex smoothing with a sliding boundary: ``boundary="slide"`` against ``boundary="frozen"`` on a box.

The box is [0,1] x [0,1] x [0,0.6] meshed 7 x 7 x 4.  The perturbed input moves *every*
vertex but keeps the box exact: a face-interior vertex is displaced within its face plane,
an edge-interior vertex along its edge line, the 8 corners not at all, and an interior
vertex in 3-D (all by up to 0.3 of a cell).  So the boundary spacing is uneven and a frozen
boundary has to keep it, while ``boundary="slide"`` may even it out without leaving the
planes, lines and corners the box is made of.  Both runs use the same schedule,
``hs.suggest_schedule(q0, outer=3)``.

Printed: quality of both runs; for each, the largest distance of a face vertex from its
input plane, of an edge vertex from its input line, and the largest corner move (0.0
exactly means bit-identical); and the notes ``resolve`` wrote for the slide run.

    PYTHONPATH=. python examples/smoother/hex/02_slide_box.py

Writes ``02_slide_box_{input,frozen,slide}.vtu`` to the current directory and leaves
``mesh`` (the slide result), ``rows``, ``meshes`` and ``sj`` in the namespace.
"""

import time

import numpy as np

from nekmeshpy import writer
from nekmeshpy.hexmesh import HexMesh
from nekmeshpy.hexmesh import smoother as hs

# -- parameters --------------------------------------------------------------
SIZE = np.array([1.0, 1.0, 0.6])
N = (7, 7, 4)
PERTURB = 0.3                # fraction of a cell
SEED = 0
OUTER = 3

# -- the box ------------------------------------------------------------------
axes = [np.linspace(0.0, s, n + 1) for s, n in zip(SIZE, N)]
X, Y, Z = np.meshgrid(*axes, indexing="ij")
exact = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
idx = np.arange(exact.shape[0]).reshape(N[0] + 1, N[1] + 1, N[2] + 1)
c = [idx[:-1, :-1, :-1], idx[1:, :-1, :-1], idx[1:, 1:, :-1], idx[:-1, 1:, :-1],
     idx[:-1, :-1, 1:], idx[1:, :-1, 1:], idx[1:, 1:, 1:], idx[:-1, 1:, 1:]]
hexes = np.stack(c, axis=-1).reshape(-1, 8)

cell = SIZE / np.array(N)
on_wall = np.isclose(exact, 0.0) | np.isclose(exact, SIZE)       # (V,3): which planes a vertex is on
n_fixed = on_wall.sum(axis=1)                                     # 0 interior, 1 face, 2 edge, 3 corner
face_v, edge_v, corner_v = n_fixed == 1, n_fixed == 2, n_fixed == 3

# a vertex may move along exactly the axes it is not pinned on
rng = np.random.default_rng(SEED)
P = exact + rng.uniform(-PERTURB, PERTURB, exact.shape) * cell * ~on_wall
before = HexMesh.from_corners(P, hexes)
print("box: %d hexes %s; vertices: %d interior, %d face, %d edge, %d corner"
      % (before.n_hexes, "x".join(map(str, N)), int((n_fixed == 0).sum()), int(face_v.sum()),
         int(edge_v.sum()), int(corner_v.sum())))


def per_hex_sj(m):
    return hs.corner_metrics(m.points, m.corners).sj.min(axis=1)


def held(m):
    """Largest deviation from the input plane (face vertices), input line (edge vertices) and
    input position (corners).  A pinned coordinate is the one the vertex was on."""
    off = np.abs(m.points - before.points) * on_wall
    return dict(off_plane=float(off[face_v].max()), off_line=float(off[edge_v].max()),
                corner_move=float(np.abs(m.points[corner_v] - before.points[corner_v]).max()))


q0 = hs.quality(before.points, before.corners)
schedule = hs.suggest_schedule(q0, outer=OUTER)
print("\nschedule: suggest_schedule(outer=%d), %d stages" % (OUTER, len(schedule)))
print("%-8s inverted %3d | sj min %.4f mean %.4f | worst 5%% %.4f | F/hex %.4f"
      % ("input", q0.n_inverted, q0.sj_min, q0.sj_mean, q0.sj_q05, q0.objective_mean))

meshes = {"input": before}
rows = {"input": dict(q=q0, **held(before))}
for mode in ("frozen", "slide"):
    r = hs.resolve(before, [], boundary=mode)
    t0 = time.perf_counter()
    out, logs = hs.run_schedule(before, r, schedule)
    dt = time.perf_counter() - t0
    q = logs[-1].after
    meshes[mode] = out
    rows[mode] = dict(q=q, seconds=dt, notes=r.notes, counts=r.counts(), **held(out))
    h = rows[mode]
    print("%-8s inverted %3d | sj min %.4f mean %.4f | worst 5%% %.4f | F/hex %.4f | %.2f s"
          % (mode, q.n_inverted, q.sj_min, q.sj_mean, q.sj_q05, q.objective_mean, dt))
    print("         max off-plane (face) %.2e | max off-line (edge) %.2e | max corner move %.1f"
          % (h["off_plane"], h["off_line"], h["corner_move"]))
print("\nresolve (slide) counts: %s" % rows["slide"]["counts"])
for n in rows["slide"]["notes"]:
    print("  note: %s" % n)
print("\nslide sj min %.4f vs frozen %.4f (x%.2f)"
      % (rows["slide"]["q"].sj_min, rows["frozen"]["q"].sj_min,
         rows["slide"]["q"].sj_min / rows["frozen"]["q"].sj_min))

sj = {k: per_hex_sj(m) for k, m in meshes.items()}
for label, m in meshes.items():
    writer.to_vtu(m, "02_slide_box_%s.vtu" % label)

mesh = meshes["slide"]
