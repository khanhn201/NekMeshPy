"""Case study: the U-tube hexes, as built and with the smoothed wall carried in (examples/u_tube_box.py).

``cs_u_tube.py`` smooths the tube wall (the face tag ``interface``) as a free-standing surface.  This script asks
whether that smoothed wall can go back into the hex mesh, by the construction of ``u_tube_box.py`` itself: the wall
is the one surface between two single-layer shells, a solid shell lofted from the stair-step cavity out to it and a
fluid shell lofted from it in to the shrunk core.  A one-layer loft has no interior stations, so lofting again from
the smoothed wall is the same as putting the smoothed wall's vertices in place of the old wall's: every other vertex
(the stair-step side, the box, the core) is the example's own and stays.  Done at order 1, the corner geometry
``.re2`` holds (the example is order 2; the order-1 mesh has its corners, none of the curvature).

**The inner side does not move, so the layers shear.**  The wall moves by up to several local edges while the
stair-step side of the solid shell and the core side of the fluid shell stay; the hexes in between are squeezed.  The
readings below are the measurement: ``corner`` is the 8-corner trilinear hex, ``fine`` the same map re-read on a
``SAMPLE_ORDER``-order lattice (the solver's), ``valid`` says whether the carried mesh is usable.  Whatever it says,
nothing here is propagated into the volume by the smoother; this is a re-placement of one set of vertices.

    PYTHONPATH=. python examples/smoother/cs_u_tube_hex.py

Writes nothing; ``variants`` (``as built`` / ``smoothed wall carried in``), ``readings``, ``sj_fine``, ``where``, ``planes`` and
``region`` are what ``docs/_ext/gen_smoother_hex_assets.py`` reads.  ``mesh`` is the as-built order-1 ``HexMesh``.
"""

import logging
import os
import runpy

import numpy as np
from scipy.spatial import cKDTree

from nekmeshpy import HexMesh, QuadMesh, hexmesh
from nekmeshpy.linemesh import LineMesh
from nekmeshpy.pointmesh import PointMesh
from nekmeshpy.quadmesh.smoother import constrain as cn

SAMPLE_ORDER = 7              # the solver's order: finer lattice the trilinear map is re-read on
FLUID = "fluid"
PLANES = [("along the U", (0.0, 0.0, 0.3), (0.0, 0.0, 1.0)),            # the U lies in z = 0
          ("across both legs", (7.3, 0.0, 0.0), (1.0, 0.0, 0.0))]       # the legs run along x

ns = runpy.run_path(os.path.join(os.path.dirname(os.path.abspath(__file__)), "cs_u_tube.py"), run_name="__main__")
logging.getLogger().setLevel(logging.ERROR)
hexes, TAG = ns["hexes"], ns["TAG"]
smoothed = ns["runs"]["asis"]["out"]                  # the smoothed wall, free-standing (order 1)
unsmoothed = ns["runs"]["asis"]["before"]


def linear(h):
    """The order-1 copy of ``h``: same corners, connectivity and tags, straight sides."""
    q, lm = h.quad_mesh, h.quad_mesh.line_mesh
    lines = LineMesh(PointMesh(np.array(lm.points, copy=True), lm.point_mesh.element_tags), lm.lines, None,
                     lm.element_tags)
    return HexMesh(QuadMesh(lines, q.quads, q.orient, None, q.element_tags), h.hexes, h.orient, None, h.element_tags)


def stats(sj):
    return dict(min=float(sj.min()), mean=float(sj.mean()), q05=float(np.quantile(sj, 0.05)), n_inv=int((sj <= 0).sum()))


def reading(m, sj_fine):
    names, counts = m.face_tags.unique(return_counts=True)
    return dict(corner=stats(hexmesh.corner_scaled_jacobian(m)), fine=stats(sj_fine),
                n=m.n_hexes, conforming=bool(hexmesh.is_watertight(m) and hexmesh.is_conforming(m)),
                faces={str(k): int(v) for k, v in zip(names, counts)})


built = linear(hexes)
carried = linear(hexes)
tree = cKDTree(built.points)
dist, wall_ids = tree.query(unsmoothed.points)                # the wall's vertices in the hex mesh's own numbering
assert dist.max() < 1e-9 and np.unique(wall_ids).size == unsmoothed.n_points
carried.points[wall_ids] = smoothed.points                    # the wall carried in; nothing else moves
variants = {"as built": built, "smoothed wall carried in": carried}
sj_fine = {k: hexmesh.scaled_jacobian(v, order=SAMPLE_ORDER) for k, v in variants.items()}      # per hex, the finer lattice
readings = {k: reading(v, sj_fine[k]) for k, v in variants.items()}
valid = (readings["smoothed wall carried in"]["corner"]["n_inv"] == 0 and readings["smoothed wall carried in"]["fine"]["n_inv"] == 0
         and readings["smoothed wall carried in"]["faces"] == readings["as built"]["faces"]
         and readings["smoothed wall carried in"]["conforming"])

# where the damage is: the hexes that touch the wall are the two shells (one solid, one fluid)
corners = built.corners
touch = np.isin(corners, wall_ids).any(axis=1)
sj = {k: hexmesh.corner_scaled_jacobian(v) for k, v in variants.items()}
region = np.asarray(built.element_tags.to_dense())
bad = sj["smoothed wall carried in"] <= 0
where = dict(n_touch=int(touch.sum()), n_bad=int(bad.sum()), bad_touch=int((bad & touch).sum()),
             bad_fluid=int((bad & (region == FLUID)).sum()), bad_solid=int((bad & (region != FLUID)).sum()),
             touch_min_built=float(sj["as built"][touch].min()), touch_min_carried=float(sj["smoothed wall carried in"][touch].min()),
             off_min_built=float(sj["as built"][~touch].min()), off_min_carried=float(sj["smoothed wall carried in"][~touch].min()))
moved = np.linalg.norm(smoothed.points - unsmoothed.points, axis=1) / cn.vertex_length(unsmoothed.points, unsmoothed.corners)
where.update(mean_move=float(moved.mean()), max_move=float(moved.max()))
planes = PLANES
mesh = built
for k, r in readings.items():
    print("%-26s %6d hexes | corner min %7.4f mean %.4f q05 %.4f inverted %d | order-%d min %7.4f inverted %d | "
          "conforming+watertight %s" % (k, r["n"], r["corner"]["min"], r["corner"]["mean"], r["corner"]["q05"],
                                        r["corner"]["n_inv"], SAMPLE_ORDER, r["fine"]["min"], r["fine"]["n_inv"], r["conforming"]))
print("smoothed wall carried in: %d inverted hexes (%d fluid, %d solid), %d of them touch the wall; the %d hexes that touch "
      "the wall read min %.4f -> %.4f, the rest min %.4f -> %.4f; valid: %s"
      % (where["n_bad"], where["bad_fluid"], where["bad_solid"], where["bad_touch"], where["n_touch"], where["touch_min_built"],
         where["touch_min_carried"], where["off_min_built"], where["off_min_carried"], valid))
