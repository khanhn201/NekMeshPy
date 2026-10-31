"""High-order meshes: smooth the vertices linearly, then rebuild the curved nodes on the wall.

The smoother moves *vertices only*.  A curved mesh handed to ``SmoothSession`` is smoothed through its
straight-sided shadow, and ``session.rebuild_high_order()`` gives the order-N mesh back for the accepted
vertices: every node keeps its offset from its straight-sided element (it moves with its element), then
the nodes on the wall are projected onto the circle.

The case is a disk, order 4, whose wall is an exact circle: an O-grid with the wall vertices slipped
along the circle and the interior jittered.  Printed, all computed here:

* the wall's deviation from the circle **between** its nodes -- the stored wall edge is a degree-4
  polynomial, read on an order-24 lattice -- for the exact input, after the carry alone, and after the
  projection.  (The nodes themselves lie on the circle to rounding after the projection; the polynomial
  through them does not, a floor set by the order and the wall's segment count.)
* the scaled Jacobian of the rebuilt mesh *sampled on finer lattices* (``HighOrderReport``), and area.
* what is **not** guaranteed: the rebuild is not a smoother -- nothing makes a curved element valid,
  the sampled reading says what it finds, and a value at the mesh's own nodes certifies nothing between them.

    PYTHONPATH=. python examples/smoother/05_high_order.py

Produces ``05_{before,after}.vtu`` (the linear vertices, with ``sj_min``), ``05_curved.vtu`` (the order-4
Lagrange cells of the rebuilt mesh) and ``05_frames.json`` (playback of the linear smoothing).  ``mesh``
is the smoothed *linear* mesh; the rebuilt one is ``curved``.
"""

import time

import numpy as np

from nekmeshpy import linemesh, quadmesh
from nekmeshpy.core import interp
from nekmeshpy.io.writer import quad_to_vtu
from nekmeshpy.quadmesh.smoother import SmoothSession, frames, highorder, smooth
from nekmeshpy.quadmesh.smoother import constrain as cn

# -- parameters --------------------------------------------------------------
RADIUS = 1.0
WALL_N = 24                  # wall segments
ORDER = 4                    # polynomial order of the curved mesh
JITTER = 0.1                 # interior jitter, fraction of the wall spacing
SLIP = 0.25                  # wall vertex slip along the circle, fraction of the wall spacing
CHECK_ORDERS = (4, 6, 8, 12)  # lattices the scaled Jacobian is read on
BETWEEN_ORDER = 24           # lattice for the between-node deviation of the wall
SEED = 3
SCHEDULE = smooth.user_schedule(outer=5)

t_start = time.perf_counter()
circle = cn.circle([0, 0, 0], RADIUS)

# -- the exact curved disk, then the same disk with perturbed vertices -----
exact = quadmesh.ogrid(linemesh.circle(RADIUS, WALL_N, element_tag="wall", order=ORDER), 6, 4, wall_tag="wall")
lin0 = highorder.linearize(exact)
rng = np.random.default_rng(SEED)
P = np.array(lin0.points, dtype=float)
wall_v = np.flatnonzero(np.isclose(np.hypot(P[:, 0], P[:, 1]), RADIUS, atol=1e-9))
inner = np.setdiff1d(np.arange(lin0.n_points), wall_v)
h_wall = 2 * np.pi * RADIUS / WALL_N
P[inner, :2] += rng.uniform(-JITTER, JITTER, (inner.size, 2)) * h_wall
th = np.arctan2(P[wall_v, 1], P[wall_v, 0]) + rng.uniform(-SLIP, SLIP, wall_v.size) * h_wall
P[wall_v, 0], P[wall_v, 1] = RADIUS * np.cos(th), RADIUS * np.sin(th)
perturbed, _ = highorder.rebuild_high_order(exact, P, None, project=False)   # the curved input: nodes carried, not projected


def wall_between_node_deviation(m):
    """Largest |distance to the circle| of the wall polynomials read on the order-24 lattice."""
    wall = m.edge_tags.flatnonzero("wall")
    blocks = linemesh.element_blocks(m.line_mesh)[wall]
    dense = interp.resample_block(blocks, m.order, BETWEEN_ORDER, 1)
    return float(np.abs(np.hypot(dense[..., 0], dense[..., 1]) - RADIUS).max())


def wall_node_deviation(m):
    wall = m.edge_tags.flatnonzero("wall")
    blocks = linemesh.element_blocks(m.line_mesh)[wall]
    return float(np.abs(np.hypot(blocks[..., 0], blocks[..., 1]) - RADIUS).max())


def per_quad_sj(m):
    return smooth.corner_metrics(m.points, m.corners, nrm).sj.min(axis=1)


# -- linear smoothing through the session --------------------------------------
session = SmoothSession(perturbed, [cn.OnCurve("wall", circle)])
nrm = session.normals
before = session.current
q0 = session.quality()
proposal = session.propose(SCHEDULE)
session.accept(proposal)
after = session.current
q1 = session.quality()
print("disk of %d quads, order %d wall, %d wall vertices slipped by up to %.2f segment, interior jitter %.2f"
      % (before.n_quads, perturbed.order, wall_v.size, SLIP, JITTER))
print("linear smoothing: inverted %d -> %d | scaled J min %.4f -> %.4f | F/quad %.4f -> %.4f"
      % (q0.n_inverted, q1.n_inverted, q0.sj_min, q1.sj_min, q0.objective_mean, q1.objective_mean))

# -- the rebuild ---------------------------------------------------------------
curved, rep = session.rebuild_high_order()
carried, rep_c = highorder.rebuild_high_order(perturbed, after.points, None)       # the same carry, no projection
print("\nwall of the stored order-%d mesh against the circle (r = %g):" % (ORDER, RADIUS))
print("  %-34s nodes %.2e | between nodes (order-%d lattice) %.2e"
      % ("exact input", wall_node_deviation(exact), BETWEEN_ORDER, wall_between_node_deviation(exact)))
print("  %-34s nodes %.2e | between nodes %.2e" % ("perturbed vertices, carried (input)",
                                                   wall_node_deviation(perturbed), wall_between_node_deviation(perturbed)))
print("  %-34s nodes %.2e | between nodes %.2e" % ("smoothed vertices, carried, no projection",
                                                   wall_node_deviation(carried), wall_between_node_deviation(carried)))
print("  %-34s nodes %.2e | between nodes %.2e" % ("smoothed vertices, projected",
                                                   wall_node_deviation(curved), wall_between_node_deviation(curved)))
print("  the projection takes the between-node deviation %.1e -> %.1e: not zero, because the degree-%d polynomial through "
      "nodes that are on the circle is not the circle (the exact input has a floor of the same kind, %.1e); report: nodes %.1e -> %.1e"
      % (wall_between_node_deviation(carried), wall_between_node_deviation(curved), ORDER,
         wall_between_node_deviation(exact), rep.max_distance_before, rep.max_distance_after))
print("report: %s" % str(rep).splitlines()[0])
print("  corners of the rebuilt mesh are exactly the smoothed vertices: %s | conforming: %s | tags kept: %s"
      % (np.array_equal(curved.points, after.points), quadmesh.is_conforming(curved),
         list(curved.edge_tags.flatnonzero("wall")) == list(perturbed.edge_tags.flatnonzero("wall"))))
print("  area: exact %.6f | linear smoothed %.6f (%+.2e) | curved rebuilt %+.2e"
      % (np.pi * RADIUS ** 2, quadmesh.area(after), quadmesh.area(after) - np.pi * RADIUS ** 2,
         quadmesh.area(curved, high_order=True) - np.pi * RADIUS ** 2))

print("\nscaled Jacobian of the rebuilt mesh, sampled on finer lattices (min over quads):")
for o in CHECK_ORDERS:
    sj = quadmesh.scaled_jacobian(curved, order=o)
    print("  order %2d: min %.4f, %d inverted" % (o, sj.min(), int((sj <= 0).sum())))
print("  HighOrderReport (order %d): input %.4f, carried %.4f, rebuilt %.4f"
      % (rep.check_order, rep.sj_min_input, rep.sj_min_carried, rep.sj_min_rebuilt))

sj_orders = [float(quadmesh.scaled_jacobian(curved, order=o).min()) for o in CHECK_ORDERS]
print("\nnot guaranteed: the rebuild only carries and projects, it does not smooth or repair the curved elements.  A "
      "reading at the mesh's own %d nodes per edge is exact there and silent between them, so the report samples at "
      "order %d and the table above reads the same map at orders %s: min %.4f .. %.4f (spread %.1e).  Here the wall "
      "curvature is mild against the element size and the readings agree; a coarser wall or a thinner layer next to it "
      "would be the case to check at the solver's own order."
      % (ORDER + 1, rep.check_order, list(CHECK_ORDERS), min(sj_orders), max(sj_orders), max(sj_orders) - min(sj_orders)))
print("(%.1f s)" % (time.perf_counter() - t_start))

# -- outputs -------------------------------------------------------------------
quad_to_vtu(before, "05_before.vtu", cell_data={"sj_min": per_quad_sj(before)})
quad_to_vtu(after, "05_after.vtu", point_data={"displacement": after.points - before.points},
            cell_data={"sj_min": per_quad_sj(after)})
quad_to_vtu(curved, "05_curved.vtu", cell_data={"sj_min_order_%d" % rep.check_order:
                                                quadmesh.scaled_jacobian(curved, order=rep.check_order)})
_, logs = smooth.run_schedule(before, cn.resolve(before, [cn.OnCurve("wall", circle)]), SCHEDULE, normals=nrm, record=True)
frames.write_frames("05_frames.json", before, logs, nrm,
                    title="05 high order: linear smoothing of a disk with an order-4 wall", decimals=5)

mesh = after
