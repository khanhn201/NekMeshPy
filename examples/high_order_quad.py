"""High-order (order-N) surface: a spectral-element cubed-sphere rendered as its
true sphere.

The motivating high-order case. :meth:`QuadMesh.sphere` with ``order=N`` projects
**every** ``(N+1)**2`` Gauss-Lobatto-Legendre node of each cube face onto the exact
sphere (not just the four corners), so every node lies at the true radius. The corner
connectivity stays linear -- ``re2`` export is unaffected -- while ``vtu`` export emits
``VTK_LAGRANGE_QUADRILATERAL`` cells that a viewer renders as a smooth sphere.

    PYTHONPATH=. python examples/high_order_quad.py

Produces ``high_order_quad.vtu`` (high-order Lagrange quadrilateral cells).
"""

import logging

import numpy as np

from nekmeshpy import export, quadmesh

logging.basicConfig(level=logging.INFO, format="%(message)s")
_log = logging.getLogger("nekmeshpy")

# -- parameters --------------------------------------------------------------
RADIUS = 2.0                 # sphere radius
N_CELL = 4                   # cells per cube-face axis (6 * N_CELL**2 quads)
ORDER = 2                    # polynomial order: (ORDER+1)**2 GLL nodes per quad
OUT_NAME = "high_order_quad"

# -- build the high-order surface --------------------------------------------
mesh = quadmesh.sphere(RADIUS, N_CELL, order=ORDER)

# every curved node sits on the true sphere, to machine precision.  The entity
# B-rep numbers each node once: corner ``points``, the shared ``edge_nodes``, and
# each quad's private ``interior``.
radii = np.linalg.norm(np.vstack([
    mesh.points,
    mesh.edge_nodes.reshape(-1, 3),
    mesh.interior.reshape(-1, 3)]), axis=1)
_log.info("order-%d sphere: %d quads, %d nodes/quad",
          mesh.order, mesh.quads.shape[0], (ORDER + 1) ** 2)
_log.info("node radius: min=%.15f max=%.15f (target %.1f)",
          radii.min(), radii.max(), RADIUS)

# -- export ------------------------------------------------------------------
export.quad_to_vtu(mesh, OUT_NAME + ".vtu")   # XML: high-order Lagrange quad cells
_log.info("wrote %s.vtu", OUT_NAME)
