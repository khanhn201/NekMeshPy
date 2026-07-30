"""High-order (order-N) curve: a spectral-element circle rendered as its true arc.

The motivating high-order case one dimension down. :meth:`LineMesh.circle` with
``order=N`` places ``N+1`` Gauss-Lobatto-Legendre nodes per arc element **on the
exact circle** (not on the straight chord between corners), so every node lies at
the true radius. The corner connectivity stays linear -- ``re2`` export is
unaffected -- while ``vtu`` export emits ``VTK_LAGRANGE_CURVE`` cells that a viewer
renders as smooth arcs.

    PYTHONPATH=. python examples/high_order_curve.py

Produces ``high_order_curve.vtu`` (high-order Lagrange curve cells).
"""

import logging

import numpy as np

from nekmeshpy import LineMesh, export

logging.basicConfig(level=logging.INFO, format="%(message)s")
_log = logging.getLogger("nekmeshpy")

# -- parameters --------------------------------------------------------------
RADIUS = 2.0                 # circle radius
N_ELEM = 8                   # arc elements (corner points) around the loop
ORDER = 2                    # polynomial order: ORDER+1 GLL nodes per element
OUT_NAME = "high_order_curve"

# -- build the high-order loop -----------------------------------------------
mesh = LineMesh.circle(RADIUS, N_ELEM, order=ORDER)

# every curved node sits on the true circle, to machine precision.  A LineMesh's
# high-order state is just its corner ``points`` plus each line's private
# ``interior`` block -- together, every node of the mesh, each numbered once.
radii = np.linalg.norm(
    np.vstack([mesh.points, mesh.interior.reshape(-1, 3)]), axis=1)
_log.info("order-%d circle: %d elements, %d nodes/element",
          mesh.order, mesh.lines.shape[0], ORDER + 1)
_log.info("node radius: min=%.15f max=%.15f (target %.1f)",
          radii.min(), radii.max(), RADIUS)

# -- export ------------------------------------------------------------------
export.line_to_vtu(mesh, OUT_NAME + ".vtu")   # XML: high-order Lagrange curve cells
_log.info("wrote %s.vtu", OUT_NAME)
