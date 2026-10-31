"""High order and the hex smoother: the native smoother is **linear**.

It moves the corner vertices of an order-1 mesh.  A curved input is smoothed as its
straight-sided shadow (:func:`linearize`): same corners, connectivity, element tags and face
tags, no edge, face or interior nodes.  Rebuilding the curved nodes of the smoothed vertices --
the hex counterpart of :func:`nekmeshpy.quadmesh.smoother.highorder.rebuild_high_order` -- is not
available yet, so the result of smoothing a curved mesh is the linear mesh.  What ``.re2``
exports is exactly that linear mesh in any case (it has no curved format).
"""

from __future__ import annotations

from ...quadmesh.smoother.highorder import linearize as _linearize_quad
from ..hexmesh import HexMesh

__all__ = ["linearize"]


def linearize(mesh: HexMesh) -> HexMesh:
    """The straight-sided order-1 mesh with the same corners, connectivity and tags -- what the
    native smoother works on.  A new mesh; ``mesh`` is untouched (a linear mesh comes back as a
    copy)."""
    return HexMesh(_linearize_quad(mesh.quad_mesh), mesh.hexes, mesh.orient, None, mesh.element_tags)
