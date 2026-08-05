"""Rung-preserving :class:`QuadMesh <nekmeshpy.quadmesh.quadmesh.QuadMesh>` operations.

``blend`` morphs between two index-paired sections; the rest are unary placements that
change only coordinates -- connectivity and tags ride through verbatim, so the input's
numbering *is* the output's.
"""

from ._morph import blend, rotate, scale, transform, translate

__all__ = ["blend", "rotate", "scale", "transform", "translate"]
