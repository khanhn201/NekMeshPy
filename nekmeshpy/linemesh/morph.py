"""Rung-preserving :class:`LineMesh <nekmeshpy.linemesh.linemesh.LineMesh>` factories.

``blend`` is binary -- it morphs between two index-paired profiles.  The rest are
unary placements: ``translate`` / ``rotate`` / ``scale`` / ``transform`` place a
finished curve, and ``reverse`` flips its traversal direction.
"""

from ._morph import blend, reverse, rotate, scale, transform, translate

__all__ = ["blend", "reverse", "rotate", "scale", "transform", "translate"]
