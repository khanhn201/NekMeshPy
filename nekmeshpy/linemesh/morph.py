"""Rung-preserving :class:`LineMesh <nekmeshpy.linemesh.linemesh.LineMesh>` factories.

Only ``blend`` lives here: it is binary, so it takes no single mesh to bind to and is
reached through this namespace.  The *unary* placements from the same sibling
(``translate`` / ``rotate`` / ``scale`` / ``transform`` / ``reverse``) do take the mesh
they act on, so they are bound onto the class instead and spelled ``lm.translate(v)``.
"""

from ._morph import blend

__all__ = ["blend"]
