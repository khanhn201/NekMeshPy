"""Variable-arity :class:`LineMesh <nekmeshpy.linemesh.linemesh.LineMesh>` factories --
the only operations at this rung that manufacture a numbering.

``loft`` numbers the chain it authors, ``loft_fn`` is ``loft`` with the points evaluated
from a parametrization rather than handed in, and ``merge`` builds the weld's
``remap`` / ``survivors`` / ``point_id`` tables.  Every fixed-arity operation either
reuses an existing numbering or delegates here.

A facade over ``_assemble``, which also holds the private lattice/weld internals the
other rungs import.  Only the public factories are re-exported.
"""

from ._assemble import loft, loft_fn, merge

__all__ = ["loft", "loft_fn", "merge"]
