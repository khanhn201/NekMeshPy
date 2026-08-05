"""Variable-arity :class:`HexMesh <nekmeshpy.hexmesh.hexmesh.HexMesh>` factories -- the
only operations at this rung that manufacture a numbering.

``loft`` lifts a stack of ``QuadMesh`` sections into a block, ``loft_fn`` is ``loft``
with the sections **evaluated** from a caller-supplied parametrization at every node
level of the sweep lattice rather than handed in, and ``merge`` welds blocks into one
index space.
"""

from ._assemble import loft, loft_fn, merge

__all__ = ["loft", "loft_fn", "merge"]
