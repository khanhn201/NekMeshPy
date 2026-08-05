"""Variable-arity :class:`QuadMesh <nekmeshpy.quadmesh.quadmesh.QuadMesh>` factories --
the only operations at this rung that manufacture a numbering.

``loft`` lifts a stack of profiles into a surface, ``loft_fn`` is ``loft`` with the
profiles **evaluated** from a caller-supplied ``f(t) -> LineMesh`` at every node level
of the sweep lattice rather than handed in -- which is what makes a swept curved
surface exact above order 1 -- and ``merge`` welds sections into one index space.
"""

from ._assemble import loft, loft_fn, merge

__all__ = ["loft", "loft_fn", "merge"]
