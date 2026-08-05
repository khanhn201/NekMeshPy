"""Fixed-arity rung-raising factories: a ``LineMesh`` curve in, a
:class:`QuadMesh <nekmeshpy.quadmesh.quadmesh.QuadMesh>` out.

Unlike the ``region`` / ``surface`` factories these own no *shape model* -- they are
generic over whatever curve they are handed.  ``sweep`` carries one profile along a
curved path by a moving frame, the curved generalization of ``extrude``.
"""

from ._lift import annulus, extrude, from_grid, sweep

__all__ = ["annulus", "extrude", "from_grid", "sweep"]
