"""Fixed-arity rung-raising factories: a ``QuadMesh`` section in, a
:class:`HexMesh <nekmeshpy.hexmesh.hexmesh.HexMesh>` out.

These own no *shape model* -- they are generic over whatever section they are handed.
``sweep`` carries one section along a curved path by a moving frame, the curved
generalization of ``extrude``.
"""

from ._lift import annulus, extrude, from_grid, sweep

__all__ = ["annulus", "extrude", "from_grid", "sweep"]
