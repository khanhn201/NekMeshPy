"""Shape factories for :class:`HexMesh <nekmeshpy.hexmesh.hexmesh.HexMesh>` -- the ones
owning a *shape model* rather than being generic over any input.

Only ``tetra`` so far: the degenerate-corner block set that fills a tetrahedron with
hexes.
"""

from ._open import tetra

__all__ = ["tetra"]
