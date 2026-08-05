"""Read-only :class:`LineMesh <nekmeshpy.linemesh.linemesh.LineMesh>` queries -- the
operations that leave the ladder.

They take the mesh and return plain arrays, never another mesh.
"""

from ._query import boundary_elements, boundary_points

__all__ = ["boundary_elements", "boundary_points"]
