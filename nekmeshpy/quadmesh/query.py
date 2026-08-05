"""Read-only :class:`QuadMesh <nekmeshpy.quadmesh.quadmesh.QuadMesh>` queries -- the
operations that leave the ladder, returning plain arrays and reports rather than a mesh.
"""

from ._query import (
    boundary_edges,
    boundary_elements,
    boundary_points,
    quality_summary,
    scaled_jacobian,
)

__all__ = [
    "boundary_edges",
    "boundary_elements",
    "boundary_points",
    "quality_summary",
    "scaled_jacobian",
]
