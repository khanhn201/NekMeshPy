"""Read-only :class:`HexMesh <nekmeshpy.hexmesh.hexmesh.HexMesh>` queries -- the
operations that leave the ladder.

The topology group (``is_watertight`` / ``is_conforming`` / ``topology_report`` /
``classify_points``) answers whether the block set is a valid closed domain; ``weld``
hands back the flat shared-point triple the smoothers and exporters want in place of
the ladder; ``report`` formats the quality and topology summaries together.
"""

from ._query import (
    boundary_elements,
    boundary_faces,
    boundary_points,
    classify_points,
    is_conforming,
    is_watertight,
    quality_summary,
    report,
    scaled_jacobian,
    topology_report,
    weld,
)

__all__ = [
    "boundary_elements",
    "boundary_faces",
    "boundary_points",
    "classify_points",
    "is_conforming",
    "is_watertight",
    "quality_summary",
    "report",
    "scaled_jacobian",
    "topology_report",
    "weld",
]
