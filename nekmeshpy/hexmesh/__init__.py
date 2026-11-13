"""3-D hex mesh container (``HexMesh``), and its operations."""

from typing import Any

from .assemble import (
    Seam,
    attach,
    components,
    face_group,
    loft,
    loft_fn,
    loft_spline,
    merge,
    refine,
    remove,
    select,
)
from .hexmesh import HexMesh
from .lift import (
    adapter,
    annulus,
    bridge,
    extrude,
    from_grid,
    sweep,
    sweep_path,
)
from .lower import boundary_mesh
from .morph import blend, mirror, rotate, scale, transform, transform_fn, translate
from .periodic import Periodic, PeriodicPairs, periodic_pairs
from .quality import order_scan
from .query import (
    boundary_elements,
    boundary_face_ids,
    boundary_faces,
    boundary_points,
    bounds,
    centroid,
    classify_points,
    corner_scaled_jacobian,
    corner_summary,
    element_blocks,
    element_volumes,
    face_rows,
    is_conforming,
    is_overlap_free,
    is_watertight,
    linear_order_scan,
    linear_scaled_jacobian,
    quality_summary,
    report,
    scaled_jacobian,
    solver_metrics,
    tag_report,
    tagged_faces,
    topology_report,
    volume,
)
from .shape import autosphere, octant_autosphere, tetra
from .tag import retag_element, retag_face, tag_faces

__all__ = [
    "bounds",
    "centroid",
    "components",
    "element_blocks",
    "element_volumes",
    "mirror",
    "remove",
    "retag_element",
    "retag_face",
    "select",
    "tag",
    "volume",
    "HexMesh",
    "Periodic",
    "PeriodicPairs",
    "Seam",
    "adapter",
    "attach",
    "annulus",
    "assemble",
    "blend",
    "boundary_elements",
    "boundary_face_ids",
    "boundary_faces",
    "boundary_mesh",
    "boundary_points",
    "bridge",
    "classify_points",
    "corner_scaled_jacobian",
    "corner_summary",
    "extrude",
    "face_group",
    "face_rows",
    "from_grid",
    "is_conforming",
    "is_overlap_free",
    "is_watertight",
    "lift",
    "linear_order_scan",
    "solver_metrics",
    "linear_scaled_jacobian",
    "loft",
    "loft_fn",
    "loft_spline",
    "lower",
    "merge",
    "morph",
    "order_scan",
    "periodic",
    "periodic_pairs",
    "quality_summary",
    "query",
    "refine",
    "report",
    "rotate",
    "scale",
    "scaled_jacobian",
    "shape",
    "sweep",
    "sweep_path",
    "tag_faces",
    "tag_report",
    "tagged_faces",
    "autosphere",
    "octant_autosphere",
    "tetra",
    "topology_report",
    "transform",
    "transform_fn",
    "translate",
]


def __getattr__(name: str) -> Any:
    """``hexmesh.smoother`` -- the (experimental) hex smoother lives in ``hexmesh/smoother/`` and is
    loaded on first use, as ``quadmesh.smoother`` is: it builds on the quad smoother, which imports
    ``io``, which imports the rungs, so importing it eagerly here would be circular."""
    if name == "smoother":
        import importlib

        return importlib.import_module(".smoother", __name__)
    raise AttributeError("module %r has no attribute %r" % (__name__, name))
