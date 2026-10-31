"""Quad-mesh smoothing.

This directory is only for code organisation: everything public is re-exported here, so callers write

    from nekmeshpy.quadmesh.smoother import SmoothSession, OnSurface, OnCurve, laplace, optimize

and never need to know which file a name lives in.  The modules keep their names and can also be used as namespaces
(``smoother.smooth``, ``smoother.constrain``, ...):

* ``constrain``  -- what each vertex may do: targets, constraints, the snap, the on-target check
* ``smooth``     -- the shape metric, objective and gradient, Laplace, the optimizer, schedules, quality readings
* ``session``    -- ``SmoothSession``: reviewed rounds, the round report, undo, the gap gate
* ``highorder``  -- rebuilding the curved nodes after the linear smoothing
* ``frames``     -- the positions after every stage as a small JSON, for the docs playback viewer

The ``nekmeshpy.quadmesh`` package keeps ``smooth``, ``constrain``, ``session`` and ``highorder`` reachable by attribute
(``from nekmeshpy.quadmesh import smooth``), loaded on first use.
"""

from . import constrain, frames, highorder, session, smooth
from .constrain import (
    FREE,
    FROZEN,
    ON_CURVE,
    ON_SURFACE,
    TIED,
    Curve,
    Frozen,
    KeepLayer,
    NormalReport,
    OnCurve,
    OnSurface,
    OnTargetReport,
    Resolved,
    SlideColumns,
    Surface,
    circle,
    claimed_surfaces,
    constrain_direction,
    grid_columns,
    layer_rings,
    line,
    on_target_report,
    plane,
    resolve,
    snap,
    sphere,
    surface_from_trimesh,
    surface_normals,
    vertex_length,
)
from .frames import frames_of, write_frames
from .highorder import (
    HighOrderReport,
    linearize,
    rebuild_high_order,
)
from .session import (
    ConnectivityError,
    Proposal,
    RoundReport,
    SmoothSession,
    adjust_schedule,
)
from .smooth import (
    DELTA,
    EPS,
    KAPPA,
    CornerMetrics,
    Quality,
    ShapeStats,
    StageLog,
    corner_metrics,
    corner_vectors,
    failing_elements,
    gradient,
    laplace,
    objective,
    optimize,
    quad_neighbours,
    quality,
    reference_normals,
    run_schedule,
    shape_stats,
    size_ratios,
    suggest_schedule,
    untangle_schedule,
    user_schedule,
    with_points,
)

__all__ = [
    "constrain",
    "frames",
    "frames_of",
    "write_frames",
    "highorder",
    "session",
    "smooth",
    "ConnectivityError",
    "CornerMetrics",
    "Curve",
    "DELTA",
    "EPS",
    "FREE",
    "FROZEN",
    "Frozen",
    "HighOrderReport",
    "KAPPA",
    "KeepLayer",
    "ON_CURVE",
    "ON_SURFACE",
    "OnCurve",
    "OnSurface",
    "OnTargetReport",
    "Proposal",
    "Quality",
    "Resolved",
    "RoundReport",
    "ShapeStats",
    "SlideColumns",
    "SmoothSession",
    "StageLog",
    "Surface",
    "TIED",
    "adjust_schedule",
    "circle",
    "constrain_direction",
    "NormalReport",
    "claimed_surfaces",
    "surface_from_trimesh",
    "surface_normals",
    "corner_metrics",
    "corner_vectors",
    "failing_elements",
    "gradient",
    "grid_columns",
    "laplace",
    "layer_rings",
    "line",
    "linearize",
    "objective",
    "on_target_report",
    "optimize",
    "plane",
    "quad_neighbours",
    "quality",
    "rebuild_high_order",
    "reference_normals",
    "resolve",
    "run_schedule",
    "shape_stats",
    "size_ratios",
    "snap",
    "sphere",
    "suggest_schedule",
    "untangle_schedule",
    "user_schedule",
    "vertex_length",
    "with_points",
]
