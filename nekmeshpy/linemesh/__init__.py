"""1-D mesh container (:class:`LineMesh`), its operations, and its shape factories."""

from . import assemble, morph, query, shape
from .assemble import loft, loft_fn, merge
from .linemesh import LineMesh
from .morph import blend, reverse, rotate, scale, transform, translate
from .query import boundary_elements, boundary_points
from .shape import (
    arc,
    arclength_fractions,
    circle,
    line,
    on_surface,
    path_fractions,
    rectangle,
    sweep_fractions,
)

__all__ = [
    "LineMesh",
    "arc",
    "arclength_fractions",
    "assemble",
    "blend",
    "boundary_elements",
    "boundary_points",
    "circle",
    "line",
    "loft",
    "loft_fn",
    "merge",
    "morph",
    "on_surface",
    "path_fractions",
    "query",
    "rectangle",
    "reverse",
    "rotate",
    "scale",
    "shape",
    "sweep_fractions",
    "transform",
    "translate",
]
