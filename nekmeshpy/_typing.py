"""Shared numpy array type aliases used across the package."""

from __future__ import annotations

from typing import Literal, Union

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
BoolArray = NDArray[np.bool_]
StrArray = NDArray[np.str_]         # per-element string labels (boundary names)

# shape-documentation aliases (not shape-enforced by mypy)
Point = NDArray[np.float64]        # a single (3,) location
Vec3 = NDArray[np.float64]         # a single (3,) direction/vector
PointArray = NDArray[np.float64]   # point coordinates: any leading shape + trailing 3

#: The section-smoothing strategies shipped in ``quadmesh.smoothing``'s
#: ``SECTION_METHODS`` registry.  The registry is **open** -- a caller can add its own
#: with ``@register_section_smoothing("name")`` -- so the alias keeps a bare ``str`` arm
#: alongside the ``Literal``: mypy and an IDE surface the built-ins by name without the
#: annotation closing the registry to third-party strategies.  It lives here rather than
#: beside the registry because every rung that forwards a ``smoothing_method=`` through
#: to it (the region fills, both ``annulus``) needs the same spelling.
SmoothingMethod = Union[
    Literal["conduction", "winslow", "bilinear", "tfi", "none", ""], str]

__all__ = ["FloatArray", "IntArray", "BoolArray", "StrArray",
           "Point", "Vec3", "PointArray", "SmoothingMethod"]
