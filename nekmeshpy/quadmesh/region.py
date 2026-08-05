"""Region fills: the open-region :class:`QuadMesh
<nekmeshpy.quadmesh.quadmesh.QuadMesh>` factories, which own a *shape model* rather
than being generic over any input.

Each meshes its boundary **exactly** -- no factory resamples what it is given -- and
carries the input walls' curvature into the interior.  ``spine_fractions``,
``quadrant_seam_fractions`` and ``quadrant_core`` come along because they answer a
question *about* one of those input contracts: they return a plain array rather than a
mesh, and exist so a caller can derive the sampling a factory demands (or, for
``quadrant_core``, land a neighbouring block on the very same points) instead of
reproducing the formula.
"""

from ._open import (
    half_ogrid,
    ogrid,
    quadrant_core,
    quadrant_disc,
    quadrant_ogrid,
    quadrant_seam_fractions,
    rectangle,
    spine_fractions,
    spined_ogrid,
    structured,
)

__all__ = [
    "half_ogrid",
    "ogrid",
    "quadrant_core",
    "quadrant_disc",
    "quadrant_ogrid",
    "quadrant_seam_fractions",
    "rectangle",
    "spine_fractions",
    "spined_ogrid",
    "structured",
]
