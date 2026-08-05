"""Shape factories for :class:`LineMesh <nekmeshpy.linemesh.linemesh.LineMesh>` -- the ones that own a *shape
model* rather than being generic over any input.

A facade over the two private siblings that split on open vs closed, which is a
storage distinction rather than a caller-facing one: ``_open`` holds the open curves
(``line`` / ``arc``) and ``_closed`` the loops (``circle`` / ``rectangle``).  A caller
asking for "the shapes" wants one namespace, so open and closed are merged here while
the modules stay split.

``arclength_fractions`` and ``sweep_fractions`` come along because they answer a
question *about* a shape factory's input contract -- no factory resamples its input, so
deriving the right sampling is an explicit caller step -- and they return a plain array
rather than a mesh, which is what keeps them distinct from the factories proper.
"""

from ._closed import circle, rectangle
from ._open import arc, arclength_fractions, line, sweep_fractions

__all__ = [
    "arc",
    "arclength_fractions",
    "circle",
    "line",
    "rectangle",
    "sweep_fractions",
]
