"""2-D quad mesh container (``QuadMesh``), its section factories, and smoothing.

``quadmesh.py`` is a pure container; the region-fill and closed-surface factories are
free functions in the sibling ``_open.py`` (``structured`` / ``rectangle`` / ``ogrid``
/ ``half_ogrid`` / ``annulus``) and ``_closed.py`` (``box`` / ``sphere``) modules, each
exposing a ``FACTORIES`` registry. This package binds them onto the class below so
callers write ``QuadMesh.ogrid(...)`` -- adding a factory only touches the sibling
module (add the function + one ``FACTORIES`` entry), never the container or this file.
"""

from ._closed import FACTORIES as _CLOSED_FACTORIES
from ._open import FACTORIES as _OPEN_FACTORIES
from .quadmesh import NO_BOUNDARY, QuadMesh

# The factories are plain free functions (no ``cls``); bind as ``staticmethod`` so
# ``QuadMesh.ogrid(boundary, ...)`` passes no implicit first argument.
for _name, _fn in {**_CLOSED_FACTORIES, **_OPEN_FACTORIES}.items():
    setattr(QuadMesh, _name, staticmethod(_fn))

__all__ = ["NO_BOUNDARY", "QuadMesh"]
