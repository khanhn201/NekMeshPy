"""1-D mesh container (:class:`LineMesh`) and its shape factories.

``linemesh.py`` is a pure container; the parametric shape factories are free
functions in the sibling ``_closed.py`` (``circle`` / ``rectangle``) and ``_open.py``
(``line``) modules, each exposing a ``FACTORIES`` registry. This package binds them
onto the class below so callers write ``LineMesh.circle(...)`` -- adding a shape only
touches the sibling module (add the function + one ``FACTORIES`` entry), never the
container or this file.
"""

from ._closed import FACTORIES as _CLOSED_FACTORIES
from ._open import FACTORIES as _OPEN_FACTORIES
from .linemesh import LineMesh

# The factories are plain free functions (no ``cls``); bind as ``staticmethod`` so
# ``LineMesh.circle(radius, n)`` passes no implicit first argument.
for _name, _fn in {**_CLOSED_FACTORIES, **_OPEN_FACTORIES}.items():
    setattr(LineMesh, _name, staticmethod(_fn))

__all__ = ["LineMesh"]
