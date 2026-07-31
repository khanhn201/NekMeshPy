"""2-D quad mesh container (``QuadMesh``), its operations, and smoothing.

``quadmesh.py`` is a pure container -- storage, validation and the derived views.
Everything that *acts* on a finished section is a free function in a sibling module,
split by **arity** and by **rung delta** (how far up or down the line -> quad -> hex
ladder the operation moves):

============== ======== ===== ===============================================
module         arity    delta contents
============== ======== ===== ===============================================
``_assemble``  n-ary    +1/0  ``loft``, ``merge`` -- build a new index space
``_lift``      fixed    +1    ``extrude``/``annulus``/``from_grid`` -> ``loft``
``_morph``     fixed     0    ``blend`` + ``translate``/``rotate``/``scale``
``_query``     fixed     exit read-only queries returning plain arrays
``_open``      fixed    +1    region fills (``structured`` / ``ogrid`` / ...)
``_closed``    fixed    +1    closed surfaces (``box`` / ``sphere`` / ...)
============== ======== ===== ===============================================

``_open`` / ``_closed`` are also rung-raising, but they own a *shape* model rather than
being generic over any input, which is what keeps them separate from ``_lift``.  Each
module ends in a registry -- ``FACTORIES`` for the ``staticmethod``-bound combinators,
``METHODS`` for the instance-method-bound queries -- and this package binds them onto
the class below, so callers write ``QuadMesh.ogrid(...)`` / ``qm.boundary_edges()``
while adding an operation touches only the sibling module (the function plus one
registry entry), never the container or this file.  The shared ``_apply_smoothing`` /
``_check_boundary`` / ``_elevate`` factory internals live in ``_helpers.py``.
"""

from ._assemble import FACTORIES as _ASSEMBLE_FACTORIES
from ._closed import FACTORIES as _CLOSED_FACTORIES
from ._lift import FACTORIES as _LIFT_FACTORIES
from ._morph import FACTORIES as _MORPH_FACTORIES
from ._morph import METHODS as _MORPH_METHODS
from ._open import FACTORIES as _OPEN_FACTORIES
from ._query import METHODS as _QUERY_METHODS
from .quadmesh import NO_BOUNDARY, QuadMesh

# The combinators are plain free functions (no ``cls``); bind as ``staticmethod`` so
# ``QuadMesh.ogrid(boundary, ...)`` passes no implicit first argument.
for _name, _fn in {**_CLOSED_FACTORIES, **_OPEN_FACTORIES, **_ASSEMBLE_FACTORIES,
                   **_LIFT_FACTORIES, **_MORPH_FACTORIES}.items():
    setattr(QuadMesh, _name, staticmethod(_fn))

# The queries and the unary placements take the mesh (or bare connectivity) first.
# ``_boundary_mask`` reads only connectivity, so it stays a ``staticmethod``; the rest
# become instance methods.
for _name, _fn in {**_QUERY_METHODS, **_MORPH_METHODS}.items():
    setattr(QuadMesh, _name,
            staticmethod(_fn) if _name == "_boundary_mask" else _fn)

__all__ = ["NO_BOUNDARY", "QuadMesh"]
