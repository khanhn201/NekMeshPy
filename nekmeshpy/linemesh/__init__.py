"""1-D mesh container (:class:`LineMesh`), its operations, and its shape factories.

``linemesh.py`` is a pure container -- storage, validation and the derived views.
Everything that *acts* on a finished mesh is a free function in a sibling module,
split by **arity** and by **rung delta** (how far up or down the line -> quad -> hex
ladder the operation moves):

============== ======== ===== ===============================================
module         arity    delta contents
============== ======== ===== ===============================================
``_assemble``  n-ary    +1/0  ``loft``, ``merge`` -- build a new index space
``_morph``     fixed     0    ``blend``, ``translate``/``rotate``/``scale``, ``reverse``
``_query``     fixed     exit read-only queries returning plain arrays
``_open``      fixed    +1    open curves (``line`` / ``arc`` / ``curve``)
``_closed``    fixed    +1    closed-loop shape factories (``circle`` / ...)
============== ======== ===== ===============================================

Each module ends in a registry -- ``FACTORIES`` for the ``staticmethod``-bound
combinators, ``METHODS`` for the instance-method-bound queries -- and this package
binds them onto the class below.  ``_open`` carries a third one, ``HELPERS``: also
``staticmethod``-bound, but for functions that answer a question *about* a factory's
input contract and return a plain array rather than a mesh, which is what keeps them
out of ``FACTORIES`` (``LineMesh.arclength_fractions``, the ``curve`` grading that
spaces nodes evenly by arc length -- the inversion is an explicit caller step, since
``curve`` meshes exactly at the fractions given).  So callers write ``LineMesh.circle(...)`` /
``lm.boundary_points()`` while adding an operation touches only the sibling module
(the function plus one registry entry), never the container or this file.
"""

from ._assemble import FACTORIES as _ASSEMBLE_FACTORIES
from ._closed import FACTORIES as _CLOSED_FACTORIES
from ._morph import FACTORIES as _MORPH_FACTORIES
from ._morph import METHODS as _MORPH_METHODS
from ._open import FACTORIES as _OPEN_FACTORIES
from ._open import HELPERS as _OPEN_HELPERS
from ._query import METHODS as _QUERY_METHODS
from .linemesh import LineMesh

# The combinators are plain free functions (no ``cls``); bind as ``staticmethod`` so
# ``LineMesh.circle(radius, n)`` passes no implicit first argument.
for _name, _fn in {**_CLOSED_FACTORIES, **_OPEN_FACTORIES,
                   **_ASSEMBLE_FACTORIES, **_MORPH_FACTORIES,
                   **_OPEN_HELPERS}.items():
    setattr(LineMesh, _name, staticmethod(_fn))

# The queries and the unary placements take the mesh they act on first, so a plain
# function binding makes them methods.
for _name, _fn in {**_QUERY_METHODS, **_MORPH_METHODS}.items():
    setattr(LineMesh, _name, _fn)

__all__ = ["LineMesh"]
