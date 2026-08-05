"""1-D mesh container (:class:`LineMesh`), its operations, and its shape factories.

``linemesh.py`` is a pure container -- storage, validation and the derived views.
Everything that *acts* on a finished mesh is a free function in a sibling module,
split by **arity** and by **rung delta** (how far up or down the line -> quad -> hex
ladder the operation moves):

============== ======== ===== ===============================================
module         arity    delta contents
============== ======== ===== ===============================================
``_assemble``  n-ary    +1/0  ``loft``, ``loft_fn``, ``merge`` -- new index space
``_morph``     fixed     0    ``blend``, ``translate``/``rotate``/``scale``, ``reverse``
``_query``     fixed     exit read-only queries returning plain arrays
``_open``      fixed    +1    open shape factories (``line`` / ``arc``)
``_closed``    fixed    +1    closed-loop shape factories (``circle`` / ...)
============== ======== ===== ===============================================

Every operation is a free function taking the mesh first, reached through one of the
namespaces re-exported here.  Nothing is bound onto the container::

    linemesh.assemble.loft(points)             # n-ary: loft / loft_fn / merge
    linemesh.shape.circle(radius, n)           # shape factories
    linemesh.morph.translate(lm, vector)       # rung-preserving placements + blend
    linemesh.query.boundary_points(lm)         # reads that leave the ladder

``linemesh.py`` is therefore pure storage -- it holds the arrays, their validation and
the derived views, and imports no sibling.  That makes the package a strict DAG: every
sibling does a plain ``from .linemesh import LineMesh`` like any normal module, so
there are no deferred function-body imports and no ``TYPE_CHECKING`` guards anywhere.

The namespaces are real modules rather than aliases of the private siblings
(``from . import _assemble as assemble``), because Sphinx registers a target under a
module's own ``__name__`` -- an alias documents nothing and every cross-reference to it
breaks.  ``shape`` merges ``_open`` and ``_closed``, since open vs closed is a storage
split rather than a caller-facing one.
"""

from . import assemble, morph, query, shape
from .linemesh import LineMesh

__all__ = ["LineMesh", "assemble", "morph", "query", "shape"]
