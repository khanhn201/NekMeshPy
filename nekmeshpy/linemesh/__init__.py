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

Operations reach callers two ways, split by whether they take a mesh.

**Mesh-first** operations -- the unary placements and the reads -- are assigned into
the ``LineMesh`` class body by ``linemesh.py``, so ``lm.translate(v)`` /
``lm.boundary_points()`` read as they should.

**Factories** take no mesh, so binding them onto the class buys nothing.  They are
reached through the namespaces re-exported here instead::

    linemesh.assemble.loft(points)        # n-ary: loft / loft_fn / merge
    linemesh.shape.circle(radius, n)      # shape factories: line / arc / circle / ...
    linemesh.morph.blend(a, b, fractions)

That split is also what keeps the import graph a DAG.  ``linemesh.py`` imports only
``_morph`` and ``_query`` (the two holding mesh-first operations), so every factory
module is free to ``from .linemesh import LineMesh`` at module level like any normal
module -- no deferred function-body imports, and ``TYPE_CHECKING`` only in those two.

Assignment rather than ``setattr`` is the load-bearing detail for the bound half: mypy
resolves a class-body assignment to the function's real signature, so a wrong argument
is a type error.  (A class-scoped ``import`` would not work -- mypy rejects it outright
with ``Unsupported class scoped import`` and types every name ``Any``.)
"""

from . import assemble, morph, shape
from .linemesh import LineMesh

__all__ = ["LineMesh", "assemble", "morph", "shape"]
