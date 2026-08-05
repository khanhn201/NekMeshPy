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

Each sibling holds plain free functions taking the mesh first, and ``linemesh.py``
assigns them into the ``LineMesh`` class body -- one manifest line per operation,
grouped there by how each binds.  A function taking the mesh first is assigned bare
and becomes an instance method; a pure factory takes an explicit ``staticmethod``.
That keeps the container pure data while leaving ``LineMesh.circle(...)`` /
``lm.boundary_points()`` reachable, and adding an operation still touches only the
sibling module plus that one manifest line.

Assignment rather than ``setattr`` is the load-bearing detail: mypy resolves a
class-body assignment to the function's real signature, so internal toolkit code may
call ``LineMesh.loft(...)`` and still have its arguments checked.  (Neither
``setattr`` nor a class-scoped ``import`` works -- mypy rejects the latter outright
with ``Unsupported class scoped import`` and types every name ``Any``.)
"""

from .linemesh import LineMesh

__all__ = ["LineMesh"]
