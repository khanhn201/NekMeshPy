"""3-D hex mesh container (``HexMesh``), its operations, and constrained smoothing.

``hexmesh.py`` is a pure container -- storage, validation and the derived views.
Everything that *acts* on a finished block is a free function in a sibling module,
split by **arity** and by **rung delta** (how far up or down the line -> quad -> hex
ladder the operation moves):

============== ======== ===== ===============================================
module         arity    delta contents
============== ======== ===== ===============================================
``_assemble``  n-ary    +1/0  ``loft``, ``loft_curve``, ``merge`` -- new index space
``_lift``      fixed    +1    ``extrude``/``sweep``/``annulus``/``from_grid`` -> ``loft``
``_morph``     fixed     0    ``blend`` + ``translate``/``rotate``/``scale``
``_query``     fixed     exit read-only queries, topology and reporting
============== ======== ===== ===============================================

Each module ends in a registry -- ``FACTORIES`` for the ``staticmethod``-bound
combinators, ``METHODS`` for the instance-method-bound queries -- and this package
binds them onto the class below, so callers write ``HexMesh.extrude(...)`` /
``mesh.is_watertight()`` while adding an operation touches only the sibling module (the
function plus one registry entry), never the container or this file.

There is no ``_project`` module: the delta -1 cell (a block's boundary *as* a
``QuadMesh``) is empty at every rung today -- ``boundary_faces`` returns
``[element, face]`` index pairs, not a mesh.
"""

from ._assemble import FACTORIES as _ASSEMBLE_FACTORIES
from ._lift import FACTORIES as _LIFT_FACTORIES
from ._morph import FACTORIES as _MORPH_FACTORIES
from ._morph import METHODS as _MORPH_METHODS
from ._open import FACTORIES as _OPEN_FACTORIES
from ._query import METHODS as _QUERY_METHODS
from .hexmesh import HexMesh

# The combinators are plain free functions (no ``cls``); bind as ``staticmethod`` so
# ``HexMesh.extrude(section, ...)`` passes no implicit first argument.
for _name, _fn in {**_ASSEMBLE_FACTORIES, **_LIFT_FACTORIES,
                   **_MORPH_FACTORIES, **_OPEN_FACTORIES}.items():
    setattr(HexMesh, _name, staticmethod(_fn))

# The queries and the unary placements take the mesh they act on first, except the
# few queries that read bare connectivity.
_STATIC_QUERIES = {"_boundary_mask", "_boundary_points", "_unique_edges"}
for _name, _fn in {**_QUERY_METHODS, **_MORPH_METHODS}.items():
    setattr(HexMesh, _name,
            staticmethod(_fn) if _name in _STATIC_QUERIES else _fn)

__all__ = ["HexMesh"]
