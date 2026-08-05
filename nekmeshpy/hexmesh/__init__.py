"""3-D hex mesh container (``HexMesh``), its operations, and constrained smoothing.

``hexmesh.py`` is a pure container -- storage, validation and the derived views.
Everything that *acts* on a finished block is a free function in a sibling module,
split by **arity** and by **rung delta** (how far up or down the line -> quad -> hex
ladder the operation moves):

============== ======== ===== ===============================================
module         arity    delta contents
============== ======== ===== ===============================================
``_assemble``  n-ary    +1/0  ``loft``, ``loft_fn``, ``merge`` -- new index space
``_lift``      fixed    +1    ``extrude``/``sweep``/``annulus``/``from_grid`` -> ``loft``
``_morph``     fixed     0    ``blend`` + ``translate``/``rotate``/``scale``
``_query``     fixed     exit read-only queries, topology and reporting
============== ======== ===== ===============================================

Each module ends in a registry -- ``FACTORIES`` for the ``staticmethod``-bound
combinators, ``METHODS`` for the instance-method-bound queries -- and this package
binds them onto the class below, so callers write ``HexMesh.extrude(...)`` /
Every operation is a free function taking the mesh first, reached through one of the
namespaces re-exported here.  Nothing is bound onto the container::

    hexmesh.assemble.loft(sections, fractions)   # n-ary: loft / loft_fn / merge
    hexmesh.lift.extrude(section, length=1.0)    # generic rung-raising
    hexmesh.shape.tetra(corners, n)              # owns a shape model
    hexmesh.morph.translate(mesh, vector)        # rung-preserving + blend
    hexmesh.query.is_watertight(mesh)            # reads that leave the ladder

``hexmesh.py`` is therefore pure storage -- arrays, validation and the derived views --
and imports no sibling.  That makes the package a strict DAG: every sibling does a
plain ``from .hexmesh import HexMesh`` like any normal module, so there are no deferred
function-body imports and no ``TYPE_CHECKING`` guards anywhere.

The namespaces are real modules rather than aliases of the private siblings, because
Sphinx registers a target under a module's own ``__name__`` -- an alias documents
nothing and every cross-reference to it breaks.

There is no ``_project`` module: the delta -1 cell (a block's boundary *as* a
``QuadMesh``) is empty at every rung today -- ``boundary_faces`` returns
``[element, face]`` index pairs, not a mesh.
"""

from . import assemble, lift, morph, query, shape
from .hexmesh import HexMesh

__all__ = ["HexMesh", "assemble", "lift", "morph", "query", "shape"]
