"""2-D quad mesh container (``QuadMesh``), its operations, and smoothing.

``quadmesh.py`` is a pure container -- storage, validation and the derived views.
Everything that *acts* on a finished section is a free function in a sibling module,
split by **arity** and by **rung delta** (how far up or down the line -> quad -> hex
ladder the operation moves):

============== ======== ===== ===============================================
module         arity    delta contents
============== ======== ===== ===============================================
``_assemble``  n-ary    +1/0  ``loft``, ``loft_fn``, ``merge`` -- new index space
``_lift``      fixed    +1    ``extrude``/``sweep``/``annulus``/``from_grid`` -> ``loft``
``_morph``     fixed     0    ``blend`` + ``translate``/``rotate``/``scale``
``_query``     fixed     exit read-only queries returning plain arrays
``_open``      fixed    +1    region fills (``structured`` / ``ogrid`` / ...)
``_closed``    fixed    +1    closed surfaces (``box`` / ``sphere`` / ...)
============== ======== ===== ===============================================

``_open`` / ``_closed`` are also rung-raising, but they own a *shape* model rather than
being generic over any input, which is what keeps them separate from ``_lift``.

Every operation is a free function taking the mesh first, reached through one of the
namespaces re-exported here.  Nothing is bound onto the container::

    quadmesh.assemble.loft(sections, fractions)   # n-ary: loft / loft_fn / merge
    quadmesh.lift.extrude(curve, length=1.0)      # generic rung-raising
    quadmesh.region.ogrid(boundary, n, radial)    # region fills (own a shape model)
    quadmesh.surface.box(lo, hi, n)               # closed shells
    quadmesh.morph.translate(qm, vector)          # rung-preserving + blend
    quadmesh.query.boundary_edges(qm)             # reads that leave the ladder

``quadmesh.py`` is therefore pure storage -- arrays, validation and the derived views --
and imports no sibling.  That makes the package a strict DAG: every sibling does a
plain ``from .quadmesh import QuadMesh`` like any normal module, so there are no
deferred function-body imports and no ``TYPE_CHECKING`` guards anywhere.

The namespaces are real modules rather than aliases of the private siblings, because
Sphinx registers a target under a module's own ``__name__`` -- an alias documents
nothing and every cross-reference to it breaks.

The shared ``_apply_smoothing`` / ``_check_boundary`` / ``_elevate`` factory internals
live in ``_helpers.py``.
"""

from . import assemble, lift, morph, query, region, surface
from .quadmesh import NO_TAG, QuadMesh

__all__ = ["NO_TAG", "QuadMesh", "assemble", "lift", "morph", "query", "region",
           "surface"]
