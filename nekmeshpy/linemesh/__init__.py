"""1-D mesh container (:class:`LineMesh`), its operations, and its shape factories.

``linemesh.py`` is a pure container -- storage, validation and the derived views.
Everything that *acts* on a finished mesh is a free function in a sibling module,
split by **arity** and by **rung delta** (how far up or down the line -> quad -> hex
ladder the operation moves):

============ ======== ===== ==================================================
module       arity    delta contents
============ ======== ===== ==================================================
``assemble`` n-ary    +1/0  ``loft``, ``loft_fn``, ``merge`` -- new index space
``shape``    fixed    +1    shape factories: ``line``/``arc``/``circle``/...
``morph``    fixed    0     ``blend``, ``translate``/``rotate``/``scale``/...
``query``    fixed    exit  read-only queries returning plain arrays
============ ======== ===== ==================================================

Every operation is a free function taking the mesh first, reached through one of the
namespaces re-exported here.  Nothing is bound onto the container::

    linemesh.loft(points)             # n-ary: loft / loft_fn / merge
    linemesh.circle(radius, n)           # shape factories
    linemesh.translate(lm, vector)       # rung-preserving placements + blend
    linemesh.boundary_points(lm)         # reads that leave the ladder

``linemesh.py`` is therefore pure storage -- it holds the arrays, their validation and
the derived views, and imports no sibling.  That makes the package a strict DAG: every
sibling does a plain ``from .linemesh import LineMesh`` like any normal module, so
there are no deferred function-body imports and no ``TYPE_CHECKING`` guards anywhere.

These namespace modules hold the code directly -- there is no private sibling behind
them and no facade layer.  Sphinx registers each object under the ``__name__`` of the
module that defines it, so a module alias would document nothing; that is why they are
real modules.

Each rung's operations are also re-exported here, so naming the namespace module is
optional at the call site -- ``linemesh.circle(...)`` and ``linemesh.shape.circle(...)`` are the same
function.  The short form is usually what a caller wants: it keeps the *rung* explicit,
which is the part that carries meaning, without also spelling out which module the
operation happens to live in.  Names are unique within a rung so the flattening is
unambiguous; across rungs they deliberately collide (each rung has its own ``loft`` and
``merge``), which is why there is no flat namespace above this one.
"""

from . import assemble, morph, query, shape
from .assemble import loft, loft_fn, merge
from .linemesh import LineMesh
from .morph import blend, reverse, rotate, scale, transform, translate
from .query import boundary_elements, boundary_points
from .shape import (
    arc,
    arclength_fractions,
    circle,
    line,
    path_fractions,
    rectangle,
    sweep_fractions,
)

__all__ = [
    "LineMesh",
    "arc",
    "arclength_fractions",
    "assemble",
    "blend",
    "boundary_elements",
    "boundary_points",
    "circle",
    "line",
    "loft",
    "loft_fn",
    "merge",
    "morph",
    "path_fractions",
    "query",
    "rectangle",
    "reverse",
    "rotate",
    "scale",
    "shape",
    "sweep_fractions",
    "transform",
    "translate",
]
