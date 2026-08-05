"""2-D quad mesh container (``QuadMesh``), its operations, and smoothing.

``quadmesh.py`` is a pure container -- storage, validation and the derived views.
Everything that *acts* on a finished section is a free function in a sibling module,
split by **arity** and by **rung delta** (how far up or down the line -> quad -> hex
ladder the operation moves):

============ ======== ===== ==================================================
module       arity    delta contents
============ ======== ===== ==================================================
``assemble`` n-ary    +1/0  ``loft``, ``loft_fn``, ``merge`` -- new index space
``lift``     fixed    +1    ``extrude``/``sweep``/``annulus``/``from_grid``
``shape``    fixed    +1    region fills and closed surfaces; own a *shape*
                            model, unlike ``lift``
``morph``    fixed    0     ``blend`` + ``translate``/``rotate``/``scale``/...
``query``    fixed    exit  read-only queries returning plain arrays
============ ======== ===== ==================================================

Every operation is a free function taking the mesh first, reached through one of the
namespaces re-exported here.  Nothing is bound onto the container::

    quadmesh.loft(sections, fractions)   # n-ary: loft / loft_fn / merge
    quadmesh.extrude(curve, length=1.0)      # generic rung-raising
    quadmesh.ogrid(boundary, n, radial)     # region fills and closed shells
    quadmesh.box(lo, hi, n)                 #   -- both own a shape model
    quadmesh.translate(qm, vector)          # rung-preserving + blend
    quadmesh.boundary_edges(qm)             # reads that leave the ladder

``quadmesh.py`` is therefore pure storage -- arrays, validation and the derived views --
and imports no sibling.  That makes the package a strict DAG: every sibling does a
plain ``from .quadmesh import QuadMesh`` like any normal module, so there are no
deferred function-body imports and no ``TYPE_CHECKING`` guards anywhere.

These namespace modules hold the code directly -- there is no private sibling behind
them and no facade layer.  Sphinx registers each object under the ``__name__`` of the
module that defines it, so a module alias would document nothing; that is why they are
real modules.

Each rung's operations are also re-exported here, so naming the namespace module is
optional at the call site -- ``quadmesh.ogrid(...)`` and ``quadmesh.shape.ogrid(...)`` are the same
function.  The short form is usually what a caller wants: it keeps the *rung* explicit,
which is the part that carries meaning, without also spelling out which module the
operation happens to live in.  Names are unique within a rung so the flattening is
unambiguous; across rungs they deliberately collide (each rung has its own ``loft`` and
``merge``), which is why there is no flat namespace above this one.
"""

from . import assemble, lift, morph, query, shape
from .assemble import loft, loft_fn, merge
from .lift import annulus, extrude, from_grid, sweep
from .morph import blend, rotate, scale, transform, translate
from .quadmesh import NO_TAG, QuadMesh
from .query import (
    boundary_edges,
    boundary_elements,
    boundary_points,
    quality_summary,
    scaled_jacobian,
)
from .shape import (
    box,
    half_box,
    half_ogrid,
    hemisphere,
    ogrid,
    quadrant_core,
    quadrant_disc,
    quadrant_ogrid,
    quadrant_seam_fractions,
    rectangle,
    sphere,
    spine_fractions,
    spined_ogrid,
    structured,
)

__all__ = [
    "NO_TAG",
    "QuadMesh",
    "annulus",
    "assemble",
    "blend",
    "boundary_edges",
    "boundary_elements",
    "boundary_points",
    "box",
    "extrude",
    "from_grid",
    "half_box",
    "half_ogrid",
    "hemisphere",
    "lift",
    "loft",
    "loft_fn",
    "merge",
    "morph",
    "ogrid",
    "quadrant_core",
    "quadrant_disc",
    "quadrant_ogrid",
    "quadrant_seam_fractions",
    "quality_summary",
    "query",
    "rectangle",
    "rotate",
    "scale",
    "scaled_jacobian",
    "shape",
    "sphere",
    "spine_fractions",
    "spined_ogrid",
    "structured",
    "sweep",
    "transform",
    "translate",
]
