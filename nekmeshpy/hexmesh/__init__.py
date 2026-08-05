"""3-D hex mesh container (``HexMesh``), its operations, and constrained smoothing.

``hexmesh.py`` is a pure container -- storage, validation and the derived views.
Everything that *acts* on a finished block is a free function in a sibling module,
split by **arity** and by **rung delta** (how far up or down the line -> quad -> hex
ladder the operation moves):

============ ======== ===== ==================================================
module       arity    delta contents
============ ======== ===== ==================================================
``assemble`` n-ary    +1/0  ``loft``, ``loft_fn``, ``merge`` -- new index space
``lift``     fixed    +1    ``extrude``/``sweep``/``annulus``/``from_grid``
``shape``    fixed    +1    ``tetra`` -- owns a *shape* model, unlike ``lift``
``morph``    fixed    0     ``blend`` + ``translate``/``rotate``/``scale``/...
``query``    fixed    exit  read-only queries, topology and reporting
============ ======== ===== ==================================================

Every operation is a free function taking the mesh first, reached through one of the
namespaces re-exported here.  Nothing is bound onto the container::

    hexmesh.loft(sections, fractions)   # n-ary: loft / loft_fn / merge
    hexmesh.extrude(section, length=1.0)    # generic rung-raising
    hexmesh.tetra(corners, n)              # owns a shape model
    hexmesh.translate(mesh, vector)        # rung-preserving + blend
    hexmesh.is_watertight(mesh)            # reads that leave the ladder

``hexmesh.py`` is therefore pure storage -- arrays, validation and the derived views --
and imports no sibling.  That makes the package a strict DAG: every sibling does a
plain ``from .hexmesh import HexMesh`` like any normal module, so there are no deferred
function-body imports and no ``TYPE_CHECKING`` guards anywhere.

These namespace modules hold the code directly -- there is no private sibling behind
them and no facade layer.  Sphinx registers each object under the ``__name__`` of the
module that defines it, so a module alias would document nothing; that is why they are
real modules.

Each rung's operations are also re-exported here, so naming the namespace module is
optional at the call site -- ``hexmesh.extrude(...)`` and ``hexmesh.lift.extrude(...)`` are the same
function.  The short form is usually what a caller wants: it keeps the *rung* explicit,
which is the part that carries meaning, without also spelling out which module the
operation happens to live in.  Names are unique within a rung so the flattening is
unambiguous; across rungs they deliberately collide (each rung has its own ``loft`` and
``merge``), which is why there is no flat namespace above this one.
"""

from . import assemble, lift, morph, query, shape
from .assemble import loft, loft_fn, merge
from .hexmesh import HexMesh
from .lift import annulus, extrude, from_grid, sweep, sweep_path
from .morph import blend, rotate, scale, transform, translate
from .query import (
    boundary_elements,
    boundary_faces,
    boundary_points,
    classify_points,
    is_conforming,
    is_watertight,
    quality_summary,
    report,
    scaled_jacobian,
    topology_report,
    weld,
)
from .shape import tetra

__all__ = [
    "HexMesh",
    "annulus",
    "assemble",
    "blend",
    "boundary_elements",
    "boundary_faces",
    "boundary_points",
    "classify_points",
    "extrude",
    "from_grid",
    "is_conforming",
    "is_watertight",
    "lift",
    "loft",
    "loft_fn",
    "merge",
    "morph",
    "quality_summary",
    "query",
    "report",
    "rotate",
    "scale",
    "scaled_jacobian",
    "shape",
    "sweep",
    "sweep_path",
    "tetra",
    "topology_report",
    "transform",
    "translate",
    "weld",
]
