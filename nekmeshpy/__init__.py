"""NekMeshPy -- an object-oriented, all-hex meshing toolkit with Nek5000/NekRS export.

The package is a library of composable primitives, not a collection of
geometry-specific meshers:

* **geometry** -- :class:`~nekmeshpy.geometry.curve.Curve` / ``CurveLoop``
  (an ``(N,3)`` coordinate array with open/closed semantics),
  :class:`~nekmeshpy.geometry.trimesh.TriMesh`,
  :class:`~nekmeshpy.geometry.quadmesh.QuadMesh`,
  :class:`~nekmeshpy.geometry.hexmesh.HexMesh` (built with the ``extrude`` /
  ``merge`` / ``from_grid`` factories), and the shared-point
  :class:`~nekmeshpy.model.mesh.Mesh`;
* **model** -- physical groups, scaled-Jacobian :mod:`quality`, watertight /
  conformal :mod:`topology`, and sizing :mod:`fields`;
* **ops** -- interior repositioning, smoothing, and surface algorithms
  (:mod:`~nekmeshpy.ops.trisurf`);
* **io** -- :mod:`~nekmeshpy.io.export` (``.re2`` / ``.rea`` / meshio) and
  :mod:`~nekmeshpy.io.viz`.

Concrete geometry meshers (bifurcation, pipes, transfinite block) are built on
top of these primitives and live in ``examples/``.
"""

from .geometry.curve import Curve, CurveLoop
from .geometry.hexmesh import HexMesh
from .geometry.quadmesh import NO_BOUNDARY, QuadMesh
from .geometry.trimesh import TriMesh
from .io import export, viz
from .model import fields, quality, topology
from .model.fields import AxisLinearField, ConstantField, DistanceField, Field, MinField
from .model.mesh import Mesh
from .model.physical import PhysicalGroup, PhysicalGroups
from .ops import smoothing, trisurf
from .ops.interior import SECTION_METHODS, register_section_interior, set_section_interior

__all__ = [
    "Curve", "CurveLoop",
    "TriMesh",
    "QuadMesh",
    "HexMesh",
    "NO_BOUNDARY",
    "Mesh",
    "PhysicalGroup", "PhysicalGroups",
    "quality",
    "topology",
    "fields",
    "export", "trisurf", "smoothing", "viz",
    "Field", "ConstantField", "AxisLinearField", "DistanceField", "MinField",
    "register_section_interior", "SECTION_METHODS", "set_section_interior",
]
