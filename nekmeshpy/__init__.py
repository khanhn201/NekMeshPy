"""NekMeshPy -- an object-oriented, all-hex meshing toolkit with Nek5000/NekRS export.

The package is a library of composable primitives, not a collection of
geometry-specific meshers:

* **geometry** -- :class:`~nekmeshpy.linemesh.LineMesh`
  (a 1-D mesh -- ``(N,3)`` points with ``(L,2)`` line connectivity, open or
  closed, carrying ``element_tags`` / ``boundary_tags``),
  :class:`~nekmeshpy.trimesh.TriMesh`,
  :class:`~nekmeshpy.quadmesh.QuadMesh`,
  :class:`~nekmeshpy.hexmesh.HexMesh` (built with the ``extrude`` /
  ``merge`` / ``from_grid`` factories), and the shared-point
  :class:`~nekmeshpy.model.mesh.Mesh`.  Each mesh type is its own subpackage; the
  quad section-smoothing strategies live in
  :mod:`~nekmeshpy.quadmesh.smoothing`, the constrained hex smoother in
  :mod:`~nekmeshpy.hexmesh.smoothing`, the scaled-Jacobian quality metrics
  in :mod:`~nekmeshpy.hexmesh.quality` /
  :mod:`~nekmeshpy.quadmesh.quality`, and the ``TriMesh`` surface
  algorithms in :mod:`~nekmeshpy.trimesh.ops`, each beside their container;
* **model** -- physical groups, watertight / conformal :mod:`topology`, and sizing
  :mod:`fields`;
* **io** -- :mod:`~nekmeshpy.io.export` (``.re2`` / ``.rea`` / meshio) and
  :mod:`~nekmeshpy.io.viz`.

Concrete geometry meshers (bifurcation, pipes, transfinite block) are built on
top of these primitives and live in ``examples/``.
"""

from .hexmesh import HexMesh, smoothing
from .io import export, viz
from .linemesh import LineMesh
from .model import fields, topology
from .model.fields import AxisLinearField, ConstantField, DistanceField, Field, MinField
from .model.mesh import Mesh
from .model.physical import PhysicalGroup, PhysicalGroups
from .quadmesh import NO_BOUNDARY, QuadMesh
from .quadmesh.smoothing import (
    SECTION_METHODS,
    register_section_smoothing,
    set_section_smoothing,
)
from .trimesh import TriMesh
from .trimesh import ops as trisurf

__all__ = [
    "LineMesh",
    "TriMesh",
    "QuadMesh",
    "HexMesh",
    "NO_BOUNDARY",
    "Mesh",
    "PhysicalGroup", "PhysicalGroups",
    "topology",
    "fields",
    "export", "trisurf", "smoothing", "viz",
    "Field", "ConstantField", "AxisLinearField", "DistanceField", "MinField",
    "register_section_smoothing", "SECTION_METHODS", "set_section_smoothing",
]
