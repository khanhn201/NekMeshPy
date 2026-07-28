"""NekMeshPy -- an all-hex meshing toolkit with Nek5000/NekRS export.

A library of composable primitives, not geometry-specific meshers:

* **geometry** -- ``LineMesh``, ``TriMesh``, ``QuadMesh``, ``HexMesh``, and the
  shared-point ``Mesh``.  Each mesh type is its own subpackage, with quality
  metrics, smoothing, and surface ops beside their container;
* **model** -- physical groups, ``topology`` checks, and sizing ``fields``;
* **io** -- ``export`` (``.re2`` / ``.rea`` / meshio) and ``viz``.

Concrete meshers (bifurcation, pipes, transfinite block) live in ``examples/``.
"""

from . import trimesh
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

__all__ = [
    "LineMesh",
    "TriMesh",
    "trimesh",
    "QuadMesh",
    "HexMesh",
    "NO_BOUNDARY",
    "Mesh",
    "PhysicalGroup", "PhysicalGroups",
    "topology",
    "fields",
    "export", "smoothing", "viz",
    "Field", "ConstantField", "AxisLinearField", "DistanceField", "MinField",
    "register_section_smoothing", "SECTION_METHODS", "set_section_smoothing",
]
