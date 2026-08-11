"""NekMeshPy -- an all-hex meshing toolkit with Nek5000/NekRS export."""

from . import trimesh
from .core import fields, topology
from .core.fields import AxisLinearField, ConstantField, DistanceField, Field, MinField
from .core.mesh import Mesh
from .core.physical import PhysicalGroup, PhysicalGroups
from .core.tags import (
    EdgeTags,
    ElementTags,
    FaceTags,
    PointTags,
    TagBuilder,
)
from .hexmesh import HexMesh, smoothing
from .io import export, viz
from .linemesh import LineMesh
from .pointmesh import PointMesh
from .quadmesh import NO_TAG, QuadMesh
from .quadmesh.smoothing import (
    SECTION_METHODS,
    register_section_smoothing,
    set_section_smoothing,
)
from .trimesh import TriMesh

__all__ = [
    "LineMesh",
    "PointMesh",
    "TriMesh",
    "trimesh",
    "QuadMesh",
    "HexMesh",
    "NO_TAG",
    "PointTags", "EdgeTags", "FaceTags", "TagBuilder", "ElementTags",
    "Mesh",
    "PhysicalGroup", "PhysicalGroups",
    "topology",
    "fields",
    "export", "smoothing", "viz",
    "Field", "ConstantField", "AxisLinearField", "DistanceField", "MinField",
    "register_section_smoothing", "SECTION_METHODS", "set_section_smoothing",
]
