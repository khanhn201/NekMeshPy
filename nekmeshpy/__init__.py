"""NekMeshPy -- an all-hex meshing toolkit with Nek5000/NekRS export."""

from . import tetmesh, trimesh
from .core import fields, topology
from .core.fields import AxisLinearField, ConstantField, DistanceField, Field, MinField
from .core.mesh import Mesh
from .core.physical import PhysicalGroup, PhysicalGroups
from .core.tags import (
    ElementTags,
)
from .hexmesh import HexMesh, smoothing
from .io import viz, writer
from .linemesh import LineMesh
from .pointmesh import PointMesh
from .quadmesh import NO_TAG, QuadMesh
from .quadmesh.smoothing import (
    SECTION_METHODS,
    register_section_smoothing,
    set_section_smoothing,
)
from .tetmesh import TetMesh
from .trimesh import TriMesh

__all__ = [
    "LineMesh",
    "PointMesh",
    "TetMesh",
    "TriMesh",
    "tetmesh",
    "trimesh",
    "QuadMesh",
    "HexMesh",
    "NO_TAG",
    "ElementTags",
    "Mesh",
    "PhysicalGroup", "PhysicalGroups",
    "topology",
    "fields",
    "smoothing", "viz", "writer",
    "Field", "ConstantField", "AxisLinearField", "DistanceField", "MinField",
    "register_section_smoothing", "SECTION_METHODS", "set_section_smoothing",
]
