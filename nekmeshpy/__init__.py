"""NekMeshPy -- object-oriented port of the SURFACE pipeline of the bifurcation
hex-mesh generator.

The pipeline is built up through mesh objects:

    tri surface (TriMesh)
        -> seam fields + cut into legs (CutSurface)
        -> per-leg cross-section rings (Ring) + O-grid slices (QuadMesh via OGridLeg)
        -> assembled all-hex mesh (HexMesh): interior, smoothing, export, plot

Entry point::

    from nekmeshpy import Config, BifurcationMesher
    hexmesh = BifurcationMesher(Config()).run()

or ``python -m nekmeshpy``.
"""

from .algorithms.bifurcation import BifurcationMesher
from .algorithms.blocks import TransfiniteBlock
from .algorithms.cutsurface import CutSurface
from .algorithms.ogrid import OGridLeg
from .algorithms.pipes import (
    CircularPipe,
    RectangularPipe,
    circular_section,
    rectangular_section,
)
from .algorithms.registry import (
    ALGORITHMS,
    HexAlgorithm,
    available,
    make,
    register_algorithm,
)
from .config import Config
from .geometry.hexmesh import HexMesh
from .geometry.polyline import Arc, Polyline, Ring
from .geometry.quadmesh import QuadMesh
from .geometry.trimesh import TriMesh
from .io import export, viz
from .model import fields, quality
from .model.fields import AxisLinearField, ConstantField, DistanceField, Field, MinField
from .model.mesh import Mesh
from .model.physical import PhysicalGroup, PhysicalGroups
from .ops import smoothing, trisurf
from .ops.interior import INTERIOR_METHODS, register_interior, set_interior

__all__ = [
    "Config",
    "Polyline", "Arc", "Ring",
    "TriMesh",
    "QuadMesh",
    "OGridLeg",
    "CutSurface",
    "Mesh",
    "PhysicalGroup", "PhysicalGroups",
    "quality",
    "fields",
    "export", "trisurf", "smoothing", "viz",
    "Field", "ConstantField", "AxisLinearField", "DistanceField", "MinField",
    "register_interior", "INTERIOR_METHODS", "set_interior",
    "HexAlgorithm", "register_algorithm", "ALGORITHMS", "available", "make",
    "HexMesh",
    "TransfiniteBlock",
    "CircularPipe", "RectangularPipe", "circular_section", "rectangular_section",
    "BifurcationMesher",
]
