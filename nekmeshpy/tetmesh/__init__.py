"""Linear tetrahedral volume-mesh container (:class:`TetMesh`) and its ``ops``.

Deliberately thin, mirroring :mod:`trimesh <nekmeshpy.trimesh>`: a tet mesh here exists
to solve a field or walk a volume, not to be exported -- a mesher's actual output is
the hex mesh built *around* what the solve decided. gmsh (:func:`tetmesh.ops.tet_mesh
<nekmeshpy.tetmesh.ops.tet_mesh>`) is an optional extra, imported only inside that one
function.
"""

from . import ops
from .tetmesh import TetMesh

__all__ = ["TetMesh", "ops"]
