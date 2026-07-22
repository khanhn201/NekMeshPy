"""Quad mesh of a single cross-section slice.

``QuadMesh`` is a plain container: node coordinates, quad connectivity, and the
set of quad edges that lie on the outer wall boundary (``wall_edges``, each a
``frozenset({i, j})`` of node indices).  It carries no notion of how the section
was generated -- dedicated section meshers (the O-grid builder in
:mod:`nekmeshpy.algorithms.ogrid`, the pipe/duct builders in :mod:`nekmeshpy.algorithms.pipes`)
produce ``QuadMesh`` instances.  A stack of these slices (sharing connectivity)
is extruded into hexes by
:meth:`~nekmeshpy.geometry.hexmesh.HexMesh.add_extruded_section`.
"""

import numpy as np


class QuadMesh:
    def __init__(self, nodes, quads, wall_edges=None):
        self.nodes = np.asarray(nodes, dtype=float)
        self.quads = np.asarray(quads, dtype=np.int64)
        self.wall_edges = (set() if wall_edges is None
                           else {frozenset(e) for e in wall_edges})

    @property
    def n_nodes(self):
        return self.nodes.shape[0]

    @property
    def n_quads(self):
        return self.quads.shape[0]
