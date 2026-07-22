"""Generic shared-node unstructured mesh -- the gmsh/meshio-style data model.

:class:`Mesh` is the interchange representation the whole package can convert to
and from: a single shared ``points`` array plus ``cells`` grouped by element
type, with named ``point_sets`` and ``cell_sets`` (physical groups).  It is a
plain data container -- algorithms live on the typed meshes
(:class:`~nekmeshpy.geometry.trimesh.TriMesh`, :class:`~nekmeshpy.geometry.hexmesh.HexMesh`); this
is what you serialize, hand to :mod:`meshio`, or inspect.

Cell-type keys follow the meshio vocabulary: ``"vertex"``, ``"line"``,
``"triangle"``, ``"quad"``, ``"tetra"``, ``"hexahedron"``.
"""

import numpy as np

# nodes-per-cell for the types we use
_NODES_PER_CELL = {
    "vertex": 1, "line": 2, "triangle": 3, "quad": 4,
    "tetra": 4, "hexahedron": 8,
}


class Mesh:
    def __init__(self, points, cells=None, point_sets=None,
                 cell_sets=None, field_data=None):
        self.points = np.asarray(points, dtype=float).reshape(-1, 3)
        # {type: (M,k) int connectivity into points}
        self.cells = {}
        for ctype, conn in (cells or {}).items():
            self.cells[ctype] = np.asarray(conn, dtype=np.int64).reshape(
                -1, _NODES_PER_CELL[ctype])
        # {name: (n,) point ids}
        self.point_sets = {k: np.asarray(v, dtype=np.int64).ravel()
                           for k, v in (point_sets or {}).items()}
        # {name: {type: (m,) local cell ids}}
        self.cell_sets = {}
        for name, block in (cell_sets or {}).items():
            self.cell_sets[name] = {t: np.asarray(ids, dtype=np.int64).ravel()
                                    for t, ids in block.items()}
        # {name: (tag, dim)} gmsh physical-group metadata
        self.field_data = dict(field_data or {})

    # -- sizes -----------------------------------------------------------
    @property
    def n_points(self):
        return self.points.shape[0]

    @property
    def cell_types(self):
        return list(self.cells.keys())

    def n_cells(self, ctype=None):
        if ctype is not None:
            return self.cells[ctype].shape[0] if ctype in self.cells else 0
        return sum(c.shape[0] for c in self.cells.values())

    # -- meshio bridge ---------------------------------------------------
    def to_meshio(self):
        """Return an equivalent :class:`meshio.Mesh` (requires ``meshio``)."""
        import meshio
        cells = [(t, conn) for t, conn in self.cells.items()]
        return meshio.Mesh(
            points=self.points, cells=cells,
            point_sets={k: v for k, v in self.point_sets.items()},
            cell_sets={n: [b.get(t, np.empty(0, np.int64)) for t, _ in cells]
                       for n, b in self.cell_sets.items()},
            field_data={k: np.asarray(v) for k, v in self.field_data.items()},
        )

    @classmethod
    def from_meshio(cls, m):
        """Build a :class:`Mesh` from a :class:`meshio.Mesh`."""
        cells = {cb.type: cb.data for cb in m.cells}
        ordered_types = [cb.type for cb in m.cells]
        cell_sets = {}
        for name, blocks in getattr(m, "cell_sets", {}).items():
            block = {}
            for t, ids in zip(ordered_types, blocks):
                if ids is not None and len(ids):
                    block[t] = ids
            cell_sets[name] = block
        return cls(points=m.points, cells=cells,
                   point_sets=getattr(m, "point_sets", None),
                   cell_sets=cell_sets,
                   field_data=getattr(m, "field_data", None))

    def write(self, path, file_format=None):
        """Write via :mod:`meshio` (``.vtu``, ``.msh``, ``.xdmf``, ...)."""
        self.to_meshio().write(path, file_format=file_format)
        return path

    @classmethod
    def read(cls, path, file_format=None):
        """Read any meshio-supported file into a :class:`Mesh`."""
        import meshio
        return cls.from_meshio(meshio.read(path, file_format=file_format))

    def __repr__(self):
        parts = ", ".join("%s=%d" % (t, c.shape[0]) for t, c in self.cells.items())
        return "Mesh(points=%d, %s, groups=%d)" % (
            self.n_points, parts or "empty", len(self.cell_sets))
