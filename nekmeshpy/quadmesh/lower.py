"""Fixed-arity ``QuadMesh`` operations that **lower** a rung (delta -1)."""

from __future__ import annotations

import numpy as np

from .._typing import IntArray, PointArray
from ..core import conform
from ..core.tags import ElementTags
from ..linemesh import LineMesh
from .quadmesh import QuadMesh
from .query import boundary_edges


def boundary_mesh(mesh: QuadMesh, tag: str | None = None) -> LineMesh:
    """A section's boundary as a ``LineMesh``, carrying the section's **own** nodes."""
    if tag is None:
        sel: IntArray = boundary_edges(mesh)
    else:
        rows = mesh.edge_tags.select(mesh.edge_tags.mask_for(tag))
        if len(rows) == 0:
            raise ValueError(
                "boundary_mesh: no edge carries the tag %r; this section has %s"
                % (tag, sorted(mesh.edge_tags.group_tags) or "no tagged edges"))
        sel = np.column_stack([rows.elements, rows.sides]).astype(np.int64)

    pairs: IntArray = mesh.quads[sel[:, 0][:, None],
                                 QuadMesh.EDGE_POINTS[sel[:, 1] - 1, :]]
    gids: IntArray = np.unique(pairs)
    local: IntArray = np.searchsorted(gids, pairs)

    e_idx = conform.locate_rows(mesh.edges, pairs, who="boundary_mesh", what="edge")
    en: PointArray = np.asarray(mesh.edge_nodes, dtype=float)[e_idx].copy()
    # the parent stores an edge's nodes min->max corner; flip those this loop traverses
    # the other way, so they read along the extracted element's own direction
    rev = mesh.edges[e_idx, 0] != pairs[:, 0]
    if en.size:
        en[rev] = en[rev][:, ::-1]

    if tag is not None:
        elem = ElementTags.uniform(pairs.shape[0], tag)
    else:
        named = mesh.edge_tags.as_dict()
        elem = ElementTags.from_dense(
            np.asarray([named.get((int(e), int(s)), "") for e, s in sel],
                       dtype=np.str_))
    return LineMesh(mesh.points[gids], local, en, elem)


__all__ = ["boundary_mesh"]
