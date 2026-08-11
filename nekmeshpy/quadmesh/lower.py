"""Fixed-arity ``QuadMesh`` operations that **lower** a rung (delta -1)."""

from __future__ import annotations

import numpy as np

from .._typing import IntArray, PointArray, StrArray
from ..core import conform
from ..core.tags import ElementTags
from ..linemesh import LineMesh
from .quadmesh import QuadMesh
from .query import boundary_edges


def boundary_mesh(mesh: QuadMesh, tag: str | None = None) -> LineMesh:
    """A section's boundary as a ``LineMesh``, carrying the section's **own** nodes."""
    # a tag names a shared *edge*, so the quads carrying it are looked up rather than
    # stored: an edge on the boundary has one, an interior one has both of its own.
    named: StrArray = mesh.edge_tags.dense(mesh.line_mesh.n_lines)
    if tag is None:
        sel: IntArray = boundary_edges(mesh)
    else:
        hit = np.argwhere(named[np.asarray(mesh.quad, dtype=np.int64)] == tag)
        if hit.shape[0] == 0:
            raise ValueError(
                "boundary_mesh: no edge carries the tag %r; this section has %s"
                % (tag, sorted(mesh.edge_tags.group_tags) or "no tagged edges"))
        sel = np.column_stack([hit[:, 0], hit[:, 1] + 1]).astype(np.int64)

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
        elem = ElementTags.from_dense(
            named[np.asarray(mesh.quad, dtype=np.int64)[sel[:, 0], sel[:, 1] - 1]])
    return LineMesh(mesh.points[gids], local, en, elem)


__all__ = ["boundary_mesh"]
