"""Fixed-arity ``HexMesh`` operations that **lower** a rung (delta -1)."""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from .._typing import IntArray, PointArray
from ..core import conform
from ..core.tags import ElementTags
from ..linemesh import LineMesh
from ..pointmesh import PointMesh
from ..quadmesh import QuadMesh
from .hexmesh import HexMesh
from .query import boundary_faces


def _selected_faces(mesh: HexMesh, tag: str | None) -> IntArray:
    """``(K,2)`` of ``[element, local face]`` for the requested group: the *tagged* rows
    when ``tag`` is given, the *topological* boundary when it is not."""
    if tag is None:
        return boundary_faces(mesh)
    rows = mesh.face_tags.select(mesh.face_tags.mask_for(tag))
    if len(rows) == 0:
        raise ValueError(
            "boundary_mesh: no face carries the tag %r; this mesh has %s"
            % (tag, sorted(mesh.face_tags.group_tags) or "no tagged faces"))
    return np.column_stack([rows.elements, rows.sides]).astype(np.int64)


def _face_corners(mesh: HexMesh, sel: IntArray) -> IntArray:
    """``(K,4)`` global corner ids of each selected face, in the hex's own CCW
    winding for that local face."""
    return mesh.hexes[sel[:, 0][:, None], HexMesh.FACE_POINTS[sel[:, 1] - 1, :]]


def _high_order_nodes(mesh: HexMesh, quads_global: IntArray, edges_global: IntArray,
                      ) -> tuple[PointArray, PointArray]:
    """The parent's own shared edge-interior and face-interior nodes for the given
    global edges and quads -- read out, never re-derived, which is the whole point."""
    e_idx = conform.locate_rows(mesh.edges, edges_global,
                                who="boundary_mesh", what="edge")
    en: PointArray = np.asarray(mesh.edge_nodes, dtype=float)[e_idx].copy()
    # the parent stores an edge's nodes min->max corner; flip those our edge traverses
    # the other way, so they read along the extracted edge's own direction
    rev = mesh.edges[e_idx, 0] != edges_global[:, 0]
    if en.size:
        en[rev] = en[rev][:, ::-1]
    f_idx = conform.locate_rows(mesh.faces, quads_global,
                                who="boundary_mesh", what="face")
    # the parent stores a face's interior in the frame of the row *it* chose, not of the
    # winding we are extracting it with -- turn it into ours, the face-family
    # counterpart of the edge reversal just above
    fn: PointArray = conform.face_nodes_in_frame(
        np.asarray(mesh.face_nodes, dtype=float)[f_idx], quads_global,
        np.asarray(mesh.quads.quads, dtype=np.int64)[f_idx])
    return en, fn


def boundary_mesh(mesh: HexMesh, tag: str | None = None, *,
                  template: QuadMesh | None = None) -> QuadMesh:
    """A block's boundary surface as a ``QuadMesh``, carrying the block's **own** nodes.
    """
    sel = _selected_faces(mesh, tag)
    poly = _face_corners(mesh, sel)
    gids: IntArray = np.unique(poly)

    if template is not None:
        return _templated(mesh, tag, poly, gids, template)

    # compact the parent's point ids into the surface's own numbering
    local: IntArray = np.searchsorted(gids, poly)
    edges, elem_edges, flip = conform.unique_edges(local, 2)
    en, fn = _high_order_nodes(mesh, poly, gids[edges])
    tags = mesh.face_tags.as_dict() if tag is None else None
    if tag is not None:
        elem = ElementTags.uniform(poly.shape[0], tag)
    else:
        names = [tags.get((int(e), int(s)), "") for e, s in sel]   # type: ignore[union-attr]
        elem = ElementTags.from_dense(np.asarray(names, dtype=np.str_))
    lines = LineMesh(mesh.points[gids], edges, en)
    return QuadMesh(lines, elem_edges, flip, fn, elem)


def _templated(mesh: HexMesh, tag: str | None, poly: IntArray, gids: IntArray,
               template: QuadMesh) -> QuadMesh:
    """``template``'s structure carrying ``mesh``'s own coordinates over the selected
    faces, paired to the template by nearest corner."""
    dist, loc = cKDTree(mesh.points[gids]).query(template.points)
    g: IntArray = gids[loc]
    if len(set(g.tolist())) != g.size:
        raise ValueError(
            "boundary_mesh: the template does not pair one-for-one with the %s face "
            "group -- %d of its %d points share a nearest port corner, so the "
            "template is not this port's own pattern"
            % (tag, g.size - len(set(g.tolist())), g.size))
    tl = template.lines
    en, fn = _high_order_nodes(mesh, g[np.asarray(template.quads, dtype=np.int64)],
                               g[np.asarray(tl.lines, dtype=np.int64)])
    lines = LineMesh(PointMesh(mesh.points[g], tl.point_tags), tl.lines, en,
                     tl.element_tags)
    _log_pairing(tag, float(np.max(dist)))
    return QuadMesh(lines, template.quad, template.flip, fn,
                    template.element_tags)


def _log_pairing(tag: str | None, worst: float) -> None:
    import logging
    logging.getLogger(__name__).debug(
        "boundary_mesh[%s]: template paired to %.3e, now exact on the mesh's own nodes",
        tag, worst)


__all__ = ["boundary_mesh"]
