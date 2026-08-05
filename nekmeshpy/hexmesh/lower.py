"""Fixed-arity ``HexMesh`` operations that **lower** a rung (delta -1).

One operation: :func:`boundary_mesh`, a block's boundary surface as a real
:class:`QuadMesh <nekmeshpy.quadmesh.quadmesh.QuadMesh>` rather than as the
``[element, face]`` pairs :func:`boundary_faces
<nekmeshpy.hexmesh.query.boundary_faces>` returns.

This direction was deliberately empty for a long time, on the reasoning that a caller
wanting the boundary wants to *index* it, not to mesh it.  What overturned that is
building **onto** a finished block: a connector swept off a port has to start from that
port's own nodes, and re-deriving them from the recipe that built the block lands close
rather than exact -- at ``order > 1`` :func:`merge
<nekmeshpy.hexmesh.assemble.merge>` verifies shared high-order edge and face nodes
against ``conform.entity_tol`` (~1e-9 of the model extent), and close fails there.
Reading the section straight off the mesh removes the guess.

Note this makes ``lower`` the third place a global index space is manufactured, beside
``assemble``'s ``loft`` and ``merge``: an extracted surface is genuinely new numbering,
not a view of the parent's.

Free functions bound onto :class:`HexMesh <nekmeshpy.hexmesh.hexmesh.HexMesh>` by
``hexmesh/__init__.py``; internal toolkit code imports them from here directly rather
than through the bound ``HexMesh.<name>`` sugar.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from .._typing import IntArray, PointArray
from ..linemesh import LineMesh
from ..model import conform
from ..model.tags import EdgeTags, ElementTags
from ..quadmesh import QuadMesh
from .hexmesh import HexMesh
from .query import boundary_faces


def _selected_faces(mesh: HexMesh, tag: str | None) -> IntArray:
    """``(K,2)`` of ``[element, local face]`` for the requested group: the *tagged*
    rows when ``tag`` is given, the *topological* boundary when it is not.

    The two really do differ -- ``face_tags`` may name interior planes as well, and the
    domain boundary may be partly untagged -- which is why the argument selects rather
    than filters."""
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
    fn: PointArray = np.asarray(mesh.face_nodes, dtype=float)[f_idx]
    return en, fn


def boundary_mesh(mesh: HexMesh, tag: str | None = None, *,
                  template: QuadMesh | None = None) -> QuadMesh:
    """A block's boundary surface as a ``QuadMesh``, carrying the block's **own** nodes.

    ``tag`` selects a named face group; omit it for the whole topological domain
    boundary.  Corners, shared edge-interior nodes and per-quad interior nodes are all
    read straight out of ``mesh``, so the result is bit-exact on the parent's geometry
    at any order -- which is what lets a piece built from it weld back onto the parent.

    The surface gets its **own** index space, compacted from the parent's point ids in
    ascending order.  Per-quad ``element_tags`` carry each face's own tag where it has
    one, so a multi-group extraction stays self-describing; edge tags are left empty,
    since an extracted surface's boundary edges are an artefact of where it was cut
    rather than anything the parent named.

    ``template=`` instead reuses a caller-supplied section's B-rep *structure*, pairing
    it to the port by nearest corner and filling in the parent's own nodes.  Reach for
    it when the result has to pair index-for-index with a section you already hold --
    :func:`adapter <nekmeshpy.hexmesh.lift.adapter>` and :func:`bridge
    <nekmeshpy.hexmesh.lift.bridge>` both require identical connectivity, and the plain
    form only guarantees a numbering consistent with *itself*: whether it also agrees
    with some section built elsewhere is not something to rely on.  The template
    supplies only numbering; every coordinate still comes from ``mesh``."""
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
    lines = LineMesh(mesh.points[gids], edges, en, order=mesh.order)
    return QuadMesh(lines, elem_edges, flip, fn, EdgeTags.empty(), elem,
                    order=mesh.order)


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
    lines = LineMesh(mesh.points[g], tl.lines, en, tl.point_tags, tl.element_tags,
                     order=tl.order)
    _log_pairing(tag, float(np.max(dist)))
    return QuadMesh(lines, template.quad, template.flip, fn, template.edge_tags,
                    template.element_tags, order=template.order)


def _log_pairing(tag: str | None, worst: float) -> None:
    import logging
    logging.getLogger(__name__).debug(
        "boundary_mesh[%s]: template paired to %.3e, now exact on the mesh's own nodes",
        tag, worst)


__all__ = ["boundary_mesh"]
