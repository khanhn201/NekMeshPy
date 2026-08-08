"""Entity-based conformal high-order storage (line + quad + hex B-rep).

The high-order layer mirrors the corner layer -- shared edges (and hex faces)
resolve to the *same* nodes from every incident element, decided by topology
(corner ids), not a coordinate search.  Three invariants ride through:

* **round-trip** -- the entity decomposition is lossless: rebuilding a mesh from its
  own B-rep tables (``quad_from_entities`` at the quad rung, ``hexmesh.merge`` of a lone
  block at the hex rung) reproduces the conformal node block exactly;
* **conformality** -- the ``conform.conformal_*`` walk deduplicates shared entity nodes
  (node count == the geometric unique-node count, well below the un-welded
  ``E*(N+1)^d``) and its corner slots are always ``points[conn]``;
* **structural exactness** -- non-conforming element-local entity nodes (incident copies
  that disagree beyond ``conform.entity_tol``) are rejected by ``scatter_edge_nodes`` /
  ``scatter_face_nodes`` rather than silently welded.

Order 1 stays a no-op: every high-order table is empty (while the edge / face topology
stays first-class) and the walk is just ``points`` + ``conn`` in block order.
"""

import numpy as np
import pytest
from conftest import conformal, curved, quad_from_entities

from nekmeshpy import HexMesh, LineMesh, hexmesh, linemesh, quadmesh
from nekmeshpy.model import conform
from nekmeshpy.model.interp import corner_indices


def _shell(order, n_face=2, n_radial=2):
    """A cubed-sphere annulus: many hexes with genuinely curved shared faces in varied
    relative orientations -- the real exerciser for the D4 face machinery."""
    cube = quadmesh.box(3.0, n_face, order=order)
    sphere = quadmesh.sphere(1.0, n_face, order=order)
    return hexmesh.annulus(sphere, cube, radial=np.linspace(0.0, 1.0, n_radial + 1))


def _unique_coord_count(block, tol=1e-9):
    """Geometric unique-node count: how many distinct points the un-welded block holds
    (the target the conformal walk must dedup down to for a non-degenerate mesh)."""
    pts = block.reshape(-1, 3)
    q = np.round(pts / tol).astype(np.int64)
    return np.unique(q, axis=0).shape[0]


# -- quad round-trip through the entity store ---------------------------
@pytest.mark.parametrize("order", [2, 3, 5])
def test_quad_round_trip_through_entity_store(order):
    """The B-rep tables are a lossless description: feeding a mesh's own entity tables
    back through ``quad_from_entities`` rebuilds an identical mesh."""
    qm = quadmesh.rectangle([[0, 0, 0], [4, 0, 0], [4, 3, 0], [0, 3, 0]],
                            3, 2, order=order)
    rebuilt = quad_from_entities(qm.points, qm.quads, edge_nodes=qm.edge_nodes,
                                 interior=qm.interior, order=order)
    assert np.array_equal(rebuilt.edges, qm.edges)
    assert np.allclose(rebuilt.edge_nodes, qm.edge_nodes, atol=1e-12)
    assert np.allclose(rebuilt.interior, qm.interior, atol=1e-12)
    assert curved(rebuilt).shape == curved(qm).shape
    assert np.allclose(curved(rebuilt), curved(qm), atol=1e-12)


@pytest.mark.parametrize("order", [2, 3])
def test_ogrid_round_trip(order):
    # order rides on the boundary loop; no repositioning smoother (rejected at N>1)
    loop = linemesh.circle(2.0, 16, order=order)
    qm = quadmesh.ogrid(loop, 4, [0.0, 0.5, 1.0])
    assert qm.order == order
    rebuilt = quad_from_entities(qm.points, qm.quads, edge_nodes=qm.edge_nodes,
                                 interior=qm.interior, order=order)
    assert np.allclose(curved(rebuilt), curved(qm), atol=1e-12)
    # the walk reconstructs the curved wall exactly across the shared O-ring edges
    nodes, conn = conformal(qm)
    assert np.allclose(nodes[conn][:, corner_indices(order, 2), :],
                       qm.points[qm.quads], atol=1e-12)
    assert nodes.shape[0] < qm.n_quads * (order + 1) ** 2   # dedup happened


# -- quad conformality: the walk dedups shared edges --------------------
@pytest.mark.parametrize("order", [2, 3, 5])
def test_conformal_walk_dedups_and_reconstructs(order):
    nx, ny = 3, 2
    qm = quadmesh.rectangle([[0, 0, 0], [4, 0, 0], [4, 3, 0], [0, 3, 0]],
                            nx, ny, order=order)
    nodes, conn = conformal(qm)
    m = (order + 1) ** 2

    # dense connectivity into one global node array reconstructs the full block from
    # the B-rep: corners from points[quads], private nodes from .interior
    assert conn.shape == (qm.n_quads, m)
    block = nodes[conn]
    assert np.allclose(block[:, corner_indices(order, 2), :], qm.points[qm.quads],
                       atol=1e-12)
    assert np.allclose(block[:, conform._interior_slots(2, order), :], qm.interior,
                       atol=1e-12)

    # dedup actually happened: far fewer nodes than the un-welded block
    unwelded = qm.n_quads * m
    assert nodes.shape[0] < unwelded
    # and it matches the topological count P + Ne*(N-1) + Q*(N-1)^2 ...
    p = qm.n_points
    ne = qm.edges.shape[0]
    expect = p + ne * (order - 1) + qm.n_quads * (order - 1) ** 2
    assert nodes.shape[0] == expect
    # ... which for this non-degenerate grid equals the geometric unique count
    assert nodes.shape[0] == _unique_coord_count(block)


@pytest.mark.parametrize("order", [2, 3])
def test_shared_edge_resolves_to_same_nodes(order):
    """Two quads sharing an edge reference identical global node ids for it."""
    nx, ny = 2, 1
    qm = quadmesh.rectangle([[0, 0, 0], [2, 0, 0], [2, 1, 0], [0, 1, 0]],
                            nx, ny, order=order)
    assert qm.n_quads == 2
    _, conn = conformal(qm)
    # the shared internal edge nodes appear in both quads' connectivity
    shared = set(conn[0].tolist()) & set(conn[1].tolist())
    # 2 shared corners + (order-1) shared edge-interior nodes
    assert len(shared) == 2 + (order - 1)
    # fewer unique edges than 2*4 = 8 un-shared local edges (one edge is shared)
    assert qm.edges.shape[0] == 7


@pytest.mark.parametrize("order", [2, 3])
def test_edge_nodes_canonical_between_incident_quads(order):
    """The stored edge_nodes are read back consistently regardless of each quad's
    traversal direction: gather (canonical -> element order, honouring flip) and
    scatter (element order -> canonical) are exact inverses."""
    qm = quadmesh.rectangle([[0, 0, 0], [2, 0, 0], [2, 1, 0], [0, 1, 0]],
                            2, 1, order=order)
    # edge_nodes has one row per unique edge, each (order-1, 3)
    assert qm.edge_nodes.shape == (qm.edges.shape[0], order - 1, 3)
    # every unique edge is referenced by some quad
    assert set(np.unique(qm.quad).tolist()) == set(range(qm.edges.shape[0]))
    # gather/scatter round-trip: at least one quad traverses an edge anti-canonically
    assert qm.flip.any()
    local = conform.gather_edge_nodes(qm.edge_nodes, qm.quad, qm.flip)
    back = conform.scatter_edge_nodes(local, qm.quad, qm.flip, qm.edges.shape[0],
                                      conform.entity_tol(qm.points), "test")
    assert np.allclose(back, qm.edge_nodes, atol=1e-12)


# -- structural exactness: non-conforming input is rejected -------------
@pytest.mark.parametrize("order", [2, 3])
def test_non_conforming_edge_nodes_rejected(order):
    """Element-local edge nodes whose incident copies disagree beyond ``entity_tol``
    are a loud error, never a silent weld."""
    qm = quadmesh.rectangle([[0, 0, 0], [2, 0, 0], [2, 2, 0], [0, 2, 0]],
                            2, 2, order=order)
    local = conform.gather_edge_nodes(qm.edge_nodes, qm.quad, qm.flip)
    # perturb every edge-interior node of quad 0 -- at least one of its edges is
    # internal (shared), so the incident copies now disagree beyond tol
    local[0] += 100.0
    with pytest.raises(ValueError, match="non-conforming high-order edge"):
        conform.scatter_edge_nodes(local, qm.quad, qm.flip, qm.edges.shape[0],
                                   conform.entity_tol(qm.points), "QuadMesh.test")


def test_corners_are_single_sourced_so_cannot_disagree():
    """Corner consistency is now *structural*, not validated: the conformal walk reads
    the corner slots straight out of ``points[quads]``, so no stored high-order copy can
    ever contradict them -- including after an in-place point move."""
    qm = quadmesh.rectangle([[0, 0, 0], [2, 0, 0], [2, 2, 0], [0, 2, 0]],
                            2, 2, order=3)
    nodes, conn = conformal(qm)
    assert np.allclose(nodes[conn][:, corner_indices(3, 2), :], qm.points[qm.quads])
    qm.points[:] = qm.points + np.array([10.0, -3.0, 1.0])      # in-place corner move
    nodes, conn = conformal(qm)
    assert np.allclose(nodes[conn][:, corner_indices(3, 2), :], qm.points[qm.quads])


# -- line: interior is private (no shared edges) ------------------------
@pytest.mark.parametrize("order", [2, 3, 5])
def test_line_conformal_private_interior(order):
    lm = linemesh.line([0, 0, 0], [3, 0, 0], [0.0, 0.25, 0.6, 1.0], order=order)
    nodes, conn = conformal(lm)
    assert conn.shape == (lm.lines.shape[0], order + 1)
    block = nodes[conn]
    assert np.allclose(block[:, [0, order], :], lm.points[lm.lines], atol=1e-12)
    assert np.allclose(block[:, 1:order, :], lm.interior, atol=1e-12)
    # endpoints shared between consecutive lines; interiors private
    expect = lm.n_points + lm.lines.shape[0] * (order - 1)
    assert nodes.shape[0] == expect


# -- order-1 no-op ------------------------------------------------------
def test_order1_conformal_is_points_and_conn():
    qm = quadmesh.rectangle([[0, 0, 0], [2, 0, 0], [2, 2, 0], [0, 2, 0]], 2, 2)
    nodes, conn = conformal(qm)
    assert np.allclose(nodes, qm.points)
    # conn is in lexicographic block order (== quads under the corner winding perm)
    assert conn.shape == (qm.n_quads, 4)
    assert np.allclose(nodes[conn][:, corner_indices(1, 2), :], qm.points[qm.quads])
    assert conn[:, corner_indices(1, 2)].tolist() == qm.quads.tolist()
    # edges are first-class B-rep storage at every order: the shared edge topology is
    # present at order 1 (a 2x2 quad grid has 12 unique edges); only the per-edge
    # interior HO nodes are empty at order 1.
    assert qm.edges.shape == (12, 2)
    assert qm.edge_nodes.shape == (12, 0, 3)
    assert qm.interior.shape == (qm.n_quads, 0, 3)


# -- native B-rep storage (QuadMesh over its edge LineMesh) --------------
@pytest.mark.parametrize("order", [1, 3])
def test_quadmesh_brep_storage(order):
    """QuadMesh stores its edges as a shared LineMesh (structural conformality): the
    corners live once on ``lines.points``, ``quad``/``flip`` index its edges, and the
    derived ``.points``/``.quads`` round-trip the corner input exactly."""
    # two quads in a row, sharing exactly the middle vertical edge
    src = quadmesh.rectangle([[0, 0, 0], [2, 0, 0], [2, 1, 0], [0, 1, 0]],
                             2, 1, order=order)
    qm = quad_from_entities(src.points, src.quads, edge_nodes=src.edge_nodes,
                            interior=src.interior, order=order)

    # B-rep fields: a real LineMesh holding the shared edges + per-quad edge indices.
    assert isinstance(qm.lines, LineMesh)
    assert qm.lines.order == order
    assert qm.quad.shape == (2, 4) and qm.quad.dtype == np.int64
    assert qm.flip.shape == (2, 4) and qm.flip.dtype == bool
    # the two quads share exactly one edge
    assert len(set(qm.quad[0].tolist()) & set(qm.quad[1].tolist())) == 1

    # .points is a live view of the shared corners (single source of truth)
    assert qm.points is qm.lines.points
    # .quads is the lossless inverse of the edge decomposition
    assert np.array_equal(qm.quads, src.quads)
    # the conformal walk reproduces the source block exactly (flip handling included)
    assert np.allclose(curved(qm), curved(src))


def test_quadmesh_brep_shares_edge_nodes_across_incident_quads():
    """At order > 1 the seam edge's interior nodes are stored once and read back
    identically from both incident quads -- edge conformality is structural."""
    order = 3
    qm = quadmesh.rectangle([[0, 0, 0], [2, 0, 0], [2, 1, 0], [0, 1, 0]],
                            2, 1, order=order)
    # the shared seam edge is one row of the edge LineMesh, referenced by both quads
    shared = set(qm.quad[0].tolist()) & set(qm.quad[1].tolist())
    assert len(shared) == 1
    e = shared.pop()
    # its interior nodes are stored once on the edge LineMesh
    assert qm.edge_nodes.shape == (qm.edges.shape[0], order - 1, 3)
    assert np.array_equal(qm.edge_nodes[e], qm.lines.interior[e])


@pytest.mark.parametrize("order", [2, 3])
def test_line_merge_propagates_high_order(order):
    """LineMesh.merge welds endpoints but must carry each line's private interior
    through (previously it silently dropped the high-order nodes)."""
    a = linemesh.line([0, 0, 0], [1, 0, 0], [0.0, 0.5, 1.0], order=order)
    b = linemesh.line([1, 0, 0], [2, 0, 0], [0.0, 0.5, 1.0], order=order)
    merged = linemesh.merge([a, b])
    assert merged.order == order
    assert merged.interior.shape == (merged.lines.shape[0], order - 1, 3)
    # the merged interior nodes equal the originals' (welding only touches endpoints)
    assert np.allclose(merged.interior,
                       np.concatenate([a.interior, b.interior], axis=0), atol=1e-12)
    nodes, conn = conformal(merged)
    assert np.allclose(nodes[conn][:, 1:order, :], merged.interior, atol=1e-12)


def test_line_merge_rejects_mismatched_order():
    a = linemesh.line([0, 0, 0], [1, 0, 0], [0.0, 0.5, 1.0], order=2)
    b = linemesh.line([1, 0, 0], [2, 0, 0], [0.0, 0.5, 1.0], order=3)
    with pytest.raises(ValueError, match="same order"):
        linemesh.merge([a, b])


def test_order1_line_conformal():
    lm = linemesh.loft([[0, 0, 0], [1, 0, 0], [2, 0, 0]])
    nodes, conn = conformal(lm)
    assert np.allclose(nodes, lm.points)
    assert np.array_equal(conn, lm.lines)
    assert lm.interior.shape == (lm.n_lines, 0, 3)


# ======================================================================
# hex edges + faces (D4 orientation)
# ======================================================================

# -- slot decomposition partitions the block ---------------------------
@pytest.mark.parametrize("order", [2, 3, 5])
def test_entity_slots_partition_the_hex_block(order):
    """corners + edge-interior + face-interior + private-interior slots cover every
    hex block node slot exactly once (the topological basis for the entity store)."""
    m = (order + 1) ** 3
    corners = corner_indices(order, 3)
    edges = conform._edge_slots(3, order)[:, 1:-1].ravel()
    faces = conform._face_interior_slots(order).ravel()
    interior = conform._interior_slots(3, order)
    allslots = np.concatenate([corners, edges, faces, interior])
    assert sorted(allslots.tolist()) == list(range(m))     # exact partition


# -- exhaustive D4 kernel ----------------------------------------------
@pytest.mark.parametrize("order", [2, 3, 4, 5])
def test_d4_perm_tables_exhaustive(order):
    p, inv = conform._perm_tables(order)
    k2 = (order - 1) ** 2
    assert p.shape == (8, k2)
    ident = np.arange(k2)
    for code in range(8):
        # each code is a genuine permutation ...
        assert sorted(p[code].tolist()) == ident.tolist()
        # ... whose recorded inverse really inverts it (both directions)
        assert np.array_equal(p[code][inv[code]], ident)
        assert np.array_equal(inv[code][p[code]], ident)
    # code 0 is the identity; for k>=2 the 8 symmetries are all distinct
    assert np.array_equal(p[0], ident)
    if order >= 3:
        assert len({tuple(p[c].tolist()) for c in range(8)}) == 8


def test_face_code_maps_element_frame_to_canonical():
    """``_face_code``'s postcondition: applying the returned D4 code to each element
    corner's (u,v) lands on the canonical layout (origin=min id, its smaller-id
    neighbour=+U, other=+V, opposite=(1,1)) -- checked for every face and every ordering
    of 4 corner ids."""
    from itertools import permutations
    for _pinned, _pin, _u, _v, uv in conform._face_axes():
        for ids_t in permutations([5, 8, 13, 21]):
            ids = np.array(ids_t, dtype=np.int64)
            code = conform._face_code(ids, uv)
            origin = int(np.argmin(ids))
            o = uv[origin]
            nb = sorted((i for i in range(4)
                         if i != origin and int(np.sum(uv[i] != o)) == 1),
                        key=lambda i: int(ids[i]))
            opp = next(i for i in range(4) if i != origin and i not in nb)
            want = {origin: (0, 0), nb[0]: (1, 0), nb[1]: (0, 1), opp: (1, 1)}
            for i in range(4):
                assert conform._d4_apply(int(uv[i, 0]), int(uv[i, 1]), code, 1) == want[i]


# -- hex round-trip through the entity store ---------------------------
@pytest.mark.parametrize("order", [2, 3])
def test_hex_round_trip_curved_shell(order):
    hm = _shell(order)
    # ``merge`` of one block is the round trip: it gathers the entity store back into
    # element-local order and re-derives the whole B-rep from the corners, scattering
    # into its own numbering.  The block's numbering is its own business; its nodes are
    # not, and a table that disagrees with the corners it claims raises in the scatter.
    rebuilt = hexmesh.merge([hm])
    assert np.allclose(curved(rebuilt), curved(hm), atol=1e-12)
    nodes, conn = conformal(hm)
    assert np.allclose(nodes[conn][:, corner_indices(order, 3), :],
                       hm.points[hm.hexes], atol=1e-12)


# -- hex edge incidence is read off the faces, not re-deduplicated -----
def test_hex_edges_from_faces_resolve_to_their_own_corner_pairs():
    """The complete statement, and the only one that does not name a numbering: reading
    the stored row each ``(hex, local edge)`` claims -- reversed where its flip says so
    -- must give back that edge's own two corners, for every block builder, since each
    hands the container a differently-built shared-face ``QuadMesh``."""
    x = np.linspace(0, 1, 3)
    X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
    ring = linemesh.circle(1.0, 8)
    disc = quadmesh.ogrid(ring, 2, np.linspace(0.5, 1.0, 3))
    blocks = [_shell(1),                                    # annulus
              hexmesh.from_grid(np.stack([X, Y, Z], axis=-1)),
              hexmesh.loft([quadmesh.translate(disc, (0.0, 0.0, z))
                            for z in np.linspace(0.0, 1.0, 4)]),
              hexmesh.loft([quadmesh.translate(disc, (0.0, 0.0, z))
                            for z in np.linspace(0.0, 1.0, 4)], loop=True),
              hexmesh.merge([_shell(1, n_face=1, n_radial=1)])]
    for hm in blocks:
        rows = hm.edges[hm._elem_edges]                     # (E,12,2) as stored
        walked = np.where(hm._edge_flip[:, :, None], rows[:, :, ::-1], rows)
        assert np.array_equal(walked, hm.hexes[:, conform._LOCAL_EDGES[3]])


@pytest.mark.parametrize("order", [2, 3])
def test_hex_does_not_care_what_order_its_edges_are_stored_in(order):
    """The container reads its edge ids out of the shared-face table, so relabelling
    that table -- rows permuted, ``quad`` remapped, nodes carried along -- is invisible
    in the mesh's geometry.  Under the old dedup this silently misplaced every
    high-order edge node."""
    hm = _shell(order, n_face=1, n_radial=2)
    ne = hm.edges.shape[0]
    sigma = np.random.default_rng(0).permutation(ne)        # old id -> new slot
    lines = LineMesh(hm.points, hm.edges[sigma],
                     interior=hm.edge_nodes[sigma])
    inv = np.argsort(sigma)
    faces = quadmesh.QuadMesh(lines, inv[hm.quads.quad], hm.quads.flip,
                              hm.face_nodes)
    relabelled = HexMesh(faces, hm.hex, hm.face_orient, hm.interior)
    assert not np.array_equal(relabelled._elem_edges, hm._elem_edges)
    assert np.allclose(curved(relabelled), curved(hm), atol=1e-12)


# -- hex conformality: the walk dedups shared edges + faces ------------
@pytest.mark.parametrize("order", [2, 3])
def test_hex_conformal_walk_dedups(order):
    hm = _shell(order)
    nodes, conn = conformal(hm)
    m = (order + 1) ** 3
    assert conn.shape == (hm.n_hexes, m)
    block = nodes[conn]
    assert np.allclose(block[:, corner_indices(order, 3), :], hm.points[hm.hexes],
                       atol=1e-12)
    assert np.allclose(block[:, conform._interior_slots(3, order), :], hm.interior,
                       atol=1e-12)

    unwelded = hm.n_hexes * m
    assert nodes.shape[0] < unwelded                          # dedup happened
    p = hm.n_points
    ne, nf = hm.edges.shape[0], hm.faces.shape[0]
    expect = (p + ne * (order - 1) + nf * (order - 1) ** 2
              + hm.n_hexes * (order - 1) ** 3)
    assert nodes.shape[0] == expect                           # topological count
    assert nodes.shape[0] == _unique_coord_count(block)       # == geometric unique


@pytest.mark.parametrize("order", [2, 3])
def test_shared_hex_face_resolves_to_same_nodes(order):
    # two hexes sharing exactly one face
    x = np.array([0.0, 1.0, 2.0])
    y = z = np.array([0.0, 1.0])
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    grid = np.stack([X, Y, Z], axis=-1)
    hm = hexmesh.from_grid(grid, order=order)
    assert hm.n_hexes == 2
    _, conn = conformal(hm)
    shared = set(conn[0].tolist()) & set(conn[1].tolist())
    # a full shared face: (order+1)^2 nodes (4 corners + 4 edges + interior)
    assert len(shared) == (order + 1) ** 2
    # and the two hexes share exactly one face in the topology
    fa = {tuple(sorted(hm.hexes[0, HexMesh.FACE_POINTS[f]].tolist())) for f in range(6)}
    fb = {tuple(sorted(hm.hexes[1, HexMesh.FACE_POINTS[f]].tolist())) for f in range(6)}
    assert len(fa & fb) == 1


# -- structural exactness: non-conforming hex entities rejected --------
@pytest.mark.parametrize("order", [2, 3])
def test_non_conforming_hex_face_rejected(order):
    hm = _shell(order)
    local = conform.gather_face_nodes(hm.face_nodes, hm.hex, hm.face_orient)
    # perturb every face-interior node of hex 0; its shared faces now disagree
    local[0] += 100.0
    with pytest.raises(ValueError, match="non-conforming high-order face"):
        conform.scatter_face_nodes(local, hm.hex, hm.face_orient, hm.faces.shape[0],
                                   conform.entity_tol(hm.points), "HexMesh.test")


@pytest.mark.parametrize("order", [2, 3])
def test_non_conforming_hex_edge_rejected(order):
    hm = _shell(order)
    local = conform.gather_edge_nodes(hm.edge_nodes, hm._elem_edges, hm._edge_flip)
    local[0] += 100.0
    with pytest.raises(ValueError, match="non-conforming high-order edge"):
        conform.scatter_edge_nodes(local, hm._elem_edges, hm._edge_flip,
                                   hm.edges.shape[0], conform.entity_tol(hm.points),
                                   "HexMesh.test")


@pytest.mark.parametrize("order", [2, 3])
def test_hex_entity_gather_scatter_round_trip(order):
    """Unperturbed, gather (canonical -> element frame) and scatter (element frame ->
    canonical, owner-wins + verify) are exact inverses for both edges and faces."""
    hm = _shell(order)
    tol = conform.entity_tol(hm.points)
    e_local = conform.gather_edge_nodes(hm.edge_nodes, hm._elem_edges, hm._edge_flip)
    assert np.allclose(
        conform.scatter_edge_nodes(e_local, hm._elem_edges, hm._edge_flip,
                                   hm.edges.shape[0], tol, "test"),
        hm.edge_nodes, atol=1e-12)
    f_local = conform.gather_face_nodes(hm.face_nodes, hm.hex, hm.face_orient)
    assert np.allclose(
        conform.scatter_face_nodes(f_local, hm.hex, hm.face_orient,
                                   hm.faces.shape[0], tol, "test"),
        hm.face_nodes, atol=1e-12)


# -- order-1 hex no-op -------------------------------------------------
def test_order1_hex_conformal():
    x = np.linspace(0, 1, 3)
    X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
    hm = hexmesh.from_grid(np.stack([X, Y, Z], axis=-1))
    nodes, conn = conformal(hm)
    assert np.allclose(nodes, hm.points)
    assert np.allclose(nodes[conn][:, corner_indices(1, 3), :], hm.points[hm.hexes])
    # edges/faces are first-class B-rep storage: present at every order (a 2x2x2 grid
    # has 54 unique edges, 36 unique faces); only the HO interior nodes are empty.
    assert hm.edges.shape == (54, 2)
    assert hm.faces.shape == (36, 4)
    assert hm.edge_nodes.shape == (54, 0, 3)
    assert hm.face_nodes.shape == (36, 0, 3)
    assert hm.interior.shape == (hm.n_hexes, 0, 3)
