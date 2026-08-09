"""The product index space every ``loft`` numbers by: one slice, copied across levels
and joined across layers.

Each rung writes that arithmetic inline -- an entity is ``level * n_carried + k`` if it
is *carried* and ``nlev * n_carried + layer * n_swept + k`` if it is *swept*, two
contiguous blocks that cannot collide -- so what is pinned here is the behaviour, never a
helper.

These closed forms replace a dedup pass, so the thing to hold them to is that they
*agree* with one: the same entity partition ``conform.unique_edges`` would find, and the
same incidence per element.  Numbering itself is not pinned -- nothing here compares ids.

They also replace the ``scatter_*`` owner election, which was the only thing checking two
elements against each other, so the node-level checks carry that weight now.  Both use a
sweep that is a rigid motion (translation, then rotation), where the whole node block is a
tensor product and so is known outright from the section's own conformal block.  A B-rep
that is right topologically and wrong by a frame passes everything else and fails these.
"""

import numpy as np
import pytest

from nekmeshpy import linemesh, quadmesh
from nekmeshpy.model import conform
from nekmeshpy.model.fields import gll_nodes


def _profiles(n_prof, loop, order=1):
    ring = linemesh.circle(1.0, 8, order=order)
    out = [linemesh.translate(ring, (0.0, 0.0, z))
           for z in np.linspace(0.0, 1.0, n_prof + (1 if loop else 0))]
    return out[:n_prof]


# -- the element numbering every rung shares ----------------------------------
@pytest.mark.parametrize("loop", [False, True])
def test_elements_are_layer_major(loop):
    """``e = layer * n_per_slice + k`` at both rungs, and a closed sweep adds exactly one
    more layer.  Every tagging and node check below indexes by it, and ``loft``'s
    ``element_tags`` contract -- an ``ElementTags`` over *one* slice's elements tags each
    swept column -- is only meaningful because of it."""
    from nekmeshpy.model.tags import ElementTags
    profs = _profiles(4, loop)
    L, nz = profs[0].n_lines, 4 if loop else 3
    per_line = ElementTags.from_dense(np.array(["e%d" % k for k in range(L)]))
    qm = quadmesh.loft(profs, loop=loop, element_tags=per_line)
    assert qm.n_quads == nz * L                       # a closed sweep adds one layer
    assert qm.element_tags.dense(nz * L).tolist() == ["e%d" % k for k in range(L)] * nz


def _relabel_lines(m):
    """The same ``LineMesh``, same corners and geometry, shared lines renumbered."""
    from nekmeshpy import LineMesh
    sigma = np.random.default_rng(0).permutation(m.n_lines)
    return LineMesh(m.points, m.lines[sigma], interior=m.interior[sigma])


def _relabel_edges(m):
    """The same ``QuadMesh``, same corners and geometry, shared edges renumbered."""
    from nekmeshpy import LineMesh, QuadMesh
    sigma = np.random.default_rng(0).permutation(m.lines.n_lines)
    lines = LineMesh(m.points, m.lines.lines[sigma], interior=m.lines.interior[sigma])
    return QuadMesh(lines, np.argsort(sigma)[m.quad], m.flip, m.interior)


def test_loft_rejects_slices_that_are_not_index_paired():
    """Both rungs read every slice's high-order nodes by the *first* slice's entity ids,
    so a stack that agrees on corners but numbers its shared entities differently is a
    loud error.  It used to be a silently wrong mesh: structurally perfect, with every
    shared edge / face node read off the wrong entity."""
    from nekmeshpy import hexmesh
    ring = linemesh.circle(1.0, 8, order=3)
    with pytest.raises(ValueError, match="index-paired with the first"):
        quadmesh.loft([ring, _relabel_lines(linemesh.translate(ring, (0.0, 0.0, 1.0)))])
    sec = _section(3)
    top = quadmesh.translate(sec, (0.0, 0.0, 1.0))
    with pytest.raises(ValueError, match="index-paired with the first"):
        hexmesh.loft([sec, _relabel_edges(top)])
    # ... and the affine ops, which is how a stack is meant to be built, still pass
    assert hexmesh.loft([sec, top]).n_hexes == sec.n_quads


def test_loft_rejects_a_stack_that_spans_no_layer():
    """A sweep needs a layer to build anything, and a closed one needs two.  Left
    unchecked these were silent bad meshes, not errors: one profile built an empty
    section, a one-profile loop built elements whose four corners were two points, and an
    empty stack raised a bare ``IndexError``.  Two layers *is* enough to close a sweep --
    see :func:`test_a_two_layer_loop_closes_a_torus`."""
    from nekmeshpy import hexmesh
    ring = linemesh.circle(1.0, 8)
    sec = _section(1)
    pair = [ring, linemesh.translate(ring, (0.0, 0.0, 1.0))]
    secs = [sec, quadmesh.translate(sec, (0.0, 0.0, 1.0))]
    with pytest.raises(ValueError, match="at least 2 profiles"):
        quadmesh.loft([ring])
    with pytest.raises(ValueError, match="at least 2 profiles"):
        quadmesh.loft([])
    for stack in ([ring], pair):                     # one layer, then two
        with pytest.raises(ValueError, match="at least 3 profiles"):
            quadmesh.loft(stack, loop=True)
    with pytest.raises(ValueError, match="at least 2 slices"):
        hexmesh.loft([sec])
    for stack in ([sec], secs):
        with pytest.raises(ValueError, match="at least 3 slices"):
            hexmesh.loft(stack, loop=True)
    # the smallest stacks that *do* span a layer still build
    assert quadmesh.loft(pair).n_quads == ring.n_lines
    assert hexmesh.loft(secs).n_hexes == sec.n_quads
    assert quadmesh.loft(pair + [linemesh.translate(ring, (0.0, 0.0, 2.0))],
                         loop=True).n_quads == 3 * ring.n_lines


def _torus_levels(order, n_prof, a=0.6, R=3.0):
    """``(levels, sweep_nodes)`` for a torus closed in ``n_prof`` layers: a circle in the
    x-z plane at radius ``R``, revolved about z, sampled on the full GLL lattice."""
    ring = linemesh.circle(a, 8, order=order)
    prof = linemesh.translate(linemesh.rotate(ring, np.pi / 2, (1.0, 0.0, 0.0)),
                              (R, 0.0, 0.0))
    def f(t):
        return linemesh.rotate(prof, 2.0 * np.pi * t)
    g = gll_nodes(order)[1:order]
    return ([f(k / n_prof) for k in range(n_prof)],
            [[f((i + gg) / n_prof) for gg in g] for i in range(n_prof)])


@pytest.mark.parametrize("n_prof", [3, 4])
@pytest.mark.parametrize("order", [2, 3, 4])
def test_a_tight_loop_closes_a_torus(order, n_prof):
    """Three layers is the minimum a closed sweep can be *represented* at, and it is
    plenty: ``sweep_nodes`` carries each layer's own intermediate profiles, so even the
    tightest legal ring lands on the true torus rather than chording it.  Two would be
    geometrically fine and topologically unrepresentable -- both layers span the same
    pair of levels, so their rungs collide on corner ids."""
    a, R = 0.6, 3.0
    lev, sw = _torus_levels(order, n_prof, a, R)
    qm = quadmesh.loft(lev, loop=True, sweep_nodes=sw)
    assert qm.n_quads == n_prof * 8
    # a one-layer loop would leave every quad with two corners, not four
    assert all(len(set(row.tolist())) == 4 for row in np.asarray(qm.quads))
    nodes, _ = conform.conformal_quad(qm.points, qm.quads, qm.quad, qm.flip,
                                      qm.lines.interior, qm.interior, qm.order)
    tube = np.hypot(np.hypot(nodes[:, 0], nodes[:, 1]) - R, nodes[:, 2])
    assert np.max(np.abs(tube - a)) < 1e-12


@pytest.mark.parametrize("order", [2, 3])
def test_a_tight_loop_closes_a_solid_torus(order):
    """The same at the hex rung, where the collision would show as duplicate *face* rows
    too: a solid torus shut in three layers is watertight."""
    from nekmeshpy import hexmesh
    lev, sw = _torus_levels(order, 3)
    disc = [quadmesh.ogrid(m, 2, np.linspace(0.5, 1.0, 3)) for m in lev]
    disc_sw = [[quadmesh.ogrid(m, 2, np.linspace(0.5, 1.0, 3)) for m in level]
               for level in sw]
    blk = hexmesh.loft(disc, loop=True, sweep_nodes=disc_sw)
    assert blk.n_hexes == 3 * disc[0].n_quads
    assert all(len(set(row.tolist())) == 8 for row in np.asarray(blk.hexes))
    assert hexmesh.is_watertight(blk)


# -- the closed form agrees with the dedup it replaces ------------------------
@pytest.mark.parametrize("loop", [False, True])
@pytest.mark.parametrize("n_prof", [2, 3, 5])
def test_quad_loft_edges_match_a_dedup_of_its_own_corners(n_prof, loop):
    if loop and n_prof < 3:
        pytest.skip("a closed sweep needs at least two layers")
    qm = quadmesh.loft(_profiles(n_prof, loop), loop=loop)
    want_e, want_i, want_f = conform.unique_edges(np.asarray(qm.quads, np.int64), 2)
    assert qm.lines.n_lines == want_e.shape[0]
    # numbering is free; the partition into entities is not
    assert np.array_equal(np.sort(qm.lines.lines[qm.quad], axis=2),
                          np.sort(want_e[want_i], axis=2))
    # ... and the stored flip must agree with the traversal it claims
    assert np.array_equal(qm.quads, qm._derive_corners())


@pytest.mark.parametrize("order", [2, 3, 4])
@pytest.mark.parametrize("loop", [False, True])
def test_curved_nodes_survive_the_closed_form(order, loop):
    """The node-level check: every conformal node of a curved sweep sits on the true
    cylinder, which fails the moment an entity's nodes land in the wrong frame."""
    qm = quadmesh.loft(_profiles(4, loop, order=order), loop=loop)
    nodes, _ = conform.conformal_quad(qm.points, qm.quads, qm.quad, qm.flip,
                                      qm.lines.interior, qm.interior, qm.order)
    assert np.max(np.abs(np.hypot(nodes[:, 0], nodes[:, 1]) - 1.0)) < 1e-12


def test_isolated_points_get_no_rung():
    """A point no line carries borders no quad, so it must not spawn a dangling edge."""
    pts = np.array([[0.0, 0, 0], [1, 0, 0], [2, 0, 0], [9, 9, 9]])
    from nekmeshpy import LineMesh
    prof = LineMesh(pts, [[0, 1], [1, 2]])
    slices = [linemesh.translate(prof, (0.0, 0.0, z)) for z in (0.0, 1.0)]
    qm = quadmesh.loft(slices)
    referenced = np.unique(qm.quad)
    assert referenced.shape[0] == qm.lines.n_lines      # no unreferenced edge


# -- the hex rung: the gate step 3 has to pass ---------------------------------
def _section(order):
    return quadmesh.ogrid(linemesh.circle(1.0, 8, order=order), 2,
                          np.linspace(0.5, 1.0, 3))


def _levels(nz, loop, down=False):
    """The ``z`` of each level: ``nz`` layers open, ``nz`` levels closed.  Sweeping
    *down* reverses the section's handedness, which is the branch that reverses the
    element's local face / edge template."""
    n = nz if loop else nz + 1
    return np.linspace(0.0, -1.0 if down else 1.0, n)


def _inplane(sec, blk):
    """``curved(sec)`` in the *block's* in-plane frame: a left-handed section makes the
    loft reverse its corner template, which transposes the in-plane grid with it.  Read
    which off the block's own corners rather than re-deciding the handedness here."""
    from conftest import curved
    sb = curved(sec)
    row = sec.order + 1
    kk = np.arange(row * row)
    reversed_template = not np.array_equal(blk.hexes[0, :4] % sec.n_points,
                                           sec.quads[0])
    return sb[:, (kk // row) + row * (kk % row), :] if reversed_template else sb


def _shell(order, nz=3, loop=False, down=False):
    from nekmeshpy import hexmesh
    sec = _section(order)
    zs = _levels(nz, loop, down)
    slices = [quadmesh.translate(sec, (0.0, 0.0, float(z))) for z in zs]
    return sec, zs, hexmesh.loft(slices, loop=loop)


@pytest.mark.parametrize("down", [False, True])
@pytest.mark.parametrize("loop", [False, True])
def test_hex_loft_face_tables_round_trip_to_its_own_corners(loop, down):
    """``faces`` / ``elem_faces`` / ``face_orient`` must be mutually consistent: the
    container recovers the very corners the loft built."""
    _, _, blk = _shell(1, nz=4, loop=loop, down=down)
    assert np.array_equal(blk.hexes, conform.hex_corners_from_faces(
        blk.quads.quads, blk.hex, blk.face_orient))


@pytest.mark.parametrize("order", [2, 3, 4])
@pytest.mark.parametrize("down", [False, True])
@pytest.mark.parametrize("loop", [False, True])
def test_hex_loft_nodes_are_its_section_lattice_swept(order, loop, down):
    """The node-level gate, and the only check that catches a wrong *frame*.

    A pure translation sweep is a tensor product, so the answer is known outright: hex
    ``i*M + q`` holds section quad ``q``'s conformal in-plane block at every GLL level
    between the two bounding levels' ``z``.  Numbering is free and this does not look at
    it -- but a face or edge whose nodes land in the wrong frame moves them to a
    different slot of the block, which this sees."""
    from conftest import curved
    sec, zs, blk = _shell(order, nz=3, loop=loop, down=down)
    row, M = order + 1, sec.n_quads
    m2 = row * row
    sb = _inplane(sec, blk)                             # (M, m2, 3), in-plane
    g = gll_nodes(order)

    got = curved(blk)
    e, q = np.divmod(np.arange(blk.n_hexes), M)
    j = (e + 1) % zs.shape[0] if loop else e + 1
    want = np.empty_like(got)
    want[:, :, :2] = np.tile(sb[q][:, :, :2], (1, row, 1))
    want[:, :, 2] = np.repeat(
        zs[e][:, None] + (zs[j] - zs[e])[:, None] * g[None, :], m2, axis=1)
    assert np.allclose(got, want, atol=1e-12)


@pytest.mark.parametrize("order", [2, 3, 4])
@pytest.mark.parametrize("loop", [False, True])
def test_hex_loft_fn_nodes_are_its_section_lattice_revolved(order, loop):
    """The same gate for a **curved** sweep, and for the other node path.

    Revolving is a rigid motion, so the tensor product still holds exactly -- hex
    ``i*M + q`` at GLL level ``k`` is section quad ``q``'s block turned by that level's
    own angle.  ``loft_fn`` hands ``loft`` genuine intermediate sections, so this walks
    the ``sweep_nodes`` gather rather than the straight-sweep blend; between them the two
    tests cover every source a shared edge or face node can have."""
    from conftest import curved

    from nekmeshpy import hexmesh
    from nekmeshpy.model import affine
    from nekmeshpy.model.stations import refined_lattice

    R = 4.0
    disc = quadmesh.rotate(_section(order), np.pi / 2, (1.0, 0.0, 0.0))
    sec = quadmesh.translate(disc, (R, 0.0, 0.0))
    turn = 2.0 * np.pi if loop else 0.7

    def f(t):
        return quadmesh.rotate(sec, turn * float(t))

    fr = np.linspace(0.0, 1.0, 5)
    blk = hexmesh.loft_fn(f, fr, order=order, loop=loop)
    t = refined_lattice(fr, order)
    nz = fr.shape[0] - 1

    row, M = order + 1, sec.n_quads
    m2 = row * row
    sb = _inplane(sec, blk)                             # (M, m2, 3) at angle 0
    got = curved(blk)
    assert got.shape == (nz * M, m2 * row, 3)
    want = np.empty_like(got)
    for e in range(nz * M):
        i, q = divmod(e, M)
        for k in range(row):
            rot, _ = affine.rotation(turn * float(t[i * order + k]))
            want[e, k * m2:(k + 1) * m2, :] = sb[q] @ rot.T
    assert np.allclose(got, want, atol=1e-12)


@pytest.mark.parametrize("down", [False, True])
@pytest.mark.parametrize("loop", [False, True])
def test_hex_loft_entity_partition_matches_a_dedup(loop, down):
    _, _, blk = _shell(1, nz=4, loop=loop, down=down)
    hexes = np.asarray(blk.hexes, np.int64)
    want_e, want_i, _ = conform.unique_edges(hexes, 3)
    # the block's *stored* tables, against a fresh dedup of its own corners
    assert blk.edges.shape[0] == want_e.shape[0]
    assert np.array_equal(np.sort(blk.edges[blk._elem_edges], axis=2),
                          np.sort(want_e[want_i], axis=2))
    assert blk.quads.n_quads == conform.canonical_faces(hexes)[0].shape[0]


@pytest.mark.parametrize("loop", [False, True])
def test_hex_loft_leaves_no_entity_an_element_does_not_carry(loop):
    """A section point no quad touches -- and, with it, the section edge no quad
    references -- must spawn neither a dangling rung nor an unreferenced face."""
    from nekmeshpy import LineMesh, QuadMesh, hexmesh
    pts = np.array([[0.0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [9, 9, 0]])
    lm = LineMesh(pts, [[0, 1], [1, 2], [2, 3], [3, 0], [0, 4]])
    sec = QuadMesh(lm, [[0, 1, 2, 3]], np.zeros((1, 4), bool))
    slices = [quadmesh.translate(sec, (0.0, 0.0, float(z)))
              for z in _levels(3, loop)]
    blk = hexmesh.loft(slices, loop=loop)
    assert np.unique(blk._elem_edges).shape[0] == blk.edges.shape[0]
    assert np.unique(blk.hex).shape[0] == blk.quads.n_quads
    assert blk.edges.shape[0] == conform.unique_edges(
        np.asarray(blk.hexes, np.int64), 3)[0].shape[0]
