"""``model.sweep`` -- the product index space every ``loft`` numbers by.

The closed forms replace a dedup pass, so what has to be pinned is that they *agree*
with one: the same entity partition ``conform.unique_edges`` would find, and the same
incidence per element.  Numbering is not pinned -- nothing here compares ids.

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
from nekmeshpy.model.sweep import Sweep


def _profiles(n_prof, loop, order=1):
    ring = linemesh.circle(1.0, 8, order=order)
    out = [linemesh.translate(ring, (0.0, 0.0, z))
           for z in np.linspace(0.0, 1.0, n_prof + (1 if loop else 0))]
    return out[:n_prof]


# -- the index space itself ---------------------------------------------------
def test_nxt_is_where_loop_lives():
    assert Sweep(4, 3, 8).nxt.tolist() == [1, 2, 3]          # open: no wrap
    assert Sweep(4, 4, 8, loop=True).nxt.tolist() == [1, 2, 3, 0]
    assert Sweep(1, 0, 8).nxt.tolist() == []                 # one slice, no layer


def test_the_two_families_never_collide():
    """Carried and swept ids partition ``0 .. total-1`` exactly once."""
    s = Sweep(4, 3, 8)
    nc, ns = 5, 6
    ids = np.concatenate([
        s.carried(np.repeat(np.arange(4), nc), np.tile(np.arange(nc), 4), nc),
        s.swept(np.repeat(np.arange(3), ns), np.tile(np.arange(ns), 3), nc, ns)])
    assert sorted(ids.tolist()) == list(range(s.total(nc, ns)))


def test_elements_are_layer_major():
    i, k, j = Sweep(3, 2, 8).elements(4)
    assert i.tolist() == [0, 0, 0, 0, 1, 1, 1, 1]        # e = layer * n + k
    assert k.tolist() == [0, 1, 2, 3, 0, 1, 2, 3]
    assert j.tolist() == [1, 1, 1, 1, 2, 2, 2, 2]


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
    from nekmeshpy.linemesh.assemble import _refined_lattice
    from nekmeshpy.model import affine

    R = 4.0
    disc = quadmesh.rotate(_section(order), np.pi / 2, (1.0, 0.0, 0.0))
    sec = quadmesh.translate(disc, (R, 0.0, 0.0))
    turn = 2.0 * np.pi if loop else 0.7

    def f(t):
        return quadmesh.rotate(sec, turn * float(t))

    fr = np.linspace(0.0, 1.0, 5)
    blk = hexmesh.loft_fn(f, fr, order=order, loop=loop)
    t = _refined_lattice(fr, order)
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
