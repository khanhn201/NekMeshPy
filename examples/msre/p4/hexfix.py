"""Last-resort repair of curved-only folded hexes: the edges of every curved-invalid hex go
straight between their corners (GLL spacing); every face touching a changed edge gets Coons
interior nodes, every hex touching a changed face gets transfinite interior nodes.  Corners and
topology are untouched (so .re2 and conformity are unchanged); repeated until clean."""
import numpy as np
from nekmeshpy import hexmesh, quadmesh, HexMesh
from nekmeshpy.quadmesh import QuadMesh
from nekmeshpy.linemesh import LineMesh
from nekmeshpy.core import conform
from nekmeshpy.core.fields import gll_nodes
from nekmeshpy.quadmesh.query import element_blocks as qblocks
from nekmeshpy.hexmesh.query import element_blocks as hblocks


def _min_sj(m):
    s = np.asarray(hexmesh.scaled_jacobian(m))
    return s.min(1) if s.ndim > 1 else s


def coons2(B, g):
    """(K,n,n,3) quad blocks -> transfinite (Coons) blocks from their boundary rows."""
    u = g[None, None, :, None]; v = g[None, :, None, None]
    return ((1 - u) * B[:, :, :1] + u * B[:, :, -1:] + (1 - v) * B[:, :1, :] + v * B[:, -1:, :]
            - ((1 - u) * (1 - v) * B[:, :1, :1] + u * (1 - v) * B[:, :1, -1:] + (1 - u) * v * B[:, -1:, :1] + u * v * B[:, -1:, -1:]))


def coons3(B, g):
    """(K,n,n,n,3) hex blocks (axes a,b,c) -> transfinite interpolation of their 6 faces."""
    a = g[None, :, None, None, None]; b = g[None, None, :, None, None]; c = g[None, None, None, :, None]
    A0, A1 = B[:, :1], B[:, -1:]; B0, B1 = B[:, :, :1], B[:, :, -1:]; C0, C1 = B[:, :, :, :1], B[:, :, :, -1:]
    F = (1 - a) * A0 + a * A1 + (1 - b) * B0 + b * B1 + (1 - c) * C0 + c * C1
    E = ((1 - a) * (1 - b) * B[:, :1, :1] + a * (1 - b) * B[:, -1:, :1] + (1 - a) * b * B[:, :1, -1:] + a * b * B[:, -1:, -1:]
         + (1 - a) * (1 - c) * B[:, :1, :, :1] + a * (1 - c) * B[:, -1:, :, :1] + (1 - a) * c * B[:, :1, :, -1:] + a * c * B[:, -1:, :, -1:]
         + (1 - b) * (1 - c) * B[:, :, :1, :1] + b * (1 - c) * B[:, :, -1:, :1] + (1 - b) * c * B[:, :, :1, -1:] + b * c * B[:, :, -1:, -1:])
    Cc = sum(((a if i else 1 - a) * (b if j else 1 - b) * (c if k else 1 - c)) * B[:, [-1 if i else 0]][:, :, [-1 if j else 0]][:, :, :, [-1 if k else 0]]
             for i in (0, 1) for j in (0, 1) for k in (0, 1))
    return F - E + Cc


def straighten_bad(mesh, thr=0.0, rounds=8, verbose=True):
    order = mesh.order
    if order < 2:
        return mesh
    n = order + 1
    g = gll_nodes(order)
    for it in range(rounds):
        sj = _min_sj(mesh)
        bad = np.where(sj <= thr)[0]
        if verbose:
            print("  hexfix round %d: %d curved-invalid hexes (min %.3f)" % (it, len(bad), sj.min()), flush=True)
        if not len(bad):
            break
        qm = mesh.quad_mesh; lm = qm.line_mesh
        lines = np.asarray(lm.lines); pts = np.asarray(lm.points)
        E = np.unique(np.asarray(mesh._elem_edges)[bad].ravel())
        lint = np.asarray(lm.interior).copy()
        lint[E] = pts[lines[E, 0]][:, None] + g[1:-1][None, :, None] * (pts[lines[E, 1]] - pts[lines[E, 0]])[:, None]
        lm2 = LineMesh(lm.point_mesh, lines, lint, lm.element_tags)
        q2 = QuadMesh(lm2, qm.quads, qm.orient, qm.interior, qm.element_tags)
        Fa = np.where(np.isin(np.asarray(qm.quads), E).any(1))[0]
        QB = qblocks(q2)[Fa].reshape(len(Fa), n, n, 3)
        fint = np.asarray(qm.interior).copy()
        fint[Fa] = coons2(QB, g).reshape(len(Fa), n * n, 3)[:, conform._interior_slots(2, order), :]
        q3 = QuadMesh(lm2, qm.quads, qm.orient, fint, qm.element_tags)
        m2 = HexMesh(q3, mesh.hexes, mesh.orient, mesh.interior, mesh.element_tags)
        Ha = np.where(np.isin(np.asarray(mesh.hexes), Fa).any(1))[0]
        HB = hblocks(m2)[Ha].reshape(len(Ha), n, n, n, 3)
        hint = np.asarray(mesh.interior).copy()
        hint[Ha] = coons3(HB, g).reshape(len(Ha), n ** 3, 3)[:, conform._interior_slots(3, order), :]
        mesh = HexMesh(q3, mesh.hexes, mesh.orient, hint, mesh.element_tags)
    return mesh
