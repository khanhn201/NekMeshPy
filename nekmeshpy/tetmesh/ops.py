"""Volumetric algorithms on a ``TetMesh`` as free functions. All indices 0-based.

The gmsh dependency lives entirely inside :func:`tet_mesh`, imported at call time --
mirrors the toolkit-wide rule that ``gmsh`` is an optional extra (``pip install
.[mesh]``), never a hard import of any module here.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.spatial import cKDTree

from .._typing import BoolArray, FloatArray, IntArray, PointArray
from ..trimesh import TriMesh
from .tetmesh import TetMesh


# -- surface -> volume ----------------------------------------------------
def cap_surface(
    surf: TriMesh, loops: Sequence[IntArray],
) -> tuple[TriMesh, list[IntArray]]:
    """Close an open surface by fan-triangulating each boundary loop ``loops`` (point
    ids into ``surf``) from its centroid.  Returns ``(capped, cap_node_sets)``; the
    wall vertices keep their indices."""
    V, F = surf.points, surf.tris
    caps: list[PointArray] = []
    faces: list[IntArray] = [F]
    n0 = V.shape[0]
    sets: list[IntArray] = []
    for ring in loops:
        ring = np.asarray(ring, dtype=np.int64).ravel()
        ci = n0 + len(caps)
        caps.append(V[ring].mean(axis=0))
        faces.append(np.column_stack([np.full(ring.size, ci), ring, np.roll(ring, -1)]))
        sets.append(np.concatenate([ring, [ci]]))
    capped = TriMesh(np.vstack([V, np.array(caps)]), np.vstack(faces))
    return capped, sets


def tet_mesh(
    capped: TriMesh, near: float, far: float, ramp: float,
    centre: PointArray | Sequence[float],
) -> TetMesh:
    """Tet-mesh the closed surface ``capped`` with gmsh, keeping its nodes in place.

    Element size is **graded** by a background field: ``near`` at ``centre``, growing
    to ``far`` over ``ramp``.

    gmsh cannot refine a *discrete* surface, so the triangles handed in set the floor on
    the size and shape of every tet at the wall -- the surface has to be good before it
    gets here.  Reparametrizing it (``classifySurfaces`` + ``createGeometry``) so gmsh
    could remesh it does not work on a smooth closed tube with no sharp edges to split
    on: it comes out either as one unparametrizable patch or, at an angle small enough
    to succeed, as hundreds of them, every boundary of which would then be forced into
    the mesh as a feature curve.

    The three ``MeshSizeFrom*`` switches are turned off so the field is the *only*
    thing setting size; otherwise the surface mesh's own spacing is extended into the
    volume and quietly overrides it.

    **gmsh does not tetrahedralize the same way twice** -- not across machines, not run
    to run on one.  A caller freezing a result against this should freeze the tets
    themselves, not just the knobs that produced them."""
    import gmsh
    V, F = capped.points, capped.tris
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("vol")
        tag = gmsh.model.addDiscreteEntity(2)
        gmsh.model.mesh.addNodes(2, tag, np.arange(1, V.shape[0] + 1), V.ravel())
        gmsh.model.mesh.addElementsByType(tag, 2, [], (F + 1).ravel())
        gmsh.model.geo.addVolume([gmsh.model.geo.addSurfaceLoop([tag])])
        gmsh.model.geo.synchronize()
        cx, cy, cz = (float(c) for c in centre)
        # Distance + Threshold, not MathEval: the same linear ramp from ``near`` at the
        # junction to ``far`` at ``ramp`` away, but built in.  MathEval goes through
        # gmsh's expression parser once per candidate point, which comes to dominate the
        # run as soon as the junction is refined -- refining it is what this field is for.
        pid = gmsh.model.geo.addPoint(cx, cy, cz)
        gmsh.model.geo.synchronize()
        dist = gmsh.model.mesh.field.add("Distance")
        gmsh.model.mesh.field.setNumbers(dist, "PointsList", [pid])
        fid = gmsh.model.mesh.field.add("Threshold")
        gmsh.model.mesh.field.setNumber(fid, "InField", dist)
        gmsh.model.mesh.field.setNumber(fid, "SizeMin", near)
        gmsh.model.mesh.field.setNumber(fid, "SizeMax", far)
        gmsh.model.mesh.field.setNumber(fid, "DistMin", 0.0)
        gmsh.model.mesh.field.setNumber(fid, "DistMax", ramp)
        gmsh.model.mesh.field.setAsBackgroundMesh(fid)
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
        # HXT, not the serial Delaunay: with a graded field over a discrete surface the
        # latter grinds through thousands of refinement iterations rejecting every
        # candidate point ("0 nodes created"), and never finishes at junction sizes worth
        # asking for.  HXT is parallel and does not stall.
        gmsh.option.setNumber("Mesh.Algorithm3D", 10)
        gmsh.option.setNumber("General.NumThreads", 0)
        # HXT is fast but leaves slivers; its own optimizers are what bring the worst
        # aspect ratios down, and Netgen's is the one that actually reshapes elements
        # rather than only flipping them.  Both are cheap next to generation.
        gmsh.option.setNumber("Mesh.Optimize", 1)
        gmsh.option.setNumber("Mesh.OptimizeNetgen", 1)
        # the threshold is the quality below which an element gets worked on.  Chasing
        # the last few slivers with 0.9 and eight passes costs minutes and moves the
        # *worst* aspect ratio only; the median barely notices, so keep it modest.
        gmsh.option.setNumber("Mesh.OptimizeThreshold", 0.5)
        gmsh.model.mesh.generate(3)
        for _ in range(2):
            try:
                gmsh.model.mesh.optimize("Netgen")
            except Exception:
                break
            gmsh.model.mesh.optimize("")
        nt, nc, _ = gmsh.model.mesh.getNodes()
        _, tc = gmsh.model.mesh.getElementsByType(4)
    finally:
        gmsh.finalize()
    # Find the surface nodes by *position* and put them first, in the order they came in.
    # Their tags do not survive -- gmsh renumbers on generate, and any entity the model
    # gains shifts things further -- but their positions must, since the whole point of
    # handing gmsh a discrete surface is that it fills the volume without moving it.  The
    # check below enforces that rather than assuming it.
    coords = nc.reshape(-1, 3)
    nv = V.shape[0]
    d, row = cKDTree(coords).query(V)
    if d.max() > 1e-9 or np.unique(row).size != nv:
        raise RuntimeError("tet_mesh: gmsh moved the input surface nodes")
    rest = np.setdiff1d(np.arange(coords.shape[0]), row)
    P = np.vstack([coords[row], coords[rest]])
    order = np.empty(coords.shape[0], dtype=np.int64)
    order[row] = np.arange(nv)
    order[rest] = np.arange(nv, coords.shape[0])
    tag_row: IntArray = np.zeros(int(nt.max()) + 1, dtype=np.int64)
    tag_row[nt] = np.arange(nt.size)
    TET = order[tag_row[tc.reshape(-1, 4).astype(np.int64)]]
    # Drop nodes no tet references.  gmsh meshes every entity it owns, so the size
    # field's source point arrives as a node belonging to nothing -- and an unreferenced
    # node is an empty row in the Laplacian, which divides to NaN the moment anything
    # preconditions on the diagonal.  The wall keeps its indices: it is all referenced,
    # and it is all at the front.
    keep = np.zeros(P.shape[0], dtype=bool)
    keep[TET.ravel()] = True
    if not keep[:nv].all():
        raise RuntimeError("tet_mesh: %d surface nodes ended up in no tet"
                           % int((~keep[:nv]).sum()))
    if not keep.all():
        remap: IntArray = np.cumsum(keep) - 1
        P, TET = P[keep], remap[TET]
    return TetMesh(P, TET)


def cap_nodes(
    mesh: TetMesh, loops: Sequence[PointArray], tol: float = 1e-6,
) -> list[IntArray]:
    """Which boundary nodes of ``mesh`` lie on each opening ``loops`` (point
    *coordinates*, one array per opening).

    Remeshing throws away the numbering the caps used to be identified by, so find them
    geometrically instead.  Each opening is planar and nothing else in the domain lies
    in its plane, which makes the test both simple and unambiguous."""
    P = mesh.points
    bnd = np.unique(mesh.boundary_faces())
    out = []
    for R in loops:
        R = np.asarray(R, dtype=float)
        c = R.mean(axis=0)
        n = np.linalg.svd(R - c, full_matrices=False)[2][2]
        on = np.abs((P[bnd] - c) @ n) < tol
        if not on.any():
            raise RuntimeError("cap_nodes: found no boundary nodes on an opening")
        out.append(bnd[on])
    return out


# -- P1 conduction --------------------------------------------------------
def tet_gradients(mesh: TetMesh) -> tuple[FloatArray, FloatArray]:
    """``(grad (E,4,3), volume (E,))`` of the P1 basis on each tet."""
    p = mesh.points[mesh.tets]
    M = np.stack([p[:, 1] - p[:, 0], p[:, 2] - p[:, 0], p[:, 3] - p[:, 0]], axis=1)
    vol: FloatArray = np.abs(np.linalg.det(M)) / 6.0
    g = np.zeros((mesh.n_tets, 4, 3))
    g[:, 1:, :] = np.transpose(np.linalg.inv(M), (0, 2, 1))
    g[:, 0, :] = -g[:, 1:, :].sum(axis=1)
    return g, vol


def tet_laplacian(mesh: TetMesh) -> sp.csr_matrix:
    """P1 stiffness ``K_ij = vol * grad(lam_i) . grad(lam_j)`` (recomputed each call)
    -- the tet sibling of :func:`trimesh.ops.cotan_laplacian
    <nekmeshpy.trimesh.ops.cotan_laplacian>`."""
    g, vol = tet_gradients(mesh)
    K = vol[:, None, None] * np.einsum("eid,ejd->eij", g, g)
    TET = mesh.tets
    I = np.repeat(TET, 4, axis=1).ravel()
    J = np.tile(TET, (1, 4)).ravel()
    n = mesh.n_points
    return sp.coo_matrix((K.ravel(), (I, J)), shape=(n, n)).tocsr()


def _solve_dirichlet(L: sp.csr_matrix, n: int, nodes: IntArray, vals: FloatArray) -> FloatArray:
    """Laplace with Dirichlet ``vals`` at ``nodes`` against a precomputed stiffness
    ``L``; every other boundary is no-flux, the natural condition, so needs nothing
    said about it."""
    u = np.zeros(n)
    u[nodes] = vals
    free: BoolArray = np.ones(n, dtype=bool)
    free[nodes] = False
    f = np.flatnonzero(free)
    rhs = -np.asarray(L[f, :][:, nodes] @ u[nodes]).ravel()
    A = L[f, :][:, f].tocsr()
    if f.size <= 120_000:
        u[f] = spla.spsolve(A.tocsc(), rhs)
        return u
    # A direct factorization of a 3D Laplacian fills in badly once the junction is
    # refined; the matrix is symmetric positive definite, so switch to CG.  Jacobi is a
    # weak preconditioner but each iteration is one sparse matvec, and the field is
    # smooth enough that it converges in far less time than the factorization takes.
    d = A.diagonal()
    if not np.all(d > 0.0):
        raise RuntimeError("solve_dirichlet: %d rows have no diagonal -- the mesh has "
                           "nodes belonging to no element" % int((d <= 0.0).sum()))
    M = spla.LinearOperator(A.shape, lambda x: x / d)
    x, info = spla.cg(A, rhs, rtol=1e-12, maxiter=20_000, M=M)
    if info != 0:
        raise RuntimeError("conduction solve did not converge (cg info %d)" % info)
    u[f] = x
    return u


def solve_dirichlet(mesh: TetMesh, dpoints: IntArray, dvals: FloatArray) -> FloatArray:
    """Solve the Laplace system with Dirichlet values at ``dpoints``; returns an
    ``(n_points,)`` field.  Assembles :func:`tet_laplacian` fresh -- a caller solving
    several right-hand sides against the same mesh (like :func:`seam_fields`) should
    assemble once and go through :func:`tet_laplacian` directly instead."""
    L = tet_laplacian(mesh)
    return _solve_dirichlet(L, mesh.n_points, np.asarray(dpoints, dtype=np.int64).ravel(),
                            np.asarray(dvals, dtype=float).ravel())


def seam_fields(mesh: TetMesh, caps: Sequence[IntArray]) -> FloatArray:
    """The three conduction fields, one per leg: no-flux on that leg's cap, ``0`` and
    ``1`` on the other two, shifted to zero mean on the free cap.

    Exactly :func:`trimesh.ops.seam_fields <nekmeshpy.trimesh.ops.seam_fields>`, with
    the solve moved into the volume -- and the one stiffness assembly shared across all
    three solves, since :func:`tet_laplacian` is the expensive part at junction
    resolution."""
    n = mesh.n_points
    L = tet_laplacian(mesh)
    dvals = [[np.nan, 0.0, 1.0], [1.0, np.nan, 0.0], [0.0, 1.0, np.nan]]
    U = np.zeros((n, 3))
    for k in range(3):
        nodes = np.concatenate([caps[j] for j in range(3) if j != k])
        vals = np.concatenate([np.full(caps[j].size, dvals[k][j])
                               for j in range(3) if j != k])
        u = _solve_dirichlet(L, n, nodes, vals)
        U[:, k] = u - u[caps[k]].mean()
    return U


def leg_label(mesh: TetMesh, U: FloatArray) -> IntArray:
    """Which leg each tet of ``mesh`` belongs to (1, 2 or 3), from the sign pattern of
    the three seam fields ``U`` ``(n_points, 3)`` at its centroid -- the volumetric
    form of :func:`trimesh.ops.leg_label <nekmeshpy.trimesh.ops.leg_label>`, which
    reads a *node's* sign pattern since the surface cut retriangulates; a tet is not
    split, so its whole-element label comes from the centroid instead."""
    c = U[mesh.tets].mean(axis=1)
    a, b, d = c[:, 0], c[:, 1], c[:, 2]
    lab: IntArray = np.zeros(mesh.n_tets, dtype=np.int64)
    lab[(b > 0) & (d < 0)] = 1
    lab[(a < 0) & (d > 0)] = 2
    lab[(a > 0) & (b < 0)] = 3
    return lab
