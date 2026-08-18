"""Reader for Nek5000's ASCII ``.rea`` mesh format.

Only the geometry and the fluid boundary conditions are read -- the surrounding
parameters, curved-side, thermal, restart, and drive-force sections either don't bear
on a mesh's shape or (curved sides) aren't supported yet and are refused rather than
silently dropped. The row layout below (element/face packed into one field once
``NELGT`` reaches four digits, corners grouped ``X,Y,Z`` per half rather than per axis)
follows Nek5000's own ``core/reader_rea.f`` (``rdmesh``/``rdbdry``) exactly -- verified
against its upstream source, not reverse-engineered from one file.
"""

from __future__ import annotations

import logging

import numpy as np

from .._typing import IntArray, PointArray
from ..hexmesh import HexMesh
from ..hexmesh.tag import tag_faces

_log = logging.getLogger("nekmeshpy")

_FACE_POINTS = HexMesh.FACE_POINTS  # (6,4) local corner indices per Nek face


class ReaFormatError(ValueError):
    """The ``.rea`` file doesn't match the section this reader expected next."""


class _UnionFind:
    """Plain array union-find over the ``nel*8`` raw (unwelded) corner slots."""

    def __init__(self, n: int) -> None:
        self.parent: IntArray = np.arange(n, dtype=np.int64)

    def find(self, x: int) -> int:
        p = self.parent
        root = x
        while p[root] != root:
            root = int(p[root])
        while p[x] != root:
            p[x], x = root, int(p[x])
        return root

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def _find(lines: list[str], i: int, needle: str) -> int:
    """The index of the next line at or after ``i`` containing ``needle`` (case-
    insensitive), or a :class:`ReaFormatError` if the file runs out first."""
    upper = needle.upper()
    for j in range(i, len(lines)):
        if upper in lines[j].upper():
            return j
    raise ReaFormatError("rea: could not find a %r section" % needle)


def _floats(lines: list[str], i: int, n: int) -> tuple[list[float], int]:
    """``(values, next line index)``: ``n`` whitespace-separated floats, read from
    ``lines[i]`` onward -- Nek's own list-directed reads don't care about line breaks,
    so a short row is topped up from the next line rather than assumed short."""
    vals: list[float] = []
    while len(vals) < n:
        vals.extend(float(tok) for tok in lines[i].split())
        i += 1
    if len(vals) != n:
        raise ReaFormatError(
            "rea: expected %d values, found %d run together across lines" % (n, len(vals)))
    return vals, i


def _match_face(p1: PointArray, p2: PointArray) -> IntArray:
    """``(4,)``: for each of ``p1``'s 4 corners, the index into ``p2`` of the same
    physical corner -- found by nearest distance within just these two quartets, so it
    needs no global coincidence tolerance at all (a real hex face's 4 corners are never
    close enough to each other to confuse a nearest-corner match against its own
    partner face). Falls back to the optimal assignment only if that greedy match isn't
    already a bijection, which real geometry never triggers."""
    d = np.linalg.norm(p1[:, None, :] - p2[None, :, :], axis=2)
    match: IntArray = np.argmin(d, axis=1)
    if len(set(match.tolist())) == 4:
        return match
    from scipy.optimize import linear_sum_assignment
    _, cols = linear_sum_assignment(d)
    return cols.astype(np.int64)


def read_rea(path: str) -> HexMesh:
    """Read a 3-D Nek5000 ``.rea`` mesh into a linear (order-1) :class:`HexMesh
    <nekmeshpy.hexmesh.HexMesh>`.

    The ASCII format stores each element's 8 corners independently -- there is no
    shared point numbering to read directly, and floating-point round-trip through
    7-significant-digit text makes a coincidence-tolerance weld genuinely fragile (on
    the reference file this reader was built against, no single tolerance across four
    orders of magnitude was both watertight *and* conformal). Instead this reader uses
    the file's own ``"E"`` (element-to-element) boundary rows -- which name the exact
    connected element and face, not a inferred one -- and matches each connected face
    pair's 4 corners by nearest distance *within that pair alone*, which needs no
    tolerance because the search space is 4 points against 4, not the whole mesh.

    Named fluid rows (``"W"``, ``"v"``, ``"O"``, ...) are carried over as face tags via
    :func:`hexmesh.tag_faces <nekmeshpy.hexmesh.tag.tag_faces>`, keyed by the raw Nek
    code -- this reader does not rename them.

    Raises :class:`ReaFormatError` for anything outside that scope: a 2-D mesh, or a
    nonzero curved-side count (this reader reconstructs straight-sided hexes only, and
    silently dropping curvature would hand back the wrong geometry rather than an
    approximate one)."""
    with open(path, "r") as f:
        raw = f.readlines()

    i = _find(raw, 0, "**MESH DATA**") + 1
    nel, ndim, _nelv = (int(float(tok)) for tok in raw[i].split()[:3])
    i += 1
    if ndim != 3:
        raise ReaFormatError(
            "rea: read_rea only builds a HexMesh from a 3-D run, got NDIM=%d" % ndim)

    corners: PointArray = np.empty((nel, 8, 3), dtype=float)
    for e in range(nel):
        i += 1                                     # "ELEMENT n [tag] GROUP g" header
        x1, i = _floats(raw, i, 4)
        y1, i = _floats(raw, i, 4)
        z1, i = _floats(raw, i, 4)
        x2, i = _floats(raw, i, 4)
        y2, i = _floats(raw, i, 4)
        z2, i = _floats(raw, i, 4)
        corners[e, 0:4, 0] = x1
        corners[e, 0:4, 1] = y1
        corners[e, 0:4, 2] = z1
        corners[e, 4:8, 0] = x2
        corners[e, 4:8, 1] = y2
        corners[e, 4:8, 2] = z2

    i = _find(raw, i, "CURVED SIDE DATA") + 1
    n_curved = int(float(raw[i].split()[0]))
    if n_curved:
        raise ReaFormatError(
            "rea: %d curved side(s) present; read_rea reconstructs straight-sided hexes "
            "only, and dropping curvature would silently hand back the wrong geometry "
            "rather than a linear approximation of it" % n_curved)
    i += 1

    i = _find(raw, i, "BOUNDARY CONDITIONS") + 1
    internal: list[tuple[int, int, int, int]] = []       # (e1, f1, e2, f2), 0-based
    named_faces: list[tuple[int, int]] = []
    named_tags: list[str] = []
    if "NO " not in raw[i].upper():
        i += 1                                     # "***** FLUID BOUNDARY CONDITIONS *****"
        for _ in range(6 * nel):
            tok = raw[i].split()
            i += 1
            code = tok[0].strip()
            if not code:
                continue
            iel, iface = divmod(int(float(tok[1])), 10)
            elem, side = iel - 1, iface - 1
            if code == "E":
                elem2, side2 = int(float(tok[2])) - 1, int(float(tok[3])) - 1
                internal.append((elem, side, elem2, side2))
            else:
                named_faces.append((elem, side))
                named_tags.append(code)

    uf = _UnionFind(nel * 8)
    for e1, f1, e2, f2 in internal:
        idx1, idx2 = _FACE_POINTS[f1], _FACE_POINTS[f2]
        match = _match_face(corners[e1, idx1, :], corners[e2, idx2, :])
        for a, b in zip(idx1, idx2[match]):
            uf.union(e1 * 8 + a, e2 * 8 + b)

    roots = np.array([uf.find(k) for k in range(nel * 8)], dtype=np.int64)
    _, welded_id = np.unique(roots, return_inverse=True)
    n_points = int(welded_id.max()) + 1 if welded_id.size else 0
    flat: PointArray = corners.reshape(nel * 8, 3)
    sums = np.zeros((n_points, 3))
    counts = np.zeros(n_points)
    np.add.at(sums, welded_id, flat)
    np.add.at(counts, welded_id, 1.0)
    points = sums / counts[:, None]
    hexes = welded_id.reshape(nel, 8)

    mesh = HexMesh.from_corners(points, hexes)
    if named_faces:
        face_ids = np.array([mesh.hexes[elem, side] for elem, side in named_faces],
                           dtype=np.int64)
        mesh = tag_faces(mesh, face_ids, named_tags)

    _log.info("rea: read %d hexes, %d welded points (%d raw) from %s",
             nel, n_points, flat.shape[0], path)
    return mesh


__all__ = ["ReaFormatError", "read_rea"]
