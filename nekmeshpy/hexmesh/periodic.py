"""Periodic face pairing: ``attach``'s stated correspondence, without the weld.

A Nek5000 ``'P  '`` boundary row names the element and face on the *other* side of the
domain, so a periodic boundary is not a name on a face -- it is a **correspondence**
between two face groups.  This module computes it and nothing else: no coordinate is
moved, no entity is fused, no numbering is invented, and the mesh comes back untouched.
What is produced is a table the ``.re2`` writer fills ``bc(1)`` / ``bc(2)`` from.

The pairing is :func:`attach <nekmeshpy.hexmesh.assemble.attach>`'s -- each of a's
corners to its nearest on b, proved by bijectivity rather than by a tolerance -- with one
difference that the whole design turns on: a periodic pair is **told its transform**.
``attach`` needs none because its two halves are meant to end up in the same place, so
nearest-neighbour reads the intended correspondence directly.  Periodic halves sit a
lattice vector apart and are often each other's rotations, where the nearest face across
the gap is emphatically not the periodic image.  Stating the map both makes the pairing
possible and makes it *checkable*: the worst residual after mapping is compared against
``conform.entity_tol`` -- the toolkit-wide coincidence tolerance -- so a mis-stated pitch is
an error quoting the number rather than a silently twisted mesh.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import NamedTuple, Union

import numpy as np
from scipy.spatial import cKDTree

from .._typing import IntArray, PointArray
from ..core import affine, conform
from .assemble import face_group
from .hexmesh import HexMesh
from .query import face_rows

#: What names one side of a pair: a face-tag name, or explicit face ids (see
#: :func:`tagged_faces <nekmeshpy.hexmesh.query.tagged_faces>`).
FaceGroup = Union[str, IntArray, Sequence[int]]


class Periodic(NamedTuple):
    """One stated periodic pair: which face group maps onto which, and by what map.

    ``tag_a`` / ``tag_b`` are face-tag names or explicit face id arrays, exactly as
    :class:`Seam <nekmeshpy.hexmesh.assemble.Seam>`'s are.  They must name **different**
    groups: a single tag covering both ends of a periodic cell cannot say which end is
    which, so split it at the point it is applied rather than guessing here from a
    coordinate.

    ``transform`` is an ``affine.Affine`` pair ``(matrix, offset)`` carrying ``a``
    **onto** ``b`` -- build it with :func:`affine.translation
    <nekmeshpy.core.affine.translation>` or :func:`affine.rotation
    <nekmeshpy.core.affine.rotation>`, the same constructors :func:`hexmesh.translate
    <nekmeshpy.hexmesh.morph.translate>` and :func:`hexmesh.rotate
    <nekmeshpy.hexmesh.morph.rotate>` take.  It is verified, not trusted.

    Two things it cannot verify:

    A group with a **rotational symmetry of its own** pairs bijectively at residual
    zero onto a cyclic shift of the intended correspondence -- the same trap
    ``attach`` carries, and a real one for a rotationally periodic cut through a
    bladed or lattice section, where nothing in the geometry distinguishes the two
    readings.  A translation-periodic pair is safe from it.

    And Nek identifies the two sides' values in the **global Cartesian frame**: it does
    not rotate a vector across a periodic face.  A rotational pair is therefore a
    periodicity for a scalar, and for a velocity field only where that field is
    invariant in the global frame.  A translation is unconditionally fine.
    """

    tag_a: FaceGroup
    tag_b: FaceGroup
    transform: affine.Affine


class PeriodicPairs(NamedTuple):
    """The resolved correspondence: what a ``'P  '`` boundary row needs.

    ``rows`` is ``(2K, 4)`` of ``[element, local face 1-6, partner element, partner
    local face]``, lexsorted by ``(element, face)``.  **Both** directions are in it --
    Nek wants a ``P`` row on each side of the pair, and emitting the two from one table
    is what makes them agree by construction rather than by a second pass.

    ``worst`` is the largest residual left by any spec's transform and ``tol`` what it
    was checked against, kept for a caller that wants to log how exactly the two halves
    corresponded."""

    rows: IntArray
    worst: float
    tol: float

    def partner_of(self) -> dict[tuple[int, int], tuple[int, int]]:
        """``{(element, face): (partner element, partner face)}`` -- the lookup a writer
        indexes its boundary rows by."""
        return {(int(e), int(f)): (int(pe), int(pf))
                for e, f, pe, pf in self.rows.tolist()}


def _paired_points(mesh: HexMesh, fa: IntArray, fb: IntArray,
                   transform: affine.Affine, who: str) -> tuple[IntArray, float]:
    """``(perm, worst)``: ``perm`` maps every corner id of group ``a`` to its partner on
    ``b`` (``-1`` elsewhere), and ``worst`` is the largest distance left after the map.

    The transform is applied to ``a`` first, so the nearest-neighbour query runs on two
    point sets meant to be *coincident* -- the same move :func:`bridge
    <nekmeshpy.hexmesh.lift.bridge>` makes when it centres both ends before pairing."""
    corners: IntArray = np.asarray(mesh.quad_mesh.corners, dtype=np.int64)
    pa: IntArray = np.unique(corners[fa])
    pb: IntArray = np.unique(corners[fb])
    if pa.size != pb.size:
        raise ValueError(
            "%s: the two groups are not the same surface -- %d faces / %d points on a, "
            "%d faces / %d points on b. Equal face counts with unequal point counts "
            "usually means the two sides are refined differently, which has no periodic "
            "correspondence." % (who, fa.size, pa.size, fb.size, pb.size))
    moved: PointArray = affine.apply(mesh.points[pa], *transform)
    dist, loc = cKDTree(mesh.points[pb]).query(moved)
    dup = loc.size - np.unique(loc).size
    if dup:
        raise ValueError(
            "%s: the pairing is not one-to-one -- %d of a's %d points share a nearest "
            "point on b after the transform, so the two patterns do not correspond one "
            "for one. Either the groups are the same surface meshed differently, or one "
            "of them is the wrong group." % (who, dup, loc.size))
    worst = float(np.max(dist)) if dist.size else 0.0
    tol = conform.entity_tol(mesh.points)
    if worst > tol:
        raise ValueError(
            "%s: the stated transform leaves a worst residual of %.3e, over the %.3e "
            "these points call coincident. The two groups are not related by this map -- "
            "check the offset or angle, and its sign (it must carry a onto b)."
            % (who, worst, tol))
    perm: IntArray = np.full(mesh.n_points, -1, dtype=np.int64)
    perm[pa] = pb[loc]
    return perm, worst


def _one_row_each(mesh: HexMesh, faces: IntArray, who: str,
                  side: str) -> IntArray:
    """``(K,2)`` ``[element, local face]`` for ``faces``, in the given order, insisting
    each face yields exactly one row -- which a boundary face does and an interior one
    does not.  :func:`face_group` has already rejected the interior ones, so a count
    other than one here means the incidence table itself is inconsistent."""
    rows, counts = face_rows(mesh, faces)
    if not np.all(counts == 1):
        bad = int(faces[int(np.flatnonzero(counts != 1)[0])])
        raise ValueError(
            "%s: %s's face %d is carried by %d hexes, not 1; a periodic face must be on "
            "the domain boundary" % (who, side, bad, int(counts[counts != 1][0])))
    return rows


def periodic_pairs(mesh: HexMesh, specs: Sequence[Periodic]) -> PeriodicPairs:
    """The face-to-face correspondence every :class:`Periodic` in ``specs`` states, as
    the ``(2K,4)`` table :func:`to_re2 <nekmeshpy.io.writer.to_re2>` writes ``'P  '``
    rows from.

    Nothing is welded and nothing is renamed: the two sides stay two distinct boundary
    faces with their own names, which is what a periodic boundary *is*.  Contrast
    :func:`attach <nekmeshpy.hexmesh.assemble.attach>`, which pairs the same way and then
    fuses the two into one interior face::

        pairs = hexmesh.periodic_pairs(mesh, [
            hexmesh.Periodic("inlet", "outlet", affine.translation([0, 0, LEAD])),
            hexmesh.Periodic("cut_lo", "cut_hi", affine.translation([0, 0, LEAD]))])

    Every face may be claimed once.  A face in two specs, or in both sides of one, would
    need two partners and get whichever spec ran last -- so it raises instead."""
    specs = list(specs)
    if not specs:
        return PeriodicPairs(np.zeros((0, 4), dtype=np.int64), 0.0,
                             conform.entity_tol(mesh.points))
    rows: list[IntArray] = []
    claimed: dict[int, str] = {}
    worst = 0.0
    for k, sp in enumerate(specs):
        who = "periodic_pairs: specs[%d]" % k
        fa = face_group(mesh, sp.tag_a, "tag_a", who)
        fb = face_group(mesh, sp.tag_b, "tag_b", who)
        if fa.size != fb.size:
            raise ValueError(
                "%s pairs groups of different face counts (%d and %d), so they cannot be "
                "the two ends of one periodic cell." % (who, fa.size, fb.size))
        if fa.size == 0:
            raise ValueError("%s names empty groups; there is nothing to pair" % who)
        for side, ids in (("tag_a", fa), ("tag_b", fb)):
            seen = np.array([f in claimed for f in ids.tolist()], dtype=bool)
            if seen.any():
                f = int(ids[np.argmax(seen)])
                raise ValueError(
                    "%s.%s: face %d is already claimed by %s. A face has one periodic "
                    "partner, so it may appear in one group of one spec -- naming the "
                    "same plane twice, or pairing a group with itself, gives it two."
                    % (who, side, f, claimed[f]))
            claimed.update({int(f): "%s.%s" % (who, side) for f in ids.tolist()})

        perm, w = _paired_points(mesh, fa, fb, sp.transform, who)
        worst = max(worst, w)
        corners: IntArray = np.asarray(mesh.quad_mesh.corners, dtype=np.int64)
        # the face map falls out of the point map: a's face, read through ``perm``, is
        # the id set of exactly one of b's faces -- ``locate_rows`` finds it and raises
        # when it is not there, so face bijectivity needs no check of its own
        into_b = conform.locate_rows(corners[fb], perm[corners[fa]],
                                     who=who, what="periodic face")
        ra = _one_row_each(mesh, fa, who, "tag_a")
        rb = _one_row_each(mesh, fb[into_b], who, "tag_b")
        rows.append(np.concatenate([np.hstack([ra, rb]), np.hstack([rb, ra])], axis=0))

    out: IntArray = np.concatenate(rows, axis=0)
    p = np.lexsort((out[:, 1], out[:, 0]))
    return PeriodicPairs(out[p], worst, conform.entity_tol(mesh.points))


__all__ = [
    "Periodic",
    "PeriodicPairs",
    "periodic_pairs",
]
