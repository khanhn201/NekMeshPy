"""Vocabulary-only ``HexMesh`` operations: rename the tags, touch nothing else.

Separate from :mod:`morph <nekmeshpy.hexmesh.morph>`, which is the delta-0 rung for
operations on the *geometry*. These change neither coordinates nor connectivity nor
numbering -- only what the two tag tables call things -- so a caller reading the
sibling list can tell at a glance that a retag cannot have moved a node.
"""

from __future__ import annotations

from collections.abc import Mapping

from ..quadmesh import tag as quadmesh
from .hexmesh import HexMesh


def retag_element(mesh: HexMesh, mapping: Mapping[str, str]) -> HexMesh:
    """The same mesh with its ``element_tags`` renamed through ``mapping``.

    A tag the map does not name is left alone; a tag renamed to ``NO_TAG`` becomes
    untagged. The map applies simultaneously, so ``{"a": "b", "b": "a"}`` swaps the
    two, and two keys may share an image, which merges those regions. A key that
    names no tag on this mesh raises -- a rename matching nothing is a typo, and a
    mis-named region is not visible again until the solver reads it.

    The region vocabulary and the boundary-condition vocabulary are different tables,
    so renaming one never disturbs the other even where they share a word."""
    return HexMesh(mesh.quads, mesh.hex, mesh.face_orient, mesh.interior,
                   mesh.element_tags.renamed(mapping, "hexmesh.retag_element"))


def retag_face(mesh: HexMesh, mapping: Mapping[str, str]) -> HexMesh:
    """The same mesh with its ``face_tags`` renamed through ``mapping``, rows kept in
    stored order (see :func:`retag_element` for the map's own rules).

    Order matters here beyond tidiness: ``.re2`` writes boundary rows in it, and
    ``.vtu`` gives a node touched by several rows the last one's tag.

    Renaming a tag to ``NO_TAG`` **drops** its rows rather than storing an empty name:
    a side-tag table is a named subset of the boundary, so leaving it is leaving the
    table. That is the way to retire a name that has stopped meaning anything --
    an ``"inlet"`` welded shut into an interior plane, which would otherwise export
    as a boundary condition on a face that is no longer on the boundary::

        mesh = hexmesh.retag_face(mesh, {"inlet": "", "outlet": ""})

    :func:`tag_report <nekmeshpy.hexmesh.query.tag_report>` is what finds those."""
    return HexMesh(quadmesh.retag_element(mesh.quads, mapping),
                   mesh.hex, mesh.face_orient, mesh.interior, mesh.element_tags)


__all__ = [
    "retag_element",
    "retag_face",
]
