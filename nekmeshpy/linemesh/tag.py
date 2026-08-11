"""Vocabulary-only ``LineMesh`` operations: rename the tags, touch nothing else.

Separate from :mod:`morph <nekmeshpy.linemesh.morph>`, which is the delta-0 rung for
operations on the *geometry*. These change neither coordinates nor connectivity nor
numbering -- only what the two tag tables call things -- so a caller reading the
sibling list can tell at a glance that a retag cannot have moved a node.
"""

from __future__ import annotations

from collections.abc import Mapping

from ..pointmesh import tag as pointmesh
from .linemesh import LineMesh


def retag_element(mesh: LineMesh, mapping: Mapping[str, str]) -> LineMesh:
    """The same mesh with its ``element_tags`` renamed through ``mapping``.

    A tag the map does not name is left alone; a tag renamed to ``""`` becomes
    untagged. The map applies simultaneously, so ``{"a": "b", "b": "a"}`` swaps the
    two, and two keys may share an image, which merges those regions. A key that
    names no tag on this mesh raises -- a rename matching nothing is a typo, and a
    mis-named region is not visible again until the solver reads it."""
    return LineMesh(mesh.vertices, mesh.lines, mesh.interior,
                    mesh.element_tags.renamed(mapping, "linemesh.retag_element"))


def retag_point(mesh: LineMesh, mapping: Mapping[str, str]) -> LineMesh:
    """The same mesh with its ``point_tags`` renamed through ``mapping``, rows kept in
    stored order (see :func:`retag_element` for the map's own rules).

    Renaming a tag to ``""`` **drops** its rows rather than storing an empty name: a
    side-tag table is a named subset, so leaving it is leaving the table. That is how
    a boundary name that has stopped meaning anything is retired."""
    return LineMesh(pointmesh.retag_element(mesh.vertices, mapping),
                    mesh.lines, mesh.interior, mesh.element_tags)


__all__ = [
    "retag_element",
    "retag_point",
]
