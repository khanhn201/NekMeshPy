"""Vocabulary-only ``QuadMesh`` operations: rename the tags, touch nothing else.

Separate from :mod:`morph <nekmeshpy.quadmesh.morph>`, which is the delta-0 rung for
operations on the *geometry*. These change neither coordinates nor connectivity nor
numbering -- only what the two tag tables call things -- so a caller reading the
sibling list can tell at a glance that a retag cannot have moved a node.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from .._typing import IntArray, StrArray
from ..core.tags import ElementTags
from ..linemesh import LineMesh
from ..linemesh import tag as linemesh
from .quadmesh import QuadMesh


def retag_element(mesh: QuadMesh, mapping: Mapping[str, str]) -> QuadMesh:
    """The same mesh with its ``element_tags`` renamed through ``mapping``.

    A tag the map does not name is left alone; a tag renamed to ``NO_TAG`` becomes
    untagged. The map applies simultaneously, so ``{"a": "b", "b": "a"}`` swaps the
    two, and two keys may share an image, which merges those regions. A key that
    names no tag on this mesh raises -- a rename matching nothing is a typo, and a
    mis-named region is not visible again until the solver reads it."""
    return QuadMesh(mesh.line_mesh, mesh.quads, mesh.orient, mesh.interior,
                    mesh.element_tags.renamed(mapping, "quadmesh.retag_element"))


def retag_edge(mesh: QuadMesh, mapping: Mapping[str, str]) -> QuadMesh:
    """The same mesh with its ``edge_tags`` renamed through ``mapping``, rows kept in
    stored order (see :func:`retag_element` for the map's own rules).

    Renaming a tag to ``NO_TAG`` **drops** its rows rather than storing an empty name:
    a side-tag table is a named subset, so leaving it is leaving the table. That is
    how a boundary name that has stopped meaning anything is retired."""
    return QuadMesh(linemesh.retag_element(mesh.line_mesh, mapping),
                    mesh.quads, mesh.orient, mesh.interior, mesh.element_tags)


def tag_edges(mesh: QuadMesh, rows: IntArray,
              tags: Sequence[str] | StrArray) -> QuadMesh:
    """The same mesh with the shared edge each ``(quad, side)`` row names given its tag.

    The authoring bridge. A factory knows its geometry element-locally -- "side 1 of the
    outermost ring is the wall" -- while storage is one tag per shared edge, so the row
    is resolved through ``mesh.quads[quad, side - 1]`` and the tag written there. What
    changes is only where it lands: two rows that name the same edge from either side of
    it no longer become two entries that could disagree, and the later row wins.

    Rows tagged ``NO_TAG`` name nothing, so a partly-tagged row block can be handed over
    whole."""
    edge_of: IntArray = np.asarray(mesh.quads, dtype=np.int64)
    r: IntArray = np.asarray(rows, dtype=np.int64).reshape(-1, 2)
    names: StrArray = np.asarray(tags, dtype=np.str_).reshape(-1)
    named = np.asarray(mesh.edge_tags.dense(mesh.line_mesh.n_lines), dtype=object)
    for (q, side), nm in zip(r, names):
        if nm:
            named[edge_of[int(q), int(side) - 1]] = str(nm)
    return QuadMesh(
        LineMesh(mesh.line_mesh.point_mesh, mesh.line_mesh.lines, mesh.line_mesh.interior,
                 ElementTags.from_dense(np.asarray(named, dtype=np.str_))),
        mesh.quads, mesh.orient, mesh.interior, mesh.element_tags)


__all__ = [
    "retag_edge",
    "retag_element",
    "tag_edges",
]
