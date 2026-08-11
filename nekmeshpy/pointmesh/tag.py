"""Vocabulary-only ``PointMesh`` operations: rename the tags, touch nothing else."""

from __future__ import annotations

from collections.abc import Mapping

from .pointmesh import PointMesh


def retag_element(mesh: PointMesh, mapping: Mapping[str, str]) -> PointMesh:
    """The same mesh with its ``element_tags`` renamed through ``mapping``.

    A tag the map does not name is left alone; a tag renamed to ``NO_TAG`` becomes
    untagged. The map applies simultaneously, so ``{"a": "b", "b": "a"}`` swaps the
    two, and two keys may share an image. A key that names no tag on this mesh raises.

    These are the tags a :class:`LineMesh <nekmeshpy.linemesh.linemesh.LineMesh>` built on this
    point set reads as its own point tags, so this is what
    :func:`linemesh.retag_point <nekmeshpy.linemesh.tag.retag_point>` calls through to.
    """
    return PointMesh(mesh.points,
                     mesh.element_tags.renamed(mapping, "pointmesh.retag_element"))


__all__ = [
    "retag_element",
]
