"""Combining tag tables that meet on one shared entity."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .._typing import BoolArray, IntArray, StrArray
from ._arrays import _str_array
from .tags import Tags

__all__ = ["weld"]


def weld(tables: Sequence[Tags]) -> Tags:
    """Combine tables that may name the **same** element -- what a ``merge`` produces
    once its weld has carried two blocks' tags onto one shared entity.

    Plain :meth:`Tags.concatenate` cannot do this: it would hand the constructor two
    rows for one id, which is rejected. Here the duplicate is expected and meaningful,
    so it is resolved rather than refused -- but only when the two agree. Two different
    non-empty names on one entity is the contradiction the shared-entity storage exists
    to rule out, and there is no honest way to pick between them, so it raises.

    An entity named by one side and left untagged by the other simply takes the name:
    that is the ordinary case of a tagged block welding onto an untagged neighbour.

    Every table must be over the same element count, which the result keeps."""
    if not tables:
        return Tags.empty(0)
    size = tables[0].size
    if any(t.size != size for t in tables):
        raise ValueError("weld: tables are over different element counts %s"
                         % sorted({t.size for t in tables}))
    live = [t for t in tables if len(t)]
    if not live:
        return Tags.empty(size)
    # concatenated as raw columns rather than through ``concatenate``: the duplicate ids
    # this resolves are exactly what the constructor is there to reject
    raw_ids: IntArray = np.concatenate([t.ids for t in live])
    raw_names: StrArray = _str_array(np.concatenate([t.tags for t in live]))
    order = np.argsort(raw_ids, kind="stable")
    ids, names = raw_ids[order], raw_names[order]
    dup: BoolArray = np.zeros(ids.shape[0], dtype=bool)
    dup[1:] = ids[1:] == ids[:-1]
    if not dup.any():
        return Tags(ids, names, size)
    differs: BoolArray = np.zeros(ids.shape[0], dtype=bool)
    differs[1:] = names[1:] != names[:-1]
    clash = np.flatnonzero(dup & differs)
    if clash.size:
        i = int(clash[0])
        raise ValueError(
            "the weld puts two different names on one entity -- element %d is "
            "tagged both %r and %r. A shared entity carries one tag, so leave one of "
            "the two sides untagged, or give them the same name."
            % (int(ids[i]), str(names[i - 1]), str(names[i])))
    return Tags(ids[~dup], names[~dup], size)
