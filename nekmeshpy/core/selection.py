"""Resolving a ``select`` / ``remove`` argument to a mask over a mesh's elements."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Union

import numpy as np

from .._typing import BoolArray, IntArray, StrArray
from ..tags import Tags

__all__ = ["Selection", "mask_for_selection"]

#: A ``select`` / ``remove`` argument: one tag name, several tag names, a boolean mask
#: over the elements, or element ids.
Selection = Union[str, Sequence[str], StrArray, BoolArray, IntArray,
                 Sequence[int]]


def mask_for_selection(which: Selection, tags: Tags) -> BoolArray:
    """The ``(n_elements,)`` boolean mask a ``select`` / ``remove`` argument names.

    ``which`` is a tag name or a list / array of them (resolved by
    :meth:`Tags.flatnonzero`; a name absent from the mesh's vocabulary is an error,
    since a silent empty selection is almost always a typo), a ready
    ``(n_elements,)`` boolean mask, or an array of element ids."""
    arr = np.asarray(which)
    n_elements = tags.size
    ids: IntArray
    if arr.dtype.kind == "U": # if str, check tags
        unknown = [n for n in np.unique(arr).tolist() if n not in tags.unique()]
        if unknown:
            raise ValueError(
                "no element carries the tag %s; this mesh has %s"
                % (", ".join(repr(u) for u in unknown),
                   tags.unique() or "no tagged elements"))
        ids = tags.flatnonzero(arr)
    elif arr.size == 0:
        ids = np.zeros(0, dtype=np.int64)
    elif arr.dtype == bool:
        if arr.ndim != 1 or arr.shape[0] != n_elements:
            raise ValueError("a boolean mask must cover all %d elements, got shape %s"
                             % (n_elements, arr.shape))
        return arr
    elif np.issubdtype(arr.dtype, np.integer):
        ids = arr.reshape(-1).astype(np.int64)
        if int(ids.min()) < 0 or int(ids.max()) >= n_elements:
            raise ValueError("element ids must lie in [0, %d); got [%d, %d]"
                             % (n_elements, int(ids.min()), int(ids.max())))
    else:
        raise TypeError(
            "select by tag names, a (%d,) boolean mask, or an array of element "
            "ids; got a %s array" % (n_elements, arr.dtype))
    out: BoolArray = np.zeros(n_elements, dtype=bool)
    out[ids] = True
    return out
