"""The sparse tag table every rung stores: :class:`Tags`."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, overload

import numpy as np

from .._typing import BoolArray, IntArray, StrArray
from ._arrays import _frozen, _str_array


@dataclass(frozen=True, eq=False)
class Tags:
    """A mesh's per-element region tags over ``size`` elements, stored **sparsely**:
    only tagged elements.

    ``size`` is the total element count the table is over, so a tagged id outside it
    is refused at construction and ``to_dense`` needs no length argument."""

    ids: IntArray
    tags: StrArray
    size: int

    def __post_init__(self) -> None:
        size = int(self.size)
        if size < 0:
            raise ValueError("Tags: size must be non-negative, got %d" % size)
        i = np.asarray(self.ids, dtype=np.int64).reshape(-1)
        t = _str_array(self.tags)
        if i.shape[0] != t.shape[0]:
            raise ValueError("Tags: ids (%d) and tags (%d) must have the same "
                             "length" % (i.shape[0], t.shape[0]))
        keep = t != ""
        if not np.all(keep):
            i, t = i[keep], t[keep]
        if i.shape[0]:
            p = np.argsort(i, kind="stable")
            i, t = i[p], t[p]
            if np.any(np.diff(i) == 0):
                dup = int(i[:-1][np.diff(i) == 0][0])
                raise ValueError("Tags: element %d is tagged more than once" % dup)
            if int(i[0]) < 0:
                raise ValueError("Tags: negative element id %d" % int(i[0]))
            if int(i[-1]) >= size:
                raise ValueError("Tags: element %d is tagged but the table is over "
                                 "only %d elements" % (int(i[-1]), size))
        object.__setattr__(self, "size", size)
        object.__setattr__(self, "ids", _frozen(i))
        object.__setattr__(self, "tags", _frozen(t))

    # -- construction ----------------------------------------------------
    @classmethod
    def empty(cls, size: int) -> Tags:
        """The nothing-tagged table over ``size`` elements -- what an untagged mesh
        stores."""
        return cls(np.zeros(0, np.int64), np.empty(0, dtype=np.str_), size)

    @classmethod
    def from_dense(cls, array: Sequence[str] | StrArray) -> Tags:
        """From a dense per-element array where ``""`` means untagged."""
        v = _str_array(array)
        ids: IntArray = np.flatnonzero(v != "").astype(np.int64)
        return cls(ids, v[ids], v.shape[0])

    @classmethod
    def full(cls, shape: int, fill_value: str) -> Tags:
        """``shape`` elements all named ``fill_value`` (or :meth:`empty` when it is
        empty), as ``np.full``."""
        if not fill_value:
            return cls.empty(shape)
        return cls(np.arange(shape, dtype=np.int64), np.full(shape, fill_value), shape)

    # -- views -----------------------------------------------------------
    def to_dense(self) -> StrArray:
        """The equivalent dense ``(size,)`` array, ``""`` where untagged."""
        out: StrArray = (np.full(self.size, "") if not len(self)
                         else np.full(self.size, "", dtype=self.tags.dtype))
        if len(self):
            out[self.ids] = self.tags
        return out

    @overload
    def unique(self) -> list[str]: ...
    @overload
    def unique(self, return_inverse: Literal[True]) -> tuple[list[str], IntArray]: ...
    def unique(self, return_inverse: bool = False
               ) -> list[str] | tuple[list[str], IntArray]:
        """Sorted unique tags, as ``np.unique``.

        With ``return_inverse`` also the index into them of each stored **row**'s tag
        (aligned with ``ids``), so ``inverse + 1`` is a per-row integer code."""
        if not return_inverse:
            return sorted(set(self.tags.tolist()))
        names, inverse = np.unique(self.tags, return_inverse=True)
        return names.tolist(), np.asarray(inverse, dtype=np.int64).reshape(-1)

    def is_uniform(self) -> bool:
        """True when all ``size`` elements carry the same single tag."""
        return len(self) == self.size and len(self.unique()) == 1

    def __len__(self) -> int:
        """The number of **tagged** elements (see the class warning)."""
        return int(self.ids.shape[0])

    def __bool__(self) -> bool:
        return bool(self.ids.shape[0])

    def __iter__(self) -> Iterator[tuple[int, str]]:
        """``(element, tag)`` per tagged element, in stored (ascending id) order."""
        for i, t in zip(self.ids, self.tags):
            yield int(i), str(t)

    def count(self, tag: str) -> int:
        """How many elements are named ``tag``."""
        return int(np.count_nonzero(self.tags == tag))

    def isin(self, test_elements: str | Sequence[str] | StrArray) -> BoolArray:
        """Boolean mask over the **rows** (not the elements) whose tag is one of
        ``test_elements``, as ``np.isin``; a single string is one name."""
        return np.asarray(np.isin(self.tags, test_elements), dtype=bool)

    def flatnonzero(self, test_elements: str | Sequence[str] | StrArray) -> IntArray:
        """Ids of the elements whose tag is one of ``test_elements``, ascending, as
        ``np.flatnonzero(np.isin(dense, test_elements))``; a single string is one
        name."""
        ids: IntArray = self.ids[self.isin(test_elements)]
        return ids

    def compress(self, condition: BoolArray | Sequence[bool]) -> Tags:
        """The rows where ``condition`` is true, ascending id order preserved, as
        ``np.compress`` over the stored rows (``size`` is unchanged).

        ``condition`` is 1-D and may be **shorter** than the table, the missing rows
        counting as false; a true entry past the last row raises ``IndexError``."""
        c = np.asarray(condition)
        if c.ndim != 1:
            raise ValueError("compress: condition must be 1-D, got %d-D" % c.ndim)
        rows: IntArray = np.flatnonzero(c)
        if rows.shape[0] and int(rows[-1]) >= len(self):
            raise IndexError("compress: condition selects row %d but the table has "
                             "only %d rows" % (int(rows[-1]), len(self)))
        return Tags(self.ids[rows], self.tags[rows], self.size)

    def __repr__(self) -> str:
        return "<Tags %d tagged {%s}>" % (len(self), ",".join(self.unique()))

    # -- operations ------------------------------------------------------
    def rename(self, mapping: Mapping[str, str]) -> Tags:
        """The same elements under a new vocabulary: a tag the map does not name is
        left alone, the map applies simultaneously, two keys may share an image, and a
        key that names nothing raises.

        Renaming a region to ``""`` drops it back to untagged, which is what this
        sparse table stores as no row at all.

        The map is read once and written once, hence simultaneous: ``{"a": "b", "b":
        "a"}`` swaps rather than collapsing. The result is re-widened, so a longer name
        is not truncated to the input's dtype. A key naming nothing raises, since a
        rename that silently matches nothing is almost always a typo."""
        vocabulary = self.unique()
        unknown = sorted(set(mapping) - set(vocabulary))
        if unknown:
            raise ValueError(
                "nothing is tagged %s; this table has %s"
                % (", ".join(repr(u) for u in unknown),
                   ", ".join(repr(v) for v in vocabulary) or "no tags at all"))
        if not len(self):
            return self
        uniq, inverse = np.unique(self.tags, return_inverse=True)
        renamed: StrArray = _str_array([mapping.get(str(u), str(u))
                                        for u in uniq.tolist()])
        return Tags(self.ids, renamed[inverse.reshape(-1)], self.size)

    def _index(self, indices: IntArray, who: str) -> IntArray:
        """``indices`` as flat ids over this table, negatives counted from the end as
        ``np.take`` does; one outside ``[-size, size)`` raises ``IndexError``."""
        idx = np.asarray(indices, dtype=np.int64).reshape(-1)
        if idx.shape[0]:
            lo, hi = int(idx.min()), int(idx.max())
            if lo < -self.size or hi >= self.size:
                raise IndexError("%s: index %d is out of bounds for %d elements"
                                 % (who, lo if lo < -self.size else hi, self.size))
            idx = np.where(idx < 0, idx + self.size, idx)
        return idx

    def take(self, indices: IntArray) -> Tags:
        """Tags for a new element list whose element ``k`` copies source
        ``indices[k]``, as ``np.take`` (``mode='raise'``): negative indices count from
        the end, one out of bounds raises ``IndexError``. The result is over
        ``len(indices)`` elements.

        The sparse form of ``dense[indices]``."""
        idx = self._index(indices, "take")
        if not len(self) or not idx.shape[0]:
            return Tags.empty(idx.shape[0])
        pos: IntArray = np.asarray(np.searchsorted(self.ids, idx), dtype=np.int64)
        # searchsorted alone maps a miss onto its neighbour, so confirm the hit
        ok = (pos < self.ids.shape[0]) & (self.ids[np.clip(pos, 0, len(self) - 1)] == idx)
        hit = np.flatnonzero(ok)
        return Tags(hit, self.tags[pos[hit]], idx.shape[0])

    def take_dense(self, indices: IntArray) -> StrArray:
        """The name of each element in ``indices`` as a string array, ``""`` where
        untagged -- ``take(indices).to_dense()``, as ``np.take`` on the dense array."""
        return self.take(indices).to_dense()

    def put(self, ids: IntArray, values: str | Sequence[str] | StrArray) -> Tags:
        """This table with ``ids`` named ``values`` (one string for all, or one per
        id), as ``np.put`` (``mode='raise'``): a name already on an id is overwritten,
        when an id repeats the last value wins, and one out of bounds raises
        ``IndexError``.

        A ``""`` value names nothing and leaves that element as it was (use
        :meth:`clear` to untag)."""
        i = self._index(ids, "put")
        v = (np.full(i.shape[0], values) if isinstance(values, str)
             else _str_array(values))
        if v.shape[0] != i.shape[0]:
            raise ValueError("put: %d ids but %d values" % (i.shape[0], v.shape[0]))
        hit = v != ""
        i, v = i[hit], v[hit]
        if not i.shape[0]:
            return self
        _, last_from_end = np.unique(i[::-1], return_index=True)
        last = i.shape[0] - 1 - last_from_end
        keep = ~np.isin(self.ids, i[last])
        return Tags(np.concatenate([self.ids[keep], i[last]]),
                    np.concatenate([self.tags[keep], v[last]]), self.size)

    def clear(self, ids: IntArray) -> Tags:
        """This table with the tags on ``ids`` removed (``size`` is unchanged); an id
        that is not tagged is ignored, one out of bounds raises ``IndexError``."""
        i = self._index(ids, "clear")
        if not len(self) or not i.shape[0]:
            return self
        return self.compress(~np.isin(self.ids, i))

    def tile(self, reps: int) -> Tags:
        """This table tiled ``reps`` times, over ``reps * size`` elements: element
        ``i*size + q``.

        The sparse form of ``np.tile(dense, reps)``."""
        nb = int(reps)
        if not len(self):
            return Tags.empty(nb * self.size)
        ids = (np.arange(nb, dtype=np.int64)[:, None] * self.size
               + self.ids[None, :]).ravel()
        return Tags(ids, np.tile(self.tags, nb), nb * self.size)

    @staticmethod
    def concatenate(arrays: Sequence[Tags]) -> Tags:
        """The tables one after another over the summed element count, as
        ``np.concatenate``: each table's ids are offset by the sizes before it."""
        offsets = np.concatenate([[0], np.cumsum([t.size for t in arrays])]
                                 ).astype(np.int64)
        total = int(offsets[-1])
        live = [(int(o), t) for o, t in zip(offsets[:-1], arrays) if len(t)]
        if not live:
            return Tags.empty(total)
        return Tags(np.concatenate([t.ids + o for o, t in live]),
                    np.concatenate([t.tags for _, t in live]), total)

    def renumber(self, new_id_of: IntArray, size: int) -> Tags:
        """Tags carried onto new ids in a table over ``size`` elements, where old
        element ``e`` becomes ``new_id_of[e]``."""
        if not len(self):
            return Tags.empty(size)
        m = np.asarray(new_id_of, dtype=np.int64).reshape(-1)
        return Tags(m[self.ids], self.tags, size)
