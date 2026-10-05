"""The two tag tables every rung stores: a side-tag table and :class:`Tags`."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from .._typing import BoolArray, IntArray, StrArray

__all__ = ["Tags", "mask_for_selection", "sweep_tags", "sweep_cap",
           "weld"]


def _frozen(arr: np.ndarray) -> np.ndarray:  # type: ignore[type-arg]
    """A read-only *view* of ``arr``."""
    v = arr.view()
    v.flags.writeable = False
    return v


def _str_array(values: Sequence[str] | StrArray) -> StrArray:
    """``values`` as a 1-D ``np.str_`` array, width inferred (never clipped to ``<U1``)."""
    return np.asarray(values, dtype=np.str_).reshape(-1)


def _empty_str() -> StrArray:
    """A zero-length string array."""
    return np.empty(0, dtype=np.str_)


@dataclass(frozen=True, eq=False)
class Tags:
    """A mesh's per-element region tags, stored **sparsely**: only tagged elements."""

    ids: IntArray
    tags: StrArray

    def __post_init__(self) -> None:
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
        object.__setattr__(self, "ids", _frozen(i))
        object.__setattr__(self, "tags", _frozen(t))

    # -- construction ----------------------------------------------------
    @classmethod
    def empty(cls) -> Tags:
        """The nothing-tagged table -- what an untagged mesh stores."""
        return cls(np.zeros(0, np.int64), _empty_str())

    @classmethod
    def from_dense(cls, array: Sequence[str] | StrArray) -> Tags:
        """From a dense per-element array where ``""`` means untagged."""
        v = _str_array(array)
        ids: IntArray = np.flatnonzero(v != "").astype(np.int64)
        return cls(ids, v[ids])

    @classmethod
    def full(cls, shape: int, fill_value: str) -> Tags:
        """``shape`` elements all named ``fill_value`` (or :meth:`empty` when it is
        empty), as ``np.full``."""
        if not fill_value:
            return cls.empty()
        return cls(np.arange(shape, dtype=np.int64), np.full(shape, fill_value))

    # -- views -----------------------------------------------------------
    def to_dense(self, size: int) -> StrArray:
        """The equivalent dense ``(size,)`` array, ``""`` where untagged."""
        out: StrArray = (np.full(size, "") if not len(self)
                         else np.full(size, "", dtype=self.tags.dtype))
        if len(self):
            out[self.ids] = self.tags
        return out

    def unique(self) -> list[str]:
        """Sorted unique tags, as ``np.unique``."""
        return sorted(set(self.tags.tolist()))

    def is_uniform(self, size: int) -> bool:
        """True when all ``size`` elements carry the same single tag."""
        return len(self) == size and len(self.unique()) == 1

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

    def compress(self, condition: BoolArray | Sequence[bool]) -> Tags:
        """The rows where ``condition`` is true, ascending id order preserved, as
        ``np.compress`` over the stored rows.

        ``condition`` is 1-D and may be **shorter** than the table, the missing rows
        counting as false; a true entry past the last row raises ``IndexError``."""
        c = np.asarray(condition)
        if c.ndim != 1:
            raise ValueError("compress: condition must be 1-D, got %d-D" % c.ndim)
        rows: IntArray = np.flatnonzero(c)
        if rows.shape[0] and int(rows[-1]) >= len(self):
            raise IndexError("compress: condition selects row %d but the table has "
                             "only %d rows" % (int(rows[-1]), len(self)))
        return Tags(self.ids[rows], self.tags[rows])

    def __repr__(self) -> str:
        return "<Tags %d tagged {%s}>" % (len(self), ",".join(self.unique()))

    # -- operations ------------------------------------------------------
    def rename(self, mapping: Mapping[str, str],
               who: str = "rename") -> Tags:
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
                "%s: nothing is tagged %s; this table has %s"
                % (who, ", ".join(repr(u) for u in unknown),
                   ", ".join(repr(v) for v in vocabulary) or "no tags at all"))
        if not len(self):
            return self
        uniq, inverse = np.unique(self.tags, return_inverse=True)
        renamed: StrArray = _str_array([mapping.get(str(u), str(u))
                                        for u in uniq.tolist()])
        return Tags(self.ids, renamed[inverse.reshape(-1)])

    def take(self, indices: IntArray) -> Tags:
        """Tags for a new element list whose element ``k`` copies source
        ``indices[k]``, as ``np.take``.

        The sparse form of ``dense[indices]``. The table does not know how many
        elements the mesh has, so an index past the last one reads as untagged rather
        than raising; a negative one raises."""
        idx = np.asarray(indices, dtype=np.int64).reshape(-1)
        if idx.shape[0] and int(idx.min()) < 0:
            raise IndexError("take: negative element index %d" % int(idx.min()))
        if not len(self) or not idx.shape[0]:
            return Tags.empty()
        pos: IntArray = np.asarray(np.searchsorted(self.ids, idx), dtype=np.int64)
        # searchsorted alone maps a miss onto its neighbour, so confirm the hit
        ok = (pos < self.ids.shape[0]) & (self.ids[np.clip(pos, 0, len(self) - 1)] == idx)
        hit = np.flatnonzero(ok)
        return Tags(hit, self.tags[pos[hit]])

    def tile(self, reps: int, size: int) -> Tags:
        """This table, over ``size`` elements, tiled ``reps`` times: element
        ``i*size + q``.

        The sparse form of ``np.tile(dense, reps)``."""
        if not len(self):
            return Tags.empty()
        nb, bs = int(reps), int(size)
        ids = (np.arange(nb, dtype=np.int64)[:, None] * bs + self.ids[None, :]).ravel()
        return Tags(ids, np.tile(self.tags, nb))

    def shift(self, delta: int) -> Tags:
        """The same tags with every element id shifted by ``delta`` (for ``merge``)."""
        if not len(self):
            return self
        return Tags(self.ids + int(delta), self.tags)

    @staticmethod
    def concatenate(arrays: Sequence[Tags]) -> Tags:
        """The tables' tags together, as ``np.concatenate`` (ids must already be
        disjoint -- shift first)."""
        live = [p for p in arrays if len(p)]
        if not live:
            return Tags.empty()
        return Tags(np.concatenate([p.ids for p in live]),
                           np.concatenate([p.tags for p in live]))

    def renumber(self, new_id_of: IntArray) -> Tags:
        """Tags carried onto new ids, where old element ``e`` becomes ``new_id_of[e]``."""
        if not len(self):
            return self
        m = np.asarray(new_id_of, dtype=np.int64).reshape(-1)
        return Tags(m[self.ids], self.tags)

    def validate(self, size: int) -> None:
        """Raise if any tagged id names an element beyond the mesh's ``size``."""
        if len(self) and int(self.ids[-1]) >= size:
            raise ValueError("element_tags names element %d but there are only %d "
                             "elements" % (int(self.ids[-1]), size))



def weld(tables: Sequence[Tags], who: str) -> Tags:
    """Combine tables that may name the **same** element -- what a ``merge`` produces
    once its weld has carried two blocks' tags onto one shared entity.

    Plain :meth:`Tags.concatenate` cannot do this: it would hand the constructor two
    rows for one id, which is rejected. Here the duplicate is expected and meaningful,
    so it is resolved rather than refused -- but only when the two agree. Two different
    non-empty names on one entity is the contradiction the shared-entity storage exists
    to rule out, and there is no honest way to pick between them, so it raises.

    An entity named by one side and left untagged by the other simply takes the name:
    that is the ordinary case of a tagged block welding onto an untagged neighbour."""
    live = [t for t in tables if len(t)]
    if not live:
        return Tags.empty()
    # concatenated as raw columns rather than through ``concat``: the duplicate ids
    # this resolves are exactly what the constructor is there to reject
    raw_ids: IntArray = np.concatenate([t.ids for t in live])
    raw_names: StrArray = _str_array(np.concatenate([t.tags for t in live]))
    order = np.argsort(raw_ids, kind="stable")
    ids, names = raw_ids[order], raw_names[order]
    dup: BoolArray = np.zeros(ids.shape[0], dtype=bool)
    dup[1:] = ids[1:] == ids[:-1]
    if not dup.any():
        return Tags(ids, names)
    differs: BoolArray = np.zeros(ids.shape[0], dtype=bool)
    differs[1:] = names[1:] != names[:-1]
    clash = np.flatnonzero(dup & differs)
    if clash.size:
        i = int(clash[0])
        raise ValueError(
            "%s: the weld puts two different names on one entity -- element %d is "
            "tagged both %r and %r. A shared entity carries one tag, so leave one of "
            "the two sides untagged, or give them the same name."
            % (who, int(ids[i]), str(names[i - 1]), str(names[i])))
    return Tags(ids[~dup], names[~dup])


def mask_for_selection(which: str | BoolArray | IntArray | Sequence[int],
                 tags: Tags, n_elements: int, who: str) -> BoolArray:
    """The ``(n_elements,)`` boolean mask a ``select`` / ``remove`` argument names.

    ``which`` is a **tag string** (every element carrying it -- absent from the mesh's
    vocabulary is an error, since a silent empty selection is almost always a typo), a
    ready ``(n_elements,)`` boolean mask, or an array of element ids."""
    if isinstance(which, str):
        if which not in tags.unique():
            raise ValueError(
                "%s: no element carries the tag %r; this mesh has %s"
                % (who, which, tags.unique() or "no tagged elements"))
        return np.asarray(tags.to_dense(n_elements) == which, dtype=bool)
    arr = np.asarray(which)
    if arr.size == 0:
        return np.zeros(n_elements, dtype=bool)
    if arr.dtype == bool:
        m: BoolArray = arr.reshape(-1)
        if m.shape[0] != n_elements:
            raise ValueError("%s: a boolean mask must cover all %d elements, got %d"
                             % (who, n_elements, m.shape[0]))
        return m
    if not np.issubdtype(arr.dtype, np.integer):
        raise TypeError(
            "%s: select by a tag string, a (%d,) boolean mask, or an array of element "
            "ids; got a %s array" % (who, n_elements, arr.dtype))
    ids: IntArray = arr.reshape(-1).astype(np.int64)
    if int(ids.min()) < 0 or int(ids.max()) >= n_elements:
        raise ValueError("%s: element ids must lie in [0, %d); got [%d, %d]"
                         % (who, n_elements, int(ids.min()), int(ids.max())))
    out: BoolArray = np.zeros(n_elements, dtype=bool)
    out[ids] = True
    return out


def sweep_tags(spec: str | Tags | None, n_layers: int,
                       n_slice: int, who: str) -> Tags:
    """The swept elements' region tags from a ``loft``'s ``element_tags`` argument.

    ``None`` tags nothing, a ``str`` tags every swept element, and an
    :class:`Tags` over one slice's ``n_slice`` elements tags each element by
    the slice element it was extruded from (element ``i*n_slice + k``)."""
    if spec is None:
        return Tags.empty()
    if isinstance(spec, str):
        return Tags.full(int(n_layers) * int(n_slice), spec)
    if isinstance(spec, Tags):
        spec.validate(int(n_slice))
        return spec.tile(int(n_layers), int(n_slice))
    raise TypeError(
        "%s: element_tags must be a tag string, a Tags over the %d elements "
        "of one slice, or None; got %s"
        % (who, n_slice, type(spec).__name__))


def sweep_cap(spec: str | Tags | None, default: Tags,
                   n_slice: int, who: str) -> StrArray:
    """One cap's dense ``(n_slice,)`` tag row from a ``loft``'s ``first_tag`` /
    ``last_tag`` argument, falling back to ``default`` (the bounding slice's own
    element tags -- a cap side *is* that slice element) when the argument is ``None``.
    """
    if spec is None:
        return default.to_dense(int(n_slice))
    if isinstance(spec, str):
        return np.full(int(n_slice), spec)
    if isinstance(spec, Tags):
        spec.validate(int(n_slice))
        return spec.to_dense(int(n_slice))
    raise TypeError(
        "%s: a cap tag must be a tag string, a Tags over the %d elements of "
        "one slice, or None; got %s" % (who, n_slice, type(spec).__name__))
