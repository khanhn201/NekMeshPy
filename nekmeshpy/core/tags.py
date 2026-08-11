"""The two tag tables every rung stores: a side-tag table and :class:`ElementTags`."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from .._typing import BoolArray, IntArray, StrArray

__all__ = ["ElementTags", "element_mask", "sweep_element_tags", "sweep_cap_tags",
           "welded_element_tags"]


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


def _renamed(tags: StrArray, mapping: Mapping[str, str], vocabulary: list[str],
             who: str) -> StrArray:
    """``tags`` with every entry the ``mapping`` names replaced by its image, entries
    it does not name left alone.

    Read once and written once, so the map applies **simultaneously**: ``{"a": "b",
    "b": "a"}`` swaps the two rather than collapsing both onto one. Two keys may share
    an image, which merges those groups. The result is re-widened rather than written
    into the input's dtype, so a longer name does not come back truncated.

    A key that names nothing is an error, on the same reasoning
    :func:`element_mask` refuses an absent tag: a rename that silently matches
    nothing is almost always a typo, and in a mesh a mis-spelled region or boundary
    name is not visible again until the solver reads it."""
    unknown = sorted(set(mapping) - set(vocabulary))
    if unknown:
        raise ValueError(
            "%s: nothing is tagged %s; this table has %s"
            % (who, ", ".join(repr(u) for u in unknown),
               ", ".join(repr(v) for v in vocabulary) or "no tags at all"))
    if not tags.shape[0]:
        return tags
    uniq, inverse = np.unique(tags, return_inverse=True)
    renamed: StrArray = _str_array([mapping.get(str(u), str(u)) for u in uniq.tolist()])
    return renamed[inverse.reshape(-1)]


@dataclass(frozen=True, eq=False)
class ElementTags:
    """A mesh's per-element region tags, stored **sparsely**: only tagged elements."""

    ids: IntArray
    tags: StrArray

    def __post_init__(self) -> None:
        i = np.asarray(self.ids, dtype=np.int64).reshape(-1)
        t = _str_array(self.tags)
        if i.shape[0] != t.shape[0]:
            raise ValueError("ElementTags: ids (%d) and tags (%d) must have the same "
                             "length" % (i.shape[0], t.shape[0]))
        keep = t != ""
        if not np.all(keep):
            i, t = i[keep], t[keep]
        if i.shape[0]:
            p = np.argsort(i, kind="stable")
            i, t = i[p], t[p]
            if np.any(np.diff(i) == 0):
                dup = int(i[:-1][np.diff(i) == 0][0])
                raise ValueError("ElementTags: element %d is tagged more than once" % dup)
            if int(i[0]) < 0:
                raise ValueError("ElementTags: negative element id %d" % int(i[0]))
        object.__setattr__(self, "ids", _frozen(i))
        object.__setattr__(self, "tags", _frozen(t))

    # -- construction ----------------------------------------------------
    @classmethod
    def empty(cls) -> ElementTags:
        """The nothing-tagged table -- what an untagged mesh stores."""
        return cls(np.zeros(0, np.int64), _empty_str())

    @classmethod
    def from_dense(cls, values: Sequence[str] | StrArray) -> ElementTags:
        """From a dense per-element array where ``""`` means untagged."""
        v = _str_array(values)
        ids: IntArray = np.flatnonzero(v != "").astype(np.int64)
        return cls(ids, v[ids])

    @classmethod
    def uniform(cls, n: int, tag: str) -> ElementTags:
        """``n`` elements all named ``tag`` (or :meth:`empty` when ``tag`` is empty)."""
        if not tag:
            return cls.empty()
        return cls(np.arange(n, dtype=np.int64), np.full(n, tag))

    @classmethod
    def blocks(cls, per_block: Sequence[str] | StrArray, block_size: int) -> ElementTags:
        """Block ``i``'s tag applied to elements ``i*block_size ...`` (untagged blocks
        contribute nothing) -- the per-layer override axis of ``loft``."""
        b = _str_array(per_block)
        hit = np.flatnonzero(b != "")
        if not hit.shape[0]:
            return cls.empty()
        ids = (hit[:, None] * int(block_size)
               + np.arange(int(block_size), dtype=np.int64)[None, :]).ravel()
        return cls(ids, np.repeat(b[hit], int(block_size)))

    # -- views -----------------------------------------------------------
    def dense(self, n: int) -> StrArray:
        """The equivalent dense ``(n,)`` array, ``""`` where untagged."""
        out: StrArray = (np.full(n, "") if not len(self)
                         else np.full(n, "", dtype=self.tags.dtype))
        if len(self):
            out[self.ids] = self.tags
        return out

    @property
    def group_tags(self) -> list[str]:
        """Sorted unique tags."""
        return sorted(set(self.tags.tolist()))

    def is_uniform(self, n: int) -> bool:
        """True when all ``n`` elements carry the same single tag."""
        return len(self) == n and len(self.group_tags) == 1

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

    def mask_for(self, tag: str) -> BoolArray:
        """Boolean mask of the **rows** named ``tag`` (not of the elements)."""
        return np.asarray(self.tags == tag, dtype=bool)

    def select(self, mask: BoolArray) -> ElementTags:
        """The rows where ``mask`` is True, ascending id order preserved."""
        m = np.asarray(mask, dtype=bool)
        return ElementTags(self.ids[m], self.tags[m])

    def __repr__(self) -> str:
        return "<ElementTags %d tagged {%s}>" % (len(self), ",".join(self.group_tags))

    # -- operations ------------------------------------------------------
    def renamed(self, mapping: Mapping[str, str],
                who: str = "renamed") -> ElementTags:
        """The same elements under a new vocabulary: a tag the map does not name is
        left alone, the map applies simultaneously, two keys may share an image, and a
        key that names nothing raises.

        Renaming a region to ``""`` drops it back to untagged, which is what this
        sparse table stores as no row at all."""
        return ElementTags(self.ids, _renamed(self.tags, mapping,
                                              self.group_tags, who))

    def gather(self, index: IntArray) -> ElementTags:
        """Tags for a new element list whose element ``k`` copies source ``index[k]``.

        The sparse form of ``dense[index]``."""
        idx = np.asarray(index, dtype=np.int64).reshape(-1)
        if not len(self) or not idx.shape[0]:
            return ElementTags.empty()
        pos: IntArray = np.asarray(np.searchsorted(self.ids, idx), dtype=np.int64)
        # searchsorted alone maps a miss onto its neighbour, so confirm the hit
        ok = (pos < self.ids.shape[0]) & (self.ids[np.clip(pos, 0, len(self) - 1)] == idx)
        hit = np.flatnonzero(ok)
        return ElementTags(hit, self.tags[pos[hit]])

    def repeat_blocks(self, n_blocks: int, block_size: int) -> ElementTags:
        """This table tiled over ``n_blocks`` blocks, element ``i*block_size + q``.

        The sparse form of ``np.tile(dense, n_blocks)``."""
        if not len(self):
            return ElementTags.empty()
        nb, bs = int(n_blocks), int(block_size)
        ids = (np.arange(nb, dtype=np.int64)[:, None] * bs + self.ids[None, :]).ravel()
        return ElementTags(ids, np.tile(self.tags, nb))

    def overlay(self, over: ElementTags) -> ElementTags:
        """``over`` wins wherever it names an element, this table stands elsewhere.

        The sparse form of ``np.where(over != "", over, base)``."""
        if not len(over):
            return self
        if not len(self):
            return over
        keep = ~np.isin(self.ids, over.ids)
        return ElementTags(np.concatenate([self.ids[keep], over.ids]),
                           np.concatenate([self.tags[keep], over.tags]))

    def offset(self, delta: int) -> ElementTags:
        """The same tags with every element id shifted by ``delta`` (for ``merge``)."""
        if not len(self):
            return self
        return ElementTags(self.ids + int(delta), self.tags)

    @staticmethod
    def concat(parts: Sequence[ElementTags]) -> ElementTags:
        """The tables' tags together (ids must already be disjoint -- offset first)."""
        live = [p for p in parts if len(p)]
        if not live:
            return ElementTags.empty()
        return ElementTags(np.concatenate([p.ids for p in live]),
                           np.concatenate([p.tags for p in live]))

    def renumber(self, new_id_of: IntArray) -> ElementTags:
        """Tags carried onto new ids, where old element ``e`` becomes ``new_id_of[e]``."""
        if not len(self):
            return self
        m = np.asarray(new_id_of, dtype=np.int64).reshape(-1)
        return ElementTags(m[self.ids], self.tags)

    def check_within(self, n_elements: int) -> None:
        """Raise if any tagged id names an element the mesh does not have."""
        if len(self) and int(self.ids[-1]) >= n_elements:
            raise ValueError("element_tags names element %d but there are only %d "
                             "elements" % (int(self.ids[-1]), n_elements))



def welded_element_tags(tables: Sequence[ElementTags], who: str) -> ElementTags:
    """Combine tables that may name the **same** element -- what a ``merge`` produces
    once its weld has carried two blocks' tags onto one shared entity.

    Plain :meth:`ElementTags.concat` cannot do this: it would hand the constructor two
    rows for one id, which is rejected. Here the duplicate is expected and meaningful,
    so it is resolved rather than refused -- but only when the two agree. Two different
    non-empty names on one entity is the contradiction the shared-entity storage exists
    to rule out, and there is no honest way to pick between them, so it raises.

    An entity named by one side and left untagged by the other simply takes the name:
    that is the ordinary case of a tagged block welding onto an untagged neighbour."""
    live = [t for t in tables if len(t)]
    if not live:
        return ElementTags.empty()
    # concatenated as raw columns rather than through ``concat``: the duplicate ids
    # this resolves are exactly what the constructor is there to reject
    raw_ids: IntArray = np.concatenate([t.ids for t in live])
    raw_names: StrArray = _str_array(np.concatenate([t.tags for t in live]))
    order = np.argsort(raw_ids, kind="stable")
    ids, names = raw_ids[order], raw_names[order]
    dup: BoolArray = np.zeros(ids.shape[0], dtype=bool)
    dup[1:] = ids[1:] == ids[:-1]
    if not dup.any():
        return ElementTags(ids, names)
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
    return ElementTags(ids[~dup], names[~dup])


def element_mask(which: str | BoolArray | IntArray | Sequence[int],
                 tags: ElementTags, n_elements: int, who: str) -> BoolArray:
    """The ``(n_elements,)`` boolean mask a ``select`` / ``remove`` argument names.

    ``which`` is a **tag string** (every element carrying it -- absent from the mesh's
    vocabulary is an error, since a silent empty selection is almost always a typo), a
    ready ``(n_elements,)`` boolean mask, or an array of element ids."""
    if isinstance(which, str):
        if which not in tags.group_tags:
            raise ValueError(
                "%s: no element carries the tag %r; this mesh has %s"
                % (who, which, tags.group_tags or "no tagged elements"))
        return np.asarray(tags.dense(n_elements) == which, dtype=bool)
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


def sweep_element_tags(spec: str | ElementTags | None, n_layers: int,
                       n_slice: int, who: str) -> ElementTags:
    """The swept elements' region tags from a ``loft``'s ``element_tags`` argument.

    ``None`` tags nothing, a ``str`` tags every swept element, and an
    :class:`ElementTags` over one slice's ``n_slice`` elements tags each element by
    the slice element it was extruded from (element ``i*n_slice + k``)."""
    if spec is None:
        return ElementTags.empty()
    if isinstance(spec, str):
        return ElementTags.uniform(int(n_layers) * int(n_slice), spec)
    if isinstance(spec, ElementTags):
        spec.check_within(int(n_slice))
        return spec.repeat_blocks(int(n_layers), int(n_slice))
    raise TypeError(
        "%s: element_tags must be a tag string, an ElementTags over the %d elements "
        "of one slice, or None; got %s"
        % (who, n_slice, type(spec).__name__))


def sweep_cap_tags(spec: str | ElementTags | None, default: ElementTags,
                   n_slice: int, who: str) -> StrArray:
    """One cap's dense ``(n_slice,)`` tag row from a ``loft``'s ``first_tag`` /
    ``last_tag`` argument, falling back to ``default`` (the bounding slice's own
    element tags -- a cap side *is* that slice element) when the argument is ``None``.
    """
    if spec is None:
        return default.dense(int(n_slice))
    if isinstance(spec, str):
        return np.full(int(n_slice), spec)
    if isinstance(spec, ElementTags):
        spec.check_within(int(n_slice))
        return spec.dense(int(n_slice))
    raise TypeError(
        "%s: a cap tag must be a tag string, an ElementTags over the %d elements of "
        "one slice, or None; got %s" % (who, n_slice, type(spec).__name__))
