"""The two tag tables every rung stores: :class:`BoundaryTable` and :class:`ElementTags`.

Both are rung-agnostic *model data* -- they take no container and know nothing about
lines, quads or hexes beyond an integer element id -- so they live here beside
``model.physical``'s :class:`~nekmeshpy.model.physical.PhysicalGroup` and
``model.quality``'s ``QualitySummary`` rather than in a sibling operation module.

**Why they are types and not loose arrays.**  A tagged boundary is a *row*: an element,
one of its sides, and a name.  Stored as two parallel arrays, every operation that
reorders, offsets, filters or concatenates rows has to do it twice, by hand, correctly --
which is what ``_order_bnd`` used to exist for, three times over.  Here the row is one
object and those become single calls that cannot desynchronize.

**Why element tags are sparse.**  ``""`` was never a meaningful element tag -- every
consumer read it as "absent" -- so a dense ``(E,)`` string array spent one slot per
element to say nothing about most of them.  :class:`ElementTags` stores only the tagged
ids, so an untagged mesh costs nothing at all, and a wide tag no longer widens the whole
array (a 16-character tag over 486k hexes was 31 MB of mostly-empty strings).

.. note::
   ``np.full(n, s, dtype=np.str_)`` and ``np.empty(n, dtype=np.str_)`` both clip to
   ``<U1`` -- ``np.full(3, "wall", dtype=np.str_)`` yields ``'w'``, silently.  This
   module never uses either: it builds string arrays with ``np.full(n, s)`` (width
   inferred from the fill value) or ``np.asarray(seq, dtype=np.str_)`` (width inferred
   from the sequence).  Keeping that discipline in one file is a large part of why these
   types exist.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass

import numpy as np

from .._typing import BoolArray, IntArray, StrArray

__all__ = ["BoundaryTable", "BoundaryBuilder", "ElementTags", "NO_TAGS",
           "check_tag_range"]


def _frozen(arr: np.ndarray) -> np.ndarray:  # type: ignore[type-arg]
    """A read-only *view* of ``arr``.

    Freezing a view rather than ``arr`` itself leaves the caller's own buffer writable,
    so this hardens the tables against accidental mutation (they are shared by reference
    across every pass-through operation) without reaching back into caller state."""
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
class BoundaryTable:
    """The tagged boundary rows of a mesh: ``elements``, ``sides``, ``tags``.

    Row ``r`` names side ``sides[r]`` of element ``elements[r]``.  The side numbering is
    the rung's own -- 1-2 for a ``LineMesh`` end point, 1-4 for a ``QuadMesh`` edge, 1-6
    for a ``HexMesh`` face -- and element ids are 0-based.

    Three parallel 1-D columns rather than an ``(Nb,2)`` block plus tags: it keeps
    :meth:`offset` a plain add, lets consumers gather with ``sides - 1`` directly, and
    removes the ``int(rows[r, 0])`` idiom from every reader.  :attr:`rows` recovers the
    paired form for the few callers that want it.

    **Row order is meaningful and is never changed implicitly.**  Construction preserves
    the order it is given; :meth:`ordered` applies the canonical sort explicitly.  Two
    consumers depend on this: ``loft`` reads a section's rows in stored order, and the
    ``.vtu`` writer resolves a node touched by several rows to the *last* one, so a
    reordering that moved rows tied on ``(element, side)`` would change its output.

    ``eq`` is disabled: the generated ``__eq__`` compares field tuples, which for ndarray
    fields raises ``ValueError: truth value of an array ... is ambiguous``.  Compare the
    columns explicitly instead."""

    elements: IntArray
    sides: IntArray
    tags: StrArray

    def __post_init__(self) -> None:
        e = np.asarray(self.elements, dtype=np.int64).reshape(-1)
        s = np.asarray(self.sides, dtype=np.int64).reshape(-1)
        t = _str_array(self.tags)
        if not (e.shape[0] == s.shape[0] == t.shape[0]):
            raise ValueError(
                "BoundaryTable: elements (%d), sides (%d) and tags (%d) must have the "
                "same length" % (e.shape[0], s.shape[0], t.shape[0]))
        object.__setattr__(self, "elements", _frozen(e))
        object.__setattr__(self, "sides", _frozen(s))
        object.__setattr__(self, "tags", _frozen(t))

    # -- construction ----------------------------------------------------
    @classmethod
    def empty(cls) -> BoundaryTable:
        """The no-rows table."""
        return cls(np.zeros(0, np.int64), np.zeros(0, np.int64), _empty_str())

    @classmethod
    def from_pairs(cls, rows: Sequence[Sequence[int]] | IntArray,
                   tags: Sequence[str] | StrArray) -> BoundaryTable:
        """From the ``(Nb,2)`` ``[element, side]`` block + parallel ``tags``."""
        r: IntArray = np.asarray(rows, dtype=np.int64).reshape(-1, 2)
        return cls(r[:, 0], r[:, 1], tags)

    # -- views -----------------------------------------------------------
    @property
    def rows(self) -> IntArray:
        """``(Nb,2)`` ``[element, side]`` -- the paired form."""
        return np.column_stack([self.elements, self.sides])

    @property
    def group_tags(self) -> list[str]:
        """Sorted unique tags -- nothing is filtered out, since a row exists in the
        first place only because it was named."""
        return sorted(set(self.tags.tolist()))

    def as_dict(self) -> dict[tuple[int, int], str]:
        """``{(element, side): tag}``, built in **row order** so duplicate keys resolve
        to the last row, matching the comprehension this replaces in ``HexMesh.loft``."""
        return {(int(e), int(s)): str(t)
                for e, s, t in zip(self.elements, self.sides, self.tags)}

    def __len__(self) -> int:
        return int(self.elements.shape[0])

    def __bool__(self) -> bool:
        return bool(self.elements.shape[0])

    def __iter__(self) -> Iterator[tuple[int, int, str]]:
        """``(element, side, tag)`` per row, in stored order."""
        for e, s, t in zip(self.elements, self.sides, self.tags):
            yield int(e), int(s), str(t)

    def __repr__(self) -> str:
        return "<BoundaryTable %d rows {%s}>" % (len(self), ",".join(self.group_tags))

    # -- operations ------------------------------------------------------
    def ordered(self) -> BoundaryTable:
        """The rows stably sorted by ``(element, side)`` -- the canonical storage order.

        Exactly ``np.lexsort((sides, elements))``.  Call it where the old ``_order_bnd``
        was called and nowhere else: it is what fixes the ``.re2`` boundary block's
        byte-for-byte order."""
        if not len(self):
            return self
        p = np.lexsort((self.sides, self.elements))
        return BoundaryTable(self.elements[p], self.sides[p], self.tags[p])

    def offset(self, delta: int) -> BoundaryTable:
        """The same rows with every element id shifted by ``delta`` (for ``merge``)."""
        if not len(self):
            return self
        return BoundaryTable(self.elements + int(delta), self.sides, self.tags)

    @staticmethod
    def concat(tables: Sequence[BoundaryTable]) -> BoundaryTable:
        """The tables' rows end to end, order preserved."""
        parts = [t for t in tables if len(t)]
        if not parts:
            return BoundaryTable.empty()
        return BoundaryTable(
            np.concatenate([t.elements for t in parts]),
            np.concatenate([t.sides for t in parts]),
            np.concatenate([t.tags for t in parts]))

    def mask_for(self, tag: str) -> BoolArray:
        """Boolean mask of the rows named ``tag``."""
        return np.asarray(self.tags == tag, dtype=bool)

    def select(self, mask: BoolArray) -> BoundaryTable:
        """The rows where ``mask`` is True, order preserved."""
        m = np.asarray(mask, dtype=bool)
        return BoundaryTable(self.elements[m], self.sides[m], self.tags[m])

    def count(self, tag: str) -> int:
        """How many rows are named ``tag``."""
        return int(np.count_nonzero(self.tags == tag))


class BoundaryBuilder:
    """Accumulates :class:`BoundaryTable` rows one at a time.

    The factories emit rows inside loops over elements and sides, which is what the
    paired ``bnd.append(...)`` / ``names.append(...)`` idiom used to be.  Backed by plain
    lists so **insertion order is preserved exactly** -- a dict- or set-backed builder
    would silently dedupe or reorder rows, and both stored order and ``(element, side)``
    ties are observable downstream (see :class:`BoundaryTable`)."""

    def __init__(self) -> None:
        self._elements: list[int] = []
        self._sides: list[int] = []
        self._tags: list[str] = []

    def add(self, element: int, side: int, tag: str) -> None:
        """Append one row unconditionally."""
        self._elements.append(int(element))
        self._sides.append(int(side))
        self._tags.append(str(tag))

    def add_if_tagged(self, element: int, side: int, tag: str) -> None:
        """Append one row only if ``tag`` is non-empty.

        Deliberately separate from :meth:`add`: the call sites' skip conditions are not
        all the same, and folding them together is how a behaviour change sneaks in."""
        if tag:
            self.add(element, side, tag)

    def extend(self, table: BoundaryTable) -> None:
        """Append every row of ``table``, in its order."""
        for e, s, t in table:
            self.add(e, s, t)

    def __len__(self) -> int:
        return len(self._elements)

    def __bool__(self) -> bool:
        return bool(self._elements)

    def build(self) -> BoundaryTable:
        """The rows as stored, in insertion order."""
        if not self._elements:
            return BoundaryTable.empty()
        return BoundaryTable(np.asarray(self._elements, dtype=np.int64),
                             np.asarray(self._sides, dtype=np.int64),
                             _str_array(self._tags))

    def build_ordered(self) -> BoundaryTable:
        """:meth:`build` then :meth:`BoundaryTable.ordered`."""
        return self.build().ordered()


@dataclass(frozen=True, eq=False)
class ElementTags:
    """A mesh's per-element region tags, stored **sparsely**: only tagged elements.

    ``ids`` are the tagged elements (strictly ascending, unique) and ``tags[i]`` names
    ``ids[i]``.  Every tag is non-empty -- an element is either named or absent, which is
    what ``""`` always meant in the dense array this replaces.  An untagged mesh is
    :data:`NO_TAGS`, which allocates nothing.

    .. warning::
       ``len()`` is the number of **tagged** elements, not the mesh's element count.
       Use the container's own ``n_lines`` / ``n_quads`` / ``n_hexes`` for that.

    Construction normalizes: empty tags are dropped, rows are sorted by id, and duplicate
    ids are rejected.  That is safe here (unlike :class:`BoundaryTable`) because element
    tags are keyed by id -- nothing downstream observes row order.

    ``eq`` is disabled for the same reason as :class:`BoundaryTable`."""

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
        """The nothing-tagged table."""
        return cls(np.zeros(0, np.int64), _empty_str())

    @classmethod
    def from_dense(cls, values: Sequence[str] | StrArray, n: int | None = None,
                   what: str = "elements") -> ElementTags:
        """From a dense per-element array where ``""`` means untagged.

        ``n``, when given, is the element count the array must match -- the bridge for
        every factory whose public argument stays dense.  ``what`` names the rung's
        elements so the error reads in the caller's own vocabulary."""
        v = _str_array(values)
        if n is not None and v.shape[0] != n:
            raise ValueError("element_tags length (%d) must match %s (%d)"
                             % (v.shape[0], what, n))
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
        """The equivalent dense ``(n,)`` array, ``""`` where untagged.

        For the few consumers that genuinely want one element per slot; storage stays
        sparse."""
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

    def __repr__(self) -> str:
        return "<ElementTags %d tagged {%s}>" % (len(self), ",".join(self.group_tags))

    # -- operations ------------------------------------------------------
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


#: The shared zero-length :class:`ElementTags` -- an untagged mesh allocates nothing.
NO_TAGS: ElementTags = ElementTags.empty()


def check_tag_range(element_tags: ElementTags, boundaries: BoundaryTable,
                    n_elements: int, n_sides: int, what: str) -> None:
    """Raise if either table names an element or side the mesh does not have.

    A check the containers could not perform while the tags were loose arrays: a
    dense per-element array is in range by construction, and nothing ever validated
    the boundary rows at all.  Sparse ids make it cheap and worth doing -- it catches
    the class of bug where a ``merge`` forgets to offset one of its blocks, which
    previously surfaced only as a wrong tag in the exported file.

    ``what`` names the rung's elements (``"lines"`` / ``"quads"`` / ``"hexes"``) so
    the message points at the caller's own vocabulary."""
    if len(element_tags) and (int(element_tags.ids[-1]) >= n_elements):
        raise ValueError("element_tags names element %d but there are only %d %s"
                         % (int(element_tags.ids[-1]), n_elements, what))
    if len(boundaries):
        hi = int(boundaries.elements.max())
        if hi >= n_elements or int(boundaries.elements.min()) < 0:
            raise ValueError("boundaries names element %d but there are only %d %s"
                             % (hi, n_elements, what))
        s_lo, s_hi = int(boundaries.sides.min()), int(boundaries.sides.max())
        if s_lo < 1 or s_hi > n_sides:
            raise ValueError("boundaries side %d is outside 1..%d"
                             % (s_hi if s_hi > n_sides else s_lo, n_sides))
