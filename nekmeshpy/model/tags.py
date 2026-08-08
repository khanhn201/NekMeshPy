"""The two tag tables every rung stores: a side-tag table and :class:`ElementTags`."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import ClassVar, Generic, TypeVar

import numpy as np

from .._typing import BoolArray, IntArray, StrArray

T = TypeVar("T", bound="SideTags")

__all__ = ["SideTags", "PointTags", "EdgeTags", "FaceTags", "TagBuilder",
           "ElementTags", "sweep_element_tags", "sweep_cap_tags"]


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
class SideTags:
    """Shared implementation of the three side-tag tables."""

    #: How many sides the rung's element has -- 2 / 4 / 6 going up the ladder.  The
    #: subclasses set it, and it is the whole reason they exist as separate types:
    #: it makes ``1 <= side <= SIDES`` checkable by the table itself, with no mesh in
    #: sight.  ``0`` on the base means "rung unknown, upper bound unchecked".
    SIDES: ClassVar[int] = 0

    #: What the rung calls the thing an element id names -- lines / quads / hexes going
    #: up the ladder.  Fixed by the subclass for the same reason :attr:`SIDES` is, so
    #: the "only 3 quads" error needs no noun passed in from the mesh.
    ELEMENT: ClassVar[str] = "elements"

    elements: IntArray
    sides: IntArray
    tags: StrArray

    def __post_init__(self) -> None:
        cls = type(self).__name__
        e = np.asarray(self.elements, dtype=np.int64).reshape(-1)
        s = np.asarray(self.sides, dtype=np.int64).reshape(-1)
        t = _str_array(self.tags)
        if not (e.shape[0] == s.shape[0] == t.shape[0]):
            raise ValueError(
                "%s: elements (%d), sides (%d) and tags (%d) must have the same length"
                % (cls, e.shape[0], s.shape[0], t.shape[0]))
        if e.shape[0]:
            if int(e.min()) < 0:
                raise ValueError("%s: negative element id %d"
                                 % (cls, int(e.min())))
            lo, hi = int(s.min()), int(s.max())
            top = self.SIDES or hi
            if lo < 1 or hi > top:
                raise ValueError("%s: side %d is outside 1..%d"
                                 % (cls, lo if lo < 1 else hi, top))
        object.__setattr__(self, "elements", _frozen(e))
        object.__setattr__(self, "sides", _frozen(s))
        object.__setattr__(self, "tags", _frozen(t))

    def check_within(self, n_elements: int) -> None:
        """Raise if any row names an element the mesh does not have."""
        if not len(self):
            return
        hi = int(self.elements.max())
        if hi >= n_elements:
            raise ValueError("%s names element %d but there are only %d %s"
                             % (type(self).__name__, hi, n_elements, self.ELEMENT))

    # -- construction ----------------------------------------------------
    @classmethod
    def empty(cls: type[T]) -> T:
        """The no-rows table."""
        return cls(np.zeros(0, np.int64), np.zeros(0, np.int64), _empty_str())

    @classmethod
    def from_pairs(cls: type[T], rows: Sequence[Sequence[int]] | IntArray,
                   tags: Sequence[str] | StrArray) -> T:
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
        return "<%s %d rows {%s}>" % (type(self).__name__, len(self),
                                      ",".join(self.group_tags))

    # -- operations ------------------------------------------------------
    def ordered(self: T) -> T:
        """The rows stably sorted by ``(element, side)`` -- the canonical storage order.
        """
        if not len(self):
            return self
        p = np.lexsort((self.sides, self.elements))
        return type(self)(self.elements[p], self.sides[p], self.tags[p])

    def offset(self: T, delta: int) -> T:
        """The same rows with every element id shifted by ``delta`` (for ``merge``)."""
        if not len(self):
            return self
        return type(self)(self.elements + int(delta), self.sides, self.tags)

    @classmethod
    def concat(cls: type[T], tables: Sequence[T]) -> T:
        """The tables' rows end to end, order preserved."""
        parts = [t for t in tables if len(t)]
        if not parts:
            return cls.empty()
        return cls(
            np.concatenate([t.elements for t in parts]),
            np.concatenate([t.sides for t in parts]),
            np.concatenate([t.tags for t in parts]))

    def mask_for(self, tag: str) -> BoolArray:
        """Boolean mask of the rows named ``tag``."""
        return np.asarray(self.tags == tag, dtype=bool)

    def select(self: T, mask: BoolArray) -> T:
        """The rows where ``mask`` is True, order preserved."""
        m = np.asarray(mask, dtype=bool)
        return type(self)(self.elements[m], self.sides[m], self.tags[m])

    def count(self, tag: str) -> int:
        """How many rows are named ``tag``."""
        return int(np.count_nonzero(self.tags == tag))


class PointTags(SideTags):
    """Tagged **end points** of a ``LineMesh``'s lines: ``side`` 1-2 -> local vertex
    ``side - 1``. See :class:`SideTags` for the shared row semantics."""

    SIDES: ClassVar[int] = 2
    ELEMENT: ClassVar[str] = "lines"


class EdgeTags(SideTags):
    """Tagged **edges** of a ``QuadMesh``'s quads: ``side`` 1-4 -> the edge
    ``EDGE_POINTS[side - 1]``.  See :class:`SideTags` for the shared row semantics."""

    SIDES: ClassVar[int] = 4
    ELEMENT: ClassVar[str] = "quads"


class FaceTags(SideTags):
    """Tagged **faces** of a ``HexMesh``'s hexes: ``side`` 1-6 -> the face
    ``FACE_POINTS[side - 1]``. See :class:`SideTags` for the shared row semantics."""

    SIDES: ClassVar[int] = 6
    ELEMENT: ClassVar[str] = "hexes"


class TagBuilder(Generic[T]):
    """Accumulates side-tag rows one at a time, then builds them into ``table_type``."""

    def __init__(self, table_type: type[T]) -> None:
        self._type: type[T] = table_type
        self._elements: list[int] = []
        self._sides: list[int] = []
        self._tags: list[str] = []

    def add(self, element: int, side: int, tag: str) -> None:
        """Append one row unconditionally."""
        self._elements.append(int(element))
        self._sides.append(int(side))
        self._tags.append(str(tag))

    def add_if_tagged(self, element: int, side: int, tag: str) -> None:
        """Append one row only if ``tag`` is non-empty."""
        if tag:
            self.add(element, side, tag)

    def extend(self, table: T) -> None:
        """Append every row of ``table``, in its order."""
        for e, s, t in table:
            self.add(e, s, t)

    def __len__(self) -> int:
        return len(self._elements)

    def __bool__(self) -> bool:
        return bool(self._elements)

    def build(self) -> T:
        """The rows as stored, in insertion order."""
        if not self._elements:
            return self._type.empty()
        return self._type(np.asarray(self._elements, dtype=np.int64),
                          np.asarray(self._sides, dtype=np.int64),
                          _str_array(self._tags))

    def build_ordered(self) -> T:
        """:meth:`build` then :meth:`SideTags.ordered`."""
        return self.build().ordered()


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

    def check_within(self, n_elements: int) -> None:
        """Raise if any tagged id names an element the mesh does not have."""
        if len(self) and int(self.ids[-1]) >= n_elements:
            raise ValueError("element_tags names element %d but there are only %d "
                             "elements" % (int(self.ids[-1]), n_elements))



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
