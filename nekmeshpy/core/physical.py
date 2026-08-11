"""Named physical groups (gmsh-style) and their Nek5000 boundary-condition codes."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Iterable, Iterator


@dataclass(frozen=True)
class PhysicalGroup:
    """One named physical group: ``name``, integer ``tag``, ``dim`` (2 =
    boundary/surface, 3 = volume), and a 3-char Nek BC ``code``."""

    name: str
    tag: int
    dim: int = 2          # 2 = boundary/surface, 3 = volume
    code: str = "E  "     # Nek BC character code, padded to exactly 3 chars

    #: Per-**region** codes, keyed by the ``element_tags`` name of the element the row
    #: is written for (``""`` for an untagged one).  ``None`` when the group reads the
    #: same from every side, which is every ordinary boundary.
    #:
    #: A face is one shared object with one name, so an asymmetric boundary condition
    #: cannot live on the face -- it lives in the *regions* either side of it. A
    #: conjugate interface is a face between a ``"fluid"`` hex and a ``"solid"`` one;
    #: naming it once and giving it ``{"fluid": "W  ", "solid": None}`` writes the
    #: fluid's wall condition and nothing at all on the solid side.  A ``None`` value
    #: emits no row for that region, which is how a face gets a condition from one
    #: side only.
    side_codes: Mapping[str, str | None] | None = None

    def __post_init__(self) -> None:
        if len(self.code) != 3:
            object.__setattr__(self, "code", (self.code + "   ")[:3])
        if self.side_codes is not None:
            object.__setattr__(self, "side_codes", {
                r: None if c is None else (c + "   ")[:3]
                for r, c in self.side_codes.items()})

    def code_for_side(self, region: str) -> str | None:
        """The code to write for a row owned by an element in ``region``.

        ``None`` means "write no row from this side". A group with no
        :attr:`side_codes` reads the same from everywhere, so its own ``code`` is the
        answer; one with them must name every region it actually borders, since a
        missing key is far more likely a typo than an intent to drop the face."""
        if self.side_codes is None:
            return self.code
        if region not in self.side_codes:
            raise ValueError(
                "boundary %r has per-region codes for %s, but borders an element in "
                "region %r. Give that region a code, or None to write no row there."
                % (self.name,
                   ", ".join(repr(k) for k in sorted(self.side_codes)) or "no region",
                   region))
        return self.side_codes[region]


class PhysicalGroups:
    """Bidirectional registry of :class:`PhysicalGroup` entries.

    There are deliberately **no presets** here. A name-to-code table is a statement
    about one piece of geometry -- which opening is the inlet, which surface is a
    measurement plane -- so it belongs in the mesher that knows, next to the tags it
    names, where a reader meets it at the same time as the mesh. A built-in
    ``nek_default()`` put the carotid's own vocabulary in the toolkit and let two other
    examples inherit a mapping neither of them stated."""

    def __init__(self, groups: Iterable[PhysicalGroup] = ()) -> None:
        self._by_tag: dict[int, PhysicalGroup] = {}
        self._by_name: dict[str, PhysicalGroup] = {}
        for g in groups:
            self.add(g)

    # -- mutation --------------------------------------------------------
    def add(self, group: PhysicalGroup) -> PhysicalGroup:
        """Register ``group``; returns it."""
        if group.tag in self._by_tag:
            raise ValueError("tag %d already registered (%s)"
                             % (group.tag, self._by_tag[group.tag].name))
        if group.name in self._by_name:
            raise ValueError("name %r already registered" % group.name)
        self._by_tag[group.tag] = group
        self._by_name[group.name] = group
        return group

    def define(self, name: str, tag: int, dim: int = 2, code: str = "E  "
               ) -> PhysicalGroup:
        """Convenience: build and register a group in one call."""
        return self.add(PhysicalGroup(name, tag, dim, code))

    # -- lookup ----------------------------------------------------------
    def code_for(self, tag: int) -> str | None:
        """The Nek BC code for ``tag``, or ``None`` if not registered."""
        g = self._by_tag.get(tag)
        return g.code if g is not None else None

    def name_for(self, tag: int) -> str | None:
        """The group name for ``tag``, or ``None`` if not registered."""
        g = self._by_tag.get(tag)
        return g.name if g is not None else None

    def tag_for(self, name: str) -> int | None:
        """The integer tag for group ``name``, or ``None`` if not registered."""
        g = self._by_name.get(name)
        return g.tag if g is not None else None

    def get(self, key: int | str) -> PhysicalGroup | None:
        """Look up by tag (int) or name (str)."""
        if isinstance(key, str):
            return self._by_name.get(key)
        return self._by_tag.get(key)

    # -- container protocol ---------------------------------------------
    def __contains__(self, key: int | str) -> bool:
        """``True`` if a group with this tag (int) or name (str) is registered."""
        return self.get(key) is not None

    def __iter__(self) -> Iterator[PhysicalGroup]:
        """Iterate the registered groups in ascending tag order."""
        return iter(sorted(self._by_tag.values(), key=lambda g: g.tag))

    def __len__(self) -> int:
        """Number of registered groups."""
        return len(self._by_tag)

    def __repr__(self) -> str:
        items = ", ".join("%s=%d(%r)" % (g.name, g.tag, g.code) for g in self)
        return "PhysicalGroups(%s)" % items
