"""Named physical groups (gmsh-style) and their Nek5000 boundary-condition codes."""

from __future__ import annotations

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

    def __post_init__(self) -> None:
        if len(self.code) != 3:
            object.__setattr__(self, "code", (self.code + "   ")[:3])


class PhysicalGroups:
    """Bidirectional registry of :class:`PhysicalGroup` entries."""

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

    # -- presets ---------------------------------------------------------
    @classmethod
    def nek_default(cls) -> "PhysicalGroups":
        """The original built-in tag->code table."""
        return cls([
            PhysicalGroup("wall",         1, 2, "W  "),
            PhysicalGroup("trunk_outlet", 2, 2, "v  "),
            PhysicalGroup("top_outlet_1", 3, 2, "int"),
            PhysicalGroup("top_outlet_2", 4, 2, "O  "),
            PhysicalGroup("flux_1",       5, 2, "f1 "),
            PhysicalGroup("flux_2",       6, 2, "f2 "),
        ])

    @classmethod
    def duct(cls, wall: int = 1, inlet: int = 2, outlet: int = 3) -> "PhysicalGroups":
        """Wall / inlet / outlet registry for a duct or pipe (codes ``W``/``v``/``O``)."""
        return cls([
            PhysicalGroup("wall",   wall,   2, "W  "),
            PhysicalGroup("inlet",  inlet,  2, "v  "),
            PhysicalGroup("outlet", outlet, 2, "O  "),
        ])

    @classmethod
    def from_tags(cls, tag_wall: int = 1, tag_trunk: int = 2, tag_top1: int = 3,
                  tag_top2: int = 4, tag_f1: int = 5, tag_f2: int = 6
                  ) -> "PhysicalGroups":
        """Build the carotid registry from explicit boundary tags."""
        return cls([
            PhysicalGroup("wall",         tag_wall,  2, "W  "),
            PhysicalGroup("trunk_outlet", tag_trunk, 2, "v  "),
            PhysicalGroup("top_outlet_1", tag_top1,  2, "int"),
            PhysicalGroup("top_outlet_2", tag_top2,  2, "O  "),
            PhysicalGroup("flux_1",       tag_f1,    2, "f1 "),
            PhysicalGroup("flux_2",       tag_f2,    2, "f2 "),
        ])
