"""Named boundary conditions: a face tag name and its Nek5000 BC code."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import Union

# What one side of a boundary carries: a ``cbc`` (Nek5000's name for the 3-char
# boundary-condition code, ``"W  "``), a ``boundaryID`` (an int), or ``(cbc, boundaryID)``
# with either left ``None``. Whatever is not stated takes its default, ``DEFAULT_CBC``
# and ``DEFAULT_BOUNDARY_ID`` -- but only if the whole table leaves it out; see
# :class:`BoundaryConditions`.
Side = Union[str, int, tuple[Union[str, None], Union[int, None]]]

# One boundary: a ``Side`` read the same from every side of the face, or a
# ``{region: Side | None}`` mapping read against the ``element_tags`` of the element
# each row is written for (``""`` for an untagged one).
#
# A face is one shared object with one name, so an asymmetric boundary condition
# cannot live on the face -- it lives in the *regions* either side of it. A
# conjugate interface is a face between a ``"fluid"`` hex and a ``"solid"`` one;
# naming it once and giving it ``{"fluid": "W  ", "solid": None}`` writes the
# fluid's wall condition and nothing at all on the solid side.  A ``None`` value
# emits no row for that region, which is how a face gets a condition from one
# side only.
BCSpec = Union[Side, Mapping[str, Union[Side, None]]]

DEFAULT_CBC = "E  "
DEFAULT_BOUNDARY_ID = 0


def _pad(cbc: str) -> str:
    return (cbc + "   ")[:3]


def _parts(spec: Side) -> tuple[str | None, int | None]:
    """``(cbc, boundaryID)`` as stated, ``None`` for what is left out."""
    if isinstance(spec, str):
        return spec, None
    if isinstance(spec, int):
        return None, spec
    return spec


def _table(field: str, table: Mapping[str, BCSpec]
           ) -> tuple[dict[str, tuple[str, int] | dict[str, tuple[str, int] | None]], bool]:
    """One field's table with every entry resolved to ``(cbc, boundaryID)``, and whether
    it states ``boundaryID`` at all.

    ``cbc`` and ``boundaryID`` are each all-or-none across the table: one name stating
    a ``cbc`` means every name must, since a table that codes some boundaries and
    silently leaves the rest ``'E  '`` is far more likely a forgotten name than an
    intent. A stated ``boundaryID`` set must be ``1..k`` with no gap -- ids are groups,
    so several names may share one, but none may be skipped. ``0`` is always allowed
    beside them: it is the id of "no group", and ``"E  "`` the code of "no condition",
    so stating either clears that boundary on purpose."""
    raw: dict[str, tuple[str | None, int | None]
              | dict[str, tuple[str | None, int | None] | None]] = {
        name: _parts(spec) if not isinstance(spec, Mapping) else {
            r: None if v is None else _parts(v) for r, v in spec.items()}
        for name, spec in table.items()}
    sides = [(name if isinstance(e, tuple) else "%s[%r]" % (name, r), x)
             for name, e in raw.items()
             for r, x in ([(None, e)] if isinstance(e, tuple) else e.items())
             if x is not None]
    for i, label in ((0, "cbc"), (1, "boundaryID")):
        missing = [n for n, x in sides if x[i] is None]
        if missing and len(missing) < len(sides):
            raise ValueError(
                "%s: some boundaries state a %s and some do not -- state it for every "
                "one or for none. Missing: %s."
                % (field, label, ", ".join(missing)))
    ids = sorted({x[1] for _, x in sides if x[1] is not None} - {0})
    if ids != list(range(1, len(ids) + 1)):
        raise ValueError(
            "%s: boundaryIDs must run 1..k with none skipped (names may share an id; "
            "0 means none); got %s." % (field, ids))

    def resolve(x: tuple[str | None, int | None]) -> tuple[str, int]:
        return (DEFAULT_CBC if x[0] is None else _pad(x[0]),
                DEFAULT_BOUNDARY_ID if x[1] is None else int(x[1]))
    resolved = {name: resolve(e) if isinstance(e, tuple) else {
                    r: None if x is None else resolve(x) for r, x in e.items()}
                for name, e in raw.items()}
    return resolved, any(x[1] is not None for _, x in sides)


class BoundaryConditions:
    """Boundary names to Nek BC codes, one table per solved field.

    ``velocity`` is the momentum field and is required. ``temperature`` is the second
    field Nek's ``.re2`` carries once a mesh has a solid region (``nelgv < nelgt``);
    it covers every element, and a name left out of it stays conformal (``'E  '``).
    Each table maps a name to a ``BCSpec``.

    Each entry carries a ``cbc`` and a ``boundaryID`` (see ``Side``). Within one
    table each is **all-or-none**: state it for every boundary or for none, and a table
    that states none takes the defaults. A table's ``boundaryID`` set is its own --
    velocity's and temperature's are separate -- and runs ``1..k`` without a gap, though
    several boundaries may share one. ``.vtu`` paints it as ``bc_id``; a table that
    states none is painted by :meth:`tag_of` instead, so its boundaries still differ.

    There are deliberately **no presets** here. A name-to-code table is a statement
    about one piece of geometry -- which opening is the inlet, which surface is a
    measurement plane -- so it belongs in the mesher that knows, next to the tags it
    names, where a reader meets it at the same time as the mesh. A built-in
    ``nek_default()`` put the carotid's own vocabulary in the toolkit and let two other
    examples inherit a mapping neither of them stated.

    Periodicity is not stated here: a ``'P  '`` code needs a *correspondence* between
    two named face groups, which ``periodic=`` supplies to the writer."""

    def __init__(self, velocity: Mapping[str, BCSpec],
                 temperature: Mapping[str, BCSpec] | None = None) -> None:
        self._sides: dict[str, dict[str, tuple[str, int]
                                    | dict[str, tuple[str, int] | None]]] = {}
        self._numbered: dict[str, bool] = {}
        self._names: dict[str, None] = {}
        for field, table in (("velocity", velocity), ("temperature", temperature)):
            if table is None:
                continue
            self._sides[field], self._numbered[field] = _table(field, table)
            self._names.update(dict.fromkeys(table))

    @property
    def fields(self) -> tuple[str, ...]:
        """The fields stated, velocity first."""
        return tuple(self._sides)

    def tag_of(self, field: str, name: str) -> int:
        """The 1-based position of ``name`` in ``field``'s own table: the id a viewer
        paints when the table states no ``boundaryID``."""
        return list(self._sides[field]).index(name) + 1

    def numbered(self, field: str) -> bool:
        """Whether ``field``'s table states ``boundaryID`` (for every boundary, since it
        is all-or-none)."""
        return self._numbered[field]

    def has(self, field: str, name: str) -> bool:
        """Whether ``field``'s table states ``name``."""
        return name in self._sides.get(field, ())

    def side(self, field: str, name: str, region: str) -> tuple[str, int] | None:
        """``(cbc, boundaryID)`` ``field`` writes for a row of boundary ``name`` owned
        by an element in ``region``; ``None`` means "write no row from this side".

        A name with one ``Side`` reads the same from everywhere. One with per-region
        entries must name every region it actually borders, since a missing key is far
        more likely a typo than an intent to drop the face."""
        spec = self._sides[field][name]
        if isinstance(spec, tuple):
            return spec
        if region not in spec:
            raise ValueError(
                "%s: boundary %r has per-region entries for %s, but borders an element "
                "in region %r. Give that region an entry, or None to write no row there."
                % (field, name, ", ".join(repr(k) for k in sorted(spec)) or "no region",
                   region))
        return spec[region]

    def names_with(self, field: str, cbc: str) -> set[str]:
        """Names ``field`` gives ``cbc`` from at least one side."""
        cbc = _pad(cbc)
        return {n for n, spec in self._sides[field].items()
                if (spec[0] == cbc if isinstance(spec, tuple)
                    else any(v is not None and v[0] == cbc for v in spec.values()))}

    def __contains__(self, name: object) -> bool:
        return name in self._names

    def __iter__(self) -> Iterator[str]:
        return iter(self._names)

    def __len__(self) -> int:
        return len(self._names)

    def __repr__(self) -> str:
        return "BoundaryConditions(%s)" % ", ".join(
            "%s=%r" % (f, t) for f, t in self._sides.items())
