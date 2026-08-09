"""Where a sweep's stations are, and what lays out across them.

Rung-agnostic by nature: a *station* is a parameter along the sweep, and a *level* is the
copy of the slice sitting at one.  Validating a sweep's ``fractions``, refining them onto
the node lattice, sampling a path at them, and laying a slice's entity table out across
them are all that one subject, which belongs to no rung in particular -- every ``loft`` /
``sweep`` / ``extrude`` on the ladder reads from here.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Protocol, TypeVar

import numpy as np

from .._typing import FloatArray, IntArray, PointArray
from .conform import entity_tol
from .fields import gll_nodes


class Sampled(Protocol):
    """What an evaluated sweep needs of a slice, at any rung.  The three containers
    share no base class, so this is structural -- it names the reads, nothing more."""

    @property
    def order(self) -> int: ...
    @property
    def n_points(self) -> int: ...
    @property
    def points(self) -> PointArray: ...


M = TypeVar("M", bound=Sampled)


def split_evaluated(profs: Sequence[M], t: FloatArray, order: int, *, loop: bool,
                    conn: Callable[[M], IntArray], noun: str, elems: str,
                    name: str) -> tuple[list[M], list[list[M]]]:
    """Validate a sweep whose slices were **evaluated** on the refined node lattice
    rather than handed in, and split them into ``(slices, sweep_nodes)``.

    Every rung above the line does this identically -- one slice per lattice value, all
    index-paired with the first, the wrap level dropped on a loop -- so the only things
    that vary are what to compare (``conn``, which doubles as the element count) and what
    to call the pieces in the errors (``noun`` / ``elems``).  Those nouns are not
    decoration: they are what the rungs' own tests match on."""
    slices = list(profs)
    if len(slices) != t.shape[0]:
        raise ValueError(
            "%s: expected one %s per sweep lattice value (%d), got %d"
            % (name, noun, t.shape[0], len(slices)))
    nz = (len(slices) - 1) // order
    ref = slices[0]
    ref_conn = conn(ref)
    for k, m in enumerate(slices):
        if m.order != order:
            raise ValueError(
                "%s: f(%g) returned an order-%d %s, but order=%d was requested"
                % (name, t[k], m.order, noun, order))
        if m.n_points != ref.n_points or not np.array_equal(conn(m), ref_conn):
            raise ValueError(
                "%s: every %s must be index-paired and conformal with the first, but "
                "f(%g) returned %d points / %d %s against f(%g)'s %d / %d.  Place one "
                "%s with the affine ops rather than rebuilding it per parameter."
                % (name, noun, t[k], m.n_points, conn(m).shape[0], elems, t[0],
                   ref.n_points, ref_conn.shape[0], noun))

    if loop:
        # the wrap level must land back on level 0; drop it, but keep the seam layer's
        # own intermediate levels -- they are what curve the seam.
        P0: PointArray = np.asarray(ref.points, dtype=float).reshape(-1, 3)
        Pw: PointArray = np.asarray(slices[-1].points, dtype=float).reshape(-1, 3)
        gap = float(np.max(np.linalg.norm(Pw - P0, axis=1))) if P0.size else 0.0
        tol = entity_tol(P0)
        if gap > tol:
            raise ValueError(
                "%s(loop=True) needs the last fraction to map back to the first %s, "
                "but f(%g) and f(%g) are %g apart (tolerance %g).  Pass the trailing "
                "wrap value as the final fraction, or use loop=False."
                % (name, noun, t[-1], t[0], gap, tol))
        slices = slices[:-1]

    return (slices[::order],
            [slices[i * order + 1:(i + 1) * order] for i in range(nz)])


def refined_lattice(fractions: FloatArray, order: int) -> FloatArray:
    """The ``n*order + 1`` parameter positions of **every** node of the order-``order``
    chain graded by ``fractions`` (``n = len(fractions) - 1`` elements): element ``i``'s
    node ``a`` sits at ``fr[i] + g[a]*(fr[i+1] - fr[i])`` for the GLL nodes ``g`` on
    ``[0, 1]``, and the chain ends at ``fr[-1]``."""
    g: FloatArray = gll_nodes(order)
    fr = fractions
    u: FloatArray = (fr[:-1, None]
                     + g[None, :order] * np.diff(fr)[:, None]).ravel()
    return np.concatenate([u, fr[-1:]])


def check_fraction_count(fr: FloatArray, *, loop: bool, name: str) -> None:
    """Raise unless ``fr`` carries enough fractions for at least one sweep layer (two,
    with ``loop=True`` -- the last fraction is the wrap back onto the first profile
    rather than a level of its own)."""
    if fr.shape[0] - 1 < 1:
        raise ValueError("%s needs at least 2 fractions (one layer), got %d"
                         % (name, fr.shape[0]))
    if loop and fr.shape[0] - 1 < 2:
        raise ValueError(
            "%s(loop=True) needs at least 3 fractions (two layers), got %d -- the "
            "last one is the wrap back to the first profile, so it is not a level of "
            "its own" % (name, fr.shape[0]))


def sweep_lattice(fractions: FloatArray, order: int, *, loop: bool,
                   name: str) -> tuple[FloatArray, FloatArray]:
    """Validate a sweep's ``fractions`` and return ``(fr, node lattice)``."""
    fr: FloatArray = np.atleast_1d(np.asarray(fractions, dtype=float))
    check_fraction_count(fr, loop=loop, name=name)
    return fr, refined_lattice(fr, order)


def sweep_path(path: Callable[[FloatArray], PointArray],
                tangent: Callable[[FloatArray], PointArray] | None,
                tv: FloatArray) -> tuple[PointArray, PointArray | None]:
    """Sample a sweep's centreline (and, if given, its analytic derivative) on the
    station parameters ``tv``, as ``(K,3)`` arrays."""
    P: PointArray = np.asarray(path(tv), dtype=float)
    if P.shape != (tv.shape[0], 3):
        raise ValueError("sweep: path must map the (%d,) sweep lattice to a (%d,3) "
                         "array of centreline points, got %s"
                         % (tv.shape[0], tv.shape[0], (P.shape,)))
    if tangent is None:
        return P, None
    T: PointArray = np.asarray(tangent(tv), dtype=float)
    if T.shape != (tv.shape[0], 3):
        raise ValueError("sweep: tangent must map the (%d,) sweep lattice to a "
                         "(%d,3) array of unit tangents, got %s"
                         % (tv.shape[0], tv.shape[0], (T.shape,)))
    return P, T / np.linalg.norm(T, axis=1)[:, None]


def at_levels(table: IntArray, level: IntArray, per_level: int) -> IntArray:
    """``table`` laid out at every ``level`` of a sweep -- ``level * per_level + table``,
    level-major, keeping ``table``'s trailing shape:
    ``(n_level,) x (k, ...) -> (n_level * k, ...)``.

    Every id a sweep writes is one of these: a section point / edge / quad id, shifted by
    how many of that entity one level owns.  The ``loft`` bodies above pick the table and
    the levels; this is the layout they share."""
    shift: IntArray = level.reshape((-1,) + (1,) * table.ndim) * per_level
    return (table[None] + shift).reshape((-1,) + table.shape[1:])


__all__ = [
    "Sampled",
    "at_levels",
    "check_fraction_count",
    "refined_lattice",
    "sweep_lattice",
    "split_evaluated",
    "sweep_path",
]
