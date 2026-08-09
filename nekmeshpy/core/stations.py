"""Where a sweep's stations are, and what lays out across them.

Rung-agnostic by nature: a *station* is a parameter along the sweep, and a *level* is the
copy of the slice sitting at one.  Validating a sweep's ``fractions``, refining them onto
the node lattice, sampling a path at them, and laying a slice's entity table out across
them are all that one subject, which belongs to no rung in particular -- every ``loft`` /
``sweep`` / ``extrude`` on the ladder reads from here.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from .._typing import FloatArray, IntArray, PointArray
from .fields import gll_nodes


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
    "at_levels",
    "check_fraction_count",
    "refined_lattice",
    "sweep_lattice",
    "sweep_path",
]
