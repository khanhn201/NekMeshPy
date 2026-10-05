"""Reading a ``loft``'s three tag arguments into stored tables."""

from __future__ import annotations

import numpy as np

from .._typing import StrArray
from .tags import Tags

__all__ = ["sweep_tags", "sweep_cap"]


def _check_slice(spec: Tags, n_slice: int) -> None:
    if spec.size != int(n_slice):
        raise ValueError("a Tags handed to a loft must be over the %d elements of one "
                         "slice, got one over %d" % (n_slice, spec.size))


def sweep_tags(spec: str | Tags | None, n_layers: int, n_slice: int) -> Tags:
    """The swept elements' region tags from a ``loft``'s ``element_tags`` argument.

    ``None`` tags nothing, a ``str`` tags every swept element, and an
    :class:`Tags` over one slice's ``n_slice`` elements tags each element by
    the slice element it was extruded from (element ``i*n_slice + k``)."""
    if spec is None:
        return Tags.empty(int(n_layers) * int(n_slice))
    if isinstance(spec, str):
        return Tags.full(int(n_layers) * int(n_slice), spec)
    if isinstance(spec, Tags):
        _check_slice(spec, n_slice)
        return spec.tile(int(n_layers))
    raise TypeError(
        "element_tags must be a tag string, a Tags over the %d elements "
        "of one slice, or None; got %s"
        % (n_slice, type(spec).__name__))


def sweep_cap(spec: str | Tags | None, default: Tags, n_slice: int) -> StrArray:
    """One cap's dense ``(n_slice,)`` tag row from a ``loft``'s ``first_tag`` /
    ``last_tag`` argument, falling back to ``default`` (the bounding slice's own
    element tags -- a cap side *is* that slice element) when the argument is ``None``.
    """
    if spec is None:
        return default.to_dense()
    if isinstance(spec, str):
        return np.full(int(n_slice), spec)
    if isinstance(spec, Tags):
        _check_slice(spec, n_slice)
        return spec.to_dense()
    raise TypeError(
        "a cap tag must be a tag string, a Tags over the %d elements of "
        "one slice, or None; got %s" % (n_slice, type(spec).__name__))
