"""A cross-section together with the two facts a bare section cannot state about itself:
**which way it faces** and **where its axis is**."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .._typing import Point, Vec3
from .quadmesh import QuadMesh
from .query import plane_normal

#: How far ``normal`` may stray from unit length before ``Port`` refuses it.
NORMAL_TOL = 1e-12


@dataclass(frozen=True, eq=False)
class Port:
    """An open end of a meshed component: its cross-section, the outward direction, the
    axis point, and the nominal radius."""

    #: The cross-section itself.
    section: QuadMesh
    #: Unit vector pointing **out** of the component, along which a connector leaves.
    normal: Vec3
    #: The axis point the section was built about -- deliberately *not* the centroid,
    #: which an O-grid's grading shifts slightly off it.
    center: Point
    #: Nominal radius, for checking that two ports being joined are the same size.
    radius: float

    def __post_init__(self) -> None:
        n = np.asarray(self.normal, dtype=float).reshape(-1)
        c = np.asarray(self.center, dtype=float).reshape(-1)
        if n.shape != (3,):
            raise ValueError("Port: normal must be a (3,) vector, got %s"
                             % (np.shape(self.normal),))
        if c.shape != (3,):
            raise ValueError("Port: center must be a (3,) point, got %s"
                             % (np.shape(self.center),))
        off = abs(float(n @ n) - 1.0)
        if off > NORMAL_TOL:
            raise ValueError(
                "Port: normal %s is not a unit vector (|n|^2 is %.3g off 1). It is used "
                "verbatim as a sweep direction, so a non-unit one rescales the "
                "connector rather than just naming a side." % (np.array2string(n), off))
        if not float(self.radius) > 0.0:
            raise ValueError("Port: radius must be positive, got %g" % self.radius)
        object.__setattr__(self, "normal", n)
        object.__setattr__(self, "center", c)
        object.__setattr__(self, "radius", float(self.radius))

    def reversed(self) -> Port:
        """The same section faced the other way -- for the end of a component that is
        about to be continued *into* rather than out of."""
        return Port(self.section, -self.normal, self.center, self.radius)

    def faces(self, other: Port) -> bool:
        """Whether the two ports point at each other, rather than the same way or
        apart.  The check :func:`hexmesh.bridge <nekmeshpy.hexmesh.lift.bridge>` cannot
        make from geometry alone."""
        return float(self.normal @ other.normal) < 0.0

    def __repr__(self) -> str:
        return ("<Port r=%.4g at %s facing %s, %d quads>"
                % (self.radius, np.array2string(self.center, precision=4),
                   np.array2string(self.normal, precision=4),
                   self.section.quads.shape[0]))


def port(section: QuadMesh, *, outward: Vec3 | Sequence[float],
         center: Point | Sequence[float] | None = None,
         radius: float | None = None) -> Port:
    """A :class:`Port` from a section plus the side that faces out."""
    n = plane_normal(section, hint=outward, check=False)
    c: Point = (np.asarray(section.points, dtype=float).mean(axis=0)
                if center is None else np.asarray(center, dtype=float).reshape(-1))
    r = (float(np.linalg.norm(np.asarray(section.points, dtype=float) - c, axis=1).max())
         if radius is None else float(radius))
    return Port(section, n, c, r)


__all__ = ["NORMAL_TOL", "Port", "port"]
