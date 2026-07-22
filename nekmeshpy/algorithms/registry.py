"""Hex-meshing algorithm interface and registry.

Every hex-generation strategy -- the bifurcation template, a transfinite block,
a future tube/extrude mesher -- implements the same tiny contract: a ``run()``
that returns a :class:`~nekmeshpy.geometry.hexmesh.HexMesh`.  Registering algorithms by
name lets the CLI and user code select one generically, gmsh-style.

    from nekmeshpy.algorithms.registry import make
    mesh = make("transfinite_block",
                corners=[...], divisions=(4, 4, 8)).run()
"""

from __future__ import annotations

from typing import Any, Callable, Protocol, runtime_checkable


@runtime_checkable
class HexAlgorithm(Protocol):
    """Anything that can produce a hex mesh."""

    def run(self) -> Any:
        """Build and return a :class:`~nekmeshpy.geometry.hexmesh.HexMesh`."""
        ...


ALGORITHMS: dict[str, type] = {}


def register_algorithm(name: str) -> Callable[[type], type]:
    """Class decorator: register a :class:`HexAlgorithm` implementation."""
    def deco(cls: type) -> type:
        ALGORITHMS[name.lower()] = cls
        return cls
    return deco


def available() -> list[str]:
    """Sorted names of registered algorithms."""
    return sorted(ALGORITHMS)


def get(name: str) -> type:
    """Return the registered algorithm class for ``name``."""
    cls = ALGORITHMS.get(name.lower())
    if cls is None:
        raise ValueError("unknown algorithm %r (available: %s)"
                         % (name, ", ".join(available())))
    return cls


def make(name: str, *args: Any, **kwargs: Any) -> HexAlgorithm:
    """Instantiate a registered algorithm by name."""
    return get(name)(*args, **kwargs)
