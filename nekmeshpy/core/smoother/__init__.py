"""The smoother engine: everything about smoothing that does not depend on the mesh type.

``core/`` is rung-agnostic, so nothing here imports ``quadmesh`` or ``hexmesh``.  A rung's smoother
(``quadmesh/smoother/``, ``hexmesh/smoother/``) extracts corner edge vectors, boundary chains / patches and
neighbour tables from its connectivity, hands plain arrays to the engine, and re-exports the result under
its own namespace, so ``quadmesh.smoother.optimize`` and ``hexmesh.smoother.optimize`` are thin wrappers over
the one :func:`~nekmeshpy.core.smoother.descent.minimize`.

* ``constraints`` -- what each vertex may do: mobility kinds, targets (surfaces, curves), constraints,
  ``Resolved``, ``snap`` and ``constrain_direction``
* ``metric``      -- the shape metric on corner Jacobians, its objective and analytic gradient
* ``quality``     -- reductions from corner readings to the numbers a report prints
* ``schedule``    -- ``StageLog``, the method ``Registry``, the preset schedules
* ``descent``     -- the conjugate-gradient driver with a line search
* ``session``     -- ``SmoothSession`` on a ``RungAdapter``: propose / accept / undo and the round report
"""

from . import constraints, descent, metric, quality, schedule, session

__all__ = ["constraints", "descent", "metric", "quality", "schedule", "session"]
