"""Closed-surface :class:`QuadMesh <nekmeshpy.quadmesh.quadmesh.QuadMesh>` factories.

The watertight shells -- a ``box`` / ``sphere`` and their half forms.  Split from
``region`` because these close on themselves and so have no boundary to fill *into*.
"""

from ._closed import box, half_box, hemisphere, sphere

__all__ = ["box", "half_box", "hemisphere", "sphere"]
