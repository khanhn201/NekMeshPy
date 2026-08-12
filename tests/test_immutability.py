"""No operation in the toolkit mutates the mesh it is given.

Every op returns a new container. That is easy to believe of the pure ones -- a
``translate`` obviously builds a new mesh -- and was for a long time untrue of the
smoothers, which wrote through ``points`` -- the mesh's own live array -- and
returned the same object. A caller could then drop the return
value and still see the effect, which is exactly what both examples did.

So the checks here are of two kinds: that the ordinary operations leave their input
alone, and that the smoothers *both* leave it alone **and** actually move something --
a smoother that silently did nothing would pass the first half on its own.
"""

import numpy as np
import pytest

from nekmeshpy import hexmesh, linemesh, quadmesh
from nekmeshpy.core.fields import uniform_spacing
from nekmeshpy.quadmesh import smoothing as qsmooth


def _rungs():
    ring = linemesh.circle(1.0, 8)
    sec = quadmesh.ogrid(ring, 2, np.linspace(0.5, 1.0, 3), wall_tag="wall")
    blk = hexmesh.extrude(sec, 1.0, uniform_spacing(2), last_tag="top")
    return ring, sec, blk


@pytest.mark.parametrize("rung", ["line", "quad", "hex"])
def test_operations_leave_their_input_untouched(rung):
    ring, sec, blk = _rungs()
    mesh, ns = {"line": (ring, linemesh), "quad": (sec, quadmesh),
                "hex": (blk, hexmesh)}[rung]
    before = mesh.points.copy()
    ops = [lambda m: ns.translate(m, (1.0, 2.0, 3.0)),
           lambda m: ns.rotate(m, 0.3),
           lambda m: ns.scale(m, 2.0),
           lambda m: ns.mirror(m, (1.0, 0.0, 0.0)),
           lambda m: ns.retag_element(m, {}),
           lambda m: ns.merge([m])]
    for op in ops:
        out = op(mesh)
        assert out is not mesh
        assert np.array_equal(mesh.points, before), op


@pytest.mark.parametrize("smoother", [qsmooth.conduction_section,
                                      qsmooth.winslow_section])
def test_section_smoothers_return_a_new_mesh_and_move_something(smoother):
    """Both halves matter: untouched input, and a result that is genuinely different.

    The interior of an O-grid is not at the smoother's fixed point, so a working
    smoother has to move it; one that returned its input unchanged would satisfy the
    immutability half alone."""
    _, sec, _ = _rungs()
    before = sec.points.copy()
    out = smoother(sec)
    assert out is not sec
    assert np.array_equal(sec.points, before)            # input untouched
    assert not np.allclose(out.points, before)           # and it actually fired
    # the boundary is held fixed; only interior points move
    ring = np.isclose(np.linalg.norm(before[:, :2], axis=1), 1.0)
    assert np.allclose(out.points[ring], before[ring])
