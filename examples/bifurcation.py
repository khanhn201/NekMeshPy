"""Run the bifurcation vessel surface mesher (the original algorithm).

Run with::

    python examples/bifurcation.py

This drives the full surface pipeline with the bundled geometry and the default
configuration, writing ``bifurcation.re2`` / ``.rea`` / ``.vtk``.
"""

import logging

from nekmeshpy import BifurcationMesher, Config

logging.basicConfig(level=logging.INFO, format="%(message)s")

cfg = Config()
cfg.interior_method = "winslow"   # bilinear | harmonic | harmonic3d | winslow
cfg.plot = False

mesh = BifurcationMesher(cfg).run()
print("bifurcation: %d hex elements" % mesh.n_elements)
