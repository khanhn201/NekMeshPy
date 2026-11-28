"""usage: export_bl.py mesh.pkl OUT [--vtu] -> OUT.re2 (corners + 'm' edge midpoints) [and OUT.vtu]"""
import sys, pickle
from nekmeshpy.io.writer import to_re2, to_vtu
m = pickle.load(open(sys.argv[1], "rb"))
to_re2(m, sys.argv[2] + ".re2", bc={"inlet": ("v  ", 1), "outlet": ("O  ", 2), "wall": ("W  ", 3), "channel": ("W  ", 4)})
if "--vtu" in sys.argv:
    to_vtu(m, sys.argv[2] + ".vtu")
