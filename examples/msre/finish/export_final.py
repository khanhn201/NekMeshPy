"""Report + export a finished (tagged, linear) model.  usage: export_final.py in.pkl OUTNAME [--vtu]"""
import sys, pickle, time
from nekmeshpy import hexmesh
from nekmeshpy.io.writer import to_re2, to_vtu
t0 = time.time()
m = pickle.load(open(sys.argv[1], "rb"))
print(hexmesh.report(m), flush=True)
to_re2(m, sys.argv[2] + ".re2", bc={"inlet": ("v  ", 1), "outlet": ("O  ", 2), "wall": ("W  ", 3), "channel": ("W  ", 4)})
pickle.dump(m, open(sys.argv[2] + ".pkl", "wb"))
if "--vtu" in sys.argv:
    to_vtu(m, sys.argv[2] + ".vtu")
print("exported %s (%.0fs)" % (sys.argv[2], time.time() - t0))
