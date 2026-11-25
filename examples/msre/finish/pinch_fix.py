"""Make an assembled model watertight: at every non-manifold boundary edge (two fluid hexes meeting only
along an edge) drop the more exposed of the two (more boundary faces; then the smaller), tag what it uncovers
'wall', repeat.  usage: pinch_fix.py in.pkl out.pkl"""
import sys, pickle, time, collections
import numpy as np
from nekmeshpy import hexmesh

t0 = time.time()
m = pickle.load(open(sys.argv[1], "rb"))
for it in range(8):
    X = np.asarray(m.points); H = np.asarray(m.corners)
    bfid = np.asarray(hexmesh.boundary_face_ids(m))
    bf = np.where(bfid)[0]
    C = np.asarray(m.quad_mesh.corners)[bf]
    E = np.sort(np.stack([C, np.roll(C, -1, 1)], 2).reshape(-1, 2), 1)
    u, c = np.unique(E, axis=0, return_counts=True)
    op = u[c != 2]
    print("pass %d: %d non-manifold edges (%.0fs)" % (it, len(op), time.time() - t0), flush=True)
    if not len(op):
        break
    hexf = np.asarray(m.hexes)                                       # (E,6) face ids
    nb = bfid[hexf].sum(1)
    inc = collections.defaultdict(set)
    want = set(op.ravel().tolist())
    for e, row in enumerate(H):
        for v in row:
            if v in want:
                inc[v].add(e)
    vol = np.abs(np.linalg.det(np.stack([X[H[:, 1]] - X[H[:, 0]], X[H[:, 3]] - X[H[:, 0]], X[H[:, 4]] - X[H[:, 0]]], 1)))
    drop = set()
    for a, b in op:
        hs_ = sorted(inc[a] & inc[b])
        if len(hs_) < 2 or drop & set(hs_):
            continue
        drop.add(max(hs_, key=lambda e: (nb[e], -vol[e])))
    dr = np.array(sorted(drop))
    cc = X[H[dr]].mean(1)
    print("  dropping %d hexes: r %s z %s" % (len(dr), np.round(np.percentile(np.hypot(*cc[:, :2].T), [0, 50, 100]), 1), np.round(np.percentile(cc[:, 2], [0, 50, 100]), 1)), flush=True)
    keep = np.setdiff1d(np.arange(len(H)), dr)
    m = hexmesh.select(m, keep)
    dense = m.face_tags.to_dense()
    un = np.where(np.asarray(hexmesh.boundary_face_ids(m)) & (dense == ""))[0]
    if len(un):
        m = hexmesh.tag_faces(m, un, "wall")
print(hexmesh.topology_report(m), flush=True)
pickle.dump(m, open(sys.argv[2], "wb"))
print("done (%.0fs)" % (time.time() - t0))
