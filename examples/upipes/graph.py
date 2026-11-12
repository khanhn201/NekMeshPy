"""Embedded Voronoi graph: edges as polylines, vertices as clusters of edge ends.

Approximate graph (paper, sec. 2): elements smaller than TAU are merged into one node.
"""
import numpy as np, pickle
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import minimum_spanning_tree, connected_components, shortest_path
from geom import PLATE
from tubes import TUBES

TAU = 1.5          # approximate Voronoi graph tolerance
MERGE = 6.0        # search radius when joining edge ends into vertices
LINK = 1.2         # edge point connectivity


def gname(c):
    c = int(c)
    if c >= 2000: return f"F{c-2000}"
    if c >= 1000: return f"S{c-1000}"
    if c < 0: return ["x-", "x+", "z-", "z+", "y+"][-1 - c]
    return str(c)


def order_chain(P):
    """longest path through the MST of the point cloud"""
    n = len(P)
    if n < 3:
        return P
    t = cKDTree(P)
    d, nb = t.query(P, k=min(10, n))
    r = np.repeat(np.arange(n), nb.shape[1] - 1)
    c = nb[:, 1:].ravel(); w = d[:, 1:].ravel() + 1e-9
    G = coo_matrix((w, (r, c)), shape=(n, n)).tocsr()
    T = minimum_spanning_tree(G + G.T)
    T = T + T.T
    d0 = shortest_path(T, indices=0, unweighted=False)
    a = int(np.argmax(np.where(np.isfinite(d0), d0, -1)))
    da, pred = shortest_path(T, indices=a, return_predecessors=True)
    b = int(np.argmax(np.where(np.isfinite(da), da, -1)))
    path = [b]
    while path[-1] != a:
        path.append(pred[path[-1]])
    return P[path]


def resample(C, h=0.5):
    L = np.r_[0, np.cumsum(np.linalg.norm(np.diff(C, axis=0), axis=1))]
    if L[-1] < 1e-9:
        return C[:1]
    # light smoothing of the jagged pixel order
    k = 5
    if len(C) > 2 * k:
        Cs = C.copy()
        for j in range(3):
            Cs[1:-1] = (Cs[:-2] + Cs[1:-1] + Cs[2:]) / 3
        C = Cs
        L = np.r_[0, np.cumsum(np.linalg.norm(np.diff(C, axis=0), axis=1))]
    n = max(int(np.ceil(L[-1] / h)), 1)
    u = np.linspace(0, L[-1], n + 1)
    return np.stack([np.interp(u, L, C[:, k]) for k in range(3)], 1)


def build():
    D = np.load("maps/edgepts.npz")
    keys, pts, src = [], [], []
    for pid in sorted(TUBES):
        keys.append(D[f"key_{pid}"]); pts.append(D[f"E_{pid}"]); src.append(np.full(len(D[f"E_{pid}"]), pid))
    keys = np.concatenate(keys); pts = np.concatenate(pts); src = np.concatenate(src)
    edges = []
    for k in np.unique(keys, axis=0):
        m = np.all(keys == k, axis=1)
        # one strand only: points from the map that sees this edge best
        srcs, cnt = np.unique(src[m], return_counts=True)
        m &= src == srcs[cnt.argmax()]
        P = pts[m]
        t = cKDTree(P)
        pr = t.query_pairs(LINK, output_type="ndarray")
        G = coo_matrix((np.ones(len(pr)), (pr[:, 0], pr[:, 1])), shape=(len(P), len(P)))
        nc, cl = connected_components(G, directed=False)
        for c in range(nc):
            Q = P[cl == c]
            if len(Q) < 3:
                continue
            C = resample(order_chain(Q))
            Lc = np.linalg.norm(np.diff(C, axis=0), axis=1).sum()
            edges.append(dict(gov=tuple(int(v) for v in k), pts=C, L=Lc, n=len(Q)))
    # vertices: cluster edge ends.  Two ends join when they are close, their edges
    # share at least two governors, and they are not the two ends of one real edge.
    ends = np.array([e["pts"][[0, -1]] for e in edges]).reshape(-1, 3)
    plate = np.abs(ends[:, 1] - PLATE) < 1.0
    ends[plate, 1] = PLATE
    t = cKDTree(ends)
    pr = t.query_pairs(MERGE, output_type="ndarray")
    keep = []
    for i, j in pr:
        ei, ej = i // 2, j // 2
        if plate[i] or plate[j]:
            continue
        if ei == ej and edges[ei]["L"] > TAU:
            continue
        if len(set(edges[ei]["gov"]) & set(edges[ej]["gov"])) >= 2:
            keep.append((i, j))
    keep = np.array(keep).reshape(-1, 2)
    G = coo_matrix((np.ones(len(keep)), (keep[:, 0], keep[:, 1])), shape=(len(ends), len(ends)))
    nc, cl = connected_components(G, directed=False)
    verts = []
    for c in range(nc):
        idx = np.where(cl == c)[0]
        X = ends[idx].mean(0)
        govs = set()
        for i in idx:
            govs |= set(edges[i // 2]["gov"])
        verts.append(dict(X=X, ends=idx.tolist(), gov=tuple(sorted(govs)), plate=bool(plate[idx[0]])))
    for e in edges:
        e["v"] = [None, None]
    for vi, v in enumerate(verts):
        for i in v["ends"]:
            edges[i // 2]["v"][i % 2] = vi
            edges[i // 2]["pts"][0 if i % 2 == 0 else -1] = v["X"]
    return edges, verts


if __name__ == "__main__":
    edges, verts = build()
    pickle.dump((edges, verts), open("maps/graph.pkl", "wb"))
    print(len(edges), "edges,", len(verts), "vertices",
          sum(v["plate"] for v in verts), "on the plate")
    # degree / governor counts of interior vertices
    from collections import Counter
    deg = Counter()
    for v in verts:
        if not v["plate"]:
            deg[(len(v["ends"]), len(v["gov"]))] += 1
    print("interior vertices (edge ends, governors):", sorted(deg.items()))
    short = [e for e in edges if e["L"] < 3 * TAU]
    print(len(short), "edges shorter than", 3 * TAU)
    for e in sorted(short, key=lambda e: e["L"])[:30]:
        print("  ", [gname(g) for g in e["gov"]], round(e["L"], 2), e["v"])
    # export
    import meshio
    pts, lines, gid = [], [], []
    for i, e in enumerate(edges):
        o = len(pts)
        pts.extend(e["pts"]); lines.extend([[o + k, o + k + 1] for k in range(len(e["pts"]) - 1)])
        gid.extend([i] * (len(e["pts"]) - 1))
    vo = len(pts)
    pts.extend([v["X"] for v in verts])
    meshio.write("out_graph.vtu", meshio.Mesh(np.array(pts), [("line", np.array(lines)), ("vertex", np.arange(vo, len(pts))[:, None])],
                 cell_data={"id": [np.array(gid), np.arange(len(verts))],
                            "ngov": [np.array([len(edges[g]["gov"]) for g in gid]), np.array([len(v["gov"]) for v in verts])]}))
