"""Channel boundary layer by dicing one hex sheet.

Seeds: the wall-normal edges of every hex with a face tagged TAG.  The sheet is every hex reached through
topologically parallel edges (an edge class), carried with which end is the wall side.  Each class edge
(u -> v, u on the wall side) is split at
    t = w * min(BL / |uv|, 1/2) + (1 - w) / 2,     w = cosine ramp of the distance from u to the TAG wall (D)
so the layer is BL thick on the wall and relaxes to an ordinary halving wherever the sheet runs on (bands,
heads).  Every sheet hex is split along each sheet axis it carries (tensor split).  Tags are carried from
the parent faces.
usage: dice.py in.pkl out.pkl   env TAG=channel BL=0.762 D=25"""
import os, sys, time, pickle, collections
import numpy as np
from scipy.spatial import cKDTree
from nekmeshpy import HexMesh, hexmesh

TAG = os.environ.get("TAG", "channel")
BL = float(os.environ.get("BL", "0.762"))
D = float(os.environ.get("D", "25"))
LOC = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]])
FP = np.asarray(HexMesh.FACE_POINTS)                       # (6,4) local faces
FAX = np.array([int(np.where((LOC[f] == LOC[f][0]).all(0))[0][0]) for f in FP])     # normal axis of each face
FSIDE = np.array([int(LOC[f][0][a]) for f, a in zip(FP, FAX)])                       # its side (0/1)
AXE = []                                                   # AXE[a] = 4 (low, high) local corner pairs along axis a
for a in range(3):
    AXE.append([(i, j) for i in range(8) for j in range(8)
                if LOC[j][a] == 1 and LOC[i][a] == 0 and (LOC[i] == LOC[j]).sum() == 2])
AXE = np.array(AXE)                                        # (3,4,2)


def sheet(H, seeds):
    """seeds: list of (hex, axis, wall_side).  -> dict (hex, axis) -> wall_side, dict edge-key -> wall node,
    set of conflicting (hex, axis).  Graph nodes are (hex, axis) = 3*hex + axis, linked through every shared
    edge with a parity (do the two hexes run that edge the same way); one multi-source BFS labels the side."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    E = len(H)
    lo = H[:, AXE[:, :, 0]].reshape(E, 12); hi = H[:, AXE[:, :, 1]].reshape(E, 12)
    node = (np.arange(E)[:, None] * 3 + np.repeat(np.arange(3), 4)[None, :]).ravel()
    lo, hi = lo.ravel(), hi.ravel()
    kmin, kmax = np.minimum(lo, hi), np.maximum(lo, hi)
    key = kmin * np.int64(H.max() + 1) + kmax
    ori = (lo != kmin).astype(np.int8)                       # 0: this hex runs the edge low id -> high id
    o = np.argsort(key, kind="stable")
    ks, ns, os_ = key[o], node[o], ori[o]
    same = ks[1:] == ks[:-1]
    a_, b_ = ns[:-1][same], ns[1:][same]
    par = (os_[:-1][same] ^ os_[1:][same]).astype(np.int8)
    keep = a_ != b_
    a_, b_, par = a_[keep], b_[keep], par[keep]
    N = 3 * E
    G = coo_matrix((np.ones(len(a_)), (a_, b_)), shape=(N, N))
    nc, lab = connected_components(G, directed=False)
    seed_nodes = np.array([3 * h + a for h, a, _ in seeds])
    comps = np.unique(lab[seed_nodes])
    insheet = np.isin(lab, comps)
    print("  sheet components %d, nodes %d (%.0f%% of all hex-axes)" % (len(comps), insheet.sum(), 100 * insheet.mean()), flush=True)
    # CSR adjacency restricted to the sheet, with parity
    m_ = insheet[a_]
    A2 = np.r_[a_[m_], b_[m_]]; B2 = np.r_[b_[m_], a_[m_]]; P2 = np.r_[par[m_], par[m_]]
    oo = np.argsort(A2, kind="stable"); A2, B2, P2 = A2[oo], B2[oo], P2[oo]
    ptr = np.searchsorted(A2, np.arange(N + 1))
    lbl = -np.ones(N, np.int8)
    bad = set()
    q = collections.deque()
    for h, a, s0 in seeds:
        n = 3 * h + a
        if lbl[n] < 0:
            lbl[n] = s0; q.append(n)
        elif lbl[n] != s0:
            bad.add((h, a))
    while q:
        n = q.popleft()
        for k in range(ptr[n], ptr[n + 1]):
            m2 = B2[k]; want = lbl[n] ^ P2[k]
            if lbl[m2] < 0:
                lbl[m2] = want; q.append(m2)
            elif lbl[m2] != want:
                bad.add((int(m2) // 3, int(m2) % 3))
    sn = np.where(lbl >= 0)[0]
    side = {(int(n) // 3, int(n) % 3): int(lbl[n]) for n in sn}
    # class edges with their wall node
    hh, aa = sn // 3, sn % 3
    wall = {}
    for j in range(4):
        u = np.where(lbl[sn] == 0, H[hh, AXE[aa, j, 0]], H[hh, AXE[aa, j, 1]])
        v = np.where(lbl[sn] == 0, H[hh, AXE[aa, j, 1]], H[hh, AXE[aa, j, 0]])
        for uu, vv, h_, a2 in zip(u.tolist(), v.tolist(), hh.tolist(), aa.tolist()):
            ek = (min(uu, vv), max(uu, vv))
            if ek in wall and wall[ek] != uu:
                bad.add((h_, a2))
            wall.setdefault(ek, uu)
    return side, wall, bad


def main():
    t0 = time.time()
    m = pickle.load(open(sys.argv[1], "rb"))
    X = np.asarray(m.points, float); H = np.asarray(m.corners); E = len(H)
    HF = np.asarray(m.hexes)                               # (E,6) face ids
    ftag = m.face_tags.to_dense()
    etag = m.element_tags.to_dense() if len(m.element_tags) else None
    he, lf = np.nonzero(ftag[HF] == TAG)
    seeds = [(int(h), int(FAX[f]), int(FSIDE[f])) for h, f in zip(he, lf)]
    print("%d hexes, %d '%s' faces (%.0fs)" % (E, len(seeds), TAG, time.time() - t0), flush=True)
    side, wall, bad = sheet(H, seeds)
    sh = collections.Counter(h for h, a in side)
    print("sheet: %d (hex, axis), %d hexes (%d with 2+ axes), %d class edges, %d conflicts (%.0fs)" % (
        len(side), len(sh), sum(1 for c in sh.values() if c > 1), len(wall), len(bad), time.time() - t0), flush=True)
    # split parameter per class edge
    wn = np.unique(H[he[:, None], FP[lf]])
    tree = cKDTree(X[wn])
    ek = np.array(list(wall.keys()), dtype=np.int64); wu = np.array([wall[(a, b)] for a, b in ek])
    wv = np.where(ek[:, 0] == wu, ek[:, 1], ek[:, 0])
    L = np.linalg.norm(X[wv] - X[wu], axis=1)
    d, _ = tree.query(X[wu])
    w = 0.5 * (1 + np.cos(np.pi * np.clip(d / D, 0, 1)))
    t = w * np.minimum(BL / L, 0.5) + (1 - w) * 0.5
    badedges = {(min(H[h, i], H[h, j]), max(H[h, i], H[h, j])) for h, a in bad for i, j in AXE[a]}
    for n, (a_, b_) in enumerate(ek):
        if (a_, b_) in badedges:
            t[n] = 0.5
    newp = X[wu] + t[:, None] * (X[wv] - X[wu])
    nid = {(int(a_), int(b_)): len(X) + n for n, (a_, b_) in enumerate(ek)}
    print("split t: inside the BL %d edges (t <= %.3f), relaxed %d; thinnest new layer %.3f mm (%.0fs)" % (
        (w > 0.999).sum(), t[w > 0.999].max() if (w > 0.999).any() else 0, (w < 0.001).sum(), (t * L).min(), time.time() - t0), flush=True)
    P = [X, newp]
    Xbase = X
    extra = {}                                             # face / centre points of multi-axis splits

    def pt(key, make):
        if key not in extra:
            extra[key] = len(X) + len(newp) + len(extra)
            P.append(make()[None])
        return extra[key]
    axes_of = collections.defaultdict(dict)
    for (h, a), s in side.items():
        axes_of[h][a] = s
    newH, parent, ptag_rows = [], [], []
    keep = np.ones(E, bool)
    # single-axis hexes, vectorised
    one = [(h, next(iter(ax.items()))) for h, ax in axes_of.items() if len(ax) == 1]
    hs = np.array([h for h, _ in one], np.int64); as_ = np.array([a for _, (a, _) in one], np.int64)
    ss = np.array([s_ for _, (_, s_) in one], np.int64)
    M = np.int64(len(X) + 1)
    code = ek[:, 0] * M + ek[:, 1]; oc = np.argsort(code); code_s = code[oc]
    PART = np.array([[int(np.where((LOC == np.where(np.arange(3) == a, 1 - LOC[q], LOC[q])).all(1))[0][0]) for q in range(8)] for a in range(3)])
    C = H[hs]
    S = np.empty_like(C)
    for q in range(8):
        pq = C[np.arange(len(hs)), PART[as_, q]]
        k_ = np.minimum(C[:, q], pq) * M + np.maximum(C[:, q], pq)
        p_ = np.searchsorted(code_s, k_)
        S[:, q] = len(X) + oc[p_]
    onside = LOC[:, as_].T == ss[:, None]
    near = np.where(onside, C, S); far = np.where(onside, S, C)
    keep[hs] = False
    tg = ftag[HF[hs]]                                     # (n,6)
    fa = FAX[None, :] == as_[:, None]
    tn = np.where(fa & (FSIDE[None, :] != ss[:, None]), "", tg)
    tf = np.where(fa & (FSIDE[None, :] == ss[:, None]), "", tg)
    vH = [near, far]; vpar = [hs, hs]; vtag = [tn, tf]
    for h, ax in axes_of.items():
        if len(ax) == 1:
            continue
        keep[h] = False
        c = H[h]

        def node(ijk):
            # ijk: per axis 0, 'm' (split), 1
            mids = [a for a in range(3) if ijk[a] == "m"]
            if not mids:
                return int(c[np.where((LOC == np.array(ijk)).all(1))[0][0]])
            if len(mids) == 1:
                a = mids[0]
                base = [x for x in ijk]; base[a] = 0; i = int(np.where((LOC == np.array(base)).all(1))[0][0])
                base[a] = 1; j = int(np.where((LOC == np.array(base)).all(1))[0][0])
                return nid[(min(c[i], c[j]), max(c[i], c[j]))]
            idx = [k for k in range(3) if ijk[k] != "m"]
            corners_ = [int(c[q]) for q in range(8) if all(LOC[q][k] == ijk[k] for k in idx)]
            key = tuple(sorted(corners_))
            def make():
                # multilinear point of the sub-cell at the mean split parameter of each split axis
                Xc = Xbase
                par_ = {}
                for a in mids:
                    ts = []
                    for q in range(8):
                        if all(LOC[q][k] == ijk[k] for k in idx) and LOC[q][a] == 0:
                            base = LOC[q].copy(); base[a] = 1
                            j2 = int(np.where((LOC == base).all(1))[0][0])
                            sp_ = newp[nid[(min(c[q], c[j2]), max(c[q], c[j2]))] - len(Xc)]
                            ts.append(np.linalg.norm(sp_ - Xc[c[q]]) / np.linalg.norm(Xc[c[j2]] - Xc[c[q]]))
                    par_[a] = float(np.mean(ts))
                acc = np.zeros(3)
                for q in range(8):
                    if not all(LOC[q][k] == ijk[k] for k in idx):
                        continue
                    wq = 1.0
                    for a in mids:
                        wq *= par_[a] if LOC[q][a] == 1 else 1 - par_[a]
                    acc += wq * Xc[c[q]]
                return acc
            return pt(key, make)
        levels = [[0, 1] if a not in ax else [0, "m", 1] for a in range(3)]
        for i0 in range(len(levels[0]) - 1):
            for j0 in range(len(levels[1]) - 1):
                for k0 in range(len(levels[2]) - 1):
                    child = []
                    for q in range(8):
                        lx = levels[0][i0 + LOC[q][0]]; ly = levels[1][j0 + LOC[q][1]]; lz = levels[2][k0 + LOC[q][2]]
                        child.append(node([lx, ly, lz]))
                    newH.append(child); parent.append(h)
                    # which parent faces each child face lies on
                    tg = []
                    for f in range(6):
                        a, s = FAX[f], FSIDE[f]
                        pos = (i0, j0, k0)[a] + s
                        on_parent = pos == 0 or pos == len(levels[a]) - 1
                        tg.append(ftag[HF[h, f]] if on_parent else "")
                    ptag_rows.append(tg)
    old = np.where(keep)[0]
    multi = [np.array(newH, dtype=np.int64).reshape(-1, 8)]
    allH = np.concatenate([H[old]] + vH + multi)
    par = np.concatenate([old] + vpar + [np.array(parent, dtype=np.int64)])
    ctag = np.concatenate([ftag[HF[old]].astype(str)] + [t.astype(str) for t in vtag]
                          + [np.array(ptag_rows, dtype=str).reshape(-1, 6)])
    Xall = np.concatenate(P)
    print("split %d hexes into %d; mesh %d -> %d hexes (%.0fs)" % (E - len(old), len(allH) - len(old), E, len(allH), time.time() - t0), flush=True)
    from nekmeshpy.tags import Tags
    et = Tags.from_dense(etag[par]) if etag is not None else None
    nm = HexMesh.from_corners(Xall, allH, et)
    NF = np.asarray(nm.hexes)
    bnd = np.asarray(hexmesh.boundary_face_ids(nm))
    for name in sorted(set(ctag[ctag != ""].tolist())):
        hh, ff = np.nonzero(ctag == name)
        ids = np.unique(NF[hh, ff])
        ids = ids[bnd[ids]]
        nm = hexmesh.tag_faces(nm, ids, name)
    print(hexmesh.topology_report(nm), flush=True)
    pickle.dump(nm, open(sys.argv[2], "wb"))
    np.savez(os.path.splitext(sys.argv[2])[0] + "_split.npz", wall=wu, far=wv, new=np.arange(len(X), len(X) + len(newp)), w=w)
    print("done (%.0fs)" % (time.time() - t0))


if __name__ == "__main__":
    main()
