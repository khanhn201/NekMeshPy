"""orientation repair and smoothing of one pipe's inside: wall nodes fixed, plate nodes in the
plate plane, everything else free"""
import os
import sys, numpy as np
from collections import defaultdict
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from columns import scaled_jacobian
from geom import PLATE
from inflate import FACES, fkey


def _cyc(f):
    """canonical directed cycle of a face (rotation starting at its min node)"""
    f = [int(v) for v in f]
    i = f.index(min(f))
    return tuple(f[i:] + f[:i])


def orient_by_neighbours(H, good):
    """flip hexes not in `good` so that every shared face is listed in opposite directions"""
    H = np.asarray(H).copy()
    good = np.asarray(good, bool).copy()
    ft = defaultdict(list)
    for h, row in enumerate(H):
        for j, f in enumerate(FACES):
            ft[fkey(row[f])].append((h, j))
    todo = [h for h in range(len(H)) if good[h]]
    while todo:
        h = todo.pop()
        for j, f in enumerate(FACES):
            k = fkey(H[h][f])
            for h2, j2 in ft[k]:
                if h2 == h or good[h2]:
                    continue
                c1 = _cyc(H[h][f]); c2 = _cyc(H[h2][FACES[j2]])
                if c1 == c2:
                    H[h2] = H[h2][[4, 5, 6, 7, 0, 1, 2, 3]]
                    # face slots move with the flip; rebuild this hex's entries
                    for jj, ff in enumerate(FACES):
                        kk = fkey(H[h2][ff])
                        ft[kk] = [(a, b) if a != h2 else (a, jj) for a, b in ft[kk]]
                good[h2] = True
                todo.append(h2)
    assert good.all(), "hexes not reached by orientation propagation"
    return H


EP = np.array([(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)])


def lattice_sj(P, n=3):
    r = np.linspace(0, 1, n); worst = np.full(len(P), np.inf)
    for u in r:
        for v in r:
            for w in r:
                du = ((1-v)*(1-w)*(P[:,1]-P[:,0]) + v*(1-w)*(P[:,2]-P[:,3]) + (1-v)*w*(P[:,5]-P[:,4]) + v*w*(P[:,6]-P[:,7]))
                dv = ((1-u)*(1-w)*(P[:,3]-P[:,0]) + u*(1-w)*(P[:,2]-P[:,1]) + (1-u)*w*(P[:,7]-P[:,4]) + u*w*(P[:,6]-P[:,5]))
                dw = ((1-u)*(1-v)*(P[:,4]-P[:,0]) + u*(1-v)*(P[:,5]-P[:,1]) + (1-u)*v*(P[:,7]-P[:,3]) + u*v*(P[:,6]-P[:,2]))
                j = np.einsum("ij,ij->i", np.cross(du, dv), dw)
                worst = np.minimum(worst, j / (np.linalg.norm(du, axis=1) * np.linalg.norm(dv, axis=1) * np.linalg.norm(dw, axis=1) + 1e-300))
    return worst


FP = np.array([[0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7], [0, 3, 2, 1], [4, 5, 6, 7]])


def twisted(P):
    """per hex: min cosine between a face's corner normals and the face's mean normal (<= 0: twisted)"""
    q = P[:, FP]
    turn = np.cross(np.roll(q, -1, axis=2) - q, np.roll(q, 1, axis=2) - q)
    mean = turn.sum(axis=2, keepdims=True)
    agree = np.einsum("efkc,efkc->efk", turn, np.broadcast_to(mean, turn.shape))
    m = np.linalg.norm(turn, axis=3) * np.linalg.norm(mean, axis=3)
    return (agree / np.maximum(m, 1e-300)).min(axis=(1, 2))


def merit(P):
    return np.minimum(np.minimum(scaled_jacobian(P), lattice_sj(P)), twisted(P))


def merit_e(P, lmin):
    q = merit(P)
    if lmin <= 0:
        return q
    e = np.linalg.norm(P[:, EP[:, 0]] - P[:, EP[:, 1]], axis=2).min(1)
    return np.where(e < lmin, np.minimum(q, e / lmin - 1.0), q)


def untangle(X, H, fixed, plate, thr=0.2, sweeps=60, verbose=True, lfrac=0.3, slide=None):
    """Gauss-Seidel pattern search: each free node moves to raise the worst merit of its hexes
    (merit goes negative when a hex edge shrinks below lfrac x the median edge)"""
    X = X.copy()
    dd = np.linalg.norm(X[H[:, EP[:, 0]]] - X[H[:, EP[:, 1]]], axis=2)
    lmin = lfrac * np.median(dd)
    merit = lambda P: merit_e(P, lmin)
    inc = defaultdict(list)
    for h, row in enumerate(H):
        for v in row:
            inc[int(v)].append(h)
    dirs = np.array([[a, b, c] for a in (-1, 0, 1) for b in (-1, 0, 1) for c in (-1, 0, 1) if (a, b, c) != (0, 0, 0)], float)
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    sj = merit(X[H])
    if verbose:
        print("   untangle before: inverted %d, <%.2f: %d, min %.3f" % ((sj < 0).sum(), thr, (sj < thr).sum(), sj.min()))
    for sw in range(sweeps):
        bad = np.where(sj < thr)[0]
        if not len(bad):
            break
        cand = np.zeros(len(X), bool); cand[H[bad].ravel()] = True
        hs_ = cand[H].any(1); cand[H[hs_].ravel()] = True
        moved = 0
        for v in np.where(cand)[0]:
            if v in fixed and (slide is None or v not in slide):
                continue
            hs = np.array(inc[v]); f0 = sj[hs].min()
            if f0 >= thr:
                continue
            E = H[hs]
            nb = np.unique(E.ravel()); nb = nb[nb != v]
            ell = np.min(np.linalg.norm(X[nb] - X[v], axis=1))
            ell = max(ell, 1e-3)
            best, bestP = f0, None
            D = dirs.copy()
            if v in plate:
                D[:, 1] = 0.0
                n = np.linalg.norm(D, axis=1); D = D[n > 0] / n[n > 0, None]
            for step in (1.0, 0.5, 0.25, 0.1, 0.04):
                P = X[v] + step * ell * D
                if slide is not None and v in slide:
                    P = slide[v](P)
                Pa = X[E][None].repeat(len(P), 0)
                mask = (E == v)
                Pa[:, mask] = P[:, None, :].repeat(mask.sum(), 1)
                f = merit(Pa.reshape(-1, 8, 3)).reshape(len(P), len(hs)).min(1)
                k = int(np.argmax(f))
                if f[k] > best + 1e-7:
                    best, bestP = f[k], P[k]
            if bestP is not None:
                X[v] = bestP; sj[hs] = merit(X[H[hs]]); moved += 1
        if verbose and (sw % 10 == 9 or moved == 0):
            print("   sweep %d: moved %d, inverted %d, <%.2f: %d, min %.3f" % (sw, moved, (sj < 0).sum(), thr, (sj < thr).sum(), sj.min()))
        if moved == 0:
            break
    return X, sj


def laplace(X, H, fixed, plate, iters=30, w=0.5):
    """sizing pass: move each free node toward the mean of its edge neighbours if no incident hex
    gets worse than 0.2 (or worse than it already was)"""
    X = X.copy()
    nb = defaultdict(set); inc = defaultdict(list)
    for h, row in enumerate(H):
        for a, b in EP:
            nb[int(row[a])].add(int(row[b])); nb[int(row[b])].add(int(row[a]))
        for v in row:
            inc[int(v)].append(h)
    free = [v for v in nb if v not in fixed]
    for _ in range(iters):
        moved = 0
        for v in free:
            hs = np.array(inc[v]); f0 = merit(X[H[hs]]).min()
            tgt = X[list(nb[v])].mean(0)
            if v in plate:
                tgt[1] = PLATE
            old = X[v].copy()
            X[v] = old + w * (tgt - old)
            if merit(X[H[hs]]).min() < min(f0, 0.2):
                X[v] = old
            else:
                moved += 1
        if not moved:
            break
    return X


def plain_laplace(X, H, fixed, plate, iters=50, w=0.6):
    """unconditional Laplacian (vectorised) to spread the nodes of a freshly inflated mesh"""
    X = X.copy()
    e = np.concatenate([H[:, EP[:, 0]].ravel()[:, None], H[:, EP[:, 1]].ravel()[:, None]], 1)
    e = np.unique(np.sort(e, 1), axis=0)
    n = len(X)
    deg = np.bincount(e.ravel(), minlength=n).astype(float)
    free = np.zeros(n, bool); free[np.unique(H)] = True
    for v in fixed:
        free[v] = False
    pl = np.zeros(n, bool)
    for v in plate:
        pl[v] = True
    for _ in range(iters):
        acc = np.zeros_like(X)
        np.add.at(acc, e[:, 0], X[e[:, 1]]); np.add.at(acc, e[:, 1], X[e[:, 0]])
        tgt = acc / np.maximum(deg, 1)[:, None]
        Xn = X + w * (tgt - X)
        Xn[pl, 1] = X[pl, 1]
        X[free] = Xn[free]
    return X


def untangle_fast(X, H, fixed, plate, thr=0.2, sweeps=60, verbose=True, chunk=200):
    """vectorised Gauss-Seidel pattern search over colour classes (no two nodes of a class
    share a hex)"""
    X = X.copy()
    n = len(X)
    inc = defaultdict(list)
    for h, row in enumerate(H):
        for c, v in enumerate(row):
            inc[int(v)].append(h)
    nodes = np.array(sorted(v for v in inc if v not in fixed))
    K = max(len(inc[v]) for v in nodes)
    IH = np.full((len(nodes), K), -1)
    for i, v in enumerate(nodes):
        IH[i, :len(inc[v])] = inc[v]
    pos = {int(v): i for i, v in enumerate(nodes)}
    isplate = np.array([v in plate for v in nodes])
    # greedy colouring
    colour = -np.ones(len(nodes), int)
    for i, v in enumerate(nodes):
        used = set()
        for h in inc[v]:
            for u in H[h]:
                j = pos.get(int(u))
                if j is not None and colour[j] >= 0:
                    used.add(colour[j])
        c = 0
        while c in used:
            c += 1
        colour[i] = c
    dirs = np.array([[a, b, c] for a in (-1, 0, 1) for b in (-1, 0, 1) for c in (-1, 0, 1) if (a, b, c) != (0, 0, 0)], float)
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    steps = np.array([1.0, 0.5, 0.25, 0.1, 0.04])
    cand = (steps[:, None, None] * dirs[None]).reshape(-1, 3)          # (C,3) unit-length scaled
    sj = merit(X[H])
    if verbose:
        print("   untangle before: inverted %d, <%.2f: %d, min %.3f" % ((sj < 0).sum(), thr, (sj < thr).sum(), sj.min()), flush=True)
    for sw in range(sweeps):
        moved = 0
        for c in range(colour.max() + 1):
            sel = np.where(colour == c)[0]
            hs = IH[sel]
            valid = hs >= 0
            cur = np.where(valid, sj[np.maximum(hs, 0)], np.inf).min(1)
            sel = sel[cur < thr]
            if not len(sel):
                continue
            for a in range(0, len(sel), chunk):
                s_ = sel[a:a + chunk]
                hs = IH[s_]; valid = hs >= 0; hs0 = np.maximum(hs, 0)
                E = H[hs0]                                             # (B,K,8)
                P0 = X[E]                                              # (B,K,8,3)
                v = nodes[s_]
                ell = np.linalg.norm(P0 - X[v][:, None, None, :], axis=3)
                ell = np.where((E != v[:, None, None]) & valid[..., None], ell, np.inf).min((1, 2))
                ell = np.maximum(ell, 1e-3)
                D = cand[None] * ell[:, None, None]                      # (B,C,3)
                D[isplate[s_], :, 1] = 0.0
                mask = (E == v[:, None, None])                          # (B,K,8)
                f0 = np.where(valid, sj[hs0], np.inf).min(1)
                best = f0.copy(); bestd = np.zeros((len(s_), 3))
                for ci in range(D.shape[1]):
                    P = P0 + mask[..., None] * D[:, ci][:, None, None, :]
                    m = merit(P.reshape(-1, 8, 3)).reshape(len(s_), -1)
                    m = np.where(valid, m, np.inf).min(1)
                    better = m > best + 1e-7
                    best = np.where(better, m, best)
                    bestd[better] = D[better, ci]
                mv = best > f0 + 1e-7
                X[v[mv]] += bestd[mv]
                moved += mv.sum()
                ch = np.unique(IH[s_[mv]]); ch = ch[ch >= 0]
                sj[ch] = merit(X[H[ch]])
        if verbose and (sw % 10 == 9 or moved == 0):
            print("   sweep %d: moved %d, inverted %d, <%.2f: %d, min %.3f" % (sw, moved, (sj < 0).sum(), thr, (sj < thr).sum(), sj.min()), flush=True)
        if moved == 0:
            break
    return X, sj


CN = [(0, 1, 3, 4), (1, 2, 0, 5), (2, 3, 1, 6), (3, 0, 2, 7), (4, 7, 5, 0), (5, 4, 6, 1), (6, 5, 7, 2), (7, 6, 4, 3)]


def global_untangle(X, H, fixed, plate, tau=0.25, lam=0.02, iters=400, verbose=True, mu=1.0, lfrac=0.3, own_size=False):
    """L-BFGS on  sum_corners max(0, tau - det/L^3)^2  +  lam * sum_edges |dx|^2 / L^2,
    wall nodes fixed, plate nodes keep their y"""
    from scipy.optimize import minimize
    X = X.copy()
    used = np.unique(H)
    loc = -np.ones(len(X), int); loc[used] = np.arange(len(used))
    h = loc[H]
    P0 = X[used].copy()
    fx = np.array([int(v) in fixed for v in used])
    pl = np.array([int(v) in plate for v in used])
    free = ~fx
    mask = np.repeat(free[:, None], 3, 1); mask[pl, 1] = False
    idx = np.where(mask.ravel())[0]
    L = np.linalg.norm(P0[h[:, EP[:, 0]]] - P0[h[:, EP[:, 1]]], axis=2).mean(1)        # per hex
    if not own_size:
        L = np.maximum(L, np.median(L))
    L3 = L ** 3
    e = np.unique(np.sort(np.concatenate([h[:, EP[:, 0]].reshape(-1, 1), h[:, EP[:, 1]].reshape(-1, 1)], 1), 1), axis=0)
    Le = np.linalg.norm(P0[e[:, 0]] - P0[e[:, 1]], axis=1).mean()
    l0 = lfrac * np.median(np.linalg.norm(P0[e[:, 0]] - P0[e[:, 1]], axis=1))
    def fg(z):
        P = P0.copy(); P.reshape(-1)[idx] = z
        G = np.zeros_like(P); f = 0.0
        for c, a, b, d in CN:
            e1 = P[h[:, a]] - P[h[:, c]]; e2 = P[h[:, b]] - P[h[:, c]]; e3 = P[h[:, d]] - P[h[:, c]]
            det = np.einsum("ij,ij->i", np.cross(e1, e2), e3) / L3
            r = np.maximum(0.0, tau - det)
            f += (r ** 2).sum()
            w = (-2 * r / L3)[:, None]
            g1 = w * np.cross(e2, e3); g2 = w * np.cross(e3, e1); g3 = w * np.cross(e1, e2)
            np.add.at(G, h[:, a], g1); np.add.at(G, h[:, b], g2); np.add.at(G, h[:, d], g3)
            np.add.at(G, h[:, c], -(g1 + g2 + g3))
        dv = P[e[:, 0]] - P[e[:, 1]]
        f += lam * (dv ** 2).sum() / Le ** 2
        ge = 2 * lam * dv / Le ** 2
        nl = np.linalg.norm(dv, axis=1)
        rr = np.maximum(0.0, l0 - nl)
        f += mu * (rr ** 2).sum() / l0 ** 2
        ge += (-2 * mu * rr / l0 ** 2 / np.maximum(nl, 1e-12))[:, None] * dv
        np.add.at(G, e[:, 0], ge); np.add.at(G, e[:, 1], -ge)
        return f, G.reshape(-1)[idx]
    z0 = P0.reshape(-1)[idx]
    res = minimize(fg, z0, jac=True, method="L-BFGS-B", options=dict(maxiter=iters, maxcor=20))
    P = P0.copy(); P.reshape(-1)[idx] = res.x
    X[used] = P
    if verbose:
        sj = merit(X[H])
        print("   global untangle: %d its, f %.3g, inverted %d, min %.3f" % (res.nit, res.fun, (sj < 0).sum(), sj.min()), flush=True)
    return X


def edge_stats(X, H):
    d = np.linalg.norm(X[H[:, EP[:, 0]]] - X[H[:, EP[:, 1]]], axis=2)
    return d.min(), (d.max(1) / np.maximum(d.min(1), 1e-30)).max()
