import numpy as np
def corner_sj(X, Q):
    P = X[Q]
    e1 = np.roll(P, -1, 1) - P; e0 = np.roll(P, 1, 1) - P
    cr = e1[..., 0] * e0[..., 1] - e1[..., 1] * e0[..., 0]
    return (cr / (np.linalg.norm(e1, axis=-1) * np.linalg.norm(e0, axis=-1) + 1e-300)).min(1)


def quad_csj(P):
    """P (..., 4, 2) -> min corner sj per quad."""
    e1 = np.roll(P, -1, -2) - P; e0 = np.roll(P, 1, -2) - P
    cr = e1[..., 0] * e0[..., 1] - e1[..., 1] * e0[..., 0]
    return (cr / (np.linalg.norm(e1, axis=-1) * np.linalg.norm(e0, axis=-1) + 1e-300)).min(-1)


def untangle(X, Q, free, project=None, target=0.15, rounds=60, grow=2, seed=0):
    """greedy local untangler: each free node near a bad quad moves to the candidate (around its
    neighbour centroid / current spot) that maximizes the worst corner sj of its quads.
    project(i, p) -> p maps a candidate onto node i's constraint (circle / line); free: bool mask."""
    import collections
    X = X.copy()
    inc = collections.defaultdict(list)
    nbr = collections.defaultdict(set)
    for qi, q in enumerate(Q):
        for a in range(4):
            inc[q[a]].append(qi)
            nbr[q[a]].update((q[(a + 1) % 4], q[(a - 1) % 4]))
    rng = np.random.default_rng(seed)
    for it in range(rounds):
        s = corner_sj(X, Q)
        bad = np.where(s < target)[0]
        if not len(bad):
            break
        zone = set(np.unique(Q[bad]).tolist())
        for _ in range(grow):
            zone |= {w for v in list(zone) for w in nbr[v]}
        moved = 0
        for v in sorted(zone, key=lambda v: min(s[inc[v]])):
            if not free[v]:
                continue
            qs = inc[v]
            cur = quad_csj(X[Q[qs]]).min()
            c0 = X[list(nbr[v])].mean(0)
            h = 0.5 * np.mean([np.linalg.norm(X[w] - X[v]) for w in nbr[v]])
            cands = [c0, 0.5 * (c0 + X[v])] + [X[v] + h * rng.normal(size=2) * f for f in (0.2, 0.5, 1.0) for _ in range(6)]
            best, bp = cur, None
            for p in cands:
                if project is not None:
                    p = project(v, p)
                old = X[v].copy(); X[v] = p
                val = quad_csj(X[Q[qs]]).min()
                X[v] = old
                if val > best + 1e-9:
                    best, bp = val, p
            if bp is not None:
                X[v] = bp; moved += 1
        if not moved:
            break
    return X
