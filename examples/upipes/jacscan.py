"""min trilinear Jacobian determinant (scaled) over an n^3 lattice in every hex"""
import numpy as np, sys
D = np.load(sys.argv[1]); X, H = D["X"], D["H"]
n = int(sys.argv[2]) if len(sys.argv) > 2 else 5
P = X[H]                                    # (E, 8, 3), VTK order
r = np.linspace(0, 1, n)
worst = np.full(len(H), np.inf)
for u in r:
    for v in r:
        for w in r:
            # derivatives of trilinear map (VTK: 0(000) 1(100) 2(110) 3(010) 4(001) 5(101) 6(111) 7(011))
            du = ((1-v)*(1-w)*(P[:,1]-P[:,0]) + v*(1-w)*(P[:,2]-P[:,3]) + (1-v)*w*(P[:,5]-P[:,4]) + v*w*(P[:,6]-P[:,7]))
            dv = ((1-u)*(1-w)*(P[:,3]-P[:,0]) + u*(1-w)*(P[:,2]-P[:,1]) + (1-u)*w*(P[:,7]-P[:,4]) + u*w*(P[:,6]-P[:,5]))
            dw = ((1-u)*(1-v)*(P[:,4]-P[:,0]) + u*(1-v)*(P[:,5]-P[:,1]) + (1-u)*v*(P[:,7]-P[:,3]) + u*v*(P[:,6]-P[:,2]))
            j = np.einsum("ij,ij->i", np.cross(du, dv), dw)
            s = j / (np.linalg.norm(du, axis=1) * np.linalg.norm(dv, axis=1) * np.linalg.norm(dw, axis=1))
            worst = np.minimum(worst, s)
print("lattice %d^3: min scaled J %.4f, negative %d, <0.1 %d" % (n, worst.min(), (worst < 0).sum(), (worst < 0.1).sum()))
np.save(sys.argv[1].replace(".npz", "_jmin.npy"), worst)
