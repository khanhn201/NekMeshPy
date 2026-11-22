import gmsh, numpy as np, collections, pickle
gmsh.initialize(); gmsh.option.setNumber("General.Terminal",0)
import os
STEP = os.environ.get("MSRE_STEP", "core_fluid_cleanup_v5.STEP")
gmsh.model.occ.importShapes(STEP); gmsh.model.occ.synchronize()
S=[s for _,s in gmsh.model.getEntities(2)]
rec=[]
for s in S:
    t=gmsh.model.getType(2,s)
    bb=gmsh.model.getBoundingBox(2,s)
    a=gmsh.model.occ.getMass(2,s)
    c=gmsh.model.occ.getCenterOfMass(2,s)
    rec.append((s,t,a,c,bb))
pickle.dump(rec,open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "faces.pkl"),"wb"))
C=[]
for _,c in gmsh.model.getEntities(1):
    C.append((c,gmsh.model.getType(1,c),gmsh.model.occ.getMass(1,c),gmsh.model.getBoundingBox(1,c)))
pickle.dump(C,open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "curves.pkl"),"wb"))
print(collections.Counter(r[1] for r in rec))
print(collections.Counter(r[1] for r in C))
print("vol", gmsh.model.occ.getMass(3,1))
