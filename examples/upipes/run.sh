#!/bin/bash
# U-pipe bundle (pipes.txt, 35 U-tubes in a box above the plate): EVG hex mesh of the fluid outside the
# pipes -> res_al1u.{npz,vtu}, res_al1u_hexmesh.pkl (16,702 hexes).  Writes everything in this folder.
# Set SKIP_MAPS=1 to reuse maps/ from an earlier run.
set -eo pipefail
cd "$(dirname "$0")"
tag=${1:-al1u}

export EVG_ALIGNF=73-81,80-82,81-90,82-95 EVG_VACSPLIT=4 EVG_T1=0 EVG_TRANS=ringsafe EVG_NORING=-1
export EVG_H=7 EVG_HCORE=2.5 EVG_NLMIN=1 EVG_NLMAX=2 EVG_NLDIV=6 EVG_NLWALL=1 EVG_CORE=1,2,2
export EVG_HUB=2,4,2 EVG_HUBNN=3

# 1. Voronoi diagram of the pipes + walls: ray label maps, edges, graph
if [ -z "$SKIP_MAPS" ]; then
    mkdir -p maps
    python -u rays.py > rays.log
    python -u edges.py > edges.log
    python -u graph.py > graph.log
fi

# 2. graph fixes, decomposition into patches + ILP interval counts + columns -> out_full
python graphfix.py > graphfix.log
python -u build3.py > build_$tag.log 2>&1
grep -E "hexes;|mesh fails" build_$tag.log

# 3. surface unfold, surface shape optimisation of every boundary surface
python -u surfsmooth.py out_full 60 | tail -1
python -u gsmooth2.py out_full_ss 300 0 2>&1 | grep -E "^before"

# 4. untangle with every boundary node fixed
python -u outer_untangle.py out_full_ss_g.npz out_full_ou.npz | grep -E "start|final"
cp out_full_ss_tags.pkl out_full_ou_tags.pkl

# 5. volume shape optimisation, final untangle, checks, export
python -u gsmooth2.py out_full_ou 0 300 2>&1 | grep -E "^before|^after|CG"
python -u untangle2.py out_full_ou_g 0.2 40 | grep -E "before|sweep 39|moved 0"
f=out_full_ou_g_u
python jacscan.py $f.npz 9
python export.py $f 2>&1 | grep -E "validity|Quality|untagged" | cut -c1-200
python bquality.py $f.npz | head -1
python -c "
import pickle, sys
sys.path.insert(0, '../..')
from nekmeshpy.hexmesh import query
m = pickle.load(open('${f}_hexmesh.pkl', 'rb')); sm = query.solver_metrics(m)
print('solver: spacing_min %.4f jac_ratio_min %.4f aspect_max %.1f' % (sm.spacing_min, sm.jac_ratio_min, sm.aspect_max))
"
cp $f.npz res_$tag.npz; cp ${f}_tagged.vtu res_$tag.vtu; cp ${f}_hexmesh.pkl res_${tag}_hexmesh.pkl
echo DONE res_$tag
