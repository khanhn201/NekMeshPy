#!/bin/bash
# Rebuild the MSRE fuel-salt hex mesh (order 2, core NZ=32) from the shipped CAD-derived inputs
# (p1/lattice.pkl, p4/topo.pkl, p4/bore_axes.npy, p4/pipe_axis.npy).  Every output lands next to its script.
#   ./run.sh             full chain
#   FROM=bands ./run.sh  start at a stage: section core top bottom bands head vanes L2 downcomer volute inlet pocket assemble
#   PAR=0 ./run.sh       everything in sequence (default PAR=1: independent work runs concurrently)
# Steps run in a memory-capped systemd scope when available (MEM, default 22G per step); NOCAP=1 skips it.
set -eo pipefail
export ORDER=${ORDER:-2} CORE_NZ=${CORE_NZ:-32} CORE_O=${ORDER:-2}
ROOT=$(cd "$(dirname "$0")" && pwd)
export PYTHONPATH=$(cd "$ROOT/../.." && pwd)${PYTHONPATH:+:$PYTHONPATH}
MEM=${MEM:-22G}
PAR=${PAR:-1}
LEVELS="AB BC CD DE EF F3F2 F2F1 F1F0 F0H HI IJ JK KN"
BANDS="A B C D E F3 F2 F1 F0 H I J KN"
mkdir -p "$ROOT/logs"
go=0; [ -z "$FROM" ] && go=1
BG=()

cap() {   # cap <cmd...> : run in a memory-capped scope when possible
    if [ -z "$NOCAP" ] && command -v systemd-run >/dev/null && systemd-run --user --scope -q true 2>/dev/null; then
        systemd-run --user --scope -q -p MemoryMax=$MEM "$@"
    else
        "$@"
    fi
}
stage() {   # stage <name> <dir> <cmd...> : run one stage, log to logs/<name>.log
    local st=$1 d=$2; shift 2
    [ "$st" = "$FROM" ] && go=1
    [ $go = 1 ] || return 0
    echo "== $st ($(date +%H:%M:%S))"
    (cd "$ROOT/$d" && cap "$@" > "$ROOT/logs/$st.log" 2>&1) || { echo "!! $st failed, see logs/$st.log"; tail -5 "$ROOT/logs/$st.log"; exit 1; }
    grep -v "Warning\|transform_fn\|^ *avg = \|^ *g = ell" "$ROOT/logs/$st.log" | tail -2
}
bg() {      # bg <name> <dir> <cmd...> : like stage, concurrently when PAR=1
    if [ "$PAR" = 1 ]; then
        local st=$1
        [ "$st" = "$FROM" ] && go=1
        [ $go = 1 ] || return 0
        stage "$@" & BG+=($!)
    else
        stage "$@"
    fi
}
fan() {     # fan <name> <dir> <script> <args...> : one process per argument, all at once when PAR=1
    local st=$1 d=$2 s=$3; shift 3
    [ "$st" = "$FROM" ] && go=1
    [ $go = 1 ] || return 0
    echo "== $st x$# ($(date +%H:%M:%S))"
    local pids=()
    for a in "$@"; do
        (cd "$ROOT/$d" && NPROC=${NPROC:-4} cap python $s $a > "$ROOT/logs/$st.$a.log" 2>&1) &
        pids+=($!)
        [ "$PAR" = 1 ] || wait $!
    done
    for p in "${pids[@]}"; do wait $p || { echo "!! $st failed, see logs/$st.*.log"; exit 1; }; done
}

stage section p1 bash -c "python build_section.py && cp section.pkl section_presmooth.pkl && python smooth_bores.py"
bg    core    p1 python upper.py $ORDER $CORE_NZ                         # core -> upper_o2_nz32.pkl
bg    top     p5 bash -c "python head_top.py && python skin.py"          # top head -> head_top.pkl
[ "$FROM" = bottom ] && go=1
if [ $go = 1 ]; then
    stage bottom p4 python positions.py top                              # top level (AB copies it)
    fan   levels p4 positions.py $LEVELS                                 # other 13 levels -> pos_*.npz
fi
fan   bands     p4 bands.py $BANDS                                       # bands A..KN (rim) -> band_*.pkl
stage head      p4 python head.py                                        # lower-head section -> head_sec.pkl
stage vanes     p4 bash -c "python vanes2.py && python vanepos2.py"      # vanes -> vanepos.pkl
stage L2        p4 python head3d.py                                      # lower head -> head_L2.pkl
stage downcomer p4 python dc3.py                                         # downcomer + tilted bores -> dc3.pkl
stage volute    p4 python volute.py                                      # volute -> volute_inner.pkl
stage inlet     p4 python inlet2.py                                      # O-grid window + inlet pipe -> volute.pkl
for p in "${BG[@]}"; do wait $p || { echo "!! a background stage failed (logs/core.log, logs/top.log)"; exit 1; }; done
stage pocket    p5 python pocket.py                                      # side region under the dome rim -> pocket.pkl
stage assemble  p4 bash -c "OUT=msre python assemble.py --vtu"           # -> p4/msre.{pkl,re2,vtu}
echo "== done ($(date +%H:%M:%S))"
