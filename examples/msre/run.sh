#!/bin/bash
# Rebuild the MSRE fuel-salt hex mesh (linear, core NZ=32) from the shipped CAD-derived inputs
# (p1/lattice.pkl, p4/rim_exclude.npy, p4/bore_axes.npy, p4/pipe_axis.npy).  Every output lands next to its
# script; the finished mesh is finish/msre.{re2,vtu,pkl}.
#   ./run.sh             full chain
#   FROM=bands ./run.sh  start at a stage: section core top topology bottom bands head downcomer L2 untangle
#                        l2overlap pocket assemble pinch smooth polish overlap revert export
#   PAR=0 ./run.sh       everything in sequence (default PAR=1: independent work runs concurrently)
# Steps run in a memory-capped systemd scope when available (MEM, default 64G per step); NOCAP=1 skips it.
set -eo pipefail
export ORDER=${ORDER:-2} CORE_NZ=${CORE_NZ:-32} CORE_O=${ORDER:-2}
ROOT=$(cd "$(dirname "$0")" && pwd)
export PYTHONPATH=$(cd "$ROOT/../.." && pwd)${PYTHONPATH:+:$PYTHONPATH}
MEM=${MEM:-64G}
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
stage topology  p4 python topo2.py                                       # lattice + lofted rim strip -> topo2.pkl
[ "$FROM" = bottom ] && go=1
if [ $go = 1 ]; then
    stage bottom p4 python positions.py top                              # top level (AB copies it)
    fan   levels p4 positions.py $LEVELS                                 # other 13 levels -> pos_*.npz
fi
fan   bands     p4 bands.py $BANDS                                       # bands A..KN (rim) -> band_*.pkl
stage head      p4 python head.py                                        # lower-head section -> head_sec.pkl
bg    downcomer p4 bash -c "python dc3.py && python volute.py && python inlet2.py"   # downcomer, volute, inlet
stage L2        p4 python vloft.py                                       # lower head, vanes lofted -> head_L2_vloft.pkl
stage untangle  p4 bash -c "IN=head_L2_vloft.pkl OUT=head_L2_tw TGT=0.02 GROW=2 ROUNDS=40 python vskin_twist.py"
stage l2overlap p4 bash -c "IN=head_L2_tw.pkl OUT=head_L2_lin HMIN=0 python vskin_ovl_fix.py"   # -> head_L2_lin.pkl
for p in "${BG[@]}"; do wait $p || { echo "!! a background stage failed (logs/core.log, top.log, downcomer.log)"; exit 1; }; done
stage pocket    p5 python pocket.py                                      # side region under the dome rim -> pocket.pkl
stage assemble  p4 bash -c "L2PKL=head_L2_lin.pkl LINEAR=1 OUT=msre_raw python assemble.py"   # -> p4/msre_raw.pkl
# finishing, on the assembled linear model (finish/)
stage pinch     finish python pinch_fix.py ../p4/msre_raw.pkl msre_wt.pkl      # cells meeting only along an edge
stage smooth    finish bash -c "SJ=0.15 RINGS=2 python region_fix.py msre_wt.pkl msre_sm.pkl"
stage polish    finish bash -c "SJ=0.15 RINGS=2 ROUNDS=10 python greedy_fix.py msre_sm.pkl msre_p1.pkl && SJ=0.15 RINGS=2 ROUNDS=10 ROT=45 python greedy_fix.py msre_p1.pkl msre_p2.pkl"
stage overlap   p4 bash -c "IN=../finish/msre_p2.pkl OUT=../finish/msre_ov HMIN=0 python vskin_ovl_fix.py"
stage revert    finish python revert_sampled.py msre_sm.pkl msre_ov.pkl msre_fin.pkl  # clean at the solver's order
stage export    finish bash -c "python export_final.py msre_fin.pkl msre --vtu && python ../p4/re2check.py msre.re2 && python bc_check.py msre.re2 && python qreg.py msre.re2"
echo "== done ($(date +%H:%M:%S))"
