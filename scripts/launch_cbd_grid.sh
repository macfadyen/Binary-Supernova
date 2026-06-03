#!/usr/bin/env bash
# scripts/launch_cbd_grid.sh — one-command driver for the CBD negative-result
# confirming grid (see docs/cbd_grid_runbook.md).  Wraps the qsub probe -> launch
# sequence over scripts/run_relax_optimal.pbs, with the G3 checkpoint-continuation
# check baked in so you don't paste five qsub lines by hand.
#
# Run on cabeus (PBS).  The grid axes (the rest of the config is fixed in
# run_relax_optimal.pbs: masses 15/30/20, relax-IC, L=6, bomb-mass-frac 0.5,
# kick -0.015):
#
#   G1  NX=256 a0=2.5 E=0.02 t=20   weaker SN / more bound mass
#   G2  NX=256 a0=2.0 E=0.05 t=20   deeper Roche overflow
#   G3  NX=256 a0=2.5 E=0.05 t=50   long integration (continues G0's checkpoint)
#   G4  NX=256 a0=2.0 E=0.02 t=30   combined worst case
#   G5  NX=384 a0=2.5 E=0.05 t=20   resolution check (auto-chains the 24 h cap)
#
# Usage:
#   scripts/launch_cbd_grid.sh probe              # devel-queue wall probes (G2/G5 shapes)
#   scripts/launch_cbd_grid.sh launch             # submit G1 G2 G3 G4 G5
#   scripts/launch_cbd_grid.sh launch G2 G5       # submit a subset
#   scripts/launch_cbd_grid.sh --dry-run launch   # print the qsub lines, submit nothing
#   scripts/launch_cbd_grid.sh status             # qstat for this user + grid OUTDIR state
#
# The decision gate after the runs complete is in docs/cbd_grid_runbook.md §4.

set -euo pipefail
cd "$(dirname "$0")/.."

PBS=scripts/run_relax_optimal.pbs
DRY=0

usage() { sed -n '2,33p' "$0" | sed 's/^# \{0,1\}//'; }

# OUTDIR matching the run_relax_optimal.pbs TAG.  KEEP IN SYNC with the fixed
# config in that file (masses 15/20, L=6, bombmf 0.5, kick m0.015); only NX, a0
# and E_SN vary here.  ${v/./} strips the single decimal point (2.5->25, 0.05->005).
outdir_for() {  # args: NX A0_RSUN E_SN_FRAC
    printf 'demo1/output_relax_optimal_a%srsun_L6_bh15_20_esn%s_bombmf05_kickm0.015_nx%s' \
           "${2/./}" "${3/./}" "$1"
}

# Submit (or, in --dry-run, just print) one qsub invocation.
submit() {
    if [ "$DRY" -eq 1 ]; then
        echo "  [dry-run] qsub $*"
    else
        qsub "$@"
    fi
}

# G3 is the long-integration run.  Cheapest as a continuation of G0's checkpoint
# (reuse G0's OUTDIR; the PBS script auto-detects chkpt.h5 and resumes the IC-built
# state).  If G0's checkpoint is gone, fall back to a fresh t=50 run in its own dir.
launch_g3() {
    local g0 g3fresh
    g0="$(outdir_for 256 2.5 0.05)"
    g3fresh="demo1/output_G3_t50_nx256"
    if [ -f "$g0/chkpt.h5" ]; then
        echo "  G3: continuing G0 checkpoint ($g0/chkpt.h5) -> t_end=50"
        submit -v NX=256,A0_RSUN=2.5,E_SN_FRAC=0.05,T_END=50,OUTDIR="$g0" "$PBS"
    else
        echo "  G3: no checkpoint at $g0/chkpt.h5 -> fresh t_end=50 run in $g3fresh"
        submit -v NX=256,A0_RSUN=2.5,E_SN_FRAC=0.05,T_END=50,OUTDIR="$g3fresh" "$PBS"
    fi
}

launch_one() {  # arg: run name G1..G5
    case "$1" in
        G1) submit -v NX=256,A0_RSUN=2.5,E_SN_FRAC=0.02,T_END=20 "$PBS" ;;
        G2) submit -v NX=256,A0_RSUN=2.0,E_SN_FRAC=0.05,T_END=20 "$PBS" ;;
        G3) launch_g3 ;;
        G4) submit -v NX=256,A0_RSUN=2.0,E_SN_FRAC=0.02,T_END=30 "$PBS" ;;
        G5) submit -v NX=384,A0_RSUN=2.5,E_SN_FRAC=0.05,T_END=20 "$PBS" ;;
        *)  echo "  !! unknown run '$1' (expected G1..G5; G0 is already done)" >&2; return 1 ;;
    esac
}

do_probe() {
    echo ">> devel-queue wall probes (2 min of sim each; read Mcell/s from the log)"
    echo "   G2-shape  NX=256 a0=2.0"
    submit -q devel -l walltime=00:45:00 \
        -v T_END=2,DT_SNAP=2,WALL_BUDGET_MIN=40,NX=256,A0_RSUN=2.0,E_SN_FRAC=0.05 "$PBS"
    echo "   G5-shape  NX=384 a0=2.5"
    submit -q devel -l walltime=00:45:00 \
        -v T_END=2,DT_SNAP=2,WALL_BUDGET_MIN=40,NX=384,A0_RSUN=2.5,E_SN_FRAC=0.05 "$PBS"
}

do_launch() {  # args: optional run-name subset; default = all five
    local runs
    if [ "$#" -gt 0 ]; then runs=("$@"); else runs=(G1 G2 G3 G4 G5); fi
    local rc=0 r
    for r in "${runs[@]}"; do
        echo ">> submitting $r"
        launch_one "$r" || { echo "  !! $r submission failed" >&2; rc=1; }
    done
    [ "$DRY" -eq 1 ] && echo ">> dry-run: nothing was submitted."
    return "$rc"
}

do_status() {
    echo "== qstat (this user) =="
    qstat -u "${USER:-$(id -un)}" 2>/dev/null || echo "  (qstat unavailable — not on a PBS host?)"
    echo
    echo "== grid output dirs (diag rows / checkpoint / completion) =="
    echo "   verdict: DONE = done.marker present (finished cleanly, no relaunch);"
    echo "            RESUME = checkpoint but no done.marker (wall-budget truncated);"
    echo "            re-run = dir exists but no checkpoint (stub/empty — start fresh)."
    local spec name d
    for spec in "G0 256 2.5 0.05" "G1 256 2.5 0.02" "G2 256 2.0 0.05" \
                "G4 256 2.0 0.02" "G5 384 2.5 0.05"; do
        # shellcheck disable=SC2086
        set -- $spec
        name="$1"; d="$(outdir_for "$2" "$3" "$4")"
        if [ -d "$d" ]; then
            local rows="-" chk="no" done_="no" verdict
            [ -f "$d/diagnostics.csv" ] && rows="$(wc -l < "$d/diagnostics.csv" | tr -d ' ')"
            [ -f "$d/chkpt.h5" ]    && chk="yes"
            [ -f "$d/done.marker" ] && done_="yes"
            # A finished run has done.marker (run_sn50_fiducial.jl writes it at t_end);
            # chkpt.h5 alone can't tell "finished" from "wall-budget truncated".
            if   [ "$done_" = "yes" ]; then verdict="DONE"
            elif [ "$chk"   = "yes" ]; then verdict="RESUME (mid-flight)"
            else                            verdict="re-run (no chkpt)"
            fi
            printf "  %-3s %-78s diag=%-6s chkpt=%-3s done=%-3s %s\n" \
                   "$name" "$d" "$rows" "$chk" "$done_" "$verdict"
        else
            printf "  %-3s %-78s [not started]\n" "$name" "$d"
        fi
    done
    # G3 lives in G0's dir (continuation) or its own fresh dir; report its
    # completion the same way so the verdict column is consistent.
    local g3="demo1/output_G3_t50_nx256"
    if [ -d "$g3" ]; then
        local g3v="re-run (no chkpt)"
        [ -f "$g3/chkpt.h5" ]    && g3v="RESUME (mid-flight)"
        [ -f "$g3/done.marker" ] && g3v="DONE"
        printf "  %-3s %-78s (fresh) %s\n" "G3" "$g3" "$g3v"
    fi
}

# ---- arg parse -------------------------------------------------------------
SUBCMD=""
RUNS=()
while [ "$#" -gt 0 ]; do
    case "$1" in
        --dry-run|-n)        DRY=1; shift ;;
        -h|--help)           usage; exit 0 ;;
        probe|launch|status) SUBCMD="$1"; shift ;;
        [Gg][0-9])           RUNS+=( "$(printf '%s' "$1" | tr 'a-z' 'A-Z')" ); shift ;;
        *) echo "unknown arg: $1" >&2; usage; exit 1 ;;
    esac
done

[ -n "$SUBCMD" ] || { usage; exit 1; }
[ -f "$PBS" ]   || { echo "error: $PBS not found (run from the repo)" >&2; exit 1; }

case "$SUBCMD" in
    probe)  do_probe ;;
    launch) do_launch ${RUNS[@]+"${RUNS[@]}"} ;;
    status) do_status ;;
esac
