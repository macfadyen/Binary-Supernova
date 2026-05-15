#!/usr/bin/env bash
# Optimal CBD-formation IC for the BinarySupernova campaign.
#
# Targets the narrow sweet spot identified in the a₀=2,5 R☉ scan:
#   - E_SN small enough that the bound envelope survives  (--e-sn-frac 0.05)
#   - bipolar bomb axis-aligned with stellar spin         (preserves equatorial AM)
#   - inner-half deposition                                (outer envelope shock-driven)
#   - Blaauw-safe mass partition (ΔM/M_preSN ≈ 0.22)      (binary stays bound)
#   - small prograde tangential kick                       (widens apastron, periastron at r_0)
#   - SCF α = 0.72                                         (Ω_spin/Ω_orb ≈ 1.5; near breakup)
#   - sink delay = 5 code-t                                (gas rearranges before BH2 eats)
#
# Resolution requirement is set by the inner-half shell thickness
#   shell ≈ (0.5 - r_core/R_star) · R_star/a₀
# At a₀=3 R☉ with M_BH2/M_star = 0.67 (n=3 polytrope: r_core/R_star ≈ 0.42):
#   shell = (0.5 - 0.42) · 0.333 ≈ 0.027 code
# Need shell/dx ≥ 2.5 cells → NX ≥ 380 at L=2.
#
# Usage:
#   scripts/run_cbd_optimal.sh                # default NX=384, GPU
#   scripts/run_cbd_optimal.sh --nx 512       # override NX
#   scripts/run_cbd_optimal.sh --cpu --nx 192 # CPU run (testing)
#   scripts/run_cbd_optimal.sh --t-end 30     # extend evolution
#
# Output dir is auto-named from the key parameters.

set -euo pipefail

# ---------- defaults
NX=384
T_END=20.0
DT_SNAP=0.5
USE_GPU=1
THREADS=16

# ---------- arg parse
while [[ $# -gt 0 ]]; do
  case "$1" in
    --nx)      NX="$2";      shift 2 ;;
    --t-end)   T_END="$2";   shift 2 ;;
    --dt-snap) DT_SNAP="$2"; shift 2 ;;
    --cpu)     USE_GPU=0;    shift   ;;
    --threads) THREADS="$2"; shift 2 ;;
    *) echo "Unknown arg: $1" >&2; exit 1 ;;
  esac
done

# ---------- physics + numerics
M_BH1_MSUN=15
M_STAR_MSUN=30
M_BH2_MSUN=20             # ΔM = 10 M☉, ΔM/M_preSN = 10/45 = 0.22 (Blaauw-safe)
A0_RSUN=3.0
L=2.0                     # ±2 a₀ box
ALPHA=0.72                # SCF axis ratio → Ω_spin/Ω_orb ≈ 1.5
E_SN_FRAC=0.05            # 5 × 10⁴⁹ erg
BOMB_OUTER=0.5            # inner-half deposition (r_core < r < R_star/2)
BIPOLAR_DEG=20            # narrow polar cones, axis along z (= spin axis)
V_KICK_Y=-0.015           # small prograde tangential kick (BH2 moves in -y at t=0)
SINK_DELAY=5.0            # let bomb gas clear r_sink(BH2) before sink arms
RHO_FLOOR=3e-4
P_FLOOR=1e-5

# ---------- GPU / CPU dispatch
GPU_FLAG=""
if [[ "$USE_GPU" == "1" ]]; then
  GPU_FLAG="--gpu"
fi

# ---------- output dir
TAG="cbd_optimal_a${A0_RSUN%.*}rsun_bh${M_BH1_MSUN}_${M_BH2_MSUN}_alpha${ALPHA/./}_bipolar${BIPOLAR_DEG}_innerhalf_esn${E_SN_FRAC/./}_kick${V_KICK_Y/-/m}_nx${NX}"
OUTDIR="demo1/output_${TAG}"
mkdir -p "$OUTDIR"

# ---------- launch
LOG="$OUTDIR/run.log"
echo "Launching CBD-optimal run → $OUTDIR"
echo "  NX=$NX  L=$L  a₀=$A0_RSUN R☉  GPU=$USE_GPU  t_end=$T_END"

cd "$(dirname "$0")/.."

CMD=(julia --project=. -t "$THREADS" scripts/run_sn50_fiducial.jl
     --m-bh1-msun "$M_BH1_MSUN"
     --m-star-msun "$M_STAR_MSUN"
     --m-bh2-msun "$M_BH2_MSUN"
     --a0-rsun "$A0_RSUN"
     --L "$L"
     --scf-ic --scf-axis-ratio "$ALPHA"
     --self-gravity --torque-free
     --e-sn-frac "$E_SN_FRAC"
     --r-bomb-outer-frac "$BOMB_OUTER"
     --bipolar-theta-deg "$BIPOLAR_DEG"
     --v-kick-y "$V_KICK_Y"
     --bh2-sink-delay "$SINK_DELAY"
     --rho-floor "$RHO_FLOOR"
     --p-floor "$P_FLOOR"
     --nx "$NX"
     --t-end "$T_END"
     --dt-snap "$DT_SNAP"
     --outdir "$OUTDIR")

if [[ -n "$GPU_FLAG" ]]; then
  CMD+=("$GPU_FLAG")
fi

printf '%s ' "${CMD[@]}" >&2; echo >&2

nohup "${CMD[@]}" > "$LOG" 2>&1 &
PID=$!
echo "Run PID: $PID"
echo "$PID" > "$OUTDIR/run.pid"

if command -v caffeinate >/dev/null 2>&1; then
  nohup caffeinate -s -i -w "$PID" > /dev/null 2>&1 &
  echo "Caffeinate PID: $!"
fi

echo
echo "Log:    tail -f $LOG"
echo "Diags:  tail -f $OUTDIR/diagnostics.csv"
