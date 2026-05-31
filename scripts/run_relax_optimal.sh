#!/usr/bin/env bash
# Co-rotating Roche-relaxation IC run for the BinarySupernova CBD campaign.
#
# The --relax-ic analogue of run_cbd_optimal.sh.  Instead of the Hachisu SCF
# figure (rotationally flattened but tidally SYMMETRIC — no L1/L2 bulge), the
# progenitor is built by relaxing a Lane-Emden polytrope in the binary's
# CO-ROTATING frame (relax_ic! with Ω = Ω_orb): centrifugal + Coriolis forces
# settle the star into the genuine tidal (Roche) shape — the L1/L2-facing
# bulge that carries orbital-scale angular momentum, the CBD-seeding feature
# the SCF IC structurally lacks.  See memory project_predict_disk.md.
#
# Config:
#   - --relax-ic               co-rotating Roche relaxation; self-gravity
#                              auto-on; relaxation auto-stops at its KE minimum
#   - mass-based bomb          --bomb-mass-frac 0.5: deposit E_SN in the inner
#                              half of the envelope BY MASS — robust to the
#                              relaxed star's expansion — spherical (no cone)
#   - E_SN small (--e-sn-frac 0.05)   bound envelope survives → fallback-rich
#   - Blaauw-safe partition (ΔM/M_preSN = 10/45 = 0.22)   binary stays bound
#   - small prograde tangential kick   widens apastron, periastron near r_0
#   - a₀ = 2.5 R☉              tight orbit → strong tidal distortion / near
#                              Roche-filling → pronounced L1/L2 bulge
#
# Usage:
#   scripts/run_relax_optimal.sh                  # default NX=384, GPU
#   scripts/run_relax_optimal.sh --nx 512
#   scripts/run_relax_optimal.sh --cpu --nx 256   # CPU run (L=6 needs NX≥256)
#   scripts/run_relax_optimal.sh --relax-t-max 9  # longer relaxation
#   scripts/run_relax_optimal.sh --t-end 30
#
# NOTE: the co-rotating relaxation prepends a damped relaxation phase to the
# run (self-gravity every RK substage) — significant cost at NX=384.
# Production target is the A100; the L=6 box needs NX ≥ 256 to resolve the star
# (R_star/dx = NX/30), and ≥ 384 to match the earlier L=2 runs.
#
# Output dir is auto-named from the key parameters.

set -euo pipefail

# ---------- defaults
NX=384
T_END=20.0
DT_SNAP=0.5
RELAX_T_MAX=3.0
USE_GPU=1
THREADS=16

# ---------- arg parse
while [[ $# -gt 0 ]]; do
  case "$1" in
    --nx)          NX="$2";          shift 2 ;;
    --t-end)       T_END="$2";       shift 2 ;;
    --dt-snap)     DT_SNAP="$2";     shift 2 ;;
    --relax-t-max) RELAX_T_MAX="$2"; shift 2 ;;
    --cpu)         USE_GPU=0;        shift   ;;
    --threads)     THREADS="$2";     shift 2 ;;
    *) echo "Unknown arg: $1" >&2; exit 1 ;;
  esac
done

# ---------- physics + numerics
M_BH1_MSUN=15
M_STAR_MSUN=30
M_BH2_MSUN=20             # ΔM = 10 M☉, ΔM/M_preSN = 10/45 = 0.22 (Blaauw-safe)
A0_RSUN=2.5               # tight orbit — strong tidal distortion in the relax IC
L=6.0                     # ±6 a₀ box — holds the post-SN-widened binary + the R>2 r_sep CBD region
E_SN_FRAC=0.05            # 5 × 10⁴⁹ erg — weak SN, bound envelope survives
BOMB_MASS_FRAC=0.5        # bomb the inner half of the envelope BY MASS (spherical)
V_KICK_Y=-0.015           # small prograde tangential kick (BH2 moves in -y at t=0)
SINK_DELAY=5.0            # let bomb gas clear r_sink(BH2) before sink arms
RHO_FLOOR=3e-4
P_FLOOR=1e-5

# ---------- GPU / CPU dispatch
GPU_FLAG=""
if [[ "$USE_GPU" == "1" ]]; then
  GPU_FLAG="--gpu"
fi

# ---------- run from the project root so relative paths resolve
cd "$(dirname "$0")/.."

# ---------- output dir
TAG="relax_optimal_a${A0_RSUN/./}rsun_L${L%.*}_bh${M_BH1_MSUN}_${M_BH2_MSUN}_esn${E_SN_FRAC/./}_bombmf${BOMB_MASS_FRAC/./}_kick${V_KICK_Y/-/m}_nx${NX}"
OUTDIR="demo1/output_${TAG}"
mkdir -p "$OUTDIR"

# ---------- launch
LOG="$OUTDIR/run.log"
echo "Launching Roche-relaxation run → $OUTDIR"
echo "  NX=$NX  L=$L  a₀=$A0_RSUN R☉  relax_t_max=$RELAX_T_MAX  GPU=$USE_GPU  t_end=$T_END"

CMD=(julia --project=. -t "$THREADS" scripts/run_sn50_fiducial.jl
     --m-bh1-msun "$M_BH1_MSUN"
     --m-star-msun "$M_STAR_MSUN"
     --m-bh2-msun "$M_BH2_MSUN"
     --a0-rsun "$A0_RSUN"
     --L "$L"
     --relax-ic
     --relax-t-max "$RELAX_T_MAX"
     --self-gravity --torque-free
     --e-sn-frac "$E_SN_FRAC"
     --bomb-mass-frac "$BOMB_MASS_FRAC"
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
