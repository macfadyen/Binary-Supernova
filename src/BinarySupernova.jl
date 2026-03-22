module BinarySupernova

# Ghost-cell count — shared by all spatial modules.
const NG = 3

# ---------------------------------------------------------------------------
# WENO5-Z reconstruction and SSP-RK3 integrator — verbatim from HighMachCBD.
# Borges et al. (2008) WENO-Z; Shu & Osher (1988) SSP-RK3.

include("weno5.jl")
include("rk3.jl")

export weno5_left, weno5_right, weno5_reconstruct_interface
export linrec5_left, linrec5_right
export rk3_step!

# ---------------------------------------------------------------------------
# Phase 1: 3D adiabatic hydrodynamics.

include("hllc.jl")
include("euler3d.jl")

export hllc_flux_x, hllc_flux_y, hllc_flux_z
export euler3d_step!, cfl_dt_3d
export fill_ghost_3d_outflow!, fill_ghost_3d_periodic!
export apply_floors_3d!
export sedov_ic_3d!

# ---------------------------------------------------------------------------
# Physical constants (CGS)

const G_CGS     = 6.67430e-8    # cm³ g⁻¹ s⁻²
const C_CGS     = 2.99792458e10 # cm s⁻¹
const M_SUN     = 1.98892e33    # g
const R_SUN     = 6.95700e10    # cm

# ---------------------------------------------------------------------------
# PhysicalUnits — maps code units to CGS given two physical scales.
#
# Code units: G = 1, M_total = 1, a0 = 1
#   → velocity unit  v_unit = sqrt(G_phys * M_total_phys / a0_phys)
#   → time unit      t_unit = a0_phys / v_unit
#   → c in code units         c_code = C_CGS / v_unit

"""
    PhysicalUnits(M_total_Msun, a0_Rsun)

Derive code-unit conversion factors from two physical scales:
  - `M_total_Msun`: total system mass in solar masses (M_BH1 + M_star)
  - `a0_Rsun`     : initial binary separation in solar radii

In code units G = M_total = a0 = 1, so:
  v_unit = sqrt(G M_total / a0)   [cm/s]
  t_unit = a0 / v_unit             [s]
  c_code = C_CGS / v_unit          [dimensionless; ~500 for compact binary at 10 R☉]
"""
struct PhysicalUnits
    M_total_Msun :: Float64   # input
    a0_Rsun      :: Float64   # input
    M_unit       :: Float64   # g  (= M_total_phys)
    L_unit       :: Float64   # cm (= a0_phys)
    v_unit       :: Float64   # cm/s
    t_unit       :: Float64   # s
    rho_unit     :: Float64   # g/cm³  (= M_unit / L_unit³)
    E_unit       :: Float64   # erg/cm³ (= M_unit / (L_unit t_unit²))  [energy density]
    c_code       :: Float64   # speed of light in code units
end

function PhysicalUnits(M_total_Msun::Real, a0_Rsun::Real)
    M_unit  = M_total_Msun * M_SUN               # g
    L_unit  = a0_Rsun      * R_SUN               # cm
    v_unit  = sqrt(G_CGS * M_unit / L_unit)      # cm/s
    t_unit  = L_unit / v_unit                    # s
    rho_unit = M_unit / L_unit^3                 # g/cm³
    E_unit   = M_unit / (L_unit * t_unit^2)      # erg/cm³  (pressure unit)
    c_code   = C_CGS / v_unit
    return PhysicalUnits(Float64(M_total_Msun), Float64(a0_Rsun),
                         M_unit, L_unit, v_unit, t_unit,
                         rho_unit, E_unit, c_code)
end

export PhysicalUnits

# ---------------------------------------------------------------------------
# BlackHole — mutable point-mass sink.
#
# r_sink = max(6 G M_BH / c², r_floor)   [ISCO radius; floor = ~2 Δx_fine]
# t_sink = f_sink * r_sink / v_ff        [free-fall timescale at sink boundary]
#           where v_ff = sqrt(2 G M_BH / r_sink) = sqrt(2 M_BH / r_sink)  [G=1]

"""
    BlackHole(pos, vel, mass, eps, c_code, r_floor)

Mutable point-mass black hole for N-body + sink integration.

Fields
- `pos`    : position [x, y, z] in code units
- `vel`    : velocity [vx, vy, vz] in code units
- `mass`   : current mass in code units (grows via accretion)
- `eps`    : gravitational softening radius (code units)
- `c_code` : speed of light in code units (from PhysicalUnits.c_code)
- `r_floor`: minimum sink radius (code units); set to ~2 Δx_fine at init

Derived quantities (functions, not stored):
- `r_sink(bh)` = max(6 bh.mass / bh.c_code², bh.r_floor)   [ISCO or floor]
- `t_sink(bh, f)` = f * r_sink(bh) / sqrt(2 bh.mass / r_sink(bh))
"""
mutable struct BlackHole
    pos    :: Vector{Float64}   # length 3, [x, y, z]
    vel    :: Vector{Float64}   # length 3, [vx, vy, vz]
    mass   :: Float64
    eps    :: Float64           # gravitational softening (code units)
    c_code :: Float64           # speed of light in code units
    r_floor:: Float64           # minimum sink radius (code units)
end

"""
    BlackHole(; pos, vel, mass, eps, units::PhysicalUnits, r_floor)

Convenience constructor using a `PhysicalUnits` instance to supply `c_code`.
"""
function BlackHole(; pos::Vector{Float64},
                     vel::Vector{Float64},
                     mass::Float64,
                     eps::Float64,
                     units::PhysicalUnits,
                     r_floor::Float64)
    return BlackHole(pos, vel, mass, eps, units.c_code, r_floor)
end

"""
    r_sink(bh::BlackHole) -> Float64

Sink radius: ISCO = 6 G M_BH / c² (G=1 in code units), floored at bh.r_floor.
Grows automatically as bh.mass increases via fallback accretion.
"""
r_sink(bh::BlackHole) = max(6.0 * bh.mass / bh.c_code^2, bh.r_floor)

"""
    t_sink(bh::BlackHole, f_sink=1.0) -> Float64

Sink removal timescale: f_sink × (free-fall time at sink boundary).
  t_ff = r_sink / v_ff,  v_ff = sqrt(2 M_BH / r_sink)   [G=1]
"""
function t_sink(bh::BlackHole, f_sink::Float64=1.0)
    rs = r_sink(bh)
    v_ff = sqrt(2.0 * bh.mass / rs)
    return f_sink * rs / v_ff
end

export BlackHole, r_sink, t_sink

# ---------------------------------------------------------------------------
# SimParams — all physics parameters for a run.

"""
    SimParams

All physical and numerical parameters for a BinarySupernova run.

Binary system
- `M_BH1`       : BH1 mass (code units; M_BH1 + M_star = 1)
- `M_star`      : initial stellar mass (code units)
- `M_BH2_init`  : BH2 mass at collapse (code units); M_ejecta = M_star − M_BH2_init
- `a0`          : initial separation (= 1 in code units)

Explosion
- `E_SN`        : supernova energy (code units)
- `r_bomb`      : thermal bomb deposition radius (code units; default = R_star)
- `v_kick`      : natal kick velocity vector for BH2, length-3 (code units)

EOS
- `gamma`       : adiabatic index (4/3 for radiation-dominated star; 5/3 for ideal gas)

Sinks
- `f_sink`      : sink timescale multiplier (default 1.0)
- `torque_free` : use torque-free sink prescription (default true)

Numerics
- `rho_floor`   : density floor (code units)
- `cfl`         : CFL number (default 0.4)
"""
struct SimParams
    # Binary
    M_BH1       :: Float64
    M_star      :: Float64
    M_BH2_init  :: Float64
    a0          :: Float64
    # Explosion
    E_SN        :: Float64
    r_bomb      :: Float64
    v_kick      :: Vector{Float64}   # length 3
    # EOS
    gamma       :: Float64
    # Sinks
    f_sink      :: Float64
    torque_free :: Bool
    # Numerics
    rho_floor   :: Float64
    cfl         :: Float64
end

function SimParams(;
    M_BH1       :: Float64,
    M_star      :: Float64,
    M_BH2_init  :: Float64,
    a0          :: Float64  = 1.0,
    E_SN        :: Float64,
    r_bomb      :: Float64,
    v_kick      :: Vector{Float64} = [0.0, 0.0, 0.0],
    gamma       :: Float64  = 4/3,
    f_sink      :: Float64  = 1.0,
    torque_free :: Bool     = true,
    rho_floor   :: Float64  = 1e-10,
    cfl         :: Float64  = 0.4)

    @assert M_BH1 > 0        "M_BH1 must be positive"
    @assert M_star > 0       "M_star must be positive"
    @assert M_BH2_init > 0   "M_BH2_init must be positive"
    @assert M_BH2_init < M_star  "BH2 remnant mass must be less than stellar mass"
    @assert E_SN > 0         "E_SN must be positive"
    @assert r_bomb > 0       "r_bomb must be positive"
    @assert length(v_kick) == 3  "v_kick must be a length-3 vector"
    @assert gamma > 1        "gamma must be > 1"
    @assert f_sink > 0       "f_sink must be positive"
    @assert rho_floor >= 0   "rho_floor must be non-negative"
    @assert 0 < cfl < 1      "CFL must be in (0, 1)"

    return SimParams(M_BH1, M_star, M_BH2_init, a0,
                     E_SN, r_bomb, v_kick,
                     gamma, f_sink, torque_free,
                     rho_floor, cfl)
end

"""
    M_ejecta(p::SimParams) -> Float64

Mass of gas placed on the grid at t=0: M_star minus the collapsed BH2 core.
"""
M_ejecta(p::SimParams) = p.M_star - p.M_BH2_init

export SimParams, M_ejecta

end # module BinarySupernova
