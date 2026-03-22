# SSP-RK3 N-body integrator for two or more black holes.
#
# Equations of motion (G = 1):
#   dr_i/dt = v_i
#   dv_i/dt = Σ_{j≠i} M_j (r_j − r_i) / (|r_j − r_i|² + ε_j²)^(3/2)
#            + F_gas→i / M_i
#
# Softening ε_j is the softening of the source BH j.
# F_gas→i is the pre-computed gravitational force on BH i from the gas
# (evaluated before the step and held constant throughout; see gravity_bh.jl).
#
# Integration uses the same SSP-RK3 Shu-Osher scheme as the gas solver
# (Shu & Osher 1988) for consistent time-centering.

# ---------------------------------------------------------------------------
# Internal helpers

"""
    _bh_bh_accel(bhi, bhj) -> (ax, ay, az)

Gravitational acceleration on `bhi` due to `bhj` (G = 1, Plummer softening ε_j).
Direction: from bhi toward bhj (attractive).
"""
@inline function _bh_bh_accel(bhi::BlackHole, bhj::BlackHole)
    dx = bhj.pos[1] - bhi.pos[1]
    dy = bhj.pos[2] - bhi.pos[2]
    dz = bhj.pos[3] - bhi.pos[3]
    r2  = dx^2 + dy^2 + dz^2 + bhj.eps^2
    r3  = r2 * sqrt(r2)
    fac = bhj.mass / r3
    return fac * dx, fac * dy, fac * dz
end

"""
    _nbody_rhs!(dpos, dvel, bhs, F_gas)

Fill `dpos[i] = v_i` and `dvel[i] = total acceleration on BH i`
from BH-BH gravity plus the pre-computed gas forces `F_gas[i]`.
"""
function _nbody_rhs!(dpos::Vector{Vector{Float64}},
                     dvel::Vector{Vector{Float64}},
                     bhs ::Vector{BlackHole},
                     F_gas::Vector{Vector{Float64}})
    n = length(bhs)
    for i in 1:n
        dpos[i] = copy(bhs[i].vel)
        a = F_gas[i] / bhs[i].mass
        for j in 1:n
            j == i && continue
            ax, ay, az = _bh_bh_accel(bhs[i], bhs[j])
            a = a + [ax, ay, az]
        end
        dvel[i] = a
    end
    return nothing
end

# ---------------------------------------------------------------------------
# Public integrator

"""
    nbody_step!(bhs, dt, F_gas)

Advance all BHs by one SSP-RK3 step of size `dt`.

`F_gas[i]` is the gravitational force vector (length 3) on BH i from the gas,
computed from the gas state at the beginning of the step and held constant
throughout the RK substages.  Pass `[zeros(3) for _ in bhs]` for a gas-free run.

For a coupled gas+N-body simulation, call `gas_force_on_bh` for each BH before
calling this function, then advance the gas with its own SSP-RK3 step (which
includes BH gravity sources via `add_bh_gravity_source!`).
"""
function nbody_step!(bhs::Vector{BlackHole}, dt::Float64,
                     F_gas::Vector{Vector{Float64}})
    n = length(bhs)

    pos0 = [copy(bhs[i].pos) for i in 1:n]
    vel0 = [copy(bhs[i].vel) for i in 1:n]

    dpos = [zeros(3) for _ in 1:n]
    dvel = [zeros(3) for _ in 1:n]

    # --- Stage 1: U⁽¹⁾ = Uⁿ + dt * L(Uⁿ) ---
    _nbody_rhs!(dpos, dvel, bhs, F_gas)
    pos1 = [pos0[i] + dt * dpos[i] for i in 1:n]
    vel1 = [vel0[i] + dt * dvel[i] for i in 1:n]
    for i in 1:n; bhs[i].pos = pos1[i]; bhs[i].vel = vel1[i]; end

    # --- Stage 2: U⁽²⁾ = 3/4 Uⁿ + 1/4 (U⁽¹⁾ + dt * L(U⁽¹⁾)) ---
    _nbody_rhs!(dpos, dvel, bhs, F_gas)
    pos2 = [0.75 * pos0[i] + 0.25 * (pos1[i] + dt * dpos[i]) for i in 1:n]
    vel2 = [0.75 * vel0[i] + 0.25 * (vel1[i] + dt * dvel[i]) for i in 1:n]
    for i in 1:n; bhs[i].pos = pos2[i]; bhs[i].vel = vel2[i]; end

    # --- Stage 3: Uⁿ⁺¹ = 1/3 Uⁿ + 2/3 (U⁽²⁾ + dt * L(U⁽²⁾)) ---
    _nbody_rhs!(dpos, dvel, bhs, F_gas)
    for i in 1:n
        bhs[i].pos = (1/3) * pos0[i] + (2/3) * (pos2[i] + dt * dpos[i])
        bhs[i].vel = (1/3) * vel0[i] + (2/3) * (vel2[i] + dt * dvel[i])
    end

    return nothing
end
