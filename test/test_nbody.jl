# Phase 3 test: N-body integrator — Kepler orbit.
#
# Two equal-mass BHs in a circular orbit, no gas.
# M1 = M2 = 0.5,  M_total = 1,  a0 = 1 (separation),  G = 1.
# Circular velocity: v_circ = sqrt(G M_total / a0) / 2 = 0.5.
# Each BH orbits the centre of mass at radius a0/2 = 0.5.
# Orbital period: T = 2π a0 / v_rel = 2π  (v_rel = 2 * v_circ = 1).
#
# Checks (after 10 orbits, t = 20π ≈ 62.83):
#   1. Relative energy error |ΔE/E0| < 0.1%.
#   2. Relative angular-momentum error |ΔL/L0| < 0.1%.

@testset "Kepler orbit — energy and angular momentum" begin

    # -----------------------------------------------------------------------
    # Initial conditions: equal-mass circular orbit in xy-plane.
    M1    = 0.5;  M2    = 0.5
    a0    = 1.0                      # separation
    v_circ = 0.5                     # speed of each BH in COM frame
    eps   = 1e-4                     # softening << a0

    # BH1 at (+a0/2, 0, 0), velocity in +y
    # BH2 at (−a0/2, 0, 0), velocity in −y
    bh1 = BlackHole([+a0/2, 0.0, 0.0],
                    [0.0,  +v_circ, 0.0],
                    M1, eps, 1e6, 0.01)
    bh2 = BlackHole([-a0/2, 0.0, 0.0],
                    [0.0,  -v_circ, 0.0],
                    M2, eps, 1e6, 0.01)
    bhs = BlackHole[bh1, bh2]

    # -----------------------------------------------------------------------
    # Conserved-quantity helpers
    function total_energy(bhs)
        KE = 0.5 * bhs[1].mass * sum(bhs[1].vel .^ 2) +
             0.5 * bhs[2].mass * sum(bhs[2].vel .^ 2)
        dx = bhs[1].pos .- bhs[2].pos
        r  = sqrt(sum(dx .^ 2) + bhs[1].eps^2)
        PE = -bhs[1].mass * bhs[2].mass / r   # G = 1
        return KE + PE
    end

    function total_Lz(bhs)
        L = 0.0
        for bh in bhs
            L += bh.mass * (bh.pos[1] * bh.vel[2] - bh.pos[2] * bh.vel[1])
        end
        return L
    end

    E0 = total_energy(bhs)
    L0 = total_Lz(bhs)

    # -----------------------------------------------------------------------
    # Integrate for 10 orbits.
    T_orb  = 2π             # orbital period = 2π a0 / v_rel (v_rel = 1)
    t_end  = 10.0 * T_orb
    dt     = T_orb / 500    # 500 steps per orbit → well below 0.1% error for SSP-RK3
    F_gas  = [zeros(3), zeros(3)]

    t = 0.0
    while t < t_end
        step = min(dt, t_end - t)
        nbody_step!(bhs, step, F_gas)
        t += step
    end

    E1 = total_energy(bhs)
    L1 = total_Lz(bhs)

    E_err = abs(E1 - E0) / abs(E0)
    L_err = abs(L1 - L0) / abs(L0)

    @test E_err < 1e-3   # < 0.1%
    @test L_err < 1e-3   # < 0.1%

    @info "Kepler orbit: E_err=$(round(E_err*100, digits=4))%, L_err=$(round(L_err*100, digits=4))%"

end
