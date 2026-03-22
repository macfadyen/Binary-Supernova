# Phase 5 tests: thermal bomb and BH2 fallback accretion.
#
# Test 1 — Thermal bomb energy: Σ ΔE = E_SN exactly (floating-point).
# Test 2 — Sedov-Taylor from bomb: blast radius at t_end within 5% of exact.
#           Uses uniform background (ρ=1) — same geometry as test_sedov.jl
#           but energy deposited via thermal_bomb! rather than sedov_ic_3d!.
# Test 3 — BH2 fallback: after a thermal bomb the BH2 at the centre accretes;
#           BH2 mass strictly increases over the run.
# Test 4 — Energy budget: total gas energy at t_end vs t=0+ (after bomb) is
#           within 15% (outflow BC allows some loss; bomb energy dominates).

@testset "Thermal bomb — energy deposition" begin

    ng = BinarySupernova.NG
    nx = ny = nz = 16
    dx = 1.0 / nx            # dx = 0.0625; domain [0, 1]³
    x0 = 0.0

    γ = 5.0 / 3.0
    E_SN = 1.0
    r_bomb = 0.2

    # Uniform background gas
    U = zeros(5, nx + 2ng, ny + 2ng, nz + 2ng)
    for k in ng+1:ng+nz, j in ng+1:ng+ny, i in ng+1:ng+nx
        U[1, i, j, k] = 1.0
        U[5, i, j, k] = 1e-5 / (γ - 1)
    end

    E_before = sum(U[5, ng+1:ng+nx, ng+1:ng+ny, ng+1:ng+nz])
    xc = 0.5;  yc = 0.5;  zc = 0.5    # centre of domain [0,1]³

    M_bomb = thermal_bomb!(U, nx, ny, nz, dx, dx, dx;
                            E_SN = E_SN, r_bomb = r_bomb,
                            x0 = x0, y0 = x0, z0 = x0,
                            x_center = xc, y_center = yc, z_center = zc)

    E_after = sum(U[5, ng+1:ng+nx, ng+1:ng+ny, ng+1:ng+nz])
    ΔE = E_after - E_before

    # ΔE should equal E_SN * dx^3 * (number of cells in r_bomb) / M_bomb * ρ *...
    # Actually: Σ ΔE[cell] = Σ (E_SN * ρ * dV / M_bomb) = E_SN * M_bomb / M_bomb = E_SN
    # Wait — ΔE in code is the DENSITY energy; we need to multiply by dV for total energy.
    ΔE_total = ΔE * dx^3   # (sum of E[i,j,k] is in erg/cm³; multiply by cell volume)
    # Actually U[5] is energy density E [code units: erg/cm³], so:
    # ΔE_total = Σ (U[5,after] - U[5,before]) * dV = E_SN exactly.
    # But sum(U[5,...]) is already summed over cells, so multiply by dV:
    @test abs(ΔE * dx^3 - E_SN) / E_SN < 1e-10

    @info "Thermal bomb: M_bomb=$(round(M_bomb, digits=5)), ΔE_total=$(round(ΔE*dx^3, digits=8)), E_SN=$E_SN"

end

@testset "Sedov-Taylor from thermal bomb" begin

    ng = BinarySupernova.NG
    γ  = 5.0 / 3.0
    ρ_bg   = 1.0
    E_SN   = 1.0
    cfl    = 0.4
    t_end  = 0.1
    ρ_floor = 1e-10
    P_floor = 1e-10

    # 32³ grid on [−0.5, 0.5]³ (same as test_sedov.jl)
    nx = ny = nz = 32
    L  = 0.5; dx = 2L / nx; x0 = -L

    # Uniform background
    U = zeros(5, nx + 2ng, ny + 2ng, nz + 2ng)
    for k in ng+1:ng+nz, j in ng+1:ng+ny, i in ng+1:ng+nx
        U[1, i, j, k] = ρ_bg
        U[5, i, j, k] = 1e-5 / (γ - 1)
    end

    # Inject explosion energy over r_bomb = 0.1 (same as test_sedov.jl's r_inject)
    r_bomb = 0.1
    thermal_bomb!(U, nx, ny, nz, dx, dx, dx;
                  E_SN = E_SN, r_bomb = r_bomb,
                  x0 = x0, y0 = x0, z0 = x0)
    fill_ghost_3d_outflow!(U, nx, ny, nz)

    # Evolve to t_end
    t = 0.0
    while t < t_end
        dt = cfl_dt_3d(U, nx, ny, nz, dx, dx, dx, γ, cfl)
        dt = min(dt, t_end - t)
        euler3d_step!(U, nx, ny, nz, dx, dx, dx, dt, γ;
                      bc = :outflow, ρ_floor = ρ_floor, P_floor = P_floor)
        t += dt
    end

    # Sedov-Taylor shock radius (Sedov 1959): R_s = (E_SN / (α ρ))^{1/5} t^{2/5}
    # α_sedov = 0.4942 for γ=5/3 in 3D (matches test_sedov.jl)
    α_sedov = 0.4942
    R_exact = (E_SN / (α_sedov * ρ_bg))^0.2 * t_end^0.4

    # Estimate numerical shock position: radius of max pressure
    P_max = 0.0; R_num = 0.0
    for k in ng+1:ng+nz, j in ng+1:ng+ny, i in ng+1:ng+nx
        xc = x0 + (i - ng - 0.5) * dx
        yc = x0 + (j - ng - 0.5) * dx   # x0 == y0 == z0 here
        zc = x0 + (k - ng - 0.5) * dx
        ρ  = U[1, i, j, k]
        KE = 0.5 * (U[2,i,j,k]^2 + U[3,i,j,k]^2 + U[4,i,j,k]^2) / ρ
        P  = (γ - 1) * (U[5, i, j, k] - KE)
        if P > P_max
            P_max = P
            R_num = sqrt(xc^2 + yc^2 + zc^2)
        end
    end

    err = abs(R_num - R_exact) / R_exact
    @test err < 0.05   # < 5%

    @info "Sedov from bomb: R_exact=$(round(R_exact, digits=4)), R_num=$(round(R_num, digits=4)), err=$(round(err*100, digits=2))%"

end

@testset "BH2 fallback accretion" begin

    ng = BinarySupernova.NG
    γ      = 5.0 / 3.0
    cfl    = 0.4
    ρ_floor = 1e-10
    P_floor = 1e-8

    # 16³ grid on [−0.5, 0.5]³
    nx = ny = nz = 16
    L  = 0.5; dx = 2L / nx; x0 = -L

    # Polytrope IC (provides the background gas for the bomb)
    M_star     = 0.7; R_star    = 0.3
    M_BH2_init = 0.2
    E_SN       = 0.5

    U = zeros(5, nx + 2ng, ny + 2ng, nz + 2ng)
    polytrope_ic_3d!(U, nx, ny, nz, dx, dx, dx, γ;
                     M_star = M_star, R_star = R_star,
                     x0 = x0, y0 = x0, z0 = x0,
                     ρ_floor = ρ_floor, P_floor = P_floor)
    fill_ghost_3d_outflow!(U, nx, ny, nz)

    # Apply thermal bomb over the whole star
    thermal_bomb!(U, nx, ny, nz, dx, dx, dx;
                  E_SN = E_SN, r_bomb = R_star,
                  x0 = x0, y0 = x0, z0 = x0)

    # Activate BH2 at the stellar centre
    r_floor_bh2 = 2.0 * dx
    bh2 = BlackHole([0.0, 0.0, 0.0], [0.0, 0.0, 0.0],
                     M_BH2_init, 0.01 * r_floor_bh2, 1e6, r_floor_bh2)
    bhs = BlackHole[bh2]
    F_gas = [zeros(3)]
    M0_bh2 = bh2.mass

    # Evolve for 5 steps; BH2 should accrete from the inner gas
    for _ in 1:5
        dt = cfl_dt_3d(U, nx, ny, nz, dx, dx, dx, γ, cfl)
        dU = zeros(size(U))
        # Gas RHS: hydro + sink sources
        Fx = zeros(size(U)); Fy = zeros(size(U)); Fz = zeros(size(U))
        euler3d_rhs!(dU, Fx, Fy, Fz, U, nx, ny, nz, dx, dx, dx, γ;
                     bc = :outflow, ρ_floor = ρ_floor, P_floor = P_floor)
        add_sink_sources!(dU, U, nx, ny, nz, dx, dx, dx, bhs, x0, x0, x0)
        # SSP-RK3 (single-stage Euler for simplicity in this test)
        @. U += dt * dU
        fill_ghost_3d_outflow!(U, nx, ny, nz)
        # N-body + accretion
        nbody_step!(bhs, dt, F_gas)
        accrete!(bh2, U, nx, ny, nz, dx, dx, dx, x0, x0, x0, dt)
    end

    @test bh2.mass > M0_bh2   # BH2 must have accreted gas

    @info "BH2 fallback: M_BH2_0=$(round(M0_bh2, digits=5)), M_BH2_final=$(round(bh2.mass, digits=5))"

end

@testset "Thermal bomb — energy budget" begin

    ng = BinarySupernova.NG
    γ  = 5.0 / 3.0
    ρ_floor = 1e-10; P_floor = 1e-10

    nx = ny = nz = 32
    L  = 0.5; dx = 2L / nx; x0 = -L
    dV = dx^3

    E_SN = 1.0

    U = zeros(5, nx + 2ng, ny + 2ng, nz + 2ng)
    for k in ng+1:ng+nz, j in ng+1:ng+ny, i in ng+1:ng+nx
        U[1, i, j, k] = 1.0
        U[5, i, j, k] = 1e-5 / (γ - 1)
    end

    # Bomb at origin, r_bomb = 0.1
    thermal_bomb!(U, nx, ny, nz, dx, dx, dx;
                  E_SN = E_SN, r_bomb = 0.1,
                  x0 = x0, y0 = x0, z0 = x0)
    fill_ghost_3d_outflow!(U, nx, ny, nz)

    E0 = sum(U[5, ng+1:ng+nx, ng+1:ng+ny, ng+1:ng+nz]) * dV

    # Evolve to t = 0.01 (before shock reaches boundary at L=0.5)
    t = 0.0
    t_end = 0.01
    cfl = 0.4
    while t < t_end
        dt = min(cfl_dt_3d(U, nx, ny, nz, dx, dx, dx, γ, cfl), t_end - t)
        euler3d_step!(U, nx, ny, nz, dx, dx, dx, dt, γ;
                      bc = :outflow, ρ_floor = ρ_floor, P_floor = P_floor)
        t += dt
    end

    E1 = sum(U[5, ng+1:ng+nx, ng+1:ng+ny, ng+1:ng+nz]) * dV
    # Energy can only decrease via outflow BC; the shock hasn't reached the boundary
    # at t=0.01 (R_s ≈ 1.033*(0.01²)^0.2 ≈ 0.16, well inside L=0.5).
    E_loss = (E0 - E1) / E0

    @test abs(E_loss) < 0.15   # energy nearly conserved in early phase (outflow BC)

    @info "Energy budget: E0=$(round(E0, digits=5)), E1=$(round(E1, digits=5)), loss=$(round(E_loss*100, digits=3))%"

end
