# Phase 4 tests: Lane-Emden polytrope solver and 3D stellar IC.
#
# Test 1 — Lane-Emden ODE: ξ_1 within 0.1% of Chandrasekhar (1939) values.
# Test 2 — 3D mass integral: ∫ρ dV matches M_star within 2% on a 32³ grid.
# Test 3 — Pressure profile: P = K ρ^γ satisfied to floating-point precision.
# Test 4 — Short hydro evolution (5 steps): no NaN / Inf, solver stable.
# Test 5 — BH1 orbit check: Keplerian initial velocity gives centripetal
#           acceleration consistent with the BH-BH force from N-body.
#
# Note: long-time stability of the star requires self-gravity (Phase 7).
# Phase 4 tests focus on IC quality and one-step numerical stability.

@testset "Lane-Emden solver — ξ_1 accuracy" begin

    # Chandrasekhar (1939) first-zero values for several polytrope indices
    exact = Dict(1.0 => π, 1.5 => 3.65375, 3.0 => 6.89685)

    for (n, ξ_ref) in sort(collect(exact))
        ξs, θs, dθs = BinarySupernova.lane_emden(n)
        ξ_1 = ξs[end]
        err = abs(ξ_1 - ξ_ref) / ξ_ref
        @test err < 1e-3   # < 0.1%
        @info "Lane-Emden n=$n: ξ_1=$(round(ξ_1, digits=5)), exact=$(round(ξ_ref, digits=5)), err=$(round(err*100, digits=4))%"
    end

end

@testset "Polytrope IC — 3D mass integral" begin

    ng = BinarySupernova.NG
    γ  = 5.0 / 3.0       # n = 1.5

    # 32³ active cells; domain [-0.4, 0.4]³; R_star = 0.3 → ~12 cells per radius
    nx = ny = nz = 32
    L  = 0.4
    dx = 2L / nx
    x0 = -L

    M_star = 0.7
    R_star = 0.3

    U = zeros(5, nx + 2ng, ny + 2ng, nz + 2ng)
    ρ_c, r_scale, K = polytrope_ic_3d!(U, nx, ny, nz, dx, dx, dx, γ;
                                        M_star = M_star, R_star = R_star,
                                        x0 = x0, y0 = x0, z0 = x0)

    dV     = dx^3
    M_grid = sum(U[1, ng+1:ng+nx, ng+1:ng+ny, ng+1:ng+nz]) * dV
    err    = abs(M_grid - M_star) / M_star

    @test err < 0.02   # < 2% grid-integration error (finite-volume discretisation)

    @info "Polytrope mass: M_grid=$(round(M_grid, digits=5)), M_star=$M_star, err=$(round(err*100, digits=3))%"

end

@testset "Polytrope IC — pressure profile P = K ρ^γ" begin

    ng = BinarySupernova.NG
    γ  = 5.0 / 3.0
    nx = ny = nz = 32
    L  = 0.4; dx = 2L / nx; x0 = -L

    M_star = 0.7; R_star = 0.3
    U = zeros(5, nx + 2ng, ny + 2ng, nz + 2ng)
    ρ_c, r_scale, K = polytrope_ic_3d!(U, nx, ny, nz, dx, dx, dx, γ;
                                        M_star = M_star, R_star = R_star,
                                        x0 = x0, y0 = x0, z0 = x0)

    # At every cell with ρ > ρ_floor, check P = K ρ^γ to floating-point precision.
    # (Both P and ρ were set from the same θ, so they share the same rounding.)
    max_err  = 0.0
    n_inside = 0
    ρ_floor  = 1e-10
    for k in ng+1:ng+nz, j in ng+1:ng+ny, i in ng+1:ng+nx
        ρ = U[1, i, j, k]
        ρ > 10 * ρ_floor || continue
        E      = U[5, i, j, k]
        P_grid = (γ - 1.0) * E    # v = 0 at t=0
        P_poly = K * ρ^γ
        max_err = max(max_err, abs(P_grid - P_poly) / P_poly)
        n_inside += 1
    end

    @test max_err < 1e-10   # exact: both computed from same formula

    @info "Pressure profile: max |(P_grid - K ρ^γ)/(K ρ^γ)| = $max_err over $n_inside interior cells"

end

@testset "Polytrope IC — short hydro evolution (no BH)" begin

    ng = BinarySupernova.NG
    γ  = 5.0 / 3.0

    # Coarse grid for speed
    nx = ny = nz = 16
    L  = 1.0; dx = 2L / nx; x0 = -L

    M_star = 0.7; R_star = 0.3
    ρ_floor = 1e-10; P_floor = 1e-8

    U = zeros(5, nx + 2ng, ny + 2ng, nz + 2ng)
    polytrope_ic_3d!(U, nx, ny, nz, dx, dx, dx, γ;
                     M_star = M_star, R_star = R_star,
                     x0 = x0, y0 = x0, z0 = x0,
                     ρ_floor = ρ_floor, P_floor = P_floor)
    fill_ghost_3d_outflow!(U, nx, ny, nz)

    # Evolve 5 steps; star expands freely without self-gravity (expected),
    # but the solver must remain stable and NaN-free.
    cfl = 0.4
    for _ in 1:5
        dt = cfl_dt_3d(U, nx, ny, nz, dx, dx, dx, γ, cfl)
        euler3d_step!(U, nx, ny, nz, dx, dx, dx, dt, γ;
                      bc = :outflow, ρ_floor = ρ_floor, P_floor = P_floor)
    end

    @test !any(isnan, U)
    @test !any(isinf, U)

    @info "Short evolution (5 steps): solver stable, no NaN / Inf"

end

@testset "BH1 Keplerian initial conditions" begin

    # For a Keplerian orbit: the centripetal acceleration of BH1
    # from the total potential equals v²/r (circular orbit).
    # With G=1, M_total = M_BH1 + M_star = 1, a0 = 1:
    #   v_rel = sqrt(M_total / a0) = 1
    # BH1 speed in COM frame: v_BH1 = M_star / M_total * v_rel = M_star.
    M_star = 0.7; M_BH1 = 0.3; a0 = 1.0
    M_total = M_star + M_BH1

    # COM frame positions
    r_BH1 = M_star / M_total * a0    # = 0.7
    r_star = M_BH1 / M_total * a0   # = 0.3

    # Circular velocity of BH1 in COM frame
    v_BH1 = sqrt(M_total / a0) * (M_star / M_total)

    # Set up BH1 and a "phantom" BH2 at the stellar centre (point-mass approximation)
    bh1 = BlackHole([r_BH1, 0.0, 0.0], [0.0, v_BH1, 0.0],
                    M_BH1, 0.05, 1e6, 0.05)
    bh_star = BlackHole([-r_star, 0.0, 0.0], [0.0, 0.0, 0.0],
                         M_star, 0.05, 1e6, 0.05)

    # BH-BH gravitational acceleration on BH1 from bh_star
    ax, ay, az = BinarySupernova._bh_bh_accel(bh1, bh_star)

    # Expected centripetal acceleration: v² / r = v_BH1² / r_BH1 (inward = -x)
    a_centripetal = v_BH1^2 / r_BH1

    # The computed acceleration should point toward the star (negative x)
    # and its magnitude should match the centripetal requirement.
    @test ax < 0.0                                          # attractive, toward star
    @test abs(abs(ax) - a_centripetal) / a_centripetal < 0.01  # within 1%

    @info "Keplerian IC: |a_BH1| = $(round(abs(ax), digits=5)), centripetal = $(round(a_centripetal, digits=5))"

end
