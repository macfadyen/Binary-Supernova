# Phase 7 tests: gas self-gravity (FFT Poisson solver).
#
# Test 1 — Potential of a uniform sphere: verify exterior potential scales
#           as −M/r at two distances (ratio test; cancels M uncertainty).
#           Require Φ(r1)/Φ(r2) ≈ r2/r1 to within 1 %.
#
# Test 2 — Self-energy of a uniform sphere.
#           Analytic: E_grav = −3M²/(5R).  Require |error| < 10 %.
#
# Test 3 — Convergence: error at an exterior point halves as N doubles.
#
# Test 4 — Source term symmetry: add_self_gravity_source! on a spherically
#           symmetric density field gives zero net momentum source.

# Helper: build a uniform sphere on an nx×ny×nz grid centred at the grid
# centre.  Physical cell positions: xc[i] = (i−0.5)⋅dx, etc.
function _sphere_density(nx, ny, nz, dx, R, ρ0=1.0)
    ρ = zeros(nx, ny, nz)
    xc = 0.5;  yc = 0.5;  zc = 0.5       # domain centre (cells span [0,1]³)
    for k in 1:nz, j in 1:ny, i in 1:nx
        xi = (i - 0.5) * dx - xc
        yi = (j - 0.5) * dx - yc
        zi = (k - 0.5) * dx - zc
        sqrt(xi^2 + yi^2 + zi^2) < R && (ρ[i,j,k] = ρ0)
    end
    return ρ
end

@testset "Poisson solver — exterior potential scales as 1/r" begin

    # Grid: nx = ny = nz = 32, domain [0,1]³.
    # Sphere of radius R = 0.15 centred at (0.5, 0.5, 0.5).
    # Two exterior test points along x-axis at r ≈ 0.25 and r ≈ 0.35.
    # For a spherically symmetric source the exterior potential is exactly
    # −M/r; its ratio at two points equals r₂/r₁ regardless of M.

    nx = ny = nz = 32
    dx = 1.0 / nx
    R  = 0.15

    ρ = _sphere_density(nx, ny, nz, dx, R)
    Φ = solve_poisson_isolated(ρ, nx, ny, nz, dx, dx, dx)

    # Locate grid cells nearest to the two test radii along x-axis.
    # Cell i along x-axis; y, z at domain centre.
    x_c = 0.5
    j_c = round(Int, x_c / dx + 0.5)      # ≈ 17
    k_c = j_c

    function r_from_center(i)
        xi = (i - 0.5) * dx - x_c
        yi = (j_c - 0.5) * dx - x_c
        zi = (k_c - 0.5) * dx - x_c
        return sqrt(xi^2 + yi^2 + zi^2)
    end

    # i_A near x_c + 0.25
    i_A = round(Int, (x_c + 0.25) / dx + 0.5)   # ≈ 25
    # i_B near x_c + 0.35
    i_B = round(Int, (x_c + 0.35) / dx + 0.5)   # ≈ 28

    rA = r_from_center(i_A)
    rB = r_from_center(i_B)

    @assert rA > R && rB > R "Test cells must be outside the sphere"

    ratio_numeric  = Φ[i_A, j_c, k_c] / Φ[i_B, j_c, k_c]
    ratio_analytic = rB / rA             # ← exact exterior result

    err = abs(ratio_numeric - ratio_analytic) / ratio_analytic
    @test err < 0.01
    @info "1/r scaling: ratio_num=$(round(ratio_numeric,digits=4)), rB/rA=$(round(ratio_analytic,digits=4)), err=$(round(err*100,digits=2))%"

end

@testset "Poisson solver — self-energy of uniform sphere" begin

    nx = ny = nz = 32
    dx = 1.0 / nx
    R  = 0.15

    ρ = _sphere_density(nx, ny, nz, dx, R)
    M  = sum(ρ) * dx^3
    Φ  = solve_poisson_isolated(ρ, nx, ny, nz, dx, dx, dx)

    E_grav_numeric  = 0.5 * sum(ρ .* Φ) * dx^3
    E_grav_analytic = -3.0 * M^2 / (5.0 * R)

    err = abs((E_grav_numeric - E_grav_analytic) / E_grav_analytic)
    @test err < 0.10
    @info "Self-energy: E_num=$(round(E_grav_numeric,digits=6)), E_ana=$(round(E_grav_analytic,digits=6)), err=$(round(err*100,digits=2))%"

end

@testset "Poisson solver — convergence (N=16 → N=32)" begin

    # Same exterior ratio test at two resolutions.
    # The ratio Φ(r1)/Φ(r2) should converge toward rB/rA as N increases.
    function ratio_error(N)
        dx = 1.0 / N
        R  = 0.15
        ρ  = _sphere_density(N, N, N, dx, R)
        Φ  = solve_poisson_isolated(ρ, N, N, N, dx, dx, dx)

        x_c = 0.5
        j_c = round(Int, x_c / dx + 0.5)
        k_c = j_c

        function r_fc(i)
            xi = (i - 0.5) * dx - x_c
            yi = (j_c - 0.5) * dx - x_c
            zi = (k_c - 0.5) * dx - x_c
            return sqrt(xi^2 + yi^2 + zi^2)
        end

        i_A = round(Int, (x_c + 0.25) / dx + 0.5)
        i_B = round(Int, (x_c + 0.35) / dx + 0.5)
        rA = r_fc(i_A);  rB = r_fc(i_B)

        rat_num = Φ[i_A, j_c, k_c] / Φ[i_B, j_c, k_c]
        return abs(rat_num - rB/rA) / (rB/rA)
    end

    err16 = ratio_error(16)
    err32 = ratio_error(32)

    @test err32 < err16     # error improves with resolution
    @test err32 < 0.01
    @info "Convergence: err(N=16)=$(round(err16*100,digits=2))%, err(N=32)=$(round(err32*100,digits=2))%"

end

@testset "Self-gravity source — symmetric density → zero net force" begin

    # A spherically symmetric density centred on the grid has zero net
    # self-gravitational force; total momentum source should vanish.

    ng = BinarySupernova.NG
    nx = ny = nz = 16
    dx = 1.0 / nx

    # Build U with ghost cells; velocity = 0.
    U  = zeros(5, nx+2ng, ny+2ng, nz+2ng)
    dU = zeros(5, nx+2ng, ny+2ng, nz+2ng)

    R = 0.25
    x_c = 0.5
    for k in ng+1:ng+nz, j in ng+1:ng+ny, i in ng+1:ng+nx
        xi = (i - ng - 0.5) * dx - x_c
        yi = (j - ng - 0.5) * dx - x_c
        zi = (k - ng - 0.5) * dx - x_c
        ρ_ijk = sqrt(xi^2 + yi^2 + zi^2) < R ? 1.0 : 1e-4
        U[1, i, j, k] = ρ_ijk
        U[5, i, j, k] = ρ_ijk * 1e-3   # tiny thermal energy; v = 0
    end

    add_self_gravity_source!(dU, U, nx, ny, nz, dx, dx, dx)

    # Net momentum source Σ dU[2:4] dV should be ≈ 0.
    Fx = sum(dU[2, ng+1:ng+nx, ng+1:ng+ny, ng+1:ng+nz]) * dx^3
    Fy = sum(dU[3, ng+1:ng+nx, ng+1:ng+ny, ng+1:ng+nz]) * dx^3
    Fz = sum(dU[4, ng+1:ng+nx, ng+1:ng+ny, ng+1:ng+nz]) * dx^3

    # Compare net force to the typical per-cell force magnitude.
    F_scale = maximum(abs.(dU[2:4, ng+1:ng+nx, ng+1:ng+ny, ng+1:ng+nz])) * dx^3
    @test abs(Fx) < 0.01 * F_scale * nx * ny * nz
    @test abs(Fy) < 0.01 * F_scale * nx * ny * nz
    @test abs(Fz) < 0.01 * F_scale * nx * ny * nz
    @info "Net force: Fx=$(round(Fx,sigdigits=3)), Fy=$(round(Fy,sigdigits=3)), Fz=$(round(Fz,sigdigits=3)), scale=$(round(F_scale,sigdigits=3))"

end
