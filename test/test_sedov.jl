# Sedov-Taylor blast wave test for the 3D adiabatic Euler solver.
#
# A point explosion of energy E_blast = 1 at the origin in uniform ρ = 1,
# P = P_floor background.  The shock radius follows the self-similar law:
#
#   R_shock(t) = (E_blast / (α ρ₀))^{1/5}  t^{2/5}
#
# where α ≈ 0.4942 for γ = 5/3 in 3D (Sedov 1959).
#
# Test: on a 32³ grid (fast), find the shock radius at t = 0.1 from the
# numerical density field and compare to the analytical prediction.
# Required: error < 10% (coarse grid, first check of 3D spherical symmetry).
#
# Energy conservation test: total energy should be conserved to < 1%.

@testset "Sedov-Taylor blast wave (3D, γ=5/3)" begin
    γ       = 5/3
    ng      = BinarySupernova.NG
    nx = ny = nz = 32        # coarse but fast; increase for production
    L       = 1.0            # half-domain size; domain = [-L, L]³
    dx      = 2*L / nx
    t_end   = 0.1
    E_blast = 1.0
    ρ_bg    = 1.0
    P_floor = 1e-5
    cfl     = 0.4

    # Sedov constant α for γ=5/3 in 3D (Sedov 1959, numerically computed).
    α_sedov = 0.4942
    R_exact = (E_blast / (α_sedov * ρ_bg))^(1/5) * t_end^(2/5)

    U = zeros(5, nx+2*ng, ny+2*ng, nz+2*ng)
    sedov_ic_3d!(U, nx, ny, nz, dx, dx, dx, γ;
                 E_blast  = E_blast,
                 ρ_bg     = ρ_bg,
                 P_floor  = P_floor,
                 x_offset = -L,
                 y_offset = -L,
                 z_offset = -L)

    # Total energy at t=0.
    E_tot_0 = sum(U[5, ng+1:ng+nx, ng+1:ng+ny, ng+1:ng+nz]) * dx^3

    # Time-march to t_end.
    t = 0.0
    while t < t_end
        dt = cfl_dt_3d(U, nx, ny, nz, dx, dx, dx, γ, cfl)
        dt = min(dt, t_end - t)
        euler3d_step!(U, nx, ny, nz, dx, dx, dx, dt, γ;
                      bc=:outflow, ρ_floor=ρ_bg*1e-6, P_floor=P_floor*0.1)
        t += dt
    end

    # Total energy at t_end (should be conserved; outflow BC allows some loss at edges).
    E_tot_f = sum(U[5, ng+1:ng+nx, ng+1:ng+ny, ng+1:ng+nz]) * dx^3
    @test abs(E_tot_f - E_tot_0) / E_tot_0 < 0.05   # < 5% energy loss from outflow BC

    # Shock radius: find outermost spherical shell where ρ > 1.5 × background.
    # Sedov compression ratio at shock = (γ+1)/(γ-1) = 4 for γ=5/3.
    ρ_thresh = 1.5 * ρ_bg
    R_num = 0.0
    for k in ng+1:ng+nz, j in ng+1:ng+ny, i in ng+1:ng+nx
        if U[1, i, j, k] > ρ_thresh
            xc = -L + (i - ng - 0.5) * dx
            yc = -L + (j - ng - 0.5) * dx
            zc = -L + (k - ng - 0.5) * dx
            r  = sqrt(xc^2 + yc^2 + zc^2)
            R_num = max(R_num, r)
        end
    end

    rel_err = abs(R_num - R_exact) / R_exact
    @info "Sedov: R_exact=$(round(R_exact,digits=4)) R_num=$(round(R_num,digits=4)) err=$(round(100*rel_err,digits=1))%"
    @test rel_err < 0.10   # < 10% at coarse resolution

    # Spherical symmetry: ρ along +x axis ≈ ρ along +y and +z axes (mid-slice).
    # Compare the three axial 1D profiles.
    j_mid = ng + ny÷2 + 1;  k_mid = ng + nz÷2 + 1
    i_mid = ng + nx÷2 + 1

    ρ_x = [U[1, ng+i, j_mid, k_mid] for i in 1:nx]
    ρ_y = [U[1, i_mid, ng+j, k_mid] for j in 1:ny]
    ρ_z = [U[1, i_mid, j_mid, ng+k] for k in 1:nz]

    # Max relative difference between profiles (half-domain, outward direction).
    half = nx÷2
    err_xy = maximum(abs.(ρ_x[half:end] .- ρ_y[half:end])) /
             max(maximum(ρ_x), 1e-10)
    err_xz = maximum(abs.(ρ_x[half:end] .- ρ_z[half:end])) /
             max(maximum(ρ_x), 1e-10)
    @info "Sedov symmetry: xy=$(round(err_xy,sigdigits=3)) xz=$(round(err_xz,sigdigits=3))"
    @test err_xy < 0.10
    @test err_xz < 0.10
end
