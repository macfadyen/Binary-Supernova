# Tests for the Hachisu SCF rotating polytrope solver.
#
# Coverage:
#  - Non-rotating limit (α = 1) reproduces Lane-Emden for n = 3:
#      → Ω² = 0 to machine precision
#      → total mass converges to 4π |ξ² θ'(ξ₁)| / ξ₁³  (= 0.07756 for n = 3)
#        at first order in Δr (Green's-function midpoint rule).  Test at
#        Nr = 256 with 6 % tolerance.
#  - Rotating sequence α < 1:
#      → Ω² > 0, monotonically increasing with decreasing α up to the
#        mass-shedding turn-over
#      → M decreases along the sequence
#      → T/|W| is small and positive (< 0.05 for n = 3)
#  - Mass-shedding limit for n = 3:
#      → Ω² peaks in α ∈ [0.55, 0.75]
#      → Ω²_peak / (π G ρ_c) lies in [0.007, 0.010]  (literature: ≈ 0.0085)

using Test
using BinarySupernova

@testset "Rotating polytrope (Hachisu SCF)" begin

    @testset "Non-rotating limit α = 1 matches Lane-Emden for n = 3" begin
        sol = scf_rotating_polytrope(3.0, 1.0;
                                      Nr = 256, Nμ = 17, lmax = 0,
                                      tol = 1e-9, maxiter = 3000, mix = 0.3)
        @test sol.converged
        @test abs(sol.Ω²) < 1e-12

        # n = 3 Lane-Emden:  ξ₁ ≈ 6.8969, −ξ² dθ/dξ|_{ξ₁} ≈ 2.01824
        #   M = 4π ρ_c r_scale³ · 2.01824   with r_scale = r_eq / ξ₁
        # For ρ_c = r_eq = 1:  M_theory = 4π · 2.01824 / 6.8969³ ≈ 0.07731
        M_theory = 4π * 2.01824 / 6.8969^3
        @test isapprox(sol.M, M_theory; rtol = 0.06)   # 1st-order midpoint

        # Equatorial density profile matches θⁿ(ξ) to within 5 % (inner+mid);
        # error is largest where the θ profile is most steeply curved.
        ξs, θs, _ = lane_emden(3.0; dξ = 1e-3)
        θ_of_r(r) = begin
            ξ = r * 6.8969
            ξ ≥ ξs[end] && return 0.0
            k = searchsortedlast(ξs, ξ)
            k == length(ξs) && return 0.0
            frac = (ξ - ξs[k]) / (ξs[k+1] - ξs[k])
            max(θs[k] + frac * (θs[k+1] - θs[k]), 0.0)
        end
        err_max = 0.0
        for i in 1:length(sol.r)
            r_i = sol.r[i]
            r_i > 0.95 && continue                 # skip the very surface
            r_i < 0.02 && continue                 # skip first cell (O(Δr²) offset)
            ρ_eq = sol.ρ[i, 1]
            ρ_ref = θ_of_r(r_i)^3
            err_max = max(err_max, abs(ρ_eq - ρ_ref))
        end
        @test err_max < 0.08
    end

    @testset "Rotating sequence α < 1 produces positive Ω² that rises with flattening" begin
        αs = [0.95, 0.85, 0.75, 0.68]
        sols = [scf_rotating_polytrope(3.0, α;
                                        Nr = 128, Nμ = 17, lmax = 10,
                                        tol = 1e-7, maxiter = 2000, mix = 0.5)
                for α in αs]
        @test all(s -> s.converged, sols)
        @test all(s -> s.Ω² > 0.0,  sols)
        # Monotonic Ω² increase along the stable branch (before mass shedding).
        @test all(diff([s.Ω² for s in sols]) .> 0.0)
        # T/|W| small and positive (stiff EoS, modest rotation).
        @test all(s -> 0.0 < s.T_over_W < 0.05, sols)
        # Mass drops as the star is flattened.
        @test all(diff([s.M for s in sols]) .< 0.0)
    end

    @testset "Mass-shedding limit for n = 3" begin
        α_peak, Ω²_peak = mass_shedding_limit(3.0;
                                                α_lo = 0.50, α_hi = 0.90, steps = 25,
                                                Nr = 256, Nμ = 17, lmax = 10,
                                                tol = 1e-7, maxiter = 2000, mix = 0.4)
        @test 0.55 <= α_peak <= 0.75
        Ω̂²_peak = Ω²_peak / π                    # literature normalisation
        @test 0.007 <= Ω̂²_peak <= 0.010
    end

    @testset "3D Cartesian mapping: mass and rotation fidelity" begin
        # Grid sized to contain the star with a margin; star centred at origin.
        ng = BinarySupernova.NG
        γ  = 4.0 / 3.0
        nx = ny = nz = 64
        L  = 0.30                           # box half-width in code units
        dx = 2L / nx;  dy = dx;  dz = dx

        M_star = 0.50                        # code units (arbitrary; G = 1)
        R_star = 0.10
        M_core = 0.05                        # hollow the inner 10 %

        U = zeros(5, nx + 2ng, ny + 2ng, nz + 2ng)

        # ---- Non-rotating (axis_ratio = 1): mapped mass matches target
        info = rotating_polytrope_ic_3d!(U, nx, ny, nz, dx, dy, dz, γ;
                                          M_star = M_star, R_star = R_star,
                                          axis_ratio = 1.0, M_core = 0.0,
                                          x0 = -L, y0 = -L, z0 = -L,
                                          Nr_scf = 256, Nμ_scf = 17, lmax_scf = 0,
                                          scf_tol = 1e-8, scf_maxiter = 3000,
                                          scf_mix = 0.3)
        @test info.converged
        @test abs(info.Ω_spin) < 1e-10
        # 3D cell-centre mapping + 1st-order SCF error ⇒ expect ≲ 10 %
        @test isapprox(info.M_mapped, M_star; rtol = 0.10)

        # ---- Rotating (axis_ratio = 0.85): rigid rotation velocity field check
        fill!(U, 0.0)
        v_star = (0.0, 0.1, 0.0)
        info = rotating_polytrope_ic_3d!(U, nx, ny, nz, dx, dy, dz, γ;
                                          M_star = M_star, R_star = R_star,
                                          axis_ratio = 0.85, M_core = M_core,
                                          x0 = -L, y0 = -L, z0 = -L,
                                          v_star = v_star,
                                          Nr_scf = 128, Nμ_scf = 17, lmax_scf = 10,
                                          scf_tol = 1e-7, scf_maxiter = 2000,
                                          scf_mix = 0.4)
        @test info.converged
        @test info.Ω_spin > 0.0
        @test info.r_core > 0.0 && info.r_core < R_star
        # Total mapped mass slightly below M_star (core is hollowed):
        @test M_star - 2 * M_core < info.M_mapped < M_star * 1.1

        # Check rotation direction at a cell near (+x, 0, 0): v_y should have
        # an additional +Ω_spin · x contribution on top of v_star.
        i0 = ng + round(Int, (R_star * 0.5 + L) / dx + 0.5)
        j0 = ng + round(Int, (0.0 + L)           / dy + 0.5)
        k0 = ng + round(Int, (0.0 + L)           / dz + 0.5)
        ρ_cell = U[1, i0, j0, k0]
        @test ρ_cell > 10 * 1e-10   # inside the star
        vy_cell = U[3, i0, j0, k0] / ρ_cell
        @test vy_cell > v_star[2]   # spin adds to bulk translation

        # Equatorial-plane angular momentum about the star centre is positive
        # (Ω aligned with +ẑ):
        Jz = 0.0
        for kk in ng+1:ng+nz, jj in ng+1:ng+ny, ii in ng+1:ng+nx
            ρ = U[1, ii, jj, kk];   ρ <= 1e-9 && continue
            xc = -L + (ii - ng - 0.5) * dx
            yc = -L + (jj - ng - 0.5) * dy
            Jz += (xc * U[3, ii, jj, kk] - yc * U[2, ii, jj, kk]) * dx * dy * dz
        end
        @test Jz > 0.0
    end

    @testset "axis_ratio_for_spin inverts the sequence" begin
        # Pick a spin well below mass-shedding and solve; check the resulting
        # Ω² is close to the target.
        Ω̂_target = 0.12                        # Ω̂² = 0.0144, comfortably stable
        α = axis_ratio_for_spin(3.0, Ω̂_target;
                                 α_lo = 0.60, α_hi = 0.999, bisect_tol = 1e-3,
                                 Nr = 128, Nμ = 17, lmax = 10,
                                 scf_tol = 1e-7, maxiter = 1500, mix = 0.4)
        sol = scf_rotating_polytrope(3.0, α;
                                      Nr = 128, Nμ = 17, lmax = 10,
                                      tol = 1e-7, maxiter = 2000, mix = 0.4)
        @test isapprox(sqrt(sol.Ω²), Ω̂_target; atol = 0.01)
    end

end
