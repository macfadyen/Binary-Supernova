# FMR-coupled self-gravity tests (Phase 7 extension, ported from Binary-PISN).
#
# Tests:
#   1. Regression: fmr3d_step!(G, dt) ≡ fmr3d_step!(G, dt; self_gravity=false)
#      — opting in with the default kwarg value must not change output.
#   2. solve_fmr_poisson agrees with a manual restrict + solve_poisson_isolated
#      on the composite coarse density.
#   3. Self-gravity step on a centred sphere preserves net momentum to ≈ 1e-12
#      (symmetry check).
#   4. A self-gravitating sphere develops an inward (negative) radial momentum
#      after one step — the gas falls toward the centre as expected.

# Helper: build a two-level FMR grid with a uniform-density sphere centred at
# the origin, v = 0, modest internal energy, and the fine patch covering the
# sphere. Domain [−L, L]³.
function _make_sg_grid(; nc::Int = 8, L::Float64 = 0.5,
                        ratio::Int = 4, R::Float64 = 0.2,
                        ρ0::Float64 = 1.0, ρ_bg::Float64 = 1e-4,
                        γ::Float64 = 5/3, e_int::Float64 = 1e-3)
    ng   = BinarySupernova.NG
    dx_c = 2L / nc
    coarse_lv = FMRLevel3D(nc, nc, nc, dx_c, dx_c, dx_c)

    # Coarse IC: uniform sphere at origin.
    for k in ng+1:ng+nc, j in ng+1:ng+nc, i in ng+1:ng+nc
        xc = (i - ng - 0.5) * dx_c - L
        yc = (j - ng - 0.5) * dx_c - L
        zc = (k - ng - 0.5) * dx_c - L
        r  = sqrt(xc^2 + yc^2 + zc^2)
        ρ  = r < R ? ρ0 : ρ_bg
        coarse_lv.U[1, i, j, k] = ρ
        coarse_lv.U[5, i, j, k] = ρ * e_int         # v = 0 → E = ρ e_int
    end
    fill_ghost_3d_outflow!(coarse_lv.U, nc, nc, nc)

    # Fine patch spans coarse cells (nc/4+1):(3nc/4) in each axis → centred on origin.
    ci_lo = div(nc, 4) + 1;   ci_hi = 3 * div(nc, 4)
    G = FMRGrid3D(coarse_lv, ci_lo, ci_hi, ci_lo, ci_hi, ci_lo, ci_hi,
                  ratio, γ; bc=:outflow, ρ_floor=1e-12, P_floor=1e-12)

    # Fine IC: same uniform-sphere field on the fine grid.
    dx_f     = dx_c / ratio
    nf       = (ci_hi - ci_lo + 1) * ratio
    x_fine_lo = (ci_lo - 1) * dx_c - L
    for k in ng+1:ng+nf, j in ng+1:ng+nf, i in ng+1:ng+nf
        xf = (i - ng - 0.5) * dx_f + x_fine_lo
        yf = (j - ng - 0.5) * dx_f + x_fine_lo
        zf = (k - ng - 0.5) * dx_f + x_fine_lo
        r  = sqrt(xf^2 + yf^2 + zf^2)
        ρ  = r < R ? ρ0 : ρ_bg
        G.fine.U[1, i, j, k] = ρ
        G.fine.U[5, i, j, k] = ρ * e_int
    end
    BinarySupernova._prolong_fine_ghosts!(G.fine.U, G.coarse.U, G)
    return G
end

@testset "fmr3d_step! self_gravity=false regression" begin
    # Default kwarg value must be a no-op vs the positional-only call.
    G1 = _make_sg_grid()
    G2 = _make_sg_grid()

    dt = 1e-4
    fmr3d_step!(G1, dt)
    fmr3d_step!(G2, dt; self_gravity = false)

    @test G1.coarse.U == G2.coarse.U
    @test G1.fine.U   == G2.fine.U
end

@testset "solve_fmr_poisson agrees with manual restrict + Poisson" begin
    G = _make_sg_grid()
    ng = BinarySupernova.NG

    # Reference: restrict density ourselves, then call solve_poisson_isolated
    # on the coarse active-cell block.
    restrict_density_to_coarse!(G)
    nxc, nyc, nzc = G.coarse.nx, G.coarse.ny, G.coarse.nz
    ρ_c_ref = collect(view(G.coarse.U, 1, ng+1:ng+nxc, ng+1:ng+nyc, ng+1:ng+nzc))
    Φ_c_ref = solve_poisson_isolated(ρ_c_ref, nxc, nyc, nzc,
                                      G.coarse.dx, G.coarse.dy, G.coarse.dz)

    # solve_fmr_poisson does restrict + solve + prolong internally.
    G2 = _make_sg_grid()
    Φ_c, Φ_f = solve_fmr_poisson(G2)

    @test Φ_c == Φ_c_ref
    @test size(Φ_f) == (G2.fine.nx, G2.fine.ny, G2.fine.nz)
end

@testset "Self-gravity step preserves net momentum (symmetric sphere)" begin
    # Symmetric ρ, v=0 → force is spherically symmetric → total Δ(ρv) = 0.
    G = _make_sg_grid()
    ng = BinarySupernova.NG

    # Record initial coarse momentum (fine region is overridden by restrict,
    # so coarse alone is not a conserved composite — but for a spherically
    # symmetric IC both levels individually have zero net momentum before and
    # after a step with v=0 initial conditions).
    P0_c = (sum(G.coarse.U[2, ng+1:ng+G.coarse.nx, ng+1:ng+G.coarse.ny, ng+1:ng+G.coarse.nz]),
            sum(G.coarse.U[3, ng+1:ng+G.coarse.nx, ng+1:ng+G.coarse.ny, ng+1:ng+G.coarse.nz]),
            sum(G.coarse.U[4, ng+1:ng+G.coarse.nx, ng+1:ng+G.coarse.ny, ng+1:ng+G.coarse.nz]))
    @test all(abs.(P0_c) .< 1e-14)

    dt = 1e-4
    fmr3d_step!(G, dt; self_gravity = true)

    Pc = (sum(G.coarse.U[2, ng+1:ng+G.coarse.nx, ng+1:ng+G.coarse.ny, ng+1:ng+G.coarse.nz]),
          sum(G.coarse.U[3, ng+1:ng+G.coarse.nx, ng+1:ng+G.coarse.ny, ng+1:ng+G.coarse.nz]),
          sum(G.coarse.U[4, ng+1:ng+G.coarse.nx, ng+1:ng+G.coarse.ny, ng+1:ng+G.coarse.nz]))
    Pf = (sum(G.fine.U[2, ng+1:ng+G.fine.nx, ng+1:ng+G.fine.ny, ng+1:ng+G.fine.nz]),
          sum(G.fine.U[3, ng+1:ng+G.fine.nx, ng+1:ng+G.fine.ny, ng+1:ng+G.fine.nz]),
          sum(G.fine.U[4, ng+1:ng+G.fine.nx, ng+1:ng+G.fine.ny, ng+1:ng+G.fine.nz]))

    # Scale momentum tolerance to the total mass × typical gravity impulse.
    # A conservative bound: |P_net| / M_total ≪ typical per-cell |v| after dt.
    M_c = sum(G.coarse.U[1, ng+1:ng+G.coarse.nx, ng+1:ng+G.coarse.ny, ng+1:ng+G.coarse.nz])
    @test all(abs.(Pc) ./ M_c .< 1e-10)
    @test all(abs.(Pf) ./ M_c .< 1e-10)
end

@testset "Self-gravitating sphere falls inward" begin
    # After one self-gravity step the radial momentum at each cell should
    # point toward the origin (ρ v · r̂ < 0 averaged over the sphere).
    G = _make_sg_grid(R = 0.2, ρ0 = 1.0, ρ_bg = 1e-6, e_int = 1e-4)
    ng = BinarySupernova.NG

    dt = 5e-4
    fmr3d_step!(G, dt; self_gravity = true)

    # Integrate ρ v · r̂ on the fine level (which surrounds the sphere centre).
    nf       = G.fine.nx
    dx_f     = G.fine.dx
    # Fine-region physical offset: active cell (1,1,1) corresponds to the
    # coordinate x_fine_lo + 0.5 dx_f.
    x_fine_lo = (G.ci_lo - 1) * G.coarse.dx - 0.5       # L=0.5 in _make_sg_grid
    radial_mom = 0.0
    for k in ng+1:ng+nf, j in ng+1:ng+nf, i in ng+1:ng+nf
        xf = (i - ng - 0.5) * dx_f + x_fine_lo
        yf = (j - ng - 0.5) * dx_f + x_fine_lo
        zf = (k - ng - 0.5) * dx_f + x_fine_lo
        r  = sqrt(xf^2 + yf^2 + zf^2)
        r < 1e-12 && continue
        r̂x, r̂y, r̂z = xf/r, yf/r, zf/r
        radial_mom += (G.fine.U[2, i, j, k] * r̂x +
                       G.fine.U[3, i, j, k] * r̂y +
                       G.fine.U[4, i, j, k] * r̂z) * dx_f^3
    end

    @test radial_mom < 0.0          # net inward
    @info "Self-gravitating sphere: ∫ρv·r̂ dV = $(round(radial_mom, sigdigits=3)) (should be < 0)"
end
