# Phase 2 tests: 3D FMR (4:1 refinement ratio).
#
# Test: Sedov-Taylor blast wave on a two-level FMR grid.
#   Coarse:  8³ active cells, domain [−0.5, 0.5]³, dx_c = 1/8
#   Fine:   16³ active cells, inner [−0.25, 0.25]³, dx_f = 1/32, ratio=4
#
# Reference: uniform 32³ at dx_f on the full [−0.5, 0.5]³ domain.
# Both FMR and reference use the same r_inject = 0.15 so that the blast
# sphere is well-resolved on the fine mesh and non-trivially small on the
# coarse mesh.  At t_end = 0.01 the shock radius ≈ 0.18, still inside the
# fine region (half-width 0.25), so the C-F ghost cells see only ambient
# background and the two runs should agree well.
#
# Checks:
#   1. L1 density error inside fine region < 5% vs uniform 32³ reference.
#   2. Total energy loss < 15% (outflow BC allows some loss).
#   3. No NaN / Inf.

@testset "FMR 4:1 — Sedov blast crossing C-F boundary" begin

    ng       = BinarySupernova.NG

    γ        = 5.0 / 3.0
    E_blast  = 1.0
    ρ_bg     = 1.0
    P_floor  = 1e-5
    ρ_floor  = 1e-10
    r_inject = 0.15        # fixed physical injection radius (same for all grids)
    cfl      = 0.4
    t_end    = 0.01

    # Coarse grid metadata
    nc       = 8
    L        = 0.5
    dx_c     = 2L / nc           # = 0.125
    ratio    = 4
    dx_f     = dx_c / ratio      # = 0.03125

    # Fine region: coarse cells 3:6 in each direction → [−0.25, 0.25]³
    ci_lo = 3;  ci_hi = 6
    cj_lo = 3;  cj_hi = 6
    ck_lo = 3;  ck_hi = 6
    nf    = (ci_hi - ci_lo + 1) * ratio   # = 16 fine cells per side

    x_lo      = -L               # domain left physical edge
    x_fine_lo = (ci_lo - 1) * dx_c + x_lo   # = −0.25, left edge of fine patch

    # -----------------------------------------------------------------------
    # FMR run

    coarse_lv = FMRLevel3D(nc, nc, nc, dx_c, dx_c, dx_c)
    sedov_ic_3d!(coarse_lv.U, nc, nc, nc, dx_c, dx_c, dx_c, γ;
                 E_blast   = E_blast,
                 r_inject  = r_inject,
                 ρ_bg      = ρ_bg,
                 P_floor   = P_floor,
                 x_offset  = x_lo,
                 y_offset  = x_lo,
                 z_offset  = x_lo)
    fill_ghost_3d_outflow!(coarse_lv.U, nc, nc, nc)

    G = FMRGrid3D(coarse_lv, ci_lo, ci_hi, cj_lo, cj_hi, ck_lo, ck_hi, ratio, γ;
                  bc = :outflow, ρ_floor = ρ_floor, P_floor = P_floor)

    # Override fine active cells with fine-resolution IC (same r_inject).
    # This ensures FMR fine and reference start from identical physics.
    sedov_ic_3d!(G.fine.U, nf, nf, nf, dx_f, dx_f, dx_f, γ;
                 E_blast   = E_blast,
                 r_inject  = r_inject,
                 ρ_bg      = ρ_bg,
                 P_floor   = P_floor,
                 x_offset  = x_fine_lo,
                 y_offset  = x_fine_lo,
                 z_offset  = x_fine_lo)
    # Re-fill ghost cells from coarse after overwriting active cells.
    BinarySupernova._prolong_fine_ghosts!(G.fine.U, G.coarse.U, G)

    E0_fmr = sum(G.coarse.U[5, ng+1:ng+nc, ng+1:ng+nc, ng+1:ng+nc]) * dx_c^3

    t = 0.0
    n_steps = 0
    while t < t_end
        dt = cfl_dt_fmr3d(G, cfl)
        dt = min(dt, t_end - t)
        fmr3d_step!(G, dt)
        t += dt
        n_steps += 1
    end

    E1_fmr = sum(G.coarse.U[5, ng+1:ng+nc, ng+1:ng+nc, ng+1:ng+nc]) * dx_c^3

    # -----------------------------------------------------------------------
    # Reference: uniform 32³ at dx_f on full [−0.5, 0.5]³.
    # Same r_inject so ICs match in the fine region.

    nref  = nc * ratio    # = 32 cells per side at dx_f resolution
    U_ref = zeros(5, nref + 2ng, nref + 2ng, nref + 2ng)
    sedov_ic_3d!(U_ref, nref, nref, nref, dx_f, dx_f, dx_f, γ;
                 E_blast   = E_blast,
                 r_inject  = r_inject,
                 ρ_bg      = ρ_bg,
                 P_floor   = P_floor,
                 x_offset  = x_lo,
                 y_offset  = x_lo,
                 z_offset  = x_lo)

    t_ref = 0.0
    while t_ref < t_end
        dt_ref = cfl_dt_3d(U_ref, nref, nref, nref, dx_f, dx_f, dx_f, γ, cfl)
        dt_ref = min(dt_ref, t_end - t_ref)
        euler3d_step!(U_ref, nref, nref, nref, dx_f, dx_f, dx_f, dt_ref, γ;
                      bc = :outflow, ρ_floor = ρ_floor, P_floor = P_floor)
        t_ref += dt_ref
    end

    # -----------------------------------------------------------------------
    # Comparison: extract fine region from the 32³ reference.
    # Fine patch starts at physical x = x_fine_lo = −0.25.
    # Reference cell i (1-indexed active) has centre x_lo + (i − 0.5)*dx_f.
    # We want x_lo + (i_lo − 0.5)*dx_f = x_fine_lo + 0.5*dx_f
    #   → i_lo = (x_fine_lo − x_lo)/dx_f + 1 = 0.25/dx_f + 1 = 9.
    i_lo_ref = round(Int, (x_fine_lo - x_lo) / dx_f) + 1   # = 9
    i_hi_ref = i_lo_ref + nf - 1                             # = 24

    ρ_fmr = G.fine.U[1, ng+1:ng+nf, ng+1:ng+nf, ng+1:ng+nf]
    ρ_ref = U_ref[1,
                  ng+i_lo_ref : ng+i_hi_ref,
                  ng+i_lo_ref : ng+i_hi_ref,
                  ng+i_lo_ref : ng+i_hi_ref]

    L1_rel   = mean(abs.(ρ_fmr .- ρ_ref)) / mean(ρ_ref)
    E_err    = abs(E1_fmr - E0_fmr) / E0_fmr

    # -----------------------------------------------------------------------
    @test !any(isnan, G.coarse.U)
    @test !any(isinf, G.coarse.U)
    @test !any(isnan, G.fine.U)
    @test !any(isinf, G.fine.U)
    @test E_err   < 0.15
    @test L1_rel  < 0.05

    @info "FMR: n_steps=$n_steps, E_err=$(round(E_err*100,digits=2))%, L1_rel=$(round(L1_rel*100,digits=3))%"

end
