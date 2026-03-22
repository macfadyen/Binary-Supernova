# Phase 2 plotting script: FMR 4:1 Sedov test — density comparison and error.
# Reproduces the test computation from test_fmr3d.jl.
# Saves figures to docs/figures/.

using BinarySupernova
using CairoMakie
using Statistics: mean

CairoMakie.activate!()

println("Running FMR 4:1 Sedov test (reproducing test_fmr3d.jl)...")

ng       = BinarySupernova.NG
γ        = 5.0 / 3.0
E_blast  = 1.0
ρ_bg     = 1.0
P_floor  = 1e-5
ρ_floor  = 1e-10
r_inject = 0.15
cfl      = 0.4
t_end    = 0.01

nc       = 8
L        = 0.5
dx_c     = 2L / nc           # = 0.125
ratio    = 4
dx_f     = dx_c / ratio      # = 0.03125

ci_lo = 3;  ci_hi = 6
cj_lo = 3;  cj_hi = 6
ck_lo = 3;  ck_hi = 6
nf    = (ci_hi - ci_lo + 1) * ratio   # = 16

x_lo      = -L
x_fine_lo = (ci_lo - 1) * dx_c + x_lo   # = -0.25

# --- FMR run ---
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

sedov_ic_3d!(G.fine.U, nf, nf, nf, dx_f, dx_f, dx_f, γ;
             E_blast   = E_blast,
             r_inject  = r_inject,
             ρ_bg      = ρ_bg,
             P_floor   = P_floor,
             x_offset  = x_fine_lo,
             y_offset  = x_fine_lo,
             z_offset  = x_fine_lo)
BinarySupernova._prolong_fine_ghosts!(G.fine.U, G.coarse.U, G)

let t = 0.0
    while t < t_end
        dt = cfl_dt_fmr3d(G, cfl)
        dt = min(dt, t_end - t)
        fmr3d_step!(G, dt)
        t += dt
    end
end

# --- Reference: uniform 32³ ---
nref  = nc * ratio    # = 32
U_ref = zeros(5, nref + 2ng, nref + 2ng, nref + 2ng)
sedov_ic_3d!(U_ref, nref, nref, nref, dx_f, dx_f, dx_f, γ;
             E_blast   = E_blast,
             r_inject  = r_inject,
             ρ_bg      = ρ_bg,
             P_floor   = P_floor,
             x_offset  = x_lo,
             y_offset  = x_lo,
             z_offset  = x_lo)

let t = 0.0
    while t < t_end
        dt = cfl_dt_3d(U_ref, nref, nref, nref, dx_f, dx_f, dx_f, γ, cfl)
        dt = min(dt, t_end - t)
        euler3d_step!(U_ref, nref, nref, nref, dx_f, dx_f, dx_f, dt, γ;
                      bc = :outflow, ρ_floor = ρ_floor, P_floor = P_floor)
        t += dt
    end
end

# --- Extract 1D x-profiles through y=z=midplane ---
# FMR fine level: 16 active cells from x_fine_lo to x_fine_lo + nf*dx_f
j_mid_f = ng + nf÷2 + 1
k_mid_f = ng + nf÷2 + 1
xs_fmr = [x_fine_lo + (i - 0.5)*dx_f for i in 1:nf]
ρ_fmr_1d = [G.fine.U[1, ng+i, j_mid_f, k_mid_f] for i in 1:nf]

# Reference: matching region
i_lo_ref = round(Int, (x_fine_lo - x_lo) / dx_f) + 1   # = 9
i_hi_ref = i_lo_ref + nf - 1                             # = 24
j_mid_r = ng + nref÷2 + 1
k_mid_r = ng + nref÷2 + 1
xs_ref = [x_lo + (i - 0.5)*dx_f for i in i_lo_ref:i_hi_ref]
ρ_ref_1d = [U_ref[1, ng+i, j_mid_r, k_mid_r] for i in i_lo_ref:i_hi_ref]

# Full reference profile for context
xs_ref_full = [x_lo + (i - 0.5)*dx_f for i in 1:nref]
ρ_ref_full  = [U_ref[1, ng+i, j_mid_r, k_mid_r] for i in 1:nref]

ρ_err = abs.(ρ_fmr_1d .- ρ_ref_1d)

println("Max relative L1 error in fine region: $(round(mean(ρ_err)/mean(ρ_ref_1d)*100, digits=3))%")

# --- Figure ---
fig = Figure(size=(900, 400))

ax1 = Axis(fig[1,1], xlabel="x", ylabel="Density ρ",
           title="FMR vs Reference (t=$(t_end))")
# Shade fine region
poly!(ax1, [x_fine_lo, x_fine_lo + nf*dx_f, x_fine_lo + nf*dx_f, x_fine_lo],
      [0, 0, maximum(ρ_ref_full)*1.1, maximum(ρ_ref_full)*1.1],
      color=(:lightblue, 0.3), strokecolor=:transparent)
lines!(ax1, xs_ref_full, ρ_ref_full, color=:black, linewidth=2, label="Reference 32³")
scatter!(ax1, xs_fmr, ρ_fmr_1d, color=:royalblue, markersize=6, label="FMR fine (16³)")
vlines!(ax1, [x_fine_lo, x_fine_lo + nf*dx_f], color=:gray, linestyle=:dash, linewidth=1)
axislegend(ax1, position=:rt)
text!(ax1, x_fine_lo + 0.01, maximum(ρ_ref_full)*0.85, text="Fine patch", fontsize=11)

ax2 = Axis(fig[1,2], xlabel="x", ylabel="|ρ_FMR − ρ_ref|",
           title="Density error in fine region",
           yscale=log10)
poly!(ax2, [x_fine_lo, x_fine_lo + nf*dx_f, x_fine_lo + nf*dx_f, x_fine_lo],
      [1e-8, 1e-8, maximum(ρ_err)*2, maximum(ρ_err)*2],
      color=(:lightblue, 0.3), strokecolor=:transparent)
scatter!(ax2, xs_fmr, max.(ρ_err, 1e-12), color=:crimson, markersize=6)
vlines!(ax2, [x_fine_lo, x_fine_lo + nf*dx_f], color=:gray, linestyle=:dash, linewidth=1)
ylims!(ax2, 1e-6, nothing)

save("docs/figures/phase2_fmr_density.png", fig, px_per_unit=2)
println("Saved: docs/figures/phase2_fmr_density.png")
