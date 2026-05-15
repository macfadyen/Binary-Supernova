#!/usr/bin/env julia
# Plot summary figure for the SN30 fixed-orbit run.
#
# Usage:  julia --project=. scripts/plot_sn30_fixedorbit.jl
# Output: demo1/figures_sn30_fixedorbit/sn30_fixedorbit_summary.png

using BinarySupernova
using CairoMakie
using DelimitedFiles: readdlm
using Printf

CairoMakie.activate!()

const OUTDIR  = "demo1/output_sn30_fixedorbit"
const FIGDIR  = "demo1/figures_sn30_fixedorbit"
const DT_SNAP = 0.5
mkpath(FIGDIR)

# ---------------------------------------------------------------------------
# Load trajectory + diagnostics

@info "Reading trajectory + diagnostics..."
t_traj, bh_data = read_trajectory(joinpath(OUTDIR, "trajectory.h5"))
bh1 = bh_data[1];  bh2 = bh_data[2]

D        = readdlm(joinpath(OUTDIR, "diagnostics.csv"), ',', Float64; skipstart=1)
t_diag   = D[:, 1]
E_gas    = D[:, 7]
Jz_gas   = D[:, 8]
M_bound  = D[:, 9]
Fg1      = D[:, 10:12]     # gas→BH1 force (x, y, z)
Fg2      = D[:, 13:15]     # gas→BH2 force
Fg1_mag  = sqrt.(sum(Fg1.^2, dims=2)[:])
Fg2_mag  = sqrt.(sum(Fg2.^2, dims=2)[:])

# BH state from the trajectory file — includes the pre-step t=0 sample, so
# initial masses read as exact 1/3 (diagnostics.csv records post-step only).
N_traj = length(t_traj)
r_bh   = [sqrt(sum((bh2.pos[k, :] .- bh1.pos[k, :]).^2)) for k in 1:N_traj]
M_BH1  = bh1.mass
M_BH2  = bh2.mass

# ---------------------------------------------------------------------------
# Pick 4 density-slice snapshots: t = 0, 5, 15, 30

snap_times = [0, 5, 15, 30]
snap_idx   = [round(Int, t / DT_SNAP) for t in snap_times]
snap_files = [joinpath(OUTDIR, @sprintf("snap_t%03d.h5", i)) for i in snap_idx]
existing   = [(t, f) for (t, f) in zip(snap_times, snap_files) if isfile(f)]
@info "Density panels" existing

# ---------------------------------------------------------------------------
# Build figure

fig = Figure(size = (1400, 750))

Label(fig[0, 1:4],
      "Binary-SN 30+30 M☉ — fixed Kepler orbit, NX=128  (Ω = 1, r_sep clamped to 1)",
      fontsize = 15, font = :bold)

# ---- Top row: density slices (z = 0 midplane, log scale)
for (k, (snap_t, snap_f)) in enumerate(existing)
    ax = Axis(fig[1, k],
              title  = "t = $snap_t",
              xlabel = k == 1 ? "x" : "",
              ylabel = k == 1 ? "y" : "",
              aspect = DataAspect())
    k > 1 && hideydecorations!(ax, ticks=false)

    U_s, nx, ny, nz, dx, dy, dz, t_s, γ = read_snapshot(snap_f)
    kz = nz ÷ 2 + 1
    ρ_slice = U_s[1, :, :, kz]
    L_dom = nx * dx / 2
    xs = LinRange(-L_dom + dx/2, L_dom - dx/2, nx)

    hm = heatmap!(ax, xs, xs, log10.(max.(ρ_slice, 1e-12)'),
                   colormap = :inferno, colorrange = (-4, 1))
    if k == length(existing)
        Colorbar(fig[1, k+1], hm, label = "log₁₀ ρ")
    end

    idx = argmin(abs.(t_traj .- Float64(snap_t)))
    scatter!(ax, [bh1.pos[idx, 1]], [bh1.pos[idx, 2]],
             color = :cyan, markersize = 9, marker = :circle,
             strokecolor = :black, strokewidth = 0.5)
    scatter!(ax, [bh2.pos[idx, 1]], [bh2.pos[idx, 2]],
             color = :yellow, markersize = 9, marker = :diamond,
             strokecolor = :black, strokewidth = 0.5)
end

# ---- Bottom row: three time-series panels

# (1) BH separation — confirms the orbit clamp
ax_r = Axis(fig[2, 1],
            xlabel = "t (code units)", ylabel = "BH separation",
            title  = "Binary separation (clamp check)")
lines!(ax_r, t_traj, r_bh, color = :royalblue, linewidth = 1.5)
hlines!(ax_r, [1.0], color = :gray, linestyle = :dash,
        linewidth = 1, label = "a₀ = 1")
ylims!(ax_r, 0.99, 1.01)
axislegend(ax_r, position = :rb)

# (2) BH masses
ax_m = Axis(fig[2, 2],
            xlabel = "t", ylabel = "M_BH",
            title  = "BH masses (accretion on, orbit fixed)")
lines!(ax_m, t_traj, M_BH1, color = :crimson,   linewidth = 2, label = "M_BH1")
lines!(ax_m, t_traj, M_BH2, color = :darkgreen, linewidth = 2, label = "M_BH2")
hlines!(ax_m, [1/3], color = :black, linestyle = :dash,
        linewidth = 1, label = "M₀ = 1/3")
axislegend(ax_m, position = :lt)

# (3) Bound gas mass
ax_bnd = Axis(fig[2, 3],
              xlabel = "t", ylabel = "M_bound",
              title  = "Bound gas mass")
lines!(ax_bnd, t_diag, M_bound, color = :teal, linewidth = 1.5)
hlines!(ax_bnd, [1/3], color = :black,
        linestyle = :dash, linewidth = 1,
        label = "M_ejecta = 1/3")
axislegend(ax_bnd, position = :rb)

# (4) Gas→BH force magnitude — diagnostic only (not fed into motion)
ax_f = Axis(fig[2, 4],
            xlabel = "t", ylabel = "|F_gas→BH|",
            title  = "Gas force on BH (diagnostic)",
            yscale = log10)
pos_mask1 = Fg1_mag .> 0.0
pos_mask2 = Fg2_mag .> 0.0
lines!(ax_f, t_diag[pos_mask1], Fg1_mag[pos_mask1],
       color = :crimson, linewidth = 1.5, label = "on BH1")
lines!(ax_f, t_diag[pos_mask2], Fg2_mag[pos_mask2],
       color = :darkgreen, linewidth = 1.5, label = "on BH2")
axislegend(ax_f, position = :rb)

outpath = joinpath(FIGDIR, "sn30_fixedorbit_summary.png")
save(outpath, fig, px_per_unit = 2)
@info "Saved figure" path=outpath

# ---------------------------------------------------------------------------
# Brief summary printout

println("\n=== SN30 fixed-orbit summary ===")
@printf "  Final t           : %.2f  (%.2f P₀, P₀ = 2π/Ω = 2π)\n"  t_diag[end]  t_diag[end]/(2π)
@printf "  Mean r_sep        : %.6f   (should be exactly 1.0)\n"   sum(r_bh)/length(r_bh)
@printf "  max |r_sep − 1|   : %.3e\n"                             maximum(abs.(r_bh .- 1.0))
@printf "  M_BH1: %.3f → %.3f   (ΔM = %.3f)\n"                     M_BH1[1]  M_BH1[end]  M_BH1[end]-M_BH1[1]
@printf "  M_BH2: %.3f → %.3f   (ΔM = %.3f)\n"                     M_BH2[1]  M_BH2[end]  M_BH2[end]-M_BH2[1]
@printf "  Final M_bound     : %.3f  (includes ambient floor × volume)\n" M_bound[end]
@printf "  max |F_gas→BH1|   : %.3e\n"                             maximum(Fg1_mag)
@printf "  max |F_gas→BH2|   : %.3e\n"                             maximum(Fg2_mag)
@printf "  E_gas drift       : %.3e (initial %.3e → final %.3e)\n" (E_gas[end]-E_gas[1]) E_gas[1] E_gas[end]
@printf "  Jz_gas drift      : %.3e\n"                             (Jz_gas[end]-Jz_gas[1])
