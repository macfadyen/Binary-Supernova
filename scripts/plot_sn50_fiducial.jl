#!/usr/bin/env julia
# Plot summary figure for the SN50 fiducial run (live orbit + sinks).
#
# Usage:  julia --project=. scripts/plot_sn50_fiducial.jl [--outdir PATH]
#                                                          [--a0-rsun A0]
# Output: <outdir>/figures/sn50_fiducial_summary.png

using BinarySupernova
using CairoMakie
using DelimitedFiles: readdlm
using Printf

CairoMakie.activate!()

outdir_arg = let v = "demo1/output_sn50_fiducial"
    for (i, arg) in enumerate(ARGS)
        if arg == "--outdir" && i < length(ARGS)
            v = ARGS[i+1]
        end
    end
    v
end
a0_rsun_arg = let v = 20.0
    for (i, arg) in enumerate(ARGS)
        if arg == "--a0-rsun" && i < length(ARGS)
            v = parse(Float64, ARGS[i+1])
        end
    end
    v
end

const OUTDIR  = outdir_arg
const FIGDIR  = joinpath(OUTDIR, "figures")
const DT_SNAP = 0.5
const A0_RSUN = a0_rsun_arg
mkpath(FIGDIR)

# Code-unit reference values for the 50+50 setup
const M_TOT_MSUN = 105.0
const M_BH1_0    = 50.0 / M_TOT_MSUN     # 0.4762
const M_BH2_0    = 45.0 / M_TOT_MSUN     # 0.4286
const M_EJECTA   = 10.0 / M_TOT_MSUN     # 0.0952
const units      = PhysicalUnits(M_TOT_MSUN, A0_RSUN)

# ---------------------------------------------------------------------------
# Load trajectory + diagnostics

@info "Reading trajectory + diagnostics..." outdir=OUTDIR
t_traj, bh_data = read_trajectory(joinpath(OUTDIR, "trajectory.h5"))
bh1 = bh_data[1];  bh2 = bh_data[2]

D        = readdlm(joinpath(OUTDIR, "diagnostics.csv"), ',', Float64; skipstart=1)
t_diag   = D[:, 1]
E_gas    = D[:, 7]
Jz_gas   = D[:, 8]
M_bound  = D[:, 9]
Fg1      = D[:, 10:12]
Fg2      = D[:, 13:15]
Fg1_mag  = sqrt.(sum(Fg1.^2, dims=2)[:])
Fg2_mag  = sqrt.(sum(Fg2.^2, dims=2)[:])

N_traj  = length(t_traj)
r_sep   = [sqrt(sum((bh2.pos[k, :] .- bh1.pos[k, :]).^2)) for k in 1:N_traj]
M_BH1   = bh1.mass
M_BH2   = bh2.mass

# ---------------------------------------------------------------------------
# Density-slice snapshots: t = 0, 5, 15, 30

snap_times = [0, 5, 15, 30]
snap_idx   = [round(Int, t / DT_SNAP) for t in snap_times]
snap_files = [joinpath(OUTDIR, @sprintf("snap_t%03d.h5", i)) for i in snap_idx]
existing   = [(t, f) for (t, f) in zip(snap_times, snap_files) if isfile(f)]
@info "Density panels" existing=[t for (t,_) in existing]

# ---------------------------------------------------------------------------
# Build figure

fig = Figure(size = (1500, 850))

Label(fig[0, 1:5],
      @sprintf("Binary-SN 50+55 M☉ → 50+45 M☉, a₀ = %.0f R☉, live orbit + sinks",
               A0_RSUN),
      fontsize = 15, font = :bold)

# ---- Top row: density slices (z = 0 midplane, log scale)
for (k, (snap_t, snap_f)) in enumerate(existing)
    ax = Axis(fig[1, k],
              title  = "t = $snap_t  ($(round(snap_t/(2π), digits=2)) P₀)",
              xlabel = k == 1 ? "x / a₀" : "",
              ylabel = k == 1 ? "y / a₀" : "",
              aspect = DataAspect())
    k > 1 && hideydecorations!(ax, ticks=false)

    U_s, nx, ny, nz, dx, dy, dz, t_s, γ = read_snapshot(snap_f)
    kz = nz ÷ 2 + 1
    ρ_slice = U_s[1, :, :, kz]
    L_dom = nx * dx / 2
    xs = LinRange(-L_dom + dx/2, L_dom - dx/2, nx)

    hm = heatmap!(ax, xs, xs, log10.(max.(ρ_slice, 1e-12)),
                   colormap = :inferno, colorrange = (-5, 1))
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

# ---- Bottom row: time-series panels

# (1) BH separation — binary fate
ax_r = Axis(fig[2, 1],
            xlabel = "t (code)", ylabel = "r_sep / a₀",
            title  = "Binary separation")
lines!(ax_r, t_traj, r_sep, color = :royalblue, linewidth = 1.5)
hlines!(ax_r, [1.0], color = :gray, linestyle = :dash,
        linewidth = 1, label = "a₀")
axislegend(ax_r, position = :rb)

# (2) Orbit in xy-plane
ax_orb = Axis(fig[2, 2],
              xlabel = "x / a₀", ylabel = "y / a₀",
              title  = "BH orbits (xy)",
              aspect = DataAspect())
lines!(ax_orb, bh1.pos[:, 1], bh1.pos[:, 2], color = :crimson,   linewidth = 1.0, label = "BH1")
lines!(ax_orb, bh2.pos[:, 1], bh2.pos[:, 2], color = :darkgreen, linewidth = 1.0, label = "BH2")
scatter!(ax_orb, [bh1.pos[1, 1]], [bh1.pos[1, 2]], color = :crimson,   markersize = 8)
scatter!(ax_orb, [bh2.pos[1, 1]], [bh2.pos[1, 2]], color = :darkgreen, markersize = 8)
axislegend(ax_orb, position = :rt)

# (3) BH masses
ax_m = Axis(fig[2, 3],
            xlabel = "t", ylabel = "M_BH (code units)",
            title  = "BH accretion histories")
lines!(ax_m, t_traj, M_BH1, color = :crimson,   linewidth = 2, label = "M_BH1")
lines!(ax_m, t_traj, M_BH2, color = :darkgreen, linewidth = 2, label = "M_BH2")
hlines!(ax_m, [M_BH1_0], color = :crimson,   linestyle = :dash, linewidth = 1)
hlines!(ax_m, [M_BH2_0], color = :darkgreen, linestyle = :dash, linewidth = 1)
hlines!(ax_m, [M_BH2_0 + M_EJECTA], color = :black, linestyle = :dot, linewidth = 1,
        label = "M_BH2_0 + M_ejecta")
axislegend(ax_m, position = :rb)

# (4) Bound gas mass
ax_bnd = Axis(fig[2, 4],
              xlabel = "t", ylabel = "M_bound",
              title  = "Bound gas mass")
lines!(ax_bnd, t_diag, M_bound, color = :teal, linewidth = 1.5)
hlines!(ax_bnd, [M_EJECTA], color = :black, linestyle = :dash,
        linewidth = 1, label = "M_ejecta")
axislegend(ax_bnd, position = :rt)

# (5) Gas total energy drift
ax_e = Axis(fig[2, 5],
            xlabel = "t", ylabel = "E_gas",
            title  = "Total gas energy")
lines!(ax_e, t_diag, E_gas, color = :indigo, linewidth = 1.5)

outpath = joinpath(FIGDIR, "sn50_fiducial_summary.png")
save(outpath, fig, px_per_unit = 2)
@info "Saved figure" path=outpath

# ---------------------------------------------------------------------------
# Brief summary printout

println("\n=== SN50 fiducial summary ($(OUTDIR)) ===")
@printf "  a₀                : %.1f R☉  (v_unit = %.1f km/s, c_code = %.1f)\n" A0_RSUN units.v_unit/1e5 units.c_code
@printf "  Final t           : %.2f  (%.2f P₀)\n"                       t_diag[end]  t_diag[end]/(2π)
@printf "  r_sep range       : [%.4f, %.4f]  (final %.4f)\n"            minimum(r_sep) maximum(r_sep) r_sep[end]
@printf "  M_BH1: %.4f → %.4f   (ΔM = %+.4f, %.1f%% of M_BH1_0)\n"      M_BH1[1]  M_BH1[end]  M_BH1[end]-M_BH1[1]  100*(M_BH1[end]-M_BH1[1])/M_BH1_0
@printf "  M_BH2: %.4f → %.4f   (ΔM = %+.4f, %.1f%% of M_ejecta)\n"     M_BH2[1]  M_BH2[end]  M_BH2[end]-M_BH2[1]  100*(M_BH2[end]-M_BH2[1])/M_EJECTA
@printf "  M_bound (final)   : %.4f   (M_ejecta = %.4f)\n"              M_bound[end]  M_EJECTA
@printf "  E_gas drift       : %+.3e  (init %.3e → final %.3e)\n"       (E_gas[end]-E_gas[1]) E_gas[1] E_gas[end]
@printf "  max |F_gas→BH1|   : %.3e\n"                                  maximum(Fg1_mag)
@printf "  max |F_gas→BH2|   : %.3e\n"                                  maximum(Fg2_mag)
