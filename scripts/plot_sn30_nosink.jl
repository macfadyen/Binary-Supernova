#!/usr/bin/env julia
# Plot summary figure for the SN30 no-sink run.
#
# Usage:  julia --project=. scripts/plot_sn30_nosink.jl
# Output: demo1/figures_sn30_nosink/sn30_nosink_summary.png

using BinarySupernova
using CairoMakie
using DelimitedFiles: readdlm
using Printf

CairoMakie.activate!()

const OUTDIR  = "demo1/output_sn30_nosink"
const FIGDIR  = "demo1/figures_sn30_nosink"
const DT_SNAP = 0.5
mkpath(FIGDIR)

# ---------------------------------------------------------------------------
# Load trajectory + diagnostics

@info "Reading trajectory + diagnostics..."
t_traj, bh_data = read_trajectory(joinpath(OUTDIR, "trajectory.h5"))
bh1 = bh_data[1];  bh2 = bh_data[2]

D       = readdlm(joinpath(OUTDIR, "diagnostics.csv"), ',', Float64; skipstart=1)
t_diag  = D[:, 1]
r_sep   = D[:, 6]
E_gas   = D[:, 7]
Jz_gas  = D[:, 8]
M_bound = D[:, 9]

# ---------------------------------------------------------------------------
# Orbit elements from trajectory

N_traj = length(t_traj)
a_orb  = zeros(N_traj);  e_orb = zeros(N_traj);  eps_orb = zeros(N_traj)
r_bh   = zeros(N_traj)

for k in 1:N_traj
    r1 = bh1.pos[k, :];  v1 = bh1.vel[k, :]
    r2 = bh2.pos[k, :];  v2 = bh2.vel[k, :]
    dr = r2 .- r1;  dv = v2 .- v1
    r  = sqrt(sum(dr.^2));  r_bh[k] = r
    M  = bh1.mass[k] + bh2.mass[k]
    eps = 0.5 * sum(dv.^2) - M / r
    eps_orb[k] = eps
    hx = dr[2]*dv[3] - dr[3]*dv[2]
    hy = dr[3]*dv[1] - dr[1]*dv[3]
    hz = dr[1]*dv[2] - dr[2]*dv[1]
    h² = hx^2 + hy^2 + hz^2
    if eps < 0.0
        a_orb[k] = -M / (2 * eps)
        e_orb[k] = sqrt(max(0.0, 1.0 + 2 * eps * h² / M^2))
    else
        a_orb[k] = NaN;  e_orb[k] = NaN
    end
end
bound_mask = eps_orb .< 0.0

# ---------------------------------------------------------------------------
# Pick 4 density-slice snapshots: t = 0, 5, 15, 30

snap_times = [0, 5, 15, 30]
snap_idx   = [round(Int, t / DT_SNAP) for t in snap_times]
snap_files = [joinpath(OUTDIR, @sprintf("snap_t%03d.h5", i)) for i in snap_idx]
existing   = [(t, f) for (t, f) in zip(snap_times, snap_files) if isfile(f)]
@info "Density panels" existing

# ---------------------------------------------------------------------------
# Build figure: 2 rows × 4 cols = density-slice strip on top, 3 time-series
# panels on bottom (one column reserved for shared colourbar).

fig = Figure(size = (1400, 750))

Label(fig[0, 1:4],
      "Binary-SN 30+30 M☉ — no-sink run, NX=128  (stripped WR progenitor, R★=0.1, E_SN=0.3)",
      fontsize = 15, font = :bold)

# ---- Top row: density slices (z=0 midplane, log scale)
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

# ---- Bottom row: time series (3 panels)
ax_r = Axis(fig[2, 1:2],
            xlabel = "t (code units)", ylabel = "BH separation",
            title  = "Binary separation")
lines!(ax_r, t_traj, r_bh, color = :royalblue, linewidth = 1.5)
hlines!(ax_r, [1.0], color = :gray, linestyle = :dash,
        linewidth = 1, label = "a₀ = 1")
axislegend(ax_r, position = :rt)

ax_ae = Axis(fig[2, 3],
             xlabel = "t", ylabel = "a  /  e",
             title  = "Orbital elements")
t_b = t_traj[bound_mask]
if !isempty(t_b)
    lines!(ax_ae, t_b, a_orb[bound_mask], color = :crimson,
           linewidth = 2, label = "a(t)")
    lines!(ax_ae, t_b, e_orb[bound_mask], color = :darkorange,
           linewidth = 2, label = "e(t)")
    axislegend(ax_ae, position = :rt)
else
    text!(ax_ae, 0.5, 0.5, text = "Binary disrupted",
          align = (:center, :center), space = :relative)
end

ax_bnd = Axis(fig[2, 4],
              xlabel = "t", ylabel = "M_bound",
              title  = "Bound gas mass")
lines!(ax_bnd, t_diag, M_bound, color = :teal, linewidth = 1.5)
hlines!(ax_bnd, [1/3], color = :black,
        linestyle = :dash, linewidth = 1,
        label = "M_ejecta = 1/3")
axislegend(ax_bnd, position = :rt)

outpath = joinpath(FIGDIR, "sn30_nosink_summary.png")
save(outpath, fig, px_per_unit = 2)
@info "Saved figure" path=outpath

# ---------------------------------------------------------------------------
# Brief summary printout

println("\n=== SN30 no-sink summary ===")
@printf "  Final t        : %.2f  (%.2f P₀)\n"  t_diag[end]  t_diag[end]/(2π)
@printf "  Initial r_sep  : %.3f\n"               r_bh[1]
@printf "  Final   r_sep  : %.3f\n"               r_bh[end]
if any(bound_mask)
    i_last = findlast(bound_mask)
    @printf "  Binary final   : BOUND  (a = %.3f, e = %.3f)\n"  a_orb[i_last]  e_orb[i_last]
else
    println("  Binary final   : DISRUPTED")
end
@printf "  Final M_bound  : %.3f  (includes ambient floor × volume)\n" M_bound[end]
@printf "  E_gas drift    : %.3e (initial %.3e → final %.3e)\n" (E_gas[end]-E_gas[1]) E_gas[1] E_gas[end]
@printf "  Jz_gas drift   : %.3e\n" (Jz_gas[end]-Jz_gas[1])
