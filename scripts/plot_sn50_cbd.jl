#!/usr/bin/env julia
# CBD-focused visualization for a single snapshot:
#   (1) zoomed xy log-ρ slice (inner ±zoom*a₀) with BH positions + orbit trace
#   (2) xz log-ρ slice (same zoom) — shows vertical disc scale height
#   (3) azimuthally-averaged surface density Σ(R) around binary COM
#   (4) azimuthally-averaged v_φ(R) with Kepler curve overlay
#
# Usage:
#   julia --project=. scripts/plot_sn50_cbd.jl --outdir PATH [--snap N] [--zoom 3.0] [--a0-rsun 2.0]

using BinarySupernova
using CairoMakie
using Printf

CairoMakie.activate!()

outdir = let v = "."
    for (i,a) in enumerate(ARGS); a == "--outdir" && i < length(ARGS) && (v = ARGS[i+1]); end; v
end
snap_arg = let v = -1
    for (i,a) in enumerate(ARGS); a == "--snap" && i < length(ARGS) && (v = parse(Int, ARGS[i+1])); end; v
end
zoom = let v = 3.0    # half-width in a₀ units
    for (i,a) in enumerate(ARGS); a == "--zoom" && i < length(ARGS) && (v = parse(Float64, ARGS[i+1])); end; v
end
a0_rsun = let v = 2.0
    for (i,a) in enumerate(ARGS); a == "--a0-rsun" && i < length(ARGS) && (v = parse(Float64, ARGS[i+1])); end; v
end

snap_idx = snap_arg
if snap_idx < 0
    snaps = filter(f -> startswith(f, "snap_t") && endswith(f, ".h5"), readdir(outdir))
    idxs = [parse(Int, match(r"snap_t(\d+)\.h5", s).captures[1]) for s in snaps]
    isempty(idxs) && error("No snapshots in $outdir")
    snap_idx = maximum(idxs)
end
snap_f = joinpath(outdir, @sprintf("snap_t%03d.h5", snap_idx))
@info "Reading" snap_f

U, nx, ny, nz, dx, dy, dz, t_s, γ = read_snapshot(snap_f)

traj_f = joinpath(outdir, "trajectory.h5")
t_traj, bh_data = read_trajectory(traj_f)
bh1 = bh_data[1]; bh2 = bh_data[2]
ti = argmin(abs.(t_traj .- t_s))

# Binary COM (mass-weighted)
M1 = bh1.mass[ti]; M2 = bh2.mass[ti]
x_com = (M1 * bh1.pos[ti,1] + M2 * bh2.pos[ti,1]) / (M1 + M2)
y_com = (M1 * bh1.pos[ti,2] + M2 * bh2.pos[ti,2]) / (M1 + M2)
z_com = (M1 * bh1.pos[ti,3] + M2 * bh2.pos[ti,3]) / (M1 + M2)
M_bin = M1 + M2

L  = nx * dx / 2
xs = collect(LinRange(-L + dx/2, L - dx/2, nx))
ys = copy(xs); zs = copy(xs)

# Midplane (z=z_com) slice — pick closest k
kz = clamp(round(Int, (z_com + L) / dz + 0.5), 1, nz)
ky = clamp(round(Int, (y_com + L) / dy + 0.5), 1, ny)

ρ_xy = U[1, :, :, kz]                    # (nx, ny)
vx_xy = U[2, :, :, kz] ./ max.(ρ_xy, 1e-30)
vy_xy = U[3, :, :, kz] ./ max.(ρ_xy, 1e-30)

ρ_xz = U[1, :, ky, :]                    # (nx, nz)

# ---------------------------------------------------------------------------
# Radial profile around binary COM (midplane slice)
#   Σ(R) from midplane ρ times a characteristic scale height dz (proxy, not
#   exact Σ = ∫ρ dz). We want *shape*, so normalize with <ρ> in the midplane.
#   Also compute azimuthally-averaged v_φ = (x-x_com)*vy - (y-y_com)*vx / R

nR   = 40
Rmax = zoom
Redges = LinRange(0.02, Rmax, nR+1)
Rcen   = 0.5 .* (Redges[1:end-1] .+ Redges[2:end])

ρ_bin  = zeros(nR)
vphi_bin = zeros(nR)
cnt    = zeros(Int, nR)

for j in 1:ny, i in 1:nx
    xrel = xs[i] - x_com
    yrel = ys[j] - y_com
    R    = sqrt(xrel^2 + yrel^2)
    R >= Redges[end] && continue
    R <= Redges[1]   && continue
    b = searchsortedlast(Redges, R)
    b = clamp(b, 1, nR)
    ρ_bin[b] += ρ_xy[i, j]
    if R > 1e-6
        vφ = (xrel * vy_xy[i,j] - yrel * vx_xy[i,j]) / R
        vphi_bin[b] += vφ
    end
    cnt[b] += 1
end
ρ_R  = [cnt[b] > 0 ? ρ_bin[b]/cnt[b] : 0.0 for b in 1:nR]
vφ_R = [cnt[b] > 0 ? vphi_bin[b]/cnt[b] : 0.0 for b in 1:nR]

# Σ proxy (midplane ρ × one cell thickness)
Σ_R = ρ_R .* dz

# Kepler curve v_K(R) = sqrt(M_bin / R)
vK_R = [sqrt(M_bin / max(R, 1e-6)) for R in Rcen]

# ---------------------------------------------------------------------------

fig = Figure(size = (1300, 950))

Label(fig[0, 1:2],
      @sprintf("%s — snap %d  t = %.3f (%.3f P₀)  r_sep = %.3f R☉",
               basename(outdir), snap_idx, t_s, t_s/(2π),
               a0_rsun * sqrt((bh1.pos[ti,1]-bh2.pos[ti,1])^2
                            + (bh1.pos[ti,2]-bh2.pos[ti,2])^2
                            + (bh1.pos[ti,3]-bh2.pos[ti,3])^2));
      fontsize = 15, font = :bold)

# --- Panel 1: xy zoom, midplane
ax1 = Axis(fig[1, 1], title = "xy midplane (z ≈ z_COM)  —  zoom ±$(zoom) a₀",
           xlabel = "x / a₀", ylabel = "y / a₀", aspect = DataAspect())
hm1 = heatmap!(ax1, xs, ys, log10.(max.(ρ_xy, 1e-12));
               colormap = :inferno, colorrange = (-4, 1))
xlims!(ax1, x_com - zoom, x_com + zoom)
ylims!(ax1, y_com - zoom, y_com + zoom)

# BH orbit traces up to this time
trace_mask = t_traj .<= t_s
lines!(ax1, bh1.pos[trace_mask, 1], bh1.pos[trace_mask, 2];
       color = :cyan, linewidth = 0.8)
lines!(ax1, bh2.pos[trace_mask, 1], bh2.pos[trace_mask, 2];
       color = :yellow, linewidth = 0.8)
scatter!(ax1, [bh1.pos[ti,1]], [bh1.pos[ti,2]];
         color = :cyan, markersize = 12, marker = :circle,
         strokecolor = :black, strokewidth = 1, label = "BH1")
scatter!(ax1, [bh2.pos[ti,1]], [bh2.pos[ti,2]];
         color = :yellow, markersize = 12, marker = :diamond,
         strokecolor = :black, strokewidth = 1, label = "BH2")
scatter!(ax1, [x_com], [y_com];
         color = :white, markersize = 8, marker = :star5,
         strokecolor = :black, strokewidth = 0.5, label = "COM")
axislegend(ax1, position = :rt)
Colorbar(fig[1, 3], hm1, label = "log₁₀ ρ")

# --- Panel 2: xz zoom, through y=y_COM
ax2 = Axis(fig[1, 2], title = "xz slice (y ≈ y_COM)  —  disc thickness",
           xlabel = "x / a₀", ylabel = "z / a₀", aspect = DataAspect())
hm2 = heatmap!(ax2, xs, zs, log10.(max.(ρ_xz, 1e-12));
               colormap = :inferno, colorrange = (-4, 1))
xlims!(ax2, x_com - zoom, x_com + zoom)
ylims!(ax2, z_com - zoom, z_com + zoom)
scatter!(ax2, [bh1.pos[ti,1]], [bh1.pos[ti,3]];
         color = :cyan, markersize = 12, marker = :circle,
         strokecolor = :black, strokewidth = 1)
scatter!(ax2, [bh2.pos[ti,1]], [bh2.pos[ti,3]];
         color = :yellow, markersize = 12, marker = :diamond,
         strokecolor = :black, strokewidth = 1)

# --- Panel 3: Σ(R) azimuthally averaged
ax3 = Axis(fig[2, 1], title = "Azimuthally-averaged midplane ρ(R)",
           xlabel = "R / a₀  (from binary COM)",
           ylabel = "⟨ρ⟩_φ  (code, log scale)",
           yscale = log10)
lines!(ax3, Rcen, max.(ρ_R, 1e-10);
       color = :royalblue, linewidth = 2)

r_sep_code = sqrt((bh1.pos[ti,1]-bh2.pos[ti,1])^2 +
                  (bh1.pos[ti,2]-bh2.pos[ti,2])^2 +
                  (bh1.pos[ti,3]-bh2.pos[ti,3])^2)
vlines!(ax3, [r_sep_code]; color = :crimson, linestyle = :dash, label = "r_sep")
vlines!(ax3, [2 * r_sep_code]; color = :gray, linestyle = :dot, label = "2 r_sep (CBD inner edge)")
axislegend(ax3, position = :rt)

# --- Panel 4: v_φ(R) with Kepler
ax4 = Axis(fig[2, 2], title = "Azimuthally-averaged v_φ(R)  vs  Kepler",
           xlabel = "R / a₀",
           ylabel = "⟨v_φ⟩  (code)")
lines!(ax4, Rcen, vφ_R;     color = :royalblue, linewidth = 2, label = "⟨v_φ⟩ gas")
lines!(ax4, Rcen, vK_R;     color = :crimson,   linewidth = 2, linestyle = :dash,
       label = "v_K = √(M_bin/R)")
vlines!(ax4, [r_sep_code]; color = :black, linestyle = :dot, alpha = 0.5)
axislegend(ax4, position = :rt)

figdir = joinpath(outdir, "figures")
mkpath(figdir)
outpath = joinpath(figdir, @sprintf("cbd_t%03d.png", snap_idx))
save(outpath, fig; px_per_unit = 2)
@info "Saved" outpath

# --- CLI summary
r_sep_Rsun = r_sep_code * a0_rsun
@printf "\nt = %.3f   r_sep = %.3f R☉   M_bin = %.3f code (%.1f M☉)\n" t_s r_sep_Rsun M_bin (M_bin*50)
@printf "COM: (%.3f, %.3f, %.3f) / a₀\n" x_com y_com z_com
# Integrated Σ(R) proxy bound gas
# Report rough "disc mass inside [2 r_sep, zoom] R" approximation
mask_disc = (Rcen .>= 2*r_sep_code) .& (Rcen .<= zoom)
if any(mask_disc)
    M_disc_approx = sum(ρ_R[mask_disc] .* (2π .* Rcen[mask_disc]) .* (Redges[2]-Redges[1]) * dz)
    @printf "M_gas (rough) in R ∈ [2 r_sep, %g]·a₀:  %.4f code  = %.3f M☉\n" zoom M_disc_approx (M_disc_approx*50)
end
