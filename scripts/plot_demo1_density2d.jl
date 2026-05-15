#!/usr/bin/env julia
# 2D equatorial density plots for demo1.
# Reads all snap_tNNN.h5 files from demo1/output/ and produces:
#   demo1/figures/density2d_tNNN.png  — one panel per snapshot
#   demo1/figures/density2d_grid.png  — multi-panel grid of all snapshots
#
# Usage: julia --project=. scripts/plot_demo1_density2d.jl

using BinarySupernova
using CairoMakie
using Printf
using Statistics: mean

CairoMakie.activate!()

const OUTDIR = "demo1/output"
const FIGDIR = "demo1/figures"
mkpath(FIGDIR)

# ---------------------------------------------------------------------------
# Find all snapshot files, sorted by time index

snap_files = sort(filter(f -> startswith(basename(f), "snap_t") && endswith(f, ".h5"),
                          readdir(OUTDIR, join=true)))

if isempty(snap_files)
    error("No snapshot files found in $OUTDIR — run scripts/run_demo1.jl first.")
end

@info "Found $(length(snap_files)) snapshots"

# ---------------------------------------------------------------------------
# Load BH trajectory for position overlays

traj_file = joinpath(OUTDIR, "trajectory.h5")
has_traj  = isfile(traj_file)
if has_traj
    t_traj, bh_data = read_trajectory(traj_file)
    bh1_traj = bh_data[1]
    bh2_traj = bh_data[2]
    @info "Trajectory loaded" N=length(t_traj)
else
    @warn "No trajectory.h5 found — BH positions will not be overlaid"
end

# ---------------------------------------------------------------------------
# Helper: find nearest trajectory index for time t

function traj_idx(t_query)
    has_traj || return nothing
    argmin(abs.(t_traj .- t_query))
end

# ---------------------------------------------------------------------------
# Helper: read snapshot and return z=0 slice + metadata

function load_equatorial_slice(filename)
    U, nx, ny, nz, dx, dy, dz, t, γ = read_snapshot(filename)
    # z = 0 equatorial plane: cell index kz in the nz active cells
    kz   = nz ÷ 2 + 1
    ρ    = U[1, :, :, kz]          # (nx, ny)
    # Physical coordinate axes (active cells, no ghosts)
    L_x  = nx * dx
    L_y  = ny * dy
    xs   = range(-L_x/2 + dx/2, L_x/2 - dx/2, length=nx)
    ys   = range(-L_y/2 + dy/2, L_y/2 - dy/2, length=ny)
    return ρ, xs, ys, t, nx, ny, dx
end

# ---------------------------------------------------------------------------
# Colour scale: fix a global range across all snapshots for comparability

# Sample a few snapshots to find global ρ range
log_lo, log_hi = let rmax = 0.0, rmin = Inf
    for f in snap_files[1:min(end, length(snap_files))]   # all snapshots
        U, nx, ny, nz, dx, dy, dz, t, γ = read_snapshot(f)
        kz = nz ÷ 2 + 1
        ρ  = U[1, :, :, kz]
        finite_pos = filter(x -> isfinite(x) && x > 0, vec(ρ))
        isempty(finite_pos) && continue
        rmax = max(rmax, maximum(finite_pos))
        rmin = min(rmin, minimum(finite_pos))
    end
    rmax > 0 || (rmax = 1.0)
    rmin < Inf || (rmin = 1e-6)
    floor(log10(max(rmin * 1e-4, 1e-12))), ceil(log10(rmax))
end
@info "Colour range" log_lo=log_lo log_hi=log_hi

# ---------------------------------------------------------------------------
# Individual panels

for snap_f in snap_files
    ρ, xs, ys, t, nx, ny, dx = load_equatorial_slice(snap_f)

    fig = Figure(size=(600, 520))
    ax  = Axis(fig[1, 1],
               xlabel = "x  (code units)",
               ylabel = "y  (code units)",
               title  = @sprintf("ρ(z=0)   t = %.2f  (P₀ = %.2f)",
                                 t, t / (2π)),
               aspect = DataAspect())

    ρ_plot = [isfinite(v) && v > 0 ? log10(v) : Float32(log_lo)
              for v in ρ']
    hm = heatmap!(ax, xs, ys, Float32.(ρ_plot);
                  colormap    = :inferno,
                  colorrange  = (Float32(log_lo), Float32(log_hi)))
    Colorbar(fig[1, 2], hm; label = "log₁₀ ρ  (code units)", width = 15)

    # BH positions
    if has_traj
        ki = traj_idx(t)
        x1, y1 = bh1_traj.pos[ki, 1], bh1_traj.pos[ki, 2]
        x2, y2 = bh2_traj.pos[ki, 1], bh2_traj.pos[ki, 2]
        scatter!(ax, [x1], [y1]; color=:cyan,   markersize=12,
                 marker=:circle,  label="BH1 (10 M☉)")
        scatter!(ax, [x2], [y2]; color=:yellow, markersize=12,
                 marker=:diamond, label="BH2 (10 M☉)")
        axislegend(ax; position=:lt, labelsize=10, framecolor=:white,
                   bgcolor=(:black, 0.5), labelcolor=:white)
    end

    # Scale bar: 0.5 code units
    xlo, xhi = minimum(xs), maximum(xs)
    ylo, yhi = minimum(ys), maximum(ys)
    sb_x = [xhi - 0.7, xhi - 0.2]
    sb_y = [ylo + 0.15, ylo + 0.15]
    lines!(ax, sb_x, sb_y; color=:white, linewidth=2.5)
    text!(ax, mean(sb_x), ylo + 0.22; text="0.5 a₀",
          color=:white, fontsize=10, align=(:center, :bottom))

    out = joinpath(FIGDIR, replace(basename(snap_f), ".h5" => "_density2d.png"))
    save(out, fig; px_per_unit=2)
    @info "Saved" file=basename(out)
end

# ---------------------------------------------------------------------------
# Multi-panel grid figure

@info "Building grid figure..."

# Use a subset if there are many snapshots: at most 16 panels
max_panels = 16
step_select = max(1, length(snap_files) ÷ max_panels)
selected = snap_files[1:step_select:end]
# Always include the first and last
selected = unique(vcat(snap_files[1:1], selected, snap_files[end:end]))

ncols = min(length(selected), 4)
nrows = ceil(Int, length(selected) / ncols)

fig_grid = Figure(size=(ncols*300, nrows*280 + 60))
Label(fig_grid[0, 1:ncols],
      "Demo 1 — Equatorial Density  (z = 0)   " *
      "M_BH1=10 M☉  M_star=20 M☉  M_BH2=10 M☉  v_kick=0";
      fontsize=13, font=:bold)

for (panel, snap_f) in enumerate(selected)
    row = (panel - 1) ÷ ncols + 1
    col = (panel - 1) % ncols + 1

    ρ, xs, ys, t, nx, ny, dx = load_equatorial_slice(snap_f)

    ax = Axis(fig_grid[row, col],
              title  = @sprintf("t=%.1f  (%.2f P₀)", t, t/(2π)),
              xlabel = col == 1 ? "x" : "",
              ylabel = row == nrows ? "y" : "",
              aspect = DataAspect(),
              titlesize = 11)
    hidedecorations!(ax; label=false, ticklabels=(col>1 && row<nrows), ticks=false)

    hm = heatmap!(ax, xs, ys, log10.(max.(ρ', 10.0^log_lo));
                  colormap   = :inferno,
                  colorrange = (log_lo, log_hi))

    if panel == length(selected)
        Colorbar(fig_grid[1:nrows, ncols+1], hm;
                 label="log₁₀ ρ", width=14, labelsize=11)
    end

    if has_traj
        ki = traj_idx(t)
        scatter!(ax, [bh1_traj.pos[ki,1]], [bh1_traj.pos[ki,2]];
                 color=:cyan,   markersize=8, marker=:circle)
        scatter!(ax, [bh2_traj.pos[ki,1]], [bh2_traj.pos[ki,2]];
                 color=:yellow, markersize=8, marker=:diamond)
    end
end

grid_out = joinpath(FIGDIR, "density2d_grid.png")
save(grid_out, fig_grid; px_per_unit=2)
@info "Saved grid figure" file=grid_out

println("\nDone. Figures in $FIGDIR/")
println("  Individual panels: density2d_tNNN_density2d.png")
println("  Grid:              density2d_grid.png")
