#!/usr/bin/env julia
# Demo 1 plotting script.
# Reads demo1/output/ and generates all diagnostic figures.
#
# Usage:
#   julia --project=. scripts/plot_demo1.jl
#
# Figures saved to demo1/figures/:
#   demo1_density_slices.png   — z=0 density at 7 snapshots
#   demo1_orbital.png          — BH separation, semi-major axis, eccentricity
#   demo1_accretion.png        — Mdot and cumulative BH masses
#   demo1_torque.png           — gas torque on BH2
#   demo1_energy.png           — energy budget
#   demo1_bound_mass.png       — bound gas mass vs time

using BinarySupernova
using CairoMakie
using DelimitedFiles: readdlm
using Statistics: mean
using Printf

CairoMakie.activate!()

const OUTDIR  = "demo1/output"
const FIGDIR  = "demo1/figures"
const DT_SNAP = 1.0

mkpath(FIGDIR)

# ---------------------------------------------------------------------------
# Load trajectory (BH positions, velocities, masses)

@info "Reading trajectory..."
t_traj, bh_data = read_trajectory(joinpath(OUTDIR, "trajectory.h5"))
# bh_data is a Vector of NamedTuples: (mass, rsink, pos, vel)
# pos shape: (N_dumps, 3), vel: (N_dumps, 3)

bh1 = bh_data[1]
bh2 = bh_data[2]

# ---------------------------------------------------------------------------
# Load diagnostics CSV

@info "Reading diagnostics..."
D = readdlm(joinpath(OUTDIR, "diagnostics.csv"), ',', Float64; skipstart=1)
t_diag   = D[:, 1]
Mdot1    = D[:, 2]
Mdot2    = D[:, 3]
M_BH1    = D[:, 4]
M_BH2    = D[:, 5]
r_sep    = D[:, 6]
E_gas    = D[:, 7]
Jz_gas   = D[:, 8]
M_bound  = D[:, 9]
Fg1      = D[:, 10:12]   # gas force on BH1 [N×3]
Fg2      = D[:, 13:15]   # gas force on BH2 [N×3]

# ---------------------------------------------------------------------------
# Orbital element computation from trajectory
# G = 1, so specific orbital energy and angular momentum in code units.

function orbital_elements(r1, v1, r2, v2, M1, M2)
    dr   = r2 .- r1
    dv   = v2 .- v1
    r    = sqrt(sum(dr.^2))
    M    = M1 + M2
    # Specific energy (energy per reduced mass)
    eps  = 0.5 * sum(dv.^2) - M / r
    # Specific angular momentum vector
    hx = dr[2]*dv[3] - dr[3]*dv[2]
    hy = dr[3]*dv[1] - dr[1]*dv[3]
    hz = dr[1]*dv[2] - dr[2]*dv[1]
    h  = sqrt(hx^2 + hy^2 + hz^2)
    # Semi-major axis and eccentricity (only valid when bound: eps < 0)
    if eps < 0.0
        a = -M / (2 * eps)
        e = sqrt(max(0.0, 1.0 + 2 * eps * h^2 / M^2))
    else
        a = Inf    # unbound
        e = NaN
    end
    return a, e, eps
end

N_traj = length(t_traj)
a_orb  = zeros(N_traj)
e_orb  = zeros(N_traj)
eps_orb = zeros(N_traj)
r_bh   = zeros(N_traj)

for k in 1:N_traj
    r1 = bh1.pos[k, :]
    v1 = bh1.vel[k, :]
    r2 = bh2.pos[k, :]
    v2 = bh2.vel[k, :]
    r_bh[k] = sqrt(sum((r2 .- r1).^2))
    a_orb[k], e_orb[k], eps_orb[k] = orbital_elements(r1, v1, r2, v2,
                                                         bh1.mass[k], bh2.mass[k])
end

# Binary is bound when eps_orb < 0
bound_mask = eps_orb .< 0.0

# ---------------------------------------------------------------------------
# Torque on BH2 from gas about the instantaneous BH–BH centre of mass
# τz = (r_BH2 − r_cm) × F_gas2 |_z  = Δx Fgy2 − Δy Fgx2

N_diag = length(t_diag)
torque_z = zeros(N_diag)
# Trajectory has one extra entry at t=0; skip it (k+1 aligns with diagnostics step k).
for k in 1:N_diag
    k_traj = min(k + 1, N_traj)    # trajectory index 1 is t=0; index k+1 is after step k
    r1   = bh1.pos[k_traj, :]
    r2   = bh2.pos[k_traj, :]
    M1   = bh1.mass[k_traj]
    M2   = bh2.mass[k_traj]
    r_cm = (M1 .* r1 .+ M2 .* r2) ./ (M1 + M2)
    dr   = r2 .- r_cm              # BH2 displacement from CoM
    Fg   = Fg2[k, :]
    torque_z[k] = dr[1]*Fg[2] - dr[2]*Fg[1]
end

# BH–BH potential energy from trajectory
E_grav_bh = -bh1.mass .* bh2.mass ./ max.(r_bh, 1e-10)  # G = 1
E_kin_bh  = 0.5 .* bh1.mass .* sum(bh1.vel.^2, dims=2)[:] .+
             0.5 .* bh2.mass .* sum(bh2.vel.^2, dims=2)[:]

# ---------------------------------------------------------------------------
# Load snapshots for density slices

snap_times = [0, 1, 2, 5, 10, 15, 20]
snap_files = [joinpath(OUTDIR, @sprintf("snap_t%03d.h5", t)) for t in snap_times]

# Filter to files that actually exist
existing_snaps = [(t, f) for (t, f) in zip(snap_times, snap_files) if isfile(f)]
if isempty(existing_snaps)
    @warn "No snapshot files found in $OUTDIR — skipping density plots"
end

# ---------------------------------------------------------------------------
# ---- Figure 1: Density slices -------------------------------------------

@info "Plotting density slices..."

n_snaps = length(existing_snaps)
if n_snaps > 0
    ncols = min(n_snaps, 4)
    nrows = ceil(Int, n_snaps / ncols)
    fig_dens = Figure(size=(ncols*280, nrows*260))

    for (panel, (snap_t, snap_f)) in enumerate(existing_snaps)
        row = (panel - 1) ÷ ncols + 1
        col = (panel - 1) % ncols + 1
        ax  = Axis(fig_dens[row, col],
                   title = "t = $snap_t",
                   xlabel = col == 1 ? "x" : "",
                   ylabel = row == nrows ? "y" : "",
                   aspect = DataAspect())
        hidedecorations!(ax, label=false, ticklabels=(col > 1 || row < nrows), ticks=false)

        U_snap, nx, ny, nz, dx, dy, dz, t_s, γ = read_snapshot(snap_f)
        # z = 0 midplane: slice at k = nz÷2 + 1 (1-indexed active cells, no ghosts)
        kz = nz ÷ 2 + 1
        ρ_slice = U_snap[1, :, :, kz]    # shape (nx, ny)

        # Physical coordinate arrays
        L_dom  = nx * dx / 2
        xs_ax  = LinRange(-L_dom + dx/2, L_dom - dx/2, nx)

        hm = heatmap!(ax, xs_ax, xs_ax, log10.(max.(ρ_slice, 1e-12)'),
                      colormap = :inferno,
                      colorrange = (-8, 1))
        if panel == 1
            Colorbar(fig_dens[nrows+1, 1:ncols], hm,
                     label = "log₁₀ ρ", vertical = false)
        end

        # Overplot BH positions if trajectory available
        if snap_t < length(t_traj)
            idx = argmin(abs.(t_traj .- Float64(snap_t)))
            scatter!(ax, [bh1.pos[idx, 1]], [bh1.pos[idx, 2]],
                     color = :cyan, markersize = 8, marker = :circle)
            scatter!(ax, [bh2.pos[idx, 1]], [bh2.pos[idx, 2]],
                     color = :yellow, markersize = 8, marker = :diamond)
        end
    end

    Label(fig_dens[0, :], "Demo 1 — Density (z=0 slice)   [cyan=BH1, yellow=BH2]",
          fontsize = 14, font = :bold)
    save(joinpath(FIGDIR, "demo1_density_slices.png"), fig_dens, px_per_unit=2)
    @info "Saved demo1_density_slices.png"
end

# ---------------------------------------------------------------------------
# ---- Figure 2: Orbital evolution ----------------------------------------

@info "Plotting orbital evolution..."

fig_orb = Figure(size=(800, 600))
ax_r = Axis(fig_orb[1, 1],
            xlabel = "Time (code units)",
            ylabel = "BH separation r",
            title  = "BH–BH Separation")
lines!(ax_r, t_traj, r_bh, color=:royalblue, linewidth=1.5, label="r(t)")
hlines!(ax_r, [1.0], color=:gray, linestyle=:dash, linewidth=1, label="a₀ = 1")
axislegend(ax_r, position=:rt)

ax_ae = Axis(fig_orb[2, 1],
             xlabel = "Time (code units)",
             ylabel = "Semi-major axis a  /  eccentricity e")
# Only plot where binary is bound
t_bound = t_traj[bound_mask]
a_bound = a_orb[bound_mask]
e_bound = e_orb[bound_mask]
if !isempty(t_bound)
    lines!(ax_ae, t_bound, a_bound, color=:crimson, linewidth=2, label="a(t)")
    lines!(ax_ae, t_bound, e_bound, color=:darkorange, linewidth=2, label="e(t)")
    axislegend(ax_ae, position=:rt)
else
    text!(ax_ae, 0.5, 0.5, text="Binary disrupted (unbound at all times)",
          align=(:center, :center), space=:relative, fontsize=13)
end

Label(fig_orb[0, 1], "Demo 1 — Binary Orbital Evolution", fontsize=14, font=:bold)
save(joinpath(FIGDIR, "demo1_orbital.png"), fig_orb, px_per_unit=2)
@info "Saved demo1_orbital.png"

# ---------------------------------------------------------------------------
# ---- Figure 3: Accretion rates ------------------------------------------

@info "Plotting accretion rates..."

# Smooth Mdot with a running average to reduce per-step noise
window = max(1, length(t_diag) ÷ 200)
function smooth(v, w)
    n = length(v)
    out = similar(v)
    for i in 1:n
        lo = max(1, i - w÷2)
        hi = min(n, i + w÷2)
        out[i] = mean(v[lo:hi])
    end
    return out
end
Mdot1_sm = smooth(abs.(Mdot1), window)
Mdot2_sm = smooth(abs.(Mdot2), window)

fig_acc = Figure(size=(800, 600))
ax_md = Axis(fig_acc[1, 1],
             xlabel = "Time (code units)",
             ylabel = "Ṁ (code units)",
             yscale = log10,
             title  = "Accretion Rates")
lines!(ax_md, t_diag, max.(Mdot1_sm, 1e-14), color=:royalblue,
       linewidth=1.5, label="Ṁ₁ (BH1)")
lines!(ax_md, t_diag, max.(Mdot2_sm, 1e-14), color=:crimson,
       linewidth=1.5, label="Ṁ₂ (BH2)")
# t^{-5/3} reference line starting from peak Mdot2
idx_peak = argmax(Mdot2_sm)
if idx_peak < length(t_diag)
    t_pk = t_diag[idx_peak]
    M_pk = Mdot2_sm[idx_peak]
    t_ref = t_diag[idx_peak:end]
    ref   = M_pk .* (t_ref ./ t_pk).^(-5/3)
    lines!(ax_md, t_ref, max.(ref, 1e-14), color=:gray, linestyle=:dash,
           linewidth=1.5, label="t^{-5/3}")
end
axislegend(ax_md, position=:rt)

ax_mass = Axis(fig_acc[2, 1],
               xlabel = "Time (code units)",
               ylabel = "BH mass (code units)",
               title  = "Cumulative BH Masses")
lines!(ax_mass, t_diag, M_BH1, color=:royalblue, linewidth=2, label="M_BH1")
lines!(ax_mass, t_diag, M_BH2, color=:crimson,   linewidth=2, label="M_BH2")
hlines!(ax_mass, [0.5],  color=:royalblue, linestyle=:dot, linewidth=1)
hlines!(ax_mass, [0.1],  color=:crimson,   linestyle=:dot, linewidth=1)
axislegend(ax_mass, position=:rt)

Label(fig_acc[0, 1], "Demo 1 — Accretion History", fontsize=14, font=:bold)
save(joinpath(FIGDIR, "demo1_accretion.png"), fig_acc, px_per_unit=2)
@info "Saved demo1_accretion.png"

# ---------------------------------------------------------------------------
# ---- Figure 4: Gas torque on BH2 ----------------------------------------

@info "Plotting torque..."

torque_sm = smooth(torque_z, window)

fig_torq = Figure(size=(800, 400))
ax_torq = Axis(fig_torq[1, 1],
               xlabel = "Time (code units)",
               ylabel = "τ_z = (r₂ − r_cm) × F_gas₂  [z-component]",
               title  = "Gas Torque on BH2 (about CoM)")
lines!(ax_torq, t_diag, torque_sm,
       color=:darkorange, linewidth=1.5)
hlines!(ax_torq, [0.0], color=:black, linewidth=0.8, linestyle=:dash)
Label(fig_torq[0, 1], "Demo 1 — Gas Torque on BH2", fontsize=14, font=:bold)
save(joinpath(FIGDIR, "demo1_torque.png"), fig_torq, px_per_unit=2)
@info "Saved demo1_torque.png"

# ---------------------------------------------------------------------------
# ---- Figure 5: Energy budget --------------------------------------------

@info "Plotting energy budget..."

# Trajectory has N_traj = N_steps + 1 entries (initial t=0 plus one per step).
# Diagnostics has N_diag = N_steps entries (one per step, after first advance).
# Plot gas quantities on t_diag and BH quantities on t_traj separately.

fig_eng = Figure(size=(800, 500))
ax_eng = Axis(fig_eng[1, 1],
              xlabel = "Time (code units)",
              ylabel = "Energy (code units)",
              title  = "Energy Budget")
lines!(ax_eng, t_diag, E_gas,    color=:royalblue, linewidth=1.5, label="E_gas (grid)")
lines!(ax_eng, t_traj, E_kin_bh, color=:crimson,   linewidth=1.5, label="E_kin (BHs)")
lines!(ax_eng, t_traj, E_grav_bh, color=:purple,   linewidth=1.5, linestyle=:dash, label="E_grav (BH–BH)")
# Total = E_gas + E_kin_bh + E_grav_bh, aligned on trajectory times;
# interpolate E_gas onto t_traj using nearest-index matching.
E_gas_on_traj = [argmin(abs.(t_diag .- tt)) |> k -> (k <= N_diag ? E_gas[k] : E_gas[end])
                 for tt in t_traj]
E_total_traj  = E_gas_on_traj .+ E_kin_bh .+ E_grav_bh
lines!(ax_eng, t_traj, E_total_traj, color=:black, linewidth=2, label="E_total (approx)")
axislegend(ax_eng, position=:rt)
Label(fig_eng[0, 1], "Demo 1 — Energy Budget", fontsize=14, font=:bold)
save(joinpath(FIGDIR, "demo1_energy.png"), fig_eng, px_per_unit=2)
@info "Saved demo1_energy.png"

# ---------------------------------------------------------------------------
# ---- Figure 6: Bound mass -----------------------------------------------

@info "Plotting bound mass..."

fig_bnd = Figure(size=(800, 400))
ax_bnd = Axis(fig_bnd[1, 1],
              xlabel = "Time (code units)",
              ylabel = "M_bound (code units)",
              title  = "Bound Gas Mass (ε_tot < 0)")
lines!(ax_bnd, t_diag, M_bound, color=:teal, linewidth=1.5)
hlines!(ax_bnd, [0.0], color=:black, linewidth=0.8, linestyle=:dash)
Label(fig_bnd[0, 1], "Demo 1 — Bound Gas Mass", fontsize=14, font=:bold)
save(joinpath(FIGDIR, "demo1_bound_mass.png"), fig_bnd, px_per_unit=2)
@info "Saved demo1_bound_mass.png"

# ---------------------------------------------------------------------------
# ---- Summary print -------------------------------------------------------

println("\n=== Demo 1 Summary ===")
@printf "  Final time:       %.2f code units (%.2f P0)\n"  t_diag[end]  t_diag[end]/(2π)
@printf "  Final M_BH1:      %.4f  (initial 0.5000)\n"  M_BH1[end]
@printf "  Final M_BH2:      %.4f  (initial 0.1000)\n"  M_BH2[end]
@printf "  Final r_sep:      %.4f  (initial 1.0000)\n"  r_sep[end]
@printf "  Final E_gas:      %.4e\n"  E_gas[end]
@printf "  Final M_bound:    %.4e\n"  M_bound[end]
if any(bound_mask)
    @printf "  Binary:           BOUND at end (a = %.3f, e = %.3f)\n" a_orb[findlast(bound_mask)] e_orb[findlast(bound_mask)]
else
    @printf "  Binary:           DISRUPTED (never bound after explosion)\n"
end
println("  Figures saved to: $FIGDIR/")
