# Phase 5 plotting script: thermal bomb Sedov test and BH2 fallback accretion.
# Reproduces computations from test_supernova.jl.
# Saves figures to docs/figures/.

using BinarySupernova
using CairoMakie
using Statistics: mean

CairoMakie.activate!()

# ---------------------------------------------------------------------------
# Test 1: Sedov-Taylor from thermal bomb (32³, γ=5/3)
println("Running Sedov from thermal bomb (32³, γ=5/3, t=0.1)...")

ng      = BinarySupernova.NG
γ       = 5.0 / 3.0
ρ_bg    = 1.0
E_SN    = 1.0
cfl     = 0.4
t_end1  = 0.1
ρ_floor = 1e-10
P_floor = 1e-10

nx = ny = nz = 32
L  = 0.5; dx = 2L / nx; x0 = -L

U1 = zeros(5, nx + 2ng, ny + 2ng, nz + 2ng)
for k in ng+1:ng+nz, j in ng+1:ng+ny, i in ng+1:ng+nx
    U1[1, i, j, k] = ρ_bg
    U1[5, i, j, k] = 1e-5 / (γ - 1)
end

r_bomb = 0.1
thermal_bomb!(U1, nx, ny, nz, dx, dx, dx;
              E_SN = E_SN, r_bomb = r_bomb,
              x0 = x0, y0 = x0, z0 = x0)
fill_ghost_3d_outflow!(U1, nx, ny, nz)

let t = 0.0
    while t < t_end1
        dt = cfl_dt_3d(U1, nx, ny, nz, dx, dx, dx, γ, cfl)
        dt = min(dt, t_end1 - t)
        euler3d_step!(U1, nx, ny, nz, dx, dx, dx, dt, γ;
                      bc = :outflow, ρ_floor = ρ_floor, P_floor = P_floor)
        t += dt
    end
end

α_sedov = 0.4942
R_exact = (E_SN / (α_sedov * ρ_bg))^0.2 * t_end1^0.4

# Radial profile — all cells
rs1 = Float64[];  ρs1 = Float64[]
for k in ng+1:ng+nz, j in ng+1:ng+ny, i in ng+1:ng+nx
    xc = x0 + (i - ng - 0.5)*dx
    yc = x0 + (j - ng - 0.5)*dx
    zc = x0 + (k - ng - 0.5)*dx
    push!(rs1, sqrt(xc^2 + yc^2 + zc^2))
    push!(ρs1, U1[1, i, j, k])
end

# Estimate numerical shock position (max pressure)
R_num = let P_max = 0.0, R_tmp = 0.0
    for k in ng+1:ng+nz, j in ng+1:ng+ny, i in ng+1:ng+nx
        xc = x0 + (i - ng - 0.5)*dx
        yc = x0 + (j - ng - 0.5)*dx
        zc = x0 + (k - ng - 0.5)*dx
        ρv = U1[1, i, j, k]
        KE = 0.5 * (U1[2,i,j,k]^2 + U1[3,i,j,k]^2 + U1[4,i,j,k]^2) / max(ρv, 1e-30)
        P  = (γ - 1) * (U1[5, i, j, k] - KE)
        if P > P_max
            P_max = P; R_tmp = sqrt(xc^2 + yc^2 + zc^2)
        end
    end
    R_tmp
end
println("  R_exact = $(round(R_exact,digits=4)), R_num = $(round(R_num,digits=4)), err = $(round(abs(R_num-R_exact)/R_exact*100, digits=2))%")

# ---------------------------------------------------------------------------
# Test 2: BH2 fallback accretion (5 steps)
println("Running BH2 fallback accretion test (5 steps)...")

nx2 = ny2 = nz2 = 16
L2  = 0.5; dx2 = 2L2 / nx2; x02 = -L2

M_star     = 0.7; R_star    = 0.3
M_BH2_init = 0.2
E_SN2      = 0.5
ρ_floor2   = 1e-10
P_floor2   = 1e-8

U2 = zeros(5, nx2 + 2ng, ny2 + 2ng, nz2 + 2ng)
polytrope_ic_3d!(U2, nx2, ny2, nz2, dx2, dx2, dx2, γ;
                 M_star = M_star, R_star = R_star,
                 x0 = x02, y0 = x02, z0 = x02,
                 ρ_floor = ρ_floor2, P_floor = P_floor2)
fill_ghost_3d_outflow!(U2, nx2, ny2, nz2)

thermal_bomb!(U2, nx2, ny2, nz2, dx2, dx2, dx2;
              E_SN = E_SN2, r_bomb = R_star,
              x0 = x02, y0 = x02, z0 = x02)

r_floor_bh2 = 2.0 * dx2
bh2 = BlackHole([0.0, 0.0, 0.0], [0.0, 0.0, 0.0],
                 M_BH2_init, 0.01 * r_floor_bh2, 1e6, r_floor_bh2)
bhs2  = BlackHole[bh2]
F_gas2 = [zeros(3)]
M0_bh2 = bh2.mass

mass_history = [M0_bh2]
step_arr     = [0]

for step in 1:5
    dt = cfl_dt_3d(U2, nx2, ny2, nz2, dx2, dx2, dx2, γ, 0.4)
    dU = zeros(size(U2))
    Fx = zeros(size(U2)); Fy = zeros(size(U2)); Fz = zeros(size(U2))
    euler3d_rhs!(dU, Fx, Fy, Fz, U2, nx2, ny2, nz2, dx2, dx2, dx2, γ;
                 bc = :outflow, ρ_floor = ρ_floor2, P_floor = P_floor2)
    add_sink_sources!(dU, U2, nx2, ny2, nz2, dx2, dx2, dx2, bhs2, x02, x02, x02)
    @. U2 += dt * dU
    fill_ghost_3d_outflow!(U2, nx2, ny2, nz2)
    nbody_step!(bhs2, dt, F_gas2)
    accrete!(bh2, U2, nx2, ny2, nz2, dx2, dx2, dx2, x02, x02, x02, dt)
    push!(mass_history, bh2.mass)
    push!(step_arr, step)
end
println("  M_BH2: $(round(M0_bh2,digits=5)) → $(round(bh2.mass,digits=5))")

# ---------------------------------------------------------------------------
# Figures

fig = Figure(size=(900, 400))

# Left: Sedov radial density with shock position
ax1 = Axis(fig[1,1], xlabel="r", ylabel="Density ρ",
           title="Sedov from thermal bomb (t=0.1, 32³)")
scatter!(ax1, rs1, ρs1, color=(:royalblue, 0.3), markersize=2.5, label="Grid cells")
hlines!(ax1, [ρ_bg], color=:black, linewidth=1.5, linestyle=:dot, label="ρ_bg = 1")
vlines!(ax1, [R_exact], color=:crimson, linewidth=2.5,
        label="R_exact = $(round(R_exact,digits=3))")
vlines!(ax1, [R_num], color=:orange, linewidth=2, linestyle=:dash,
        label="R_num = $(round(R_num,digits=3))")
xlims!(ax1, 0, L)
axislegend(ax1, position=:rt)

# Right: BH2 mass vs step
ax2 = Axis(fig[1,2], xlabel="Timestep", ylabel="M_BH2 (code units)",
           title="BH2 fallback accretion (5 steps)")
lines!(ax2, step_arr, mass_history, color=:royalblue, linewidth=2)
scatter!(ax2, step_arr, mass_history, color=:royalblue, markersize=8)
hlines!(ax2, [M0_bh2], color=:gray, linestyle=:dash, linewidth=1.5,
        label="M_BH2_init = $(M0_bh2)")
axislegend(ax2, position=:rb)

save("docs/figures/phase5_combined.png", fig, px_per_unit=2)
println("Saved: docs/figures/phase5_combined.png")

# Save individual panels as separate figures
fig_sedov = Figure(size=(600, 420))
ax_s2 = Axis(fig_sedov[1,1], xlabel="r", ylabel="Density ρ",
             title="Sedov from thermal bomb (t=0.1, 32³)")
scatter!(ax_s2, rs1, ρs1, color=(:royalblue, 0.3), markersize=2.5, label="Grid cells")
hlines!(ax_s2, [ρ_bg], color=:black, linewidth=1.5, linestyle=:dot, label="ρ_bg = 1")
vlines!(ax_s2, [R_exact], color=:crimson, linewidth=2.5,
        label="R_exact = $(round(R_exact,digits=3))")
vlines!(ax_s2, [R_num], color=:orange, linewidth=2, linestyle=:dash,
        label="R_num = $(round(R_num,digits=3))")
xlims!(ax_s2, 0, L)
axislegend(ax_s2, position=:rt)
save("docs/figures/phase5_sedov_bomb.png", fig_sedov, px_per_unit=2)
println("Saved: docs/figures/phase5_sedov_bomb.png")

fig_bh2 = Figure(size=(600, 420))
ax_b2 = Axis(fig_bh2[1,1], xlabel="Timestep", ylabel="M_BH2 (code units)",
             title="BH2 fallback accretion (5 steps)")
lines!(ax_b2, step_arr, mass_history, color=:royalblue, linewidth=2)
scatter!(ax_b2, step_arr, mass_history, color=:royalblue, markersize=8)
hlines!(ax_b2, [M0_bh2], color=:gray, linestyle=:dash, linewidth=1.5,
        label="M_BH2_init = $(M0_bh2)")
axislegend(ax_b2, position=:rb)
save("docs/figures/phase5_bh2_accretion.png", fig_bh2, px_per_unit=2)
println("Saved: docs/figures/phase5_bh2_accretion.png")
