# Phase 6 plotting script: diagnostics and snapshot round-trip.
# Reproduces computations from test_diagnostics.jl.
# Saves figures to docs/figures/.

using BinarySupernova
using CairoMakie
using Statistics: mean

CairoMakie.activate!()

println("Computing Phase 6 diagnostics...")

ng = BinarySupernova.NG

# --- Test 1: Gas energy ---
nx = ny = nz = 8
dx  = 0.1
γ   = 5.0/3.0

U_e = zeros(5, nx+2ng, ny+2ng, nz+2ng)
for k in ng+1:ng+nz, j in ng+1:ng+ny, i in ng+1:ng+nx
    U_e[1,i,j,k] = 1.0
    U_e[5,i,j,k] = 2.0
end
E_diag  = gas_energy_total(U_e, nx, ny, nz, dx, dx, dx)
E_ref   = 2.0 * nx*ny*nz * dx^3
E_match = abs(E_diag - E_ref) / E_ref

# --- Test 2: Gas momentum = 0 ---
U_m = zeros(5, nx+2ng, ny+2ng, nz+2ng)
for k in ng+1:ng+nz, j in ng+1:ng+ny, i in ng+1:ng+nx
    U_m[1,i,j,k] = 1.0; U_m[5,i,j,k] = 0.01
end
P_mom = gas_momentum_total(U_m, nx, ny, nz, dx, dx, dx)
P_mag = maximum(abs.(P_mom))

# --- Test 3: BH kinetic energy and Lz ---
bh1 = BlackHole([+0.5, 0.0, 0.0], [0.0, 0.5, 0.0], 0.5, 0.01, 1e6, 0.01)
bh2_diag = BlackHole([-0.5, 0.0, 0.0], [0.0,-0.5, 0.0], 0.5, 0.01, 1e6, 0.01)
bhs_diag = BlackHole[bh1, bh2_diag]
KE_BH   = bh_kinetic_total(bhs_diag)
L_BH    = bh_angular_momentum_total(bhs_diag)

# --- Test 4: Bound mass ---
bh_bound = BlackHole([0.0, 0.0, 0.0], [0.0,0.0,0.0], 10.0, 0.01, 1e6, 0.01)
bhs_bound = BlackHole[bh_bound]
x0_b = -0.4
U_b  = zeros(5, nx+2ng, ny+2ng, nz+2ng)
for k in ng+1:ng+nz, j in ng+1:ng+ny, i in ng+1:ng+nx
    U_b[1, i, j, k] = 1.0
    U_b[5, i, j, k] = 1e-6 / (γ-1)
end
M_bound = bound_gas_mass(U_b, nx, ny, nz, dx, dx, dx, x0_b, x0_b, x0_b, bhs_bound, γ)
M_total = sum(U_b[1, ng+1:ng+nx, ng+1:ng+ny, ng+1:ng+nz]) * dx^3
bound_frac = M_bound / M_total

# --- Test 5: Snapshot round-trip ---
U_snap = rand(5, nx+2ng, ny+2ng, nz+2ng)
t_snap = 1.23
fname  = tempname() * ".h5"
write_snapshot(fname, U_snap, nx, ny, nz, dx, dx, dx, t_snap, γ)
U_snap2, nx2, ny2, nz2, dx2, dy2, dz2, t2, γ2 = read_snapshot(fname)
rm(fname)
snap_maxerr = maximum(abs.(U_snap2 .- U_snap[1:5, ng+1:ng+nx, ng+1:ng+ny, ng+1:ng+nz]))

println("  E_diag match: $(round(E_match*100, sigdigits=3))%")
println("  P_mag: $(P_mag)")
println("  KE_BH: $(KE_BH) (expected 0.125)")
println("  L_BH[3]: $(L_BH[3]) (expected 0.25)")
println("  Bound fraction: $(round(bound_frac, digits=4))")
println("  Snapshot max error: $(snap_maxerr)")

# --- Figure 1: Diagnostics summary bar chart ---
fig1 = Figure(size=(900, 400))

# Panel 1: Energy agreement
ax1 = Axis(fig1[1,1], title="Gas energy diagnostic",
           xlabel="", ylabel="Total energy")
barplot!(ax1, [1, 2], [E_diag, E_ref], color=[:royalblue, :gray],
         dodge=1:2)
text!(ax1, 0.5, E_ref*0.5, text="E_diag\n$(round(E_diag,sigdigits=4))", fontsize=11)
text!(ax1, 1.5, E_ref*0.5, text="E_ref\n$(round(E_ref,sigdigits=4))", fontsize=11)
ax1.xticks = ([1, 2], ["Computed", "Reference"])
ylims!(ax1, 0, E_ref * 1.2)

# Panel 2: BH kinetic energy
ax2 = Axis(fig1[1,2], title="BH kinetic energy",
           xlabel="", ylabel="KE_BH")
barplot!(ax2, [1], [KE_BH], color=[:royalblue])
hlines!(ax2, [0.125], color=:crimson, linestyle=:dash, linewidth=2,
        label="Expected = 0.125")
ax2.xticks = ([1], ["Computed"])
axislegend(ax2, position=:rt)
ylims!(ax2, 0, 0.2)

# Panel 3: BH angular momentum Lz
ax3 = Axis(fig1[1,3], title="BH angular momentum Lz",
           xlabel="", ylabel="L_z")
barplot!(ax3, [1], [L_BH[3]], color=[:forestgreen])
hlines!(ax3, [0.25], color=:crimson, linestyle=:dash, linewidth=2,
        label="Expected = 0.25")
ax3.xticks = ([1], ["Computed"])
axislegend(ax3, position=:rt)
ylims!(ax3, 0, 0.35)

# Panel 4: Bound mass fraction
ax4 = Axis(fig1[1,4], title="Bound mass fraction\n(deep BH potential)",
           xlabel="", ylabel="M_bound / M_total")
barplot!(ax4, [1], [bound_frac], color=[:orange])
hlines!(ax4, [1.0], color=:crimson, linestyle=:dash, linewidth=2, label="All bound")
ax4.xticks = ([1], ["Computed"])
axislegend(ax4, position=:rb)
ylims!(ax4, 0, 1.15)

save("docs/figures/phase6_diagnostics.png", fig1, px_per_unit=2)
println("Saved: docs/figures/phase6_diagnostics.png")

# --- Figure 2: Snapshot round-trip density slice ---
println("Generating snapshot round-trip figure...")

# Build a structured density field for visualization
U_vis = zeros(5, nx+2ng, ny+2ng, nz+2ng)
x0_vis = -0.4
for k in ng+1:ng+nz, j in ng+1:ng+ny, i in ng+1:ng+nx
    xc = x0_vis + (i - ng - 0.5)*dx
    yc = x0_vis + (j - ng - 0.5)*dx
    r  = sqrt(xc^2 + yc^2)
    ρ  = exp(-r^2 / 0.05)
    U_vis[1,i,j,k] = ρ + 0.01
    U_vis[5,i,j,k] = ρ * 0.5 + 0.001
end

fname2 = tempname() * ".h5"
write_snapshot(fname2, U_vis, nx, ny, nz, dx, dx, dx, 0.5, γ)
U_vis2, _, _, _, _, _, _, _, _ = read_snapshot(fname2)
rm(fname2)

# Extract z-mid slice of density
k_mid = nz÷2 + 1
ρ_orig  = [U_vis[1, ng+i, ng+j, ng+k_mid] for j in 1:ny, i in 1:nx]
ρ_readback = [U_vis2[1, i, j, k_mid] for j in 1:ny, i in 1:nx]
diff_max = maximum(abs.(ρ_orig .- ρ_readback))

x_coords = [x0_vis + (i - 0.5)*dx for i in 1:nx]
y_coords = [x0_vis + (j - 0.5)*dx for j in 1:ny]

fig2 = Figure(size=(900, 380))
ax_a = Axis(fig2[1,1], xlabel="x", ylabel="y", title="Snapshot: before write",
            aspect=DataAspect())
hm_a = heatmap!(ax_a, x_coords, y_coords, ρ_orig, colormap=:viridis)
Colorbar(fig2[1,1][1,2], hm_a, label="ρ")

ax_b = Axis(fig2[1,2], xlabel="x", ylabel="y", title="Snapshot: after HDF5 round-trip",
            aspect=DataAspect())
hm_b = heatmap!(ax_b, x_coords, y_coords, ρ_readback, colormap=:viridis)
Colorbar(fig2[1,2][1,2], hm_b, label="ρ")

Label(fig2[2, 1:2],
      "Max |ρ_before − ρ_after| = $(round(diff_max, sigdigits=3)) (should be 0 to machine precision)",
      fontsize=13)

save("docs/figures/phase6_snapshot.png", fig2, px_per_unit=2)
println("Saved: docs/figures/phase6_snapshot.png")
