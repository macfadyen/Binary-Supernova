#!/usr/bin/env julia
# Plot Phase 9 figures:
#   1. KE/E_thermal vs. time during relaxation
#   2. Mid-plane density slice before and after relaxation (with and without BH)
#
# Run from the project root:
#   julia --project=. scripts/plot_phase9.jl
#
# Output: docs/figures/phase9_ke_relaxation.png
#         docs/figures/phase9_density_slice.png

using BinarySupernova
using CairoMakie

mkpath("docs/figures")

# ---------------------------------------------------------------------------
# Setup: 32³ polytrope, γ = 5/3

const NX    = 32
const γ     = 5/3
const L     = 1.0
const DX    = 2L / NX
const NG_   = BinarySupernova.NG
const NXTOT = NX + 2NG_

function make_state()
    U = zeros(Float64, 5, NXTOT, NXTOT, NXTOT)
    polytrope_ic_3d!(U, NX, NX, NX, DX, DX, DX, γ;
                     M_star  = 0.5,
                     R_star  = 0.4,
                     x0 = -L, y0 = -L, z0 = -L)
    return U
end

# Add a 5% velocity perturbation so there is initial KE to damp
function perturb!(U)
    ρ = view(U, 1, NG_+1:NG_+NX, NG_+1:NG_+NX, NG_+1:NG_+NX)
    U[2, NG_+1:NG_+NX, NG_+1:NG_+NX, NG_+1:NG_+NX] .+= 0.05 .* ρ
end

# ---------------------------------------------------------------------------
# Figure 1: KE/E_thermal vs. time — instrument relax_ic! with a callback

function relax_with_history(U; t_max = 0.5, t_damp = 0.1)
    # Manually run the relaxation loop to capture KE_ratio at each step.
    nx = ny = nz = NX
    dx = dy = dz = DX
    ng = NG_
    nxtot = nx + 2ng

    Fx = similar(U, 5, nxtot+1, nxtot,   nxtot  )
    Fy = similar(U, 5, nxtot,   nxtot+1, nxtot  )
    Fz = similar(U, 5, nxtot,   nxtot,   nxtot+1)
    dU = similar(U, 5, nxtot,   nxtot,   nxtot  )
    Un = similar(U)

    t_hist  = Float64[]
    ke_hist = Float64[]

    t = 0.0
    while t < t_max
        dt = min(cfl_dt_3d(U, nx, ny, nz, dx, dy, dz, γ, 0.4), t_max - t)
        Un .= U

        for (α, β) in ((1.0, 0.0), (0.25, 0.75), (2/3, 1/3))
            euler3d_rhs!(dU, Fx, Fy, Fz, U, nx, ny, nz, dx, dy, dz, γ;
                         bc=:outflow, ρ_floor=1e-10, P_floor=1e-10)
            relax_damping_source!(dU, U, nx, ny, nz, dx, dy, dz,
                                  -L, -L, -L, t_damp)
            @. U = β * Un + (1-β) * U + (1-β) * dt * dU
            apply_floors_3d!(U, nx, ny, nz, 1e-10, 1e-10, γ)
        end

        t += dt
        KE    = gas_kinetic_total(U, nx, ny, nz, dx, dy, dz)
        E_tot = gas_energy_total(U, nx, ny, nz, dx, dy, dz)
        E_th  = max(E_tot - KE, 0.0)
        push!(t_hist,  t)
        push!(ke_hist, E_th > 0.0 ? KE / E_th : 0.0)
    end
    return t_hist, ke_hist
end

@info "Running relaxation for KE/E_thermal history..."
U1 = make_state()
perturb!(U1)
t_hist, ke_hist = relax_with_history(U1)

fig1 = Figure(size = (700, 420))
ax = Axis(fig1[1,1],
    xlabel = "time  [code units]",
    ylabel = "KE / E_thermal",
    title  = "Phase 9 — Velocity-damping relaxation (32³, γ = 5/3)",
    yscale = log10,
    yminorticksvisible = true,
    yminorgridvisible  = true)
lines!(ax, t_hist, ke_hist, color = :royalblue, linewidth = 2, label = "KE / E_th")
hlines!(ax, [0.01], color = :crimson, linestyle = :dash, linewidth = 1.5,
        label = "1% threshold")
axislegend(ax, position = :lb)
save("docs/figures/phase9_ke_relaxation.png", fig1, px_per_unit = 2)
@info "Saved docs/figures/phase9_ke_relaxation.png"

# ---------------------------------------------------------------------------
# Figure 2: mid-plane density slice z = 0 before and after relaxation

@info "Running relaxation for density slice..."
U_before = make_state()
perturb!(U_before)
U_after  = deepcopy(U_before)

relax_ic!(U_after, NX, NX, NX, DX, DX, DX, γ;
          x0 = -L, y0 = -L, z0 = -L,
          t_damp = 0.1, t_max = 0.5,
          ρ_floor = 1e-10, P_floor = 1e-10,
          KE_tol = 0.01, verbose = true)

# Extract mid-plane slice (k = NX÷2)
k_mid = NX ÷ 2 + NG_
ρ_bf = Array(U_before[1, NG_+1:NG_+NX, NG_+1:NG_+NX, k_mid])
ρ_af = Array(U_after[ 1, NG_+1:NG_+NX, NG_+1:NG_+NX, k_mid])

xs = [-L + (i - 0.5) * DX for i in 1:NX]

fig2 = Figure(size = (900, 380))
ax1 = Axis(fig2[1,1], xlabel = "x", ylabel = "y",
           title = "Before relaxation", aspect = DataAspect())
ax2 = Axis(fig2[1,2], xlabel = "x", ylabel = "y",
           title = "After relaxation",  aspect = DataAspect())

ρ_lims = (0.0, max(maximum(ρ_bf), maximum(ρ_af)))
hm1 = heatmap!(ax1, xs, xs, ρ_bf; colorrange = ρ_lims, colormap = :plasma)
hm2 = heatmap!(ax2, xs, xs, ρ_af; colorrange = ρ_lims, colormap = :plasma)
Colorbar(fig2[1,3], hm2, label = "density  ρ  [code units]")
Label(fig2[0,:], "Phase 9 density slice (z = 0, 32³, γ = 5/3)", fontsize = 14)

save("docs/figures/phase9_density_slice.png", fig2, px_per_unit = 2)
@info "Saved docs/figures/phase9_density_slice.png"
