# Phase 3 plotting script: Kepler orbit conservation.
# Reproduces the test computation from test_nbody.jl.
# Saves figures to docs/figures/.

using BinarySupernova
using CairoMakie

CairoMakie.activate!()

println("Running Kepler orbit (10 orbits, SSP-RK3)...")

M1    = 0.5;  M2    = 0.5
a0    = 1.0
v_circ = 0.5
eps   = 1e-4

bh1 = BlackHole([+a0/2, 0.0, 0.0], [0.0,  +v_circ, 0.0], M1, eps, 1e6, 0.01)
bh2 = BlackHole([-a0/2, 0.0, 0.0], [0.0,  -v_circ, 0.0], M2, eps, 1e6, 0.01)
bhs = BlackHole[bh1, bh2]

function total_energy(bhs)
    KE = 0.5 * bhs[1].mass * sum(bhs[1].vel .^ 2) +
         0.5 * bhs[2].mass * sum(bhs[2].vel .^ 2)
    dx = bhs[1].pos .- bhs[2].pos
    r  = sqrt(sum(dx .^ 2) + bhs[1].eps^2)
    PE = -bhs[1].mass * bhs[2].mass / r
    return KE + PE
end

function total_Lz(bhs)
    L = 0.0
    for bh in bhs
        L += bh.mass * (bh.pos[1] * bh.vel[2] - bh.pos[2] * bh.vel[1])
    end
    return L
end

E0 = total_energy(bhs)
L0 = total_Lz(bhs)

T_orb  = 2π
t_end  = 10.0 * T_orb
dt     = T_orb / 500
F_gas  = [zeros(3), zeros(3)]

# Store trajectory and conservation errors
n_steps = round(Int, t_end / dt) + 1
t_arr   = Float64[]
x1_arr  = Float64[];  y1_arr = Float64[]
x2_arr  = Float64[];  y2_arr = Float64[]
E_err_arr = Float64[]
L_err_arr = Float64[]

push!(t_arr, 0.0)
push!(x1_arr, bh1.pos[1]);  push!(y1_arr, bh1.pos[2])
push!(x2_arr, bh2.pos[1]);  push!(y2_arr, bh2.pos[2])
push!(E_err_arr, 0.0)
push!(L_err_arr, 0.0)

let t = 0.0
    while t < t_end - 1e-12
        step = min(dt, t_end - t)
        nbody_step!(bhs, step, F_gas)
        t += step
        push!(t_arr, t)
        push!(x1_arr, bhs[1].pos[1]);  push!(y1_arr, bhs[1].pos[2])
        push!(x2_arr, bhs[2].pos[1]);  push!(y2_arr, bhs[2].pos[2])
        Ec = total_energy(bhs)
        Lc = total_Lz(bhs)
        push!(E_err_arr, abs(Ec - E0) / abs(E0))
        push!(L_err_arr, abs(Lc - L0) / abs(L0))
    end
end

println("Final E_err = $(round(E_err_arr[end]*100, sigdigits=3))%")
println("Final L_err = $(round(L_err_arr[end]*100, sigdigits=3))%")

# --- Figure ---
fig = Figure(size=(900, 400))

ax1 = Axis(fig[1,1], xlabel="x", ylabel="y",
           title="Kepler orbit (10 periods, M₁=M₂=0.5)",
           aspect=DataAspect())
lines!(ax1, x1_arr, y1_arr, color=:royalblue, linewidth=1.5, label="BH1")
lines!(ax1, x2_arr, y2_arr, color=:crimson,   linewidth=1.5, label="BH2")
scatter!(ax1, [x1_arr[1]], [y1_arr[1]], color=:royalblue, markersize=8)
scatter!(ax1, [x2_arr[1]], [y2_arr[1]], color=:crimson,   markersize=8)
axislegend(ax1, position=:rt)

ax2 = Axis(fig[1,2], xlabel="Time (code units)", ylabel="Relative error",
           title="Energy & angular momentum conservation",
           yscale=log10)
t_plot = t_arr[2:end]  # skip t=0 where error is 0
lines!(ax2, t_plot, max.(E_err_arr[2:end], 1e-16), color=:royalblue, linewidth=2,
       label="|ΔE/E₀|")
lines!(ax2, t_plot, max.(L_err_arr[2:end], 1e-16), color=:crimson, linewidth=2,
       linestyle=:dash, label="|ΔL/L₀|")
hlines!(ax2, [1e-3], color=:gray, linestyle=:dot, linewidth=1.5, label="0.1% threshold")
# Mark orbital periods
for k in 1:10
    vlines!(ax2, [k * T_orb], color=(:gray, 0.3), linewidth=0.5)
end
axislegend(ax2, position=:rb)
ylims!(ax2, 1e-10, 1e-2)

save("docs/figures/phase3_kepler_orbit.png", fig, px_per_unit=2)
println("Saved: docs/figures/phase3_kepler_orbit.png")
