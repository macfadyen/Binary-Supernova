# Phase 7 plotting script: FFT Poisson solver tests.
# Reproduces computations from test_self_gravity.jl.
# Saves figures to docs/figures/.

using BinarySupernova
using CairoMakie

CairoMakie.activate!()

println("Computing Phase 7 Poisson solver tests...")

# Helper: build uniform sphere on nx³ grid in unit cube [0,1]³
function sphere_density(nx, ny, nz, dx, R, ρ0=1.0)
    ρ = zeros(nx, ny, nz)
    xc = 0.5;  yc = 0.5;  zc = 0.5
    for k in 1:nz, j in 1:ny, i in 1:nx
        xi = (i - 0.5) * dx - xc
        yi = (j - 0.5) * dx - yc
        zi = (k - 0.5) * dx - zc
        sqrt(xi^2 + yi^2 + zi^2) < R && (ρ[i,j,k] = ρ0)
    end
    return ρ
end

# ---------------------------------------------------------------------------
# Main Poisson run: N=32
N32 = 32
dx32 = 1.0 / N32
R    = 0.15
ρ32  = sphere_density(N32, N32, N32, dx32, R)
Φ32  = solve_poisson_isolated(ρ32, N32, N32, N32, dx32, dx32, dx32)
M32  = sum(ρ32) * dx32^3

# Extract potential along x-axis through domain centre
x_c  = 0.5
j_c32 = round(Int, x_c / dx32 + 0.5)
k_c32 = j_c32

xs_axis32 = [(i - 0.5) * dx32 for i in 1:N32]
r_from_c32 = [abs((i - 0.5)*dx32 - x_c) for i in 1:N32]
Φ_axis32   = [Φ32[i, j_c32, k_c32] for i in 1:N32]

# Analytic: exterior = -M/r
Φ_analytic32 = [-M32 / max(r, dx32) for r in r_from_c32]

println("  N=32: M=$(round(M32, sigdigits=4))")

# ---------------------------------------------------------------------------
# Convergence: run at N=8,16,32,64

function ratio_error_N(N)
    dx  = 1.0 / N
    ρv  = sphere_density(N, N, N, dx, R)
    Φv  = solve_poisson_isolated(ρv, N, N, N, dx, dx, dx)
    x_c = 0.5
    j_c = round(Int, x_c / dx + 0.5)
    k_c = j_c
    function r_fc(i)
        xi = (i - 0.5) * dx - x_c
        yi = (j_c - 0.5) * dx - x_c
        zi = (k_c - 0.5) * dx - x_c
        return sqrt(xi^2 + yi^2 + zi^2)
    end
    i_A = round(Int, (x_c + 0.25) / dx + 0.5)
    i_B = round(Int, (x_c + 0.35) / dx + 0.5)
    rA = r_fc(i_A);  rB = r_fc(i_B)
    # clamp to valid range
    i_A = clamp(i_A, 1, N);  i_B = clamp(i_B, 1, N)
    rat_num = Φv[i_A, j_c, k_c] / Φv[i_B, j_c, k_c]
    return abs(rat_num - rB/rA) / (rB/rA)
end

Ns_conv = [8, 16, 32, 64]
println("  Computing convergence at N = $(Ns_conv)...")
errs_conv = [ratio_error_N(N) for N in Ns_conv]
println("  Errors: $(round.(errs_conv.*100, sigdigits=3))%")

# ---------------------------------------------------------------------------
# Figure

fig = Figure(size=(900, 420))

# Left panel: potential along x-axis
ax1 = Axis(fig[1,1], xlabel="x", ylabel="Φ(x, 0.5, 0.5)",
           title="Poisson potential along x-axis (N=32, R=0.15)")

# Only plot exterior points (r > R)
exterior_mask = r_from_c32 .> R .+ dx32
interior_mask = r_from_c32 .<= R + 2*dx32

# Plot full numerical result
lines!(ax1, xs_axis32, Φ_axis32, color=:royalblue, linewidth=2.5, label="FFT Poisson")

# Overlay analytic exterior only
xs_ext = xs_axis32[exterior_mask]
Φ_ext  = Φ_analytic32[exterior_mask]
lines!(ax1, xs_ext, Φ_ext, color=:black, linewidth=1.5, linestyle=:dash,
       label="Analytic −M/r")

# Mark sphere boundary
r_sphere_lo = x_c - R;  r_sphere_hi = x_c + R
vlines!(ax1, [r_sphere_lo, r_sphere_hi], color=:gray, linestyle=:dot, linewidth=1,
        label="Sphere boundary")
axislegend(ax1, position=:rb)

# Right panel: convergence
ax2 = Axis(fig[1,2], xlabel="N (cells per side)", ylabel="Ratio error (%)",
           title="Convergence of exterior 1/r scaling",
           xscale=log2, yscale=log10)
lines!(ax2, Ns_conv, errs_conv .* 100, color=:royalblue, linewidth=2)
scatter!(ax2, Ns_conv, errs_conv .* 100, color=:royalblue, markersize=8)
hlines!(ax2, [1.0], color=:gray, linestyle=:dash, linewidth=1.5, label="1% threshold")
ax2.xticks = (Ns_conv, string.(Ns_conv))
for (N, e) in zip(Ns_conv, errs_conv)
    text!(ax2, N, e*100*1.3, text="$(round(e*100,sigdigits=2))%", fontsize=10, align=(:center, :bottom))
end
axislegend(ax2, position=:rt)

save("docs/figures/phase7_poisson.png", fig, px_per_unit=2)
println("Saved: docs/figures/phase7_poisson.png")
