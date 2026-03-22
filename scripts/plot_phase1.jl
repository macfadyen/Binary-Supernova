# Phase 1 plotting script: Sod shock tube and Sedov-Taylor blast wave.
# Reproduces the test computations from test_sod3d.jl and test_sedov.jl.
# Saves figures to docs/figures/.

using BinarySupernova
using CairoMakie
using Statistics: mean

CairoMakie.activate!()

# ---------------------------------------------------------------------------
# Exact Sod solution (Toro §4.2) — same as test_sod3d.jl
function sod_exact(x::AbstractVector, t::Real; γ=1.4)
    ρL, vL, pL = 1.0,   0.0, 1.0
    ρR, vR, pR = 0.125, 0.0, 0.1
    csL = sqrt(γ * pL / ρL);  csR = sqrt(γ * pR / ρR)

    function fK_and_deriv(p, ρK, pK, csK)
        if p > pK
            AK = 2 / ((γ+1)*ρK);  BK = (γ-1)/(γ+1) * pK
            sq = sqrt(AK / (p + BK))
            fv = (p - pK) * sq;  dv = sq * (1 - (p - pK) / (2*(p + BK)))
        else
            fv = 2*csK/(γ-1) * ((p/pK)^((γ-1)/(2γ)) - 1)
            dv = (p/pK)^(-(γ+1)/(2γ)) / (ρK * csK)
        end
        return fv, dv
    end

    p = 0.5*(pL + pR)
    for _ in 1:100
        fL, dfL = fK_and_deriv(p, ρL, pL, csL)
        fR, dfR = fK_and_deriv(p, ρR, pR, csR)
        Δp = (fL + fR + vR - vL) / (dfL + dfR)
        p  = max(1e-12, p - Δp)
        abs(Δp) / p < 1e-12 && break
    end
    fL, _ = fK_and_deriv(p, ρL, pL, csL)
    fR, _ = fK_and_deriv(p, ρR, pR, csR)
    v_star = 0.5*(vL + vR) + 0.5*(fR - fL)

    ρ_starL = p > pL ? ρL*(p/pL + (γ-1)/(γ+1)) / ((γ-1)/(γ+1)*p/pL + 1) :
                       ρL*(p/pL)^(1/γ)
    ρ_starR = p > pR ? ρR*(p/pR + (γ-1)/(γ+1)) / ((γ-1)/(γ+1)*p/pR + 1) :
                       ρR*(p/pR)^(1/γ)

    p > pL ? (SL_head = SL_tail = vL - csL*sqrt((γ+1)/(2γ)*p/pL + (γ-1)/(2γ))) :
             (SL_head  = vL - csL; SL_tail = v_star - sqrt(γ*p/ρ_starL))
    p > pR ? (SR = vR + csR*sqrt((γ+1)/(2γ)*p/pR + (γ-1)/(2γ))) :
             (SR = vR + csR)

    ρ_out = similar(x);  v_out = similar(x);  p_out = similar(x)
    for (k, xk) in enumerate(x)
        ξ = t > 0 ? xk/t : sign(xk)*Inf
        if p > pL
            if ξ < SL_head;    ρ_out[k]=ρL;      v_out[k]=vL;     p_out[k]=pL
            elseif ξ < v_star; ρ_out[k]=ρ_starL; v_out[k]=v_star; p_out[k]=p
            elseif ξ < SR;     ρ_out[k]=ρ_starR; v_out[k]=v_star; p_out[k]=p
            else;              ρ_out[k]=ρR;      v_out[k]=vR;     p_out[k]=pR
            end
        else
            if ξ < SL_head;    ρ_out[k]=ρL;      v_out[k]=vL;     p_out[k]=pL
            elseif ξ < SL_tail
                cs_fan = (2*csL + (γ-1)*(vL-ξ)) / (γ+1)
                v_out[k] = ξ + cs_fan
                ρ_out[k] = ρL*(cs_fan/csL)^(2/(γ-1))
                p_out[k] = pL*(ρ_out[k]/ρL)^γ
            elseif ξ < v_star; ρ_out[k]=ρ_starL; v_out[k]=v_star; p_out[k]=p
            elseif ξ < SR;     ρ_out[k]=ρ_starR; v_out[k]=v_star; p_out[k]=p
            else;              ρ_out[k]=ρR;      v_out[k]=vR;     p_out[k]=pR
            end
        end
    end
    return ρ_out, v_out, p_out
end

# ---------------------------------------------------------------------------
# Run Sod shock tube in x-direction (256×4×4), γ=1.4, t=0.2
function run_sod(; n_normal=128, n_trans=4, γ=1.4, t_end=0.2, cfl=0.4)
    ng = BinarySupernova.NG
    nx, ny, nz = n_normal, n_trans, n_trans
    L  = 1.0
    dx = L / nx;  dy = L / ny;  dz = L / nz

    U = zeros(5, nx+2*ng, ny+2*ng, nz+2*ng)
    for k in 1:nz, j in 1:ny, i in 1:nx
        coord = (i - 0.5)*dx - 0.5
        ρ, v, pr = coord < 0.0 ? (1.0, 0.0, 1.0) : (0.125, 0.0, 0.1)
        ii, jj, kk = ng+i, ng+j, ng+k
        U[1, ii, jj, kk] = ρ
        U[2, ii, jj, kk] = ρ*v
        U[5, ii, jj, kk] = pr/(γ-1) + 0.5*ρ*v^2
    end

    t = 0.0
    while t < t_end
        dt = cfl_dt_3d(U, nx, ny, nz, dx, dy, dz, γ, cfl)
        dt = min(dt, t_end - t)
        euler3d_step!(U, nx, ny, nz, dx, dy, dz, dt, γ; bc=:outflow)
        t += dt
    end

    # Extract x-profile
    ρ_num = zeros(nx);  v_num = zeros(nx);  p_num = zeros(nx)
    for i in 1:nx
        vals = [U[:, ng+i, ng+j, ng+1] for j in 1:n_trans]
        row  = mean(vals)
        ρv   = row[1]
        mn   = row[2]
        E    = row[5]
        ρ_num[i] = ρv
        v_num[i] = mn / max(ρv, 1e-30)
        p_num[i] = (γ - 1) * (E - 0.5 * mn^2 / max(ρv, 1e-30))
    end

    xs = [-0.5 + (i - 0.5)*dx for i in 1:nx]
    ρ_ex, v_ex, p_ex = sod_exact(xs, t_end; γ=γ)
    return xs, ρ_num, v_num, p_num, ρ_ex, v_ex, p_ex
end

println("Running Sod shock tube (128×4×4, γ=1.4)...")
xs, ρ_num, v_num, p_num, ρ_ex, v_ex, p_ex = run_sod(n_normal=128)

fig1 = Figure(size=(900, 300))
ax1 = Axis(fig1[1,1], xlabel="x", ylabel="Density ρ", title="Sod: Density")
ax2 = Axis(fig1[1,2], xlabel="x", ylabel="Velocity v", title="Sod: Velocity")
ax3 = Axis(fig1[1,3], xlabel="x", ylabel="Pressure P", title="Sod: Pressure")

lines!(ax1, xs, ρ_ex, color=:black, linewidth=2, label="Exact")
scatter!(ax1, xs[1:4:end], ρ_num[1:4:end], color=:royalblue, markersize=4, label="Numerical")
axislegend(ax1, position=:rt)

lines!(ax2, xs, v_ex, color=:black, linewidth=2, label="Exact")
scatter!(ax2, xs[1:4:end], v_num[1:4:end], color=:royalblue, markersize=4, label="Numerical")
axislegend(ax2, position=:lt)

lines!(ax3, xs, p_ex, color=:black, linewidth=2, label="Exact")
scatter!(ax3, xs[1:4:end], p_num[1:4:end], color=:royalblue, markersize=4, label="Numerical")
axislegend(ax3, position=:rt)

save("docs/figures/phase1_sod_shock.png", fig1, px_per_unit=2)
println("Saved: docs/figures/phase1_sod_shock.png")

# ---------------------------------------------------------------------------
# Sedov-Taylor blast wave (32³, γ=5/3)
println("Running Sedov-Taylor blast wave (32³, γ=5/3)...")

γ_sedov  = 5/3
ng       = BinarySupernova.NG
nx = ny = nz = 32
L        = 1.0
dx       = 2*L / nx
t_end    = 0.1
E_blast  = 1.0
ρ_bg     = 1.0
P_floor  = 1e-5
cfl      = 0.4
α_sedov  = 0.4942
R_exact  = (E_blast / (α_sedov * ρ_bg))^(1/5) * t_end^(2/5)

U_sed = zeros(5, nx+2*ng, ny+2*ng, nz+2*ng)
sedov_ic_3d!(U_sed, nx, ny, nz, dx, dx, dx, γ_sedov;
             E_blast  = E_blast,
             ρ_bg     = ρ_bg,
             P_floor  = P_floor,
             x_offset = -L,
             y_offset = -L,
             z_offset = -L)

let t = 0.0
    while t < t_end
        dt = cfl_dt_3d(U_sed, nx, ny, nz, dx, dx, dx, γ_sedov, cfl)
        dt = min(dt, t_end - t)
        euler3d_step!(U_sed, nx, ny, nz, dx, dx, dx, dt, γ_sedov;
                      bc=:outflow, ρ_floor=ρ_bg*1e-6, P_floor=P_floor*0.1)
        t += dt
    end
end

# Radial profile: extract along x-axis (y=z=mid)
j_mid = ng + ny÷2 + 1;  k_mid = ng + nz÷2 + 1
rs_x = Float64[]
ρs_x = Float64[]
for i in 1:nx
    xc = -L + (i - 0.5)*dx
    push!(rs_x, abs(xc))
    push!(ρs_x, U_sed[1, ng+i, j_mid, k_mid])
end

# Collect all cells for radial density plot
rs_all = Float64[]
ρs_all = Float64[]
for k in ng+1:ng+nz, j in ng+1:ng+ny, i in ng+1:ng+nx
    xc = -L + (i - ng - 0.5)*dx
    yc = -L + (j - ng - 0.5)*dx
    zc = -L + (k - ng - 0.5)*dx
    r  = sqrt(xc^2 + yc^2 + zc^2)
    push!(rs_all, r)
    push!(ρs_all, U_sed[1, i, j, k])
end

# 2D density slice (z=mid plane)
k_slice = ng + nz÷2 + 1
ρ_slice = [U_sed[1, ng+i, ng+j, k_slice] for j in 1:ny, i in 1:nx]
x_coords = [-L + (i - 0.5)*dx for i in 1:nx]
y_coords = [-L + (j - 0.5)*dx for j in 1:ny]

fig2 = Figure(size=(900, 400))

# Left panel: 2D density slice
ax_2d = Axis(fig2[1,1], xlabel="x", ylabel="y",
             title="Sedov density slice (z=0, t=0.1)",
             aspect=DataAspect())
hm = heatmap!(ax_2d, x_coords, y_coords, log10.(max.(ρ_slice, 1e-6)),
              colormap=:inferno)
# Analytic shock circle
θ_circ = range(0, 2π, length=200)
lines!(ax_2d, R_exact.*cos.(θ_circ), R_exact.*sin.(θ_circ),
       color=:cyan, linewidth=2, linestyle=:dash, label="R_exact")
Colorbar(fig2[1,1][1,2], hm, label="log₁₀ρ")
axislegend(ax_2d, position=:rt)

# Right panel: radial density profile
ax_r = Axis(fig2[1,2], xlabel="r", ylabel="Density ρ",
            title="Sedov radial profile (t=0.1)")
# Scatter all cells
scatter!(ax_r, rs_all, ρs_all, color=:royalblue, markersize=2, alpha=0.3, label="Cells")
# Background density line
hlines!(ax_r, [ρ_bg], color=:black, linewidth=1.5, linestyle=:dot, label="ρ_bg=1")
# Post-shock density (Rankine-Hugoniot for γ=5/3)
ρ_post = ρ_bg * (γ_sedov+1)/(γ_sedov-1)  # = 4
hlines!(ax_r, [ρ_post], color=:orange, linewidth=1.5, linestyle=:dash, label="ρ_post=4")
# Vertical line at exact shock radius
vlines!(ax_r, [R_exact], color=:red, linewidth=2, label="R_exact=$(round(R_exact,digits=3))")
xlims!(ax_r, 0, L)
axislegend(ax_r, position=:rt)

save("docs/figures/phase1_sedov_taylor.png", fig2, px_per_unit=2)
println("Saved: docs/figures/phase1_sedov_taylor.png")
