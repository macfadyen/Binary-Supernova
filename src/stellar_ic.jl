# Lane-Emden polytrope solver and 3D stellar initial conditions.
#
# Lane-Emden ODE (polytrope index n, adiabatic index γ = 1 + 1/n):
#   d²θ/dξ² + (2/ξ) dθ/dξ = −θⁿ,   θ(0) = 1,  θ'(0) = 0
#
# Physical mapping (G = 1 in code units):
#   ρ(r) = ρ_c · θ(r / r_scale)ⁿ,    r < R_star
#   P(r) = K · ρ(r)^{1+1/n}
#   r_scale = R_star / ξ_1            (ξ_1 = first zero of θ)
#   ρ_c     = M_star / (4π r_scale³ ω_n)   where ω_n = −ξ_1² θ'(ξ_1) > 0
#   K       = 4π r_scale² ρ_c^{1−1/n} / (n+1)   (ensures hydrostatic equilibrium, G=1)
#
# Reference: Chandrasekhar (1939), Stellar Structure, §IV.

# ---------------------------------------------------------------------------
# ODE solver

"""
    lane_emden(n; dξ=1e-3) -> (ξs, θs, dθs)

Integrate the Lane-Emden ODE for polytrope index `n` using 4th-order Runge-Kutta.

Uses the Taylor expansion θ ≈ 1 − ξ²/6 + n ξ⁴/120 to start from ξ = dξ,
avoiding the 1/ξ singularity at the origin.  Integration stops when θ < 0.

Returns Float64 vectors with uniform step dξ.  The final element corresponds
to the first zero ξ_1 ≈ ξs[end] (slightly past zero).

Known first zeros (Chandrasekhar 1939):
  n = 1.0  → ξ_1 = π ≈ 3.14159
  n = 1.5  → ξ_1 ≈ 3.65375
  n = 3.0  → ξ_1 ≈ 6.89685
"""
function lane_emden(n::Real; dξ::Float64 = 1e-3)
    nf = Float64(n)
    # Taylor seed at ξ0 = dξ
    ξ0  = dξ
    θ0  = 1.0 - ξ0^2/6.0 + nf * ξ0^4 / 120.0
    dθ0 = -ξ0/3.0 + nf * ξ0^3 / 30.0

    ξs  = Float64[ξ0]
    θs  = Float64[θ0]
    dθs = Float64[dθ0]

    ξ = ξ0; θ = θ0; dθ = dθ0
    # d²θ/dξ² = f(ξ, θ, dθ) = −θⁿ − (2/ξ) dθ
    f(xi, th, dth) = -(max(th, 0.0))^nf - 2.0 * dth / xi

    while θ > 0.0
        k1θ = dξ * dθ;                k1d = dξ * f(ξ,        θ,        dθ       )
        k2θ = dξ * (dθ + k1d/2);      k2d = dξ * f(ξ+dξ/2,   θ+k1θ/2,  dθ+k1d/2)
        k3θ = dξ * (dθ + k2d/2);      k3d = dξ * f(ξ+dξ/2,   θ+k2θ/2,  dθ+k2d/2)
        k4θ = dξ * (dθ + k3d);        k4d = dξ * f(ξ+dξ,     θ+k3θ,    dθ+k3d  )

        θ  += (k1θ + 2k2θ + 2k3θ + k4θ) / 6
        dθ += (k1d + 2k2d + 2k3d + k4d) / 6
        ξ  += dξ

        push!(ξs, ξ); push!(θs, θ); push!(dθs, dθ)
    end

    return ξs, θs, dθs
end

# ---------------------------------------------------------------------------
# 3D IC builder

"""
    polytrope_ic_3d!(U, nx, ny, nz, dx, dy, dz, γ;
                     M_star, R_star, M_core=0.0,
                     x0=0, y0=0, z0=0,
                     x_center=0, y_center=0, z_center=0,
                     ρ_floor=1e-10, P_floor=1e-8) -> (ρ_c, r_scale, K, r_core)

Fill active cells of `U` with a Lane-Emden polytrope of index n = 1/(γ−1)
of total mass `M_star` and radius `R_star`.

If `M_core > 0`, hollow out the inner sphere of radius `r_core` (defined by
∫₀^r_core 4πρ r² dr = M_core, computed from the Lane-Emden mass profile)
by setting cells with r < r_core to floor values.  This is the design in
CLAUDE.md §6.1: the core mass becomes BH2 at t = 0 and must not appear as
gas on the grid, otherwise BH2 instantly accretes its own progenitor.

Cells outside R_star receive floor values (zero velocity).  Returns
(ρ_c, r_scale = R_star/ξ_1, K, r_core); r_core = 0 if M_core = 0.
"""
function polytrope_ic_3d!(U,
                           nx::Int, ny::Int, nz::Int,
                           dx::Real, dy::Real, dz::Real, γ::Real;
                           M_star   ::Real,
                           R_star   ::Real,
                           M_core   ::Real = 0.0,
                           x0       ::Real = 0.0,
                           y0       ::Real = 0.0,
                           z0       ::Real = 0.0,
                           x_center ::Real = 0.0,
                           y_center ::Real = 0.0,
                           z_center ::Real = 0.0,
                           ρ_floor  ::Real = 1e-10,
                           P_floor  ::Real = 1e-8)
    n = 1.0 / (γ - 1.0)     # polytrope index

    # --- Solve Lane-Emden ---
    ξs, θs, dθs = lane_emden(n)
    ξ_1  = ξs[end]
    dθ_1 = dθs[end]
    ω_n  = -ξ_1^2 * dθ_1    # > 0; total mass integral: M = 4π ρ_c r_0³ ω_n

    # --- Physical scales (G = 1) ---
    r_scale = R_star / ξ_1
    ρ_c     = M_star / (4π * r_scale^3 * ω_n)
    # K from hydrostatic equilibrium: r_0² = (n+1) K ρ_c^{1/n−1} / (4π G=1)
    K       = 4π * r_scale^2 * ρ_c^(1.0 - 1.0/n) / (n + 1.0)

    # --- θ interpolant (linear, uniform spacing dξ = ξs[2]-ξs[1]) ---
    dξ  = ξs[2] - ξs[1]
    N_ξ = length(ξs)
    function θ_at(ξq)
        ξq <= 0.0  && return 1.0
        ξq >= ξ_1  && return 0.0
        i_f = ξq / dξ
        i   = clamp(floor(Int, i_f), 1, N_ξ - 1)
        α   = i_f - i
        return (1.0 - α) * θs[i] + α * θs[i+1]
    end

    # --- Find r_core such that M(<r_core) = M_core ---
    # Lane-Emden mass interior to ξ:  M(<ξ) = 4π ρ_c r_scale³ × (-ξ² dθ/dξ)
    # so M(<ξ)/M_star = (-ξ² dθ/dξ) / ω_n.
    r_core = 0.0
    if M_core > 0.0
        M_core < M_star || error("polytrope_ic_3d!: M_core ($M_core) ≥ M_star ($M_star)")
        target = Float64(M_core) / Float64(M_star)
        ξ_core = ξ_1
        @inbounds for i in 1:N_ξ
            frac = -ξs[i]^2 * dθs[i] / ω_n
            if frac >= target
                # Linear interp in (ξ, frac) between i-1 and i for smooth root.
                if i > 1
                    f0 = -ξs[i-1]^2 * dθs[i-1] / ω_n
                    α  = (target - f0) / (frac - f0)
                    ξ_core = ξs[i-1] + α * (ξs[i] - ξs[i-1])
                else
                    ξ_core = ξs[i]
                end
                break
            end
        end
        r_core = ξ_core * r_scale
    end
    r_core2 = r_core^2

    # --- Fill grid ---
    ng = NG
    @inbounds for k in ng+1:ng+nz, j in ng+1:ng+ny, i in ng+1:ng+nx
        xc = x0 + (i - ng - 0.5) * Float64(dx) - Float64(x_center)
        yc = y0 + (j - ng - 0.5) * Float64(dy) - Float64(y_center)
        zc = z0 + (k - ng - 0.5) * Float64(dz) - Float64(z_center)
        r2 = xc^2 + yc^2 + zc^2
        r  = sqrt(r2)
        th = θ_at(r / r_scale)

        if th > 0.0 && r2 >= r_core2
            ρ = ρ_c * th^n
            P = K * ρ^(1.0 + 1.0/n)
        else
            ρ = Float64(ρ_floor)
            P = Float64(P_floor)
        end
        U[1, i, j, k] = ρ
        U[2, i, j, k] = 0.0
        U[3, i, j, k] = 0.0
        U[4, i, j, k] = 0.0
        U[5, i, j, k] = P / (γ - 1.0)
    end

    return ρ_c, r_scale, K, r_core
end

# ---------------------------------------------------------------------------
# Supernova thermal bomb (Phase 5)

"""
    thermal_bomb!(U, nx, ny, nz, dx, dy, dz;
                  E_SN, r_bomb, r_bomb_inner=0.0,
                  x0=0, y0=0, z0=0,
                  x_center=0, y_center=0, z_center=0,
                  bipolar_theta_deg=180.0) -> M_bomb

Deposit supernova energy `E_SN` as thermal energy, mass-weighted over all
active cells in the spherical shell `r_bomb_inner ≤ r < r_bomb` around
(`x_center`, `y_center`, `z_center`):

```
ΔE[cell] = E_SN × (ρ[cell] dV) / M_bomb
```

where M_bomb = ∫_{r_bomb_inner ≤ r < r_bomb} ρ dV.  Total energy deposited
equals E_SN exactly.  Setting `r_bomb_inner = r_sink(BH2)` excludes cells
that BH2 would instantly accrete on activation, so the bomb-driven blastwave
gets a chance to clear the sink region before the first sink sub-step.

`bipolar_theta_deg` restricts deposition to a pair of axial cones of half
opening angle `θ_j` around ±ẑ (the spin axis): cells are included only when
|cos θ| ≥ cos θ_j, i.e. within θ_j of either pole.  Default 180° is the
spherical bomb.  The physical motivation is magneto-rotational / jet-driven
explosions, which preferentially unbind low-specific-AM polar material
while leaving the high-AM equatorial belt bound for CBD feeding.
"""
function thermal_bomb!(U,
                        nx::Int, ny::Int, nz::Int,
                        dx::Real, dy::Real, dz::Real;
                        E_SN             ::Real,
                        r_bomb           ::Real,
                        r_bomb_inner     ::Real = 0.0,
                        x0               ::Real = 0.0,
                        y0               ::Real = 0.0,
                        z0               ::Real = 0.0,
                        x_center         ::Real = 0.0,
                        y_center         ::Real = 0.0,
                        z_center         ::Real = 0.0,
                        bipolar_theta_deg::Real = 180.0)
    ng = NG
    dV = Float64(dx) * Float64(dy) * Float64(dz)
    rin2  = Float64(r_bomb_inner)^2
    rout2 = Float64(r_bomb)^2
    μ_min = cos(Float64(bipolar_theta_deg) * π / 180.0)

    M_bomb = 0.0
    @inbounds for k in ng+1:ng+nz, j in ng+1:ng+ny, i in ng+1:ng+nx
        xc = x0 + (i - ng - 0.5) * dx - x_center
        yc = y0 + (j - ng - 0.5) * dy - y_center
        zc = z0 + (k - ng - 0.5) * dz - z_center
        r2 = xc^2 + yc^2 + zc^2
        in_shell = (rin2 <= r2 < rout2)
        in_cone  = (r2 <= 0.0) ? true : (abs(zc) >= sqrt(r2) * μ_min)
        (in_shell & in_cone) && (M_bomb += U[1, i, j, k] * dV)
    end
    M_bomb > 0.0 || error("thermal_bomb!: no gas in shell [$r_bomb_inner, $r_bomb) within cone θ_j=$(bipolar_theta_deg)°")

    fac = Float64(E_SN) / M_bomb
    @inbounds for k in ng+1:ng+nz, j in ng+1:ng+ny, i in ng+1:ng+nx
        xc = x0 + (i - ng - 0.5) * dx - x_center
        yc = y0 + (j - ng - 0.5) * dy - y_center
        zc = z0 + (k - ng - 0.5) * dz - z_center
        r2 = xc^2 + yc^2 + zc^2
        in_shell = (rin2 <= r2 < rout2)
        in_cone  = (r2 <= 0.0) ? true : (abs(zc) >= sqrt(r2) * μ_min)
        (in_shell & in_cone) && (U[5, i, j, k] += fac * U[1, i, j, k])
    end

    return M_bomb
end
