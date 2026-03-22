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
                     M_star, R_star,
                     x0=0, y0=0, z0=0,
                     x_center=0, y_center=0, z_center=0,
                     ρ_floor=1e-10, P_floor=1e-8) -> (ρ_c, r_scale, K)

Fill active cells of `U` with a Lane-Emden polytrope of index n = 1/(γ−1).

The stellar centre is at (`x_center`, `y_center`, `z_center`) in physical
coordinates.  `x0, y0, z0` are the physical left edges of the active domain
(active cell (ng+1) has centre x0 + dx/2, matching euler3d.jl convention).
Cells outside R_star receive floor values (zero velocity).

Returns ρ_c, r_scale = R_star/ξ_1, and the polytropic constant K (all in
code units, G = 1).
"""
function polytrope_ic_3d!(U,
                           nx::Int, ny::Int, nz::Int,
                           dx::Real, dy::Real, dz::Real, γ::Real;
                           M_star   ::Real,
                           R_star   ::Real,
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

    # --- Fill grid ---
    ng = NG
    @inbounds for k in ng+1:ng+nz, j in ng+1:ng+ny, i in ng+1:ng+nx
        xc = x0 + (i - ng - 0.5) * Float64(dx) - Float64(x_center)
        yc = y0 + (j - ng - 0.5) * Float64(dy) - Float64(y_center)
        zc = z0 + (k - ng - 0.5) * Float64(dz) - Float64(z_center)
        r  = sqrt(xc^2 + yc^2 + zc^2)
        th = θ_at(r / r_scale)

        if th > 0.0
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

    return ρ_c, r_scale, K
end

# ---------------------------------------------------------------------------
# Supernova thermal bomb (Phase 5)

"""
    thermal_bomb!(U, nx, ny, nz, dx, dy, dz;
                  E_SN, r_bomb,
                  x0=0, y0=0, z0=0,
                  x_center=0, y_center=0, z_center=0) -> M_bomb

Deposit supernova energy `E_SN` as thermal energy, mass-weighted over all
active cells within radius `r_bomb` of (`x_center`, `y_center`, `z_center`):

```
ΔE[cell] = E_SN × (ρ[cell] dV) / M_bomb
```

where M_bomb = ∫_{r<r_bomb} ρ dV.  The sum of all ΔE equals E_SN exactly
(up to floating-point rounding).

Returns M_bomb (total gas mass inside the bomb sphere).
"""
function thermal_bomb!(U,
                        nx::Int, ny::Int, nz::Int,
                        dx::Real, dy::Real, dz::Real;
                        E_SN     ::Real,
                        r_bomb   ::Real,
                        x0       ::Real = 0.0,
                        y0       ::Real = 0.0,
                        z0       ::Real = 0.0,
                        x_center ::Real = 0.0,
                        y_center ::Real = 0.0,
                        z_center ::Real = 0.0)
    ng = NG
    dV = Float64(dx) * Float64(dy) * Float64(dz)

    # First pass: total gas mass inside r_bomb
    M_bomb = 0.0
    @inbounds for k in ng+1:ng+nz, j in ng+1:ng+ny, i in ng+1:ng+nx
        xc = x0 + (i - ng - 0.5) * dx - x_center
        yc = y0 + (j - ng - 0.5) * dy - y_center
        zc = z0 + (k - ng - 0.5) * dz - z_center
        sqrt(xc^2 + yc^2 + zc^2) < r_bomb && (M_bomb += U[1, i, j, k] * dV)
    end
    M_bomb > 0.0 || error("thermal_bomb!: no gas within r_bomb = $r_bomb")

    # Second pass: deposit energy proportional to local mass density.
    # ΔU[5] = (E_SN / M_bomb) * ρ  is an energy density [code units / volume].
    # Total energy deposited: Σ ΔU[5] * dV = (E_SN / M_bomb) * M_bomb = E_SN exactly.
    fac = Float64(E_SN) / M_bomb
    @inbounds for k in ng+1:ng+nz, j in ng+1:ng+ny, i in ng+1:ng+nx
        xc = x0 + (i - ng - 0.5) * dx - x_center
        yc = y0 + (j - ng - 0.5) * dy - y_center
        zc = z0 + (k - ng - 0.5) * dz - z_center
        sqrt(xc^2 + yc^2 + zc^2) < r_bomb && (U[5, i, j, k] += fac * U[1, i, j, k])
    end

    return M_bomb
end
