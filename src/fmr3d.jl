# 3D Fixed Mesh Refinement (FMR) — two-level hierarchy (coarse + one fine patch).
# Refinement ratio r ∈ {2, 4}.  Fine patch is centred on the domain.
#
# Algorithm (per coarse timestep dt_c):
#   1. Coarse SSP-RK3 step over the full domain.  Accumulate coarse fluxes
#      at C-F boundaries into flux register fr_c (SSP-RK3 stage weights).
#   2. Fine subcycles: r SSP-RK3 steps of dt_f = dt_c/r.  Before each RK
#      stage, fill fine ghost cells by 5th-order Lagrange prolongation from
#      the temporally interpolated coarse state.  Accumulate fine C-F fluxes
#      into flux register fr_f (averaged over r^2 fine faces per coarse face).
#   3. Restrict: conservative averaging of r^3 fine cells → coarse cell,
#      for all coarse cells inside the fine region.
#   4. Berger-Colella flux correction: adjust coarse cells just outside the
#      fine region using (fr_c − fr_f) to replace the coarse C-F fluxes with
#      the more accurate fine fluxes.
#
# References:
#   Berger & Colella (1989) — flux correction at coarse-fine interfaces.
#   Shu & Osher (1988) — SSP-RK3 (Butcher weights 1/6, 1/6, 2/3).

# ---------------------------------------------------------------------------
# Lagrange interpolation weights

"""
    _lag5_weights(ratio) -> Matrix{Float64}   # (ratio × 5)

5-point Lagrange weights for `ratio` fine sub-cells within one coarse cell.
Stencil nodes: coarse cells at offsets −2, −1, 0, +1, +2 from the centre.
Row s+1 gives weights for fine sub-cell s (0-indexed) with normalised
position t_s = (s + 0.5 − ratio/2) / ratio relative to the coarse centre.
"""
function _lag5_weights(ratio::Int)
    nodes = Float64[-2, -1, 0, 1, 2]
    w = zeros(ratio, 5)
    for s in 0:ratio-1
        t = (s + 0.5 - ratio / 2.0) / ratio    # ∈ (−0.5, +0.5)
        for j in 1:5
            wj = 1.0
            for m in 1:5
                m != j && (wj *= (t - nodes[m]) / (nodes[j] - nodes[m]))
            end
            w[s+1, j] = wj
        end
    end
    return w
end

# ---------------------------------------------------------------------------
# Data structures

"""
    FMRLevel3D

One level of the FMR hierarchy: conserved-variable array + grid metadata.
Active cells are indices NG+1:NG+nx in each direction.
"""
struct FMRLevel3D
    U  :: Array{Float64, 4}   # (5, nx+2NG, ny+2NG, nz+2NG)
    nx :: Int
    ny :: Int
    nz :: Int
    dx :: Float64
    dy :: Float64
    dz :: Float64
end

function FMRLevel3D(nx::Int, ny::Int, nz::Int, dx::Real, dy::Real, dz::Real)
    ng = NG
    U  = zeros(5, nx + 2ng, ny + 2ng, nz + 2ng)
    return FMRLevel3D(U, nx, ny, nz, Float64(dx), Float64(dy), Float64(dz))
end

"""
    FluxReg3D

Six face flux registers (x_lo/x_hi/y_lo/y_hi/z_lo/z_hi) for
Berger-Colella flux correction.  Each register stores the time-integrated
flux through the coarse-fine boundary face, summed with SSP-RK3 weights.
Dimensions: (5, transverse_dim1, transverse_dim2).
"""
struct FluxReg3D
    x_lo :: Array{Float64, 3}   # (5, n_jc, n_kc)
    x_hi :: Array{Float64, 3}
    y_lo :: Array{Float64, 3}   # (5, n_ic, n_kc)
    y_hi :: Array{Float64, 3}
    z_lo :: Array{Float64, 3}   # (5, n_ic, n_jc)
    z_hi :: Array{Float64, 3}
end

function FluxReg3D(n_ic::Int, n_jc::Int, n_kc::Int)
    return FluxReg3D(
        zeros(5, n_jc, n_kc), zeros(5, n_jc, n_kc),
        zeros(5, n_ic, n_kc), zeros(5, n_ic, n_kc),
        zeros(5, n_ic, n_jc), zeros(5, n_ic, n_jc),
    )
end

"""
    FMRGrid3D

Two-level FMR grid: one coarse level + one fine patch.

The fine patch covers coarse active cells `ci_lo:ci_hi` × `cj_lo:cj_hi` ×
`ck_lo:ck_hi`.  Fine dimensions: `nx_f = n_ic * ratio`, etc.

Constraint (for 5-point Lagrange stencil at fine ghost cells):
  ci_lo ≥ NG + 3, ci_hi ≤ nc.nx − NG − 2  (and similarly for j, k).
"""
mutable struct FMRGrid3D
    coarse  :: FMRLevel3D
    fine    :: FMRLevel3D
    ratio   :: Int
    # Fine region in coarse 1-indexed active cell coordinates
    ci_lo :: Int;  ci_hi :: Int
    cj_lo :: Int;  cj_hi :: Int
    ck_lo :: Int;  ck_hi :: Int
    γ       :: Float64
    bc      :: Symbol
    ρ_floor :: Float64
    P_floor :: Float64
    lag_w   :: Matrix{Float64}   # (ratio × 5) Lagrange weights
end

"""
    FMRGrid3D(coarse, ci_lo, ci_hi, cj_lo, cj_hi, ck_lo, ck_hi, ratio, γ;
              bc=:outflow, ρ_floor=0.0, P_floor=0.0)

Construct a two-level FMR grid.  Fine level is initialised by prolongation
from the coarse level.  The coarse level must already have valid data.
"""
function FMRGrid3D(coarse::FMRLevel3D,
                   ci_lo::Int, ci_hi::Int,
                   cj_lo::Int, cj_hi::Int,
                   ck_lo::Int, ck_hi::Int,
                   ratio::Int, γ::Real;
                   bc::Symbol    = :outflow,
                   ρ_floor::Real = 0.0,
                   P_floor::Real = 0.0)
    n_ic = ci_hi - ci_lo + 1
    n_jc = cj_hi - cj_lo + 1
    n_kc = ck_hi - ck_lo + 1
    fine = FMRLevel3D(n_ic * ratio, n_jc * ratio, n_kc * ratio,
                      coarse.dx / ratio, coarse.dy / ratio, coarse.dz / ratio)
    lag_w = _lag5_weights(ratio)
    G = FMRGrid3D(coarse, fine, ratio,
                  ci_lo, ci_hi, cj_lo, cj_hi, ck_lo, ck_hi,
                  Float64(γ), bc, Float64(ρ_floor), Float64(P_floor), lag_w)
    # Initialise fine active + ghost cells from coarse via prolongation.
    _prolong_all!(G.fine.U, G.coarse.U, G)
    return G
end

# ---------------------------------------------------------------------------
# Coarse-stencil helper for Lagrange prolongation

# Given a fine global array index `ig_f` along one axis (x, y, or z),
# the corresponding coarse active-cell origin `c_lo` (= ci_lo / cj_lo / ck_lo),
# the refinement ratio, and the ghost-cell count `ng`, return:
#   ic0  — nearest coarse active-cell index (1-indexed; may be ≤ 0 for ghost region)
#   sx   — sub-cell index within ic0 (0:ratio-1) matching this fine cell's position
#
# Both ic0 and sx are derived from the normalised position
#   ξ = c_lo − 0.5 + (ig_f − ng − 0.5) / ratio
# such that fine active cell 1 → ξ = c_lo − 0.5 + 0.5/ratio (sub-cell 0 of c_lo).
@inline function _coarse_stencil(ig_f::Int, c_lo::Int, ratio::Int, ng::Int)
    ξ   = c_lo - 0.5 + (ig_f - ng - 0.5) / ratio
    ic0 = round(Int, ξ)
    sx  = clamp(floor(Int, (ξ - ic0 + 0.5) * ratio), 0, ratio - 1)
    return ic0, sx
end

# ---------------------------------------------------------------------------
# Prolongation

# Tensor-product 5th-order Lagrange interpolation for one fine cell.
@inline function _prolong_cell!(U_f, U_c, G::FMRGrid3D, ig_f::Int, jf::Int, kf::Int)
    r   = G.ratio
    ng  = NG
    ic0, si = _coarse_stencil(ig_f, G.ci_lo, r, ng)
    jc0, sj = _coarse_stencil(jf,   G.cj_lo, r, ng)
    kc0, sk = _coarse_stencil(kf,   G.ck_lo, r, ng)
    @inbounds for q in 1:5
        val = 0.0
        for di in -2:2
            ig_c = ng + ic0 + di
            wi   = G.lag_w[si+1, di+3]
            for dj in -2:2
                jg_c = ng + jc0 + dj
                wij  = wi * G.lag_w[sj+1, dj+3]
                for dk in -2:2
                    kg_c = ng + kc0 + dk
                    val += wij * G.lag_w[sk+1, dk+3] * U_c[q, ig_c, jg_c, kg_c]
                end
            end
        end
        U_f[q, ig_f, jf, kf] = val
    end
end

"""
    _prolong_all!(U_f, U_c, G)

Fill ALL fine cells (active + ghost) from coarse via tensor-product Lagrange.
Used once at FMRGrid3D construction to initialise the fine level.
"""
function _prolong_all!(U_f, U_c, G::FMRGrid3D)
    ng = NG
    nxtot_f = G.fine.nx + 2ng
    nytot_f = G.fine.ny + 2ng
    nztot_f = G.fine.nz + 2ng
    for kf in 1:nztot_f, jf in 1:nytot_f, ig_f in 1:nxtot_f
        _prolong_cell!(U_f, U_c, G, ig_f, jf, kf)
    end
end

"""
    _prolong_fine_ghosts!(U_f, U_c, G)

Fill fine ghost cells from the coarse state `U_c` via tensor-product Lagrange.
Active fine cells are NOT overwritten.  Fill order: x faces → y faces
(active x) → z faces (active x, y), so edge/corner ghost cells are set once.
"""
function _prolong_fine_ghosts!(U_f, U_c, G::FMRGrid3D)
    ng  = NG
    nxf = G.fine.nx;  nyf = G.fine.ny;  nzf = G.fine.nz
    nxtot_f = nxf + 2ng;  nytot_f = nyf + 2ng;  nztot_f = nzf + 2ng

    # x faces — fill full y-z slab (including corners/edges in y and z).
    for kf in 1:nztot_f, jf in 1:nytot_f
        for ig_f in 1:ng
            _prolong_cell!(U_f, U_c, G, ig_f, jf, kf)
        end
        for ig_f in ng+nxf+1:nxtot_f
            _prolong_cell!(U_f, U_c, G, ig_f, jf, kf)
        end
    end
    # y faces — active x only (corners already filled above).
    for kf in 1:nztot_f, ig_f in ng+1:ng+nxf
        for jf in 1:ng
            _prolong_cell!(U_f, U_c, G, ig_f, jf, kf)
        end
        for jf in ng+nyf+1:nytot_f
            _prolong_cell!(U_f, U_c, G, ig_f, jf, kf)
        end
    end
    # z faces — active x and y only (edges already filled above).
    for jf in ng+1:ng+nyf, ig_f in ng+1:ng+nxf
        for kf in 1:ng
            _prolong_cell!(U_f, U_c, G, ig_f, jf, kf)
        end
        for kf in ng+nzf+1:nztot_f
            _prolong_cell!(U_f, U_c, G, ig_f, jf, kf)
        end
    end
end

# ---------------------------------------------------------------------------
# Restriction

"""
    _restrict!(U_c, U_f, G)

Conservative averaging: replace coarse cells in the fine region with the
mean of the r^3 fine cells that cover each coarse cell.
"""
function _restrict!(U_c, U_f, G::FMRGrid3D)
    r      = G.ratio
    ng     = NG
    inv_r3 = 1.0 / r^3

    @inbounds for m in 0:(G.ci_hi - G.ci_lo),
                  n in 0:(G.cj_hi - G.cj_lo),
                  p in 0:(G.ck_hi - G.ck_lo)
        ic_g = ng + G.ci_lo + m
        jc_g = ng + G.cj_lo + n
        kc_g = ng + G.ck_lo + p
        for q in 1:5
            val = 0.0
            for sx in 1:r, sy in 1:r, sz in 1:r
                val += U_f[q, ng + r*m + sx, ng + r*n + sy, ng + r*p + sz]
            end
            U_c[q, ic_g, jc_g, kc_g] = val * inv_r3
        end
    end
end

# ---------------------------------------------------------------------------
# Flux register accumulation

# Accumulate coarse C-F boundary fluxes into fr_c.
# `weight` is the SSP-RK3 Butcher weight for this stage; `dt` is dt_c.
function _accumulate_fr_coarse!(fr::FluxReg3D,
                                 Fx_c, Fy_c, Fz_c,
                                 weight::Float64, dt::Float64,
                                 G::FMRGrid3D)
    ng  = NG
    fac = weight * dt

    # x faces: flux at right face of coarse cell ic → Fx_c[q, ng+ic, ng+jc, ng+kc]
    @inbounds for kc in G.ck_lo:G.ck_hi, jc in G.cj_lo:G.cj_hi
        kc_idx = kc - G.ck_lo + 1
        jc_idx = jc - G.cj_lo + 1
        for q in 1:5
            fr.x_lo[q, jc_idx, kc_idx] += fac * Fx_c[q, ng+G.ci_lo-1, ng+jc, ng+kc]
            fr.x_hi[q, jc_idx, kc_idx] += fac * Fx_c[q, ng+G.ci_hi,   ng+jc, ng+kc]
        end
    end
    # y faces: flux at right face of coarse cell jc → Fy_c[q, ng+ic, ng+jc, ng+kc]
    @inbounds for kc in G.ck_lo:G.ck_hi, ic in G.ci_lo:G.ci_hi
        kc_idx = kc - G.ck_lo + 1
        ic_idx = ic - G.ci_lo + 1
        for q in 1:5
            fr.y_lo[q, ic_idx, kc_idx] += fac * Fy_c[q, ng+ic, ng+G.cj_lo-1, ng+kc]
            fr.y_hi[q, ic_idx, kc_idx] += fac * Fy_c[q, ng+ic, ng+G.cj_hi,   ng+kc]
        end
    end
    # z faces
    @inbounds for jc in G.cj_lo:G.cj_hi, ic in G.ci_lo:G.ci_hi
        jc_idx = jc - G.cj_lo + 1
        ic_idx = ic - G.ci_lo + 1
        for q in 1:5
            fr.z_lo[q, ic_idx, jc_idx] += fac * Fz_c[q, ng+ic, ng+jc, ng+G.ck_lo-1]
            fr.z_hi[q, ic_idx, jc_idx] += fac * Fz_c[q, ng+ic, ng+jc, ng+G.ck_hi  ]
        end
    end
end

# Accumulate fine C-F boundary fluxes into fr_f.
# `weight` is the SSP-RK3 Butcher weight; `dt` is dt_f = dt_c / ratio.
# Fine faces are averaged over r^2 fine cells per coarse face (factor 1/r^2).
function _accumulate_fr_fine!(fr::FluxReg3D,
                               Fx_f, Fy_f, Fz_f,
                               weight::Float64, dt::Float64,
                               G::FMRGrid3D)
    ng    = NG
    r     = G.ratio
    fac   = weight * dt / (r * r)   # 1/r^2 area averaging + stage weight × dt_f

    nxf = G.fine.nx;  nyf = G.fine.ny;  nzf = G.fine.nz

    # x faces: left face of active cell 1 = Fx_f[q, ng, ...]
    #          right face of active cell nxf = Fx_f[q, ng+nxf, ...]
    @inbounds for kf in 1:nzf, jf in 1:nyf
        kc_idx = (kf - 1) ÷ r + 1
        jc_idx = (jf - 1) ÷ r + 1
        for q in 1:5
            fr.x_lo[q, jc_idx, kc_idx] += fac * Fx_f[q, ng,      ng+jf, ng+kf]
            fr.x_hi[q, jc_idx, kc_idx] += fac * Fx_f[q, ng+nxf,  ng+jf, ng+kf]
        end
    end
    # y faces
    @inbounds for kf in 1:nzf, ia in 1:nxf
        kc_idx = (kf - 1) ÷ r + 1
        ic_idx = (ia - 1) ÷ r + 1
        for q in 1:5
            fr.y_lo[q, ic_idx, kc_idx] += fac * Fy_f[q, ng+ia, ng,      ng+kf]
            fr.y_hi[q, ic_idx, kc_idx] += fac * Fy_f[q, ng+ia, ng+nyf,  ng+kf]
        end
    end
    # z faces
    @inbounds for jf in 1:nyf, ia in 1:nxf
        jc_idx = (jf - 1) ÷ r + 1
        ic_idx = (ia - 1) ÷ r + 1
        for q in 1:5
            fr.z_lo[q, ic_idx, jc_idx] += fac * Fz_f[q, ng+ia, ng+jf, ng     ]
            fr.z_hi[q, ic_idx, jc_idx] += fac * Fz_f[q, ng+ia, ng+jf, ng+nzf ]
        end
    end
end

# ---------------------------------------------------------------------------
# Berger-Colella flux correction

# Apply the correction δU_c = (fr_c − fr_f) / dx_c to the coarse cells just
# outside the fine region.  Sign conventions:
#   x_lo face is the RIGHT face of coarse cell ci_lo−1  → (fr_c − fr_f)/dx_c
#   x_hi face is the LEFT  face of coarse cell ci_hi+1  → (fr_f − fr_c)/dx_c
# (and analogously for y/z faces).
function _bc_correct!(G::FMRGrid3D, fr_c::FluxReg3D, fr_f::FluxReg3D)
    nc   = G.coarse
    ng   = NG
    n_ic = G.ci_hi - G.ci_lo + 1
    n_jc = G.cj_hi - G.cj_lo + 1
    n_kc = G.ck_hi - G.ck_lo + 1
    inv_dx = 1.0 / nc.dx
    inv_dy = 1.0 / nc.dy
    inv_dz = 1.0 / nc.dz

    # x faces
    @inbounds for kc_idx in 1:n_kc, jc_idx in 1:n_jc
        jc = G.cj_lo + jc_idx - 1
        kc = G.ck_lo + kc_idx - 1
        for q in 1:5
            δ_lo = (fr_c.x_lo[q,jc_idx,kc_idx] - fr_f.x_lo[q,jc_idx,kc_idx]) * inv_dx
            δ_hi = (fr_f.x_hi[q,jc_idx,kc_idx] - fr_c.x_hi[q,jc_idx,kc_idx]) * inv_dx
            nc.U[q, ng+G.ci_lo-1, ng+jc, ng+kc] += δ_lo
            nc.U[q, ng+G.ci_hi+1, ng+jc, ng+kc] += δ_hi
        end
    end
    # y faces
    @inbounds for kc_idx in 1:n_kc, ic_idx in 1:n_ic
        ic = G.ci_lo + ic_idx - 1
        kc = G.ck_lo + kc_idx - 1
        for q in 1:5
            δ_lo = (fr_c.y_lo[q,ic_idx,kc_idx] - fr_f.y_lo[q,ic_idx,kc_idx]) * inv_dy
            δ_hi = (fr_f.y_hi[q,ic_idx,kc_idx] - fr_c.y_hi[q,ic_idx,kc_idx]) * inv_dy
            nc.U[q, ng+ic, ng+G.cj_lo-1, ng+kc] += δ_lo
            nc.U[q, ng+ic, ng+G.cj_hi+1, ng+kc] += δ_hi
        end
    end
    # z faces
    @inbounds for jc_idx in 1:n_jc, ic_idx in 1:n_ic
        ic = G.ci_lo + ic_idx - 1
        jc = G.cj_lo + jc_idx - 1
        for q in 1:5
            δ_lo = (fr_c.z_lo[q,ic_idx,jc_idx] - fr_f.z_lo[q,ic_idx,jc_idx]) * inv_dz
            δ_hi = (fr_f.z_hi[q,ic_idx,jc_idx] - fr_c.z_hi[q,ic_idx,jc_idx]) * inv_dz
            nc.U[q, ng+ic, ng+jc, ng+G.ck_lo-1] += δ_lo
            nc.U[q, ng+ic, ng+jc, ng+G.ck_hi+1] += δ_hi
        end
    end
end

# ---------------------------------------------------------------------------
# Fine-level RHS (ghost cells must be filled before calling)

function _fine_rhs!(dU_f, Fx_f, Fy_f, Fz_f, U_f, G::FMRGrid3D)
    nf = G.fine
    apply_floors_3d!(U_f, nf.nx, nf.ny, nf.nz, G.ρ_floor, G.P_floor, G.γ)
    fill!(Fx_f, 0.0);  fill!(Fy_f, 0.0);  fill!(Fz_f, 0.0)
    _weno_fluxes_x!(Fx_f, U_f, nf.nx, nf.ny, nf.nz, G.γ)
    _weno_fluxes_y!(Fy_f, U_f, nf.nx, nf.ny, nf.nz, G.γ)
    _weno_fluxes_z!(Fz_f, U_f, nf.nx, nf.ny, nf.nz, G.γ)
    _flux_divergence_3d!(dU_f, Fx_f, Fy_f, Fz_f, nf.nx, nf.ny, nf.nz,
                         nf.dx, nf.dy, nf.dz)
end

# ---------------------------------------------------------------------------
# Main FMR step

"""
    fmr3d_step!(G, dt_c)

Advance the two-level FMR grid by one coarse timestep `dt_c`.
Fine sub-steps: `G.ratio` steps of `dt_c / G.ratio` each.
"""
function fmr3d_step!(G::FMRGrid3D, dt_c::Float64)
    r     = G.ratio
    dt_f  = dt_c / r
    ng    = NG
    γ     = G.γ
    bc    = G.bc
    ρ_fl  = G.ρ_floor
    P_fl  = G.P_floor
    nc    = G.coarse
    nf    = G.fine

    nxtot_c = nc.nx + 2ng;  nytot_c = nc.ny + 2ng;  nztot_c = nc.nz + 2ng
    nxtot_f = nf.nx + 2ng;  nytot_f = nf.ny + 2ng;  nztot_f = nf.nz + 2ng

    # Pre-allocate scratch arrays.
    Fx_c = zeros(5, nxtot_c+1, nytot_c,   nztot_c  )
    Fy_c = zeros(5, nxtot_c,   nytot_c+1, nztot_c  )
    Fz_c = zeros(5, nxtot_c,   nytot_c,   nztot_c+1)
    dU_c = zeros(5, nxtot_c,   nytot_c,   nztot_c  )

    Fx_f = zeros(5, nxtot_f+1, nytot_f,   nztot_f  )
    Fy_f = zeros(5, nxtot_f,   nytot_f+1, nztot_f  )
    Fz_f = zeros(5, nxtot_f,   nytot_f,   nztot_f+1)
    dU_f = zeros(5, nxtot_f,   nytot_f,   nztot_f  )

    n_ic = G.ci_hi - G.ci_lo + 1
    n_jc = G.cj_hi - G.cj_lo + 1
    n_kc = G.ck_hi - G.ck_lo + 1
    fr_c = FluxReg3D(n_ic, n_jc, n_kc)
    fr_f = FluxReg3D(n_ic, n_jc, n_kc)

    # SSP-RK3 Butcher stage weights: u^{n+1} = u^n + dt*(1/6 L1 + 1/6 L2 + 2/3 L3)
    rk_w = (1.0/6.0, 1.0/6.0, 2.0/3.0)

    # -----------------------------------------------------------------------
    # Step 1: Coarse SSP-RK3, accumulating C-F boundary fluxes.

    U_c_n   = copy(nc.U)          # save U_c^n for temporal interpolation below
    Un_c    = copy(nc.U)

    # Stage 1
    euler3d_rhs!(dU_c, Fx_c, Fy_c, Fz_c, nc.U,
                 nc.nx, nc.ny, nc.nz, nc.dx, nc.dy, nc.dz, γ;
                 bc, ρ_floor=ρ_fl, P_floor=P_fl)
    _accumulate_fr_coarse!(fr_c, Fx_c, Fy_c, Fz_c, rk_w[1], dt_c, G)
    @. nc.U = Un_c + dt_c * dU_c
    apply_floors_3d!(nc.U, nc.nx, nc.ny, nc.nz, ρ_fl, P_fl, γ)

    # Stage 2
    euler3d_rhs!(dU_c, Fx_c, Fy_c, Fz_c, nc.U,
                 nc.nx, nc.ny, nc.nz, nc.dx, nc.dy, nc.dz, γ;
                 bc, ρ_floor=ρ_fl, P_floor=P_fl)
    _accumulate_fr_coarse!(fr_c, Fx_c, Fy_c, Fz_c, rk_w[2], dt_c, G)
    @. nc.U = 0.75 * Un_c + 0.25 * nc.U + 0.25 * dt_c * dU_c
    apply_floors_3d!(nc.U, nc.nx, nc.ny, nc.nz, ρ_fl, P_fl, γ)

    # Stage 3
    euler3d_rhs!(dU_c, Fx_c, Fy_c, Fz_c, nc.U,
                 nc.nx, nc.ny, nc.nz, nc.dx, nc.dy, nc.dz, γ;
                 bc, ρ_floor=ρ_fl, P_floor=P_fl)
    _accumulate_fr_coarse!(fr_c, Fx_c, Fy_c, Fz_c, rk_w[3], dt_c, G)
    @. nc.U = (1/3) * Un_c + (2/3) * nc.U + (2/3) * dt_c * dU_c
    apply_floors_3d!(nc.U, nc.nx, nc.ny, nc.nz, ρ_fl, P_fl, γ)

    U_c_np1 = copy(nc.U)          # U_c^{n+1} for temporal interpolation

    # -----------------------------------------------------------------------
    # Step 2: Fine subcycles.

    Un_f = similar(nf.U)

    for s_sub in 0:r-1
        # Temporal interpolation fraction (at start of this fine subcycle).
        α   = s_sub / r
        # Interpolated coarse state used for ghost fill throughout this subcycle.
        U_c_interp = (1.0 - α) * U_c_n + α * U_c_np1

        Un_f .= nf.U

        # SSP-RK3 for fine level — ghost fill before each stage.

        # Stage 1
        _prolong_fine_ghosts!(nf.U, U_c_interp, G)
        _fine_rhs!(dU_f, Fx_f, Fy_f, Fz_f, nf.U, G)
        _accumulate_fr_fine!(fr_f, Fx_f, Fy_f, Fz_f, rk_w[1], dt_f, G)
        @. nf.U = Un_f + dt_f * dU_f
        apply_floors_3d!(nf.U, nf.nx, nf.ny, nf.nz, ρ_fl, P_fl, γ)

        # Stage 2
        _prolong_fine_ghosts!(nf.U, U_c_interp, G)
        _fine_rhs!(dU_f, Fx_f, Fy_f, Fz_f, nf.U, G)
        _accumulate_fr_fine!(fr_f, Fx_f, Fy_f, Fz_f, rk_w[2], dt_f, G)
        @. nf.U = 0.75 * Un_f + 0.25 * nf.U + 0.25 * dt_f * dU_f
        apply_floors_3d!(nf.U, nf.nx, nf.ny, nf.nz, ρ_fl, P_fl, γ)

        # Stage 3
        _prolong_fine_ghosts!(nf.U, U_c_interp, G)
        _fine_rhs!(dU_f, Fx_f, Fy_f, Fz_f, nf.U, G)
        _accumulate_fr_fine!(fr_f, Fx_f, Fy_f, Fz_f, rk_w[3], dt_f, G)
        @. nf.U = (1/3) * Un_f + (2/3) * nf.U + (2/3) * dt_f * dU_f
        apply_floors_3d!(nf.U, nf.nx, nf.ny, nf.nz, ρ_fl, P_fl, γ)
    end

    # -----------------------------------------------------------------------
    # Step 3: Restrict fine → coarse in the fine region.
    _restrict!(nc.U, nf.U, G)

    # -----------------------------------------------------------------------
    # Step 4: Berger-Colella flux correction at C-F boundaries.
    _bc_correct!(G, fr_c, fr_f)

    # Apply floors to coarse after correction (BC correction can introduce
    # small negative values near the blast boundary).
    apply_floors_3d!(nc.U, nc.nx, nc.ny, nc.nz, ρ_fl, P_fl, γ)

    return nothing
end

# ---------------------------------------------------------------------------
# CFL timestep for the FMR grid

"""
    cfl_dt_fmr3d(G, cfl) -> dt_c

CFL-limited coarse timestep.  Both levels are checked:
  dt_c ≤ min(dt_coarse_cfl, ratio × dt_fine_cfl)
so that each fine sub-step also satisfies the CFL condition.
"""
function cfl_dt_fmr3d(G::FMRGrid3D, cfl::Real)
    dt_c = cfl_dt_3d(G.coarse.U, G.coarse.nx, G.coarse.ny, G.coarse.nz,
                     G.coarse.dx, G.coarse.dy, G.coarse.dz, G.γ, Float64(cfl))
    dt_f = cfl_dt_3d(G.fine.U, G.fine.nx, G.fine.ny, G.fine.nz,
                     G.fine.dx, G.fine.dy, G.fine.dz, G.γ, Float64(cfl))
    return min(dt_c, G.ratio * dt_f)
end
