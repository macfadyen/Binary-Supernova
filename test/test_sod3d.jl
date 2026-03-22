# Sod shock tube tests for the 3D adiabatic Euler solver.
#
# Strategy: run 128×4×4 (or 4×128×4, 4×4×128) thin-slab grids with the shock
# oriented along each of the three coordinate axes.  Compare numerical density,
# velocity, and pressure profiles against the exact Riemann solution.
# All three directions should give identical L1 errors, confirming that the
# y- and z-sweep HLLC rotations are correct.
#
# Exact Riemann solver: sod_exact() (copied from HighMachCBD, same as Toro §4.2)
# Standard Sod ICs: ρL=1, vL=0, pL=1 | ρR=0.125, vR=0, pR=0.1, γ=1.4, t=0.2.

# Exact Sod solution (Toro §4.2) — copied verbatim from HighMachCBD.
function _sod_exact(x::AbstractVector, t::Real; γ=1.4)
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

    p > pL ? (SL_shock = vL - csL*sqrt((γ+1)/(2γ)*p/pL + (γ-1)/(2γ)); SL_head = SL_tail = SL_shock) :
             (SL_head  = vL - csL; SL_tail = v_star - sqrt(γ*p/ρ_starL); SL_shock = SL_head)
    p > pR ? (SR = vR + csR*sqrt((γ+1)/(2γ)*p/pR + (γ-1)/(2γ))) :
             (SR = vR + csR)

    ρ_out = similar(x);  v_out = similar(x);  p_out = similar(x)
    for (k, xk) in enumerate(x)
        ξ = t > 0 ? xk/t : sign(xk)*Inf
        if p > pL
            if ξ < SL_shock;      ρ_out[k]=ρL;      v_out[k]=vL;     p_out[k]=pL
            elseif ξ < v_star;    ρ_out[k]=ρ_starL; v_out[k]=v_star; p_out[k]=p
            elseif ξ < SR;        ρ_out[k]=ρ_starR; v_out[k]=v_star; p_out[k]=p
            else;                 ρ_out[k]=ρR;      v_out[k]=vR;     p_out[k]=pR
            end
        else
            if ξ < SL_head;       ρ_out[k]=ρL;      v_out[k]=vL;     p_out[k]=pL
            elseif ξ < SL_tail
                cs_fan = (2*csL + (γ-1)*(vL-ξ)) / (γ+1)
                v_out[k] = ξ + cs_fan
                ρ_out[k] = ρL*(cs_fan/csL)^(2/(γ-1))
                p_out[k] = pL*(ρ_out[k]/ρL)^γ
            elseif ξ < v_star;    ρ_out[k]=ρ_starL; v_out[k]=v_star; p_out[k]=p
            elseif ξ < SR;        ρ_out[k]=ρ_starR; v_out[k]=v_star; p_out[k]=p
            else;                 ρ_out[k]=ρR;      v_out[k]=vR;     p_out[k]=pR
            end
        end
    end
    return ρ_out, v_out, p_out
end

# Run one Sod shock tube in the given direction (:x, :y, :z).
# Returns (err_ρ, err_v, err_p) L1 errors vs exact solution.
function _run_sod_3d(direction::Symbol; n_normal=128, n_trans=4, γ=1.4,
                      t_end=0.2, cfl=0.4)
    ng = BinarySupernova.NG

    # Grid
    nx, ny, nz = direction == :x ? (n_normal, n_trans, n_trans) :
                 direction == :y ? (n_trans, n_normal, n_trans) :
                                   (n_trans, n_trans, n_normal)
    L  = 1.0
    dx = L / nx;  dy = L / ny;  dz = L / nz

    U = zeros(5, nx+2*ng, ny+2*ng, nz+2*ng)

    # Initial conditions: discontinuity at midplane of the normal direction.
    for k in 1:nz, j in 1:ny, i in 1:nx
        coord = direction == :x ? (i - 0.5)*dx - 0.5 :
                direction == :y ? (j - 0.5)*dy - 0.5 :
                                   (k - 0.5)*dz - 0.5
        ρ, v, pr = coord < 0.0 ? (1.0, 0.0, 1.0) : (0.125, 0.0, 0.1)
        ii, jj, kk = ng+i, ng+j, ng+k
        U[1, ii, jj, kk] = ρ
        U[2, ii, jj, kk] = direction == :x ? ρ*v : 0.0
        U[3, ii, jj, kk] = direction == :y ? ρ*v : 0.0
        U[4, ii, jj, kk] = direction == :z ? ρ*v : 0.0
        U[5, ii, jj, kk] = pr/(γ-1) + 0.5*ρ*v^2
    end

    # Time-march with CFL-limited steps.
    t = 0.0
    while t < t_end
        dt = cfl_dt_3d(U, nx, ny, nz, dx, dy, dz, γ, cfl)
        dt = min(dt, t_end - t)
        euler3d_step!(U, nx, ny, nz, dx, dy, dz, dt, γ; bc=:outflow)
        t += dt
    end

    # Extract 1D profile along normal direction.
    n_cells = direction == :x ? nx : direction == :y ? ny : nz
    d_cell  = direction == :x ? dx : direction == :y ? dy : dz
    ρ_num = zeros(n_cells);  v_num = zeros(n_cells);  p_num = zeros(n_cells)
    for i in 1:n_cells
        # Average over the two transverse cells (ny_trans=4, pick middle pair).
        vals = if direction == :x
            [U[:, ng+i, ng+j, ng+1] for j in 1:n_trans]
        elseif direction == :y
            [U[:, ng+1, ng+i, ng+j] for j in 1:n_trans]
        else
            [U[:, ng+1, ng+j, ng+i] for j in 1:n_trans]
        end
        row  = mean(vals)   # average over transverse cells
        ρv   = row[1]
        mn   = direction == :x ? row[2] : direction == :y ? row[3] : row[4]
        E    = row[5]
        ρ_num[i] = ρv
        v_num[i] = mn / max(ρv, 1e-30)
        p_num[i] = (γ - 1) * (E - 0.5 * mn^2 / max(ρv, 1e-30))
    end

    # Exact solution at cell centres along normal.
    xs_c   = [-0.5 + (i - 0.5)*d_cell for i in 1:n_cells]
    ρ_ex, v_ex, p_ex = _sod_exact(xs_c, t_end; γ=γ)

    # L1 errors (exclude 5-cell buffer at boundaries).
    buf = 5
    rng = buf+1 : n_cells-buf
    err_ρ = sum(abs.(ρ_num[rng] .- ρ_ex[rng])) * d_cell
    err_v = sum(abs.(v_num[rng] .- v_ex[rng])) * d_cell
    err_p = sum(abs.(p_num[rng] .- p_ex[rng])) * d_cell
    return err_ρ, err_v, err_p
end

@testset "Sod shock tube — 3D adiabatic (x-sweep)" begin
    err_ρ, err_v, err_p = _run_sod_3d(:x)
    @info "Sod x: ρ=$(round(err_ρ,sigdigits=3)) v=$(round(err_v,sigdigits=3)) p=$(round(err_p,sigdigits=3))"
    @test err_ρ < 5e-2
    @test err_v < 5e-2
    @test err_p < 5e-2
end

@testset "Sod shock tube — 3D adiabatic (y-sweep)" begin
    err_ρ, err_v, err_p = _run_sod_3d(:y)
    @info "Sod y: ρ=$(round(err_ρ,sigdigits=3)) v=$(round(err_v,sigdigits=3)) p=$(round(err_p,sigdigits=3))"
    @test err_ρ < 5e-2
    @test err_v < 5e-2
    @test err_p < 5e-2
end

@testset "Sod shock tube — 3D adiabatic (z-sweep)" begin
    err_ρ, err_v, err_p = _run_sod_3d(:z)
    @info "Sod z: ρ=$(round(err_ρ,sigdigits=3)) v=$(round(err_v,sigdigits=3)) p=$(round(err_p,sigdigits=3))"
    @test err_ρ < 5e-2
    @test err_v < 5e-2
    @test err_p < 5e-2
end

@testset "Sod sweep symmetry (x, y, z errors match)" begin
    ex = _run_sod_3d(:x)
    ey = _run_sod_3d(:y)
    ez = _run_sod_3d(:z)
    # All three directions should give identical errors (same 1D problem).
    @test abs(ex[1] - ey[1]) / max(ex[1], 1e-15) < 1e-10
    @test abs(ex[1] - ez[1]) / max(ex[1], 1e-15) < 1e-10
end
