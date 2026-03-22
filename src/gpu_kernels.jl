# GPU kernel definitions for BinarySupernova.
#
# All kernels use KernelAbstractions.jl for CPU/CUDA portability.
# CPU backend: multithreaded via Julia threads (replaces Threads.@threads).
# CUDA backend: compiled to PTX via CUDA.jl for NVIDIA A100 execution.
#
# Workgroup sizes tuned for A100 (108 SMs × 2048 threads/SM):
#   3D kernels: (8, 8, 4) = 256 threads — fills warps, good occupancy at 32³ grids
#   2D kernels: (16, 16)  = 256 threads
#
# Usage convention:
#   All GPU-accelerated public functions detect the backend via
#   KA.get_backend(U) and dispatch to the appropriate kernel.
#   CPU Array  → KA.CPU()     (multithreaded, replaces Threads.@threads)
#   CuArray    → CUDABackend() (PTX via CUDA.jl)

import KernelAbstractions as KA
using Adapt

const _WGSIZE_3D = (8, 8, 4)   # 256 threads — A100 3D stencil default
const _WGSIZE_2D = (16, 16)    # 256 threads — A100 2D face default

# ---------------------------------------------------------------------------
# Floor application

@kernel function _apply_floors_kernel!(U, nx, ny, nz, ρ_floor, P_floor, γ)
    ii, jj, kk = @index(Global, NTuple)
    ng = NG
    i = ii + ng;  j = jj + ng;  k = kk + ng
    @inbounds begin
        ρ = U[1, i, j, k]
        if ρ < ρ_floor
            U[1, i, j, k] = ρ_floor
            U[2, i, j, k] = 0.0
            U[3, i, j, k] = 0.0
            U[4, i, j, k] = 0.0
            U[5, i, j, k] = P_floor / (γ - 1)
        else
            KE = 0.5 * (U[2,i,j,k]^2 + U[3,i,j,k]^2 + U[4,i,j,k]^2) / ρ
            P  = (γ - 1) * (U[5, i, j, k] - KE)
            if P < P_floor
                U[5, i, j, k] = P_floor / (γ - 1) + KE
            end
        end
    end
end

# ---------------------------------------------------------------------------
# WENO5 + HLLC flux kernels

@kernel function _weno_fluxes_x_kernel!(Fx, @Const(U), nx, ny, nz, γ)
    # ii: 1…nx+1 → interface index i = ng+ii-1 ∈ [ng, ng+nx]
    # jj: 1…ny   → active cell j = ng+jj
    # kk: 1…nz   → active cell k = ng+kk
    ii, jj, kk = @index(Global, NTuple)
    ng = NG
    i = ii + ng - 1
    j = jj + ng
    k = kk + ng
    @inbounds begin
        ρL  = weno5_left( U[1,i-2,j,k],U[1,i-1,j,k],U[1,i,j,k],U[1,i+1,j,k],U[1,i+2,j,k])
        ρR  = weno5_right(U[1,i-1,j,k],U[1,i,j,k],U[1,i+1,j,k],U[1,i+2,j,k],U[1,i+3,j,k])
        mxL = weno5_left( U[2,i-2,j,k],U[2,i-1,j,k],U[2,i,j,k],U[2,i+1,j,k],U[2,i+2,j,k])
        mxR = weno5_right(U[2,i-1,j,k],U[2,i,j,k],U[2,i+1,j,k],U[2,i+2,j,k],U[2,i+3,j,k])
        myL = weno5_left( U[3,i-2,j,k],U[3,i-1,j,k],U[3,i,j,k],U[3,i+1,j,k],U[3,i+2,j,k])
        myR = weno5_right(U[3,i-1,j,k],U[3,i,j,k],U[3,i+1,j,k],U[3,i+2,j,k],U[3,i+3,j,k])
        mzL = weno5_left( U[4,i-2,j,k],U[4,i-1,j,k],U[4,i,j,k],U[4,i+1,j,k],U[4,i+2,j,k])
        mzR = weno5_right(U[4,i-1,j,k],U[4,i,j,k],U[4,i+1,j,k],U[4,i+2,j,k],U[4,i+3,j,k])
        pi2 = _cell_pressure(U[1,i-2,j,k],U[2,i-2,j,k],U[3,i-2,j,k],U[4,i-2,j,k],U[5,i-2,j,k],γ)
        pi1 = _cell_pressure(U[1,i-1,j,k],U[2,i-1,j,k],U[3,i-1,j,k],U[4,i-1,j,k],U[5,i-1,j,k],γ)
        p0  = _cell_pressure(U[1,i  ,j,k],U[2,i  ,j,k],U[3,i  ,j,k],U[4,i  ,j,k],U[5,i  ,j,k],γ)
        pp1 = _cell_pressure(U[1,i+1,j,k],U[2,i+1,j,k],U[3,i+1,j,k],U[4,i+1,j,k],U[5,i+1,j,k],γ)
        pp2 = _cell_pressure(U[1,i+2,j,k],U[2,i+2,j,k],U[3,i+2,j,k],U[4,i+2,j,k],U[5,i+2,j,k],γ)
        pp3 = _cell_pressure(U[1,i+3,j,k],U[2,i+3,j,k],U[3,i+3,j,k],U[4,i+3,j,k],U[5,i+3,j,k],γ)
        P_maxL = max(pi2, pi1, p0, pp1, pp2)
        P_maxR = max(pi1, p0,  pp1, pp2, pp3)
        PL = clamp(weno5_left( pi2, pi1, p0, pp1, pp2), 0.0, P_maxL)
        PR = clamp(weno5_right(pi1, p0,  pp1, pp2, pp3), 0.0, P_maxR)
        EL = _EL_from_P(PL, ρL, mxL, myL, mzL, γ)
        ER = _EL_from_P(PR, ρR, mxR, myR, mzR, γ)
        Fx[1,i,j,k],Fx[2,i,j,k],Fx[3,i,j,k],Fx[4,i,j,k],Fx[5,i,j,k] =
            hllc_flux_x(ρL,mxL,myL,mzL,EL, ρR,mxR,myR,mzR,ER, γ)
    end
end

@kernel function _weno_fluxes_y_kernel!(Fy, @Const(U), nx, ny, nz, γ)
    # ii: 1…nx   → active cell i = ng+ii
    # jj: 1…ny+1 → interface j = ng+jj-1 ∈ [ng, ng+ny]
    # kk: 1…nz   → active cell k = ng+kk
    ii, jj, kk = @index(Global, NTuple)
    ng = NG
    i = ii + ng
    j = jj + ng - 1
    k = kk + ng
    @inbounds begin
        ρL  = weno5_left( U[1,i,j-2,k],U[1,i,j-1,k],U[1,i,j,k],U[1,i,j+1,k],U[1,i,j+2,k])
        ρR  = weno5_right(U[1,i,j-1,k],U[1,i,j,k],U[1,i,j+1,k],U[1,i,j+2,k],U[1,i,j+3,k])
        mxL = weno5_left( U[2,i,j-2,k],U[2,i,j-1,k],U[2,i,j,k],U[2,i,j+1,k],U[2,i,j+2,k])
        mxR = weno5_right(U[2,i,j-1,k],U[2,i,j,k],U[2,i,j+1,k],U[2,i,j+2,k],U[2,i,j+3,k])
        myL = weno5_left( U[3,i,j-2,k],U[3,i,j-1,k],U[3,i,j,k],U[3,i,j+1,k],U[3,i,j+2,k])
        myR = weno5_right(U[3,i,j-1,k],U[3,i,j,k],U[3,i,j+1,k],U[3,i,j+2,k],U[3,i,j+3,k])
        mzL = weno5_left( U[4,i,j-2,k],U[4,i,j-1,k],U[4,i,j,k],U[4,i,j+1,k],U[4,i,j+2,k])
        mzR = weno5_right(U[4,i,j-1,k],U[4,i,j,k],U[4,i,j+1,k],U[4,i,j+2,k],U[4,i,j+3,k])
        pi2 = _cell_pressure(U[1,i,j-2,k],U[2,i,j-2,k],U[3,i,j-2,k],U[4,i,j-2,k],U[5,i,j-2,k],γ)
        pi1 = _cell_pressure(U[1,i,j-1,k],U[2,i,j-1,k],U[3,i,j-1,k],U[4,i,j-1,k],U[5,i,j-1,k],γ)
        p0  = _cell_pressure(U[1,i,j  ,k],U[2,i,j  ,k],U[3,i,j  ,k],U[4,i,j  ,k],U[5,i,j  ,k],γ)
        pp1 = _cell_pressure(U[1,i,j+1,k],U[2,i,j+1,k],U[3,i,j+1,k],U[4,i,j+1,k],U[5,i,j+1,k],γ)
        pp2 = _cell_pressure(U[1,i,j+2,k],U[2,i,j+2,k],U[3,i,j+2,k],U[4,i,j+2,k],U[5,i,j+2,k],γ)
        pp3 = _cell_pressure(U[1,i,j+3,k],U[2,i,j+3,k],U[3,i,j+3,k],U[4,i,j+3,k],U[5,i,j+3,k],γ)
        P_maxL = max(pi2, pi1, p0, pp1, pp2)
        P_maxR = max(pi1, p0,  pp1, pp2, pp3)
        PL = clamp(weno5_left( pi2, pi1, p0, pp1, pp2), 0.0, P_maxL)
        PR = clamp(weno5_right(pi1, p0,  pp1, pp2, pp3), 0.0, P_maxR)
        EL = _EL_from_P(PL, ρL, mxL, myL, mzL, γ)
        ER = _EL_from_P(PR, ρR, mxR, myR, mzR, γ)
        Fy[1,i,j,k],Fy[2,i,j,k],Fy[3,i,j,k],Fy[4,i,j,k],Fy[5,i,j,k] =
            hllc_flux_y(ρL,mxL,myL,mzL,EL, ρR,mxR,myR,mzR,ER, γ)
    end
end

@kernel function _weno_fluxes_z_kernel!(Fz, @Const(U), nx, ny, nz, γ)
    # ii: 1…nx   → active cell i = ng+ii
    # jj: 1…ny   → active cell j = ng+jj
    # kk: 1…nz+1 → interface k = ng+kk-1 ∈ [ng, ng+nz]
    ii, jj, kk = @index(Global, NTuple)
    ng = NG
    i = ii + ng
    j = jj + ng
    k = kk + ng - 1
    @inbounds begin
        ρL  = weno5_left( U[1,i,j,k-2],U[1,i,j,k-1],U[1,i,j,k],U[1,i,j,k+1],U[1,i,j,k+2])
        ρR  = weno5_right(U[1,i,j,k-1],U[1,i,j,k],U[1,i,j,k+1],U[1,i,j,k+2],U[1,i,j,k+3])
        mxL = weno5_left( U[2,i,j,k-2],U[2,i,j,k-1],U[2,i,j,k],U[2,i,j,k+1],U[2,i,j,k+2])
        mxR = weno5_right(U[2,i,j,k-1],U[2,i,j,k],U[2,i,j,k+1],U[2,i,j,k+2],U[2,i,j,k+3])
        myL = weno5_left( U[3,i,j,k-2],U[3,i,j,k-1],U[3,i,j,k],U[3,i,j,k+1],U[3,i,j,k+2])
        myR = weno5_right(U[3,i,j,k-1],U[3,i,j,k],U[3,i,j,k+1],U[3,i,j,k+2],U[3,i,j,k+3])
        mzL = weno5_left( U[4,i,j,k-2],U[4,i,j,k-1],U[4,i,j,k],U[4,i,j,k+1],U[4,i,j,k+2])
        mzR = weno5_right(U[4,i,j,k-1],U[4,i,j,k],U[4,i,j,k+1],U[4,i,j,k+2],U[4,i,j,k+3])
        pi2 = _cell_pressure(U[1,i,j,k-2],U[2,i,j,k-2],U[3,i,j,k-2],U[4,i,j,k-2],U[5,i,j,k-2],γ)
        pi1 = _cell_pressure(U[1,i,j,k-1],U[2,i,j,k-1],U[3,i,j,k-1],U[4,i,j,k-1],U[5,i,j,k-1],γ)
        p0  = _cell_pressure(U[1,i,j,k  ],U[2,i,j,k  ],U[3,i,j,k  ],U[4,i,j,k  ],U[5,i,j,k  ],γ)
        pp1 = _cell_pressure(U[1,i,j,k+1],U[2,i,j,k+1],U[3,i,j,k+1],U[4,i,j,k+1],U[5,i,j,k+1],γ)
        pp2 = _cell_pressure(U[1,i,j,k+2],U[2,i,j,k+2],U[3,i,j,k+2],U[4,i,j,k+2],U[5,i,j,k+2],γ)
        pp3 = _cell_pressure(U[1,i,j,k+3],U[2,i,j,k+3],U[3,i,j,k+3],U[4,i,j,k+3],U[5,i,j,k+3],γ)
        P_maxL = max(pi2, pi1, p0, pp1, pp2)
        P_maxR = max(pi1, p0,  pp1, pp2, pp3)
        PL = clamp(weno5_left( pi2, pi1, p0, pp1, pp2), 0.0, P_maxL)
        PR = clamp(weno5_right(pi1, p0,  pp1, pp2, pp3), 0.0, P_maxR)
        EL = _EL_from_P(PL, ρL, mxL, myL, mzL, γ)
        ER = _EL_from_P(PR, ρR, mxR, myR, mzR, γ)
        Fz[1,i,j,k],Fz[2,i,j,k],Fz[3,i,j,k],Fz[4,i,j,k],Fz[5,i,j,k] =
            hllc_flux_z(ρL,mxL,myL,mzL,EL, ρR,mxR,myR,mzR,ER, γ)
    end
end

# ---------------------------------------------------------------------------
# Flux divergence

@kernel function _flux_divergence_kernel!(dU, @Const(Fx), @Const(Fy), @Const(Fz),
                                          nx, ny, nz, inv_dx, inv_dy, inv_dz)
    ii, jj, kk = @index(Global, NTuple)
    ng = NG
    i = ii + ng;  j = jj + ng;  k = kk + ng
    @inbounds for q in 1:5
        dU[q,i,j,k] = -(Fx[q,i,j,k] - Fx[q,i-1,j,k]) * inv_dx -
                        (Fy[q,i,j,k] - Fy[q,i,j-1,k]) * inv_dy -
                        (Fz[q,i,j,k] - Fz[q,i,j,k-1]) * inv_dz
    end
end

# ---------------------------------------------------------------------------
# CFL wave speed (per cell; caller reduces with maximum)

@kernel function _cfl_speeds_kernel!(speeds, @Const(U), nx, ny, nz,
                                     γ, inv_dx, inv_dy, inv_dz)
    ii, jj, kk = @index(Global, NTuple)
    ng = NG
    i = ii + ng;  j = jj + ng;  k = kk + ng
    @inbounds begin
        ρ  = max(U[1, i, j, k], 1e-30)
        vx = U[2, i, j, k] / ρ
        vy = U[3, i, j, k] / ρ
        vz = U[4, i, j, k] / ρ
        KE = 0.5 * (vx^2 + vy^2 + vz^2)
        P  = max((γ - 1) * (U[5, i, j, k] / ρ - KE) * ρ, 0.0)
        cs = sqrt(γ * P / ρ)
        speeds[ii, jj, kk] = max((abs(vx) + cs) * inv_dx,
                                  (abs(vy) + cs) * inv_dy,
                                  (abs(vz) + cs) * inv_dz)
    end
end

# ---------------------------------------------------------------------------
# BH gravitational source term
#
# bh_px/py/pz/mass/eps are NTuples (length = number of BHs), passed by value.
# This avoids device-pointer issues with CPU-side BlackHole structs.

@kernel function _bh_gravity_kernel!(dU, @Const(U), nx, ny, nz,
                                     dx, dy, dz, x0, y0, z0,
                                     bh_px, bh_py, bh_pz, bh_mass, bh_eps)
    ii, jj, kk = @index(Global, NTuple)
    ng = NG
    i = ii + ng;  j = jj + ng;  k = kk + ng
    xc = x0 + (ii - 0.5) * dx
    yc = y0 + (jj - 0.5) * dy
    zc = z0 + (kk - 0.5) * dz
    @inbounds begin
        ρ  = U[1, i, j, k]
        vx = U[2, i, j, k] / ρ
        vy = U[3, i, j, k] / ρ
        vz = U[4, i, j, k] / ρ

        ax = 0.0;  ay = 0.0;  az = 0.0
        for n in 1:length(bh_px)
            ddx = xc - bh_px[n]
            ddy = yc - bh_py[n]
            ddz = zc - bh_pz[n]
            r2  = ddx^2 + ddy^2 + ddz^2 + bh_eps[n]^2
            r3  = r2 * sqrt(r2)
            fac = -bh_mass[n] / r3
            ax += fac * ddx;  ay += fac * ddy;  az += fac * ddz
        end

        dU[2, i, j, k] += ρ * ax
        dU[3, i, j, k] += ρ * ay
        dU[4, i, j, k] += ρ * az
        dU[5, i, j, k] += ρ * (vx * ax + vy * ay + vz * az)
    end
end

# ---------------------------------------------------------------------------
# Gas sink source terms
#
# Torque-free prescription (Dempsey+ 2020) or standard uniform drain.
# bh_* are NTuples to avoid CPU→GPU pointer issues.

@kernel function _sink_sources_kernel!(dU, @Const(U), nx, ny, nz,
                                       dx, dy, dz, x0, y0, z0,
                                       bh_px, bh_py, bh_pz,
                                       bh_vx, bh_vy, bh_vz,
                                       bh_rs, bh_ts, torque_free)
    ii, jj, kk = @index(Global, NTuple)
    ng = NG
    i = ii + ng;  j = jj + ng;  k = kk + ng
    xc = x0 + (ii - 0.5) * dx
    yc = y0 + (jj - 0.5) * dy
    zc = z0 + (kk - 0.5) * dz
    @inbounds begin
        ρ  = U[1, i, j, k]
        mx = U[2, i, j, k];  my = U[3, i, j, k];  mz = U[4, i, j, k]
        E  = U[5, i, j, k]

        for n in 1:length(bh_rs)
            ddx = xc - bh_px[n]
            ddy = yc - bh_py[n]
            ddz = zc - bh_pz[n]
            r   = sqrt(ddx^2 + ddy^2 + ddz^2)
            if r < bh_rs[n]
                ts = bh_ts[n]
                dU[1, i, j, k] -= ρ / ts

                if torque_free
                    rx = ddx / r;  ry = ddy / r;  rz = ddz / r
                    vx = mx / ρ;   vy = my / ρ;   vz = mz / ρ
                    vrx = vx - bh_vx[n];  vry = vy - bh_vy[n];  vrz = vz - bh_vz[n]
                    v_r   = vrx * rx + vry * ry + vrz * rz
                    e_int = E / ρ - 0.5 * (vx^2 + vy^2 + vz^2)
                    dU[2, i, j, k] -= ρ * v_r * rx / ts
                    dU[3, i, j, k] -= ρ * v_r * ry / ts
                    dU[4, i, j, k] -= ρ * v_r * rz / ts
                    dU[5, i, j, k] -= (0.5 * ρ * v_r^2 + ρ * e_int) / ts
                else
                    dU[2, i, j, k] -= mx / ts
                    dU[3, i, j, k] -= my / ts
                    dU[4, i, j, k] -= mz / ts
                    dU[5, i, j, k] -= E  / ts
                end
            end
        end
    end
end

# ---------------------------------------------------------------------------
# Self-gravity gradient kernel
#
# Applies ∇Φ source terms to dU given the potential Φ on the active-cell grid.

@kernel function _sg_gradient_kernel!(dU, @Const(U), @Const(Φ),
                                      nx, ny, nz, inv2dx, inv2dy, inv2dz)
    ii, jj, kk = @index(Global, NTuple)
    ng = NG
    i = ii + ng;  j = jj + ng;  k = kk + ng
    @inbounds begin
        ρ   = U[1, i, j, k]
        ρvx = U[2, i, j, k];  ρvy = U[3, i, j, k];  ρvz = U[4, i, j, k]

        ip = min(ii + 1, nx);  im = max(ii - 1, 1)
        jp = min(jj + 1, ny);  jm = max(jj - 1, 1)
        kp = min(kk + 1, nz);  km = max(kk - 1, 1)

        fxfac = ip > im ? inv2dx : 2.0 * inv2dx
        fyfac = jp > jm ? inv2dy : 2.0 * inv2dy
        fzfac = kp > km ? inv2dz : 2.0 * inv2dz

        gx = (Φ[ip, jj, kk] - Φ[im, jj, kk]) * fxfac
        gy = (Φ[ii, jp, kk] - Φ[ii, jm, kk]) * fyfac
        gz = (Φ[ii, jj, kp] - Φ[ii, jj, km]) * fzfac

        vdotg = (ρvx * gx + ρvy * gy + ρvz * gz) / ρ

        dU[2, i, j, k] -= ρ * gx
        dU[3, i, j, k] -= ρ * gy
        dU[4, i, j, k] -= ρ * gz
        dU[5, i, j, k] -= ρ * vdotg
    end
end
