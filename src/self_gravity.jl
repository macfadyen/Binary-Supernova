# Gas self-gravity for BinarySupernova.
#
# Solves ∇²Φ = 4πρ  (G=1 in code units) on a uniform Cartesian grid
# using the Hockney-Eastwood (1981) zero-padding method for isolated
# (vacuum) boundary conditions.
#
# Algorithm:
#   1. Zero-pad ρ from (nx,ny,nz) to (2nx,2ny,2nz).
#   2. Build the convolution kernel K(r) = −1/√(r²+ε²) on the padded grid,
#      where ε = cbrt(dx dy dz) softens the r=0 singularity.
#   3. Φ = dV · IFFT( FFT(ρ_pad) .* FFT(K_pad) )   [discrete convolution]
#   4. Extract the (nx,ny,nz) block.
#
# Reference: Hockney & Eastwood (1981) §6-5; James (1977) J. Comput. Phys. 25.

using FFTW

# ---------------------------------------------------------------------------
# Poisson solver

"""
    solve_poisson_isolated(ρ, nx, ny, nz, dx, dy, dz) -> Φ

Solve ∇²Φ = 4πρ (G=1) on an nx×ny×nz grid with isolated boundary conditions
via Hockney-Eastwood zero-padding.

Input  `ρ`: active-cell density array of size (nx, ny, nz) — no ghost cells.
Output `Φ`: potential array of the same size.
"""
function solve_poisson_isolated(ρ::Array{Float64,3},
                                 nx::Int, ny::Int, nz::Int,
                                 dx::Real, dy::Real, dz::Real)
    fdx = Float64(dx);  fdy = Float64(dy);  fdz = Float64(dz)
    Nx  = 2nx;          Ny  = 2ny;          Nz  = 2nz
    dV  = fdx * fdy * fdz

    # Plummer-softened Green's function K(r) = -1/sqrt(r²+ε²).
    # ε ≈ one cell diagonal; eliminates the r=0 singularity.
    ε² = (fdx * fdy * fdz)^(2.0/3.0)

    K = zeros(Nx, Ny, Nz)
    @inbounds for k in 1:Nz, j in 1:Ny, i in 1:Nx
        # Fold index so that (1,1,1) is the origin.
        fi = Float64(i <= nx ? i - 1 : i - 1 - Nx)
        fj = Float64(j <= ny ? j - 1 : j - 1 - Ny)
        fk = Float64(k <= nz ? k - 1 : k - 1 - Nz)
        r2 = (fi*fdx)^2 + (fj*fdy)^2 + (fk*fdz)^2
        K[i,j,k] = -1.0 / sqrt(r2 + ε²)
    end

    # Zero-pad density.
    ρ_pad = zeros(Nx, Ny, Nz)
    ρ_pad[1:nx, 1:ny, 1:nz] .= ρ

    # Discrete convolution via FFT.
    Φ_pad = real.(ifft(fft(K) .* fft(ρ_pad))) .* dV

    return Φ_pad[1:nx, 1:ny, 1:nz]
end

# ---------------------------------------------------------------------------
# Source term for the gas RHS

"""
    add_self_gravity_source!(dU, U, nx, ny, nz, dx, dy, dz)

Add gas self-gravity source terms to the RHS array `dU[q,i,j,k]`:

  d(ρv)/dt  -= ρ ∇Φ
  dE/dt     -= ρ v · ∇Φ

The potential Φ is computed from the active-cell density via
`solve_poisson_isolated`.  ∇Φ is evaluated with second-order centered
differences (first-order one-sided at domain edges).
"""
function add_self_gravity_source!(dU, U,
                                   nx::Int, ny::Int, nz::Int,
                                   dx::Real, dy::Real, dz::Real)
    ng = NG
    fdx = Float64(dx);  fdy = Float64(dy);  fdz = Float64(dz)

    # Extract active-cell density and solve for Φ.
    ρ_act = Array{Float64,3}(U[1, ng+1:ng+nx, ng+1:ng+ny, ng+1:ng+nz])
    Φ = solve_poisson_isolated(ρ_act, nx, ny, nz, fdx, fdy, fdz)

    inv2dx = 0.5 / fdx
    inv2dy = 0.5 / fdy
    inv2dz = 0.5 / fdz

    @inbounds for k in 1:nz, j in 1:ny, i in 1:nx
        iu = i + ng;  ju = j + ng;  ku = k + ng

        ρ   = U[1, iu, ju, ku]
        ρvx = U[2, iu, ju, ku]
        ρvy = U[3, iu, ju, ku]
        ρvz = U[4, iu, ju, ku]

        # Gradient of Φ: centered, one-sided at boundaries.
        ip = min(i+1, nx);  im = max(i-1, 1)
        jp = min(j+1, ny);  jm = max(j-1, 1)
        kp = min(k+1, nz);  km = max(k-1, 1)

        fxfac = ip > im ? inv2dx : 1.0/fdx
        fyfac = jp > jm ? inv2dy : 1.0/fdy
        fzfac = kp > km ? inv2dz : 1.0/fdz

        gx = (Φ[ip,j,k] - Φ[im,j,k]) * fxfac
        gy = (Φ[i,jp,k] - Φ[i,jm,k]) * fyfac
        gz = (Φ[i,j,kp] - Φ[i,j,km]) * fzfac

        # v · ∇Φ (using ρv to avoid dividing by ρ)
        vdotg = (ρvx * gx + ρvy * gy + ρvz * gz) / ρ

        dU[2, iu, ju, ku] -= ρ * gx
        dU[3, iu, ju, ku] -= ρ * gy
        dU[4, iu, ju, ku] -= ρ * gz
        dU[5, iu, ju, ku] -= ρ * vdotg
    end
    return nothing
end
