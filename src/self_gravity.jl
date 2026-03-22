# Gas self-gravity for BinarySupernova.
#
# Solves ∇²Φ = 4πρ  (G=1 in code units) on a uniform Cartesian grid
# using the Hockney-Eastwood (1981) zero-padding method for isolated
# (vacuum) boundary conditions.
#
# Algorithm:
#   1. Zero-pad ρ from (nx,ny,nz) to (2nx,2ny,2nz).
#   2. Build the convolution kernel K(r) = -1/√(r²+ε²) on the padded grid,
#      where ε = cbrt(dx dy dz) softens the r=0 singularity.
#   3. Φ = dV · IFFT( FFT(ρ_pad) .* FFT(K_pad) )   [discrete convolution]
#   4. Extract the (nx,ny,nz) block.
#
# GPU path: when U is a CuArray, the density slice remains on device.
# AbstractFFTs dispatch routes fft/ifft to CUFFT automatically when CUDA.jl
# is loaded (requires CUDA.jl to be imported in the calling environment).
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
           Can be a CPU Array or CuArray; FFT dispatch is automatic.
Output `Φ`: potential array of the same size and device.
"""
function solve_poisson_isolated(ρ::AbstractArray{Float64,3},
                                 nx::Int, ny::Int, nz::Int,
                                 dx::Real, dy::Real, dz::Real)
    fdx = Float64(dx);  fdy = Float64(dy);  fdz = Float64(dz)
    Nx  = 2nx;          Ny  = 2ny;          Nz  = 2nz
    dV  = fdx * fdy * fdz

    # Plummer-softened Green's function K(r) = -1/sqrt(r²+ε²).
    # ε ≈ one cell diagonal; eliminates the r=0 singularity.
    ε² = (fdx * fdy * fdz)^(2.0/3.0)

    # Build K on CPU (small loop), then adapt to the same device as ρ.
    K_cpu = zeros(Nx, Ny, Nz)
    @inbounds for k in 1:Nz, j in 1:Ny, i in 1:Nx
        # Fold index so that (1,1,1) is the origin.
        fi = Float64(i <= nx ? i - 1 : i - 1 - Nx)
        fj = Float64(j <= ny ? j - 1 : j - 1 - Ny)
        fk = Float64(k <= nz ? k - 1 : k - 1 - Nz)
        r2 = (fi*fdx)^2 + (fj*fdy)^2 + (fk*fdz)^2
        K_cpu[i,j,k] = -1.0 / sqrt(r2 + ε²)
    end

    backend = KA.get_backend(ρ)
    K = adapt(backend, K_cpu)

    # Zero-pad density onto the same device.
    ρ_pad = KA.zeros(backend, Float64, Nx, Ny, Nz)
    ρ_pad[1:nx, 1:ny, 1:nz] .= ρ

    # Discrete convolution via FFT.
    # On CPU: uses FFTW (loaded above).
    # On CUDA: CUDA.jl re-exports AbstractFFTs which routes to cuFFT automatically.
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
differences (first-order one-sided at domain edges) via _sg_gradient_kernel!.
"""
function add_self_gravity_source!(dU, U,
                                   nx::Int, ny::Int, nz::Int,
                                   dx::Real, dy::Real, dz::Real)
    ng = NG
    fdx = Float64(dx);  fdy = Float64(dy);  fdz = Float64(dz)

    # Extract active-cell density (stays on the same device as U).
    ρ_act = view(U, 1, ng+1:ng+nx, ng+1:ng+ny, ng+1:ng+nz)
    # solve_poisson_isolated needs a contiguous array
    ρ_cont = collect(ρ_act)
    Φ = solve_poisson_isolated(ρ_cont, nx, ny, nz, fdx, fdy, fdz)

    # Apply ∇Φ source terms via KA kernel (GPU/CPU portable).
    backend = KA.get_backend(U)
    kern = _sg_gradient_kernel!(backend, _WGSIZE_3D)
    kern(dU, U, Φ, nx, ny, nz,
         0.5/fdx, 0.5/fdy, 0.5/fdz;
         ndrange = (nx, ny, nz))
    KA.synchronize(backend)
    return nothing
end
