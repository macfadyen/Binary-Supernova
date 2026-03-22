# Gas-sink source terms for one or more black holes.
# Applied at each SSP-RK3 substage as part of the method-of-lines RHS.
#
# Torque-free prescription (Dempsey, Munoz & Lithwick 2020, arXiv:2002.05164):
#   d(ρ)/dt    = −ρ / t_sink
#   d(ρv)/dt   = −(ρ v_r) r̂ / t_sink        [radial component only — Dempsey+ eq. 6]
#   d(E)/dt    = −(½ρ v_r² + ρ e_int) / t_sink
#
# Standard sink (torque_free = false): drains all conserved variables at 1/t_sink.
#
# GPU: add_sink_sources! packs BH data into NTuples and dispatches to
# _sink_sources_kernel! (gpu_kernels.jl).  accrete! runs on CPU only
# (mutates BlackHole struct).

# ---------------------------------------------------------------------------

"""
    add_sink_sources!(dU, U, nx, ny, nz, dx, dy, dz,
                      bhs, x0, y0, z0; f_sink=1.0, torque_free=true)

Add gas-sink source terms to the method-of-lines RHS `dU` for all BHs.
"""
function add_sink_sources!(dU, U,
                            nx::Int, ny::Int, nz::Int,
                            dx::Real, dy::Real, dz::Real,
                            bhs, x0::Real, y0::Real, z0::Real;
                            f_sink    ::Float64 = 1.0,
                            torque_free::Bool   = true)
    nbh = length(bhs)
    bh_px = ntuple(n -> Float64(bhs[n].pos[1]),        nbh)
    bh_py = ntuple(n -> Float64(bhs[n].pos[2]),        nbh)
    bh_pz = ntuple(n -> Float64(bhs[n].pos[3]),        nbh)
    bh_vx = ntuple(n -> Float64(bhs[n].vel[1]),        nbh)
    bh_vy = ntuple(n -> Float64(bhs[n].vel[2]),        nbh)
    bh_vz = ntuple(n -> Float64(bhs[n].vel[3]),        nbh)
    bh_rs = ntuple(n -> Float64(r_sink(bhs[n])),       nbh)
    bh_ts = ntuple(n -> Float64(t_sink(bhs[n], f_sink)), nbh)

    backend = KA.get_backend(U)
    kern = _sink_sources_kernel!(backend, _WGSIZE_3D)
    kern(dU, U, nx, ny, nz,
         Float64(dx), Float64(dy), Float64(dz),
         Float64(x0), Float64(y0), Float64(z0),
         bh_px, bh_py, bh_pz, bh_vx, bh_vy, bh_vz,
         bh_rs, bh_ts, torque_free;
         ndrange = (nx, ny, nz))
    KA.synchronize(backend)
    return nothing
end

"""
    accrete!(bh, U, nx, ny, nz, dx, dy, dz, x0, y0, z0, dt)

Update `bh.mass` and `bh.vel` by accreting gas within `r_sink(bh)`.

The BH gains the **full** momentum of the drained gas (strict momentum
conservation; CLAUDE.md §4.3), regardless of the `torque_free` flag used in
`add_sink_sources!`.  The asymmetry is intentional: torque-free removes only
radial momentum from the gas, but the BH bookkeeping is fully conservative.

Runs on CPU (mutates BlackHole struct). For GPU simulations, the active-cell
gas state is copied to host before the loop.
"""
function accrete!(bh::BlackHole, U,
                  nx::Int, ny::Int, nz::Int,
                  dx::Real, dy::Real, dz::Real,
                  x0::Real, y0::Real, z0::Real, dt::Float64;
                  f_sink::Float64 = 1.0)
    ng  = NG
    rs  = r_sink(bh)
    ts  = t_sink(bh, f_sink)
    dV  = dx * dy * dz
    # Copy active-cell slice to host for BH mutation loop (tiny overhead — O(N³))
    U_host = Array(view(U, :, ng+1:ng+nx, ng+1:ng+ny, ng+1:ng+nz))
    Δm  = 0.0
    ΔPx = 0.0;  ΔPy = 0.0;  ΔPz = 0.0
    @inbounds for k in 1:nz, j in 1:ny, i in 1:nx
        xc = x0 + (i - 0.5) * dx
        yc = y0 + (j - 0.5) * dy
        zc = z0 + (k - 0.5) * dz
        r  = sqrt((xc - bh.pos[1])^2 + (yc - bh.pos[2])^2 + (zc - bh.pos[3])^2)
        r >= rs && continue
        rate = dV / ts * dt
        Δm  += U_host[1, i, j, k] * rate
        ΔPx += U_host[2, i, j, k] * rate
        ΔPy += U_host[3, i, j, k] * rate
        ΔPz += U_host[4, i, j, k] * rate
    end
    if Δm > 0.0
        M_new       = bh.mass + Δm
        bh.vel[1]   = (bh.mass * bh.vel[1] + ΔPx) / M_new
        bh.vel[2]   = (bh.mass * bh.vel[2] + ΔPy) / M_new
        bh.vel[3]   = (bh.mass * bh.vel[3] + ΔPz) / M_new
        bh.mass     = M_new
    end
    return nothing
end
