# Gas-sink source terms for one or more black holes.
# Applied at each SSP-RK3 substage as part of the method-of-lines RHS.
#
# Torque-free prescription (Dempsey, Munoz & Lithwick 2020, arXiv:2002.05164):
#   d(ρ)/dt    = −ρ / t_sink
#   d(ρv)/dt   = −(ρ v_r) r̂ / t_sink        [radial component only — Dempsey+ eq. 6]
#   d(E)/dt    = −(½ρ v_r² + ρ e_int) / t_sink
#
# where:
#   d    = r_cell − bh.pos          (displacement from BH, code units)
#   r̂   = d / |d|                   (unit radial vector)
#   v    = (ρv) / ρ                  (primitive velocity)
#   v_r  = (v − bh.vel) · r̂         (radial speed in BH rest frame)
#   e_int = E/ρ − ½|v|²             (specific internal energy)
#
# The tangential kinetic energy ½ρ|v_tan|² is NOT removed; gas surrounding the
# sink conserves angular momentum.  Torque on gas about BH centre = r̂ × r̂ = 0. ✓
#
# Standard sink (torque_free = false): drains all conserved variables at 1/t_sink,
# matching the simplest uniform-drain prescription for comparison tests.
#
# BH mass/velocity update (accrete!): BH gains the full gas momentum ρv (not just
# radial) for strict N-body momentum conservation (CLAUDE.md §4.3).

# ---------------------------------------------------------------------------

"""
    add_sink_sources!(dU, U, nx, ny, nz, dx, dy, dz,
                      bhs, x0, y0, z0; f_sink=1.0, torque_free=true)

Add gas-sink source terms to the method-of-lines RHS `dU` for all BHs.

Cells within `r_sink(bh)` of each BH are drained at rate `1 / t_sink(bh, f_sink)`.

- `x0, y0, z0` : physical left edge of the active domain.
- `f_sink`      : multiplier on the free-fall timescale (default 1.0).
- `torque_free` : if true, use the torque-free prescription (Dempsey+ 2020);
                  otherwise drain all conserved variables uniformly.
"""
function add_sink_sources!(dU, U,
                            nx::Int, ny::Int, nz::Int,
                            dx::Real, dy::Real, dz::Real,
                            bhs, x0::Real, y0::Real, z0::Real;
                            f_sink    ::Float64 = 1.0,
                            torque_free::Bool   = true)
    ng = NG
    @inbounds for bh in bhs
        rs = r_sink(bh)
        ts = t_sink(bh, f_sink)
        for k in ng+1:ng+nz, j in ng+1:ng+ny, i in ng+1:ng+nx
            xc = x0 + (i - ng - 0.5) * dx
            yc = y0 + (j - ng - 0.5) * dy
            zc = z0 + (k - ng - 0.5) * dz
            ddx = xc - bh.pos[1]
            ddy = yc - bh.pos[2]
            ddz = zc - bh.pos[3]
            r = sqrt(ddx^2 + ddy^2 + ddz^2)
            r >= rs && continue

            ρ  = U[1, i, j, k]
            mx = U[2, i, j, k]; my = U[3, i, j, k]; mz = U[4, i, j, k]
            E  = U[5, i, j, k]

            dU[1, i, j, k] -= ρ / ts

            if torque_free
                # Radial unit vector (BH frame)
                rx = ddx / r;  ry = ddy / r;  rz = ddz / r
                # Primitive velocity
                vx = mx / ρ;  vy = my / ρ;  vz = mz / ρ
                # Relative velocity: gas minus BH
                vrx = vx - bh.vel[1]
                vry = vy - bh.vel[2]
                vrz = vz - bh.vel[3]
                # Radial speed in BH rest frame
                v_r = vrx * rx + vry * ry + vrz * rz
                # Specific internal energy (total − kinetic)
                e_int = E / ρ - 0.5 * (vx^2 + vy^2 + vz^2)
                # Dempsey+ 2020 eq. 6: radial momentum only
                dU[2, i, j, k] -= ρ * v_r * rx / ts
                dU[3, i, j, k] -= ρ * v_r * ry / ts
                dU[4, i, j, k] -= ρ * v_r * rz / ts
                # Dempsey+ 2020 eq. 7: radial KE + thermal
                dU[5, i, j, k] -= (0.5 * ρ * v_r^2 + ρ * e_int) / ts
            else
                # Standard: drain all conserved variables uniformly
                dU[2, i, j, k] -= mx / ts
                dU[3, i, j, k] -= my / ts
                dU[4, i, j, k] -= mz / ts
                dU[5, i, j, k] -= E  / ts
            end
        end
    end
    return nothing
end

"""
    accrete!(bh, U, nx, ny, nz, dx, dy, dz, x0, y0, z0, dt)

Update `bh.mass` and `bh.vel` by accreting gas within `r_sink(bh)`.

The BH gains the **full** momentum of the drained gas (strict momentum
conservation; CLAUDE.md §4.3), regardless of the `torque_free` flag used in
`add_sink_sources!`.  The asymmetry is intentional: torque-free removes only
radial momentum from the gas, but the BH bookkeeping is fully conservative.

Typical usage: call once per timestep with the pre-step gas state `U`.
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
    Δm  = 0.0
    ΔPx = 0.0;  ΔPy = 0.0;  ΔPz = 0.0
    @inbounds for k in ng+1:ng+nz, j in ng+1:ng+ny, i in ng+1:ng+nx
        xc = x0 + (i - ng - 0.5) * dx
        yc = y0 + (j - ng - 0.5) * dy
        zc = z0 + (k - ng - 0.5) * dz
        r  = sqrt((xc - bh.pos[1])^2 + (yc - bh.pos[2])^2 + (zc - bh.pos[3])^2)
        r >= rs && continue
        rate = dV / ts * dt
        Δm  += U[1, i, j, k] * rate
        ΔPx += U[2, i, j, k] * rate
        ΔPy += U[3, i, j, k] * rate
        ΔPz += U[4, i, j, k] * rate
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
