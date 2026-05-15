# Fixed Keplerian circular orbit — kinematic prescription for BH motion.
#
# When activated, BH positions and velocities are set analytically at each
# step instead of being integrated by `nbody_step!`.  BHs exert gravity on
# the gas as usual and `accrete!` still grows `bh.mass`, but gas forces and
# accretion momentum do NOT back-react on the orbit.  Useful for isolating
# the hydrodynamic response (sink accretion, wake structure, torques) from
# the N-body dynamics.
#
# Restricted to two bodies in a circular orbit about a fixed COM.
#
# At t=0 the orbit is captured from the current (pos, vel, mass) of the two
# BHs; from then on,
#   r_i(t) = R_nhat(Ω t) · r_i(0),   v_i(t) = Ω n̂ × r_i(t)
# where R_nhat is a Rodrigues rotation by angle Ω t about n̂, and Ω is frozen
# at its t=0 value.  By default Ω is taken from BH 1's initial speed,
#   Ω = |v1| / |r1|      (r1 measured from the COM)
# so the initial state is preserved exactly (no velocity snap at the first
# step).  For a self-consistent circular Keplerian pair this equals the
# point-mass formula sqrt(G (M1+M2)/a³); for a pre-SN state whose velocities
# reflect the pre-collapse inertial mass (e.g. stellar gas on the grid), the
# user's chosen period is preserved.  Ω can also be set explicitly via the
# constructor keyword.  BH masses may grow between steps without changing Ω.

"""
    KeplerOrbit

Kinematic prescription of a fixed two-body circular Keplerian orbit.

Fields
- `com`   : center-of-mass position (fixed)
- `Ω`     : angular frequency (frozen at construction)
- `nhat`  : unit normal to orbit plane (right-hand rule → sense of rotation)
- `r_init`: BH positions at t=0 relative to `com` (length 2)
"""
struct KeplerOrbit
    com    :: NTuple{3, Float64}
    Ω      :: Float64
    nhat   :: NTuple{3, Float64}
    r_init :: Vector{NTuple{3, Float64}}
end

"""
    KeplerOrbit(bhs; com=(0,0,0), nhat=nothing, Ω=nothing)

Capture the current two-BH state as a fixed circular orbit.

- `com`  : center-of-mass position.  The BHs are assumed to already be on a
           circular orbit about this point; the constructor does not recentre.
- `nhat` : optional orbit-plane unit normal.  If omitted, it is inferred from
           `r1 × v1` (in COM frame) — the instantaneous angular-momentum
           direction of BH 1.
- `Ω`    : optional explicit angular frequency.  If omitted (default), Ω is
           derived from BH 1's initial speed and COM-frame radius,
           `Ω = |v1| / |r1|`, so whatever velocities are in `bhs` at
           construction time are preserved exactly by `kepler_update!` — no
           instantaneous velocity snap at the first step.  For a
           self-consistent Keplerian pair this agrees with
           `Ω = sqrt(G M_pt / a³)`; for a pre-SN state where the orbit is set
           by a larger inertial mass (e.g. stellar gas still on the grid), it
           preserves the user's intended period.
"""
function KeplerOrbit(bhs::Vector{BlackHole};
                     com ::NTuple{3,Float64} = (0.0, 0.0, 0.0),
                     nhat::Union{Nothing,NTuple{3,Float64}} = nothing,
                     Ω   ::Union{Nothing,Real} = nothing)
    @assert length(bhs) == 2 "KeplerOrbit supports exactly two BHs"

    dx = bhs[1].pos[1] - bhs[2].pos[1]
    dy = bhs[1].pos[2] - bhs[2].pos[2]
    dz = bhs[1].pos[3] - bhs[2].pos[3]
    a  = sqrt(dx*dx + dy*dy + dz*dz)
    @assert a > 0  "BHs are coincident — cannot build a Kepler orbit"

    rx1 = bhs[1].pos[1] - com[1]
    ry1 = bhs[1].pos[2] - com[2]
    rz1 = bhs[1].pos[3] - com[3]
    vx1, vy1, vz1 = bhs[1].vel[1], bhs[1].vel[2], bhs[1].vel[3]

    nh = if nhat === nothing
        Lx = ry1*vz1 - rz1*vy1
        Ly = rz1*vx1 - rx1*vz1
        Lz = rx1*vy1 - ry1*vx1
        Lm = sqrt(Lx*Lx + Ly*Ly + Lz*Lz)
        Lm > 0 || error("BH1 has zero angular momentum in COM frame — " *
                        "pass nhat explicitly or give BH1 a tangential velocity")
        (Lx/Lm, Ly/Lm, Lz/Lm)
    else
        nm = sqrt(nhat[1]^2 + nhat[2]^2 + nhat[3]^2)
        @assert abs(nm - 1) < 1e-10 "nhat must be a unit vector"
        nhat
    end

    Ω_val = if Ω === nothing
        Rmag = sqrt(rx1*rx1 + ry1*ry1 + rz1*rz1)
        vmag = sqrt(vx1*vx1 + vy1*vy1 + vz1*vz1)
        @assert Rmag > 0 "BH1 sits at the COM — cannot derive Ω from |v|/|r|; " *
                         "pass Ω explicitly"
        @assert vmag > 0 "BH1 has zero velocity — pass Ω explicitly"
        vmag / Rmag
    else
        @assert Ω > 0 "Ω must be positive"
        Float64(Ω)
    end

    r_init = [(bhs[i].pos[1] - com[1],
               bhs[i].pos[2] - com[2],
               bhs[i].pos[3] - com[3]) for i in 1:2]

    return KeplerOrbit(com, Ω_val, nh, r_init)
end

"""
    kepler_update!(bhs, orbit, t)

Set `bhs[i].pos` and `bhs[i].vel` to the analytic circular-orbit state at
absolute time `t`.  Rotates `orbit.r_init[i]` by Ω·t about `orbit.nhat`
(Rodrigues formula) and sets `v = Ω n̂ × r`.

BH masses, softening, and sink-floor fields are not touched — this function
is purely kinematic.  `bh.mass` may have been updated by `accrete!` since
construction; that does NOT affect the orbit.
"""
function kepler_update!(bhs::Vector{BlackHole}, orbit::KeplerOrbit, t::Real)
    @assert length(bhs) == 2 "kepler_update! expects two BHs"
    θ = orbit.Ω * t
    c = cos(θ);  s = sin(θ)
    nx, ny, nz = orbit.nhat
    cx, cy, cz = orbit.com
    Ω  = orbit.Ω
    @inbounds for i in 1:2
        rx0, ry0, rz0 = orbit.r_init[i]
        # Rodrigues rotation: r = r0 cosθ + (n̂×r0) sinθ + n̂(n̂·r0)(1−cosθ)
        ndotr = nx*rx0 + ny*ry0 + nz*rz0
        kx = ny*rz0 - nz*ry0
        ky = nz*rx0 - nx*rz0
        kz = nx*ry0 - ny*rx0
        rx = rx0*c + kx*s + nx*ndotr*(1 - c)
        ry = ry0*c + ky*s + ny*ndotr*(1 - c)
        rz = rz0*c + kz*s + nz*ndotr*(1 - c)
        bhs[i].pos[1] = cx + rx
        bhs[i].pos[2] = cy + ry
        bhs[i].pos[3] = cz + rz
        # v = ω × r,  ω = Ω n̂
        bhs[i].vel[1] = Ω * (ny*rz - nz*ry)
        bhs[i].vel[2] = Ω * (nz*rx - nx*rz)
        bhs[i].vel[3] = Ω * (nx*ry - ny*rx)
    end
    return nothing
end
