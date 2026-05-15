# FMR self-gravity — composite-density base-level Poisson solve + Φ prolongation.
#
# Algorithm (base-level-only variant):
#   1. Restrict fine-level ρ into the coarse level across the fine region
#      (conservative 1/r³ average — same operator as `_restrict!` but
#      acting only on the density component).
#   2. Solve ∇²Φ = 4πρ on the coarse level via `solve_poisson_isolated`
#      (Hockney-Eastwood FFT, isolated BCs).
#   3. Prolong the coarse Φ trilinearly onto the fine active cells so both
#      levels have a gradient to evaluate ∇Φ on.
#
# The full multi-level Dirichlet-BC solve (each level gets its own Poisson
# solve with BCs supplied by the parent) is deferred: it is the physically
# correct choice for very high refinement, but at 4:1 with a single fine
# patch the base-level solve is already adequate, because the composite
# density still captures the fine mass at coarse resolution.
#
# Ported from Binary-PISN/src/fmr_gravity.jl.
#
# Limitations / caveats:
#   - Φ on the fine level is only coarse-resolution (prolonged); fine
#     gradients therefore reflect coarse-cell smoothing. Source terms on
#     the fine level are physically consistent with the coarse potential
#     but do not resolve sub-coarse-cell gravitational structure.
#   - No periodic / cosmological BCs; isolated-only.

# ---------------------------------------------------------------------------
# Density restriction — fine active cells → coarse active cells (1/r³ mean).
# Same logic as `_restrict!` but touches only the ρ component (q=1), so it
# can be called on its own without rebuilding the full conserved state.

"""
    restrict_density_to_coarse!(G::FMRGrid3D)

Overwrite coarse ρ inside the fine region with the conservative 1/r³
average of fine ρ. Leaves the rest of the coarse state (momentum, energy)
untouched. Required before `solve_fmr_poisson` to build the composite
density seen by the Poisson solve.
"""
function restrict_density_to_coarse!(G::FMRGrid3D)
    U_c = G.coarse.U
    U_f = G.fine.U
    r      = G.ratio
    ng     = NG
    inv_r3 = 1.0 / r^3

    @inbounds for m in 0:(G.ci_hi - G.ci_lo),
                  n in 0:(G.cj_hi - G.cj_lo),
                  p in 0:(G.ck_hi - G.ck_lo)
        ic_g = ng + G.ci_lo + m
        jc_g = ng + G.cj_lo + n
        kc_g = ng + G.ck_lo + p
        val = 0.0
        for sx in 1:r, sy in 1:r, sz in 1:r
            val += U_f[1, ng + r*m + sx, ng + r*n + sy, ng + r*p + sz]
        end
        U_c[1, ic_g, jc_g, kc_g] = val * inv_r3
    end
    return nothing
end

# ---------------------------------------------------------------------------
# Trilinear prolongation of a scalar Φ (coarse active grid → fine active grid).
#
# Fine cell (if, jf, kf) ∈ 1:nx_f with centre at physical coordinate
#   x_f = (G.ci_lo - 1 + (if - 0.5)/r) * dx_c
# relative to the coarse active-cell origin. We interpolate Φ_c (coarse
# active-cell indices 1:nxc) at that position using 2×2×2 trilinear weights.

function _prolong_scalar_trilinear!(Φ_f::AbstractArray{Float64,3},
                                     Φ_c::AbstractArray{Float64,3},
                                     G::FMRGrid3D)
    r  = G.ratio
    nxc, nyc, nzc = size(Φ_c)
    nxf, nyf, nzf = size(Φ_f)

    @inbounds for kf in 1:nzf, jf in 1:nyf, ifi in 1:nxf
        # Position of fine cell centre in coarse active-cell coordinates.
        xc = (G.ci_lo - 1) + (ifi - 0.5) / r + 0.5
        yc = (G.cj_lo - 1) + (jf  - 0.5) / r + 0.5
        zc = (G.ck_lo - 1) + (kf  - 0.5) / r + 0.5

        ic = clamp(floor(Int, xc), 1, nxc - 1)
        jc = clamp(floor(Int, yc), 1, nyc - 1)
        kc = clamp(floor(Int, zc), 1, nzc - 1)
        fx = xc - ic
        fy = yc - jc
        fz = zc - kc

        w000 = (1-fx)*(1-fy)*(1-fz)
        w100 =    fx *(1-fy)*(1-fz)
        w010 = (1-fx)*   fy *(1-fz)
        w110 =    fx *   fy *(1-fz)
        w001 = (1-fx)*(1-fy)*   fz
        w101 =    fx *(1-fy)*   fz
        w011 = (1-fx)*   fy *   fz
        w111 =    fx *   fy *   fz

        Φ_f[ifi, jf, kf] =
            w000 * Φ_c[ic,   jc,   kc  ] + w100 * Φ_c[ic+1, jc,   kc  ] +
            w010 * Φ_c[ic,   jc+1, kc  ] + w110 * Φ_c[ic+1, jc+1, kc  ] +
            w001 * Φ_c[ic,   jc,   kc+1] + w101 * Φ_c[ic+1, jc,   kc+1] +
            w011 * Φ_c[ic,   jc+1, kc+1] + w111 * Φ_c[ic+1, jc+1, kc+1]
    end
    return nothing
end

# ---------------------------------------------------------------------------
# Composite FMR Poisson solve.

"""
    solve_fmr_poisson(G::FMRGrid3D) -> (Φ_c, Φ_f)

Solve ∇²Φ = 4πρ on the FMR composite density. Returns coarse and fine
active-cell potential arrays (no ghosts). `restrict_density_to_coarse!`
is called unconditionally since it is cheap and idempotent once the
fine state is valid.
"""
function solve_fmr_poisson(G::FMRGrid3D)
    restrict_density_to_coarse!(G)

    nxc, nyc, nzc = G.coarse.nx, G.coarse.ny, G.coarse.nz
    ng = NG
    ρ_c_act = collect(view(G.coarse.U, 1, ng+1:ng+nxc, ng+1:ng+nyc, ng+1:ng+nzc))
    Φ_c = solve_poisson_isolated(ρ_c_act, nxc, nyc, nzc,
                                 G.coarse.dx, G.coarse.dy, G.coarse.dz)

    nxf, nyf, nzf = G.fine.nx, G.fine.ny, G.fine.nz
    Φ_f = zeros(nxf, nyf, nzf)
    _prolong_scalar_trilinear!(Φ_f, Φ_c, G)

    return Φ_c, Φ_f
end

# ---------------------------------------------------------------------------
# Per-level ρ∇Φ source application (reusable by `fmr3d_step!` which solves
# Φ once per coarse step and applies it across all SSP-RK3 stages).

"""
    _apply_sg_force_level!(dU, U, Φ, nx, ny, nz, dx, dy, dz)

Apply ρ∇Φ / ρv·∇Φ source terms on one level using a pre-computed
potential `Φ` (active-cell grid). Used by `add_fmr_self_gravity_source!`
and by `fmr3d_step!` with `self_gravity=true` (frozen Φ across stages).
"""
function _apply_sg_force_level!(dU, U, Φ::AbstractArray{Float64,3},
                                 nx::Int, ny::Int, nz::Int,
                                 dx::Real, dy::Real, dz::Real)
    backend = KA.get_backend(U)
    kern    = _sg_gradient_kernel!(backend, _WGSIZE_3D)
    kern(dU, U, Φ, nx, ny, nz,
         0.5/Float64(dx), 0.5/Float64(dy), 0.5/Float64(dz);
         ndrange = (nx, ny, nz))
    KA.synchronize(backend)
    return nothing
end

"""
    add_fmr_self_gravity_source!(dU_c, dU_f, G::FMRGrid3D; G_grav=1.0)

Apply ρ∇Φ / ρv·∇Φ source terms to coarse and fine RHS arrays using a
composite-density Poisson solve. The fine Φ is a trilinearly prolonged
copy of the coarse Φ (see module docstring). `G_grav` rescales the
potential for non-unit Newton's constant (default `G_grav = 1.0` for
code units).
"""
function add_fmr_self_gravity_source!(dU_c, dU_f, G::FMRGrid3D;
                                       G_grav::Real = 1.0)
    Φ_c, Φ_f = solve_fmr_poisson(G)
    if G_grav != 1.0
        Φ_c .*= Float64(G_grav)
        Φ_f .*= Float64(G_grav)
    end
    _apply_sg_force_level!(dU_c, G.coarse.U, Φ_c,
                           G.coarse.nx, G.coarse.ny, G.coarse.nz,
                           G.coarse.dx, G.coarse.dy, G.coarse.dz)
    _apply_sg_force_level!(dU_f, G.fine.U, Φ_f,
                           G.fine.nx, G.fine.ny, G.fine.nz,
                           G.fine.dx, G.fine.dy, G.fine.dz)
    return nothing
end
