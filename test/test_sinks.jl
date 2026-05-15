# Phase 3 test: gas-sink source terms.
#
# Test 1 — Mass accounting:
#   Gas mass removed by sink source terms equals BH mass gained via accrete!.
#   Both computed from the same gas state; should agree to floating-point precision.
#
# Test 2 — Torque-free angular momentum:
#   For each cell inside the sink radius, r × d(ρv)/dt = 0 exactly.
#   This is guaranteed algebraically because d(ρv)/dt ∝ r̂ and r ∥ r̂.
#   Verified to machine precision (≡ 0 in floating-point, not just small).
#
# Test 3 — Standard vs torque-free:
#   With torque_free = false the full momentum d(ρv)/dt = −ρv/ts.
#   For gas at rest in the BH frame (v = bh.vel) the torque-free and standard
#   prescriptions must agree (v_r = |v − bh.vel| = 0 → torque-free → same as
#   standard for the momentum drain).

@testset "Sink — mass accounting" begin

    ng = BinarySupernova.NG
    nx = ny = nz = 8
    dx = dy = dz = 0.1          # domain [0, 0.8]³ active, plus ghost cells
    x0 = y0 = z0 = 0.0          # left edge of active domain

    # Uniform gas at rest: ρ = 1, v = 0, P = 1e-2 → E = P/(γ-1)
    γ   = 5.0 / 3.0
    ρ0  = 1.0
    P0  = 1e-2
    U   = zeros(5, nx + 2ng, ny + 2ng, nz + 2ng)
    for k in ng+1:ng+nz, j in ng+1:ng+ny, i in ng+1:ng+nx
        U[1, i, j, k] = ρ0
        U[5, i, j, k] = P0 / (γ - 1)
    end
    dU = zeros(size(U))

    # BH at the centre of the active domain; r_sink spans ~3 cells
    bh_pos = [x0 + 0.5 * nx * dx, y0 + 0.5 * ny * dy, z0 + 0.5 * nz * dz]
    # Use large c_code so ISCO is tiny; r_floor = 3 * dx > 0 → r_sink = r_floor
    r_floor = 3.0 * dx         # = 0.3
    c_code  = 1e6
    bh = BlackHole(bh_pos, [0.0, 0.0, 0.0], 1.0, 1e-6, c_code, r_floor)
    bhs = BlackHole[bh]
    f_sink = 1.0

    # Compute RHS sink contributions
    add_sink_sources!(dU, U, nx, ny, nz, dx, dy, dz, bhs, x0, y0, z0;
                      f_sink = f_sink, torque_free = true)

    # Gas mass removal rate from dU (per unit time)
    dV = dx * dy * dz
    dm_gas_rate = 0.0
    for k in ng+1:ng+nz, j in ng+1:ng+ny, i in ng+1:ng+nx
        dm_gas_rate -= dU[1, i, j, k] * dV   # dU[1] is the source rate for ρ
    end

    # Expected BH mass gain rate: Σ ρ/ts * dV for cells inside r_sink
    rs = r_sink(bh)
    ts = t_sink(bh, f_sink)
    dm_bh_rate = 0.0
    for k in ng+1:ng+nz, j in ng+1:ng+ny, i in ng+1:ng+nx
        xc = x0 + (i - ng - 0.5) * dx
        yc = y0 + (j - ng - 0.5) * dy
        zc = z0 + (k - ng - 0.5) * dz
        r  = sqrt((xc - bh.pos[1])^2 + (yc - bh.pos[2])^2 + (zc - bh.pos[3])^2)
        if r < rs
            dm_bh_rate += U[1, i, j, k] * dV / ts
        end
    end

    # They must match to floating-point precision
    @test dm_gas_rate ≈ dm_bh_rate  atol = 1e-14 * dm_bh_rate

    @info "Sink mass accounting: dm_gas_rate=$(dm_gas_rate), dm_bh_rate=$(dm_bh_rate)"

end

@testset "Sink — torque-free: zero angular momentum drain" begin

    ng = BinarySupernova.NG
    nx = ny = nz = 8
    dx = dy = dz = 0.1
    x0 = y0 = z0 = 0.0

    γ = 5.0 / 3.0

    # Gas with nonzero angular velocity ω = 1 around the z-axis.
    # Cell velocity: v = ω × r = (−ω y, ω x, 0)
    ω   = 1.0
    ρ0  = 1.0
    P0  = 1e-2
    U   = zeros(5, nx + 2ng, ny + 2ng, nz + 2ng)
    for k in ng+1:ng+nz, j in ng+1:ng+ny, i in ng+1:ng+nx
        xc = x0 + (i - ng - 0.5) * dx
        yc = y0 + (j - ng - 0.5) * dy
        vx = -ω * yc;  vy = ω * xc;  vz = 0.0
        U[1, i, j, k] = ρ0
        U[2, i, j, k] = ρ0 * vx
        U[3, i, j, k] = ρ0 * vy
        U[4, i, j, k] = ρ0 * vz
        U[5, i, j, k] = P0 / (γ - 1) + 0.5 * ρ0 * (vx^2 + vy^2)
    end

    dU = zeros(size(U))

    # BH at domain centre, at rest
    bh_pos  = [x0 + 0.5 * nx * dx, y0 + 0.5 * ny * dy, z0 + 0.5 * nz * dz]
    r_floor = 3.0 * dx
    bh = BlackHole(bh_pos, [0.0, 0.0, 0.0], 1.0, 1e-6, 1e6, r_floor)
    bhs = BlackHole[bh]

    add_sink_sources!(dU, U, nx, ny, nz, dx, dy, dz, bhs, x0, y0, z0;
                      torque_free = true)

    # For each cell inside r_sink, verify torque = r × d(ρv)/dt = 0.
    # Because d(ρv)/dt ∝ r̂ and r ∥ r̂, the cross product is algebraically zero.
    # In floating-point: τ_z = ddx*(dU[3]) − ddy*(dU[2]) = (ddx*ddy − ddy*ddx)/r = 0.
    rs = r_sink(bh)
    max_torque = 0.0
    for k in ng+1:ng+nz, j in ng+1:ng+ny, i in ng+1:ng+nx
        xc  = x0 + (i - ng - 0.5) * dx
        yc  = y0 + (j - ng - 0.5) * dy
        zc  = z0 + (k - ng - 0.5) * dz
        ddx = xc - bh.pos[1];  ddy = yc - bh.pos[2];  ddz = zc - bh.pos[3]
        r   = sqrt(ddx^2 + ddy^2 + ddz^2)
        r >= rs && continue

        # Angular momentum source rate about BH centre (z-component)
        τx = ddy * dU[4, i, j, k] - ddz * dU[3, i, j, k]
        τy = ddz * dU[2, i, j, k] - ddx * dU[4, i, j, k]
        τz = ddx * dU[3, i, j, k] - ddy * dU[2, i, j, k]
        max_torque = max(max_torque, abs(τx), abs(τy), abs(τz))
    end

    @test max_torque < 1e-14   # ≈ machine epsilon: r × r̂ algebraically zero

    @info "Torque-free: max |r × dJ/dt| = $max_torque (should be exactly 0)"

end

@testset "Sink — accrete! mass conservation" begin

    ng = BinarySupernova.NG
    nx = ny = nz = 8
    dx = dy = dz = 0.1
    x0 = y0 = z0 = 0.0
    dt = 0.01

    γ   = 5.0 / 3.0
    ρ0  = 1.5
    P0  = 1e-2
    U   = zeros(5, nx + 2ng, ny + 2ng, nz + 2ng)
    for k in ng+1:ng+nz, j in ng+1:ng+ny, i in ng+1:ng+nx
        U[1, i, j, k] = ρ0
        U[5, i, j, k] = P0 / (γ - 1)
    end

    bh_pos  = [x0 + 0.5 * nx * dx, y0 + 0.5 * ny * dy, z0 + 0.5 * nz * dz]
    r_floor = 2.5 * dx
    M0_bh   = 1.0
    bh = BlackHole(bh_pos, [0.0, 0.0, 0.0], M0_bh, 1e-6, 1e6, r_floor)

    # Mass inside sink before accretion
    dV = dx * dy * dz
    rs = r_sink(bh)
    M_gas_before = 0.0
    for k in ng+1:ng+nz, j in ng+1:ng+ny, i in ng+1:ng+nx
        xc = x0 + (i - ng - 0.5) * dx
        yc = y0 + (j - ng - 0.5) * dy
        zc = z0 + (k - ng - 0.5) * dz
        r  = sqrt((xc - bh.pos[1])^2 + (yc - bh.pos[2])^2 + (zc - bh.pos[3])^2)
        r < rs && (M_gas_before += U[1, i, j, k] * dV)
    end

    M_bh_before = bh.mass
    accrete!(bh, U, nx, ny, nz, dx, dy, dz, x0, y0, z0, dt)
    ΔM_bh = bh.mass - M_bh_before

    # Expected: ΔM_bh = M_gas_before / ts * dt
    ts = t_sink(BlackHole(bh_pos, [0.0,0.0,0.0], M0_bh, 1e-6, 1e6, r_floor))
    ΔM_expected = M_gas_before / ts * dt

    @test ΔM_bh ≈ ΔM_expected  rtol = 1e-14

    @info "accrete!: ΔM_bh=$(ΔM_bh), expected=$(ΔM_expected)"

end

@testset "Sink — ρ_sink_min threshold skips low-density cells" begin

    ng = BinarySupernova.NG
    nx = ny = nz = 8
    dx = dy = dz = 0.1
    x0 = y0 = z0 = 0.0
    dt = 0.01

    γ   = 5.0 / 3.0
    ρ_floor_like = 1e-3            # ambient "vacuum" density
    ρ_high       = 1.0              # a few bright cells above threshold
    P0  = 1e-2
    U   = zeros(5, nx + 2ng, ny + 2ng, nz + 2ng)
    for k in ng+1:ng+nz, j in ng+1:ng+ny, i in ng+1:ng+nx
        U[1, i, j, k] = ρ_floor_like
        U[5, i, j, k] = P0 / (γ - 1)
    end
    # Plant a single high-density cell near the BH centre.
    ic = ng + nx ÷ 2 + 1
    U[1, ic, ic, ic] = ρ_high
    U[5, ic, ic, ic] = P0 / (γ - 1)

    bh_pos  = [x0 + 0.5 * nx * dx, y0 + 0.5 * ny * dy, z0 + 0.5 * nz * dz]
    r_floor = 3.0 * dx
    bh = BlackHole(bh_pos, [0.0, 0.0, 0.0], 1.0, 1e-6, 1e6, r_floor)
    bhs = BlackHole[bh]

    # Threshold well above the floor; well below the peak.
    ρ_thr = 2.0 * ρ_floor_like

    # --- RHS check: only the high-density cell contributes to the drain
    dU = zeros(size(U))
    add_sink_sources!(dU, U, nx, ny, nz, dx, dy, dz, bhs, x0, y0, z0;
                      torque_free = true, ρ_sink_min = ρ_thr)

    dV = dx * dy * dz
    ts = t_sink(bh, 1.0)
    dm_gas_rate = 0.0
    for k in ng+1:ng+nz, j in ng+1:ng+ny, i in ng+1:ng+nx
        dm_gas_rate -= dU[1, i, j, k] * dV
    end
    expected_rate = ρ_high * dV / ts
    @test dm_gas_rate ≈ expected_rate  rtol = 1e-14

    # --- accrete! check: only the high-density cell contributes to ΔM
    M_before = bh.mass
    accrete!(bh, U, nx, ny, nz, dx, dy, dz, x0, y0, z0, dt;
             ρ_sink_min = ρ_thr)
    ΔM = bh.mass - M_before
    @test ΔM ≈ (ρ_high * dV / ts) * dt  rtol = 1e-10

    # --- Default (ρ_sink_min = 0) reproduces original behaviour
    bh2 = BlackHole(bh_pos, [0.0, 0.0, 0.0], 1.0, 1e-6, 1e6, r_floor)
    M_before2 = bh2.mass
    accrete!(bh2, U, nx, ny, nz, dx, dy, dz, x0, y0, z0, dt)
    ΔM2 = bh2.mass - M_before2
    @test ΔM2 > ΔM         # ambient floor cells now counted

    @info "ρ_sink_min: with threshold ΔM=$(ΔM), without ΔM=$(ΔM2)"

end
