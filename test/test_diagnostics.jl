# Phase 6 tests: diagnostics and I/O.
#
# Test 1 — Energy diagnostic: gas_energy_total matches Σ U[5] dV.
# Test 2 — Momentum diagnostic: gas_momentum_total = [0,0,0] for symmetric IC.
# Test 3 — Angular momentum: L = 0 for radially symmetric velocity field.
# Test 4 — BH diagnostics: kinetic energy and angular momentum are correct.
# Test 5 — Bound mass: uniform gas with no BH → all bound if KE+thermal ≤ 0.
# Test 6 — Snapshot write/read round-trip: data survives HDF5 serialisation.
# Test 7 — Trajectory append/read round-trip: BH state survives serialisation.

using Statistics: mean

@testset "Gas energy diagnostic" begin

    ng = BinarySupernova.NG
    nx = ny = nz = 8
    dx = 0.1
    γ  = 5.0/3.0

    U = zeros(5, nx+2ng, ny+2ng, nz+2ng)
    for k in ng+1:ng+nz, j in ng+1:ng+ny, i in ng+1:ng+nx
        U[1,i,j,k] = 1.0
        U[5,i,j,k] = 2.0   # energy density
    end

    E = gas_energy_total(U, nx, ny, nz, dx, dx, dx)
    E_ref = 2.0 * nx*ny*nz * dx^3   # sum of energy densities × dV

    @test E ≈ E_ref  atol=1e-14*E_ref
    @info "Energy diagnostic: E=$(round(E, digits=6)), ref=$(round(E_ref, digits=6))"

end

@testset "Gas momentum diagnostic — symmetric IC" begin

    ng = BinarySupernova.NG
    nx = ny = nz = 8
    dx = 0.1
    x0 = -0.4

    # Uniform gas at rest → zero total momentum
    U = zeros(5, nx+2ng, ny+2ng, nz+2ng)
    for k in ng+1:ng+nz, j in ng+1:ng+ny, i in ng+1:ng+nx
        U[1,i,j,k] = 1.0
        U[5,i,j,k] = 0.01
    end

    P = gas_momentum_total(U, nx, ny, nz, dx, dx, dx)
    @test maximum(abs.(P)) < 1e-14

end

@testset "BH kinetic energy and angular momentum" begin

    # BH1 at (+0.5, 0, 0) with v = (0, 0.5, 0), mass 0.5
    bh1 = BlackHole([+0.5, 0.0, 0.0], [0.0, 0.5, 0.0], 0.5, 0.01, 1e6, 0.01)
    # BH2 at (-0.5, 0, 0) with v = (0, -0.5, 0), mass 0.5
    bh2 = BlackHole([-0.5, 0.0, 0.0], [0.0,-0.5, 0.0], 0.5, 0.01, 1e6, 0.01)
    bhs = BlackHole[bh1, bh2]

    KE_BH = bh_kinetic_total(bhs)
    @test KE_BH ≈ 0.5*0.5*0.25 + 0.5*0.5*0.25   # 2 × (1/2 × 0.5 × 0.5²)
    @test KE_BH ≈ 0.125  atol=1e-14

    L_BH = bh_angular_momentum_total(bhs)
    # BH1: M r × v = 0.5 × (+0.5, 0, 0) × (0, 0.5, 0) = 0.5 × (0, 0, 0.25) = (0,0,0.125)
    # BH2: 0.5 × (-0.5, 0, 0) × (0,-0.5, 0) = 0.5 × (0, 0, 0.25) = (0,0,0.125)
    @test L_BH[3] ≈ 0.25  atol=1e-14
    @test abs(L_BH[1]) < 1e-14
    @test abs(L_BH[2]) < 1e-14

    @info "BH diagnostics: KE_BH=$(KE_BH), L_z=$(L_BH[3])"

end

@testset "Bound mass diagnostic" begin

    ng = BinarySupernova.NG
    nx = ny = nz = 8
    dx = 0.1
    x0 = -0.4
    γ  = 5.0/3.0

    # Gas at rest with tiny thermal energy, dominated by a BH potential.
    # BH at origin with M = 10 → deep potential well.
    bh = BlackHole([0.0, 0.0, 0.0], [0.0,0.0,0.0], 10.0, 0.01, 1e6, 0.01)
    bhs = BlackHole[bh]

    U = zeros(5, nx+2ng, ny+2ng, nz+2ng)
    for k in ng+1:ng+nz, j in ng+1:ng+ny, i in ng+1:ng+nx
        U[1, i, j, k] = 1.0
        U[5, i, j, k] = 1e-6 / (γ-1)   # tiny thermal energy
    end

    M_bound = bound_gas_mass(U, nx, ny, nz, dx, dx, dx, x0, x0, x0, bhs, γ)
    M_total = sum(U[1, ng+1:ng+nx, ng+1:ng+ny, ng+1:ng+nz]) * dx^3

    # With M_BH=10, most cells at r ~ 0.1-0.4 have |Φ| ~ 10/0.4 = 25 >> e_int ~ 1e-6
    # → should be almost all bound
    @test M_bound / M_total > 0.9

    @info "Bound mass: M_bound=$(round(M_bound, digits=4)), M_total=$(round(M_total, digits=4)), frac=$(round(M_bound/M_total, digits=4))"

end

@testset "Snapshot round-trip (HDF5)" begin

    ng = BinarySupernova.NG
    nx = ny = nz = 8
    dx = 0.1
    γ  = 5.0/3.0
    t  = 1.23

    U = rand(5, nx+2ng, ny+2ng, nz+2ng)

    fname = tempname() * ".h5"
    try
        write_snapshot(fname, U, nx, ny, nz, dx, dx, dx, t, γ)
        U2, nx2, ny2, nz2, dx2, dy2, dz2, t2, γ2 = read_snapshot(fname)

        @test nx2 == nx && ny2 == ny && nz2 == nz
        @test dx2 ≈ dx && t2 ≈ t && γ2 ≈ γ
        # Active cells should round-trip exactly
        @test U2 ≈ U[1:5, ng+1:ng+nx, ng+1:ng+ny, ng+1:ng+nz]
    finally
        isfile(fname) && rm(fname)
    end

end

@testset "BH trajectory round-trip (HDF5)" begin

    bh1 = BlackHole([1.0, 0.0, 0.0], [0.0, 0.5, 0.0], 0.4, 0.01, 1e6, 0.01)
    bh2 = BlackHole([-1.0,0.0, 0.0], [0.0,-0.5, 0.0], 0.6, 0.01, 1e6, 0.01)
    bhs = BlackHole[bh1, bh2]

    fname = tempname() * "_traj.h5"
    try
        init_trajectory_file(fname, length(bhs))
        for step in 0:4
            append_trajectory(fname, Float64(step) * 0.1, bhs)
        end

        t_arr, bh_data = read_trajectory(fname)

        @test length(t_arr) == 5
        @test t_arr[1] ≈ 0.0 && t_arr[end] ≈ 0.4
        @test length(bh_data) == 2
        @test bh_data[1].mass[1] ≈ bh1.mass
        @test bh_data[2].pos[1, 1] ≈ bh2.pos[1]   # first dump, x-coordinate
    finally
        isfile(fname) && rm(fname)
    end

end
