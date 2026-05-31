# Checkpoint round-trip tests: build a small FMRGrid3D + BHs, save, load,
# compare bit-for-bit.

@testset "Checkpoint round-trip: FMR grid + BHs" begin
    γ = 5/3
    ng = BinarySupernova.NG

    # Coarse level: 8³ active cells, dx=1/8. Fine region: coarse cells 4:5
    # in each axis (refinement ratio 4, so a 2×2×2 coarse patch → 8³ fine).
    nc = BinarySupernova.FMRLevel3D(8, 8, 8, 1/8, 1/8, 1/8)
    for k in ng+1:ng+8, j in ng+1:ng+8, i in ng+1:ng+8
        xc = (i - ng - 0.5) * (1/8) - 0.5
        yc = (j - ng - 0.5) * (1/8) - 0.5
        zc = (k - ng - 0.5) * (1/8) - 0.5
        r2 = xc^2 + yc^2 + zc^2
        nc.U[1, i, j, k] = 1.0 + 0.1 * r2
        nc.U[5, i, j, k] = 1.0 / (γ - 1.0)
    end

    G = FMRGrid3D(nc, 4, 5, 4, 5, 4, 5, 4, γ;
                  bc=:outflow, ρ_floor=1e-12, P_floor=1e-12)
    # Perturb fine level so the saved fine state differs from what would
    # re-prolong from coarse — forces the bit-for-bit restore path.
    G.fine.U .*= 1.01

    units = PhysicalUnits(20.0, 10.0)
    bh1 = BlackHole(pos=[-0.5, 0.0, 0.0], vel=[0.0, -0.5, 0.0],
                    mass=0.5, eps=1e-3, units=units, r_floor=1e-3)
    bh2 = BlackHole(pos=[+0.5, 0.1, 0.0], vel=[0.0, +0.5, 0.0],
                    mass=0.5, eps=1e-3, units=units, r_floor=1e-3)
    bhs = [bh1, bh2]

    path = joinpath(mktempdir(), "ckpt.h5")
    save_checkpoint(path, G, bhs; t=0.25, step=42, dt_last=0.001)

    G2, bhs2, meta = load_checkpoint(path)

    @test meta.t              == 0.25
    @test meta.step           == 42
    @test meta.dt_last        == 0.001
    @test meta.format_version == CHECKPOINT_FORMAT_VERSION

    @test G2.ratio   == G.ratio
    @test G2.γ       == G.γ
    @test G2.bc      == G.bc
    @test G2.ρ_floor == G.ρ_floor
    @test G2.P_floor == G.P_floor
    @test (G2.ci_lo, G2.ci_hi) == (G.ci_lo, G.ci_hi)
    @test (G2.cj_lo, G2.cj_hi) == (G.cj_lo, G.cj_hi)
    @test (G2.ck_lo, G2.ck_hi) == (G.ck_lo, G.ck_hi)

    @test size(G2.coarse.U) == size(G.coarse.U)
    @test G2.coarse.U == G.coarse.U           # bit-for-bit
    @test G2.fine.U   == G.fine.U

    @test length(bhs2) == 2
    for (b, b2) in zip(bhs, bhs2)
        @test b2.pos     == b.pos
        @test b2.vel     == b.vel
        @test b2.mass    == b.mass
        @test b2.eps     == b.eps
        @test b2.c_code  == b.c_code
        @test b2.r_floor == b.r_floor
    end
end

@testset "Checkpoint with empty BH list" begin
    γ = 5/3
    nc = BinarySupernova.FMRLevel3D(8, 8, 8, 1/8, 1/8, 1/8)
    nc.U[1, :, :, :] .= 1.0
    nc.U[5, :, :, :] .= 1.0 / (γ - 1.0)
    G = FMRGrid3D(nc, 4, 5, 4, 5, 4, 5, 4, γ)

    path = joinpath(mktempdir(), "ckpt_empty.h5")
    save_checkpoint(path, G, BlackHole[])
    G2, bhs2, meta = load_checkpoint(path)
    @test isempty(bhs2)
    @test meta.t == 0.0
    @test meta.step == 0
end

# Uniform single-grid checkpoint (run_sn50_fiducial.jl driver): full ghosted
# U + BHs + the time-loop resume scalars (t, step, t_snap, t_bh2_sink_on).
@testset "Uniform checkpoint round-trip: U + BHs + meta" begin
    γ = 4/3
    ng = BinarySupernova.NG
    nx, ny, nz = 6, 6, 6
    dx = 0.2
    U = zeros(Float64, 5, nx+2ng, ny+2ng, nz+2ng)
    for i in eachindex(U)        # deterministic, distinct values incl. ghosts
        U[i] = 0.5 + 1.0e-3 * i
    end

    units = PhysicalUnits(20.0, 10.0)
    bh1 = BlackHole(pos=[-0.3, 0.0, 0.0],   vel=[0.0,  -0.7, 0.0],
                    mass=0.6, eps=2e-3, units=units, r_floor=2e-3)
    bh2 = BlackHole(pos=[+0.3, 0.05, -0.1], vel=[0.01, +0.7, 0.0],
                    mass=0.4, eps=2e-3, units=units, r_floor=2e-3)
    bhs = [bh1, bh2]

    path = joinpath(mktempdir(), "ckpt_uniform.h5")
    save_checkpoint_uniform(path, U, bhs;
        t=7.5, step=12345, dt_last=2.5e-3, t_snap=8.0, t_bh2_sink_on=5.0,
        nx=nx, ny=ny, nz=nz, dx=dx, dy=dx, dz=dx, γ=γ,
        x0=-1.0, y0=-1.0, z0=-1.0, ρ_floor=3e-4, P_floor=1e-5)

    U2, bhs2, grid, meta = load_checkpoint_uniform(path)

    @test size(U2) == size(U)
    @test U2 == U                                  # bit-for-bit incl. ghosts
    @test meta.t == 7.5
    @test meta.step == 12345
    @test meta.dt_last == 2.5e-3
    @test meta.t_snap == 8.0
    @test meta.t_bh2_sink_on == 5.0
    @test meta.format_version == CHECKPOINT_FORMAT_VERSION

    @test (grid.nx, grid.ny, grid.nz) == (nx, ny, nz)
    @test grid.dx == dx
    @test grid.γ == γ
    @test (grid.x0, grid.y0, grid.z0) == (-1.0, -1.0, -1.0)
    @test grid.ρ_floor == 3e-4
    @test grid.P_floor == 1e-5

    @test length(bhs2) == 2
    for (b, b2) in zip(bhs, bhs2)
        @test b2.pos     == b.pos
        @test b2.vel     == b.vel
        @test b2.mass    == b.mass
        @test b2.eps     == b.eps
        @test b2.c_code  == b.c_code
        @test b2.r_floor == b.r_floor
    end

    # The two checkpoint kinds are not cross-loadable.
    @test_throws Exception load_checkpoint(path)            # FMR loader on uniform file
end
