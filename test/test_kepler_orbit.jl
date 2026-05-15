# Kepler orbit (kinematic prescription) tests.
#
# KeplerOrbit freezes Ω at construction and updates BH pos/vel analytically
# via Rodrigues rotation.  Because the update is exact (not integrated), we
# can hold the BHs to machine precision over arbitrary times.

@testset "KeplerOrbit — kinematic prescription" begin

    # ------------------------------------------------------------------ setup
    # Equal-mass circular orbit in xy plane, COM at origin.
    #   M1 = M2 = 0.5, a = 1, Ω = sqrt(1/1) = 1, T = 2π
    #   each BH at radius 0.5, circular speed 0.5.
    function make_pair()
        bh1 = BlackHole([+0.5, 0.0, 0.0], [0.0, +0.5, 0.0],
                        0.5, 1e-4, 1e6, 0.01)
        bh2 = BlackHole([-0.5, 0.0, 0.0], [0.0, -0.5, 0.0],
                        0.5, 1e-4, 1e6, 0.01)
        return BlackHole[bh1, bh2]
    end

    # ---------------------------------------------------------- construction
    @testset "construction infers Ω and plane normal" begin
        bhs   = make_pair()
        orbit = KeplerOrbit(bhs)

        @test isapprox(orbit.Ω, 1.0; atol=1e-12)
        @test orbit.com == (0.0, 0.0, 0.0)
        @test isapprox(orbit.nhat[1], 0.0; atol=1e-12)
        @test isapprox(orbit.nhat[2], 0.0; atol=1e-12)
        @test isapprox(orbit.nhat[3], 1.0; atol=1e-12)
        @test length(orbit.r_init) == 2
    end

    # ------------------------------------------------------ closure after 1 P
    @testset "closure to machine precision after one period" begin
        bhs   = make_pair()
        pos0  = [copy(bh.pos) for bh in bhs]
        vel0  = [copy(bh.vel) for bh in bhs]
        orbit = KeplerOrbit(bhs)

        T = 2π / orbit.Ω
        kepler_update!(bhs, orbit, T)

        for i in 1:2
            @test maximum(abs.(bhs[i].pos .- pos0[i])) < 1e-12
            @test maximum(abs.(bhs[i].vel .- vel0[i])) < 1e-12
        end
    end

    # ------------------------------------------ invariants over many periods
    @testset "separation and speed constant over 100 periods" begin
        bhs   = make_pair()
        orbit = KeplerOrbit(bhs)

        a0 = sqrt(sum((bhs[1].pos .- bhs[2].pos).^2))
        s1 = sqrt(sum(bhs[1].vel.^2))
        s2 = sqrt(sum(bhs[2].vel.^2))

        T = 2π / orbit.Ω
        for k in 1:1000
            t = k * T / 10           # sample 10 phases per period over 100 periods
            kepler_update!(bhs, orbit, t)
            a = sqrt(sum((bhs[1].pos .- bhs[2].pos).^2))
            @test isapprox(a, a0; rtol=1e-12)
            @test isapprox(sqrt(sum(bhs[1].vel.^2)), s1; rtol=1e-12)
            @test isapprox(sqrt(sum(bhs[2].vel.^2)), s2; rtol=1e-12)
        end
    end

    # ----------------------- v ⟂ r (pure circular motion) at arbitrary times
    @testset "velocity orthogonal to COM-frame position" begin
        bhs   = make_pair()
        orbit = KeplerOrbit(bhs)

        for θ in (0.1, 0.7, 1.5, 2.9, 4.2)
            kepler_update!(bhs, orbit, θ / orbit.Ω)
            for i in 1:2
                r = bhs[i].pos .- collect(orbit.com)
                @test abs(sum(r .* bhs[i].vel)) < 1e-12
            end
        end
    end

    # ----------------------------------- Ω frozen under accreted mass growth
    @testset "Ω is frozen when BH masses grow" begin
        bhs   = make_pair()
        orbit = KeplerOrbit(bhs)
        Ω0    = orbit.Ω
        a0    = sqrt(sum((bhs[1].pos .- bhs[2].pos).^2))

        # Simulate accretion: grow masses by 20% after construction.
        bhs[1].mass *= 1.2
        bhs[2].mass *= 1.2

        T = 2π / Ω0
        kepler_update!(bhs, orbit, T)
        @test orbit.Ω == Ω0
        @test isapprox(sqrt(sum((bhs[1].pos .- bhs[2].pos).^2)), a0; rtol=1e-12)
    end

    # ----------------------------- cross-check against N-body integrator
    # Over short times the SSP-RK3 N-body solution should closely agree with
    # the analytic Kepler prescription. This catches direction / phase errors.
    @testset "agrees with nbody_step! over 1/4 orbit" begin
        bhs_k = make_pair();  orbit = KeplerOrbit(bhs_k)
        bhs_n = make_pair()

        T  = 2π / orbit.Ω
        dt = T / 2000
        F_gas = [zeros(3), zeros(3)]
        t = 0.0
        steps = 500          # 500 × dt = T/4
        for _ in 1:steps
            nbody_step!(bhs_n, dt, F_gas)
            t += dt
        end
        kepler_update!(bhs_k, orbit, t)

        for i in 1:2
            dpos = maximum(abs.(bhs_k[i].pos .- bhs_n[i].pos))
            dvel = maximum(abs.(bhs_k[i].vel .- bhs_n[i].vel))
            @test dpos < 1e-5
            @test dvel < 1e-5
        end
    end

    # ----------- Ω derived from |v|/|r| preserves initial state exactly
    @testset "pre-SN initial velocities preserved (no snap)" begin
        # BH masses sum to 2/3 (post-collapse point masses), but velocities
        # reflect the pre-SN inertial mass of 1 — Kepler relation
        #     v_rel = sqrt(M_inertial / a) = 1,  v1 = (M_star/M_tot)·v_rel = 2/3
        # Point-mass Kepler would give Ω = sqrt(2/3) ≈ 0.816; default
        # derivation should instead give Ω = |v1|/|r1| = (2/3)/(2/3) = 1.
        bh1 = BlackHole([+2/3, 0.0, 0.0], [0.0, +2/3, 0.0],
                        1/3, 1e-4, 1e6, 0.01)
        bh2 = BlackHole([-1/3, 0.0, 0.0], [0.0, -1/3, 0.0],
                        1/3, 1e-4, 1e6, 0.01)
        bhs = BlackHole[bh1, bh2]

        pos0 = [copy(bh.pos) for bh in bhs]
        vel0 = [copy(bh.vel) for bh in bhs]

        orbit = KeplerOrbit(bhs)
        @test isapprox(orbit.Ω, 1.0; atol=1e-12)

        # Applying kepler_update! at t=0 must reproduce the initial state
        # to machine precision (no velocity snap).
        kepler_update!(bhs, orbit, 0.0)
        for i in 1:2
            @test maximum(abs.(bhs[i].pos .- pos0[i])) < 1e-12
            @test maximum(abs.(bhs[i].vel .- vel0[i])) < 1e-12
        end
    end

    # ----------- Explicit Ω override
    @testset "explicit Ω keyword" begin
        bhs = make_pair()          # would default to Ω = 1
        orbit = KeplerOrbit(bhs; Ω = 0.5)
        @test orbit.Ω == 0.5

        # With Ω halved, velocities set by kepler_update! are halved too.
        kepler_update!(bhs, orbit, 0.0)
        @test isapprox(sqrt(sum(bhs[1].vel.^2)), 0.25; atol=1e-12)  # 0.5 × 0.5
        @test isapprox(sqrt(sum(bhs[2].vel.^2)), 0.25; atol=1e-12)
    end

    # ----------------------------- unequal-mass (asymmetric COM radii)
    @testset "unequal masses, COM offset" begin
        # M1 = 2/3, M2 = 1/3, a = 1. COM at origin.
        # r1 = (M2/M_tot) a = 1/3, r2 = (M1/M_tot) a = 2/3
        # v_rel = sqrt(M_tot/a) = 1, v1 = (M2/M_tot) v_rel = 1/3, v2 = 2/3.
        bh1 = BlackHole([+1/3, 0.0, 0.0], [0.0, +1/3, 0.0],
                        2/3, 1e-4, 1e6, 0.01)
        bh2 = BlackHole([-2/3, 0.0, 0.0], [0.0, -2/3, 0.0],
                        1/3, 1e-4, 1e6, 0.01)
        bhs = BlackHole[bh1, bh2]
        orbit = KeplerOrbit(bhs)

        @test isapprox(orbit.Ω, 1.0; atol=1e-12)

        # Quarter period → BH1 at (0, +1/3, 0), BH2 at (0, -2/3, 0)
        kepler_update!(bhs, orbit, 2π / 4 / orbit.Ω)
        @test isapprox(bhs[1].pos[1], 0.0;  atol=1e-12)
        @test isapprox(bhs[1].pos[2], 1/3; atol=1e-12)
        @test isapprox(bhs[2].pos[1], 0.0;  atol=1e-12)
        @test isapprox(bhs[2].pos[2], -2/3; atol=1e-12)

        # After quarter period, velocities point along −x (BH1) and +x (BH2)
        @test isapprox(bhs[1].vel[1], -1/3; atol=1e-12)
        @test isapprox(bhs[2].vel[1], +2/3; atol=1e-12)
    end

    # ------------------------------------- inclined orbit plane (nhat != ẑ)
    @testset "inclined plane: n̂ inferred correctly" begin
        # Equal-mass pair on the x-axis with velocities along ∓ẑ.  The
        # angular momentum L = r × v points along +ŷ (right-hand rule).
        bh1 = BlackHole([+0.5, 0.0, 0.0], [0.0, 0.0, -0.5],
                        0.5, 1e-4, 1e6, 0.01)
        bh2 = BlackHole([-0.5, 0.0, 0.0], [0.0, 0.0, +0.5],
                        0.5, 1e-4, 1e6, 0.01)
        bhs = BlackHole[bh1, bh2]
        orbit = KeplerOrbit(bhs)

        @test isapprox(orbit.nhat[1], 0.0; atol=1e-12)
        @test isapprox(orbit.nhat[2], 1.0; atol=1e-12)
        @test isapprox(orbit.nhat[3], 0.0; atol=1e-12)

        # π/2 rotation about +ŷ takes +x̂ → −ẑ.
        kepler_update!(bhs, orbit, 2π / 4 / orbit.Ω)
        @test isapprox(bhs[1].pos[1], 0.0;  atol=1e-12)
        @test isapprox(bhs[1].pos[2], 0.0;  atol=1e-12)
        @test isapprox(bhs[1].pos[3], -0.5; atol=1e-12)
    end

end
