# Orbit-analysis tests: synthetic Keplerian two-body states → verify
# orbit_elements returns the correct semi-major axis, eccentricity,
# period, energy sign, etc.

@testset "orbit_elements: circular orbit (code units, G=1)" begin
    # Two equal masses m = 0.5, separation a = 1, circular orbit in the x-y
    # plane. In the centre-of-mass frame each mass moves on a circle of
    # radius a/2 with speed v_c = 0.5 * sqrt(μ / a) = 0.5 * sqrt(1) = 0.5,
    # giving a relative speed sqrt(μ/a) = 1.
    m1 = 0.5;  m2 = 0.5
    r1 = (-0.5, 0.0, 0.0);  v1 = (0.0, -0.5, 0.0)
    r2 = ( 0.5, 0.0, 0.0);  v2 = (0.0, +0.5, 0.0)

    el = orbit_elements(r1, v1, m1, r2, v2, m2; G_grav = 1.0)
    @test el.r     ≈ 1.0      rtol=1e-12
    @test el.a     ≈ 1.0      rtol=1e-12
    @test el.e     < 1e-10
    @test el.bound == true
    @test el.T_orb ≈ 2π       rtol=1e-12     # μ=1, a=1 → T = 2π
    @test el.r_peri ≈ 1.0     rtol=1e-10
    @test el.r_apo  ≈ 1.0     rtol=1e-10
    # Specific orbital energy ε = -μ/(2a) = -0.5
    @test el.ε     ≈ -0.5     rtol=1e-12
end

@testset "orbit_elements: unbound (hyperbolic)" begin
    # Radial relative velocity exceeds escape speed v_esc = sqrt(2μ/r).
    m1 = 0.5;  m2 = 0.5        # μ = 1
    r1 = (0.0, 0.0, 0.0)
    r2 = (1.0, 0.0, 0.0)        # r = 1 → v_esc = sqrt(2) ≈ 1.414
    v1 = (0.0, 0.0, 0.0)
    v2 = (2.0, 0.0, 0.0)        # v_rel = 2.0 > v_esc

    el = orbit_elements(r1, v1, m1, r2, v2, m2; G_grav = 1.0)
    @test el.bound == false
    @test el.ε     > 0.0
    @test el.T_orb == Inf
    @test isnan(el.r_peri)
    @test isnan(el.r_apo)
end

@testset "orbit_elements: eccentric orbit" begin
    # Place at apocenter of an a=1, e=0.5 ellipse.
    # r_apo = a(1+e) = 1.5, v_apo = sqrt(μ (1-e)/(a(1+e))) = sqrt(1/3).
    a_ref = 1.0;  e_ref = 0.5
    μ     = 1.0
    r_apo = a_ref * (1.0 + e_ref)
    v_apo = sqrt(μ * (1.0 - e_ref) / (a_ref * (1.0 + e_ref)))

    m1 = 0.5;  m2 = 0.5
    r1 = (0.0,   0.0, 0.0)
    r2 = (r_apo, 0.0, 0.0)
    v1 = (0.0, 0.0, 0.0)
    v2 = (0.0, v_apo, 0.0)

    el = orbit_elements(r1, v1, m1, r2, v2, m2; G_grav = 1.0)
    @test el.a     ≈ a_ref rtol=1e-10
    @test el.e     ≈ e_ref rtol=1e-10
    @test el.r_peri ≈ a_ref * (1.0 - e_ref) rtol=1e-10
    @test el.r_apo  ≈ r_apo                 rtol=1e-10
    @test el.bound == true
end

@testset "orbit_elements_from_trajectory: HDF5 round-trip" begin
    units = PhysicalUnits(20.0, 10.0)
    # Two BHs at circular-orbit setup.
    bh1 = BlackHole(pos=[-0.5, 0.0, 0.0], vel=[0.0, -0.5, 0.0],
                    mass=0.5, eps=1e-3, units=units, r_floor=1e-3)
    bh2 = BlackHole(pos=[+0.5, 0.0, 0.0], vel=[0.0, +0.5, 0.0],
                    mass=0.5, eps=1e-3, units=units, r_floor=1e-3)

    traj_path = joinpath(mktempdir(), "traj.h5")
    init_trajectory_file(traj_path, 2)
    for (k, t) in enumerate((0.0, 0.1, 0.2))
        append_trajectory(traj_path, t, [bh1, bh2])
    end

    el = orbit_elements_from_trajectory(traj_path; G_grav = 1.0)
    @test length(el.t) == 3
    @test all(el.bound)
    @test all(abs.(el.a .- 1.0) .< 1e-10)
    @test all(el.e .< 1e-10)
end

@testset "parse_run_csv + orbit_elements_from_csv" begin
    # Write a minimal CSV with the expected column naming.
    path = joinpath(mktempdir(), "run.csv")
    open(path, "w") do io
        println(io, "step,t,bh1_x,bh1_y,bh1_z,bh1_vx,bh1_vy,bh1_vz,bh1_m,bh2_x,bh2_y,bh2_z,bh2_vx,bh2_vy,bh2_vz,bh2_m")
        println(io, "0,0.0,-0.5,0.0,0.0,0.0,-0.5,0.0,0.5,0.5,0.0,0.0,0.0,0.5,0.0,0.5")
        println(io, "1,0.1,-0.5,0.0,0.0,0.0,-0.5,0.0,0.5,0.5,0.0,0.0,0.0,0.5,0.0,0.5")
    end
    p = parse_run_csv(path)
    @test p.n_rows == 2
    @test p.columns["bh1_m"] == [0.5, 0.5]

    el = orbit_elements_from_csv(path; G_grav = 1.0)
    @test el.step == [0, 1]
    @test el.a     ≈ [1.0, 1.0] rtol=1e-10
    @test all(el.e .< 1e-10)
    @test all(el.bound)
end
