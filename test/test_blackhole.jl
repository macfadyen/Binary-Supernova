@testset "BlackHole struct and sink radii" begin
    # PhysicalUnits: 20 M☉ total, 10 R☉ separation — typical pre-SN compact binary.
    units = PhysicalUnits(20.0, 10.0)

    # Speed of light in code units should be >> 1 (non-relativistic regime).
    # For 20 M_sun, 10 R_sun: v_unit ~ 618 km/s -> c_code ~ 485.
    @test units.c_code > 100.0

    # Unit sanity: v_unit should be hundreds of km/s for a close massive binary.
    # v_unit = sqrt(G * 20 M_sun / 10 R_sun) ≈ 618 km/s
    v_km_s = units.v_unit / 1e5
    @test 100.0 < v_km_s < 1000.0

    # Construct a BH with mass = 0.5 (half total), reasonable softening.
    bh = BlackHole(
        pos     = [0.0, 0.0, 0.0],
        vel     = [0.0, 0.5, 0.0],
        mass    = 0.5,
        eps     = 1e-3,
        units   = units,
        r_floor = 1e-3,
    )
    @test bh.mass   == 0.5
    @test bh.c_code == units.c_code

    # r_sink: ISCO = 6 M / c² is sub-grid; floor should dominate.
    rs_isco  = 6.0 * bh.mass / bh.c_code^2
    @test rs_isco < bh.r_floor                # ISCO is always sub-grid
    @test r_sink(bh) == bh.r_floor            # floor is operative

    # After mass growth, ISCO still sub-grid (floor still wins) unless c_code is tiny.
    bh.mass = 0.9
    @test r_sink(bh) == bh.r_floor

    # With an artificially small c_code the ISCO formula kicks in.
    bh_nr = BlackHole([0.,0.,0.], [0.,0.,0.], 0.5, 1e-4, 1.0, 1e-4)  # c_code = 1
    rs_nr  = r_sink(bh_nr)
    @test rs_nr ≈ 6.0 * 0.5 / 1.0^2          # = 3.0, well above floor 1e-4
    @test rs_nr > bh_nr.r_floor

    # t_sink: should be positive and scale as r_sink^{3/2} / sqrt(M).
    ts = t_sink(bh_nr)
    @test ts > 0.0
    # t_ff at r_sink = sqrt(r_sink^3 / (2 M)) ; t_sink = f_sink * t_ff
    r  = r_sink(bh_nr)
    expected_ts = 1.0 * r / sqrt(2.0 * bh_nr.mass / r)
    @test ts ≈ expected_ts rtol=1e-12
end

@testset "SimParams construction" begin
    p = SimParams(
        M_BH1      = 0.5,
        M_star     = 0.5,
        M_BH2_init = 0.1,
        E_SN       = 0.01,
        r_bomb     = 0.4,
        gamma      = 4/3,
    )
    @test M_ejecta(p) ≈ 0.4
    @test p.torque_free == true     # default
    @test p.cfl == 0.4             # default

    # M_BH2_init ≥ M_star should throw.
    @test_throws AssertionError SimParams(
        M_BH1      = 0.5,
        M_star     = 0.5,
        M_BH2_init = 0.6,   # larger than M_star — invalid
        E_SN       = 0.01,
        r_bomb     = 0.4,
    )
end

@testset "PhysicalUnits conversion" begin
    # Check that G = 1 in code units by construction.
    # G_code = G_CGS * M_unit / (L_unit * v_unit²)
    # With M_unit = M_total, L_unit = a0, v_unit = sqrt(G M / a0):
    #   G_code = G * M / (a0 * G*M/a0) = 1  ✓
    u = PhysicalUnits(10.0, 5.0)
    G_code = BinarySupernova.G_CGS * u.M_unit / (u.L_unit * u.v_unit^2)
    @test G_code ≈ 1.0 rtol=1e-12

    # c_code = C_CGS / v_unit
    @test u.c_code ≈ BinarySupernova.C_CGS / u.v_unit rtol=1e-12
end
