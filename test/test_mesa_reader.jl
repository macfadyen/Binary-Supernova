# MESA reader tests: round-trip a minimal synthetic MESA-style profile
# through read_mesa_profile and verify the center→surface ordering,
# log-space decoding, and unit auto-detection for mass/radius.

@testset "read_mesa_profile: minimal synthetic profile (M☉/R☉ units)" begin
    # Construct a surface→centre table (as MESA writes), 5 zones.
    # Columns: zone, mass (M☉), radius (R☉), logRho, logT, logP
    tmp_dir = mktempdir()
    path = joinpath(tmp_dir, "profile1.data")
    open(path, "w") do io
        # MESA preamble is typically 5 header lines before the column names
        # line; we just need _is_numeric(tokens[1]) to be false on the
        # column-names line and for `i >= 4` to be true.
        println(io, "# star_age ...")
        println(io, "         1     1e6")
        println(io, "")
        println(io, "")
        println(io, "zone   mass   radius   logRho   logT   logP")
        # Surface-first rows (MESA convention). Five zones.
        println(io, "1   10.0   5.0    -6.0   4.0   10.0")
        println(io, "2    9.0   4.0    -3.0   5.0   13.0")
        println(io, "3    7.0   3.0    -1.0   6.5   16.0")
        println(io, "4    3.0   2.0     0.5   7.5   18.0")
        println(io, "5    0.1   0.1     1.5   8.0   19.0")
    end

    p = read_mesa_profile(path)
    @test p isa StellarProfile1D{Float64}
    @test p.n_shells == 5

    # After reversal: center-first.  Innermost entry (was zone 5) has
    # mass = 0.1 M☉ → converted to grams via M_SUN; radius = 0.1 R☉ → cm.
    @test p.mass_coord[1]   ≈ 0.1 * BinarySupernova.M_SUN rtol=1e-12
    @test p.mass_coord[end] ≈ 10.0 * BinarySupernova.M_SUN rtol=1e-12
    @test p.radius[1]       ≈ 0.1 * BinarySupernova.R_SUN rtol=1e-12
    @test p.radius[end]     ≈ 5.0 * BinarySupernova.R_SUN rtol=1e-12

    # logRho, logT, logP decoded (center first).
    @test p.rho[1]         ≈ 10.0^1.5 rtol=1e-12
    @test p.rho[end]       ≈ 10.0^-6.0 rtol=1e-12
    @test p.temperature[1] ≈ 10.0^8.0 rtol=1e-12
    @test p.pressure[1]    ≈ 10.0^19.0 rtol=1e-12

    # Monotonicity: center→surface should be increasing in radius and
    # decreasing in density for this synthetic profile.
    @test issorted(p.radius)
    @test issorted(p.rho; rev=true)
end

@testset "read_mesa_profile: CGS units (no auto-conversion)" begin
    # Mass in g, radius in cm — outer values large enough to skip the
    # M☉/R☉ auto-scaling branches.
    tmp_dir = mktempdir()
    path = joinpath(tmp_dir, "profile_cgs.data")
    open(path, "w") do io
        println(io, "# header")
        println(io, "  1  1e6")
        println(io, "")
        println(io, "")
        println(io, "zone   mass   radius   logRho   logT")
        println(io, "1   1.0e34   1.0e11   -6.0   4.0")
        println(io, "2   0.5e34   0.5e11    0.0   7.0")
    end
    p = read_mesa_profile(path)
    # Outer mass is 1e34 g > 1e10 → no M☉ scaling applied.
    @test p.mass_coord[end] ≈ 1.0e34 rtol=1e-12
    # Outer radius is 1e11 cm > 1e5 → no R☉ scaling applied.
    @test p.radius[end]     ≈ 1.0e11 rtol=1e-12
    # logP absent → fallback P = (a_rad/3) T⁴.
    a_rad = 7.5646e-15
    @test p.pressure[1] ≈ (a_rad/3.0) * 10.0^(4*7.0) rtol=1e-10
end

@testset "read_mesa_profile: error on missing required column" begin
    tmp_dir = mktempdir()
    path = joinpath(tmp_dir, "profile_bad.data")
    open(path, "w") do io
        println(io, "# header")
        println(io, "  1  1e6")
        println(io, "")
        println(io, "")
        println(io, "zone   mass   radius")
        println(io, "1   1.0   1.0")
    end
    @test_throws ErrorException read_mesa_profile(path)
end
