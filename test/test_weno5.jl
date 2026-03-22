@testset "WENO5 reconstruction" begin
    # WENO5 reconstructs an interface *value* from cell *averages*.
    # For any polynomial of degree ≤ 5, the reconstruction is exact (up to rounding).
    #
    # Test 1: linear function f(x) = x.
    # Cell averages == cell-center values for linear f, so v[i] = x_i.
    # Interface at i+1/2 = 0.5 (between cells at x=0 and x=1).
    xs = Float64.(-2:3)
    v  = copy(xs)   # f(x) = x: cell avg = pointwise value
    # 1-indexed: v[3] = 0.0 (x=0), v[4] = 1.0 (x=1); interface at x=0.5
    vL, vR = weno5_reconstruct_interface(v, 3)
    @test abs(vL - 0.5) < 1e-13
    @test abs(vR - 0.5) < 1e-13

    # Test 2: quadratic f(x) = x², using proper cell averages.
    # Cell avg of x² over [i-1/2, i+1/2] = i² + 1/12  (with Δx = 1).
    v2 = xs.^2 .+ 1/12
    # Interface between x=0 and x=1 → exact value f(0.5) = 0.25
    vL2, vR2 = weno5_reconstruct_interface(v2, 3)
    @test abs(vL2 - 0.25) < 1e-13
    @test abs(vR2 - 0.25) < 1e-13
end
