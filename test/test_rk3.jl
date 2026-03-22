@testset "SSP-RK3 integrator" begin
    # Test 1: constant RHS — state must not change.
    U    = [1.0, 2.0, 3.0]
    tmp1 = similar(U)
    tmp2 = similar(U)
    L_zero!(dU, U) = fill!(dU, 0.0)
    rk3_step!(U, L_zero!, 0.1, tmp1, tmp2)
    @test U ≈ [1.0, 2.0, 3.0]

    # Test 2: scalar exponential decay  dy/dt = -y, y(0) = 1 → y(t) = e^{-t}.
    # SSP-RK3 is 3rd-order; error should be O(dt³).
    # Check that error at t=1 with dt=0.01 is small (< 1e-5).
    function integrate_decay(dt)
        y   = [1.0]
        t1  = [0.0]
        tmp1 = similar(y)
        tmp2 = similar(y)
        L!(dU, U) = (dU .= .-U)
        n = round(Int, 1.0 / dt)
        for _ in 1:n
            rk3_step!(y, L!, dt, tmp1, tmp2)
        end
        return y[1]
    end

    y_dt01  = integrate_decay(0.1)
    y_dt001 = integrate_decay(0.01)
    exact   = exp(-1.0)

    err_coarse = abs(y_dt01  - exact)
    err_fine   = abs(y_dt001 - exact)

    # Error should be O(dt³): refine by 10× → error drops by ~1000×
    @test err_fine < 1e-6
    @test err_coarse / err_fine > 100   # confirms > 2nd-order convergence

    # Test 3: conservation — RHS = 0 leaves array unchanged to machine precision.
    # Note: SSP-RK3 computes 0.75*U + 0.25*U etc., which may introduce 1-ULP rounding.
    U2      = randn(5)
    U2_orig = copy(U2)
    tmp1    = similar(U2)
    tmp2    = similar(U2)
    rk3_step!(U2, L_zero!, 1.0, tmp1, tmp2)
    @test U2 ≈ U2_orig rtol=1e-14
end
