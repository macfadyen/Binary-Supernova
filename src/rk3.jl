# SSP-RK3 integrator — Shu-Osher form (Shu & Osher 1988).
#
# rk3_step!(U, L!, dt, tmp1, tmp2)
#   U    : current state array (modified in-place → uⁿ⁺¹)
#   L!   : in-place spatial operator: L!(dU, U) fills dU = L(U)
#   dt   : time step
#   tmp1, tmp2 : pre-allocated work arrays of same size as U
#
# Shu-Osher stages:
#   u⁽¹⁾ = uⁿ + dt*L(uⁿ)
#   u⁽²⁾ = 3/4*uⁿ + 1/4*u⁽¹⁾ + 1/4*dt*L(u⁽¹⁾)
#   uⁿ⁺¹ = 1/3*uⁿ + 2/3*u⁽²⁾ + 2/3*dt*L(u⁽²⁾)

function rk3_step!(U, L!, dt, tmp1, tmp2)
    Un = tmp2      # save uⁿ
    Un .= U

    # Stage 1: u⁽¹⁾ = uⁿ + dt*L(uⁿ)
    L!(tmp1, U)
    @. U = Un + dt*tmp1

    # Stage 2: u⁽²⁾ = 3/4 uⁿ + 1/4 u⁽¹⁾ + 1/4 dt*L(u⁽¹⁾)
    L!(tmp1, U)
    @. U = 0.75*Un + 0.25*U + 0.25*dt*tmp1

    # Stage 3: uⁿ⁺¹ = 1/3 uⁿ + 2/3 u⁽²⁾ + 2/3 dt*L(u⁽²⁾)
    L!(tmp1, U)
    @. U = (1/3)*Un + (2/3)*U + (2/3)*dt*tmp1

    return nothing
end
