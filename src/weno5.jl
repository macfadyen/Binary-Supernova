# WENO-Z reconstruction — Borges et al. (2008), based on Jiang & Shu (1996)
# Pure functions, CPU+GPU compatible (no allocations in kernel-callable form).
#
# weno5_reconstruct(v, i) → (vL, vR)
#   v : AbstractVector of cell averages (1-indexed, ghost cells included)
#   i : index of the interface between cell i and cell i+1
#   returns vL = reconstructed value from the left stencil,
#           vR = reconstructed value from the right stencil.
#
# Stencil indexing follows Jiang & Shu convention:
#   left reconstruction of interface i+1/2 uses v[i-2..i+2]
#   right reconstruction of interface i+1/2 uses v[i-1..i+3]

@inline function wenoz_weights(β0, β1, β2, d0, d1, d2, ε=1e-36)
    τ5 = abs(β0 - β2)          # global smoothness indicator (Borges et al. 2008)
    α0 = d0 * (1 + (τ5 / (ε + β0))^2)
    α1 = d1 * (1 + (τ5 / (ε + β1))^2)
    α2 = d2 * (1 + (τ5 / (ε + β2))^2)
    αsum = α0 + α1 + α2
    return α0/αsum, α1/αsum, α2/αsum
end

# Left-biased reconstruction at interface i+1/2.
# Uses stencil v[i-2], v[i-1], v[i], v[i+1], v[i+2].
@inline function weno5_left(vm2, vm1, v0, vp1, vp2)
    # Candidate stencil polynomials (3rd order each)
    q0 = ( 1/3)*vm2 + (-7/6)*vm1 + (11/6)*v0
    q1 = (-1/6)*vm1 + ( 5/6)*v0  + ( 1/3)*vp1
    q2 = ( 1/3)*v0  + ( 5/6)*vp1 + (-1/6)*vp2

    # Smoothness indicators
    β0 = (13/12)*(vm2 - 2*vm1 +  v0)^2 + (1/4)*(vm2 - 4*vm1 + 3*v0)^2
    β1 = (13/12)*(vm1 - 2*v0  + vp1)^2 + (1/4)*(vm1 -  vp1)^2
    β2 = (13/12)*(v0  - 2*vp1 + vp2)^2 + (1/4)*(3*v0 - 4*vp1 + vp2)^2

    # Ideal weights (left side)
    d0, d1, d2 = 1/10, 6/10, 3/10

    ω0, ω1, ω2 = wenoz_weights(β0, β1, β2, d0, d1, d2)
    return ω0*q0 + ω1*q1 + ω2*q2
end

# Right-biased reconstruction at interface i+1/2 (mirror of left).
# Uses stencil v[i-1], v[i], v[i+1], v[i+2], v[i+3].
@inline function weno5_right(vm1, v0, vp1, vp2, vp3)
    # Candidate stencil polynomials
    q0 = (-1/6)*vm1  + ( 5/6)*v0  + ( 1/3)*vp1
    q1 = ( 1/3)*v0   + ( 5/6)*vp1 + (-1/6)*vp2
    q2 = (11/6)*vp1  + (-7/6)*vp2 + ( 1/3)*vp3

    # Smoothness indicators (same formulas, mirrored stencil)
    β0 = (13/12)*(vm1 - 2*v0  + vp1)^2 + (1/4)*(vm1 - vp1)^2
    β1 = (13/12)*(v0  - 2*vp1 + vp2)^2 + (1/4)*(v0 - 4*vp1 + 3*vp2)^2
    β2 = (13/12)*(vp1 - 2*vp2 + vp3)^2 + (1/4)*(3*vp1 - 4*vp2 + vp3)^2

    # Ideal weights (right side — reversed)
    d0, d1, d2 = 3/10, 6/10, 1/10

    ω0, ω1, ω2 = wenoz_weights(β0, β1, β2, d0, d1, d2)
    return ω0*q0 + ω1*q1 + ω2*q2
end

"""
    weno5_reconstruct_interface(v, i)

Return `(vL, vR)` at interface `i+1/2` given cell-average array `v` (1-indexed).
Requires ghost cells: v[i-2..i+3] must be valid (i.e., `i` must satisfy 3 ≤ i ≤ length(v)-3).
"""
@inline function weno5_reconstruct_interface(v::AbstractVector, i::Int)
    vL = weno5_left( v[i-2], v[i-1], v[i],   v[i+1], v[i+2])
    vR = weno5_right(v[i-1], v[i],   v[i+1], v[i+2], v[i+3])
    return vL, vR
end

# ---------------------------------------------------------------------------
# 5th-order linear reconstruction (fixed optimal weights d_k, no WENO limiting)
#
# Used for high-order interface values in smooth flows (e.g. viscous stress σ).
# Identical to WENO5 in smooth regions but cheaper (no smoothness indicators).
#
# Left-biased at interface i+1/2, stencil {v_{i-2}..v_{i+2}}:
#   linrec5_left  = (2v1 - 13v2 + 47v3 + 27v4 - 3v5) / 60
#                   where (v1,v2,v3,v4,v5) = (v_{i-2},v_{i-1},v_i,v_{i+1},v_{i+2})
# Right-biased mirror (stencil {v_{i-1}..v_{i+3}}):
#   linrec5_right = (-3v1 + 27v2 + 47v3 - 13v4 + 2v5) / 60
#                   where (v1,v2,v3,v4,v5) = (v_{i-1},v_i,v_{i+1},v_{i+2},v_{i+3})

"""
    linrec5_left(v1, v2, v3, v4, v5) → value at i+1/2

5th-order left-biased linear reconstruction at interface i+1/2 using
stencil (v_{i-2}, v_{i-1}, v_i, v_{i+1}, v_{i+2}).
Uses fixed optimal WENO5 weights d = (1/10, 6/10, 3/10); no nonlinear limiting.
"""
@inline function linrec5_left(v1, v2, v3, v4, v5)
    return (2*v1 - 13*v2 + 47*v3 + 27*v4 - 3*v5) / 60
end

"""
    linrec5_right(v1, v2, v3, v4, v5) → value at i+1/2

5th-order right-biased linear reconstruction at interface i+1/2 using
stencil (v_{i-1}, v_i, v_{i+1}, v_{i+2}, v_{i+3}).
Mirror of `linrec5_left`; uses fixed optimal weights d = (3/10, 6/10, 1/10).
"""
@inline function linrec5_right(v1, v2, v3, v4, v5)
    return (-3*v1 + 27*v2 + 47*v3 - 13*v4 + 2*v5) / 60
end
