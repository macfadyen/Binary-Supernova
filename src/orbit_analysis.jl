# Orbit-element analyzer: reduce two BH positions/velocities/masses to
# Keplerian orbital elements (separation, eccentricity, semi-major axis,
# specific energy + angular momentum, period, peri/apocenter).
#
# Pure post-processing. Two input paths:
#   - HDF5 trajectory file produced by `init_trajectory_file` + `append_trajectory`
#   - CSV file with columns `step, t, bh{k}_x/y/z/vx/vy/vz/m`
#
# All elements are computed in the same units as the inputs. For a code-units
# two-body system with G=1, the reduced gravitational parameter is μ = m1 + m2.
#
# Ported from Binary-PISN/src/orbit_analysis.jl (Phase 22).

# 3-vector helpers (avoid pulling in LinearAlgebra for three ops).
@inline _dot3(a, b)   = a[1]*b[1] + a[2]*b[2] + a[3]*b[3]
@inline _norm3(a)     = sqrt(_dot3(a, a))
@inline _cross3(a, b) = (a[2]*b[3] - a[3]*b[2],
                         a[3]*b[1] - a[1]*b[3],
                         a[1]*b[2] - a[2]*b[1])
@inline _sub3(a, b)   = (a[1]-b[1], a[2]-b[2], a[3]-b[3])

"""
    orbit_elements(r1, v1, m1, r2, v2, m2; G_grav=1.0) -> NamedTuple

Reduce a two-body state to relative-frame Keplerian orbital elements.
`r1`, `r2` are 3-element position vectors; `v1`, `v2` are velocity
vectors; `m1`, `m2` are masses. Units are caller-defined (code units
with `G_grav=1.0` by default).

Returns `(; r, v, ε, L, L_vec, e_vec, a, e, T_orb, r_peri, r_apo, bound)`:

- `r`, `v`        — relative separation magnitude / speed
- `ε`             — specific orbital energy (½v² − μ/r)
- `L`, `L_vec`    — magnitude / vector of specific angular momentum
- `e_vec`, `e`    — Laplace–Runge–Lenz eccentricity vector / magnitude
- `a`             — semi-major axis (signed: positive for bound)
- `T_orb`         — Keplerian period (Inf for unbound)
- `r_peri`, `r_apo` — peri- and apocenter
- `bound::Bool`   — true iff `ε < 0`

Reduces to a single relative two-body problem with reduced gravitational
parameter `μ = G_grav · (m1 + m2)`.
"""
function orbit_elements(r1, v1, m1::Real,
                        r2, v2, m2::Real;
                        G_grav::Real = 1.0)
    μ   = Float64(G_grav) * (Float64(m1) + Float64(m2))
    rr  = _sub3(r2, r1)
    vv  = _sub3(v2, v1)
    r   = _norm3(rr)
    v²  = _dot3(vv, vv)
    Lv  = _cross3(rr, vv)
    L   = _norm3(Lv)
    ε   = 0.5 * v² - μ / r
    a   = -μ / (2.0 * ε)
    vxL = _cross3(vv, Lv)
    ev  = (vxL[1]/μ - rr[1]/r,
           vxL[2]/μ - rr[2]/r,
           vxL[3]/μ - rr[3]/r)
    e   = _norm3(ev)
    bound  = ε < 0
    T_orb  = bound ? 2π * sqrt(a^3 / μ) : Inf
    r_peri = bound ? a * (1.0 - e) : NaN
    r_apo  = bound ? a * (1.0 + e) : NaN
    return (; r, v = sqrt(v²), ε, L, L_vec = Lv, e_vec = ev,
              a, e, T_orb, r_peri, r_apo, bound)
end

# ---------------------------------------------------------------------------
# HDF5 trajectory reader — uses the format written by init_trajectory_file /
# append_trajectory in io.jl.

"""
    orbit_elements_from_trajectory(filename; G_grav=1.0, bh_indices=(1,2)) -> NamedTuple

Apply [`orbit_elements`](@ref) to every time record in an HDF5 trajectory
file written by [`init_trajectory_file`](@ref) + [`append_trajectory`](@ref)
for the BH pair `bh_indices = (i, j)` (default `(1, 2)`).

Returns column vectors of the scalar elements plus `t`:

`(; t, r, v, ε, L, a, e, T_orb, r_peri, r_apo, bound)`

Errors if the file does not contain `/bh{i}` and `/bh{j}` groups.
"""
function orbit_elements_from_trajectory(filename::AbstractString;
                                         G_grav::Real = 1.0,
                                         bh_indices::Tuple{Int, Int} = (1, 2))
    i, j = bh_indices
    HDF5.h5open(filename, "r") do fid
        haskey(fid, "bh$i") || error("orbit_elements_from_trajectory: /bh$i missing in $filename")
        haskey(fid, "bh$j") || error("orbit_elements_from_trajectory: /bh$j missing in $filename")
        t    = HDF5.read(fid["time"])
        pos1 = HDF5.read(fid["bh$i/pos"]);  vel1 = HDF5.read(fid["bh$i/vel"]);  m1 = HDF5.read(fid["bh$i/mass"])
        pos2 = HDF5.read(fid["bh$j/pos"]);  vel2 = HDF5.read(fid["bh$j/vel"]);  m2 = HDF5.read(fid["bh$j/mass"])
        n = length(t)
        r      = Vector{Float64}(undef, n)
        v      = Vector{Float64}(undef, n)
        ε      = Vector{Float64}(undef, n)
        L      = Vector{Float64}(undef, n)
        a      = Vector{Float64}(undef, n)
        e      = Vector{Float64}(undef, n)
        T_orb  = Vector{Float64}(undef, n)
        r_peri = Vector{Float64}(undef, n)
        r_apo  = Vector{Float64}(undef, n)
        bound  = Vector{Bool}(undef, n)
        for k in 1:n
            r1 = (pos1[k, 1], pos1[k, 2], pos1[k, 3])
            v1 = (vel1[k, 1], vel1[k, 2], vel1[k, 3])
            r2 = (pos2[k, 1], pos2[k, 2], pos2[k, 3])
            v2 = (vel2[k, 1], vel2[k, 2], vel2[k, 3])
            el = orbit_elements(r1, v1, m1[k], r2, v2, m2[k]; G_grav = G_grav)
            r[k]      = el.r
            v[k]      = el.v
            ε[k]      = el.ε
            L[k]      = el.L
            a[k]      = el.a
            e[k]      = el.e
            T_orb[k]  = el.T_orb
            r_peri[k] = el.r_peri
            r_apo[k]  = el.r_apo
            bound[k]  = el.bound
        end
        return (; t, r, v, ε, L, a, e, T_orb, r_peri, r_apo, bound)
    end
end

# ---------------------------------------------------------------------------
# CSV reader — for per-step CSV streams (e.g. from custom production drivers).

"""
    parse_run_csv(path) -> NamedTuple

Read a CSV with a header row and numeric data rows into a column-keyed
dict. Returns `(; header, n_rows, columns)` where `columns::Dict{String,
Vector{Float64}}` is keyed by the header field names.
"""
function parse_run_csv(path::AbstractString)
    lines = readlines(path)
    isempty(lines) && error("parse_run_csv: empty file at $path")
    header = String.(split(strip(lines[1]), ","))
    n_rows = length(lines) - 1
    columns = Dict{String, Vector{Float64}}()
    for h in header
        columns[h] = Vector{Float64}(undef, n_rows)
    end
    for (r, line) in enumerate(lines[2:end])
        vals = split(strip(line), ",")
        length(vals) == length(header) || error(
            "parse_run_csv: row $r has $(length(vals)) fields, expected $(length(header))")
        for (k, v) in zip(header, vals)
            columns[k][r] = parse(Float64, v)
        end
    end
    return (; header, n_rows, columns)
end

@inline function _row_bh_state(cols::Dict{String, Vector{Float64}}, i::Int, k::Int)
    pre = "bh$(k)_"
    return ((cols[pre*"x"][i],  cols[pre*"y"][i],  cols[pre*"z"][i]),
            (cols[pre*"vx"][i], cols[pre*"vy"][i], cols[pre*"vz"][i]),
            cols[pre*"m"][i])
end

"""
    orbit_elements_from_csv(path; G_grav=1.0, bh_indices=(1,2)) -> NamedTuple

Apply [`orbit_elements`](@ref) to every row of a CSV with columns
`step, t, bh{k}_x/y/z/vx/vy/vz/m` for the BH pair `bh_indices = (i, j)`
(default `(1, 2)`). Returns column vectors of every scalar element plus
`step` and `t`.
"""
function orbit_elements_from_csv(path::AbstractString;
                                  G_grav::Real = 1.0,
                                  bh_indices::Tuple{Int, Int} = (1, 2))
    p = parse_run_csv(path)
    cols = p.columns
    i, j = bh_indices
    for k in (i, j), f in ("x","y","z","vx","vy","vz","m")
        haskey(cols, "bh$(k)_$(f)") || error(
            "orbit_elements_from_csv: missing column bh$(k)_$(f) in $path")
    end
    n = p.n_rows
    r       = Vector{Float64}(undef, n)
    v       = Vector{Float64}(undef, n)
    ε       = Vector{Float64}(undef, n)
    L       = Vector{Float64}(undef, n)
    a       = Vector{Float64}(undef, n)
    e       = Vector{Float64}(undef, n)
    T_orb   = Vector{Float64}(undef, n)
    r_peri  = Vector{Float64}(undef, n)
    r_apo   = Vector{Float64}(undef, n)
    bound   = Vector{Bool}(undef, n)
    for k in 1:n
        r1, v1, m1 = _row_bh_state(cols, k, i)
        r2, v2, m2 = _row_bh_state(cols, k, j)
        el = orbit_elements(r1, v1, m1, r2, v2, m2; G_grav = G_grav)
        r[k]      = el.r
        v[k]      = el.v
        ε[k]      = el.ε
        L[k]      = el.L
        a[k]      = el.a
        e[k]      = el.e
        T_orb[k]  = el.T_orb
        r_peri[k] = el.r_peri
        r_apo[k]  = el.r_apo
        bound[k]  = el.bound
    end
    step = Int.(cols["step"])
    t    = cols["t"]
    return (; step, t, r, v, ε, L, a, e, T_orb, r_peri, r_apo, bound)
end
