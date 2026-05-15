# 1D MESA stellar profile reader.
#
# Provides `StellarProfile1D` (radius, ρ, T, P, enclosed mass; CGS) and
# `read_mesa_profile(filename)` which parses MESA `profile*.data` files
# with column-alias fallback. Profiles are returned ordered center → surface.
#
# Synthetic profile generators are NOT included — Binary-Supernova's
# `polytrope_ic_3d!` (Phase 4) already covers synthetic polytropic ICs.
# Use `read_mesa_profile` for realistic progenitor structures.
#
# Ported from Binary-PISN/src/mesa_reader.jl (Heger-&-Woosley zones and
# pair-instability-specific synthetic profile stripped; they are not
# applicable to general core-collapse supernova progenitors).

"""
    StellarProfile1D{T}

One-dimensional stellar profile, ordered center → surface. All fields
are in CGS units.

Fields
- `n_shells`    : number of radial shells
- `mass_coord`  : enclosed mass [g]
- `radius`      : radial coordinate [cm]
- `rho`         : density [g cm⁻³]
- `temperature` : temperature [K]
- `pressure`    : pressure [dyn cm⁻² = erg cm⁻³]
"""
struct StellarProfile1D{T <: AbstractFloat}
    n_shells    :: Int
    mass_coord  :: Vector{T}
    radius      :: Vector{T}
    rho         :: Vector{T}
    temperature :: Vector{T}
    pressure    :: Vector{T}
end

# ---------------------------------------------------------------------------
# MESA profile reader with column-alias fallback.

"""
    read_mesa_profile(filename) -> StellarProfile1D{Float64}

Read a MESA `profile*.data` file into a `StellarProfile1D`. Accepts common
column aliases for the required columns and reverses the arrays to
center→surface order. Mass and radius units are auto-detected from the
outermost value magnitude: if outer mass < 1e10 it is interpreted as M☉
(converted to grams with `M_SUN`); if outer radius < 1e5 it is interpreted
as R☉ (converted to cm with `R_SUN`).

Required columns (with aliases tried in order):
  log ρ     : logRho, log_rho, logdensity, log_density
  log T     : logT,   log_t,   logtemp,    log_temp, logTeff
  mass      : mass, m, mass_grams, enclosed_mass

Optional columns:
  radius    : radius, r, radius_cm, log_r   (derived from mass/ρ if absent)
  log P     : logP, log_p, logpressure, log_pressure   (fallback: P = a_rad T⁴/3)
"""
function read_mesa_profile(filename::String)
    lines = readlines(filename)

    col_names_line = 0
    data_start = 0
    for (i, line) in enumerate(lines)
        tokens = split(strip(line))
        isempty(tokens) && continue
        if i >= 4 && !_is_numeric(tokens[1]) && length(tokens) > 3
            col_names_line = i
            data_start = i + 1
            break
        end
    end
    col_names_line == 0 && error("Could not find column names in MESA profile: $filename")
    col_names = split(strip(lines[col_names_line]))

    n_zones = 0
    data_rows = Vector{Vector{Float64}}()
    for i in data_start:length(lines)
        tokens = split(strip(lines[i]))
        isempty(tokens) && continue
        _is_numeric(tokens[1]) || continue
        push!(data_rows, parse.(Float64, tokens))
        n_zones += 1
    end
    n_zones == 0 && error("No data rows found in MESA profile: $filename")

    col_data = Dict{String,Vector{Float64}}()
    for (ci, name) in enumerate(col_names)
        col_data[lowercase(name)] = [data_rows[r][ci] for r in 1:n_zones]
    end

    logrho     = _get_mesa_col(col_data, ["logrho","log_rho","logdensity","log_density"])
    logT       = _get_mesa_col(col_data, ["logt","log_t","logtemp","log_temp","logteff"])
    mass       = _get_mesa_col(col_data, ["mass","m","mass_grams","enclosed_mass"])
    radius_col = _get_mesa_col(col_data, ["radius","r","radius_cm","log_r"]; required=false)
    logP_col   = _get_mesa_col(col_data, ["logp","log_p","logpressure","log_pressure"]; required=false)

    rho_out  = 10.0 .^ reverse(logrho)
    T_out    = 10.0 .^ reverse(logT)
    mass_out = reverse(mass)

    if radius_col !== nothing
        r_name = _find_mesa_colname(col_data, ["radius","r","radius_cm","log_r"])
        radius_out = startswith(r_name, "log") ? 10.0 .^ reverse(radius_col) : reverse(radius_col)
    else
        radius_out = ((3.0 ./ (4π)) .* mass_out ./ max.(rho_out, 1e-30)) .^ (1.0/3.0)
    end

    if mass_out[end] < 1.0e10;  mass_out   .*= M_SUN;  end
    if radius_out[end] < 1.0e5; radius_out .*= R_SUN;  end

    # a_rad = 4σ/c = 7.5646e-15 erg cm⁻³ K⁻⁴
    P_out = logP_col !== nothing ?
                10.0 .^ reverse(logP_col) :
                (7.5646e-15 / 3.0) .* T_out .^ 4

    return StellarProfile1D{Float64}(n_zones, mass_out, radius_out,
                                      rho_out, T_out, P_out)
end

# ---------------------------------------------------------------------------
# Helpers.

function _is_numeric(s::AbstractString)
    try parse(Float64, s); return true
    catch; return false
    end
end

function _get_mesa_col(col_data::Dict{String,Vector{Float64}},
                       aliases::Vector{String}; required::Bool = true)
    for alias in aliases
        key = lowercase(alias)
        haskey(col_data, key) && return col_data[key]
    end
    required && error("Required MESA column not found. Tried: " * join(aliases, ", "))
    return nothing
end

function _find_mesa_colname(col_data::Dict{String,Vector{Float64}},
                            aliases::Vector{String})
    for alias in aliases
        key = lowercase(alias)
        haskey(col_data, key) && return key
    end
    return ""
end
