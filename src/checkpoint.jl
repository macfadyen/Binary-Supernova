# HDF5 checkpoint/restart for FMR + N-body state.
#
# Layout (single HDF5 file):
#
#   /manifest
#       attrs: format_version (Int), t (Float64), step (Int64), dt_last (Float64)
#
#   /grid
#       attrs: ratio, gamma, bc ("outflow"|"periodic"),
#              rho_floor, P_floor,
#              ci_lo, ci_hi, cj_lo, cj_hi, ck_lo, ck_hi
#       /coarse
#           attrs: nx, ny, nz, dx, dy, dz
#           dataset: U  (NQ, nx+2NG, ny+2NG, nz+2NG)
#       /fine
#           attrs: nx, ny, nz, dx, dy, dz
#           dataset: U  (NQ, nx+2NG, ny+2NG, nz+2NG)
#
#   /bhs
#       attrs: n
#       datasets (only if n > 0):
#           pos  (3, n)   vel  (3, n)
#           mass (n)      eps  (n)
#           c_code (n)    r_floor (n)
#
# SimParams and PhysicalUnits are not serialized — they are driver
# configuration (re-supplied on restart).
#
# Ported from Binary-PISN/src/checkpoint.jl (Phase 18).

const CHECKPOINT_FORMAT_VERSION = 1

"""
    save_checkpoint(path, G, bhs; t=0.0, step=0, dt_last=0.0)

Write the full FMR grid `G` and `bhs` list to `path` (HDF5). Overwrites
any existing file at `path`.
"""
function save_checkpoint(path::AbstractString,
                         G::FMRGrid3D,
                         bhs::Vector{BlackHole};
                         t::Real = 0.0,
                         step::Integer = 0,
                         dt_last::Real = 0.0)
    HDF5.h5open(path, "w") do f
        m = HDF5.create_group(f, "manifest")
        HDF5.attrs(m)["format_version"] = CHECKPOINT_FORMAT_VERSION
        HDF5.attrs(m)["t"]               = Float64(t)
        HDF5.attrs(m)["step"]            = Int64(step)
        HDF5.attrs(m)["dt_last"]         = Float64(dt_last)

        g = HDF5.create_group(f, "grid")
        HDF5.attrs(g)["ratio"]     = Int64(G.ratio)
        HDF5.attrs(g)["gamma"]     = G.γ
        HDF5.attrs(g)["bc"]        = String(G.bc)
        HDF5.attrs(g)["rho_floor"] = G.ρ_floor
        HDF5.attrs(g)["P_floor"]   = G.P_floor
        HDF5.attrs(g)["ci_lo"]     = Int64(G.ci_lo)
        HDF5.attrs(g)["ci_hi"]     = Int64(G.ci_hi)
        HDF5.attrs(g)["cj_lo"]     = Int64(G.cj_lo)
        HDF5.attrs(g)["cj_hi"]     = Int64(G.cj_hi)
        HDF5.attrs(g)["ck_lo"]     = Int64(G.ck_lo)
        HDF5.attrs(g)["ck_hi"]     = Int64(G.ck_hi)

        _write_level(g, "coarse", G.coarse)
        _write_level(g, "fine",   G.fine)

        b = HDF5.create_group(f, "bhs")
        n = length(bhs)
        HDF5.attrs(b)["n"] = Int64(n)
        if n > 0
            pos     = zeros(Float64, 3, n)
            vel     = zeros(Float64, 3, n)
            mass    = zeros(Float64, n)
            eps_arr = zeros(Float64, n)
            c_code  = zeros(Float64, n)
            r_floor = zeros(Float64, n)
            for (i, bh) in enumerate(bhs)
                pos[:, i]   .= bh.pos
                vel[:, i]   .= bh.vel
                mass[i]      = bh.mass
                eps_arr[i]   = bh.eps
                c_code[i]    = bh.c_code
                r_floor[i]   = bh.r_floor
            end
            b["pos"]     = pos
            b["vel"]     = vel
            b["mass"]    = mass
            b["eps"]     = eps_arr
            b["c_code"]  = c_code
            b["r_floor"] = r_floor
        end
    end
    return path
end

function _write_level(parent, name::String, lvl::FMRLevel3D)
    sub = HDF5.create_group(parent, name)
    HDF5.attrs(sub)["nx"] = Int64(lvl.nx)
    HDF5.attrs(sub)["ny"] = Int64(lvl.ny)
    HDF5.attrs(sub)["nz"] = Int64(lvl.nz)
    HDF5.attrs(sub)["dx"] = lvl.dx
    HDF5.attrs(sub)["dy"] = lvl.dy
    HDF5.attrs(sub)["dz"] = lvl.dz
    sub["U"] = lvl.U
    return sub
end

"""
    load_checkpoint(path) -> (G, bhs, meta)

Read a checkpoint written by [`save_checkpoint`](@ref) and reconstruct
the FMR grid and BH list. Returns `(G, bhs, meta)` where `meta` is a
`NamedTuple` `(; t, step, dt_last, format_version)`.

The returned `G.fine.U` is the saved fine-level state (not re-prolonged
from the coarse level), so the restart is bit-for-bit faithful.
"""
function load_checkpoint(path::AbstractString)
    HDF5.h5open(path, "r") do f
        m  = f["manifest"]
        fv = HDF5.read_attribute(m, "format_version")
        fv == CHECKPOINT_FORMAT_VERSION || error(
            "checkpoint format_version=$fv, expected $CHECKPOINT_FORMAT_VERSION")
        t       = HDF5.read_attribute(m, "t")
        step    = HDF5.read_attribute(m, "step")
        dt_last = HDF5.read_attribute(m, "dt_last")

        g       = f["grid"]
        ratio   = Int(HDF5.read_attribute(g, "ratio"))
        γ       = HDF5.read_attribute(g, "gamma")
        bc      = Symbol(HDF5.read_attribute(g, "bc"))
        ρ_floor = HDF5.read_attribute(g, "rho_floor")
        P_floor = HDF5.read_attribute(g, "P_floor")
        ci_lo   = Int(HDF5.read_attribute(g, "ci_lo"))
        ci_hi   = Int(HDF5.read_attribute(g, "ci_hi"))
        cj_lo   = Int(HDF5.read_attribute(g, "cj_lo"))
        cj_hi   = Int(HDF5.read_attribute(g, "cj_hi"))
        ck_lo   = Int(HDF5.read_attribute(g, "ck_lo"))
        ck_hi   = Int(HDF5.read_attribute(g, "ck_hi"))

        coarse = _read_level(g["coarse"])
        G = FMRGrid3D(coarse, ci_lo, ci_hi, cj_lo, cj_hi, ck_lo, ck_hi,
                      ratio, γ; bc=bc, ρ_floor=ρ_floor, P_floor=P_floor)
        fine_saved = _read_level(g["fine"])
        size(fine_saved.U) == size(G.fine.U) || error(
            "checkpoint fine-level shape $(size(fine_saved.U)) does not " *
            "match reconstructed shape $(size(G.fine.U))")
        copyto!(G.fine.U, fine_saved.U)

        b = f["bhs"]
        n = Int(HDF5.read_attribute(b, "n"))
        bhs = BlackHole[]
        if n > 0
            pos     = read(b["pos"])
            vel     = read(b["vel"])
            mass    = read(b["mass"])
            eps_arr = read(b["eps"])
            c_code  = read(b["c_code"])
            r_floor_arr = read(b["r_floor"])
            for i in 1:n
                push!(bhs, BlackHole(Vector{Float64}(pos[:, i]),
                                     Vector{Float64}(vel[:, i]),
                                     mass[i], eps_arr[i], c_code[i],
                                     r_floor_arr[i]))
            end
        end

        meta = (; t, step, dt_last, format_version = fv)
        return G, bhs, meta
    end
end

function _read_level(grp)
    nx = Int(HDF5.read_attribute(grp, "nx"))
    ny = Int(HDF5.read_attribute(grp, "ny"))
    nz = Int(HDF5.read_attribute(grp, "nz"))
    dx = HDF5.read_attribute(grp, "dx")
    dy = HDF5.read_attribute(grp, "dy")
    dz = HDF5.read_attribute(grp, "dz")
    U  = read(grp["U"])
    lvl = FMRLevel3D(nx, ny, nz, dx, dy, dz)
    copyto!(lvl.U, U)
    return lvl
end
