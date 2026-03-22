# HDF5 snapshot and BH trajectory I/O for BinarySupernova.
#
# Snapshot layout (one HDF5 file per dump):
#   /gas/rho     — density array   (nx, ny, nz)
#   /gas/rhovx   — x-momentum      (nx, ny, nz)
#   /gas/rhovy   — y-momentum      (nx, ny, nz)
#   /gas/rhovz   — z-momentum      (nx, ny, nz)
#   /gas/E       — total energy    (nx, ny, nz)
#   /attrs       — scalar metadata (t, nx, ny, nz, dx, dy, dz, gamma)
#
# BH trajectory file (one HDF5 per run, datasets extended each dump):
#   /time        — simulation time     (N_dumps,)
#   /bh{i}/mass  — BH i mass          (N_dumps,)
#   /bh{i}/pos   — BH i position      (N_dumps, 3)
#   /bh{i}/vel   — BH i velocity      (N_dumps, 3)
#   /bh{i}/rsink — BH i sink radius   (N_dumps,)

using HDF5

# ---------------------------------------------------------------------------
# Snapshot I/O

"""
    write_snapshot(filename, U, nx, ny, nz, dx, dy, dz, t, γ)

Write the active-cell conserved-variable array to an HDF5 snapshot file.
Active cells are `U[:, ng+1:ng+nx, ng+1:ng+ny, ng+1:ng+nz]`.
"""
function write_snapshot(filename::String, U,
                         nx::Int, ny::Int, nz::Int,
                         dx::Real, dy::Real, dz::Real,
                         t::Real, γ::Real)
    ng = NG
    HDF5.h5open(filename, "w") do fid
        gas = HDF5.create_group(fid, "gas")
        gas["rho" ] = Array(U[1, ng+1:ng+nx, ng+1:ng+ny, ng+1:ng+nz])
        gas["rhovx"] = Array(U[2, ng+1:ng+nx, ng+1:ng+ny, ng+1:ng+nz])
        gas["rhovy"] = Array(U[3, ng+1:ng+nx, ng+1:ng+ny, ng+1:ng+nz])
        gas["rhovz"] = Array(U[4, ng+1:ng+nx, ng+1:ng+ny, ng+1:ng+nz])
        gas["E"   ] = Array(U[5, ng+1:ng+nx, ng+1:ng+ny, ng+1:ng+nz])
        HDF5.attrs(fid)["t"]     = Float64(t)
        HDF5.attrs(fid)["nx"]    = nx
        HDF5.attrs(fid)["ny"]    = ny
        HDF5.attrs(fid)["nz"]    = nz
        HDF5.attrs(fid)["dx"]    = Float64(dx)
        HDF5.attrs(fid)["dy"]    = Float64(dy)
        HDF5.attrs(fid)["dz"]    = Float64(dz)
        HDF5.attrs(fid)["gamma"] = Float64(γ)
    end
    return nothing
end

"""
    read_snapshot(filename) -> (U_active, nx, ny, nz, dx, dy, dz, t, γ)

Read a snapshot file.  Returns the active-cell 4-D array
`U_active[q, i, j, k]` (q ∈ 1:5, no ghost cells) plus grid metadata.
"""
function read_snapshot(filename::String)
    HDF5.h5open(filename, "r") do fid
        rho  = HDF5.read(fid["gas/rho" ])
        rhovx = HDF5.read(fid["gas/rhovx"])
        rhovy = HDF5.read(fid["gas/rhovy"])
        rhovz = HDF5.read(fid["gas/rhovz"])
        E    = HDF5.read(fid["gas/E"   ])
        nx   = Int(HDF5.attrs(fid)["nx"])
        ny   = Int(HDF5.attrs(fid)["ny"])
        nz   = Int(HDF5.attrs(fid)["nz"])
        dx   = Float64(HDF5.attrs(fid)["dx"])
        dy   = Float64(HDF5.attrs(fid)["dy"])
        dz   = Float64(HDF5.attrs(fid)["dz"])
        t    = Float64(HDF5.attrs(fid)["t"])
        γ    = Float64(HDF5.attrs(fid)["gamma"])

        U = zeros(5, nx, ny, nz)
        U[1, :, :, :] .= rho
        U[2, :, :, :] .= rhovx
        U[3, :, :, :] .= rhovy
        U[4, :, :, :] .= rhovz
        U[5, :, :, :] .= E
        return U, nx, ny, nz, dx, dy, dz, t, γ
    end
end

# ---------------------------------------------------------------------------
# BH trajectory I/O

"""
    init_trajectory_file(filename, n_bh) -> nothing

Create a new HDF5 trajectory file with extendable datasets for `n_bh` black holes.
Call once at the start of a run.
"""
function init_trajectory_file(filename::String, n_bh::Int)
    HDF5.h5open(filename, "w") do fid
        # Extendable time vector
        HDF5.create_dataset(fid, "time",
                             Float64, ((0,), (-1,)),
                             chunk = (128,))
        for i in 1:n_bh
            grp = HDF5.create_group(fid, "bh$i")
            HDF5.create_dataset(grp, "mass",  Float64, ((0,),    (-1,)),    chunk=(128,))
            HDF5.create_dataset(grp, "rsink", Float64, ((0,),    (-1,)),    chunk=(128,))
            HDF5.create_dataset(grp, "pos",   Float64, ((0, 3),  (-1, 3)), chunk=(128, 3))
            HDF5.create_dataset(grp, "vel",   Float64, ((0, 3),  (-1, 3)), chunk=(128, 3))
        end
    end
    return nothing
end

"""
    append_trajectory(filename, t, bhs)

Append one time record for all BHs to the trajectory file.
"""
function append_trajectory(filename::String, t::Real, bhs)
    HDF5.h5open(filename, "r+") do fid
        # Extend and write time
        ds_t = fid["time"]
        n    = HDF5.size(ds_t, 1)
        HDF5.set_extent_dims(ds_t, (n + 1,))
        ds_t[n+1] = Float64(t)

        for (i, bh) in enumerate(bhs)
            grp    = fid["bh$i"]
            ds_m   = grp["mass"];   HDF5.set_extent_dims(ds_m, (n+1,));   ds_m[n+1]     = bh.mass
            ds_rs  = grp["rsink"];  HDF5.set_extent_dims(ds_rs, (n+1,));  ds_rs[n+1]    = r_sink(bh)
            ds_pos = grp["pos"];    HDF5.set_extent_dims(ds_pos, (n+1, 3)); ds_pos[n+1, :] = bh.pos
            ds_vel = grp["vel"];    HDF5.set_extent_dims(ds_vel, (n+1, 3)); ds_vel[n+1, :] = bh.vel
        end
    end
    return nothing
end

"""
    read_trajectory(filename) -> (t, bh_data)

Read the full trajectory file.
Returns:
- `t`: time vector (length N_dumps)
- `bh_data`: vector of NamedTuples (one per BH), each with fields
  `mass`, `rsink`, `pos` (N×3), `vel` (N×3).
"""
function read_trajectory(filename::String)
    HDF5.h5open(filename, "r") do fid
        t = HDF5.read(fid["time"])
        bh_data = []
        i = 1
        while haskey(fid, "bh$i")
            grp = fid["bh$i"]
            push!(bh_data, (
                mass  = HDF5.read(grp["mass"]),
                rsink = HDF5.read(grp["rsink"]),
                pos   = HDF5.read(grp["pos"]),
                vel   = HDF5.read(grp["vel"]),
            ))
            i += 1
        end
        return t, bh_data
    end
end
