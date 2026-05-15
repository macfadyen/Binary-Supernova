#!/usr/bin/env julia
# Floor-value sweep at NX=128.  For each ρ_floor ∈ FLOORS, launches the
# fixed-orbit run and records whether it survived + final diagnostics.
#
# Usage:   julia --project=. scripts/run_floor_sweep.jl
# Output:  demo1/floor_sweep/sweep_summary.csv
#          demo1/floor_sweep/output_f{tag}/...   (per-run dirs)
#          demo1/floor_sweep/log_f{tag}.txt      (per-run log)

using Printf
using DelimitedFiles: readdlm

const FLOORS     = [1e-3, 3e-4, 1e-4, 3e-5, 1e-5]
const NX         = 64
const SWEEP_ROOT = "demo1/floor_sweep"
const SUMMARY    = joinpath(SWEEP_ROOT, "sweep_summary.csv")

mkpath(SWEEP_ROOT)

floor_tag(ρ) = replace(@sprintf("%.0e", ρ), "-0" => "m")   # 1e-3 → "1em3"

open(SUMMARY, "w") do f
    println(f,
        "rho_floor,survived,n_steps,wall_sec,t_final," *
        "M_BH1_final,M_BH2_final,dM_BH1,dM_BH2," *
        "M_bound_final,E_gas_final,E_gas_drift," *
        "max_Fg1,max_Fg2")
end

const M_BH1_0      = 30.0 / 90.0
const M_BH2_INIT_0 = 30.0 / 90.0

for ρ in FLOORS
    tag    = floor_tag(ρ)
    outdir = joinpath(SWEEP_ROOT, "output_f$tag")
    logf   = joinpath(SWEEP_ROOT, "log_f$tag.txt")
    @info "=== Running ρ_floor = $ρ ($tag) ===" outdir

    isdir(outdir) && rm(outdir; recursive=true)

    t0 = time()
    cmd = `julia --project=. scripts/run_sn30_fixedorbit.jl
           --nx $NX --rho-floor $ρ --outdir $outdir`
    survived = try
        run(pipeline(cmd; stdout=logf, stderr=logf))
        true
    catch err
        @warn "Run failed" ρ err
        false
    end
    wall = time() - t0

    # Extract final diagnostics (if the run made it that far).
    csv_path = joinpath(outdir, "diagnostics.csv")
    n_steps = 0; t_final = 0.0
    M_BH1_f = NaN; M_BH2_f = NaN; dM_BH1 = NaN; dM_BH2 = NaN
    M_bound_f = NaN; E_gas_f = NaN; E_gas_drift = NaN
    max_Fg1 = NaN; max_Fg2 = NaN

    if isfile(csv_path)
        D = try readdlm(csv_path, ',', Float64; skipstart=1) catch; nothing end
        if D !== nothing && size(D, 1) >= 1
            n_steps     = size(D, 1)
            t_final     = D[end, 1]
            M_BH1_f     = D[end, 4]
            M_BH2_f     = D[end, 5]
            dM_BH1      = M_BH1_f - M_BH1_0
            dM_BH2      = M_BH2_f - M_BH2_INIT_0
            M_bound_f   = D[end, 9]
            E_gas_f     = D[end, 7]
            E_gas_drift = D[end, 7] - D[1, 7]
            Fg1_mag     = sqrt.(D[:,10].^2 .+ D[:,11].^2 .+ D[:,12].^2)
            Fg2_mag     = sqrt.(D[:,13].^2 .+ D[:,14].^2 .+ D[:,15].^2)
            max_Fg1     = maximum(Fg1_mag)
            max_Fg2     = maximum(Fg2_mag)
        end
    end

    open(SUMMARY, "a") do f
        @printf(f, "%.3e,%d,%d,%.1f,%.4f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6e,%.6e,%.3e,%.3e\n",
                ρ, survived ? 1 : 0, n_steps, wall, t_final,
                M_BH1_f, M_BH2_f, dM_BH1, dM_BH2,
                M_bound_f, E_gas_f, E_gas_drift,
                max_Fg1, max_Fg2)
    end

    @info "finished" ρ survived wall_sec=round(wall,digits=1) ΔM_BH1=dM_BH1 ΔM_BH2=dM_BH2 M_bound=M_bound_f
end

@info "Sweep complete." summary=SUMMARY
