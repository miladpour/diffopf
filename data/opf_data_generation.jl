############################################################
# AC-OPF DATA GENERATION (IEEE CASES)
############################################################

using PowerModels
using JuMP, Ipopt
using SparseArrays
using LinearAlgebra
using Distributions
using JSON
using CSV
using DataFrames
using Random

PowerModels.silence()

############################################################
# CASE DEFINITION
############################################################

caseID = "data/test_case/pglib_opf_case118_ieee.m"

############################################################
# OPF DATA EXTRACTOR
############################################################

function opf_data(caseID)

    data = PowerModels.parse_file(caseID)

    n_g = length(data["gen"])
    n_d = length(data["load"])
    n_l = length(data["branch"])
    n_b = length(data["bus"])
    n_s = length(data["shunt"])

    bus_to_idx = calc_admittance_matrix(data).bus_to_idx

    # -------------------- LINE DATA --------------------
    f_bus = zeros(Int64, n_l)
    t_bus = zeros(Int64, n_l)
    s̅ = zeros(n_l)
    g = zeros(n_l)
    b = zeros(n_l)

    for l in parse.(Int64, keys(data["branch"]))
        br = data["branch"][string(l)]
        f_bus[l] = bus_to_idx[br["f_bus"]]
        t_bus[l] = bus_to_idx[br["t_bus"]]
        s̅[l] = br["rate_a"]
        g[l] = br["br_r"]
        b[l] = br["br_x"]
    end

    # -------------------- BUS DATA --------------------
    v̅ = zeros(n_b)
    v̲ = zeros(n_b)
    ref_bus = 0

    for bidx in parse.(Int64, keys(data["bus"]))
        bdata = data["bus"][string(bidx)]
        idx = bus_to_idx[bidx]
        v̅[idx] = bdata["vmax"]
        v̲[idx] = bdata["vmin"]
        if bdata["bus_type"] == 3
            ref_bus = idx
        end
    end

    # -------------------- DEMAND --------------------
    p_d = zeros(n_d)
    q_d = zeros(n_d)
    M_d = zeros(Int64, n_b, n_d)

    for d in parse.(Int64, keys(data["load"]))
        ld = data["load"][string(d)]
        p_d[d] = ld["pd"]
        q_d[d] = ld["qd"]
        M_d[bus_to_idx[ld["load_bus"]], d] = 1
    end

    # -------------------- GENERATION --------------------
    p̅_g = zeros(n_g)
    p̲_g = zeros(n_g)
    q̅_g = zeros(n_g)
    q̲_g = zeros(n_g)
    c1 = zeros(n_g)

    M_g = zeros(Int64, n_b, n_g)

    for gidx in parse.(Int64, keys(data["gen"]))
        gdata = data["gen"][string(gidx)]

        p̅_g[gidx] = gdata["pmax"]
        p̲_g[gidx] = gdata["pmin"]
        q̅_g[gidx] = gdata["qmax"]
        q̲_g[gidx] = gdata["qmin"]

        if length(gdata["cost"]) >= 2
            c1[gidx] = gdata["cost"][1]
        end

        M_g[bus_to_idx[gdata["gen_bus"]], gidx] = 1
    end

    return Dict(
        :dims => Dict(:n_b=>n_b, :n_d=>n_d, :n_l=>n_l, :n_g=>n_g, :n_s=>n_s),
        :gen => Dict(:p̅_g=>p̅_g, :p̲_g=>p̲_g, :q̅_g=>q̅_g, :q̲_g=>q̲_g, :c1=>c1, :M_g=>M_g),
        :demand => Dict(:p_d=>p_d, :q_d=>q_d, :M_d=>M_d),
        :bus => Dict(:v̅=>v̅, :v̲=>v̲, :ref=>ref_bus)
    )
end

############################################################
# AC OPF SOLVER
############################################################

function AC_OPF(data)

    model = Model(Ipopt.Optimizer)
    set_silent(model)

    n_b = data[:dims][:n_b]
    n_g = data[:dims][:n_g]
    n_l = data[:dims][:n_l]

    @variable(model, v[1:n_b])
    @variable(model, θ[1:n_b])
    @variable(model, p_g[1:n_g])
    @variable(model, q_g[1:n_g])

    @objective(model, Min, sum(data[:gen][:c1][i] * p_g[i] for i in 1:n_g))

    optimize!(model)

    return Dict(
        :p_g => value.(p_g),
        :q_g => value.(q_g),
        :v => value.(v),
        :θ => value.(θ)
    )
end

############################################################
# DATA GENERATION LOOP
############################################################

data = opf_data(caseID)

results = Dict()
feasible_count = 0

for i in 1:5000

    dcopy = deepcopy(data)

    scale_d = rand(Uniform(0.8, 1.2), dcopy[:dims][:n_d])
    scale_g = rand(Uniform(0.6, 1.4), dcopy[:dims][:n_g])

    dcopy[:demand][:p_d] .= dcopy[:demand][:p_d] .* scale_d
    dcopy[:demand][:q_d] .= dcopy[:demand][:q_d] .* scale_d
    dcopy[:gen][:c1] .= dcopy[:gen][:c1] .* scale_g

    sol = AC_OPF(dcopy)

    feasible_count += 1
    results[feasible_count] = Dict(
        "p_d" => dcopy[:demand][:p_d],
        "q_d" => dcopy[:demand][:q_d],
        "p_g" => sol[:p_g],
        "q_g" => sol[:q_g]
    )
end

############################################################
# EXPORT TO CSV
############################################################

df = DataFrame()

n_g = data[:dims][:n_g]
n_d = data[:dims][:n_d]

for (i, r) in pairs(results)

    row = DataFrame(iteration = [i])

    for j in 1:n_d
        row[!, Symbol("p_d_$j")] = [r["p_d"][j]]
        row[!, Symbol("q_d_$j")] = [r["q_d"][j]]
    end

    for j in 1:n_g
        row[!, Symbol("p_g_$j")] = [r["p_g"][j]]
        row[!, Symbol("q_g_$j")] = [r["q_g"][j]]
    end

    append!(df, row)
end

CSV.write("data/IEEE_OPF_dataset.csv", df)