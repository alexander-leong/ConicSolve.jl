#=
Copyright (c) 2025 Alexander Leong, and contributors

This Julia package ConicSolve.jl is released under the MIT license; see LICENSE.md
file in the root directory
=#

using ConicSolve
using CSV
using DataFrames
using Distributions
using LinearAlgebra
using Statistics

function get_expected_return(pf)
    p_bar = mean.(eachcol(pf))
    return p_bar
end

function get_price_covariance(pf)
    Σ = cov(pf)
    return Σ
end

function get_inv_cdf_std_normal(p)
    return quantile.(Normal(), p)
end

function soc_constraint_to_cone_qp_form(A, b, c, d)
    G = [b'; A]
    h = [d; c]
    return G, h
end

function get_problem_parameters(pf, α, β)
    inv_Φ_β = get_inv_cdf_std_normal(β)
    c = get_expected_return(pf)
    d = -α
    Σ = get_price_covariance(pf)
    sqrt_Σ = sqrt(Σ)
    A = sqrt_Σ / -inv_Φ_β
    b = zeros(size(Σ)[1])
    G, h = soc_constraint_to_cone_qp_form(A, c, b, d)
    return G, c, h
end

function get_qp(pf, α, β)
    G, c, h = get_problem_parameters(pf, α, β)
    P = zeros((size(G)[2], size(G)[2]))
    cones::Vector{Cone} = []
    push!(cones, SecondOrderCone(size(G)[1]))
    push!(cones, NonNegativeOrthant(size(G)[2]))
    F = ones(1, size(G)[2])
    g = [1.0]
    # set the non-negative constraint in G, h
    G = [G; -I(length(c))]
    h = [h; zeros(length(c))]
    # we're maximizing so -c instead of c
    cone_qp = ConeQP(F, G, P, g, -c, h, cones)
    return cone_qp
end

function run_example()
    # α - level of loss (scalar, vector)
    α = 1e3
    # β - risks (scalar, vector)
    β = 0.25
    # load data
    df = CSV.read("./examples/portfolio_optimization/data/portfolio.csv", DataFrame)
    pf = Matrix(df)
    # get optimization problem
    cone_qp = get_qp(pf, α, β)
    # solve optimization problem
    solver = Solver(cone_qp)
    optimize!(solver)
    x = get_solution(solver)
end

run_example()