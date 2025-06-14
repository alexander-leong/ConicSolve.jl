#=
Copyright (c) 2025 Alexander Leong, and contributors

This Julia package ConicSolve.jl is released under the MIT license; see LICENSE.md
file in the root directory
=#

using ConicSolve
using LinearAlgebra
import Random

function get_problem_parameters(n, x, λ)
    # construct 1-d finite difference matrix
    D = Matrix{Float64}(-1.0I, n-1, n)
    inds = map(x -> CartesianIndex(x, x+1), 1:n-1)
    D[inds] .= 1

    # construct objective
    P = zeros((n*4-1, n*4-1))
    inds = map(x -> CartesianIndex(n*2+x, n*2+x), 1:n)
    P[inds] .= 1
    c = zeros(n*4-1)
    c[n*3+1:n*4-1] .= λ * 1
    G = zeros((n*2-2, n*4-1))
    
    # enforce L1 constraint
    G[1:n-1, 1:n] = D
    inds = map(x -> CartesianIndex(x, n*3+x), 1:n-1)
    G[inds] .= -1
    G[n:n*2-2, 1:n] = -D
    inds = map(x -> CartesianIndex(n-1+x, n*3+x), 1:n-1)
    G[inds] .= -1
    h = zeros(size(G)[1])

    # enforce constraint on input signal
    A = zeros((n, n*4-1))
    inds = map(x -> CartesianIndex(x, n+x), 1:n)
    A[inds] .= 1
    b = x
    # enforce constraint on start of estimate
    A_est = zeros((1, n*4-1))
    A_est[1, 1] = 1
    A_est[1, n+1] = -1
    A = vcat(A, A_est)
    b = vcat(b, [0])
    # enforce L2 constraint
    A_L2 = zeros((n, n*4-1))
    inds = map(x -> CartesianIndex(x, x), 1:n)
    A_L2[inds] .= 1
    inds = map(x -> CartesianIndex(x, n+x), 1:n)
    A_L2[inds] .= -1
    inds = map(x -> CartesianIndex(x, 2*n+x), 1:n)
    A_L2[inds] .= 1
    A = vcat(A, A_L2)
    b = vcat(b, zeros(n))
    return A, G, P, b, c, h
end

function get_qp(n, x, λ)
    A, G, P, b, c, h = get_problem_parameters(n, x, λ)
    cones::Vector{Cone} = []
    push!(cones, NonNegativeOrthant(n-1))
    push!(cones, NonNegativeOrthant(n-1))
    cone_qp = ConeQP{Float64, Float64, Float64}(A, G, P, b, c, h, cones)
    return cone_qp
end

function run_example()
    Random.seed!(1)
    n = 80
    x = sin.(0:0.1:8)[1:end-1] .+ rand(0:0.1:1, n)
    λ = 1 # smoothing parameter
    cone_qp = get_qp(n, x, λ)
    # solver = Solver(cone_qp)
    # status = run_solver(solver)
    return x, cone_qp
end

run_example()