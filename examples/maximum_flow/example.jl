#=
Copyright (c) 2025 Alexander Leong, and contributors

This Julia package ConicSolve.jl is released under the MIT license; see LICENSE.md
file in the root directory
=#

using ConicSolve
using LinearAlgebra

function get_graph()
    G::Matrix{Float64} = [
        0 3 2 2 0 0 0 0;
        0 0 0 0 5 1 0 0;
        0 0 0 0 1 3 1 0;
        0 0 0 0 0 1 0 0;
        0 0 0 0 0 0 0 4;
        0 0 0 0 0 0 0 2;
        0 0 0 0 0 0 0 4;
        0 0 0 0 0 0 0 0
    ]
    min_G::Matrix{Float64} = [
    0 2 2 0 0 0 0 0;
    0 0 0 0 2 0 0 0;
    0 0 0 0 0 2 0 0;
    0 0 0 0 0 0 0 0;
    0 0 0 0 0 0 0 0;
    0 0 0 0 0 0 0 0;
    0 0 0 0 0 0 0 0;
    0 0 0 0 0 0 0 0
    ]
    return G, min_G
end

export get_graph

function get_a(i, n)
    A_vec = zeros((n, n))
    A_vec[i, :] .= -1
    A_vec[:, i] .= 1
    A_vec[i, i] = 0
    a = vec(A_vec)
    return a
end

function get_flow_conservation_constraints(min_G)
    n = size(min_G)[2]
    A = vcat([get_a(i, n)' for i=2:n-1]...)
    b = zeros(size(A)[1])
    return A, b
end

function get_capacity_constraints(G, min_G)
    G = Matrix{Float64}(I, (length(G), length(G)))
    G = [G'; -G']
    
    h = vec(G)
    h_min = -vec(min_G)
    h = [h; h_min]
    return G, h
end

function get_c(min_G)
    n = size(min_G)[2]
    c_vec = zeros((n, n))
    c_vec[1, :] .= -1
    c = vec(c_vec)
    return c
end

function get_problem_parameters(G, min_G)
    G, h = get_capacity_constraints(G, min_G)
    A, b = get_flow_conservation_constraints(min_G)

    # Set objective
    P = zeros((size(A)[2], size(A)[2]))
    c = get_c(min_G)
    return A, G, P, b, c, h
end

function get_qp(G, min_G)
    A, G, P, b, c, h = get_problem_parameters(G, min_G)
    cones::Vector{Cone} = []
    n = size(G)[1]
    push!(cones, NonNegativeOrthant(n))
    cone_qp = ConeQP(A, G, P, b, c, h, cones)
    return cone_qp
end

function run_example()
    G, min_G = get_graph()
    cone_qp = get_qp(G, min_G)
    solver = Solver(cone_qp)
    solver.max_iterations = 10
    run_solver(solver)
end

run_example()