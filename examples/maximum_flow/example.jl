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

function get_A(min_G)
    return vcat([get_a(i, size(min_G)[2])' for i=2:size(min_G)[2]-1]...)
end

function get_G(G)
    G = Matrix{Float64}(I, (length(G), length(G)))
    G = [G'; -G']
    return G
end

function get_c(min_G)
    c_vec = zeros((size(min_G)[2], size(min_G)[2]))
    c_vec[1, :] .= 1
    c = vec(c_vec)
    return c
end

function get_h(G, min_G)
    h = vec(G)
    h_min = -vec(min_G)
    h = [h; h_min]
    return h
end

export get_A
export get_G
export get_c
export get_h

function get_problem_parameters(G, min_G)
    # Set capacity constraints
    h = get_h(G, min_G)
    G = get_G(G)
    # Set flow conservation constraints
    A = get_A(min_G)
    P = zeros((size(A)[2], size(A)[2]))
    b = zeros(size(A)[1])
    # Set objective
    c = get_c(min_G)
    return A, G, P, b, c, h
end

function get_qp(G, min_G)
    A, G, P, b, c, h = get_problem_parameters(G, min_G)
    cones::Vector{Cone} = []
    n = size(G)[1]
    push!(cones, NonNegativeOrthant(n))
    cone_qp = ConeQP{Float64, Float64, Float64}(A, G, P, b, c, h, cones)
    return cone_qp
end

function run_example()
    G, min_G = get_graph()
    cone_qp = get_qp(G, min_G)
    kktsolve = "qrchol"
    solver = Solver(cone_qp, kktsolve)
    solver.max_iterations = 10
    status = run_solver(solver)

    KKT_x = cone_qp.KKT_x
    s = cone_qp.s
    z = cone_qp.z
    cone_qp = get_qp(G, min_G)
    cone_qp.KKT_x = KKT_x
    cone_qp.s = s
    cone_qp.z = z
    solver = Solver(cone_qp, kktsolve)
    status = run_solver(solver, true)
    return status
    # TODO
    # x = get_solution(solver)
end

run_example()