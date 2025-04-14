#=
Copyright (c) 2025 Alexander Leong, and contributors

This Julia package ConicSolve.jl is released under the MIT license; see LICENSE.md
file in the root directory
=#

using ConicSolve
using LinearAlgebra

function get_problem_parameters(G, min_G)
    # Set capacity constraints
    h = vec(G)
    G = Matrix{Float64}(I, (length(G), length(G)))
    G = [G'; -G']
    h_min = -vec(min_G)
    h = [h; h_min]
    # Set flow conservation constraints
    function get_a(i, n)
        A_vec = zeros((n, n))
        A_vec[i, :] .= -1
        A_vec[:, i] .= 1
        A_vec[i, i] = 0
        a = vec(A_vec)
        return a
    end
    A = vcat([get_a(i, size(min_G)[2])' for i=2:size(min_G)[2]-1]...)
    P = zeros((size(A)[2], size(A)[2]))
    b = zeros(size(A)[1])
    # Set objective
    c_vec = zeros((size(min_G)[2], size(min_G)[2]))
    c_vec[1, :] .= 1
    c = vec(c_vec)
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
    cone_qp = get_qp(G, min_G)
    kktsolve = "minres"
    solver = Solver(cone_qp, kktsolve)
    solver.max_iterations = 20
    status = optimize!(solver)
    return status
    # TODO
    # x = get_solution(solver)
end

run_example()