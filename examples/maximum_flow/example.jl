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

function get_a(i, n)
    A_vec = zeros((n, n))
    A_vec[i, :] .= -1
    A_vec[:, i] .= 1
    A_vec[i, i] = 0
    a = vec(A_vec)
    return a
end

function get_c(min_G)
    n = size(min_G)[2]
    c_vec = zeros((n, n))
    c_vec[1, :] .= -1
    c = vec(c_vec)
    return c
end

function run_example()
    G_vals, min_G = get_graph()
    G = Matrix{Float64}(I, (length(G_vals), length(G_vals)))
    c = get_c(min_G)

    program = ConeQP()
    x = add_variable(program, NonNegativeOrthant(), size(G, 2))

    n = size(min_G)[2]
    define_program(program,
                minimize(c * x),
                vcat([get_a(i, n)' for i=2:n-1]...) * x == 0.,
                [G'; -G'] * x <= [vec(G_vals); -vec(min_G)])
    
    program = build_program(program)
    # solver = Solver(program)
    # solver.max_iterations = 10
    # run_solver(solver)
end

run_example()