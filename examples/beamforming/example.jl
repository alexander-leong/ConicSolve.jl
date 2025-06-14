#=
Copyright (c) 2025 Alexander Leong, and contributors

This Julia package ConicSolve.jl is released under the MIT license; see LICENSE.md
file in the root directory
=#

using ConicSolve
using LinearAlgebra

# Utility function to return the diagram, Z of an antenna
function get_Z(a, b, m, n, λ, θ)
    ϕ_l = 0
    ϕ_u = 2*pi
    dr = (b - a) / m
    dϕ = (ϕ_u - ϕ_l) / n
    ϕs = ϕ_l .+ (ϕ_l:n) * dϕ
    rs = a .+ (a:m)*dr
    f = (r, ϕ) -> r * cos(2 * pi * r * cos(θ) * cos(ϕ) / λ)
    # approximate integral by riemann sum
    Z = sum(apply_cart_on_f(f, rs, ϕs)) / 2
    return Z
end

function get_problem_parameters(diagrams, target)
    P = zeros((length(diagrams)+1, length(diagrams)+1))
    A = Matrix{Float64}(I, (length(diagrams)+1, length(diagrams)+1))
    G = -Matrix{Float64}(I, (length(diagrams)+2, length(diagrams)+1))
    G[end, 1:end-1] .= diagrams
    G[end, end] = -target
    c = ones(length(diagrams)+1)
    b = zeros(length(diagrams)+1)
    b[end] = 1.0
    h = zeros(length(diagrams)+2)
    return A, G, P, b, c, h
end

function get_qp(diagrams, target)
    A, G, P, b, c, h = get_problem_parameters(diagrams, target)
    cones::Vector{Cone} = []
    push!(cones, NonNegativeOrthant(size(G)[1]))
    cone_qp = ConeQP{Float64, Float64, Float64}(A, G, P, b, c, h, cones)
    return cone_qp
end

function run_example()
    # target diagram
    Z_target = 1
    
    # antenna diagrams
    Z_1 = 0
    Z_2 = 3e-4
    Z_3 = -1.3346
    Z = [Z_1, Z_2, Z_3]
    cone_qp = get_qp(Z, Z_target)
    
    # solve optimization problem
    solver = Solver(cone_qp)
    solver.max_iterations = 20
    status = optimize!(solver)
    return status
    # x = get_solution(solver)
end

# run_example()