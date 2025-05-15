#=
Copyright (c) 2025 Alexander Leong, and contributors

This Julia package ConicSolve.jl is released under the MIT license; see LICENSE.md
file in the root directory
=#

using ConicSolve
using JuMP

include("../example.jl")

function run_example(A, G, c, h)
    model = Model(ConicSolve.Optimizer)
    # set_silent(model)

    n = size(A)[2]
    @variable(model, x[i=1:n])
    @variable(model, s[i=1:size(G)[1]])
    m = size(A)[1]
    b = zeros(m)

    # Cone QP
    @constraint(model, A * x == b)
    @constraint(model, h - G * x == s)
    @constraint(model, s in MOI.Nonnegatives(size(G)[1]))

    @objective(model, Min, sum(c[i] * x[i] for i in eachindex(c)))

    optimize!(model)

    return value.(x)
end

G, min_G = get_graph()
A = get_A(min_G)
h = get_h(G, min_G)
G = get_G(G)
c = get_c(min_G)

run_example(A, G, c, h)